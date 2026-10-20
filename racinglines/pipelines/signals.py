"""
Live paper signals: what each user's strategy profile (pipelines/profiles.py) would do on the current
race weekend, stage by stage, computed by the backtest's own code. Recommendations and paper fills only:
nothing here places an order.

    racinglines strategy signals f1                         # every user with a profile, the weekend in progress
    racinglines strategy signals f1 --profile A --event 15 --asof 2026-09-27T12:00 --no-fetch   # replay (prints only)

Per run:
  1. Data (live only): FastF1 fetch + ingest when a finished session's data is missing; Polymarket
     prices (5-minute) and trades for the event's markets up to now.
  2. Pricing: each stage whose cutoff has passed AND whose session data is in, as of that cutoff, with the
     profile's model settings (weekend_sweep's stage cache: model_key + data_key). A raced event is priced
     exactly as the sweep prices it (pricing.diagnostic); an upcoming one with price_race on the weekend's
     latest classification (or the last race's field) and the same seed.
  3. Markets: weekend_sweep.weekend_markets (same filters). Live, each stage reads the market when the
     stage was priced (its trades couldn't be made earlier); a replay reads it at the cutoff, like the sweep.
  4. Taker profiles: taker_weekend.run_market over the stages so far. Every trade is a signal (the latest
     stage's are the new ones); the paper position is the replay's position.
     Maker profiles: maker_replay.replay up to now. Paper fills are signals; so is a market starting or
     stopping quoting in a stage (not every requote).
  5. Store (strategy_signals, paper_positions; idempotent per profile, market, stage and side), then alert.

Venue: Polymarket unless the profile's settings say venue = "kalshi" (sweep_settings; unset by default and left
out of every settings key). A Kalshi profile reads the race's exchange='kalshi' links, one market per ticker
(never grouped by condition_id, Kalshi's event ticker), its tape per market (its own 24 h volume filter), and
its maker pays KALSHI_MAKER_FEE; its signals carry detail.venue = 'kalshi' and its positions venue = 'kalshi'.
The profile setting is the only switch: a live run on Kalshi needs it, and a replay may name the venue
(compute(venue=...), as demo_history does). Paper only on every venue; KALSHI_TRADING_ENABLED is never read here.

Heat: the modelled EV of a taker entry, shares x (our probability of the side - limit price), graded
(HEAT). Takers see the heat, never our fair value or edge (docs/webapp.md, Roles). The tiers are the
median and top quartile of profile A's 472 backtest entries (2025 + 2026, modelled EV $4.8 / $12.0 / $29.8
at the quartiles). Realised ROI by EV quartile: 2026 +10% / +10% / +40% / +70%, but 2025 +10% / +20% /
-20% / 0%, and realised P&L was about a quarter of modelled EV: heat grades the model's conviction, it
doesn't promise returns.

Following (takers): a user needn't take every recommendation. With a profile's follow_rate below 1 (set
in code, e.g. the demo taker's 0.33 in profiles.DEMO_FOLLOW; not in the UI), the user decides once per
market, at its first entry, whether to follow it: with probability follow_rate x HEAT_WEIGHT[heat] /
HEAT_MIX_MEAN (hotter entries are likelier), deterministic per user and market. A followed market is
followed through every later resize and exit; the rest are "passed". Paper positions hold followed
markets only.
"""

import hashlib

import subprocess
import sys
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.markets.strategies import taker_weekend as RB
from racinglines.pipelines import sweep_settings as SS
from racinglines.pipelines import weekend_sweep as WS

STAGE_ROUND = {"after FP1": "fp1", "after FP2": "fp2", "after FP3": "fp3", "after SQ": "sprint_qual",
               "after Sprint": "sprint", "after Quali": "qual"}
SESSION_ROUND = {"FP1": "fp1", "FP2": "fp2", "FP3": "fp3", "SQ": "sprint_qual", "Sprint": "sprint", "Quali": "qual",
                 "Race": "race"}
FASTF1_SESSIONS = "FP1,FP2,FP3,SQ,Q,S,R"
WEEKEND_LEAD = timedelta(days=4)          # signals run from 4 days before the race ...
WEEKEND_TAIL = timedelta(hours=24)        # ... until a day after it (results in: paper positions settle)
RACE_DONE = timedelta(minutes=150) + WS.DATA_LAG   # when the race's results can be fetched
SEED = 42                                 # pricing.diagnostic's seed
HEAT = ((30.0, 3), (12.0, 2), (0.0, 1))   # modelled EV ($) of the entry -> heat (see the docstring)
HEAT_LABEL = {1: "warm", 2: "hot", 3: "very hot"}
QUOTING = "quoting"
HEAT_WEIGHT = {1: 0.6, 2: 1.0, 3: 1.4}    # relative chance of following an entry, by heat
HEAT_MIX_MEAN = 0.9                       # mean weight over A's backtest entries (1/2 warm, 1/4 hot, 1/4 very hot)


def _stages(w):
    """The side stages (sports/f1.toml [sessions.sim], e.g. the sprint) this weekend has, from the schedule's
    FastF1 event format, so a live run prices them before their sessions' data exists (pricing.price_stages)."""
    from racinglines.models.inputs import laps as M
    from racinglines.sources.fastf1.fetch import SPRINT_FORMATS
    if not w or str(w.get("format", "")) not in SPRINT_FORMATS:
        return []
    return [k for k in M.SIM_SESSIONS if k != M.MAIN_STAGE]



# ---------------------------------------------------------------------------
# Pure pieces (tested without a database)
# ---------------------------------------------------------------------------

def heat(action, side, shares, limit_price, fair):
    """1-3 for an entry (buy) by its modelled EV; None for exits / trims or without a fair value."""
    if action != "buy" or fair is None or limit_price is None or shares is None:
        return None
    p = fair if side == "YES" else 1 - fair
    ev = shares * (p - limit_price)
    return next((h for lo, h in HEAT if ev >= lo), None) if ev > 0 else None


def modelled_ev(action, side, shares, limit_price, fair):
    if action != "buy" or fair is None:
        return None
    return shares * ((fair if side == "YES" else 1 - fair) - limit_price)


def follow_prob(heat_, rate):
    return 1.0 if rate is None or rate >= 1 else min(1.0, rate * HEAT_WEIGHT.get(heat_ or 1, 1.0) / HEAT_MIX_MEAN)


def follows(user_id, market_key, heat_, rate):
    """Does this user follow this market's recommendations? Deterministic per (user, market)."""
    u = int(hashlib.sha1(f"{user_id}:{market_key}".encode()).hexdigest()[:8], 16) / 2 ** 32
    return u < follow_prob(heat_, rate)


def apply_follow(signals, positions, user_id, rate):
    """Mark each taker signal followed or not (decided at the market's first entry) and keep only
    followed markets' positions. -> (signals, positions)."""
    first = {}
    for s in sorted(signals, key=lambda s: (s["signal_ts"], s["action"] != "buy")):
        if s["action"] == "buy":
            first.setdefault(s["market_key"], s.get("heat"))
    keep = {k for k, h in first.items() if follows(user_id, k, h, rate)}
    out = [dict(s, followed=s["market_key"] in keep) for s in signals]
    return out, [p for p in positions if p["market_key"] in keep]


def taker_signals(markets, p, capped=False):
    """Replay the taker over each market's stages so far (taker_weekend.run_market, h2h threshold via
    params_for). capped: honour p.max_deployed across the markets, traded in time order (taker_weekend._capped,
    the backtest's cap: buys over it are cut to fit); off, every market is run on its own, as the live pass always
    has (a profile's own max_deployed setting is not applied live). -> (signal rows, position rows)."""
    sigs, pos = [], []
    results = RB._capped(markets, p) if capped and p.max_deployed is not None else \
        [RB.run_market(mk["stages"], mk["outcome"], RB.params_for(mk["kind"], p)) for mk in markets]
    for mk, r in zip(markets, results):
        last = next((s for s in reversed(mk["stages"]) if s["price"] is not None), None)
        if r["trades"] or abs(r["yes"]) > 1e-9 or abs(r["no"]) > 1e-9:
            pos.append(dict(market_key=mk["key"], kind=mk["kind"], subject=mk["subject"], yes_shares=r["yes"],
                            no_shares=r["no"], cash=r["cash"], mark=last["price"] if last else None,
                            outcome=mk["outcome"]))
        for tr in r["trades"]:
            action = "buy" if tr["shares"] > 0 else "sell"
            shares = abs(tr["shares"])
            stage = next(s for s in mk["stages"] if s["label"] == tr["stage"])
            fair_side = tr["fair"] if tr["side"] == "YES" else 1 - tr["fair"]
            sigs.append(dict(market_key=mk["key"], kind=mk["kind"], subject=mk["subject"], stage=tr["stage"],
                             dedupe=tr["stage"], action=action, side=tr["side"], shares=shares,
                             limit_price=tr["price"], fair=tr["fair"], price=tr["mid"], edge=tr["fair"] - tr["mid"],
                             heat=heat(action, tr["side"], shares, tr["price"], tr["fair"]),
                             target_cost=shares * tr["price"], signal_ts=stage["t"],
                             detail=dict(ev=modelled_ev(action, tr["side"], shares, tr["price"], tr["fair"]),
                                         side_prob=fair_side)))
    return sigs, [_position(p) for p in pos]


def _position(p):
    """A paper position row. paper_positions.outcome is YES/NO; a cancelled race's venue rule (markets/
    settlement_rules.py) can instead pay 0.5 or refund at cost (VOID): then the position is closed to cash at
    that payout (shares 0, outcome None), which the track record reads as settled at that P&L."""
    from racinglines.markets import settlement_rules as SR
    o = p["outcome"]
    if o is None or isinstance(o, (bool, np.bool_)):
        return p
    if o == SR.FAIR:                # Kalshi's fair-price settlement with no recorded price: still open
        return dict(p, outcome=None)
    return dict(p, cash=SR.settle_position(p["yes_shares"], p["no_shares"], p["cash"], o), yes_shares=0.0,
                no_shares=0.0, outcome=None)


def maker_state(quotes, fills, markets, stage_label, now_ns):
    """Maker signals from a replay: a paper fill each, and per market and stage one 'quote' when it starts
    quoting and one 'pull' when it stops for a reason other than the pre-session pull. Also each market's
    state now. quotes/fills: maker_replay.replay frames; stage_label: run_id -> stage label."""
    by = {m.cond: m for m in markets}
    sigs, state = [], {}
    for cond, g in quotes.sort_values("ts").groupby("cond", sort=False):
        mk = by[cond]
        for run_id, gs in g.groupby("run_id", sort=False):
            quoted = gs["skip"].isna()
            if not quoted.any():
                continue
            first = gs[quoted].iloc[0]
            lab = stage_label.get(int(run_id), str(run_id))
            sigs.append(dict(market_key=cond, kind=mk.kind, subject=mk.subject, stage=lab, dedupe=lab, action="quote",
                             side="both", limit_price=None, shares=None, fair=mk.fairs.get(int(run_id)),
                             signal_ts=pd.Timestamp(int(first["ts"]), tz="UTC"),
                             detail=dict(bid=first["bid"], ask=first["ask"])))
            after = gs[gs["ts"] > first["ts"]]
            stop = after[after["skip"].notna()]
            if len(stop):
                s = stop.iloc[0]
                sigs.append(dict(market_key=cond, kind=mk.kind, subject=mk.subject, stage=lab, dedupe=lab,
                                 action="pull", side="both", fair=mk.fairs.get(int(run_id)),
                                 signal_ts=pd.Timestamp(int(s["ts"]), tz="UTC"), detail=dict(reason=s["skip"])))
        last = g.iloc[-1]
        state[cond] = dict(bid=last["bid"], ask=last["ask"],
                           quote_state=QUOTING if pd.isna(last["skip"]) else last["skip"])
    seen = {}
    for f in fills.to_dict("records") if len(fills) else []:
        mk = by[f["cond"]]
        lab = stage_label.get(int(f["run_id"]), str(f["run_id"]))
        side = "YES" if f["side"] == "buy" else "NO"          # buying YES / selling YES (= long NO exposure)
        # fills sharing a market, side and timestamp (Kalshi's tape is timed to the second) each get their own
        # key, "<ts>.<n>" from the second one on, so store keeps them all; a lone fill's key stays "<ts>"
        k = (f["cond"], side, int(f["ts"]))
        n = seen[k] = seen.get(k, -1) + 1
        sigs.append(dict(market_key=f["cond"], kind=mk.kind, subject=mk.subject, stage=lab,
                         dedupe=str(int(f["ts"])) + (f".{n}" if n else ""), action="fill", side=side, shares=float(f["qty"]),
                         limit_price=float(f["price"]), fair=f["fair"], price=f.get("mid"),
                         signal_ts=pd.Timestamp(int(f["ts"]), tz="UTC"), status="filled_paper",
                         detail=dict(maker_side=f["side"])))
    return sigs, state


def truncate_stages(stages, now_ns):
    """The maker's stages up to now: later ones dropped, the current one ends now (no pre-session pull)."""
    out = []
    for st in stages:
        if st["start"] >= now_ns:
            break
        out.append(dict(st, end=now_ns, session_end=False) if st["end"] > now_ns else st)
    return out


# ---------------------------------------------------------------------------
# Event, data, pricing
# ---------------------------------------------------------------------------

def pick_event(sched, now, event="next"):
    """(round, w): the given round, or the first event whose race hasn't finished by `now`."""
    if event not in (None, "", "next"):
        rnd = int(str(event).split("-")[-1])
        return rnd, sched[rnd]
    for rnd in sorted(sched):
        if sched[rnd]["race_start"] + WEEKEND_TAIL > now:
            return rnd, sched[rnd]
    return None, None


def in_weekend(w, now):
    return w["race_start"] - WEEKEND_LEAD <= now <= w["race_start"] + WEEKEND_TAIL


def settled(engine, profile, event_key, venue="polymarket"):
    """True once the event's race result is stored and this profile's stored rows for it (every user's, this venue)
    hold no open position: a live pass would only price the completed event again (the WEEKEND_TAIL day) and store
    the same rows. A profile with nothing stored yet for the event, or no candidate id to find its rows by, is never
    skipped (its first pass stores it)."""
    if profile.get("candidate_id") is None:
        return False
    with engine.connect() as c:
        return bool(c.execute(text("""
            SELECT EXISTS (SELECT 1 FROM results r JOIN rounds ro ON ro.id = r.round_id JOIN races ra ON ra.id = ro.race_id
                           JOIN events e ON e.id = ra.event_id WHERE e.source_key = :e AND ro.kind = 'race')
               AND (EXISTS (SELECT 1 FROM paper_positions WHERE candidate_id IS NOT DISTINCT FROM :c AND event_key = :e
                                                            AND venue = :v)
                    OR EXISTS (SELECT 1 FROM strategy_signals WHERE candidate_id IS NOT DISTINCT FROM :c
                                                                AND event_key = :e))
               AND NOT EXISTS (SELECT 1 FROM paper_positions WHERE candidate_id IS NOT DISTINCT FROM :c
                                 AND event_key = :e AND venue = :v AND outcome IS NULL
                                 AND (yes_shares <> 0 OR no_shares <> 0))"""),
            dict(e=event_key, c=profile.get("candidate_id"), v=venue or "polymarket")).scalar())


def _missing_sessions(meas, w, now):
    """Sessions of this weekend that ended (plus the data lag) but aren't in the database yet, the race
    included (its result settles the paper positions)."""
    year, rnd = (int(x) for x in w["event_key"].split("-"))
    ev = meas.res[(meas.res["year"] == year) & (meas.res["series_round"] == rnd)]
    have = set(ev["round"])
    q = ev[ev["round"] == "qual"]
    if len(q) and q["position"].isna().all():    # stored before it had an order (FastF1 right after the session):
        have.discard("qual")                     # ingest again, which reads the order from the laps
    out = [lab for lab, cutoff in w["stages"] if lab in STAGE_ROUND and cutoff <= now
           and STAGE_ROUND[lab] not in have]
    if w["race_start"] + RACE_DONE <= now and "race" not in have:
        out.append("Race")
    return out


def refresh_fastf1(w, echo=print):
    year, rnd = (int(x) for x in w["event_key"].split("-"))
    rl = [sys.executable, "-m", "racinglines", "data"]
    for argv in (["fetch", "f1", "--years", str(year), "--rounds", str(rnd), "--sessions", FASTF1_SESSIONS],
                 ["ingest", "f1", "--years", str(year)]):
        r = subprocess.run(rl + argv, capture_output=True, text=True, timeout=1800)
        echo(f"  f1 {argv[0]}: {'ok' if r.returncode == 0 else 'failed'} {(r.stdout.strip().splitlines() or [''])[-1]}")


def refresh_markets(engine, engine_url, w, now, echo=print, venue="polymarket"):
    """Polymarket 5-minute prices and the trade tape for the event's markets, up to now (venue="kalshi": its
    minute candlesticks and trades, per event ticker, from markets/kalshi/sync.py)."""
    from racinglines.db.config import get_session
    from racinglines.markets import store as MS
    if venue == "polymarket":
        from racinglines.markets.polymarket.sync import fetch_history, fetch_trades
    elif venue == "kalshi":
        from racinglines.markets.kalshi.sync import fetch_history, fetch_trades
    else:                                        # an exchange defined as data (exchanges/<code>.toml, e.g. OG.com)
        return _refresh_exchange(engine, engine_url, w, now, venue, echo)
    with engine.connect() as c:
        rid = WS._race_id(c, w["event_key"])
        links = WS._links(c, rid, venue) if rid else pd.DataFrame()
        if not len(links):
            echo(f"  no {venue.capitalize()} markets linked to this event yet")
            return
        start = (w["stages"][0][1] - timedelta(hours=36)).tz_localize("UTC")
        have = MS.read(c, "prices", tokens=links["token_id"].tolist(), start=start, end=now.tz_localize("UTC"),
                       root=MS.root_for(venue))
        since = max(start, pd.Timestamp(have["ts"].max()).tz_convert("UTC") - timedelta(hours=1)) if len(have) else start
        recent = (now - timedelta(hours=6)).tz_localize("UTC").to_pydatetime() if len(have) else None
        with get_session(engine_url) as s:
            if venue == "polymarket":
                n = fetch_history(s, c, None, since.to_pydatetime(), now.tz_localize("UTC").to_pydatetime(), fidelity=5,
                                  tokens=links["token_id"].tolist())
                k = fetch_trades(s, c, links["event_slug"].dropna().unique().tolist(), modeled_only=True, since=recent)
            else:
                events = links["condition_id"].dropna().unique().tolist()      # Kalshi: the event tickers
                n = fetch_history(s, c, events, since.to_pydatetime(), now.tz_localize("UTC").to_pydatetime(), period=1)
                k = fetch_trades(s, c, events, since=recent)
    echo(f"  {venue.capitalize()}: {n} price points, {k} trades")


def _refresh_exchange(engine, engine_url, w, now, venue, echo=print):
    """refresh_markets for an exchange driven by its schema (markets/exchange_driver.py): the trades and minute
    prices of the event's markets over the last day (idempotent upserts)."""
    from racinglines.db.config import get_session
    from racinglines.markets import exchange_driver as D
    start = (now - timedelta(hours=26)).tz_localize("UTC").to_pydatetime()
    end = now.tz_localize("UTC").to_pydatetime()
    with engine.connect() as c:
        rid = WS._race_id(c, w["event_key"])
        links = WS._links(c, rid, venue) if rid else pd.DataFrame()
        if not len(links):
            echo(f"  no {venue} markets linked to this event yet")
            return
        events = sorted(set(links["condition_id"].dropna()))
        with get_session(engine_url) as s:
            k = D.fetch_trades(s, c, venue, events=events, since=start)
            n = D.fetch_history(s, c, venue, start, end, events=events)
    echo(f"  {venue}: {n} price points, {k} trades")


def _entrants(meas, event_id, cutoff):
    """Who races, from what's known at the cutoff: this weekend's qualifying (or sprint qualifying,
    FP3, FP2) classification, else the last race's field. FP1 is skipped: reserve drivers run it."""
    from racinglines.models.inputs import laps_pricing as run
    v = meas.view(cutoff)
    if event_id is not None:
        for code in ("qual", "sprint_qual", "fp3", "fp2"):
            q = v.res[(v.res["event_id"] == event_id) & (v.res["round"] == code)]
            if len(q):
                return q[["athlete_id", "driver", "team_key"]].drop_duplicates("athlete_id").reset_index(drop=True)
    last = v.drivers.sort_values("r_ts")["event_id"].iloc[-1]
    return run.entry_list(v, last)


def _venue(meas, event_id, w):
    if event_id is not None:
        return meas.res.loc[meas.res["event_id"] == event_id, "venue"].iloc[0]
    from racinglines.db.ingest import _slugify
    from racinglines.models.inputs.laps_pricing import upcoming_schedule
    year, rnd = (int(x) for x in w["event_key"].split("-"))
    s = upcoming_schedule(year)
    return _slugify(s.loc[s["round"] == rnd, "location"].iloc[0])


def price_stages_now(meas, hist, w, st, engine, engine_url, now, echo=print):
    """[(label, cutoff, run_id, priced_at)] for the stages whose cutoff has passed and whose session data is
    in; stops at the first stage still waiting for data. Reuses the sweep's cached stage runs."""
    from racinglines.models.inputs import laps_pricing as run
    year, rnd = (int(x) for x in w["event_key"].split("-"))
    ev = meas.res[(meas.res["year"] == year) & (meas.res["series_round"] == rnd)]
    event_id = int(ev["event_id"].iloc[0]) if len(ev) else None
    raced = event_id is not None and (ev["round"] == "race").any()
    mk = st.model_key
    with engine.connect() as c:
        have = pd.read_sql(text("""SELECT id, created_at, params->>'cutoff' AS cutoff, params->>'data_key' AS d
                                   FROM model_runs WHERE kind = 'diagnostic' AND params ? 'sweep_stage'
                                     AND params->>'model_key' = :m AND params->>'event_key' = :k ORDER BY id"""),
                           c, params=dict(m=mk, k=w["event_key"]))
    have = {(str(pd.Timestamp(cu)), d): (int(i), pd.Timestamp(t).tz_convert("UTC").tz_localize(None))
            for i, t, cu, d in zip(have["id"], have["created_at"], have["cutoff"], have["d"])}
    out = []
    for label, cutoff in w["stages"]:
        if cutoff > now:
            break
        if label in STAGE_ROUND and STAGE_ROUND[label] not in set(ev["round"]):
            echo(f"  {label}: waiting for the session's data")
            break
        dk = SS.data_key(meas.view(cutoff))
        hit = have.get((str(pd.Timestamp(cutoff)), dk))
        if hit is None:
            if raced:
                _, summ, ex, _ = run.diagnostic(meas, hist, w["event_key"], cutoff, n_sims=st["sims"],
                                                use_track=st["track_features"], seed=st.rng_seed, stages=_stages(w))
            else:
                summ, ex = run.price_race(meas, hist, cutoff, event_id, n_sims=st["sims"],
                                          rng=np.random.default_rng(st.rng_seed), use_track=st["track_features"],
                                          entrants=_entrants(meas, event_id, cutoff), venue=_venue(meas, event_id, w),
                                          stages=_stages(w), event_key=w["event_key"])
            extra = dict(model_key=mk, data_key=dk, model_settings={n: st.to_json()[n] for n in SS.MODEL_NAMES},
                         live=not raced)
            if st["variant"] != "baseline":
                extra["variant"] = st["variant"]
            rid = run.save_diagnostic(engine_url, w["event_key"], cutoff, summ, ex, st["sims"], sweep_stage=label,
                                      **extra)
            if extra["live"]:                     # a live reprice moves the board: log it (once per stage stored)
                log_live_price(engine_url, w["event_key"], rid, st, dk, stage=label)
            hit = (rid, now)
            echo(f"  {label}: priced (run #{rid})")
        out.append((label, cutoff, *hit))
    return out


# ---------------------------------------------------------------------------
# One profile through the weekend
# ---------------------------------------------------------------------------

def resolve_venue(settings, venue=None, live=True):
    """The exchange a run trades: the profile's `venue` setting (polymarket while unset), or `venue` when a replay
    names one. Live, only the profile's own setting counts: a live run on another exchange without it is refused,
    so nothing changes for a deployed Polymarket profile."""
    own = SS.venue_of(settings)
    if venue is None or venue == own:
        return own
    if live:
        raise ValueError(f"venue {venue!r}: a live run trades the profile's own venue ({own}); set the profile's "
                         f"`venue` setting to trade {venue}")
    return venue


def _naive_utc(t):
    """A time as the naive UTC the stage cutoffs use: "2026-10-09T07:35Z" and "2026-10-09T07:35" are the same moment."""
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def compute(engine, engine_url, profile, now=None, event="next", live=True, fetch=True, echo=print, cache=None,
            venue=None, scale=1.0):
    """Signals and positions of one profile at `now` (naive UTC; default the current time).
    live=False is a replay: markets are read at each stage's cutoff, exactly like the sweep.
    cache: a dict reused across calls (the loaded measurements and each model's training history).
    venue: the exchange to replay on instead of the profile's own (see resolve_venue; "kalshi": Kalshi's links
    and recorded tape, its maker fee).
    scale: a taker's stake multiplier (TakerParams.scale; a blend member's weight, see compute_all).
    -> dict(event, stages, signals, positions, venue, note)."""
    from racinglines.models.inputs import laps_pricing as run
    st = SS.Settings.from_dict(profile["settings"], strict=False)
    venue = resolve_venue(st, venue, live)
    now = _naive_utc(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    year = int(str(event).split("-")[0]) if "-" in str(event or "") else now.year     # "2026-15" or a round
    sched = WS.schedule(year)
    rnd, w = pick_event(sched, now, event)
    base = dict(profile=profile, event=w, stages=[], signals=[], positions=[], maker_state={}, venue=venue)
    if w is None:
        return dict(base, note="no race weekend left this season")
    if live and not in_weekend(w, now):
        return dict(base, note=f"{w['name']}: signals start {(w['race_start'] - WEEKEND_LEAD):%a %d %b}")
    if live and settled(engine, profile, w["event_key"], venue):
        return dict(base, note=f"{w['name']}: complete, its paper positions settled: nothing left to price")
    cache = {} if cache is None else cache
    meas = cache["meas"] if "meas" in cache else run.Measurements.load(engine)
    if live and fetch:
        if _missing_sessions(meas, w, now):
            refresh_fastf1(w, echo)
            meas = run.Measurements.load(engine)
            cache.clear()
        refresh_markets(engine, engine_url, w, now, echo, venue=venue)
    cache["meas"] = meas
    with st.applied():
        hist = cache.get(("hist", st.model_key))
        if hist is None:
            hist = cache[("hist", st.model_key)] = run.history(meas, st["track_features"])
        stages = price_stages_now(meas, hist, w, st, engine, engine_url, now, echo)
    base["stages"] = stages
    if not stages:
        return dict(base, note=f"{w['name']}: no stage priced yet")
    runs = [(lab, cu, rid) for lab, cu, rid, _ in stages]
    strategy = profile["strategy"]
    with engine.connect() as c:
        base["race_id"] = WS._race_id(c, w["event_key"])
        if strategy in WS.TAKER_MODES:
            price_times = {lab: max(cu, at) for lab, cu, _, at in stages} if live else None
            opt = tuple(k for k in WS.OPT_KINDS if k in st["market_kinds"])
            markets = WS.weekend_markets(c, w, runs, min_volume_24h=st["min_volume_24h"], price_times=price_times,
                                         venue=venue, kinds=WS.KINDS + opt if opt else None,
                                         thin_depth=st["thin_edge_mult"] is not None,
                                         tol_by_kind=SS.parse_map(st["coherence_tol_by_kind"]),
                                         max_spread=st.get("max_spread"))
            markets = [m for m in markets or [] if m["kind"] in st["market_kinds"]]
            p = taker_params(st, strategy, venue)
            if scale != 1.0:
                p = replace(p, scale=p.scale * scale)
            base["signals"], base["positions"] = taker_signals(markets, p)
            base["_taker"] = taker_inputs(markets, p)
        elif strategy in WS.MAKERS:
            base.update(_maker(c, w, runs, st, strategy, now, live, venue=venue))
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
    if venue != "polymarket":                    # another exchange's rows say so (the default rows stay as they were)
        tag(base, detail=dict(venue=venue), position=dict(venue=venue))
    return base


def taker_params(st, strategy, venue):
    """The live taker's TakerParams from a profile's settings, as the sweep reads them for the same settings
    (weekend_sweep): sell_floor and the stage filter max_spread (weekend_markets) apply live too. Not applied
    live: max_deployed (see taker_signals), kelly / root_unit sizing, and taker_fill (the live pass prices
    from our own tape)."""
    return RB.TakerParams(min_edge=st["min_edge"], stake_per_edge=st["stake_per_edge"], max_stake=st["max_stake"],
                          cost=WS.taker_cost(st["cost"], venue), late_stages=st["late_stages"],
                          min_edge_h2h=st["min_edge_h2h"],
                          min_edge_by_kind=tuple(SS.parse_map(st["min_edge_by_kind"]).items()),
                          stages=None if st["taker_stages"] == SS.STAGES else st["taker_stages"], mode=strategy,
                          thin_edge_mult=st["thin_edge_mult"], taker_fee=WS.taker_fee(venue),
                          sell_floor=None if st.get("sell_floor") in (None, "none") else st.get("sell_floor"))


def taker_inputs(markets, p):
    """What a taker computation keeps so one player's copy can be re-sized (for_player): its markets, its parameters
    and the tags added to its rows since (tag)."""
    return dict(markets=markets, params=p, tags=[])


def tag(out, detail=None, position=None):
    """Add `detail` keys to every signal's detail and `position` keys to every position, and remember them so a
    player's re-sized copy (for_player) is tagged the same way. -> out"""
    if detail:
        out["signals"] = [dict(s, detail=dict(s.get("detail") or {}, **detail)) for s in out["signals"]]
    if position:
        out["positions"] = [dict(p, **position) for p in out["positions"]]
    if out.get("_taker") is not None:
        out["_taker"]["tags"].append((detail, position))
    return out


def for_player(conn, user_id, out, kelly=None):
    """One user's copy of a shared computation. A registered fantasy player's taker strategy is re-run on the same
    markets sized to their F$ (fantasy/sizing.py: stakes x their strategy bankroll / grant, open cost capped at the
    bankroll); anyone else (the demo accounts, an account without a grant, a maker, a replay) gets `out` unchanged.
    The strategy's bankroll is its own grant plus its own P&L: the player's manual trades never change it.
    kelly: the player's Kelly fraction (profiles.sizing_of of their profile; None = standard sizing)."""
    from racinglines.fantasy import sizing as FS
    t = out.get("_taker")
    if t is None or not out.get("stages"):
        return out
    bank = FS.player_bankroll(conn, user_id, out["event"]["event_key"])
    if bank is None:
        return out
    sigs, pos = taker_signals(t["markets"], FS.sized(t["params"], bank, kelly=kelly), capped=True)
    mine = dict(out, signals=sigs, positions=pos, _taker=None, bankroll=bank)
    for detail, position in t["tags"]:
        tag(mine, detail, position)
    return mine


def compute_all(engine, engine_url, profile, **kw):
    """compute() for any profile -> [out]. One strategy: [compute(profile)], exactly as before. A blend
    (profiles.is_combo): one out per member, the member's own strategy and settings with its stakes scaled by its
    weight, its signals tagged detail.member = its code, and out["profile"] the member's candidate under the blend's
    name (so each member keeps its own rows: strategy_signals and paper_positions are unique per candidate)."""
    from racinglines.pipelines import profiles as PF
    if not PF.is_combo(profile):
        return [compute(engine, engine_url, profile, **kw)]
    outs = []
    for m in profile["members"]:
        out = compute(engine, engine_url, m["member"], scale=m["weight"], **kw)
        out["profile"] = dict(m["member"], name=profile["name"], member=m["code"])
        tag(out, detail=dict(member=m["code"]))
        outs.append(out)
    return outs


def combo_key(profile):
    """What run_all groups users by: the candidate, or a blend's (code, weight) members (two basic accounts with
    the same draw share one computation; different draws never do, though both are called "Your picks")."""
    from racinglines.pipelines import profiles as PF
    if PF.is_combo(profile):
        return ("combo",) + tuple((m["code"], m["weight"]) for m in profile["members"])
    return profile.get("candidate_id") or profile["name"]


def _maker(c, w, runs, st, strategy, now, live, venue="polymarket"):
    from racinglines.markets.strategies import maker_replay as R
    sessions = None if not live else [(SESSION_ROUND.get(k, k), t) for k, t in w["sessions"]]
    ev = R.load_event(c, [r for _, _, r in runs], sessions=sessions, books=st["fill"] == "queue",
                      **({} if venue == "polymarket" else dict(exchange=venue)))
    ev = dict(ev, markets=[m for m in ev["markets"] if m.kind in st["market_kinds"]])
    return maker_run(ev, st, strategy, now, {rid: lab for lab, _, rid in runs}, venue)


def maker_run(ev, st, strategy, now, labels, venue="polymarket"):
    """The maker replay of an event (maker_replay.load_event's shape, any sport) up to `now`: its signals, paper
    positions and quote state. labels: the event's stage run_id -> stage label."""
    from racinglines.markets.strategies import maker_replay as R
    now_ns = R._ns(now)
    ev["stages"] = truncate_stages(ev["stages"], now_ns)
    opts = dict(WS.MAKERS[strategy])
    if "info_skew" in opts:
        opts["info_skew"] = st["info_skew"]
    opts.pop("widen", None)            # widening needs earlier weekends' markouts; not applied live
    p = replace(WS.maker_params(st, venue), **opts)     # the sweep's maker parameters, volume floor included
    rep = R.replay(ev, p)
    sigs, state = maker_state(rep["quotes"], rep["fills"], ev["markets"], labels, now_ns)
    pos = []
    for r in rep["positions"].to_dict("records"):
        mk = next(m for m in ev["markets"] if m.cond == r["cond"])
        s = state.get(r["cond"], {})
        pos.append(dict(market_key=r["cond"], kind=r["kind"], subject=r["subject"], yes_shares=r["inventory"],
                        no_shares=0.0, cash=r["cash"], mark=R.PublicView(mk, now_ns).mid(), outcome=r["outcome"],
                        bid=s.get("bid"), ask=s.get("ask"), quote_state=s.get("quote_state")))
    return dict(signals=sigs, positions=[_position(p) for p in pos], maker_state=state)


def format_replay(out):
    """Plain-text table of a compute() result (the replay / parity view shows our fair values)."""
    w = out["event"]
    member = out["profile"].get("member")             # a blend member's run (compute_all)
    head = f"{out['profile']['name']}{f' [{member}]' if member else ''} · {w['name'] if w else '-'}"
    if out.get("note"):
        return f"{head}: {out['note']}"
    lines = [head, "stages: " + ", ".join(f"{lab} #{rid}" for lab, _, rid, _ in out["stages"])]
    for s in out["signals"]:
        lines.append(f"  {s['stage']:13s} {s['action']:5s} {s['side']:4s} {s.get('shares') or 0:8.2f} "
                     f"@ {s.get('limit_price') or 0:.3f}  fair {s.get('fair') or 0:.3f}  {s['kind']:21s} {s['subject']}")
    lines.append(f"{len(out['signals'])} signals, {len(out['positions'])} positions")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Storage and the run over every assigned user
# ---------------------------------------------------------------------------

SIGNAL_COLS = ("market_key", "kind", "subject", "stage", "dedupe", "action", "side", "shares", "limit_price", "fair",
               "price", "edge", "heat", "target_cost", "signal_ts", "detail", "status")


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return (v.tz_localize("UTC") if v.tzinfo is None else v).to_pydatetime()
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def store(conn, user_id, out, follow_rate=None, history=False, venue=None):
    """Insert the run's signals (existing ones are kept as they are), mark superseded taker signals
    expired, and replace the user's paper positions for the event. Returns the ids of new signals.
    follow_rate: a taker user's (see Following); history: a backfilled weekend (a backtest replay shown
    as the account's track record): taken / passed statuses, already seen, flagged detail.backfill.
    venue: another exchange ("kalshi"; default: the computation's own, out["venue"]): its signals carry detail.venue
    and only that venue's positions are replaced; a Polymarket run leaves every other venue's positions alone."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from racinglines.db import models as m
    pr, w = out["profile"], out["event"]
    if venue is None and out.get("venue", "polymarket") != "polymarket":
        venue = out["venue"]
    signals, positions = out["signals"], out["positions"]
    taker = pr["strategy"] in WS.TAKER_MODES
    if taker:
        signals, positions = apply_follow(signals, positions, user_id, follow_rate)
    common = dict(user_id=user_id, candidate_id=pr.get("candidate_id"), profile=pr["name"], strategy=pr["strategy"],
                  race_id=out.get("race_id"), event_key=w["event_key"])
    run_of = {lab: rid for lab, _, rid, _ in out["stages"]}
    new = []
    for s in signals:
        row = dict(common, **{k: _clean(s.get(k)) for k in SIGNAL_COLS}, run_id=run_of.get(s["stage"]))
        row["detail"] = dict(row["detail"] or {}, **({"followed": s["followed"]} if taker else {}),
                             **({"backfill": True} if history else {}), **({"venue": venue} if venue else {}))
        row["status"] = row["status"] or "new"
        if history:
            row["status"] = row["status"] if not taker else ("filled_paper" if s["followed"] else "passed")
            row["seen_at"] = row["alerted_at"] = pd.Timestamp.now(tz="UTC").to_pydatetime()
        r = conn.execute(pg_insert(m.StrategySignal).values(row).on_conflict_do_nothing(
            index_elements=["user_id", "candidate_id", "market_key", "dedupe", "action", "side"])
            .returning(m.StrategySignal.id)).scalar()
        if r is not None:
            new.append(r)
    if out["stages"] and taker and not history:
        conn.execute(text("""UPDATE strategy_signals SET status = 'expired'
                             WHERE user_id = :u AND candidate_id IS NOT DISTINCT FROM :c AND event_key = :e
                               AND status IN ('new', 'alerted') AND stage <> :s"""),
                     dict(u=user_id, c=pr.get("candidate_id"), e=w["event_key"], s=out["stages"][-1][0]))
    # only this run's venue: a Polymarket run (venue None) replaces Polymarket's rows, never Kalshi's, OG.com's or
    # the live maker's ('private') for the same event
    conn.execute(text("""DELETE FROM paper_positions WHERE user_id = :u AND candidate_id IS NOT DISTINCT FROM :c
                         AND event_key = :e AND venue = :v"""),
                 dict(u=user_id, c=pr.get("candidate_id"), e=w["event_key"], v=venue or "polymarket"))
    for p in positions:
        row = dict(user_id=user_id, candidate_id=pr.get("candidate_id"), race_id=out.get("race_id"),
                   event_key=w["event_key"], **{k: _clean(v) for k, v in p.items()})
        if venue:
            row.setdefault("venue", venue)
        conn.execute(pg_insert(m.PaperPosition).values(row))
    return new


def run_all(engine, engine_url, users=None, profile_ref=None, now=None, event="next", fetch=True, alert=True,
            echo=print, venue="polymarket"):
    """Every active user with a profile (or `users`, usernames), one computation per distinct profile (a blend:
    one per member, see compute_all). venue: whose profiles (profiles.pref(venue); Polymarket, the timer's pass, by
    default); each still trades its own `venue` setting."""
    from racinglines.pipelines import profiles as PF
    with engine.connect() as c:
        targets = [(uid, prof) for uid, name, _, prof in PF.assigned(c, venue=venue) if not users or name in users]
        names = {uid: name for uid, name, _, _ in PF.assigned(c, venue=venue)}
        if profile_ref is not None:            # this profile instead of the users' own
            prof = PF.load(c, profile_ref)
            targets = [(uid, prof) for uid, _ in targets]
    groups, rate, kelly = {}, {}, {}
    cache = {}                         # one measurements load and one training history per model for the whole pass
    for uid, prof in targets:
        groups.setdefault(combo_key(prof), (prof, []))[1].append(uid)
        rate[uid] = prof.get("follow_rate")
        kelly[uid] = PF.sizing_of(prof)                # a player's sizing choice (for_player; None = standard)
    report = []
    for prof, uids in groups.values():
        # the Lab monitor accounts (profiles.ensure_monitors) are stored and scored like anyone, but alert nobody
        alert_g = alert and any(not names.get(u, "").startswith(PF.MONITOR_PREFIX) for u in uids)
        if PF.is_combo(prof):
            echo(f"{prof['name']} (blend of {', '.join(m['code'] for m in prof['members'])}) for users {uids}")
            outs = compute_all(engine, engine_url, prof, now=now, event=event, fetch=fetch, echo=echo, cache=cache)
            report += [_run_one(engine, engine_url, out["profile"], out, uids, rate, now, alert_g, echo, kelly=kelly,
                                guard=guard(prof, venue) if profile_ref is None else None) for out in outs]
            continue
        echo(f"{prof['name']} ({prof['strategy']}) for users {uids}")
        out = compute(engine, engine_url, prof, now=now, event=event, fetch=fetch, echo=echo, cache=cache)
        report.append(_run_one(engine, engine_url, prof, out, uids, rate, now, alert_g, echo, kelly=kelly,
                               guard=guard(prof, venue) if profile_ref is None else None))
    return report


def guard(prof, venue="polymarket", sport=None):
    """What _run_one re-checks before it stores: the group's profile (combo_key) and where it lives in users.prefs."""
    from racinglines.pipelines import profiles as PF
    return dict(key=combo_key(prof), venue=venue, sport=sport or PF.DEFAULT_SPORT)


def still_assigned(conn, uids, key, venue="polymarket", sport=None):
    """The users in `uids` who still run the profile `key` (combo_key), read with their rows locked FOR SHARE until the
    caller's transaction ends. A pass loads the profiles first and computes for minutes; a fantasy reset
    (web/accounts.reset, which locks the same rows FOR UPDATE) or a new pick in between would otherwise get the old
    profile's positions written back (cback, race 17 of 2026: written 36 s after the 12 Oct reset)."""
    from racinglines.pipelines import profiles as PF
    k = PF.pref(venue, sport or PF.DEFAULT_SPORT)
    rows = conn.execute(text(f"SELECT id, prefs->'{k}' FROM users WHERE id = ANY(:u) AND active ORDER BY id FOR SHARE"),
                        dict(u=list(uids))).all()
    return [uid for uid in uids if any(i == uid and p and combo_key(p) == key for i, p in rows)]


def _run_one(engine, engine_url, prof, out, uids, rate, now, alert, echo, upcoming="f1", kelly=None, guard=None):
    """Store one computation for its users and alert the new signals. -> the report row. upcoming: price the next
    races between weekends (price_upcoming, the F1 Markets page's calls); None for another sport's run. guard (see
    guard()): store only for the users who still run this profile when the store starts (still_assigned)."""
    from racinglines.markets import alerts
    if out.get("note"):
        echo(f"  {out['note']}")
    if upcoming and prof["strategy"] in WS.TAKER_MODES and not out["stages"]:
        try:                                         # between weekends: the Markets page's current calls
            price_upcoming(engine, engine_url, prof, now=now, echo=echo)
        except Exception as ex:                      # noqa: BLE001
            echo(f"  pricing upcoming races failed: {ex}")
    if not out["stages"]:
        return dict(profile=prof["name"], note=out.get("note"), new=0)
    with engine.begin() as c:
        if guard is not None:
            keep = still_assigned(c, uids, **guard)
            if len(keep) < len(uids):
                echo(f"  users {[u for u in uids if u not in keep]} no longer run {prof['name']}: not stored for them")
            uids = keep
        new = {uid: store(c, uid, for_player(c, uid, out, kelly=(kelly or {}).get(uid)), follow_rate=rate[uid])
               for uid in uids}
    if not uids:
        return dict(profile=prof["name"], stage=out["stages"][-1][0], signals=len(out["signals"]), new=0)
    n_new = sum(len(v) for v in new.values())
    echo(f"  {out['event']['name']} · {out['stages'][-1][0]} · {len(out['signals'])} signals "
         f"({n_new} new) · {len(out['positions'])} positions")
    if alert and n_new:
        with engine.begin() as c:
            ids = [i for v in new.values() for i in v]
            rows = c.execute(text("SELECT * FROM strategy_signals WHERE id = ANY(:i) ORDER BY id"),
                             dict(i=ids)).mappings().all()
            used = alerts.signals_alert(prof, out["event"], [dict(r) for r in rows if r["user_id"] == uids[0]])
            c.execute(text("""UPDATE strategy_signals SET status = CASE WHEN status = 'new' THEN 'alerted'
                              ELSE status END, alerted_at = now() WHERE id = ANY(:i)"""), dict(i=ids))
        echo(f"  alerted via {', '.join(used) or 'nothing'}")
    return dict(profile=prof["name"], stage=out["stages"][-1][0], signals=len(out["signals"]), new=n_new)


# ---------------------------------------------------------------------------
# A profile's current call on any market (the taker's Markets page)
# ---------------------------------------------------------------------------

NOW_STAGE = "now"                         # sweep_stage of an as-of-now pricing between race weekends


def call(profile, kind, fair, price, volume_24h=None):
    """What a taker profile would do in one market at `price`: dict(action, side, shares, limit, heat,
    why). Same rule as the backtest taker (sizing.target_shares, the h2h threshold, the price band and
    the volume filter) and the live pass's venue costs (taker_params: the profile venue's cost per share and taker
    fee). Never returns the fair value or edge (takers don't see them)."""
    from racinglines.markets.strategies.sizing import target_shares
    st = SS.Settings.from_dict(profile["settings"], strict=False)
    venue = SS.venue_of(st)
    p = RB.TakerParams(min_edge=st["min_edge"], stake_per_edge=st["stake_per_edge"], max_stake=st["max_stake"],
                       cost=WS.taker_cost(st["cost"], venue), min_edge_h2h=st["min_edge_h2h"],
                       min_edge_by_kind=tuple(SS.parse_map(st["min_edge_by_kind"]).items()),
                       taker_fee=WS.taker_fee(venue))
    p = RB.params_for(kind, p)
    if kind not in st["market_kinds"]:
        return dict(action=None, why="not a market this strategy trades")
    if fair is None:
        return dict(action=None, why="not priced yet")
    if price is None or not (p.price_band[0] <= price <= p.price_band[1]):
        return dict(action=None, why="outside the price band")
    if volume_24h is not None and volume_24h < st["min_volume_24h"]:
        return dict(action=None, why="too thin to trade")
    ty, tn = target_shares(fair, price, p.cost + p.taker_fee * price * (1 - price), p.min_edge, p.stake_per_edge,
                           p.max_stake)                       # as taker_weekend sizes it at one price
    if ty <= 0 and tn <= 0:
        return dict(action=None, why="no edge")
    side, shares = ("YES", ty) if ty > 0 else ("NO", tn)
    px = price if side == "YES" else 1 - price
    limit = px + p.cost + RB._fee(p.taker_fee, px, shares)   # what the live pass pays per share (its limit_price)
    return dict(action="buy", side=side, shares=shares, limit=limit, cost=shares * limit,
                heat=heat("buy", side, shares, limit, fair), why=None)


def latest_run(conn, profile, event_key):
    """The profile's most recent pricing of an event (a live stage, or an as-of-now view): run id or None."""
    st = SS.Settings.from_dict(profile["settings"], strict=False)
    return conn.execute(text("""SELECT id FROM model_runs WHERE kind = 'diagnostic' AND params->>'model_key' = :m
                                  AND params->>'event_key' = :k AND params ? 'sweep_stage'
                                ORDER BY (params->>'cutoff')::timestamp DESC, id DESC LIMIT 1"""),
                        dict(m=st.model_key, k=event_key)).scalar()


def price_upcoming(engine, engine_url, profile, now=None, n=3, echo=print, cache=None):
    """Between race weekends: price each of the next `n` races that has race markets listed on the profile's
    venue, as of now, with the profile's model, unless a pricing from the same data by the same code exists (the
    data only changes when a session runs and the code only when a deploy lands, so this is a one-off per race
    per deploy: the Markets board reads the newest of these runs, so a model change reaches it at the next pass).
    Returns {event_key: run id}."""
    from racinglines.db.queries import code_version
    from racinglines.models.inputs import laps_pricing as run
    st = SS.Settings.from_dict(profile["settings"], strict=False)
    venue = SS.venue_of(st)
    now = _naive_utc(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    sched = WS.schedule(now.year)
    todo = [w for r, w in sorted(sched.items()) if w["race_start"] > now][:n]
    out = {}
    with engine.connect() as c:
        listed = {w["event_key"] for w in todo if (rid := WS._race_id(c, w["event_key"]))
                  and len(WS._links(c, rid, venue))}
    todo = [w for w in todo if w["event_key"] in listed]
    if not todo:
        return out
    cache = {} if cache is None else cache
    meas = cache["meas"] if "meas" in cache else run.Measurements.load(engine)
    cache["meas"] = meas
    with st.applied():
        hist = cache.get(("hist", st.model_key))
        if hist is None:
            hist = cache[("hist", st.model_key)] = run.history(meas, st["track_features"])
        dk, cv = SS.data_key(meas.view(now)), code_version()
        for w in todo:
            with engine.connect() as c:
                have = c.execute(text("""SELECT id FROM model_runs WHERE kind = 'diagnostic' AND params->>'model_key' = :m
                                           AND params->>'event_key' = :k AND params->>'data_key' = :d
                                           AND code_version IS NOT DISTINCT FROM :cv ORDER BY id DESC LIMIT 1"""),
                                 dict(m=st.model_key, k=w["event_key"], d=dk, cv=cv)).scalar()
            if have:
                out[w["event_key"]] = have
                continue
            year, rnd = (int(x) for x in w["event_key"].split("-"))
            ev = meas.res[(meas.res["year"] == year) & (meas.res["series_round"] == rnd)]
            event_id = int(ev["event_id"].iloc[0]) if len(ev) else None
            summ, ex = run.price_race(meas, hist, now, event_id, n_sims=st["sims"], rng=np.random.default_rng(st.rng_seed),
                                      use_track=st["track_features"], entrants=_entrants(meas, event_id, now),
                                      venue=_venue(meas, event_id, w), stages=_stages(w), event_key=w["event_key"])
            extra = dict(model_key=st.model_key, data_key=dk, live=True,
                         model_settings={k: st.to_json()[k] for k in SS.MODEL_NAMES})
            if st["variant"] != "baseline":
                extra["variant"] = st["variant"]
            out[w["event_key"]] = run.save_diagnostic(engine_url, w["event_key"], now, summ, ex, st["sims"],
                                                      sweep_stage=NOW_STAGE, **extra)
            log_live_price(engine_url, w["event_key"], out[w["event_key"]], st, dk, code_version=cv, stage=NOW_STAGE)
            echo(f"  {w['name']}: priced as of now for {profile['name']} (run #{out[w['event_key']]})")
    return out


def log_live_price(engine_url, event_key, run_id, st, data_key, code_version=None, stage=None, sport="f1"):
    """A data_changes row ("live-price") for a live reprice the signal engine just stored. Called only when a new
    run is saved: a stage's pricing is stored once (reused by every later 5-minute pass) and an as-of-now pricing
    only when the data or the code changed, so this is a handful of rows per weekend, not one per pass."""
    from racinglines.db import changes
    from racinglines.db.config import get_session
    from racinglines.db.queries import code_version as cv_now
    with get_session(engine_url) as s:
        changes.record_run(s, "live-price", sport, [event_key], run_id, variant=st["variant"], sims=st["sims"],
                           code_version=code_version or cv_now(), data_key=data_key, stage=stage,
                           model_key=st.model_key)
        s.commit()


def maker_call(profile, kind, fair, price, volume_24h=None):
    """Where a maker profile would quote one market at the current price: dict(action='quote', bid, ask,
    size) or dict(action=None, why). Same rules as maker_replay.quote (spread around fair, never crossing
    the market, the price band, the volume floor, the disagreement filter), with no inventory yet."""
    st = SS.Settings.from_dict(profile["settings"], strict=False)
    p = WS.maker_params(st, SS.venue_of(st))             # the replay's rules, K's volume floor included
    if kind not in st["market_kinds"] or kind == "race_pole":
        return dict(action=None, why="not a market this strategy quotes")
    if fair is None:
        return dict(action=None, why="not priced yet")
    if price is None or not (p.price_band[0] <= price <= p.price_band[1]):
        return dict(action=None, why="outside the price band")
    if volume_24h is not None and volume_24h < p.min_volume_24h:
        return dict(action=None, why="too thin to quote")
    if p.max_disagree is not None and abs(fair - price) > p.max_disagree:
        return dict(action=None, why="model and market disagree: stays out")
    bid = min(np.floor(round((fair - p.half_spread) / p.tick, 6)) * p.tick, np.ceil(round(price / p.tick, 6)) * p.tick - p.tick)
    ask = max(np.ceil(round((fair + p.half_spread) / p.tick, 6)) * p.tick, np.floor(round(price / p.tick, 6)) * p.tick + p.tick)
    return dict(action="quote", bid=round(float(bid), 2) if bid >= p.tick else None,
                ask=round(float(ask), 2) if ask <= 1 - p.tick else None, size=p.size, why=None)
