"""
Weekend sweep: every race of a season, traded through the weekend.

For each race:
  1. Stages from the real session schedule (FastF1): before any running, then
     DATA_LAG after each session ends (practice, sprint qualifying, sprint,
     qualifying). The race itself is never traded.
  2. At each stage, price the race as of that moment (inputs.laps_pricing.price_race: only data
     from sessions that started before the cutoff) and store it as a
     kind='diagnostic' run tagged with the stage.
  3. Read the venue at that moment (5-minute prices, trades for liquidity) and
     run the taker strategies in racinglines/markets/strategies/taker_weekend.py (update / hold / last) plus the
     maker replay (racinglines/markets/strategies/maker_replay.py) through the same stages.
  4. Only then read the result: settle, score, summarise.

The venue is Polymarket unless the settings say another exchange of markets/venue_replay.py EXCHANGES
(`--venue kalshi` / `--venue og`): then the race's links on that exchange are read (one market per token, never
grouped by condition_id, which is Kalshi's event ticker), its tape per market from data/archive/markets/<venue>/,
and the maker and taker pay its fees (Kalshi's schedule; OG.com's flat fee per contract on the taker's cost).
Nothing about the default changes.

This is the sweep's "sessions" engine: stages from a session schedule (FastF1's, sports/f1.toml [stages]) and one
diagnostic run per stage. pipelines/season_sweep.py is the entry point for any sport; a sport whose schema says
`[sweep] engine = "replay"` (NASCAR, MotoGP) runs there instead, on its [replay] stages and pricing model.

The model uses qualifying, sprint and race results; practice pace isn't in the
model yet, so our fair value only moves after sprint qualifying / sprint /
qualifying, while the market moves after every session.
"""

import copy
import math
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines import progress as PG
from racinglines import sports
from racinglines.markets.strategies import taker_weekend as RB
from racinglines.core import calibration as CAL
from racinglines.core import stages as STG

_SCHEDULE = sports.load("f1")["sessions"]["schedule"]       # FastF1 session name -> [minutes, short label]
SESSION_MINUTES = {name: m for name, (m, _) in _SCHEDULE.items()}
SHORT = {name: short for name, (_, short) in _SCHEDULE.items()}
_STAGES = STG.spec("f1")                                     # sports/f1.toml [stages] (core/stages.py)
DATA_LAG = timedelta(minutes=_STAGES["lag_minutes"])         # a session's data is used 30 min after it ends
PRE = timedelta(minutes=_STAGES["pre_minutes"])              # "before any running": 1 h before the first session
KINDS = tuple(sports.load("f1")["markets"]["weekend_kinds"])
GROUP_TARGET = {"race_win": 1, "race_pole": 1, "race_constructor_top": 1, "race_podium": 3}
COHERENCE_TOL = 0.25
OPT_KINDS = ("race_top10",)                           # opt-in kinds (market_kinds): Kalshi's top-10 finishers
OPT_GROUP_TARGET = {"race_top10": 10}                 # ... a group of ten finishers
STALE = timedelta(hours=6)
MIN_VOLUME_24H = 50.0                # $ traded in the market over the previous 24 h
TAKER_MODES = ("update", "hold", "last", "early")
# maker settings replayed side by side (docs/f1-roadmap.md, F1-4); all chosen a priori, not tuned
MAKERS = {"maker": {},
          "maker_flat": dict(flatten_before_qual=True),        # go flat at the market before qualifying
          "maker_skew": dict(info_skew=2.0),                    # skew harder as each session approaches
          "maker_widen": dict(widen=True),                      # wider where earlier weekends' markouts were bad
          "maker_all": dict(flatten_before_qual=True, info_skew=2.0, widen=True)}
WIDEN = 1.5                          # half-spread multiplier for widened market kinds
FILL_COLS = ["ts", "cond", "side", "price", "qty", "mid", "kind", "subject", "run_id"]   # a maker fill, kept for CLV


from racinglines.sources.fastf1.fetch import SCHEDULE_COLS  # noqa: E402,F401  (kept importable from here)


def _event_schedule(year):
    """FastF1's event schedule, saved next to the raw files (data/raw/f1/fastf1/<year>/schedule.parquet)
    so a machine without access to FastF1's servers (e.g. a cloud session) can still build the sweep
    (sources/fastf1/fetch.event_schedule; RACINGLINES_OFFLINE=1 reads only the saved copy)."""
    from racinglines.sources.fastf1.fetch import event_schedule
    return event_schedule(year)


def schedule(year, rounds=None):
    """{round: dict(event_key, name, format, race_start, sessions, stages=[(label, cutoff)])} (naive UTC)."""
    sch = _event_schedule(year)
    out = {}
    for ev in sch.itertuples():
        rnd = int(ev.RoundNumber)
        if rounds and rnd not in rounds:
            continue
        sessions = []
        for i in range(1, 6):
            name, start = getattr(ev, f"Session{i}"), getattr(ev, f"Session{i}DateUtc")
            if name and not pd.isna(start):
                sessions.append((name, pd.Timestamp(start).tz_localize(None)))
        b = STG.build(sessions, "f1")                        # the stages, from the schema
        if b is None:
            continue
        out[rnd] = dict(event_key=f"{year}-{rnd:02d}", name=ev.EventName, format=ev.EventFormat, race_start=b["until"],
                        qual_start=next((t for n, t in sessions if n == "Qualifying"), None), closes=b["closes"],
                        sessions=[(SHORT.get(n, n), t) for n, t in sessions], stages=b["stages"],
                        pre_race=STG.pre_race(b["until"], "f1", b["stages"][-1][1] if b["stages"] else None))
    return out


def price_stages(meas, hist, sched, engine, engine_url=None, n_sims=4000, reprice=False, echo=print,
                 variant="baseline", settings=None):
    """Diagnostic run per (race, stage); a stored one is reused only when it was priced with the same
    model settings (model_key) from the same data (data_key), unless reprice. hist: the rating history,
    or a function returning it, called only when a stage has to be priced (a sweep whose stages are all
    stored never builds it). Returns {event_key: [(label, cutoff, run_id), ...]}."""
    from racinglines.models.inputs import laps_pricing as run
    from racinglines.pipelines import sweep_settings as SS
    st = settings or SS.Settings.from_dict({"variant": variant, "sims": n_sims})
    mk = st.model_key
    with engine.connect() as c:
        have = pd.read_sql(text("""SELECT id, params->>'event_key' AS k, params->>'cutoff' AS cutoff,
                                          params->>'data_key' AS d
                                   FROM model_runs WHERE kind = 'diagnostic' AND params ? 'sweep_stage'
                                     AND params->>'model_key' = :m ORDER BY id"""), c, params=dict(m=mk))
    have = {(k, str(pd.Timestamp(cu)), d): int(i) for i, k, cu, d in zip(have["id"], have["k"], have["cutoff"], have["d"])}
    out, data_keys = {}, []
    for rnd, w in PG.track(list(sched.items()), "round", name=lambda x: x[1]["event_key"]):
        ev = meas.res[(meas.res["year"] == int(w["event_key"][:4])) & (meas.res["series_round"] == rnd)]
        if ev.empty or not (ev["round"] == "race").any():
            continue                                     # not raced yet
        runs = []
        for label, cutoff in w["stages"]:
            dk = SS.data_key(meas.view(cutoff))
            data_keys.append(dk)
            k = (w["event_key"], str(pd.Timestamp(cutoff)), dk)
            if k in have and not reprice:
                runs.append((label, cutoff, have[k]))
                continue
            if callable(hist):
                hist = hist()
            _, summ, ex, _ = run.diagnostic(meas, hist, w["event_key"], cutoff, n_sims=st["sims"],
                                            use_track=st["track_features"], seed=st.rng_seed)
            extra = dict(model_key=mk, data_key=dk, model_settings={n: st.to_json()[n] for n in SS.MODEL_NAMES})
            if st["variant"] != "baseline":
                extra["variant"] = st["variant"]
            rid = run.save_diagnostic(engine_url, w["event_key"], cutoff, summ, ex, st["sims"], sweep_stage=label,
                                      **extra)
            runs.append((label, cutoff, rid))
        out[w["event_key"]] = runs
        echo(f"progress {len(out)}/{len(sched)} priced {w['event_key']} {w['name']}: "
             + ", ".join(f"{lab} #{rid}" for lab, _, rid in runs))
    price_stages.data_key = __import__("hashlib").sha1("|".join(data_keys).encode()).hexdigest()[:12]
    return out


def _token0_links(conn, race_id, kinds=None):
    return _links(conn, race_id, "polymarket", kinds)


def _links(conn, race_id, venue="polymarket", kinds=None):
    """The race's tradeable links on a venue, one row per market (the exchange's MARKET_KEY): Polymarket's first
    token per condition, or every Kalshi / OG.com token (a Kalshi condition_id is the event ticker, shared by all
    its markets)."""
    cls = _exchange(venue)
    return cls.links(conn, race_id, kinds or KINDS).drop_duplicates(cls.MARKET_KEY, keep="first")


def _exchange(venue):
    from racinglines.markets.venue_replay import EXCHANGES
    if venue not in EXCHANGES:
        raise ValueError(f"unknown venue {venue!r}: one of {sorted(EXCHANGES)}")
    return EXCHANGES[venue]


def _venue(conn, links, start, end, venue="polymarket", group_target=None, event_key=None, tape=None, levels=False):
    """The weekend's backtest venue: our recorded tape, or Probalytics' (F1's tag) when the tape_source setting
    (`tape`) or RACINGLINES_TAPE_SOURCE is probalytics (markets/probalytics.py; off by default)."""
    from racinglines.markets import probalytics as PB
    if PB.enabled(tape) and venue in PB.EXCHANGES:
        return PB.EXCHANGES[venue](conn, links, start, end, group_target or GROUP_TARGET, COHERENCE_TOL,
                                   event_key=event_key, tag=PB.tag_of("f1"), book_stale=PB.book_stale_of("f1"),
                                   levels=levels, day_books=False)   # stage files for every weekend: one rule,
                                                                     # and no day books in memory
    return _exchange(venue)(conn, links, start, end, group_target or GROUP_TARGET, COHERENCE_TOL, STALE)


def _race_id(conn, event_key):
    return conn.execute(text("SELECT ra.id FROM races ra JOIN events e ON e.id = ra.event_id WHERE e.source_key = :k"),
                        dict(k=event_key)).scalar()


def fetch_market_data(session, conn, sched, fidelity=5, force=False, echo=print):
    """Polymarket price history (one token per question) over each weekend, and the trade tape."""
    from racinglines.markets.polymarket.sync import fetch_history, fetch_trades
    for rnd, w in PG.track(list(sched.items()), "download round", name=lambda x: x[1]["event_key"]):
        rid = _race_id(conn, w["event_key"])
        if rid is None:
            continue
        links = _token0_links(conn, rid)
        if not len(links):
            continue
        start = (w["stages"][0][1] - timedelta(hours=36)).tz_localize("UTC").to_pydatetime()
        end = (w["race_start"] + timedelta(hours=3)).tz_localize("UTC").to_pydatetime()
        from racinglines.markets import store as MS
        n_have = len(MS.read(conn, "prices", tokens=links["token_id"].tolist(), start=start, end=end))
        if n_have and not force:
            echo(f"progress {w['event_key']} market data already stored ({n_have} points)")
            continue
        n = fetch_history(session, conn, None, start, end, fidelity=fidelity, tokens=links["token_id"].tolist())
        k = fetch_trades(session, conn, links["event_slug"].dropna().unique().tolist(), modeled_only=True)
        echo(f"progress {w['event_key']} {w['name']}: {n} price points, {k} trades")


def book_fill(st):
    """True when taker_fill = book: every order walks the full book, which only Probalytics' tape has (a ValueError
    on any other tape)."""
    from racinglines.markets import probalytics as PB
    if st.get("taker_fill") != "book":
        return False
    if not PB.enabled(st.get("tape_source")):
        raise ValueError("taker_fill = book walks the full book: it needs tape_source = probalytics")
    return True


def stage_levels(stage, venue, token, t):
    """taker_fill = book: the stage's whole book (`bids` / `asks`, best first) for the taker to walk."""
    b = venue.book_at(token, t) if hasattr(venue, "book_at") else None
    if b is not None:
        stage.update(bids=b["bids"], asks=b["asks"])
    return stage


def stage_quote(venue, token, t, max_spread=None):
    """(the venue's (bid, ask) at t or None, whether it is wider than max_spread)."""
    q = venue.quote(token, t)
    return q, q is not None and max_spread is not None and q[1] - q[0] > max_spread + 1e-9


def weekend_markets(conn, w, stage_runs, min_volume_24h=None, price_times=None, venue="polymarket", rules=None,
                    kinds=None, thin_depth=False, tol_by_kind=None, tape=None, taker_fill=None, max_spread=None):
    """The weekend's tradeable markets for racinglines/markets/strategies/taker_weekend.py: per market, each stage's fair,
    exchange price and tradeable flag (what was knowable then) and the outcome (settlement). The exchange is a
    backtest venue (markets/venue_replay.py: Polymarket's recorded prices and trade tape, or Kalshi's with
    `venue="kalshi"`: one market per ticker, its own 24 h volume).
    price_times: {stage label: time} to read the market at instead of the stage's cutoff (live signals:
    when the stage was priced, i.e. when its trades could first be made); None = the cutoffs (backtests).
    rules: a cancelled race's outcomes by the venue's rules (markets/settlement_rules.py: an outcome can then be
    a 0.5 payout or VOID); None reads RACINGLINES_CANCELLED_RACE_RULES, off by default.
    kinds: the market kinds to read (None = KINDS; an opt-in kind of OPT_KINDS is added by the caller and
    brings its own group target). thin_depth: read the recorded order books, and mark a stage that fails only
    the volume floor `thin` with the size at the touch (`depth_yes` / `depth_no`); the stage stays
    `tradeable=False`, so nothing but a taker with thin_edge_mult reads it. tol_by_kind: {kind: coherence
    tolerance} over COHERENCE_TOL. tape: the tape_source setting (None = ours, unless RACINGLINES_TAPE_SOURCE)."""
    at = lambda lab, cutoff: (price_times or {}).get(lab, cutoff)          # noqa: E731
    from racinglines.db import reads as D
    from racinglines.markets import private_book as house
    from racinglines.markets import settlement_rules as SR
    rid = _race_id(conn, w["event_key"])
    links = _links(conn, rid, venue, kinds)
    if not len(links):
        return None
    status = SR.race_status("f1", w["event_key"], SR.db_status(conn, rid)) if SR.enabled(rules) else None
    start, end = stage_runs[0][1] - timedelta(hours=1), max([w["race_start"], *(price_times or {}).values()])
    group_target = {**GROUP_TARGET, **{k: v for k, v in OPT_GROUP_TARGET.items() if k in (kinds or ())}}
    levels = taker_fill == "book"                           # every level of every book, for the walk
    venue = _venue(conn, links, start, end, venue, group_target, event_key=w["event_key"], tape=tape, levels=levels)
    if taker_fill in ("touch", "book") and hasattr(venue, "touch_quotes"):
        venue.touch_quotes = True                           # Probalytics' Polymarket book: buy at the ask
    venue.tol_by_kind = dict(tol_by_kind or {})
    if thin_depth:
        venue.load_books(conn, start, end)
    elif levels:                                            # the walk: only the books at the stages' price times
        venue.load_books(conn, start, end, times=[at(lab, cu) for lab, cu, _ in stage_runs])
    res = house.race_outcomes(conn, rid)
    cache, markets = {}, []
    fair = {}   # (token, stage) -> fair
    for lab, cutoff, run_id in stage_runs:
        for link in links.to_dict("records"):
            fair[(link["token_id"], lab)] = D.model_prob(conn, link, cache, run_id=run_id)[0]
    # coherence of each multi-outcome group at each stage (stale/empty books are skipped)
    coherent = {(kind, lab): venue.coherent(kind, at(lab, cutoff)) for lab, cutoff, _ in stage_runs for kind in group_target}
    vol_min = MIN_VOLUME_24H if min_volume_24h is None else min_volume_24h
    for link in venue.markets():
        kind = link["prediction"]
        stages = []
        for lab, cutoff, _ in stage_runs:
            t = at(lab, cutoff)
            price, _, liquid = venue.view(link, t, vol_min)
            f = fair[(link["token_id"], lab)]
            open_ = STG.is_open(kind, t, w.get("closes", {"race_pole": w["qual_start"]}))
            ok = liquid and f is not None and open_ and coherent.get((kind, lab), True)
            stage = dict(label=lab, t=t, fair=f, price=price, tradeable=ok, open=open_)
            q, wide = stage_quote(venue, link["token_id"], t, max_spread)
            if q is not None:                    # the taker buys at the ask and sells at the bid (Kalshi)
                stage.update(bid=q[0], ask=q[1])
                sz = venue.touch_sizes(link["token_id"], t) if hasattr(venue, "touch_sizes") else None
                if sz is not None:
                    stage.update(bid_size=sz[0], ask_size=sz[1])
            if wide:                             # max_spread: no market to fill at
                stage.update(tradeable=False, wide=True)
            if levels:
                stage_levels(stage, venue, link["token_id"], t)
            if thin_depth and not liquid and price is not None and 0 < price < 1 and f is not None and open_ \
                    and coherent.get((kind, lab), True):
                d = venue.touch_depth(link["token_id"], t)          # only the volume floor failed: is there size?
                if d is not None and (d[0] > 0 or d[1] > 0):
                    stage.update(thin=True, depth_yes=d[0], depth_no=d[1])
            stages.append(stage)
        subject = link["athlete"] or (link["params"] or {}).get("team") or link["group_title"]
        if kind == "race_h2h":
            subject = f"{link['outcome']} ({link['question'].split(': ')[-1]})"
        outcome = venue.resolve(link, res)
        if status:
            outcome = SR.apply(venue.code, status, kind, link.get("group_title") or link.get("outcome"), outcome,
                               last_price=venue.price(link["token_id"], end))
        markets.append(dict(key=link["token_id"], kind=kind, subject=subject, stages=stages, link=link, outcome=outcome))
    return markets


def weekend(conn, w, stage_runs, params_list, echo=print, widen_kinds=(), settings=None):
    """Trade one weekend. Returns dict(summary rows per mode, trades, stage scores, makers).
    widen_kinds: market kinds whose maker fills lost on 60-min markouts in EARLIER weekends.
    settings: sweep_settings.Settings (market kinds, volume filter, maker parameters); None = defaults."""
    from racinglines.pipelines import sweep_settings as SS
    st = settings or SS.Settings.from_dict()
    venue = SS.venue_of(st)
    rids = tuple(r for _, _, r in stage_runs)
    opt = tuple(k for k in OPT_KINDS if k in st["market_kinds"])
    kinds = KINDS + opt if opt else None                       # opt-in kinds only when asked for
    thin = st["thin_edge_mult"] is not None
    tol = SS.parse_map(st["coherence_tol_by_kind"])
    tape = st.get("tape_source")
    fill, spread = st.get("taker_fill"), st.get("max_spread")
    book_fill(st)                                           # taker_fill = book: Probalytics' tape or a ValueError
    markets = _memo(("markets", w["event_key"], rids, st["min_volume_24h"], venue, kinds, thin,
                     tuple(sorted(tol.items())), tape, fill, spread),
                    lambda: weekend_markets(conn, w, stage_runs, min_volume_24h=st["min_volume_24h"], venue=venue,
                                            kinds=kinds, thin_depth=thin, tol_by_kind=tol, tape=tape,
                                            taker_fill=fill, max_spread=spread))
    if markets is None:
        return None
    markets = [m for m in markets if m["kind"] in st["market_kinds"]]
    from racinglines.markets.strategies import maker_replay as R
    tape = lambda: _memo(("tape", w["event_key"], rids, st["fill"] == "queue", venue),          # noqa: E731
                         lambda: R.load_event(conn, list(rids), books=st["fill"] == "queue",
                                              **({} if venue == "polymarket" else dict(exchange=venue))))
    return trade(markets, [lab for lab, _, _ in stage_runs], params_list, st, tape, widen_kinds, KINDS, echo,
                 stage_of={rid: lab for lab, _, rid in stage_runs})


def trade(markets, labels, params_list, st, tape, widen_kinds=(), kinds=KINDS, echo=print, stage_of=None):
    """Trade one event's markets, any sport: every taker mode of params_list (and the buy_all debug mode), the
    model and the market scored per stage and kind (`kinds`), the calibration rows, and the maker variants (MAKERS)
    replayed on the event's tape. markets: [dict(key, kind, subject, stages, outcome)] (weekend_markets, or
    season_sweep's); labels: the stage labels in order; tape(): the maker replay's event (markets, stages), raising
    when the event has no tape; stage_of: {the tape's stage run_id: label} (None: run_id i is labels[i]). Returns
    dict(modes, trades, scores, calib, markets, tradeable_first / _last, makers, maker_fills (each maker variant's
    fills with their stage label, for CLV), markout_by_kind, maker)."""
    from racinglines.pipelines import sweep_settings as SS
    venue = SS.venue_of(st)
    out = dict(modes={}, trades={})
    for p in params_list:
        tr, per = RB.run_weekend(markets, p)
        out["modes"][p.mode] = RB.summarize(tr, per)
        out["trades"][p.mode] = tr
    from racinglines.markets.strategies import buy_everything as BA
    if BA.enabled() and params_list:           # debug, off by default: one YES and one NO of every priced market
        p = params_list[0]
        tr, per = BA.run_weekend(markets, cost=p.cost, taker_fee=p.taker_fee)
        out["modes"][BA.MODE] = RB.summarize(tr, per)
        out["trades"][BA.MODE] = tr
    # model vs market at each stage, scored on the result (every kind; pole before qualifying)
    scores = []
    for lab in labels:
        for kind in kinds:
            rows = [(s["fair"], s["price"], float(m["outcome"])) for m in markets if m["kind"] == kind
                    for s in m["stages"] if s["label"] == lab and s["fair"] is not None and s["price"] is not None
                    and m["outcome"] is not None and s["tradeable"]]
            if len(rows) >= 5:
                f, pm, y = np.array(rows).T
                scores.append(dict(stage=lab, kind=kind, n=len(rows), brier_model=float(np.mean((f - y) ** 2)),
                                   brier_market=float(np.mean((pm - y) ** 2))))
    out["scores"] = scores
    # every tradeable (fair, price, outcome), for calibration (run_sweep's calibration tables)
    out["calib"] = [dict(stage=s["label"], kind=m["kind"], fair=s["fair"], price=s["price"], y=float(m["outcome"]))
                    for m in markets if m["outcome"] is not None
                    for s in m["stages"] if s["fair"] is not None and s["price"] is not None and s["tradeable"]]
    out["markets"] = len(markets)
    out["tradeable_first"] = sum(m["stages"][0]["tradeable"] for m in markets)
    out["tradeable_last"] = sum(m["stages"][-1]["tradeable"] for m in markets)
    # maker replays through the same stages (conservative fills), one per setting
    out["makers"], out["maker_fills"], out["markout_by_kind"] = {}, {}, {}
    stage_of = stage_of if stage_of is not None else dict(enumerate(labels))
    try:
        from dataclasses import replace

        from racinglines.markets.strategies import maker_replay as R
        ev = tape()
        ev = dict(ev, markets=[m for m in ev["markets"] if m.kind in st["market_kinds"]])
    except Exception as ex:  # noqa: BLE001  (no tape for this weekend)
        echo(f"  maker replay skipped: {ex}")
        ev = None
    base = maker_params(st, venue)
    for name, opts in MAKERS.items() if ev is not None else []:
        opts = dict(opts)
        p = base
        if "info_skew" in opts:
            opts["info_skew"] = st["info_skew"]
        if opts.pop("widen", False):
            opts["half_spread_by_kind"] = {k: p.half_spread * st["widen"] for k in widen_kinds}
        rep = R.replay(ev, replace(p, **opts))
        sk = R.summary(rep)
        s = sk.loc["total"]
        out["makers"][name] = dict(fills=int(s["fills"]), notional=float(s["notional"]), pnl=float(s["pnl"]),
                                   markout_60m=float(s["markout_60m"]), spread_pnl=float(s["spread_pnl"]))
        f = rep["fills"]
        out["maker_fills"][name] = f.reindex(columns=FILL_COLS).assign(
            stage=f["run_id"].map(lambda i: stage_of.get(i, str(i))) if len(f) else None)
        if name == "maker":
            out["markout_by_kind"] = sk.drop(index="total")["markout_60m"].to_dict()
    out["maker"] = out["makers"].get("maker")
    return out


def taker_fee(venue):
    """The venue's taker fee rate for TakerParams.taker_fee (venue_replay.EXCHANGES: Kalshi 0.07, Polymarket 0)."""
    from racinglines.markets.venue_replay import EXCHANGES
    return EXCHANGES[venue].TAKER_FEE


def taker_cost(cost, venue):
    """The taker's cost per share on a venue: the settings' `cost` plus the venue's flat fee per contract
    (venue_replay: OG.com's; nothing on Polymarket or Kalshi, whose fee is the rate in taker_fee)."""
    from racinglines.markets.venue_replay import EXCHANGES
    fee = EXCHANGES[venue].per_share_fee()
    return cost + fee if fee else cost


def maker_venue_opts(venue):
    """maker_replay.Params overrides for a venue: its maker fee per fill, from its fee schedule
    (venue_replay.EXCHANGES; Kalshi's MAKER_FEE, Polymarket none)."""
    from racinglines.markets.venue_replay import EXCHANGES
    fee = EXCHANGES[venue].MAKER_FEE
    return dict(maker_fee=fee) if fee else {}


def maker_params(st, venue="polymarket"):
    """The maker replay's parameters from sweep settings: the quote settings, and the venue's maker fee.
    `maker_min_volume_24h` unset keeps the replay's own $100 filter (the markets group's `min_volume_24h`
    is the takers' filter)."""
    from racinglines.markets.strategies import maker_replay as R
    p = R.Params(half_spread=st["half_spread"], size=st["size"], max_pos=st["max_pos"], skew=st["skew"],
                 max_disagree=st["max_disagree"], fill=st["fill"], **maker_venue_opts(venue))
    if st["maker_min_volume_24h"] is not None:
        p = replace(p, min_volume_24h=st["maker_min_volume_24h"])
    return p


HISTORY_CACHE = "RACINGLINES_HISTORY_CACHE"      # =1: keep rating histories on disk (data/cache/history/)


def history(meas, st, echo=print):
    """The model's rating history (pricing.history) under settings `st`, which must be applied. With
    RACINGLINES_HISTORY_CACHE=1 it is kept in data/cache/history/, keyed by the settings' model_key and the
    fingerprint of all the data (sweep_settings.data_key) and clean checkout's code version. Unknown, dirty or
    untracked versions bypass the disk cache. Off, it is built every time, as before."""
    import hashlib
    import os

    from racinglines import paths
    from racinglines.db.queries import code_version
    from racinglines.models.inputs import laps_pricing as run
    from racinglines.pipelines import sweep_settings as SS
    if os.environ.get(HISTORY_CACHE, "0") != "1":
        return run.history(meas, st["track_features"])
    version = code_version()
    if not version or version.endswith(("-dirty", "-untracked")):
        echo("progress history cache bypassed: checkout code version is unknown or not clean")
        return run.history(meas, st["track_features"])
    code_key = hashlib.sha256(version.encode()).hexdigest()[:16]
    f = paths.cache("history") / f"{st.model_key}-{SS.data_key(meas.view(pd.Timestamp('2100-01-01')))}-{code_key}.pkl"
    if f.exists():
        echo(f"progress history from cache ({f.name})")
        return pd.read_pickle(f)
    hist = run.history(meas, st["track_features"])
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")            # parallel jobs: write aside, then rename
    pd.to_pickle(hist, tmp)
    tmp.replace(f)
    return hist


_SHARED = None                 # inside shared(): inputs reused between the sweeps of one process


@contextmanager
def shared():
    """Several run_sweep calls in one process (`f1 sweep --grid`) that share what doesn't depend on the
    strategy settings: the measurements, the stage pricings of one model, each weekend's markets and each
    weekend's maker tape. Every sweep's results are the same as its own process would give."""
    global _SHARED
    outer, _SHARED = _SHARED, {} if _SHARED is None else _SHARED
    try:
        yield
    finally:
        _SHARED = outer


def _memo(key, make):
    """make() once per key inside shared(); a fresh copy for every caller (callers may change it)."""
    if _SHARED is None:
        return make()
    if key not in _SHARED:
        _SHARED[key] = make()
    return copy.deepcopy(_SHARED[key])


def bankroll_scale(start, balance, growth=None):
    """Stake multiplier for bankroll-aware sizing: the balance over the starting bankroll (0 once it's gone).
    growth "sqrt" (the bankroll_growth setting): the square root of that, square-root staking, so stake = C x sqrt(balance)
    x edge with C = stake_per_edge / sqrt(start): 1.0 at the start, 2.0 at 4x, 0.5 at a quarter, 0 once it's gone."""
    r = max(balance, 0.0) / start
    return math.sqrt(r) if growth == "sqrt" else r


def run_sweep(engine, engine_url, year, rounds=None, n_sims=4000, fetch=True, reprice=False, taker=None, echo=print,
              variant="baseline", settings=None):
    """The whole season (every raced weekend unless `rounds`). settings: sweep_settings.Settings, the
    full description of the combo (model, entry timing, taker, maker, markets); without it, one is
    built from n_sims / taker / variant. Returns dict(weekends DataFrame, by_stage, by_kind, totals,
    trades, scores, calibration, reliability, params). calibration / reliability: our fair values and the
    exchange's prices at every tradeable stage, scored on the outcome (core/calibration.py): per market kind
    and stage, and pooled per kind (stage "all"; a market counts once per stage it was tradeable)."""
    from racinglines.db.config import get_session

    from racinglines.models.inputs import laps_pricing as run
    from racinglines.pipelines import sweep_settings as SS
    if settings is None:
        t = taker or RB.TakerParams()
        settings = SS.Settings.from_dict(dict(variant=variant, sims=n_sims, min_edge=t.min_edge,
                                              stake_per_edge=t.stake_per_edge, max_stake=t.max_stake, cost=t.cost))
    st = settings
    if st.get("stages") not in (None, "sessions"):
        raise ValueError(f"stages {st['stages']!r}: this is the session-schedule sweep (stages = \"sessions\"); "
                         "the weekend mode needs [replay] stages in the sport's schema (pipelines/season_sweep.py)")
    sched = schedule(year, rounds)
    if fetch and SS.venue_of(st) != "polymarket":
        echo(f"progress {SS.venue_of(st)}: its tape is what the recorder and archive hold (nothing downloaded)")
    elif fetch:
        with engine.connect() as c, get_session(engine_url) as s:
            fetch_market_data(s, c, sched, echo=echo)
    echo(f"progress 0/1 building history ({st.label()})")
    meas = _SHARED.get("meas") if _SHARED is not None else None
    if meas is None:
        meas = run.Measurements.load(engine)
        if _SHARED is not None:
            _SHARED["meas"] = meas
    skey = ("stages", st.model_key, year, tuple(sched), reprice)
    if _SHARED is not None and skey in _SHARED:
        stage_runs, data_key = _SHARED[skey]
    else:
        with st.applied():
            stage_runs = price_stages(meas, lambda: history(meas, st, echo), sched, engine, engine_url,
                                      reprice=reprice, echo=echo, settings=st)
        data_key = price_stages.data_key
        if _SHARED is not None and not reprice:
            _SHARED[skey] = (stage_runs, data_key)
    base, params_list = taker_params(st)
    events = [dict(round=rnd, event_key=w["event_key"], event=w["name"], format=w["format"],
                   stages=len(stage_runs[w["event_key"]]), last_run_id=stage_runs[w["event_key"]][-1][2], w=w)
              for rnd, w in sched.items() if stage_runs.get(w["event_key"])]

    def trade_event(ev, plist, widen_kinds):
        with engine.connect() as c:
            return weekend(c, ev["w"], stage_runs[ev["event_key"]], plist, echo=echo, widen_kinds=widen_kinds,
                           settings=st)

    out = season(events, trade_event, st, params_list, STAGE_ORDER, total=len(sched), echo=echo)
    season_clv(engine, out, SS.venue_of(st), "f1", STAGE_ORDER, echo)
    return dict(out, params=dict({k: v for k, v in base.__dict__.items() if k != "stages" and not
                                  (k in ("scale", "max_deployed", "min_edge_by_kind", "thin_edge_mult", "kelly", "bankroll", "root_unit") and v == RB.TakerParams.__dataclass_fields__[k].default)},
                                 sport="f1", venue=SS.venue_of(st), n_sims=st["sims"],
                                 variant=st["variant"], data_lag_min=DATA_LAG.seconds // 60,
                                 min_volume_24h=st["min_volume_24h"], coherence_tol=COHERENCE_TOL,
                                 settings=st.to_json(), settings_key=st.key, model_key=st.model_key, data_key=data_key,
                                 label=st.label(), weekends=len(out["weekends"])))


STAGE_ORDER = ("pre-weekend", "after FP1", "after SQ", "after Sprint", "after FP2", "after FP3", "after Quali")


def taker_params(st):
    """(base TakerParams, one per TAKER_MODES) from sweep settings, any sport: the taker settings, the venue's fee
    (taker_fee, taker_cost), entry timing (`taker_stages`: None = every stage, the setting's default)."""
    from racinglines.pipelines import sweep_settings as SS
    venue = SS.venue_of(st)
    base = RB.TakerParams(min_edge=st["min_edge"], stake_per_edge=st["stake_per_edge"], max_stake=st["max_stake"],
                          cost=taker_cost(st["cost"], venue), late_stages=st["late_stages"],
                          min_edge_h2h=st["min_edge_h2h"],
                          min_edge_by_kind=tuple(SS.parse_map(st["min_edge_by_kind"]).items()),
                          stages=None if st["taker_stages"] == st.BY["taker_stages"].default else st["taker_stages"],
                          max_deployed=st["max_deployed"], thin_edge_mult=st["thin_edge_mult"],
                          kelly=st["kelly"], bankroll=st["bankroll"] if st["kelly"] is not None else None,
                          root_unit=st.get("root_unit"), fill="book" if st.get("taker_fill") == "book" else "price",
                          sell_floor=None if st.get("sell_floor") in (None, "none") else st.get("sell_floor"),
                          taker_fee=taker_fee(venue))
    if st["kelly"] is not None and st["bankroll"] is None:
        raise ValueError("kelly sizing needs a bankroll (the `bankroll` setting)")
    if st["kelly"] is not None and st.get("root_unit") is not None:
        raise ValueError("pick one stake rule: kelly or root_unit, not both")
    return base, [RB.TakerParams(**{**base.__dict__, "mode": m}) for m in TAKER_MODES]


def season(events, trade_event, st, params_list, stage_order=(), total=None, echo=print):
    """A season of events traded in order, any sport. events: [dict(round, event_key, event, format, stages,
    last_run_id, ...)]; trade_event(event, params_list, widen_kinds) -> trade()'s dict, or None (no markets).
    Bankroll-aware sizing carries each mode's balance from event to event; the maker_widen variant widens the
    kinds whose maker markouts were negative in EARLIER events. Returns dict(weekends, by_stage, by_kind, totals,
    trades, scores, calibration, reliability, strategy_trades: {strategy: every taker mode's trades and maker
    variant's fills, with event_key and stage}, for season_clv)."""
    balance = {q.mode: st["bankroll"] for q in params_list}     # bankroll-aware sizing: each mode's balance
    rows, all_trades, all_scores, all_calib = [], [], [], []
    by_strategy = {}                   # strategy -> [trades or fills per event], for CLV (season_clv)
    markouts = {}                      # market kind -> maker's 60-min markout so far (as of each weekend)
    for ev in events:
        plist = params_list if st["bankroll"] is None else \
            [replace(q, scale=bankroll_scale(st["bankroll"], balance[q.mode], st.get("bankroll_growth")))
             for q in params_list]
        r = trade_event(ev, plist, [k for k, v in markouts.items() if v < 0])
        if r is None:
            continue
        if st["bankroll"] is not None:
            for q in plist:
                balance[q.mode] += r["modes"][q.mode]["pnl"]
                r["modes"][q.mode].update(scale=q.scale, balance=balance[q.mode])
        for k, v in r["markout_by_kind"].items():
            markouts[k] = markouts.get(k, 0.0) + v
        row = dict(round=ev["round"], event_key=ev["event_key"], event=ev["event"], format=ev["format"],
                   stages=ev["stages"], markets=r["markets"], tradeable_pre=r["tradeable_first"],
                   tradeable_quali=r["tradeable_last"], last_run_id=ev["last_run_id"])
        for mode, s in r["modes"].items():
            row.update({f"{mode}_{k}": v for k, v in s.items()})
        for name, m in r["makers"].items():
            row.update({f"{name}_{k}": v for k, v in m.items()})
        rows.append(row)
        t = r["trades"]["update"]
        if len(t):
            all_trades.append(t.assign(event_key=ev["event_key"], event=ev["event"]))
        for name, t in [*r["trades"].items(), *r.get("maker_fills", {}).items()]:
            if len(t):
                by_strategy.setdefault(name, []).append(t.assign(event_key=ev["event_key"]))
        all_scores += [dict(s, event_key=ev["event_key"]) for s in r["scores"]]
        all_calib += [dict(c, event_key=ev["event_key"]) for c in r["calib"]]
        echo(f"progress {len(rows)}/{total or len(events)} traded {ev['event_key']} {ev['event']}: "
             f"update {row.get('update_pnl', 0):+.2f} · hold {row.get('hold_pnl', 0):+.2f} · "
             f"last {row.get('last_pnl', 0):+.2f} · maker {row.get('maker_pnl', float('nan')):+.2f}")
    weekends = pd.DataFrame(rows)
    trades = RB.core(pd.concat(all_trades, ignore_index=True)) if all_trades else pd.DataFrame()
    by_stage = RB.by(trades, "stage")
    if len(by_stage):
        by_stage["o"] = by_stage["stage"].map({s: i for i, s in enumerate(stage_order)}).fillna(99)
        by_stage = by_stage.sort_values("o").drop(columns="o")
    scores = pd.DataFrame(all_scores)
    score_stage = (scores.groupby(["kind", "stage"])[["n", "brier_model", "brier_market"]]
                   .agg({"n": "sum", "brier_model": "mean", "brier_market": "mean"}).reset_index()
                   if len(scores) else scores)
    calib = pd.DataFrame(all_calib, columns=["event_key", "stage", "kind", "fair", "price", "y"]).rename(
        columns=dict(fair="model", price="market"))
    cal_stage, _ = CAL.table(calib, ("model", "market"), by=("kind", "stage"))
    cal_all, rel = CAL.table(calib, ("model", "market"), by=("kind",))
    totals = {}
    for mode in (*TAKER_MODES, *MAKERS, "buy_all"):          # buy_all: only with RACINGLINES_BUY_ALL=1
        col = f"{mode}_pnl"
        if col in weekends:
            spent = f"{mode}_notional" if mode.startswith("maker") else f"{mode}_bought"
            totals[mode] = dict(pnl=float(weekends[col].fillna(0).sum()),
                                weekends_up=int((weekends[col] > 0).sum()), weekends=int(weekends[col].notna().sum()),
                                bought=float(weekends.get(spent, pd.Series(dtype=float)).fillna(0).sum()))
    return dict(weekends=weekends, by_stage=by_stage, by_kind=RB.by(trades, "kind"), totals=totals, trades=trades,
                scores=score_stage, calibration=pd.concat([cal_all.assign(stage="all"), cal_stage], ignore_index=True),
                reliability=rel, strategy_trades={k: pd.concat(v, ignore_index=True) for k, v in by_strategy.items()})


def trades_frame(out):
    """Every trade of a sweep's `strategy_trades` (each taker mode, maker variant) as one frame: the trade rows (with
    the touch bid / ask and sizes where the tape had them, and the market's outcome) plus mode, settings_key, venue,
    year (from the stage time), event_key and event. Empty frame when there are none."""
    p, ev = out.get("params") or {}, {r["event_key"]: r["event"] for r in out["weekends"].to_dict("records")} \
        if len(out.get("weekends", ())) else {}
    parts = [t.assign(mode=name) for name, t in (out.get("strategy_trades") or {}).items() if len(t)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["settings_key"], df["venue"] = p.get("settings_key"), p.get("venue")
    df["year"] = pd.to_datetime(df["t"], errors="coerce").dt.year if "t" in df else None
    df["event"] = df["event_key"].map(ev)
    lead = ["mode", "settings_key", "venue", "year", "event_key", "event"]
    return df[lead + [c for c in df.columns if c not in lead]]


def write_trades(out, folder, name="trades"):
    """--trades-out: write trades_frame(out) to <folder>/<name>.parquet (opt-in output; nothing else changes).
    Returns the path, or None when the sweep made no trades."""
    from pathlib import Path
    df = trades_frame(out)
    if not len(df):
        return None
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{name}.parquet"
    df.to_parquet(dest, index=False)
    return dest


def season_clv(engine, out, venue, sport, stage_order=(), echo=print):
    """Closing-line value of a season's trades (pipelines/clv.py: every taker mode's trades and maker variant's
    fills against the venue's price at each market's close, read from the database), set as out["clv"] (what
    the saved run keeps as metrics.clv: per strategy, strategy x stage, strategy x event) and out["clv_bets"] (one
    row per bet; not saved). Adds to the sweep, never changes it: a failure to read the closes is echoed and kept
    as out["clv"] = dict(error=...), the sweep's own results stand."""
    from racinglines.pipelines import clv as CLV
    order = [*TAKER_MODES, *MAKERS, "buy_all"]
    try:
        out["clv"], out["clv_bets"] = CLV.sweep(engine, out.get("strategy_trades") or {}, venue, sport,
                                                strategies=order, stages=stage_order)
    except Exception as ex:  # noqa: BLE001  (CLV is an addition: the sweep's results stand without it)
        echo(f"  CLV not computed: {type(ex).__name__}: {ex}")
        out["clv"], out["clv_bets"] = dict(error=f"{type(ex).__name__}: {ex}"), None
    return out
