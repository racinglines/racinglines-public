"""
F1 live private book: a race weekend's mock book, updated at session ends from the model's stage runs (no
live feed). The F1 adapter of the live core (pipelines/live.py); the plan is docs/f1-live-roadmap.md.

    racinglines live step live/f1/2026-16.toml            # one idempotent update (the LaunchAgent, every 5 min)
    racinglines live step live/f1/2026-15.toml --now 2026-09-24T07:35 --no-fetch     # a simulated clock

The updates, each one step that acts only when something new has happened:

    1. pre-weekend   the book opens: every market listed, priced from the stage run, quoted; the demo taker's
                     hype picks; the settings frozen into meta.json
    2-5. after FP1 / FP2 / FP3 (sprint: SQ, Sprint) / Quali
                     the crowd trades the window since the last update at the quotes posted then; reprice
                     from the new stage run; requote. After qualifying: pole settles from the classification
                     and the pre-race window opens (fresh caps, a faster pace). The book never freezes: every
                     update requotes from the latest fair until lights out (owner rule, 2026-10-04)
    6. lights out    the crowd trades the pre-race window; the book closes
    7. results       every market settles from the race classification; final P&L

A stage update waits for its stage run (priced with the maker's profile, C, by the same code as the signal
engine: signals.price_stages_now), so a late archive delays that update and nothing else. Updates missed
while the engine was down are merged into the next one, its crowd batch covering the whole gap.

Markets (key: what YES means):
    race_win:<athlete>             wins the race                     win_prob
    race_podium:<athlete>          finishes on the podium            podium_prob
    race_pole:<athlete>            takes pole                        extra.pole_prob       settles after qualifying
    race_h2h:<a>:<b>               a finishes ahead of b             extra.h2h[b]          pairs: the last race Polymarket listed
    race_constructor_top:<team>    the team scores the most points   metrics.race_constructor_top
Props, off by default (listed only when [live.markets] kinds names them; models/props.py):
    race_safety_car / race_red_flag / race_rain     yes/no, per-circuit rates from the race history
    race_fastest_lap:<athlete>     sets the race's fastest lap       the stage run's finishing odds x history
Settlement: private_book.outcome_for on the official classification. A demo experiment: play money, nothing
is traded anywhere.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.markets import crowd as C
from racinglines.markets import quoting as Q
from racinglines.models import props as P
from racinglines.pipelines import live as LV

KINDS = ("race_win", "race_podium", "race_pole", "race_h2h", "race_constructor_top")     # the default market set
PROP_KINDS = P.PROP_KINDS                                                               # opt-in
ALL_KINDS = KINDS + PROP_KINDS
KIND_LABEL = {"race_win": "Winner", "race_podium": "Podium", "race_pole": "Pole position", "race_h2h": "Head-to-head",
              "race_constructor_top": "Top constructor", "race_props": "Race props", **P.LABEL}
FIELD = {"race_win": "win_prob", "race_podium": "podium_prob"}
GROUP_TARGET = {"race_win": 1.0, "race_podium": 3.0, "race_pole": 1.0, "race_constructor_top": 1.0}
VIEW_GROUPS = KINDS + ("race_props", "race_fastest_lap")      # the Live tab: the yes/no props share one card
EXCHANGE_LABELS = {"polymarket": "Polymarket", "kalshi": "Kalshi", "coinbase": "Coinbase", "og": "OG.com"}
EXCHANGE_ORDER = ("polymarket", "kalshi", "coinbase", "og")


def _utc(t):
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def _iso(t):
    return pd.Timestamp(t).tz_localize("UTC").isoformat() if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t).isoformat()


# ---------------------------------------------------------------------------
# Markets and their fair values (adapter: markets)
# ---------------------------------------------------------------------------

def mkey(kind, athlete_id=None, opponent_id=None, team=None):
    if kind == "race_h2h":
        return f"race_h2h:{athlete_id}:{opponent_id}"
    if kind == "race_constructor_top":
        return f"race_constructor_top:{team}"
    return f"{kind}:{athlete_id}"


def h2h_pairs(conn, event_key, source="last_listed"):
    """([(a, b)], source event key): Polymarket's head-to-head pairs, one per market (its first outcome is
    "a finishes ahead of b"), from `source` (an event key) or the last race listed before this one."""
    if source in (None, "", "last_listed"):
        source = conn.execute(text("""
            SELECT e.source_key FROM market_links ml JOIN races ra ON ra.id = ml.race_id JOIN events e ON e.id = ra.event_id
            WHERE ml.prediction = 'race_h2h' AND ml.exchange = 'polymarket' AND e.start_date <= (SELECT e2.start_date FROM events e2
                  WHERE e2.source_key = :k AND e2.source = 'f1timing')
            ORDER BY e.start_date DESC LIMIT 1"""), dict(k=event_key)).scalar()
    if source is None:
        return [], None
    rows = conn.execute(text("""
        SELECT DISTINCT ON (ml.condition_id) ml.athlete_id, (ml.params->>'opponent_id')::int
        FROM market_links ml JOIN races ra ON ra.id = ml.race_id JOIN events e ON e.id = ra.event_id
        WHERE e.source_key = :s AND ml.prediction = 'race_h2h' AND ml.exchange = 'polymarket' AND ml.athlete_id IS NOT NULL
        ORDER BY ml.condition_id, ml.id"""), dict(s=source)).all()
    return sorted((int(a), int(b)) for a, b in rows if b is not None), source


def run_prices(conn, run_id):
    """A stage run's prices: (per-driver DataFrame with athlete_id, driver, team_key, win_prob, podium_prob,
    pole_prob, h2h; {team: P(top constructor)})."""
    df = pd.read_sql(text("""SELECT athlete_id, win_prob, podium_prob, extra FROM race_predictions
                             WHERE model_run_id = :r ORDER BY win_prob DESC, athlete_id"""), conn, params=dict(r=run_id))
    ex = df.pop("extra").map(lambda e: e or {})
    df["driver"] = ex.map(lambda e: e.get("driver"))
    df["team_key"] = ex.map(lambda e: e.get("team_key"))
    df["pole_prob"] = ex.map(lambda e: e.get("pole_prob"))
    df["h2h"] = ex.map(lambda e: e.get("h2h") or {})
    params, metrics = conn.execute(text("SELECT params, metrics FROM model_runs WHERE id = :r"), dict(r=run_id)).first()
    ctor = ((metrics or {}).get("race_constructor_top") or {}).get((params or {}).get("event_key")) or {}
    return df, ctor


def market_set(preds, ctor, pairs, kinds=KINDS):
    """The markets and their fair values from one stage run: [dict(key, kind, athlete_id, params, subject,
    fair)]. Head-to-head pairs with a driver not in the field are left out."""
    from racinglines.markets.venues import team_name
    out = []
    name = dict(zip(preds["athlete_id"].astype(int), preds["driver"]))
    by = preds.set_index(preds["athlete_id"].astype(int))
    for kind in ("race_win", "race_podium", "race_pole"):
        if kind not in kinds:
            continue
        col = FIELD.get(kind, "pole_prob")
        for r in preds.itertuples():
            v = getattr(r, col)
            out.append(dict(key=mkey(kind, int(r.athlete_id)), kind=kind, athlete_id=int(r.athlete_id), params=None,
                            subject=r.driver, fair=None if v is None or pd.isna(v) else float(v)))
    if "race_h2h" in kinds:
        for a, b in pairs:
            if a not in by.index or b not in by.index:
                continue
            p = by.at[a, "h2h"].get(str(b))
            if p is None and by.at[b, "h2h"].get(str(a)) is not None:
                p = 1 - by.at[b, "h2h"][str(a)]
            out.append(dict(key=mkey("race_h2h", a, b), kind="race_h2h", athlete_id=a, params=dict(opponent_id=b),
                            subject=f"{name[a]} vs {name[b]}", fair=None if p is None else float(p)))
    if "race_constructor_top" in kinds:
        for team in sorted(preds["team_key"].dropna().unique()):
            out.append(dict(key=mkey("race_constructor_top", team=team), kind="race_constructor_top", athlete_id=None,
                            params=dict(team=team), subject=team_name(team), fair=float(ctor.get(team, 0.0))))
    return out


def group_sums(mkts):
    """{kind: sum of fair values} for the grouped kinds, and the worst head-to-head pair's deviation from 1
    (checks: winner 1, podium 3, pole 1, constructors 1)."""
    s = {k: sum(m["fair"] or 0 for m in mkts if m["kind"] == k) for k in GROUP_TARGET}
    fair = {m["key"]: m["fair"] for m in mkts}
    dev = [abs(m["fair"] + fair.get(mkey("race_h2h", m["params"]["opponent_id"], m["athlete_id"]), 1 - m["fair"]) - 1)
           for m in mkts if m["kind"] == "race_h2h" and m["fair"] is not None]
    return s, max(dev, default=0.0)


def _market_lookup_key(kind, athlete_id=None, params=None):
    p = params if isinstance(params, dict) else {}
    if isinstance(params, str):
        try:
            p = json.loads(params)
        except (TypeError, ValueError):
            p = {}
    try:
        athlete = int(athlete_id) if athlete_id is not None and not pd.isna(athlete_id) else None
    except (TypeError, ValueError):
        athlete = None
    opp = p.get("opponent_id")
    try:
        opp = int(opp) if opp is not None and not pd.isna(opp) else None
    except (TypeError, ValueError):
        opp = None
    team = p.get("team")
    n = p.get("n")
    try:
        n = int(n) if n is not None and not pd.isna(n) else None
    except (TypeError, ValueError):
        n = None
    return (kind, athlete, team, opp, n)


def attach_exchange_prices(mkts, links):
    """Add a compact per-market exchange summary for live pages when the event is linked to venue data."""
    if not mkts:
        return []
    rows = [] if links is None or len(links) == 0 else links.to_dict("records") if hasattr(links, "to_dict") else list(links)
    out = []
    for market in mkts:
        key = _market_lookup_key(market.get("kind"), market.get("athlete_id"), market.get("params"))
        prices = []
        for row in rows:
            if not row:
                continue
            pred = row.get("prediction") or row.get("kind") or row.get("market_kind")
            if pred is None:
                continue
            if _market_lookup_key(pred, row.get("athlete_id"), row.get("params")) != key:
                continue
            code = str(row.get("exchange") or "")
            if not code:
                continue
            mid = row.get("last_price")
            bid = row.get("last_bid")
            ask = row.get("last_ask")
            mid = None if mid is None or pd.isna(mid) else float(mid)
            bid = None if bid is None or pd.isna(bid) else float(bid)
            ask = None if ask is None or pd.isna(ask) else float(ask)
            if mid is None and bid is None and ask is None:
                continue
            synced = row.get("synced_at")       # when the sync last wrote these prices (None: unknown)
            prices.append(dict(code=code, name=EXCHANGE_LABELS.get(code, code.replace("_", " ").title()),
                               mid=mid, bid=bid, ask=ask, slug=row.get("event_slug"),
                               token=row.get("token_id"),
                               synced=None if synced is None or pd.isna(synced) else _iso(synced)))
        prices.sort(key=lambda x: (EXCHANGE_ORDER.index(x["code"]) if x["code"] in EXCHANGE_ORDER else len(EXCHANGE_ORDER), x["code"]))
        out.append(dict(market, exchange_prices=prices))
    return out


PRICE_STALE_MIN = 15   # an exchange price older than this is flagged stale (the Markets board's rule, web/board.py STALE_MIN)


def venue_price_ages(markets, asof):
    """(subject, price) for every exchange price on the snapshot's markets, each price with its age in minutes as of
    `asof` (now, or the snapshot's time in a replay) and stale=True past PRICE_STALE_MIN. A price with no sync time
    (snapshots written before it was recorded) has age None and is not called stale: its age is unknown."""
    out = []
    for m in markets:
        for q in m.get("exchange_prices") or []:
            if q.get("mid") is None and q.get("bid") is None and q.get("ask") is None:
                continue
            age = None
            if q.get("synced"):
                age = max(0, int((pd.Timestamp(asof) - pd.Timestamp(q["synced"])).total_seconds() // 60))
            out.append((m["subject"], dict(q, age_min=age, stale=age is not None and age > PRICE_STALE_MIN)))
    return out


def compare_exchange_prices(links, threshold=0.005):
    """Any Coinbase/Kalshi gap above a small threshold, logged once per poll for later inspection."""
    rows = [] if links is None or len(links) == 0 else links.to_dict("records") if hasattr(links, "to_dict") else list(links)
    by_key = {}
    for row in rows:
        if not row:
            continue
        code = str(row.get("exchange") or "").lower()
        if code not in ("coinbase", "kalshi"):
            continue
        key = _market_lookup_key(row.get("prediction") or row.get("kind") or row.get("market_kind"),
                                 row.get("athlete_id"), row.get("params"))
        val = row.get("last_price")
        val = None if val is None or pd.isna(val) else float(val)
        if val is None:
            continue
        prev = by_key.setdefault(key, {"kind": row.get("prediction") or row.get("kind") or row.get("market_kind"),
                                      "athlete_id": row.get("athlete_id"), "params": row.get("params"),
                                      "coinbase": None, "kalshi": None})
        prev[code] = val
    out = []
    for key, vals in by_key.items():
        cb, ks = vals.get("coinbase"), vals.get("kalshi")
        if cb is None or ks is None:
            continue
        diff = abs(cb - ks)
        if diff > threshold:
            out.append(dict(kind=vals.get("kind"), athlete_id=vals.get("athlete_id"), params=vals.get("params"),
                            market_key=list(key), coinbase=cb, kalshi=ks, diff=diff))
    out.sort(key=lambda r: r["diff"], reverse=True)
    return out


def markets(conn, event_key, run_id, source="last_listed", kinds=KINDS, props=None):
    """Adapter interface: the event's markets and their fair values from a stage run (props: the [live.props]
    settings, for prop kinds)."""
    preds, ctor = run_prices(conn, run_id)
    pairs, _ = h2h_pairs(conn, event_key, source)
    return market_set(preds, ctor, pairs, kinds) + prop_markets(conn, event_key, run_id, kinds, props)


def prop_markets(conn, event_key, run_id, kinds, props=None):
    """The prop markets among `kinds` ([] when none is listed, the default). The race's latest saved wet vote
    (`racinglines weather fetch ... --session race=...`, weather/wet.py), when there is one, prices race_rain and
    race_red_flag given the forecast; [live.props] weather = false turns that off."""
    pk = tuple(k for k in kinds if k in PROP_KINDS)
    if not pk:
        return []
    props = props or {}
    vote = None
    if props.get("weather", True):
        from racinglines.weather import wet as WET
        vote = WET.load_vote(event_key)
    return P.markets(conn, event_key, run_id, pk, props.get("prior_n", P.PRIOR_N), wet_vote=vote)


# ---------------------------------------------------------------------------
# The weekend's updates
# ---------------------------------------------------------------------------

QUAL_UPDATE = "after Quali"      # pole settles and the pre-race window opens at this update


def race_done_after():
    from racinglines.pipelines.signals import RACE_DONE
    return RACE_DONE


def plan(event_key):
    """The weekend's updates in order: [dict(label, kind ('open' | 'stage' | 'close' | 'results'), at (naive
    UTC), qual (the qualifying update: pole settles, the pre-race window opens))], from the FastF1 schedule
    (weekend_sweep.schedule: stage cutoffs are session end + the data lag). Plus the weekend (weekend_sweep's dict)."""
    from racinglines.pipelines import weekend_sweep as WS
    year, rnd = (int(x) for x in event_key.split("-"))
    w = WS.schedule(year, rounds=[rnd])[rnd]
    ups = [dict(label=lab, kind="open" if i == 0 else "stage", at=cut, qual=lab == QUAL_UPDATE)
           for i, (lab, cut) in enumerate(w["stages"])]
    ups += [dict(label="lights out", kind="close", at=w["race_start"], qual=False),
            dict(label="results", kind="results", at=w["race_start"] + race_done_after(), qual=False)]
    return ups, w


def load_state(out):
    p = out / "state.json"
    return json.loads(p.read_text()) if p.exists() else dict(done=[], closed=False, settled=False,
                                                             quotes=[], markets=[], outcomes={}, last_ts=None)


def save_state(out, st):
    tmp = out / "state.json.tmp"
    tmp.write_text(json.dumps(st, default=str))
    tmp.replace(out / "state.json")


def due(ups, st, now):
    """The updates due at `now` and not yet done (or skipped)."""
    seen = {d["label"] for d in st["done"]}
    return [u for u in ups if u["label"] not in seen and u["at"] <= now]


def status(spec, now=None):
    """The event's state for `racinglines live status`: dict(done, next label, due at, late by (h), settled)."""
    now = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    out = LV.folder(spec.get("run", spec["event"]), mkdir=False)
    st = load_state(out) if out.exists() else load_state(LV.base() / "__none__")
    ups, _ = plan(spec["event"])
    nxt = next_update(ups, st)
    return dict(done=[d["label"] for d in st["done"]], next=nxt["label"] if nxt else None,
                due=_iso(nxt["at"]) if nxt else None, settled=st["settled"],
                late_h=round((now - nxt["at"]).total_seconds() / 3600, 1) if nxt and now > nxt["at"] else 0.0)


def next_update(ups, st):
    seen = {d["label"] for d in st["done"]}
    return next((u for u in ups if u["label"] not in seen), None)


def late(ups, st, now, hours=2.0):
    """The update that is more than `hours` overdue (due, not done), or None: the lateness alert."""
    u = next_update(ups, st)
    return u if u is not None and now - u["at"] > timedelta(hours=hours) else None


# ---------------------------------------------------------------------------
# Settlement (adapter: outcomes)
# ---------------------------------------------------------------------------

def race_id(conn, event_key):
    return conn.execute(text("""SELECT ra.id FROM races ra JOIN events e ON e.id = ra.event_id
                                WHERE e.source_key = :k AND e.source = 'f1timing'"""), dict(k=event_key)).scalar()


def qual_classification(conn, rid):
    return pd.read_sql(text("""SELECT r.athlete_id, r.position AS qual_position FROM results r JOIN rounds ro ON ro.id = r.round_id
                               WHERE ro.race_id = :r AND ro.kind = 'qual'"""), conn, params=dict(r=rid))


def outcomes(conn, rid, mkts, stage="race"):
    """{key: True / False} for the markets decided at this point: 'qual' settles pole from the qualifying
    classification (nothing else is read); 'race' settles every market from the race classification
    (private_book.outcome_for). {} while the classification isn't in."""
    from racinglines.markets.private_book import outcome_for, race_outcomes
    out = {}
    if stage == "qual":
        q = qual_classification(conn, rid)
        if len(q) < 5:
            return {}
        res = q.assign(position=None, status=None, team_id=None, points=0.0)
        for m in mkts:
            if m["kind"] == "race_pole":
                y = outcome_for("race_pole", m["athlete_id"], m["params"], res)
                if y is not None:
                    out[m["key"]] = bool(y)
        return out
    res = race_outcomes(conn, rid)
    if res.empty or (res["status"] == "OK").sum() < 5:
        return {}
    for m in mkts:
        y = outcome_for(m["kind"], m["athlete_id"], m["params"], res)
        if y is not None:
            out[m["key"]] = bool(y)
    pm = [m for m in mkts if m["kind"] in PROP_KINDS]
    if pm:                                               # props: from the race's laps and weather
        out.update(P.outcomes(conn, rid, pm))
    return out


def classification(conn, rid, round_kind):
    """A session's classification: [dict(pos, driver, team, status, best)] (the latest session's, for the page).
    Practice and sprint qualifying carry no positions in the results: those are ranked by best valid lap."""
    df = pd.read_sql(text("""
        SELECT r.position, a.display_name AS driver, coalesce(r.team, r.extra->>'team_id') AS team, r.status,
               (SELECT min(l.lap_time_ms) FROM laps l WHERE l.result_id = r.id AND NOT coalesce(l.deleted, false)) AS best
        FROM results r JOIN rounds ro ON ro.id = r.round_id JOIN athletes a ON a.id = r.athlete_id
        WHERE ro.race_id = :r AND ro.kind = :k ORDER BY r.position NULLS LAST, best NULLS LAST, a.display_name"""),
        conn, params=dict(r=rid, k=round_kind))
    by_lap = df["position"].isna().all()
    return [dict(pos=(i + 1 if by_lap and pd.notna(r.best) else None) if by_lap else (None if pd.isna(r.position) else int(r.position)),
                 driver=r.driver, team=r.team, status=r.status, best=None if pd.isna(r.best) else int(r.best))
            for i, r in enumerate(df.itertuples())]


# ---------------------------------------------------------------------------
# Quotes and the demo taker's picks
# ---------------------------------------------------------------------------

def quotes_for(mkts, book, picks, qp, hs, decided=()):
    """The maker's quotes on every market: fair +- hs, leaning against inventory (the crowd's fills and the
    demo taker's picks); none on a decided market. qp's max_loss / floor_bid switches apply (off by default)."""
    pos = C.maker_positions(book, picks)                 # the crowd's fills and the demo taker's picks
    out = []
    for m in mkts:
        inv, cash = pos.get(m["key"], {}).get("yes", 0.0), pos.get(m["key"], {}).get("cash", 0.0)
        bid, ask = (None, None) if m["key"] in decided else Q.capped(m["fair"], hs, inv, cash, qp)
        out.append(dict(key=m["key"], kind=m["kind"], fair=None if m["fair"] is None else round(m["fair"], 4),
                        bid=bid, ask=ask, inv=round(inv, 2)))
    return out


def make_picks(mkts, quotes, spec_picks, stake, ts):
    """The demo taker's hype picks at the book's opening: each spec pick (name: a driver or team substring;
    market: a kind; vs: the opponent, head-to-head only; hype 1-5; why) bought YES at the maker's ask for
    `stake` dollars. Picks the book doesn't offer are left out."""
    qs = {q["key"]: q for q in quotes}
    done = []
    for p in spec_picks:
        kind = p["market"]
        cand = [m for m in mkts if m["kind"] == kind and p["name"].lower() in (m["subject"] or "").lower()]
        if kind == "race_h2h":
            cand = [m for m in cand if m["subject"].lower().split(" vs ")[0].find(p["name"].lower()) >= 0
                    and p.get("vs", "").lower() in m["subject"].lower().split(" vs ")[1]]
        if not cand:
            continue
        m = cand[0]
        q = qs[m["key"]]
        if q["ask"] is None:
            continue
        done.append(dict(p, key=m["key"], kind=kind, subject=m["subject"], price=q["ask"], stake=stake,
                         shares=round(stake / q["ask"], 2), fair_then=q["fair"], ts=ts, side="YES"))
    return done


# ---------------------------------------------------------------------------
# One step (adapter: step)
# ---------------------------------------------------------------------------

def frozen_settings(spec, out, unfreeze=False, echo=print):
    """The settings this event runs with: frozen into meta.json at the book's opening; a changed schema or
    spec is ignored (with a warning) unless unfreeze."""
    live = spec["live"]
    p = out / "meta.json"
    if not p.exists():
        return live
    meta = json.loads(p.read_text())
    frozen = meta.get("settings")
    if frozen is None or frozen == json.loads(json.dumps(live)):
        return live
    if unfreeze:
        echo("settings changed and --unfreeze given: running with the new settings (add a decision-log line)")
        return live
    echo("settings differ from the frozen ones in meta.json: running with the frozen settings (--unfreeze to change)")
    return frozen


def _seed(event_key, label):
    return int(hashlib.sha1(f"{event_key}:{label}".encode()).hexdigest()[:8], 16)


def step(spec, now=None, fetch=True, unfreeze=False, echo=print, engine=None, engine_url=None, cache=None, sync=True,
         alert=True):
    """One idempotent update of the event at `now` (naive UTC; default the current time): the snapshot
    written, or None when there's nothing new. fetch: refresh FastF1 data first (live); sync: write the
    positions to paper_positions; alert: send the lateness alert (an update more than late_alert_h overdue,
    once per update)."""
    from racinglines.db.config import get_engine
    event_key, run = spec["event"], spec.get("run", spec["event"])
    now = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    out = LV.folder(run)
    st = load_state(out)
    if st["settled"]:
        return None
    live = frozen_settings(spec, out, unfreeze, echo)
    ups, w = plan(event_key)
    todo = due(ups, st, now)
    if not todo:
        return None
    lu = late(ups, st, now, live.get("poll", {}).get("late_alert_h", 2.0))
    if lu is not None and lu["label"] not in st.setdefault("alerted", []):
        msg = f"{spec.get('title') or event_key}: the {lu['label']} update is late (due {_iso(lu['at'])[:16]} UTC)"
        echo(f"LATE: {msg}")
        if alert:
            try:
                from racinglines.markets.alerts import deliver
                deliver("racinglines live: update late", msg, urgent=True, payload=dict(kind="live_late", event=event_key,
                                                                                       update=lu["label"]))
            except Exception as ex:                          # noqa: BLE001
                echo(f"alert failed: {ex}")
        st["alerted"].append(lu["label"])
        save_state(out, st)
    engine = engine or get_engine(engine_url)
    cache = {} if cache is None else cache
    kinds = [k for k in live["markets"]["kinds"] if k in ALL_KINDS]
    qp = Q.Params.from_dict(live["quoting"])
    cp = C.Params.from_dict(dict(live["crowd"], max_loss=live["quoting"].get("max_loss")))
    book = C.load_book(out, cp)
    picks = json.loads((out / "picks.json").read_text()) if (out / "picks.json").exists() else []
    with engine.connect() as c:
        rid = race_id(c, event_key)
    kinds_due = {u["kind"] for u in todo}
    note, fills, update, run_id = [], [], None, st.get("run_id")
    mkts = st["markets"]

    if "results" in kinds_due and st["closed"]:
        if fetch:
            _refresh(w, now, cache, echo, engine)
        with engine.connect() as c:
            res = outcomes(c, rid, mkts, "race")
        if not res:
            echo(f"{event_key} results: waiting for the race classification")
            return None
        st["outcomes"].update(res)
        st["settled"] = True
        update = next(u for u in ups if u["kind"] == "results")
        st["done"].append(dict(label=update["label"], ts=_iso(now)))
    elif "close" in kinds_due and not st["closed"]:
        update = next(u for u in ups if u["kind"] == "close")
        for u in ups:                                   # stages that never priced (their data never came): skipped
            if u["kind"] in ("open", "stage") and u["label"] not in {d["label"] for d in st["done"]}:
                st["done"].append(dict(label=u["label"], ts=None, skipped=True))
                note.append(f"{u['label']} skipped (no stage run by lights out)")
        if st["last_ts"]:
            end = min(now, update["at"])
            fills = _crowd(st, book, cp, event_key, update["label"], end, out)
        st["closed"] = True
        st["done"].append(dict(label=update["label"], ts=_iso(now)))
    else:
        stages = [u for u in todo if u["kind"] in ("open", "stage")]
        if not stages:
            return None
        priced = _stage_runs(spec, w, now, fetch, cache, echo, engine, engine_url)
        have = {lab: r for lab, r in priced}
        ready = [u for u in stages if u["label"] in have]
        if not ready:
            return None                                 # waiting for the stage run (data not in yet)
        update = ready[-1]
        run_id = have[update["label"]]
        for u in stages:
            if u is update:
                break
            st["done"].append(dict(label=u["label"], ts=None, skipped=True, merged_into=update["label"]))
            note.append(f"{u['label']} merged into {update['label']}")
        if st["last_ts"]:
            fills = _crowd(st, book, cp, event_key, update["label"], now, out)
        prev = {m["key"]: m["fair"] for m in mkts}
        with engine.connect() as c:
            if st.get("pairs") is None:                 # the head-to-head pairs: fixed at the opening
                pairs, st["pairs_from"] = h2h_pairs(c, event_key, live["markets"].get("h2h_from", "last_listed"))
                st["pairs"] = [list(p) for p in pairs]
            new = market_set(*run_prices(c, run_id), [tuple(p) for p in st["pairs"]], kinds)
            new += prop_markets(c, event_key, run_id, kinds, live.get("props"))
        if mkts:                                        # markets listed at the opening stay; a new driver's are added
            keep = {m["key"] for m in new}
            new += [dict(m, fair=m["fair"]) for m in mkts if m["key"] not in keep]
        for m in new:
            m["prev_fair"] = prev.get(m["key"])
        mkts = new
        if update["qual"]:
            with engine.connect() as c:
                st["outcomes"].update(outcomes(c, rid, mkts, "qual"))
            C.open_late(book, cp, _iso(now))
        st["done"].append(dict(label=update["label"], ts=_iso(now), run_id=run_id))
        st["run_id"] = run_id

    # quotes: requoted from the latest fair at every update until lights out (never frozen); none once closed
    hs = Q.half_spread(live["quoting"]["half_spread"], (update or {}).get("label"), 0.03)
    decided = set(st["outcomes"])
    if st["closed"]:
        quotes = [dict(q, bid=None, ask=None) for q in st["quotes"]]
    else:
        quotes = quotes_for(mkts, book, picks, qp, hs, decided)
    if not st.get("opened") and update["kind"] in ("open", "stage"):      # the book opens
        picks = make_picks(mkts, quotes, spec.get("picks", []), live.get("picks", {}).get("stake", 25.0), _iso(now))
        (out / "picks.json").write_text(json.dumps(picks, indent=1))
        quotes = quotes_for(mkts, book, picks, qp, hs, decided)        # the maker now holds the other side
        meta = dict(sport="f1", event_key=event_key, title=spec.get("title") or w["name"], spec=spec.get("path"),
                    settings=live, opened_at=_iso(now), profile=live["sources"]["live"].get("profile"),
                    h2h_from=st.get("pairs_from"), h2h_pairs=st.get("pairs"))
        LV.write_meta(out, meta, now.strftime("%Y%m%dT%H%M%S"))
        st["opened"] = True
    st.update(markets=mkts, quotes=quotes, last_ts=_iso(now))
    fair = {m["key"]: m["fair"] for m in mkts if m["fair"] is not None}
    oc = st["outcomes"]
    pnl = C.book_pnl(book, fair, oc, picks)
    cres = C.crowd_results(book, fair, oc)
    (out / "book.json").write_text(json.dumps(book))
    subj = {m["key"]: (m["kind"], m["subject"]) for m in mkts}
    try:
        if sync:
            C.sync_positions(event_key, book, picks, lambda k: subj.get(k, ("race", k)), fair, oc, race_id=rid)
    except Exception as ex:                                  # noqa: BLE001  never stop the engine for the database
        echo(f"positions sync failed: {ex}")
    nxt = next_update(ups, st)
    rounds = {"after FP1": "fp1", "after FP2": "fp2", "after FP3": "fp3", "after SQ": "sprint_qual",
              "after Sprint": "sprint", "after Quali": "qual", "lights out": "qual", "results": "race"}
    with engine.connect() as c:
        cls_round = rounds.get(update["label"])
        cls = classification(c, rid, cls_round) if cls_round and rid else []
    qmap = {q["key"]: q for q in quotes}
    with engine.connect() as c:
        rid = race_id(c, event_key)
        if rid is not None:
            links = pd.read_sql(text("""SELECT exchange, prediction, athlete_id, params, last_price, last_bid,
                last_ask, token_id, event_slug, synced_at FROM market_links WHERE race_id = :r AND prediction IS NOT NULL"""),
                               c, params=dict(r=rid))
            mkts = attach_exchange_prices(mkts, links)
            diffs = compare_exchange_prices(links, threshold=0.005)
            LV.append(out, "coinbase_kalshi_diffs.jsonl",
                      dict(ts=_iso(now), event_key=event_key, threshold=0.005, n_diffs=len(diffs), diffs=diffs))
            if diffs:
                echo(f"{_iso(now)} {event_key}: Coinbase/Kalshi diff check: {len(diffs)} markets above 0.5%")
        else:
            diffs = []
    snap = dict(ts=_iso(now), sport="f1", event_key=event_key, title=spec.get("title") or w["name"], name=w["name"],
                update=dict(label=update["label"], kind=update["kind"], n=len([d for d in st["done"] if not d.get("skipped")])),
                updates=[dict(label=u["label"], kind=u["kind"], at=_iso(u["at"]),
                              state=next(("skipped" if d.get("skipped") else "done" for d in st["done"] if d["label"] == u["label"]), "pending"))
                         for u in ups],
                run_id=run_id, note=note, done=st["settled"], closed=st["closed"],
                next_at=_iso(nxt["at"]) if nxt else None, next_label=nxt["label"] if nxt else None,
                markets=[dict(m, bid=qmap.get(m["key"], {}).get("bid"), ask=qmap.get(m["key"], {}).get("ask"),
                              inv=qmap.get(m["key"], {}).get("inv"), outcome=oc.get(m["key"])) for m in mkts],
                classification=dict(round=cls_round, rows=cls), maker_pnl=pnl,
                betting=dict(late=bool(book.get("late")), late_at=book.get("late_at"), late_cap=cp.late_cap,
                             pace=cp.late_pace if book.get("late") else 1.0),
                crowd=dict(book["crowd"], takers=cp.takers, last_fills=len(fills),
                           active=sum(1 for x, b in zip(book["left"], book["budget"]) if x < b - 0.005),
                           left=round(sum(book["left"]), 2), results=cres),
                outcomes=[dict(key=k, yes=v) for k, v in oc.items()])
    stamp = now.strftime("%Y%m%dT%H%M%S")
    LV.write_snapshot(out, snap, stamp)
    LV.append(out, "history.jsonl", dict(ts=snap["ts"], label=update["label"], run_id=run_id,
                                         fair={m["key"]: m["fair"] for m in mkts},
                                         quotes={q["key"]: [q["bid"], q["ask"]] for q in quotes}))
    save_state(out, st)
    echo(f"{snap['ts']} {event_key} {update['label']}: {len(mkts)} markets, {sum(1 for q in quotes if q['bid'] or q['ask'])} quoted, "
         f"{len(fills)} crowd fills, maker P&L {pnl['total']:+.2f}" + (f" ({'; '.join(note)})" if note else ""))
    return snap


def _crowd(st, book, cp, event_key, label, end, out):
    """The crowd's batch for the window since the last update, at the quotes posted then; logged."""
    start = pd.Timestamp(st["last_ts"]).to_pydatetime()
    end_ = pd.Timestamp(end).tz_localize("UTC").to_pydatetime() if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).to_pydatetime()
    seed = _seed(event_key, label)
    late_ = bool(book.get("late"))
    fills = C.window(st["quotes"], book, np.random.default_rng(seed), cp, start, end_,
                     pace=cp.late_pace if late_ else 1.0, pot="late_left" if late_ else "left")
    LV.append(out, "crowd.jsonl", dict(ts=end_.isoformat(timespec="seconds"), seed=seed, label=label,
                                       window=[start.isoformat(), end_.isoformat()], late=late_, fills=fills))
    return fills


def _refresh(w, now, cache, echo, engine):
    """Live only: fetch and ingest a finished session's FastF1 data when it's missing (signals.py)."""
    from racinglines.models.inputs import laps_pricing as run
    from racinglines.pipelines import signals as SG
    meas = cache.get("meas") or run.Measurements.load(engine)
    if SG._missing_sessions(meas, w, now):
        SG.refresh_fastf1(w, echo)
        cache.clear()


def _stage_runs(spec, w, now, fetch, cache, echo, engine, engine_url):
    """[(stage label, run id)] for the stages priced so far with the maker's profile (signals.price_stages_now:
    a stored run is reused, a missing one priced and stored)."""
    from racinglines.models.inputs import laps_pricing as run
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import signals as SG
    from racinglines.pipelines import sweep_settings as SS
    if fetch:
        _refresh(w, now, cache, echo, engine)
    prof = PF.PROFILES[spec["live"]["sources"]["live"].get("profile", "C")]
    st = SS.Settings.from_dict(prof["settings"], strict=False)
    meas = cache["meas"] if "meas" in cache else run.Measurements.load(engine)
    cache["meas"] = meas
    with st.applied():
        hist = cache.get(("hist", st.model_key))
        if hist is None:
            hist = cache[("hist", st.model_key)] = run.history(meas, st["track_features"])
        stages = SG.price_stages_now(meas, hist, w, st, engine, engine_url, now, echo)
    return [(lab, rid) for lab, _, rid, _ in stages]


# ---------------------------------------------------------------------------
# In-race win chart: live timing (the Mac relay, web/f1_live.py) or, after the race, its laps
# (backfill_inrace) -> models/inrace.py
# ---------------------------------------------------------------------------

RACE_LAPS = 56       # race distance for the in-race chart; Sepang (2026-16). RACINGLINES_RACE_LAPS overrides


def inrace_store(year, rnd):
    from racinglines.web import f1_live as FL
    return FL.relay_file(year, rnd).with_suffix(".inrace.jsonl")


def inrace_points(run, snap, now=None, replay=False):
    """The in-race win-chart points so far, as hist entries (ts, fair by race_win key). Live, each fresh race
    snapshot from the relay adds one point to the store (once per relay write); in replay nothing is added and
    only the points up to the snapshot shown are returned. [] when nothing is stored."""
    import os
    import re

    from racinglines.models import inrace as IR
    from racinglines.web import f1_live as FL
    key = str(snap.get("event_key") or run)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", key)
    if not m:
        return []
    year, rnd = int(m.group(1)), int(m.group(2))
    store = inrace_store(year, rnd)
    wins = {mk["subject"]: mk["key"] for mk in snap["markets"] if mk["kind"] == "race_win"}
    data = None if replay else FL.relayed(year, rnd)
    if data and str(data.get("session")) in ("Race", "R"):
        prior = {name: next((mk["fair"] or 0.0 for mk in snap["markets"] if mk["key"] == k), 0.0)
                 for name, k in wins.items()}
        probs, laps = IR.win_probs(data.get("drivers") or [], prior, int(os.environ.get("RACINGLINES_RACE_LAPS", RACE_LAPS)))
        if probs is not None:
            t = (now if now is not None else pd.Timestamp.now(tz="UTC")) - pd.Timedelta(seconds=data.get("relay_age") or 0)
            t = t.floor("min")
            last = _read_points(store)[-1:]
            if not last or pd.Timestamp(last[0]["ts"]) < t:
                try:
                    with store.open("a") as fh:
                        fh.write(json.dumps(dict(ts=t.isoformat(), lap=laps, fair=probs)) + "\n")
                except OSError:
                    pass
    pts = _read_points(store)
    if replay:
        pts = [p for p in pts if pd.Timestamp(p["ts"]) <= pd.Timestamp(snap["ts"])]
    return [dict(ts=p["ts"], fair={wins[n]: v for n, v in p["fair"].items() if n in wins}) for p in pts]


def _read_points(store):
    try:
        return [json.loads(line) for line in store.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def race_running_order(laps, results, start):
    """The race lap by lap from its laps table (FastF1, as stored by f1 fetch: LapNumber, Position, Driver,
    LapStartTime and Time in session seconds) and its classification: [(UTC time the leader finished lap L,
    rows like the relay's)]. Each driver shows his position at the end of lap L (or of his last lap before it).
    The clock is the scheduled start plus the session time since lap 1 began. A driver who stopped before lap L
    carries his classification status, so a retirement drops out of the chart."""
    laps = laps.dropna(subset=["LapNumber", "Time"]).sort_values(["Driver", "LapNumber"])
    if not len(laps):
        return []
    t0 = float(laps.loc[laps["LapNumber"] == laps["LapNumber"].min(), "LapStartTime"].min())
    name = dict(zip(results["Abbreviation"], results["FullName"]))
    status = dict(zip(results["Abbreviation"], results["Status"].astype(str)))
    out = []
    for lap in sorted(laps["LapNumber"].unique()):
        t = float(laps.loc[laps["LapNumber"] == lap, "Time"].min())
        upto = laps[laps["LapNumber"] <= lap].groupby("Driver").tail(1)
        rows = [dict(name=name.get(r.Driver, r.Driver), code=r.Driver,
                     pos=None if pd.isna(r.Position) else int(r.Position), laps=int(r.LapNumber),
                     status=status.get(r.Driver, "") if r.LapNumber < lap else "")
                for r in upto.itertuples()]
        out.append((pd.Timestamp(start) + pd.Timedelta(seconds=t - t0), rows))
    return out


def backfill_inrace(year, rnd, echo=print):
    """After the race, the in-race win chart as if the relay had run all race: one point per lap from the race's
    laps (data/raw/f1/fastf1/<year>/<round>_R.*, pushed by scripts/deploy/f1_push.sh or f1 fetch), priced from the
    last live-book update before lights out. Rewrites the store; returns how many points it wrote."""
    from racinglines.models import inrace as IR
    from racinglines.sources.fastf1.fetch import OUT
    base = OUT / str(year) / f"{rnd:02d}_R"
    meta = json.loads(base.with_suffix(".meta.json").read_text())
    laps, results = pd.read_parquet(base.with_suffix(".laps.parquet")), pd.read_parquet(base.with_suffix(".results.parquet"))
    ev = LV.find(f"{year}-{rnd:02d}")
    if ev is None:
        raise SystemExit(f"no live event recorded for {year}-{rnd:02d}")
    snap, _, hist = LV.load(ev["run"])
    start = pd.Timestamp(meta["session_date"])
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    before = [h for h in hist if pd.Timestamp(h["ts"]) <= start] or hist[:1]
    if not before:
        raise SystemExit(f"{ev['run']} has no update before lights out to price from")
    wins = {mk["subject"]: mk["key"] for mk in snap["markets"] if mk["kind"] == "race_win"}
    prior = {n: float(before[-1]["fair"].get(k) or 0.0) for n, k in wins.items()}
    race_laps = int(meta.get("total_laps") or laps["LapNumber"].max())
    points = []
    for t, rows in race_running_order(laps, results, start):
        probs, done = IR.win_probs(rows, prior, race_laps)
        if probs is not None:
            points.append(dict(ts=t.floor("s").isoformat(), lap=done, fair=probs))
    missing = sorted(set(results["FullName"]) - set(prior))
    if missing:
        echo(f"not on the book (left out of the chart): {', '.join(missing)}")
    store = inrace_store(year, rnd)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("".join(json.dumps(p) + "\n" for p in points))
    echo(f"{ev['run']}: {len(points)} in-race points (laps {points[0]['lap']}-{points[-1]['lap']}) -> {store}"
         if points else f"{ev['run']}: no usable laps")
    return len(points)


# ---------------------------------------------------------------------------
# The Live tab's body (adapter: view; templates/live_f1.html)
# ---------------------------------------------------------------------------

def view(run, snap, picks, hist, mode, maker):
    """What the F1 body needs: the weekend's updates, the latest classification, every market grouped by
    kind with its move since the last update, the private book, the demo taker's picks."""
    oc = {o["key"]: o["yes"] for o in snap.get("outcomes") or []}
    fair = {m["key"]: m["fair"] for m in snap["markets"]}
    groups = []
    for kind in VIEW_GROUPS:
        ms = [dict(m) for m in snap["markets"] if m["kind"] == kind or (kind == "race_props" and m["kind"] in P.BINARY)]
        if not ms:
            continue
        for m in ms:
            m["move"] = (None if m.get("prev_fair") is None or m["fair"] is None else m["fair"] - m["prev_fair"])
        ms.sort(key=lambda m: -(m["fair"] or 0))
        groups.append(dict(kind=kind, label=KIND_LABEL[kind], markets=ms,
                           total=sum(m["fair"] or 0 for m in ms),
                           target=1.0 if kind == "race_fastest_lap" else GROUP_TARGET.get(kind)))
    rows, tot = [], dict(stake=0.0, value=0.0, pnl=0.0, maker_pnl=0.0, won=0, lost=0)
    qs = {m["key"]: m for m in snap["markets"]}
    for p in picks:
        m = qs.get(p["key"], {})
        res = oc.get(p["key"])
        if res is not None:
            now, state = float(res), ("won" if res else "lost")
        else:
            mid = [x for x in (m.get("bid"), m.get("ask")) if x is not None]
            now, state = (sum(mid) / len(mid) if mid else m.get("fair")), "live"
        value = p["shares"] * now if now is not None else None
        f = float(res) if res is not None else m.get("fair")
        r = dict(p, now=now, value=value, pnl=None if value is None else value - p["stake"], state=state, fair=f,
                 maker_pnl=None if f is None else p["stake"] - p["shares"] * f)
        rows.append(r)
        tot["stake"] += p["stake"]
        tot["value"] += r["value"] or 0
        tot["pnl"] += r["pnl"] or 0
        tot["maker_pnl"] += r["maker_pnl"] or 0
        tot["won"] += state == "won"
        tot["lost"] += state == "lost"
    markets_, polls = LV.book_at(run, snap["ts"] if mode == "replay" else None)
    positions, recent = [], []
    subj = {m["key"]: m["subject"] for m in snap["markets"]}
    kind_of = {m["key"]: m["kind"] for m in snap["markets"]}
    for k, mk in markets_.items():
        v = float(oc[k]) if k in oc else fair.get(k) or 0.0
        positions.append(dict(subject=subj.get(k, k), kind=KIND_LABEL.get(kind_of.get(k), ""), inv=mk["inv"],
                              pnl=mk["cash"] + mk["inv"] * v, value=v, settled=k in oc))
    positions.sort(key=lambda x: -abs(x["inv"]))
    for e in reversed(polls[-3:]):
        for f in sorted(e["fills"], key=lambda f: f.get("ts") or "", reverse=True)[:6]:
            recent.append(dict(f, subject=subj.get(f["key"], f["key"]), kind=KIND_LABEL.get(kind_of.get(f["key"]), "")))
    asof = pd.Timestamp(snap["ts"]) if mode == "replay" else pd.Timestamp.now(tz="UTC")
    venue_prices = venue_price_ages(snap["markets"], asof)
    chart = None
    hist = sorted(list(hist) + inrace_points(run, snap, replay=mode == "replay"), key=lambda h: pd.Timestamp(h["ts"]))
    if maker and len(hist) >= 2:
        from racinglines.web.viz import line_chart
        top = [m for m in sorted(snap["markets"], key=lambda m: -(m["fair"] or 0)) if m["kind"] == "race_win"][:6]
        series = {m["subject"]: [(pd.Timestamp(h["ts"]), h["fair"].get(m["key"]) or 0.0) for h in hist] for m in top}
        chart = line_chart(series, {k: k for k in series}, money=False, h=190)
    return dict(groups=groups, picks=rows, tot=tot, book=markets_ if polls else None, positions=positions[:12],
                recent=recent[:12], crowd=snap.get("crowd"), mpnl=snap.get("maker_pnl"), win_chart=chart,
                venue_prices=venue_prices, venue_stale=any(q["stale"] for _, q in venue_prices),
                stale_min=PRICE_STALE_MIN,
                age=int((pd.Timestamp.now(tz="UTC") - pd.Timestamp(snap["ts"])).total_seconds()))
