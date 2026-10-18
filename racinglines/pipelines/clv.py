"""
Closing-line value (CLV) per paper bet (`racinglines backtest clv`, docs/parity-rebuild.md "C4"; owner goal G1: CLV
above +2 % over 100 live bets before real money). Read-only: it computes CLV on read and writes nothing.

A bet is one paper trade the signal engine stored (strategy_signals): a taker's buy or sell it followed
(detail.followed not false, the track record's rule) or a maker's paper fill, on any sport and venue. A sweep's
in-memory trades (season_sweep / weekend_sweep `trades`: token, stage time, side, price) go through from_trades(), its
maker fills through from_fills(); sweep() is the CLV every sweep saves in its run's metrics.clv (per strategy, per
strategy x stage and per strategy x event).

    CLV (points)   = direction x (close - entry), both as the YES token's price; direction +1 for YES exposure (a YES
                     buy, a NO sell, a maker's bid fill), -1 for NO exposure
    CLV (percent)  = CLV points / the price paid for the exposure taken (entry for YES, 1 - entry for NO)
    mid move       = the same against the market's mid when the bet was made (no costs: did the market move our way)

The entry is what the bet paid: a taker's fill price (strategy_signals.limit_price, the side's price with the
replay's cost and fee) or a maker's fill price (a YES price).

The close (one rule for every sport, kind and venue, from what the database and schemas hold):
  1. The round that decides the market, from markets/kinds.toml: the kind's `session`, else the `stage` of a
     stage_top_n payoff or a stage_top predicate (pole: "qual"), else the race.
  2. That round's start in UTC, rounds.extra.session_date (FastF1, the MotoGP API) or the sport's [sessions] time_key
     (NASCAR's run_date_utc, the race's included since C42).
  3. A race with no stored start (a NASCAR weekend whose runs carry no UTC time; a race before its results are in):
     race day from the schema, the event's start_date + [replay] race_day_offset days, at [sweep] quote_until_hours
     UTC (default 0: 00:00 UTC on race day, before any race-day running, the time the sweep's maker stops quoting). An earlier round
     with no stored start has no close: such a bet stays undecided rather than risk a close after the result.
  4. The closing price is the venue's price at the close as its replay reads it (markets/venue_replay.EXCHANGES): the
     book's mid when the venue keeps a quote (Kalshi: the recorded book within 10 minutes, else the candle's bid/ask),
     else the last tape price at or before the close (OG.com: its observed prices, without the empty book's 0.50),
     within the venue's staleness window (6 h). No price in that window: undecided.
A bet made after its close, a market with no race (season futures) or no link, and a close still ahead are undecided,
each with its reason.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines import sports
from racinglines.markets import kinds as K
from racinglines.markets import venue_replay as VR

BET_COLS = ["source", "bet_id", "user", "profile", "strategy", "sport", "venue", "event_key", "race_id", "kind",
            "subject", "market_key", "stage", "action", "side", "shares", "entry_ts", "entry", "mid"]
GROUP = ["sport", "venue", "kind", "strategy", "profile"]

_BETS = """
SELECT ss.id AS bet_id, u.username AS "user", ss.profile, ss.strategy, ss.race_id, ss.event_key, ss.market_key,
       ss.kind, ss.subject, ss.stage, ss.action, ss.side, ss.shares, ss.limit_price AS entry, ss.price AS mid,
       ss.signal_ts AS entry_ts, coalesce(ss.detail->>'venue', 'polymarket') AS venue, ss.detail->>'sport' AS sport_tag,
       coalesce(co.code, (SELECT co2.code FROM events e2 JOIN seasons s2 ON s2.id = e2.season_id
                          JOIN competitions co2 ON co2.id = s2.competition_id
                          WHERE e2.source_key = ss.event_key ORDER BY e2.id LIMIT 1)) AS competition
FROM strategy_signals ss
LEFT JOIN users u ON u.id = ss.user_id
LEFT JOIN races ra ON ra.id = ss.race_id
LEFT JOIN events e ON e.id = ra.event_id
LEFT JOIN seasons se ON se.id = e.season_id
LEFT JOIN competitions co ON co.id = se.competition_id
WHERE ss.action IN ('buy', 'sell', 'fill') AND coalesce(ss.detail->>'followed', 'true') = 'true'
"""


def _sport_of():
    return {sports.load(s)["competition"]["code"]: s for s in sports.SPORT_CODES}


def _utc_naive(t):
    """A timestamp as naive UTC (the venue classes' convention), None for a missing one."""
    if t is None or (not isinstance(t, str) and pd.isna(t)):
        return None
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


# --- the bets -------------------------------------------------------------------------------------------------

def bets(conn, users=None, sport=None, venue=None, event=None):
    """The stored paper bets (strategy_signals), one row per trade, BET_COLS. Filters: usernames, sport code, venue
    code, event key."""
    df = pd.read_sql(text(_BETS + " ORDER BY ss.signal_ts, ss.id"), conn)
    df["sport"] = df["competition"].map(_sport_of()).fillna(df["sport_tag"]).fillna("unknown")
    df["source"] = "paper"
    if users:
        df = df[df["user"].isin(list(users))]
    if sport:
        df = df[df["sport"] == sport]
    if venue:
        df = df[df["venue"] == venue]
    if event:
        df = df[df["event_key"] == str(event)]
    return df.reindex(columns=BET_COLS).reset_index(drop=True)


def from_trades(trades, venue, sport=None, strategy="update", profile=None):
    """A sweep's or replay's taker trades (taker_weekend: key = token, t = stage time, side, signed shares, price =
    the side's fill price, mid; plus kind, subject, event_key) as bet rows, for compute()."""
    t = pd.DataFrame(trades)
    if not len(t):
        return pd.DataFrame(columns=BET_COLS)
    out = pd.DataFrame(dict(
        source="sweep", bet_id=range(len(t)), user=None, profile=profile, strategy=strategy, sport=sport, venue=venue,
        event_key=t.get("event_key"), race_id=None, kind=t.get("kind"), subject=t.get("subject"), market_key=t["key"],
        stage=t.get("stage"), action=np.where(t["shares"] > 0, "buy", "sell"), side=t["side"], shares=t["shares"].abs(),
        entry_ts=t["t"], entry=t["price"], mid=t.get("mid")))
    return out.reindex(columns=BET_COLS)


def from_fills(fills, venue, sport=None, strategy="maker", profile=None):
    """A maker replay's fills (maker_replay.replay: ts in ns, cond = the market key, side "buy" = its bid filled,
    "sell" = its ask filled, price = the YES price, qty, mid; plus kind, subject, stage, event_key) as bet rows, as
    the signal engine stores a paper fill (pipelines/signals.py: action "fill", side YES for a bid filled, NO for an
    ask filled), for compute()."""
    f = pd.DataFrame(fills)
    if not len(f):
        return pd.DataFrame(columns=BET_COLS)
    out = pd.DataFrame(dict(
        source="sweep", bet_id=range(len(f)), user=None, profile=profile, strategy=strategy, sport=sport, venue=venue,
        event_key=f.get("event_key"), race_id=None, kind=f.get("kind"), subject=f.get("subject"), market_key=f["cond"],
        stage=f.get("stage"), action="fill", side=np.where(f["side"] == "buy", "YES", "NO"), shares=f["qty"],
        entry_ts=pd.to_datetime(f["ts"].astype("int64"), unit="ns"), entry=f["price"], mid=f.get("mid")))
    return out.reindex(columns=BET_COLS)


def exposure(action, side, entry):
    """(the YES price the bet was made at, direction: +1 = YES exposure, -1 = NO). A maker's fill records the YES
    price it filled at (side YES = its bid filled, NO = its ask); a taker's buy or sell records the side's own price."""
    yes_side = side == "YES"
    if action == "fill":
        return float(entry), (1 if yes_side else -1)
    yes = float(entry) if yes_side else 1.0 - float(entry)
    return yes, (1 if action == "buy" else -1) * (1 if yes_side else -1)


# --- the close --------------------------------------------------------------------------------------------------

def deciding_round(kind):
    """The rounds.kind whose start closes a market of `kind`: its session, else the stage of a stage_top_n payoff or
    a stage_top predicate, else the race (markets/kinds.toml)."""
    k = K.KINDS.get(kind)
    if k is None:
        return "race"
    if k.session:
        return k.session
    if k.payoff == "stage_top_n":
        return k.stage
    pay = ((k.spec or {}).get("payoff") or {})
    return pay.get("stage") if pay.get("predicate") == "stage_top" else "race"


def race_day_close(start_date, sport):
    """A race's close from its schema when no start is stored: race day ([replay] race_day_offset days after the
    event's date) at [sweep] quote_until_hours UTC."""
    sch = sports.load(sport) if sport in sports.SPORT_CODES else {}
    off = int((sch.get("replay") or {}).get("race_day_offset", 0))
    hrs = float((sch.get("sweep") or {}).get("quote_until_hours", 0))
    return pd.Timestamp(start_date) + pd.Timedelta(days=off) + pd.Timedelta(hours=hrs)


def links(conn, keys_by_venue):
    """{(venue, market_key): link dict with token_id} by exact key: the link whose token_id is the key, else (a
    Polymarket maker keys its market by condition) the condition's outcome-0 token (from the trade tape), else the
    condition's first link."""
    from racinglines.markets import store as MS
    out = {}
    for venue, keys in keys_by_venue.items():
        keys = sorted({str(k) for k in keys})
        if not keys:
            continue
        df = pd.read_sql(text("""
            SELECT ml.id, ml.token_id, ml.condition_id, ml.exchange, ml.race_id, ml.prediction, ml.params,
                   co.code AS competition, e.start_date
            FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
            LEFT JOIN races ra ON ra.id = ml.race_id LEFT JOIN events e ON e.id = ra.event_id
            WHERE ml.exchange = :x AND (ml.token_id = ANY(:k) OR ml.condition_id = ANY(:k)) ORDER BY ml.id"""),
            conn, params=dict(x=venue, k=keys))
        by_token = {r["token_id"]: r for r in df.to_dict("records")}
        conds = [k for k in keys if k not in by_token]
        oi = {}
        if conds and len(df):
            tr = MS.read(conn, "trades", conditions=conds)
            if len(tr):
                oi = dict(zip(tr["token_id"], tr["outcome_index"]))
        for k in keys:
            if k in by_token:
                out[(venue, k)] = by_token[k]
                continue
            g = [r for r in df.to_dict("records") if r["condition_id"] == k]
            if g:
                g.sort(key=lambda r: (oi.get(r["token_id"]) is None, oi.get(r["token_id"]) or 0, r["id"]))
                out[(venue, k)] = g[0]
    return out


def _time_keys():
    """The rounds.extra keys that hold a round's UTC start: session_date (FastF1, the MotoGP API) and every schema's
    [sessions] time_key (NASCAR's run_date_utc), from the schemas, in a fixed order."""
    keys = ["session_date"]
    for code in sports.SPORT_CODES:
        k = sports.load(code).get("sessions", {}).get("time_key")
        if k and k not in keys and k.isidentifier():
            keys.append(k)
    return keys


def closes(conn, link_rows, sport_of=None):
    """{(race_id, kind): (close as naive UTC or None, rule or the reason there is none)}."""
    sport_of = sport_of or _sport_of()
    races = sorted({int(r["race_id"]) for r in link_rows if r.get("race_id") is not None and not pd.isna(r["race_id"])})
    starts = {}
    if races:
        keys = _time_keys()
        start = "coalesce(" + ", ".join(f"extra->>'{k}'" for k in keys) + ")"
        for rid, kind, sd in conn.execute(text(f"""SELECT race_id, kind, {start} FROM rounds
                                                   WHERE race_id = ANY(:r) AND {start} IS NOT NULL"""),
                                          dict(r=races)):
            starts[(rid, kind)] = sd
    out = {}
    for r in link_rows:
        rid = r.get("race_id")
        if rid is None or pd.isna(rid):
            continue
        rid, kind = int(rid), _kind(r)
        if (rid, kind) in out:
            continue
        rnd = deciding_round(kind)
        if (rid, rnd) in starts:
            out[(rid, kind)] = (_utc_naive(starts[(rid, rnd)]), f"{rnd} start")
        elif rnd == "race" and r.get("start_date") is not None:
            sport = sport_of.get(r["competition"], r["competition"])
            out[(rid, kind)] = (race_day_close(r["start_date"], sport), "race day (schema)")
        else:
            out[(rid, kind)] = (None, f"no stored start for the {rnd} round")
    return out


def _kind(link):
    """A link's kind: its prediction, or params.kind while it is unmodeled (the coverage counter's rule)."""
    if link["prediction"] == "unmodeled":
        return (link.get("params") or {}).get("kind") or "unmodeled"
    return link["prediction"]


def close_prices(conn, wants, stale=VR.STALE):
    """{(venue, token, close): YES price at the close or None}. wants: {(venue, close): {token: link}}. The venue
    class reads its own tape over [close - stale, close]: a quote's mid when it has one, else its price."""
    out = {}
    for (venue, close), toks in wants.items():
        cls = VR.EXCHANGES.get(venue)
        if cls is None:
            for t in toks:
                out[(venue, t, close)] = None
            continue
        lk = pd.DataFrame([dict(token_id=t, condition_id=r.get("condition_id"), prediction=_kind(r))
                           for t, r in toks.items()])
        v = cls(conn, lk, close - stale, close, stale=stale)
        for t in toks:
            q = v.quote(t, close)
            out[(venue, t, close)] = (q[0] + q[1]) / 2 if q else v.price(t, close)
    return out


# --- CLV -------------------------------------------------------------------------------------------------------

def compute(conn, bet_rows, now=None):
    """Per bet: BET_COLS plus token, close_ts, close_rule, close (YES price), clv (points), clv_pct, mid_move,
    status ('decided' or 'undecided') and reason."""
    now = _utc_naive(now or pd.Timestamp.now(tz="UTC"))
    b = pd.DataFrame(bet_rows).reindex(columns=BET_COLS).reset_index(drop=True)
    for c in ("token", "close_ts", "close_rule", "close", "clv", "clv_pct", "mid_move", "status", "reason"):
        b[c] = None
    if not len(b):
        return b
    by_venue = {}
    for v, k in zip(b["venue"], b["market_key"]):
        by_venue.setdefault(v, set()).add(k)
    lk = links(conn, by_venue)
    cl = closes(conn, list(lk.values()))
    sport_of = _sport_of()
    wants, rows = {}, []
    for i, r in b.iterrows():
        link = lk.get((r["venue"], str(r["market_key"])))
        if link is None:
            rows.append((i, None, None, None, f"no {r['venue']} link for this market key"))
            continue
        if pd.isna(r["sport"]) or r["sport"] in (None, "unknown"):
            b.at[i, "sport"] = sport_of.get(link["competition"], link["competition"])
        if pd.isna(r["kind"]) or r["kind"] is None:
            b.at[i, "kind"] = _kind(link)
        if link["race_id"] is None or pd.isna(link["race_id"]):
            rows.append((i, link, None, None, "season market: no race close"))
            continue
        close, rule = cl[(int(link["race_id"]), _kind(link))]
        entry_ts = _utc_naive(r["entry_ts"])
        if close is None:
            reason = rule
        elif close > now:
            reason = "not closed yet"
        elif entry_ts is not None and entry_ts > close:
            reason = "entered after the close"
        elif r["entry"] is None or pd.isna(r["entry"]):
            reason = "no entry price"
        else:
            reason = None
            wants.setdefault((r["venue"], close), {})[link["token_id"]] = link
        rows.append((i, link, close, rule, reason))
    px = close_prices(conn, wants)
    for i, link, close, rule, reason in rows:
        r = b.loc[i]
        b.at[i, "token"] = link["token_id"] if link else None
        b.at[i, "close_ts"], b.at[i, "close_rule"] = close, rule if close is not None else None
        if reason is None:
            p = px.get((r["venue"], link["token_id"], close))
            if p is None:
                reason = f"no {r['venue']} price in the {int(VR.STALE.total_seconds() // 3600)} h before the close"
            else:
                yes, d = exposure(r["action"], r["side"], r["entry"])
                paid = yes if d > 0 else 1 - yes
                clv = d * (p - yes)
                b.at[i, "close"], b.at[i, "clv"] = p, clv
                b.at[i, "clv_pct"] = clv / paid if paid > 0 else None
                if r["mid"] is not None and not pd.isna(r["mid"]):
                    b.at[i, "mid_move"] = d * (p - float(r["mid"]))
        b.at[i, "status"] = "decided" if reason is None else "undecided"
        b.at[i, "reason"] = reason
    for c in ("close", "clv", "clv_pct", "mid_move"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["entry_ts"] = b["entry_ts"].map(_utc_naive)            # naive UTC, as close_ts
    return b


def aggregate(per_bet, by=GROUP):
    """Per sport x venue x kind x strategy x profile, then an 'all' row: bets, decided, undecided, mean CLV in points
    and percent, the share of decided bets with a positive CLV, and the mean mid move."""
    cols = list(by) + ["bets", "decided", "undecided", "mean_clv", "mean_clv_pct", "positive_share", "mean_mid_move"]
    if not len(per_bet):
        return pd.DataFrame(columns=cols)

    def row(g):
        d = g[g["status"] == "decided"]
        return dict(bets=len(g), decided=len(d), undecided=len(g) - len(d),
                    mean_clv=float(d["clv"].mean()) if len(d) else None,
                    mean_clv_pct=float(d["clv_pct"].mean()) if d["clv_pct"].notna().any() else None,
                    positive_share=float((d["clv"] > 0).mean()) if len(d) else None,
                    mean_mid_move=float(d["mid_move"].mean()) if d["mid_move"].notna().any() else None)

    g = per_bet.assign(**{k: per_bet[k].fillna("") for k in by})
    out = [dict(zip(by, key), **row(x)) for key, x in g.groupby(list(by), sort=True)]
    out.append(dict({k: "all" for k in by}, **row(per_bet)))
    return pd.DataFrame(out, columns=cols)


def run(conn, users=None, sport=None, venue=None, event=None, now=None):
    per = compute(conn, bets(conn, users, sport, venue, event), now=now)
    return dict(bets=per, summary=aggregate(per))


# --- a sweep's CLV (season_sweep / weekend_sweep: the saved run's metrics.clv) ----------------------------------

def sweep_bets(by_strategy, venue, sport=None):
    """Bet rows for every strategy of a sweep. by_strategy: {strategy: a taker mode's trades (from_trades) or a maker
    variant's fills (from_fills: the frame has `cond`)}, each with event_key and stage."""
    parts = [from_fills(t, venue, sport, s) if "cond" in t else from_trades(t, venue, sport, s)
             for s, t in by_strategy.items() if t is not None and len(t)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BET_COLS)


def _groups(per_bet, by):
    """aggregate() without its closing 'all' row, as JSON-ready records."""
    a = aggregate(per_bet, by)
    a = a[~(a[list(by)] == "all").all(axis=1)] if len(a) else a
    return json.loads(a.to_json(orient="records"))


def sweep_summary(per_bet, strategies=(), stages=()):
    """metrics.clv of a saved sweep, from compute()'s per-bet rows: dict(
        by_strategy  one row per strategy: bets, decided, undecided, mean_clv (points), mean_clv_pct, positive_share
                     (of decided bets), mean_mid_move
        by_stage     per strategy: a row per stage, then its 'all' row (= its by_strategy row)
        by_event     per strategy x event_key
        undecided    {reason: bets} over every strategy)
    strategies / stages: the order to list them in (others after, sorted)."""
    if not len(per_bet):
        return dict(by_strategy=[], by_stage=[], by_event=[], undecided={})
    events = per_bet.assign(t=pd.to_datetime(per_bet["entry_ts"])).groupby("event_key")["t"].min().sort_values()
    order = dict(strategy={s: i for i, s in enumerate(strategies)}, stage={s: i for i, s in enumerate(stages)},
                 event_key={e: i for i, e in enumerate(events.index)})       # events in the order they were traded

    def by_order(rows, *keys):
        return sorted(rows, key=lambda r: tuple((order[k].get(r[k], len(order[k])), str(r[k])) for k in keys))
    strat = by_order(_groups(per_bet, ("strategy",)), "strategy")
    stage = by_order(_groups(per_bet, ("strategy", "stage")), "strategy", "stage")
    rows = []
    for s in strat:
        rows += [r for r in stage if r["strategy"] == s["strategy"]] + [dict(s, stage="all")]
    why = per_bet.loc[per_bet["status"] == "undecided", "reason"].value_counts()
    return dict(by_strategy=strat, by_stage=rows,
                by_event=by_order(_groups(per_bet, ("strategy", "event_key")), "strategy", "event_key"),
                undecided={str(k): int(v) for k, v in why.items()})


def sweep(engine, by_strategy, venue, sport=None, strategies=(), stages=(), now=None):
    """A sweep's CLV against the database's closes: (sweep_summary, per-bet rows). Read-only."""
    with engine.connect() as conn:
        per = compute(conn, sweep_bets(by_strategy, venue, sport), now=now)
    return sweep_summary(per, strategies, stages), per


def sweep_line(clv, strategy):
    """One strategy's CLV for a sweep's printed totals, from its metrics.clv (sweep_summary): mean CLV in percent and
    points, the share of decided bets with a positive CLV, how many were decided."""
    clv = clv or {}
    if "error" in clv:
        return "CLV not computed"
    row = next((r for r in clv.get("by_strategy", []) if r["strategy"] == strategy), None)
    if not row or not row.get("decided"):
        return f"CLV n/a (0 of {(row or {}).get('bets', 0)} bets decided)"
    pct = row.get("mean_clv_pct")
    return (f"CLV {'n/a' if pct is None else f'{pct:+.2%}'} ({row['mean_clv']:+.4f}) · "
            f"{row['positive_share']:.0%} positive of {row['decided']} decided")


# --- output ----------------------------------------------------------------------------------------------------

BET_SHOW = ["sport", "venue", "event_key", "profile", "strategy", "kind", "subject", "action", "side", "entry_ts",
            "entry", "close_ts", "close", "clv", "clv_pct", "reason"]


def format_text(res, per_bet=True):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 5000)
    s, b = res["summary"], res["bets"]
    fmt = {"mean_clv": "{:+.4f}", "mean_clv_pct": "{:+.2%}", "positive_share": "{:.0%}", "mean_mid_move": "{:+.4f}",
           "clv": "{:+.4f}", "clv_pct": "{:+.2%}", "entry": "{:.4f}", "close": "{:.4f}"}

    def show(df):
        df = df.copy()
        for c, f in fmt.items():
            if c in df:
                df[c] = df[c].map(lambda v, f=f: "" if v is None or pd.isna(v) else f.format(v))
        return df.fillna("").to_string(index=False)

    lines = [f"=== CLV per sport x venue x kind x strategy x profile ({len(b)} paper bets) ==="]
    lines.append(show(s) if len(s) else "no paper bets")
    if per_bet and len(b):
        lines += ["", "=== CLV per bet (YES-price terms; clv_pct = CLV / the price paid for the side) ===",
                  show(b[BET_SHOW])]
    if len(b):
        why = b.loc[b["status"] == "undecided", "reason"].value_counts()
        if len(why):
            lines += ["", "undecided: " + "; ".join(f"{k} ({v})" for k, v in why.items())]
    return "\n".join(lines)


def to_json(res):
    def recs(df):
        return json.loads(df.to_json(orient="records", date_format="iso"))
    return dict(summary=recs(res["summary"]), bets=recs(res["bets"]))


def write(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["bets"].to_csv(out / "clv_bets.csv", index=False)
    res["summary"].to_csv(out / "clv_summary.csv", index=False)
    return out
