"""
Cross-venue disagreement log (roadmap U7): where Polymarket and Kalshi price the same F1 outcome differently.

For every outcome linked on both exchanges (a race's win / podium / top 10 / pole / top constructor / head-to-head
markets, and the season's drivers' and constructors' champion markets) and every tick of a time grid, one row of
`market_disagreements`:

    pm_mid, pm_bid, pm_ask          Polymarket's price and top of book at the tick (the outcome's YES side)
    kalshi_mid, kalshi_bid, kalshi_ask   the same on Kalshi
    fair, run_id                    our fair value from the run in force at the tick (see `fair_runs`; no run
                                    yet = no fair and no edges, the gap columns still fill)
    pm_edge, pm_side                the better taker trade on Polymarket vs fair, net of its taker fee: buy YES at the
                                    ask (fair - ask - fee) or sell at the bid (bid - fair - fee); > 0 = an edge
    kalshi_edge, kalshi_side        the same on Kalshi, with Kalshi's taker fee (TAKER_FEE x P x (1 - P) per contract)
    gap, gap_net                    kalshi_mid - pm_mid, and |gap| net of both venues' taker fees (> 0 = the venues
                                    disagree by more than it costs to take both sides at mid)
    pm_vol24, kalshi_vol24          USD traded on each venue in the 24 h before the tick (`market_trades`; null where
                                    no tape is stored for the token). A row is *liquid* when neither venue's tape
                                    shows nothing traded: a dead book (Polymarket's history reads 0.5 for an empty
                                    book) is a gap on paper only, so the report's counts use liquid rows.

A mid is the venue's last stored price at or before the tick (`market_price_history`, Parquet and Postgres), no
older than STALE; a recorded order book within one step of the tick gives the top of book and replaces the mid
with (bid + ask) / 2. Without a book, the edge is measured against the mid (no spread), which flatters it.

`racinglines markets disagree --event …` backfills the log from stored prices and prints the report;
`record(engine)` writes the current tick (the recorder calls it on every pass when RACINGLINES_DISAGREE=1).
Everything here is off unless RACINGLINES_DISAGREE=1 (`ON`): the Markets page panel and the recorder hook.
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.db import reads as data
from racinglines.markets import store as MS
from racinglines.markets import venues as V
from racinglines.markets.venue_replay import EXCHANGES

ON = {"on": os.environ.get("RACINGLINES_DISAGREE", "0") == "1"}
TAKER_FEE = {code: v.TAKER_FEE for code, v in EXCHANGES.items()}  # per contract, as a share of P x (1 - P)
KINDS = ("race_win", "race_podium", "race_top10", "race_pole", "race_constructor_top", "race_h2h",
         "champion", "constructors_champion")
SEASON_KINDS = ("champion", "constructors_champion")
STEP = timedelta(hours=1)
STALE = timedelta(hours=6)          # a price older than this at a tick is no quote
COLS = ["pm_token", "kalshi_token", "ts", "competition_id", "race_id", "kind", "athlete_id", "team", "opponent_id", "n",
        "pm_mid", "pm_bid", "pm_ask", "kalshi_mid", "kalshi_bid", "kalshi_ask", "fair", "run_id",
        "pm_edge", "pm_side", "kalshi_edge", "kalshi_side", "gap", "gap_net", "pm_vol24", "kalshi_vol24"]
VOL_WINDOW = timedelta(hours=24)


def fee(exchange, price):
    """A venue's taker fee per contract at `price`, in price units (Kalshi: TAKER_FEE x P x (1 - P); Polymarket: 0)."""
    return TAKER_FEE.get(exchange, 0.0) * price * (1 - price)


def edge(exchange, fair, mid, bid=None, ask=None):
    """(edge, side) of the better taker trade vs `fair`: buy YES at the ask (else the mid) or sell at the bid (else
    the mid), each net of the venue's taker fee. Positive = an edge. (None, None) without a fair or a mid."""
    if fair is None or mid is None:
        return None, None
    a = ask if ask is not None else mid
    b = bid if bid is not None else mid
    buy = fair - a - fee(exchange, a)
    sell = b - fair - fee(exchange, b)
    return (buy, "buy") if buy >= sell else (sell, "sell")


def gap_net(pm_mid, kalshi_mid):
    """|kalshi_mid - pm_mid| net of both venues' taker fees at their mids."""
    return abs(kalshi_mid - pm_mid) - fee("kalshi", kalshi_mid) - fee("polymarket", pm_mid)


# ---------------------------------------------------------------------------------------------------
# which outcomes are on both venues
# ---------------------------------------------------------------------------------------------------

def _season_window(year):
    return datetime(year, 6, 1, tzinfo=timezone.utc), datetime(year + 1, 7, 1, tzinfo=timezone.utc)


def pairs(conn, race_id=None, competition_id=None, year=None):
    """One row per outcome linked on both exchanges: the outcome key (kind, athlete_id, team, opponent_id, n), its
    name, and each venue's link (token, invert, link id). A race's markets with `race_id`; the season's champion
    markets with `competition_id` and `year` (links whose end date falls in that season)."""
    if race_id is not None:
        links = data.q(conn, """SELECT ml.*, a.display_name AS athlete FROM market_links ml
                                LEFT JOIN athletes a ON a.id = ml.athlete_id
                                WHERE ml.race_id = :r AND ml.prediction = ANY(:k) ORDER BY ml.id""", r=race_id, k=list(KINDS))
    else:
        a, b = _season_window(year or datetime.now(timezone.utc).year)
        links = data.q(conn, """SELECT ml.*, a.display_name AS athlete FROM market_links ml
                                LEFT JOIN athletes a ON a.id = ml.athlete_id
                                WHERE ml.race_id IS NULL AND ml.competition_id = :c AND ml.prediction = ANY(:k)
                                  AND ml.end_date >= :a AND ml.end_date < :b ORDER BY ml.id""",
                       c=competition_id, k=list(SEASON_KINDS), a=a, b=b)
    by = {}
    seen = set()
    for link in links.to_dict("records"):
        q = link["token_id"] if link["exchange"] == "kalshi" else link["condition_id"]
        if link["prediction"] in ("race_h2h", "standings_h2h") and (link["exchange"], q) in seen:
            continue                         # a Polymarket head-to-head has two tokens: keep the first
        seen.add((link["exchange"], q))
        k = V.key(link["prediction"], link["athlete_id"], link["params"])
        by.setdefault(k, {}).setdefault(link["exchange"], link)      # the first (oldest) link per venue and outcome
    rows = []
    opponents = V._subject_names(conn, [k[3] for k in by if k[3] is not None])
    for k, ex in by.items():
        if "polymarket" not in ex or "kalshi" not in ex:
            continue
        pm, ks = ex["polymarket"], ex["kalshi"]
        kind, ath, team, opp, n = k
        subject = pm.get("athlete") or ks.get("athlete") or V.team_name(team) or ks["outcome"]
        if opp is not None:
            subject += f" ahead of {opponents.get(opp, '?')}"
        elif n is not None:
            subject += f" {n}+"
        rows.append(dict(kind=kind, athlete_id=ath, team=team, opponent_id=opp, n=n, subject=subject,
                         competition_id=int(pm["competition_id"]), race_id=race_id,
                         pm_token=pm["token_id"], pm_invert=bool(pm["invert"]), pm_link=pm,
                         kalshi_token=ks["token_id"], kalshi_invert=bool(ks["invert"]), kalshi_link=ks))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# our fair value at a time
# ---------------------------------------------------------------------------------------------------

def fair_runs(conn, race_id=None, competition_id=None):
    """The runs that priced these outcomes, oldest first, each with the time it is a price 'as of': for a race,
    diagnostics and forecasts with predictions for it (as of the diagnostic's cutoff, else the run's creation);
    for a season, the competition's forecasts (as of the cutoff, else promotion, else creation).
    -> DataFrame(run_id, as_of)"""
    if race_id is not None:
        df = data.q(conn, """
            SELECT mr.id AS run_id, coalesce((mr.params->>'cutoff')::timestamp, mr.created_at::timestamp) AS as_of
            FROM model_runs mr WHERE mr.kind IN ('diagnostic', 'forecast')
              AND EXISTS (SELECT 1 FROM race_predictions rp WHERE rp.model_run_id = mr.id AND rp.race_id = :r)
            ORDER BY as_of, mr.id""", r=race_id)
    else:
        df = data.q(conn, """
            SELECT mr.id AS run_id, coalesce((mr.params->>'cutoff')::timestamp, (mr.params->>'promoted_at')::timestamp,
                                             mr.created_at::timestamp) AS as_of
            FROM model_runs mr WHERE mr.kind = 'forecast' AND mr.competition_id = :c
              AND EXISTS (SELECT 1 FROM standings_predictions sp WHERE sp.model_run_id = mr.id)
            ORDER BY as_of, mr.id""", c=competition_id)
    if len(df):
        df["as_of"] = pd.to_datetime(df["as_of"], utc=True)
    return df


def runs_at(runs, ts):
    """For each time in `ts`, the id of the latest run as of then (None before the first run: no fair, no edge).
    -> list"""
    if runs is None or not len(runs):
        return [None] * len(ts)
    as_of = pd.DatetimeIndex(runs["as_of"]).tz_convert("UTC").tz_localize(None).to_numpy()
    ts = pd.DatetimeIndex(ts)
    ts = (ts.tz_convert("UTC") if ts.tz is not None else ts).tz_localize(None).to_numpy()
    idx = np.searchsorted(as_of, ts, side="right") - 1
    ids = runs["run_id"].to_numpy()
    return [None if i < 0 else int(ids[i]) for i in idx]


class Fair:
    """Our fair value of an outcome (its YES side, whichever side the token is) from a run, cached per run and pair;
    `runs_at(ts)` says which run is in force at each time."""

    def __init__(self, conn, runs):
        self.conn, self.runs, self.cache, self.metrics = conn, runs, {}, {}

    def runs_at(self, ts):
        return runs_at(self.runs, ts)

    def prob(self, pair, run_id):
        key = (run_id, pair["pm_token"])
        if key not in self.cache:
            link = dict(pair["pm_link"], invert=False)
            self.cache[key] = data.model_prob(self.conn, link, self.metrics, run_id=run_id)[0]
        return self.cache[key]


# ---------------------------------------------------------------------------------------------------
# the log
# ---------------------------------------------------------------------------------------------------

def _grid(start, end, step):
    a = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start).tz_localize("UTC")
    b = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end).tz_localize("UTC")
    return pd.date_range(a.floor(step), b, freq=step)


def _at(series, grid, tolerance):
    """The last value at or before each grid point, within `tolerance` (NaN otherwise)."""
    if series is None or not len(series):
        return pd.Series(np.nan, index=grid, dtype=float)
    s = series[~series.index.duplicated(keep="last")].sort_index()
    return s.reindex(grid, method="ffill", tolerance=tolerance)


def _quotes(px, bk, tok, invert, grid, stale, step):
    """One venue's (mid, bid, ask) on the grid, as the outcome's YES side: the last stored price within `stale`,
    replaced by the mid of a book recorded within `step`, which also gives the top of book. An inverted link's
    token pays when the outcome does NOT happen, so its quote is flipped (bid and ask swap)."""
    mid = _at(px.get(tok), grid, stale)
    book = bk.get(tok)
    if book is not None:
        bid, ask = _at(book["best_bid"], grid, step), _at(book["best_ask"], grid, step)
        both = bid.notna() & ask.notna()
        mid = mid.where(~both, (bid + ask) / 2)
    else:
        bid = ask = pd.Series(np.nan, index=grid, dtype=float)
    if invert:
        mid, bid, ask = 1 - mid, 1 - ask, 1 - bid
    return pd.DataFrame(dict(mid=mid, bid=bid, ask=ask))


def _vol24(tr, tok, grid, window=VOL_WINDOW):
    """USD traded on `tok` in the `window` before each grid point (None when no tape is stored for the token)."""
    g = tr.get(tok)
    if g is None:
        return pd.Series(None, index=grid, dtype=object)
    ts, cum = g.index.to_numpy(), np.concatenate([[0.0], np.cumsum(g.to_numpy(float))])
    hi = cum[np.searchsorted(ts, grid.to_numpy(), side="right")]
    lo = cum[np.searchsorted(ts, (grid - window).to_numpy(), side="right")]
    return pd.Series(hi - lo, index=grid, dtype=object)


def liquid(df):
    """Rows where neither venue's stored tape shows nothing traded in the prior 24 h (a null volume = no tape)."""
    ok = pd.Series(True, index=df.index)
    for c in ("pm_vol24", "kalshi_vol24"):
        v = pd.to_numeric(df[c], errors="coerce")
        ok &= v.isna() | (v > 0)
    return ok


def _edges(exchange, fair, mid, bid, ask):
    """Vectorised `edge`: (edge, side) Series, NaN / None without a fair."""
    a, b = ask.fillna(mid), bid.fillna(mid)
    buy = fair - a - TAKER_FEE.get(exchange, 0.0) * a * (1 - a)
    sell = b - fair - TAKER_FEE.get(exchange, 0.0) * b * (1 - b)
    best = buy.where(buy >= sell, sell)
    side = pd.Series(np.where(buy >= sell, "buy", "sell"), index=mid.index, dtype=object).where(best.notna(), None)
    return best, side


def build(conn, prs, start, end, step=STEP, grid=None, fair=None, root=None, stale=STALE):
    """The log's rows for these pairs on a time grid (every `step` from `start` to `end`, or `grid`), from the
    stored prices and books. `fair` is a `Fair` (default: from the database's runs; None with no connection: no
    fair, no edges). Rows where either venue has no fresh price are left out. -> DataFrame(COLS + subject)"""
    if not len(prs):
        return pd.DataFrame(columns=COLS + ["subject"])
    grid = pd.DatetimeIndex(grid) if grid is not None else _grid(start, end, step)
    if not len(grid):
        return pd.DataFrame(columns=COLS + ["subject"])
    toks = list(prs["pm_token"]) + list(prs["kalshi_token"])
    lo, hi = grid[0] - stale, grid[-1]
    px = MS.read(conn, "prices", tokens=toks, start=lo, end=hi, root=root)
    bk = MS.read(conn, "books", tokens=toks, start=lo, end=hi, root=root)
    tr = MS.read(conn, "trades", tokens=toks, root=root)       # the whole tape: a token with none has no tape at all
    px = {t: g.set_index("ts")["price"].astype(float) for t, g in px.groupby("token_id")} if len(px) else {}
    bk = {t: g.set_index("ts")[["best_bid", "best_ask"]].astype(float) for t, g in bk.groupby("token_id")} if len(bk) else {}
    tr = {t: (g["price"].astype(float) * g["size"].astype(float)).set_axis(g["ts"]).sort_index()
          for t, g in tr.groupby("token_id")} if len(tr) else {}
    step = pd.Timedelta(step) if step is not None else (grid[1] - grid[0] if len(grid) > 1 else stale)
    if fair is None and conn is not None:
        first = prs.iloc[0]
        fair = Fair(conn, fair_runs(conn, race_id=first["race_id"], competition_id=int(first["competition_id"])))
    run_ids = fair.runs_at(grid) if fair is not None else [None] * len(grid)
    parts = []
    for pair in prs.to_dict("records"):
        pm = _quotes(px, bk, pair["pm_token"], pair["pm_invert"], grid, stale, step)
        ks = _quotes(px, bk, pair["kalshi_token"], pair["kalshi_invert"], grid, stale, step)
        df = pd.DataFrame(dict(ts=grid, pm_mid=pm["mid"].to_numpy(), pm_bid=pm["bid"].to_numpy(), pm_ask=pm["ask"].to_numpy(),
                               kalshi_mid=ks["mid"].to_numpy(), kalshi_bid=ks["bid"].to_numpy(), kalshi_ask=ks["ask"].to_numpy(),
                               run_id=pd.Series(run_ids, dtype=object), pm_vol24=_vol24(tr, pair["pm_token"], grid).to_numpy(),
                               kalshi_vol24=_vol24(tr, pair["kalshi_token"], grid).to_numpy()))
        df = df[df["pm_mid"].notna() & df["kalshi_mid"].notna()]
        if not len(df):
            continue
        probs = {rid: fair.prob(pair, rid) for rid in set(df["run_id"]) if rid is not None}
        df["fair"] = df["run_id"].map(lambda r: np.nan if r is None or probs.get(r) is None else probs[r]).astype(float)
        df["pm_edge"], df["pm_side"] = _edges("polymarket", df["fair"], df["pm_mid"], df["pm_bid"], df["pm_ask"])
        df["kalshi_edge"], df["kalshi_side"] = _edges("kalshi", df["fair"], df["kalshi_mid"], df["kalshi_bid"], df["kalshi_ask"])
        df["gap"] = df["kalshi_mid"] - df["pm_mid"]
        df["gap_net"] = df["gap"].abs() - TAKER_FEE["kalshi"] * df["kalshi_mid"] * (1 - df["kalshi_mid"]) \
            - TAKER_FEE["polymarket"] * df["pm_mid"] * (1 - df["pm_mid"])
        for c in ("pm_token", "kalshi_token", "competition_id", "race_id", "kind", "athlete_id", "team", "opponent_id", "n", "subject"):
            df[c] = pair[c]
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=COLS + ["subject"])
    out = pd.concat(parts, ignore_index=True)[COLS + ["subject"]]
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def save(conn, df):
    """Upsert the rows into market_disagreements (one per outcome pair and tick). Returns rows written."""
    if not len(df):
        return 0
    rows = df[COLS].astype(object).where(df[COLS].notna(), None).to_dict("records")
    for r in rows:
        r["ts"] = pd.Timestamp(r["ts"]).to_pydatetime()
        for c in ("competition_id", "race_id", "athlete_id", "opponent_id", "n", "run_id"):
            r[c] = None if r[c] is None else int(r[c])
        for c in ("pm_edge", "kalshi_edge", "fair", "pm_vol24", "kalshi_vol24"):
            r[c] = None if r[c] is None or pd.isna(r[c]) else float(r[c])
    conn.execute(text(f"""
        INSERT INTO market_disagreements ({', '.join(COLS)}) VALUES ({', '.join(':' + c for c in COLS)})
        ON CONFLICT (pm_token, kalshi_token, ts) DO UPDATE SET
        {', '.join(f'{c} = EXCLUDED.{c}' for c in COLS if c not in ('pm_token', 'kalshi_token', 'ts'))}"""), rows)
    return len(rows)


def coverage(conn, prs, root=None):
    """(start, end) where both venues have stored prices for these pairs, or (None, None)."""
    lo, hi = None, None
    for col in ("pm_token", "kalshi_token"):
        px = MS.read(conn, "prices", tokens=list(prs[col]), root=root)
        if not len(px):
            return None, None
        a, b = px["ts"].min(), px["ts"].max()
        lo, hi = (a if lo is None else max(lo, a)), (b if hi is None else min(hi, b))
    return (lo, hi) if lo is not None and lo <= hi else (None, None)


# ---------------------------------------------------------------------------------------------------
# the recorder's tick and the report
# ---------------------------------------------------------------------------------------------------

def record(engine, now=None, year=None):
    """One tick of the log for every open race listed on both venues and the season's champion markets
    (the recorder's pass, with RACINGLINES_DISAGREE=1). Returns rows written."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC").floor("min")
    n = 0
    with engine.begin() as c:
        comps = data.q(c, """SELECT DISTINCT co.id FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
                             WHERE ml.exchange = 'kalshi' AND ml.race_id IS NULL AND NOT ml.closed""")
        races = data.q(c, """SELECT DISTINCT ml.race_id FROM market_links ml JOIN races ra ON ra.id = ml.race_id
                             JOIN events e ON e.id = ra.event_id WHERE ml.exchange = 'kalshi' AND e.status <> 'completed'""")
        for cid in comps["id"]:
            n += save(c, build(c, pairs(c, competition_id=int(cid), year=year or now.year), None, None, grid=[now]))
        for rid in races["race_id"]:
            n += save(c, build(c, pairs(c, race_id=int(rid)), None, None, grid=[now]))
    return n


def by_day(df):
    """The report's rows: per UTC day, ticks, outcomes, liquid rows (both venues traded in the prior 24 h, where a
    tape is stored) and, of those, the rows above fees and their share, the mean |gap|, the median net gap, the
    widest net gap with its outcome and both mids (Kalshi vs Polymarket). A jump inside one step shows as a
    one-tick gap: read the share and the median, not the max."""
    cols = ["day", "ticks", "outcomes", "rows", "liquid", "above_fees", "share", "mean_gap", "median_net", "max_gap_net",
            "widest", "kalshi_vs_pm"]
    if not len(df):
        return pd.DataFrame(columns=cols)
    d = df.assign(day=pd.to_datetime(df["ts"], utc=True).dt.strftime("%Y-%m-%d"), liquid=liquid(df))
    out = []
    for day, g in d.groupby("day"):
        lq = g[g["liquid"]]
        row = dict(day=day, ticks=int(g["ts"].nunique()), outcomes=int(g[["pm_token", "kalshi_token"]].drop_duplicates().shape[0]),
                   rows=len(g), liquid=len(lq), above_fees=int((lq["gap_net"] > 0).sum()),
                   share=float((lq["gap_net"] > 0).mean()) if len(lq) else np.nan,
                   mean_gap=float(lq["gap"].abs().mean()) if len(lq) else np.nan,
                   median_net=float(lq["gap_net"].median()) if len(lq) else np.nan,
                   max_gap_net=np.nan, widest="", kalshi_vs_pm="")
        if len(lq):
            top = lq.loc[lq["gap_net"].idxmax()]
            row.update(max_gap_net=float(top["gap_net"]),
                       widest=f"{V.KIND_LABEL.get(top['kind'], top['kind'])} · {top.get('subject') or top['pm_token']}",
                       kalshi_vs_pm=f"{top['kalshi_mid']:.3f} vs {top['pm_mid']:.3f}")
        out.append(row)
    return pd.DataFrame(out, columns=cols)


def _f(fmt):
    return lambda v: "" if v is None or pd.isna(v) else fmt.format(v)


def report(df, title=""):
    """The printed report: the day-by-day table, then the widest net gaps by outcome (liquid rows)."""
    lines = [f"=== Cross-venue disagreement: {title} ===" if title else "=== Cross-venue disagreement ==="]
    if not len(df):
        lines.append("no tick where both venues quoted the same outcome")
        return "\n".join(lines)
    lq = df[liquid(df)]
    n_above = int((lq["gap_net"] > 0).sum())
    lines.append(f"{len(df)} rows · {df[['pm_token', 'kalshi_token']].drop_duplicates().shape[0]} outcomes · "
                 f"{df['ts'].nunique()} ticks · {len(lq)} liquid (both venues traded in the prior 24 h, where a tape is "
                 f"stored) · {n_above} of those above fees ({n_above / len(lq) if len(lq) else 0:.0%}) · "
                 f"Kalshi taker fee {TAKER_FEE['kalshi']:.0%} x P(1-P), Polymarket {TAKER_FEE['polymarket']:.0%}")
    lines.append("")
    lines.append("--- by day (gap = Kalshi mid - Polymarket mid; net = |gap| - both taker fees; liquid rows) ---")
    days = by_day(df)
    lines.append(days.to_string(index=False, formatters=dict(share=_f("{:.0%}"), mean_gap=_f("{:.3f}"),
                                                             median_net=_f("{:+.3f}"), max_gap_net=_f("{:+.3f}"))))
    lines.append("")
    lines.append("--- by outcome (liquid rows above fees, widest net gap, mean gap, each venue's mid and taker edge vs fair) ---")
    rows = []
    for (kind, subj), g in lq.groupby(["kind", "subject"], dropna=False):
        top = g.loc[g["gap_net"].idxmax()]
        pe = None if pd.isna(top["pm_edge"]) else top["pm_edge"]
        ke = None if pd.isna(top["kalshi_edge"]) else top["kalshi_edge"]
        rows.append(dict(outcome=f"{V.KIND_LABEL.get(kind, kind)} · {subj}", above=int((g["gap_net"] > 0).sum()), of=len(g),
                         widest=f"{top['gap_net']:+.3f}", at=pd.Timestamp(top["ts"]).strftime("%Y-%m-%d %H:%M"),
                         mean_gap=f"{g['gap'].mean():+.3f}", fair="" if pd.isna(top["fair"]) else f"{top['fair']:.3f}",
                         pm=f"{top['pm_mid']:.3f}" + (f" {top['pm_side']} {pe:+.3f}" if pe is not None else ""),
                         kalshi=f"{top['kalshi_mid']:.3f}" + (f" {top['kalshi_side']} {ke:+.3f}" if ke is not None else "")))
    rows.sort(key=lambda r: -float(r["widest"]))
    lines.append(pd.DataFrame(rows).to_string(index=False) if rows else "no liquid row")
    return "\n".join(lines)


def panel(conn, days=7):
    """What the Markets page shows (RACINGLINES_DISAGREE=1): the latest tick per outcome pair, widest net gap
    first (`above`: liquid and above fees), and the last `days` days' summary. None when the log is empty."""
    latest = data.q(conn, """
        SELECT DISTINCT ON (d.pm_token, d.kalshi_token) d.*, a.display_name AS athlete, e.name AS event, v.name AS venue
        FROM market_disagreements d LEFT JOIN athletes a ON a.id = d.athlete_id
        LEFT JOIN races ra ON ra.id = d.race_id LEFT JOIN events e ON e.id = ra.event_id LEFT JOIN venues v ON v.id = e.venue_id
        ORDER BY d.pm_token, d.kalshi_token, d.ts DESC""")
    if not len(latest):
        return None
    latest = latest.assign(liquid=liquid(latest)).sort_values("gap_net", ascending=False)
    latest = latest.astype(object).where(latest.notna(), None)
    rows = []
    for r in latest.to_dict("records"):
        rid = None if r["race_id"] is None else int(r["race_id"])
        rows.append(dict(r, subject=r["athlete"] or V.team_name(r["team"]) or "?", kind_label=V.KIND_LABEL.get(r["kind"], r["kind"]),
                         rid=rid, event=(f"{r['venue']} GP" if r["venue"] else r["event"]) if rid else "Championship",
                         above=bool(r["liquid"]) and r["gap_net"] is not None and r["gap_net"] > 0,
                         ts=pd.Timestamp(r["ts"]).tz_convert("UTC")))
    since = datetime.now(timezone.utc) - timedelta(days=days)
    recent = data.q(conn, "SELECT * FROM market_disagreements WHERE ts >= :s", s=since)
    return dict(rows=rows, n_above=sum(r["above"] for r in rows), last=max(r["ts"] for r in rows),
                days=by_day(recent).to_dict("records") if len(recent) else [], window=days)
