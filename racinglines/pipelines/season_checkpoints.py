"""
Championship-market checkpoints: enter at fixed points in the season, hold, and score each
entry over fixed windows. Built for comparing model variants on the season markets, where
the update-every-race strategy (season_strategy.py) mostly measures how fast the market
reacts to race results.

Entries (fixed before looking at results): pre-season, after 3 grands prix, after 6. At each,
a separate book ("tranche", capital / number of entries) takes the default season sizing
against that moment's as-of forecast (season_strategy.asof_forecasts, per variant) and holds.

Windows: the next WINDOW grands prix after the entry (the same length for every entry, so
entries compare), and to date. Both end at the price one hour after a decision, as traded.

Scores, per tranche and window:
  pnl        liquidation value (price - cost) + settlements - cost basis, as in the replay
  drift      mean move of the market toward our fair value, in points, over the markets
             where |fair - price| >= min_edge at entry: (p_end - p_entry) * sign(fair - p_entry)
  hit        share of those markets where the market moved toward our fair value
  slope      least-squares slope of (p_end - p_entry) on (fair - p_entry) over every
             tradeable market: the share of our edge the market closed. 1 = the market
             went all the way to our number, 0 = our edge told nothing about the move,
             negative = the market moved away. Its SE treats markets as independent, which
             they aren't (a title's probabilities sum to one), so read it as indicative.
"""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd

from racinglines.core.stats import last_at
from racinglines.markets.strategies import season as SS
from racinglines.pipelines import season_strategy as SE

ENTRIES = (0, 3, 6)          # grands prix completed at entry
WINDOW = 3                   # grands prix per scoring window
# variants whose championship forecast differs (gridq = baseline and gridq+pretrain = pretrain
# for the season: future races have no grid yet) and the combos worth testing
VARIANTS = ("baseline", "grid", "pretrain", "gbm", "tail", "grid+pretrain", "pretrain+tail", "grid+pretrain+tail")


def _end_price(mk, t):
    """Settled outcome if the market resolved by t, else the last price at t."""
    if mk.outcome is not None and mk.closed_at is not None and mk.closed_at <= t:
        return float(mk.outcome)
    return last_at(mk.ts, mk.px, t, timedelta(days=30))


def drift(markets, decision, t_entry, t_end, p: SS.SeasonParams):
    """Market drift toward our fair values between t_entry and t_end (see module doc)."""
    rows = []
    for k, mk in markets.items():
        fair = decision["fairs"].get(k)
        if fair is None or (mk.closed_at is not None and mk.closed_at <= t_entry):
            continue
        p0 = last_at(mk.ts, mk.px, t_entry, p.max_age)
        if p0 is None or not p.price_band[0] <= p0 <= p.price_band[1]:
            continue
        p1 = _end_price(mk, t_end)
        if p1 is not None:
            rows.append((fair - p0, p1 - p0))
    if not rows:
        return dict(markets=0, edged=0, drift=np.nan, hit=np.nan, slope=np.nan, slope_se=np.nan)
    e, m = np.array(rows).T
    big = np.abs(e) >= p.min_edge
    toward = m[big] * np.sign(e[big])
    slope = se = np.nan
    if (e ** 2).sum() > 0:
        slope = float((e * m).sum() / (e ** 2).sum())                       # through the origin
        resid = m - slope * e
        if len(e) > 1:
            se = float(np.sqrt((resid ** 2).sum() / (len(e) - 1) / (e ** 2).sum()))
    return dict(markets=len(e), edged=int(big.sum()), drift=float(toward.mean() * 100) if big.any() else np.nan,
                hit=float((toward > 0).mean()) if big.any() else np.nan, slope=slope, slope_se=se)


def score(markets, decisions, times, p=SS.SeasonParams(), entries=ENTRIES, window=WINDOW, now=None):
    """One row per (entry, window) for one variant. decisions: as in season_strategy (one per
    decision time, in order); times: [(label, t)] from season_strategy.decision_times."""
    now = now or pd.Timestamp.now(tz="UTC")
    tranche = replace(p, mode="hold", capital=p.capital / len(entries))
    rows = []
    for e in entries:
        if e >= len(decisions):
            continue
        d = decisions[e]
        t0 = d["t"] + p.exec_delay
        ends = [(f"next {window} GPs", decisions[e + window]["t"] + p.exec_delay)] if e + window < len(decisions) else []
        ends.append(("to date", now))
        for win, t1 in ends:
            r = SS.replay(markets, [d], tranche, now=t1, marks=[t1])
            s = r["summary"]
            rows.append(dict(entry=d["label"], entry_n=e, window=win, end=t1, pnl=s["pnl"], bought=s["bought"],
                             positions=s["trades"], **drift(markets, d, t0, t1, p)))
    return pd.DataFrame(rows)


def run(engine, engine_url, variants=VARIANTS, year=2026, p=SS.SeasonParams(), entries=ENTRIES, window=WINDOW,
        n_sims=5000, echo=print):
    """Score every variant. Only the entry forecasts are needed; missing ones are computed and
    stored (kind 'season_asof', keyed by variant), so a new combo costs len(entries) forecasts."""
    from racinglines.models.inputs import laps_pricing as run_
    from racinglines.models import variants as V
    with engine.connect() as c:
        links = SE.season_links(c, year)
        markets = SE.build_markets(c, links)
    meas = run_.Measurements.load(engine)
    times = SE.decision_times(meas, year)
    need = [times[e] for e in entries if e < len(times)]
    out = []
    for v in variants:
        with V.use(v):
            hist = run_.history(meas)
            got = SE.asof_forecasts(engine, engine_url, meas, hist, year, need, n_sims=n_sims, echo=echo, variant=v)
        by_t = {t: rid for _, t, rid in got}
        with engine.connect() as c:
            decisions = [dict(t=t, label=label, run_id=by_t.get(t),
                              fairs=SE.fairs(c, links, by_t[t]) if t in by_t else {}) for label, t in times]
        sc = score(markets, decisions, times, p, entries, window)
        out.append(sc.assign(variant=v, runs=",".join(str(r) for _, _, r in got)))
        echo(f"progress {len(out)}/{len(variants)} {v}")
    return pd.concat(out, ignore_index=True), dict(markets=len(markets), decisions=len(times))


def format_table(df):
    """Markdown: per variant, P&L and drift for each entry over the fixed window and to date,
    plus the ladder (all tranches together) to date."""
    fmt = lambda x, f: "" if x is None or pd.isna(x) else f.format(x)          # noqa: E731
    out = []
    ents = df.drop_duplicates("entry_n").sort_values("entry_n")["entry"].tolist()
    wins = [w for w in df["window"].unique() if w != "to date"]
    head = ["Model"] + [f"{e}: {w}" for e in ents for w in wins] + [f"{e}: to date" for e in ents] + ["Ladder, to date"]
    out.append("P&L in $ (market drift toward our fair value, points; slope)")
    out.append("")
    out.append("| " + " | ".join(head) + " |")
    out.append("|---" * len(head) + "|")
    for v, g in df.groupby("variant", sort=False):
        cells = []
        for e in ents:
            for w in wins:
                r = g[(g["entry"] == e) & (g["window"] == w)]
                cells.append(_cell(r, fmt))
        for e in ents:
            cells.append(_cell(g[(g["entry"] == e) & (g["window"] == "to date")], fmt))
        lad = g[g["window"] == "to date"]
        cells.append(f"{lad['pnl'].sum():+,.0f}")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _cell(r, fmt):
    if not len(r):
        return ""
    r = r.iloc[0]
    return f"{r['pnl']:+,.0f} ({fmt(r['drift'], '{:+.1f}')}; {fmt(r['slope'], '{:.2f}')})"


def format_summary(df, window=WINDOW):
    """Compact markdown: per variant, P&L (slope) over the fixed window from each entry, and
    all tranches together to date."""
    ents = df.drop_duplicates("entry_n").sort_values("entry_n")
    win = f"next {window} GPs"
    head = ["Model"] + [("Pre-season" if n == 0 else f"After GP {n}") + f" → +{window} GPs" for n in ents["entry_n"]] \
        + ["All three, to date"]
    out = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for v, g in df.groupby("variant", sort=False):
        cells = []
        for n in ents["entry_n"]:
            r = g[(g["entry_n"] == n) & (g["window"] == win)]
            cells.append("" if not len(r) else f"{r['pnl'].iloc[0]:+,.0f} ({r['slope'].iloc[0]:+.2f})")
        cells.append(f"{g.loc[g['window'] == 'to date', 'pnl'].sum():+,.0f}")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(out)
