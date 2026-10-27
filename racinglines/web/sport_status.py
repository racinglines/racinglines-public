"""
The Markets page's per-sport status (RACINGLINES_SPORT_STATUS=1, off by default): for every sport with a schema, what
the database holds and what is missing, in one row each, so a sport with no model or no data says so instead of
disappearing from the page.

    race data     events and races stored, races with results, the latest result and the next scheduled event
    market data   links per exchange (with a race / open), the last sync
    model         the schema's model family, the latest forecast run (a live price), the stored as-of replay runs
                  (`<sport> replay --save`: a pre-race price per race), the season-forecast runs
    backtests     model backtest runs and the replay settings grid on disk (data/runs/replay-grid/<sport>)
    sweeps        saved Edge Finder strategy sweeps
    paper         the demo accounts' paper record for the sport (races, P&L), never a buy_all row

Everything is read-only and counted from the tables (no Parquet archive scan: the exchange data table below it on the
board has those counts). Each cell carries a state: ok (there), partial (some of it), none (missing), n/a (the sport
has no model by design).
"""

import os
from pathlib import Path

import pandas as pd

from racinglines import sports as SP
from racinglines.db import reads as data

SWITCH = "RACINGLINES_SPORT_STATUS"


def enabled():
    return os.environ.get(SWITCH, "").strip().lower() in ("1", "true", "yes", "on")


def _i(v):
    return 0 if v is None or pd.isna(v) else int(v)


def _grid(sport):
    """The replay settings grid on disk (scripts/vm/replay_grid.py): runs with a summary, seasons, and grid.md's time."""
    from racinglines.paths import DATA
    root = Path(DATA) / "runs" / "replay-grid" / sport
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir() if p.is_dir() and any(p.glob("*/summary.json"))]
    if not runs:
        return None
    md = root / "grid.md"
    seasons = sorted({p.name[:4] for p in runs if p.name[:4].isdigit()})
    return dict(runs=len(runs), seasons=seasons, ranked=md.is_file(),
                at=pd.Timestamp(md.stat().st_mtime, unit="s", tz="UTC") if md.is_file() else None)


def status(conn, hide=frozenset(), hide_venues=frozenset()):
    """One dict per sport (schema display order): name, competition, modeled, and a cell per column, each
    dict(state=ok|partial|none|na, text=..., sub=...)."""
    races = data.q(conn, """
        SELECT co.code AS competition, count(DISTINCT e.id) AS events, count(DISTINCT ra.id) AS races,
               min(s.year) AS first_year, max(s.year) AS last_year,
               count(DISTINCT ra.id) FILTER (WHERE EXISTS (SELECT 1 FROM rounds ro JOIN results r ON r.round_id = ro.id
                                                          WHERE ro.race_id = ra.id)) AS with_results
        FROM competitions co JOIN seasons s ON s.competition_id = co.id JOIN events e ON e.season_id = s.id
        LEFT JOIN races ra ON ra.event_id = e.id GROUP BY co.code""")
    last = data.q(conn, """
        SELECT DISTINCT ON (co.code) co.code AS competition, e.name, e.start_date
        FROM competitions co JOIN seasons s ON s.competition_id = co.id JOIN events e ON e.season_id = s.id
        JOIN races ra ON ra.event_id = e.id
        WHERE EXISTS (SELECT 1 FROM rounds ro JOIN results r ON r.round_id = ro.id WHERE ro.race_id = ra.id)
        ORDER BY co.code, e.start_date DESC""")
    nxt = data.q(conn, """
        SELECT DISTINCT ON (co.code) co.code AS competition, e.name, e.start_date
        FROM competitions co JOIN seasons s ON s.competition_id = co.id JOIN events e ON e.season_id = s.id
        WHERE e.start_date >= current_date AND e.status <> 'cancelled'
        ORDER BY co.code, e.start_date""")
    links = data.q(conn, """
        SELECT co.code AS competition, ml.exchange, count(*) AS links, count(ml.race_id) AS with_race,
               count(*) FILTER (WHERE NOT ml.closed) AS open, max(ml.synced_at) AS synced
        FROM market_links ml JOIN competitions co ON co.id = ml.competition_id GROUP BY 1, 2 ORDER BY 1, 2""")
    runs = data.q(conn, """
        SELECT co.code AS competition, mr.kind, (mr.params ? 'replay_batch') AS replay, count(*) AS n,
               count(DISTINCT mr.params->>'event_key') AS events, max(mr.created_at) AS at, max(mr.id) AS last_id
        FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id GROUP BY 1, 2, 3""")
    paper = data.q(conn, """
        SELECT co.code AS competition, u.username,
               count(DISTINCT p.event_key) AS races,
               count(*) FILTER (WHERE p.outcome IS NOT NULL) AS settled_positions,
               sum(p.cash + p.yes_shares * p.outcome::int + p.no_shares * (1 - p.outcome::int))
                   FILTER (WHERE p.outcome IS NOT NULL) AS pnl
        FROM paper_positions p JOIN users u ON u.id = p.user_id
        JOIN (SELECT DISTINCT ON (e.source_key) e.source_key, s.competition_id FROM events e
              JOIN seasons s ON s.id = e.season_id ORDER BY e.source_key, e.id) e ON e.source_key = p.event_key
        JOIN competitions co ON co.id = e.competition_id
        LEFT JOIN (SELECT DISTINCT user_id, market_key FROM strategy_signals          -- the debug mode, read once
                   WHERE strategy = 'buy_all' OR detail->>'mode' = 'buy_all') x
               ON x.user_id = p.user_id AND x.market_key = p.market_key
        WHERE u.prefs ? 'strategy_profile' AND p.venue <> 'private' AND x.user_id IS NULL
        GROUP BY 1, 2 ORDER BY 1, 2""")
    from racinglines import exchanges as EX
    from racinglines.markets.venues import VENUES
    names = {v.code: v.name for v in VENUES}
    for code in EX.CODES:
        names.setdefault(code, EX.load(code)["exchange"]["name"])

    out = []
    for schema in sorted(map(SP.load, SP.SPORT_CODES), key=lambda s: s["sport"]["display_order"]):
        comp, sport = schema["competition"]["code"], schema["sport"]["code"]
        if comp in hide:                                 # a sport this user doesn't see (web/audience.py)
            continue
        family = schema["sport"].get("model_family", "none")
        modeled = family != "none"
        row = dict(sport=sport, competition=comp, modeled=modeled, family=family,
                   name=schema["competition"].get("display_name", schema["sport"]["name"]))

        r = races[races["competition"] == comp]
        r = r.iloc[0] if len(r) else None
        lr = last[last["competition"] == comp]
        nx = nxt[nxt["competition"] == comp]
        if r is None or not _i(r["events"]):
            row["races"] = dict(state="none", text="no events stored", short="0",
                                sub="no results source" if not modeled else "results not loaded")
        else:
            n, res = _i(r["races"]), _i(r["with_results"])
            span = f"{_i(r['first_year'])}" + (f"–{_i(r['last_year'])}" if _i(r["last_year"]) != _i(r["first_year"]) else "")
            sub = [f"last result: {lr.iloc[0]['name']} ({lr.iloc[0]['start_date']:%d %b %Y})" if len(lr) else "no results yet"]
            if len(nx):
                sub.append(f"next: {nx.iloc[0]['name']} ({nx.iloc[0]['start_date']:%d %b %Y})")
            row["races"] = dict(state="ok" if res else "partial", short=f"{res:,}/{n:,}",
                                text=f"{res:,} of {n:,} races with results · {span}", sub=" · ".join(sub))

        lk = links[(links["competition"] == comp) & ~links["exchange"].isin(hide_venues)]
        if not len(lk):
            row["markets"] = dict(state="none", text="no exchange markets linked", short="0", sub="", exchanges=[])
        else:
            ex = [dict(exchange=x["exchange"], name=names.get(x["exchange"], x["exchange"]), links=_i(x["links"]),
                       with_race=_i(x["with_race"]), open=_i(x["open"]), synced=x["synced"]) for x in lk.to_dict("records")]
            synced = max((x["synced"] for x in ex if x["synced"] is not None and not pd.isna(x["synced"])), default=None)
            row["markets"] = dict(state="ok" if any(x["open"] for x in ex) else "partial",
                                  short=f"{sum(x['links'] for x in ex):,}",
                                  text=" · ".join(f"{x['name']} {x['links']:,}" for x in ex),
                                  sub=(f"{sum(x['open'] for x in ex):,} open · "
                                       f"{sum(x['with_race'] for x in ex):,} tied to a race"
                                       + (f" · synced {pd.Timestamp(synced):%d %b %H:%M} UTC" if synced is not None else "")),
                                  exchanges=ex)

        g = runs[runs["competition"] == comp]

        def kind(k, replay=None):
            x = g[g["kind"] == k]
            if replay is not None:
                x = x[x["replay"] == replay]
            return None if not len(x) else dict(n=_i(x["n"].sum()), events=_i(x["events"].sum()), at=x["at"].max(),
                                                last_id=_i(x["last_id"].max()))
        fc, asof, season = kind("forecast"), kind("diagnostic", True), kind("season_asof")
        if not modeled:
            row["model"] = dict(state="na", text="no model", sub="by design: tape-only sport")
        else:
            parts, sub = [], [f"model: {family}"]
            if fc:
                parts.append(f"live forecast run #{fc['last_id']}")
                sub.append(f"latest {pd.Timestamp(fc['at']):%d %b %Y}")
            if asof:
                parts.append(f"as-of prices for {asof['events']:,} races")
                sub.append(f"replay saves, latest {pd.Timestamp(asof['at']):%d %b %Y}")
            if season:
                parts.append(f"{season['n']:,} season forecasts")
            if not parts:
                parts.append("model built, no run stored")
                sub.append("no live price yet")
            elif not fc:
                sub.append("no live forecast run: no fair price on upcoming races")
            row["model"] = dict(state="ok" if fc else ("partial" if asof or season else "none"),
                                text=" · ".join(parts), sub=" · ".join(sub))

        bt, sw = kind("backtest"), kind("sweep")
        grid = _grid(sport)
        parts, sub = [], []
        if bt:
            parts.append(f"{bt['n']:,} backtest runs")
            sub.append(f"latest #{bt['last_id']} {pd.Timestamp(bt['at']):%d %b %Y}")
        if grid:
            parts.append(f"replay grid: {grid['runs']} runs ({', '.join(grid['seasons'])})")
            sub.append("ranked" + (f" {grid['at']:%d %b %H:%M} UTC" if grid["at"] is not None else "") if grid["ranked"]
                       else "not ranked (no grid.md)")
        if not modeled:
            parts, short = ["—"], "—"
        else:
            short = f"{bt['n']:,}" if bt else (f"{grid['runs']}" if grid else "0")
            if not parts:
                parts.append("no backtest stored")
        row["backtests"] = dict(state="ok" if bt or (grid and grid["ranked"]) else ("partial" if grid else
                                                                                   ("none" if modeled else "na")),
                                short=short, text=" · ".join(parts), sub=" · ".join(sub),
                                note="indicative, in-sample" if grid and sport != "f1" else "")
        if not modeled:
            row["sweeps"] = dict(state="na", short="—", text="no model", sub="")
        elif sw:
            row["sweeps"] = dict(state="ok", short=f"{sw['n']:,}", text=f"{sw['n']:,} saved strategy sweeps",
                                 sub=f"latest #{sw['last_id']} {pd.Timestamp(sw['at']):%d %b %Y}")
        else:
            row["sweeps"] = dict(state="none", short="0", text="no strategy sweep saved",
                                 sub="run an Edge Finder sweep to save per-weekend strategy results")

        pp = paper[paper["competition"] == comp]
        if not len(pp):
            row["paper"] = dict(state="none" if modeled else "na", short="—" if not modeled else "0",
                                text="no paper record" if modeled else "—", sub="")
        else:
            account_pnls = []
            invalid_pnl = False
            total = 0.0
            for u in pp.to_dict("records"):
                if pd.isna(u["pnl"]):
                    if _i(u["settled_positions"]):
                        pnl_text = "P&L unavailable"
                        invalid_pnl = True
                    else:
                        pnl_text = "+0"
                else:
                    value = float(u["pnl"])
                    total += value
                    pnl_text = f"{value:+,.0f}"
                account_pnls.append(f"{u['username']} {_i(u['races'])} races {pnl_text}")
            row["paper"] = dict(state="partial" if invalid_pnl else "ok",
                                short="—" if invalid_pnl else f"{total:+,.0f}",
                                text=" · ".join(account_pnls),
                                sub="paper (settled P&L, $)")
        out.append(row)
    return out
