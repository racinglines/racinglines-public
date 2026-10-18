"""
Demo paper portfolios for the result-only sports (NASCAR Cup, MotoGP): each past race's taker replay
(pipelines/position_replay.py) stored as backfilled paper positions and signals of a demo account, so the web app
shows them beside the F1 demo history (pipelines/demo_history.py). On by default for demo setups: set
RACINGLINES_SPORT_PAPER=0 to hide them, or 1 to force the explicit opt-in in a non-demo environment.

    racinglines strategy demo-history nascar --pick data/runs/replay-grid/nascar             # print the selection, write nothing
    racinglines strategy demo-history nascar --grid data/runs/replay-grid/nascar --backup FILE [--users taker] [--events latest]
    racinglines strategy demo-history nascar --reset --backup FILE [--users taker]            # this sport's rows only

The strategy per market kind is the settings grid's best (the overnight run's read-only grid, scripts/vm/replay_grid.py,
runs named <year>-e<edge>-v<volume> under data/runs/replay-grid/<sport>/, each with kalshi/trades.csv):

  per kind (--book kinds, the default): for each market kind, the grid setting (min edge, 24 h volume floor) whose
  worse season's net P&L on that kind (after Kalshi's taker fee) is highest; a kind whose best setting still lost money
  in its worse season is not traded (a season with no trade at any setting, as MotoGP 2025 on Kalshi, is left out of
  the test rather than counted as $0). The blend (--book blend): the one setting with the best worse season over every
  kind together (grid.md's top row at the default simulations).

Both are chosen on the same seasons they are then replayed on: IN-SAMPLE, a demo, not a forecast. Every row is labelled
so (profile name, detail.book, detail.in_sample). Only the taker's `update` mode is ever stored: never the debug
buy_all mode (markets/strategies/buy_everything.py), which the replay is run with switched off.

Rows: strategy_signals with detail.backfill, detail.sport, detail.venue, detail.book, detail.setting; paper_positions on
the venue ('kalshi' by default) with candidate_id NULL and the race's event_key (the sport's own event keys, which never
collide with F1's). Positions carry the venue's taker fee in their cash, so their P&L is after fees. A rebuild replaces
the account's rows for that sport, venue and race; --reset removes them all for the sport and venue, nothing else.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import text

SWITCH = "RACINGLINES_SPORT_PAPER"
SPORTS = ("nascar", "motogp")
BOOKS = ("kinds", "blend", "best")
MODE = "update"                               # the only taker mode stored
BUY_ALL = "buy_all"                           # never stored, never read by the app
NAME = re.compile(r"^(?P<year>\d{4})-e(?P<edge>[\d.]+)-v(?P<vol>[\d.]+)$")     # default simulations only

# The account's event keys that hold this module's rows, for the app's queries (bind :u). A buy_all row never counts.
SPORT_KEYS = ("SELECT event_key FROM strategy_signals WHERE user_id = :u AND detail ? 'sport' "
              "AND strategy <> 'buy_all' AND coalesce(detail->>'mode', 'update') <> 'buy_all'")


def enabled(flag=None):
    def on(v):
        if v is None:
            return True
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    if flag is not None:
        return on(flag)
    value = os.environ.get(SWITCH)
    return on(value)


def event_sources():
    """The events.source values whose source_key can be a paper position's event_key: F1's, then these sports'."""
    from racinglines.pipelines import position_replay as P
    return ["f1timing"] + [P.spec(s)["source"] for s in SPORTS]


def sport_name(sport):
    from racinglines import sports
    s = sports.load(sport)
    return s["competition"].get("display_name") or s["sport"]["name"]


# --- the selection from the grid ------------------------------------------------------------------------

def grid_nets(folder, venue="kalshi"):
    """Net P&L after the venue's taker fee, per (edge, volume floor, season, kind), from each grid run's trades.csv."""
    from racinglines.pipelines import position_replay as P
    rows = []
    for f in sorted(Path(folder).glob(f"*/{venue}/summary.json")):
        m = NAME.match(f.parent.parent.name)
        if not m:
            continue
        kinds = json.loads(f.read_text())["params"].get("kinds") or []
        key = dict(edge=float(m["edge"]), volume=float(m["vol"]), season=int(m["year"]))
        tr = f.parent / "trades.csv"
        try:
            t = pd.read_csv(tr) if tr.is_file() and tr.stat().st_size else pd.DataFrame()
        except pd.errors.EmptyDataError:
            t = pd.DataFrame()
        for k in kinds:
            tk = t[t["kind"] == k] if len(t) else t
            pnl = float(tk["pnl"].fillna(0).sum()) if len(tk) else 0.0
            fees = P.venue_fees(venue, tk) if len(tk) else 0.0
            rows.append(dict(key, kind=k, trades=len(tk), pnl=pnl, fees=fees, net=pnl - fees))
    return pd.DataFrame(rows, columns=["edge", "volume", "season", "kind", "trades", "pnl", "fees", "net"])


def _best(g, seasons):
    """The setting with the best worse season (every season run), ties to the higher edge then the higher floor."""
    wide = g.pivot_table(index=["edge", "volume"], columns="season", values="net", aggfunc="sum")
    wide = wide.dropna(subset=[s for s in seasons if s in wide.columns])
    if not len(wide) or any(s not in wide.columns for s in seasons):
        return None
    wide["worse"] = wide[list(seasons)].min(axis=1)
    wide = wide.reset_index().sort_values(["worse", "edge", "volume"], ascending=False)
    b = wide.iloc[0]
    return dict(edge=float(b["edge"]), volume=float(b["volume"]), worse=float(b["worse"]),
                seasons={int(s): float(b[s]) for s in seasons})


def pick(folder, venue="kalshi"):
    """dict(seasons, kinds={kind: best or None}, traded={kind: best} (worse season > 0), blend=best over all kinds)."""
    nets = grid_nets(folder, venue)
    if not len(nets):
        raise ValueError(f"no grid runs (<year>-e<edge>-v<volume>/{venue}/summary.json) under {folder}")
    seasons = sorted(nets["season"].unique().tolist())
    # a season in which a kind had nothing to trade at any setting (MotoGP 2025: no Kalshi race markets yet) is no
    # evidence either way: it is left out of that kind's worse-season test instead of counting as a $0 season
    traded = lambda g: sorted(g.loc[g["trades"] > 0, "season"].unique().tolist())      # noqa: E731
    kinds = {k: (_best(g, traded(g)) if traded(g) else None) for k, g in nets.groupby("kind", sort=False)}
    all_ = nets.groupby(["edge", "volume", "season"], as_index=False)[["net", "trades"]].sum()
    blend = _best(all_, traded(all_)) if traded(all_) else None
    # best effort (--book best): per kind the setting with the best total over its traded seasons; the kinds whose
    # total is positive, else (nothing positive) the one setting with the best total over every kind, win or lose, so
    # a sport with markets always has a record to show (still in-sample, and its losses shown as losses)
    def top(g):
        ss = traded(g)
        if not ss:
            return None
        w = g[g["season"].isin(ss)].pivot_table(index=["edge", "volume"], columns="season", values="net", aggfunc="sum")
        w = w.fillna(0.0)
        w["total"] = w[ss].sum(axis=1)
        b = w.reset_index().sort_values(["total", "edge", "volume"], ascending=False).iloc[0]
        return dict(edge=float(b["edge"]), volume=float(b["volume"]), total=float(b["total"]),
                    worse=float(min(b[x] for x in ss)), seasons={int(x): float(b[x]) for x in ss})
    best = {k: top(g) for k, g in nets.groupby("kind", sort=False)}
    best_traded = {k: b for k, b in best.items() if b and b["total"] > 0}
    return dict(seasons=seasons, kinds=kinds, blend=blend, venue=venue,
                traded={k: b for k, b in kinds.items() if b and b["worse"] > 0},
                best=best, best_traded=best_traded, best_all=None if best_traded else top(all_))


def selection_md(sport, sel):
    """The selection as Markdown: the rule, each kind's best setting and seasons, the blend, and the caveat."""
    ss = sel["seasons"]
    lines = [f"# {sport_name(sport)} demo paper portfolio: strategy selection ({sel['venue']})", "",
             "Rule: per market kind, the grid setting (min edge, 24 h volume floor) with the best worse-season net P&L "
             "after the taker fee; a kind whose best setting lost money in its worse season is not traded (a season with "
             "no markets for the kind is left out). The blend is "
             "the one setting with the best worse season over every kind together.", "",
             f"**In-sample:** chosen on {', '.join(map(str, ss))} and replayed on the same seasons. A demo of what the "
             "strategy would have done, not evidence of an edge. The debug buy_all mode is never included.", "",
             "| book | kind | min edge | volume floor | " + " | ".join(f"{s} net" for s in ss) + " | worse season | traded |",
             "|---|---|---:|---:|" + "---:|" * len(ss) + "---:|---|"]

    def row(book, kind, b, traded):
        if b is None:
            return f"| {book} | {kind} | | | " + " | ".join("not run" for _ in ss) + " | | no |"
        return (f"| {book} | {kind} | {b['edge']:g} | ${b['volume']:,.0f} | "
                + " | ".join(f"{b['seasons'][s]:+,.0f}" if s in b["seasons"] else "no markets" for s in ss) + f" | {b['worse']:+,.0f} | {'yes' if traded else 'no'} |")
    for k, b in sel["kinds"].items():
        lines.append(row("per kind", k, b, k in sel["traded"]))
    lines.append(row("blend", "all kinds", sel["blend"], bool(sel["blend"] and sel["blend"]["worse"] > 0)))
    tot = {s: sum(b["seasons"].get(s, 0.0) for b in sel["traded"].values()) for s in ss}
    lines += ["", "Per-kind book, the traded kinds together: " + ", ".join(f"{s} {tot[s]:+,.0f}" for s in ss) + "."]
    if "best" in sel:
        lines += ["", "Best effort (--book best): per kind, the setting with the best total; kinds with a positive total "
                  "are traded, else every kind at the one setting with the best total, win or lose.", ""]
        for k, b in sel["best"].items():
            if b:
                lines.append(f"- {k}: edge {b['edge']:g}, floor ${b['volume']:,.0f}, total {b['total']:+,.0f} "
                             f"({'traded' if k in sel['best_traded'] else 'not traded'})")
        if sel.get("best_all"):
            b = sel["best_all"]
            lines.append(f"- nothing positive: every kind at edge {b['edge']:g}, floor ${b['volume']:,.0f}, "
                         f"total {b['total']:+,.0f} (shown as a loss if it lost)")
    return "\n".join(lines) + "\n"


def settings_for(sel, book="kinds"):
    """{(edge, volume): [kinds]} to replay: each traded kind at its own best (kinds), or every kind at the blend's."""
    if book not in BOOKS:
        raise ValueError(f"book must be one of {BOOKS}")
    if book == "best":
        if sel.get("best_all"):
            b = sel["best_all"]
            return {(b["edge"], b["volume"]): sorted(k for k, x in sel["best"].items() if x)}
        out = {}
        for k, b in sel.get("best_traded", {}).items():
            out.setdefault((b["edge"], b["volume"]), []).append(k)
        return out
    if book == "blend":
        b = sel["blend"]
        return {(b["edge"], b["volume"]): sorted(sel["kinds"])} if b and b["worse"] > 0 else {}
    out = {}
    for k, b in sel["traded"].items():
        out.setdefault((b["edge"], b["volume"]), []).append(k)
    return out


# --- the rows -------------------------------------------------------------------------------------------

def _user_ids(conn, usernames):
    return dict(conn.execute(text("SELECT username, id FROM users WHERE username = ANY(:u)"), dict(u=list(usernames))).all())


def _delete(conn, uids, sport, venue, event_keys=None):
    """This module's rows for these users, sport and venue (and only these event keys, when given)."""
    ev = " AND event_key = ANY(:k)" if event_keys is not None else ""
    p = dict(i=list(uids), s=sport, v=venue, k=list(event_keys or []))
    keys = conn.execute(text(f"""SELECT DISTINCT event_key FROM strategy_signals WHERE user_id = ANY(:i)
                                 AND detail->>'sport' = :s AND detail->>'backfill' = 'true' AND detail->>'mode' = 'update'
                                 AND coalesce(detail->>'venue', 'polymarket') = :v{ev}"""), p).scalars().all()
    n = conn.execute(text(f"""DELETE FROM strategy_signals WHERE user_id = ANY(:i) AND detail->>'sport' = :s
                              AND detail->>'backfill' = 'true' AND detail->>'mode' = 'update' AND coalesce(detail->>'venue', 'polymarket') = :v{ev}"""),
                     p).rowcount
    keys = sorted(set(keys) | set(event_keys or []))
    m = conn.execute(text("""DELETE FROM paper_positions WHERE user_id = ANY(:i) AND event_key = ANY(:k) AND venue = :v
                             AND candidate_id IS NULL"""), dict(i=list(uids), k=keys, v=venue)).rowcount
    return n, m


def reset(engine, usernames, sport, venue="kalshi"):
    """Delete this sport's demo rows on `venue` for these accounts. -> (signals, positions) deleted."""
    with engine.begin() as c:
        return _delete(c, _user_ids(c, usernames).values(), sport, venue)


def race_rows(markets_by_setting, venue, sport, book):
    """One race's signal and position rows from its markets per setting: the taker's `update` trades
    (signals.taker_signals), each position's cash net of the venue's taker fee on its trades."""
    from racinglines.markets.strategies import taker_weekend as RB
    from racinglines.pipelines import position_replay as P
    from racinglines.pipelines import signals as SG
    sigs, pos = [], []
    for (edge, vol), markets in markets_by_setting.items():
        p = RB.TakerParams(min_edge=edge, mode=MODE)
        s, ps = SG.taker_signals(markets, p)
        fee = {}
        for x in s:
            f = P.venue_fees(venue, pd.DataFrame(dict(mid=[x["price"]], shares=[x["shares"]])))
            fee[x["market_key"]] = fee.get(x["market_key"], 0.0) + f
            x["detail"] = dict(x.get("detail") or {}, sport=sport, venue=venue, book=book, mode=MODE, in_sample=True,
                               setting=dict(min_edge=edge, min_volume_24h=vol), fee=f)
        for q in ps:
            q["cash"] = q["cash"] - fee.get(q["market_key"], 0.0)
        sigs += s
        pos += ps
    return sigs, pos


def backfill(engine, sport, settings, usernames=("taker",), seasons=None, venue="kalshi", events=None, book="kinds",
             echo=print, stage_mode="weekend"):
    """Replay each season (one run per season and setting, as the grid ran them, so the trades match its trades.csv)
    and store every race's trades as the accounts' demo paper rows. settings: {(edge, volume): [kinds]}. stage_mode:
    position_replay.stage_list's ("race_day" adds the [replay] race_day_stages).
    Returns [(user, event_key, race, signals, P&L)]."""
    from racinglines.markets.strategies import taker_weekend as RB
    from racinglines.pipelines import position_replay as P
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import signals as SG
    if sport not in SPORTS:
        raise ValueError(f"sport must be one of {SPORTS}")
    with engine.connect() as c:
        uids = _user_ids(c, usernames)
        seasons = seasons or sorted(P.races(c, P.spec(sport))["season"].unique().tolist())
    for u in set(usernames) - set(uids):
        echo(f"no user {u!r}: skipped")
    prof = dict(name=f"{sport_name(sport)} taker, {dict(kinds='best per kind', blend='blend', best='best effort')[book]} (demo, in-sample)",
                strategy=MODE, candidate_id=None)
    report, data = [], None
    for year in seasons:
        races = {}
        for (edge, vol), kinds in settings.items():
            def keep(r, markets, key=(edge, vol)):
                races.setdefault(r.event_key, dict(r=r, by={}))["by"][key] = markets
            out = P.run(engine, sport, [year], venue=venue, taker=RB.TakerParams(min_edge=edge), min_volume_24h=vol,
                        kinds=kinds, data=data, events=events, echo=lambda *a: None, buy_all=False, on_race=keep,
                        stage_mode=stage_mode)
            data = out.pop("data")
        for key, x in races.items():
            r = x["r"]
            sigs, pos = race_rows(x["by"], venue, sport, book)
            if not sigs:
                continue
            sp = P.spec(sport)
            stages = [(lab, t, None, None) for lab, t in P.stage_times(r.start, dict(sp, stages=P.stage_list(sp, stage_mode)))]
            o = dict(profile=prof, event=dict(event_key=key), race_id=int(r.race_id), stages=stages, signals=sigs,
                     positions=pos, venue=venue)
            for username, uid in uids.items():
                with engine.begin() as c:
                    _delete(c, [uid], sport, venue, [key])
                    SG.store(c, uid, o, follow_rate=PF.follow_rate(username, MODE), history=True, venue=venue)
                    pnl = c.execute(text("""SELECT coalesce(sum(cash + yes_shares * outcome::int + no_shares * (1 - outcome::int)), 0)
                                            FROM paper_positions WHERE user_id = :u AND event_key = :e AND venue = :v
                                            AND candidate_id IS NULL AND outcome IS NOT NULL"""),
                                    dict(u=uid, e=key, v=venue)).scalar()
                report.append((username, key, r.name, len(sigs), float(pnl)))
                echo(f"{username} {sport} {key} {r.name[:28]:28s} {len(sigs):4d} signals  P&L {pnl:+8.2f}")
    return report
