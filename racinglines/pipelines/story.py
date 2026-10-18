"""
The demo accounts' story, from data: bankroll, track record, and (maker) each strategy decision with the
evidence the account had at that moment.

The maker re-checks its setup in the Lab's Edge Finder at a few decision points (before the first race,
mid-season, before the next season). The evidence is walk-forward: each candidate setup's P&L over only
the weekends raced so far, read from its saved season sweep (every weekend priced as of its own sessions).
Rules:
  - before any weekend has been raced, start on the defaults;
  - mid-season, switch to the setup with the most P&L so far when it beats the current one by more
    than SWITCH_MARGIN;
  - before a new season (a whole season's commitment), take every setup within NOISE_FLOOR of the P&L
    leader (differences smaller than that are simulation noise, per the params-4h search) and commit to
    the most consistent one (the Edge Finder's consistency: mean / s.d. of weekend P&L x sqrt(weekends)).

The Strategy page shows each decision with its evidence (story.account); demo_history replays the result.
"""

import numpy as np

from racinglines.pipelines import sweep_settings as SS

SWITCH_MARGIN = 100.0          # $: don't churn on noise
NOISE_FLOOR = 350.0            # $ per season: makers' simulation-noise floor (params-4h report, section 2)
MAKERS = ("maker", "maker_skew", "maker_widen", "maker_flat", "maker_all")
# the maker setups the account tested (settings that differ from the defaults, strategies)
POOL = [
    ({"variant": "baseline"}, MAKERS),
    ({"variant": "gridq+pretrain"}, MAKERS),
    ({"variant": "gbm"}, MAKERS),
    ({"variant": "gridq+pretrain", "max_disagree": 0.10}, ("maker", "maker_skew")),
    ({"variant": "gbm", "max_disagree": 0.07}, ("maker", "maker_skew")),
    ({"variant": "gbm", "max_disagree": 0.05, "size": 25}, ("maker",)),
]
# decision points: (label, season whose raced weekends are the evidence, rounds of it known)
DECISIONS = [("Before the first race of 2025", 2025, 0, "mid"), ("After 2025 round 8", 2025, 8, "mid"),
             ("After 2025 round 16", 2025, 16, "mid"), ("Before the 2026 season", 2025, 99, "season")]
START = ({"variant": "baseline"}, "maker")      # the defaults
# The taker's pool: the taker re-sweep's round 1, fixed before any result (four models x four minimum edges, the
# update and stage-aware takers), judged by the same rule with the takers' own noise floor (measured there).
TAKERS = ("update", "early")
TAKER_POOL = [({"variant": v, "min_edge": e} if (v, e) != ("baseline", 0.05) else {"variant": v}, TAKERS)
              for v in ("baseline", "gridq+pretrain", "gridq+pretrain+reset", "gbm") for e in (0.05, 0.08, 0.10, 0.15)]
TAKER_START = ({"variant": "baseline"}, "update")
TAKER_NOISE_FLOOR = 400.0      # $ per season: the update taker's seed spread (taker-resweep report, section 2)


def _weekly(conn, year, pool=None):
    """{(settings key, strategy): [(round, pnl), ...]} for every pool setup (default POOL) with a saved `year`
    sweep."""
    from sqlalchemy import text

    from racinglines.web import edge
    cfgs = edge.configs(conn, year, "f1", "polymarket")
    out = {}
    for settings, strategies in (POOL if pool is None else pool):
        st = SS.Settings.from_dict(settings)
        cfg = cfgs.get(st.key)
        if cfg is None:
            continue
        wk = conn.execute(text("SELECT metrics->'weekends' FROM model_runs WHERE id = :i"),
                          dict(i=cfg["run_id"])).scalar() or []
        for s in strategies:
            out[(st.key, s)] = [(int(w["round"]), float(w.get(f"{s}_pnl") or 0.0)) for w in wk]
    return out


def label(settings, strategy):
    st = SS.Settings.from_dict(settings)
    return f"{strategy} · {st.label()}"


def decisions(conn, taker=False):
    """[dict(label, known, table=[dict(setup, strategy, key, pnl, weekends, up, worst)], current, chosen,
    switched)] following the rule, with the evidence each decision saw. taker: the demo taker's pool, start and
    noise floor (same decision points and rule)."""
    pool, start, noise = (TAKER_POOL, TAKER_START, TAKER_NOISE_FLOOR) if taker else (POOL, START, NOISE_FLOOR)
    weekly = {2025: _weekly(conn, 2025, pool)}
    names = {(SS.Settings.from_dict(s).key, k): (s, k) for s, ks in pool for k in ks}
    cur = (SS.Settings.from_dict(start[0]).key, start[1])
    out = []
    for lab, year, known, kind in DECISIONS:
        table = []
        for key, rows in weekly[year].items():
            got = [p for r, p in rows if r <= known]
            if not got:
                continue
            settings, strategy = names[key]
            cum = np.cumsum(got)
            sd = float(np.std(got, ddof=1)) if len(got) > 1 else 0.0
            table.append(dict(key=key[0], setup=label(settings, strategy), settings=settings, strategy=strategy,
                              pnl=float(cum[-1]), weekends=len(got), up=sum(p > 0 for p in got), worst=min(got),
                              max_dd=float((cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]).min()),
                              consistency=float(np.mean(got) / sd * np.sqrt(len(got))) if sd > 0 else None))
        table.sort(key=lambda r: -r["pnl"])
        prev = cur
        if table and kind == "season":
            band = [r for r in table if r["pnl"] >= table[0]["pnl"] - noise and r["consistency"] is not None]
            best = max(band, key=lambda r: r["consistency"])
            cur = (best["key"], best["strategy"])
            for r in table:
                r["in_band"] = r in band
        elif table:
            best, now = table[0], next((r for r in table if (r["key"], r["strategy"]) == cur), None)
            if now is None or best["pnl"] > now["pnl"] + SWITCH_MARGIN:
                cur = (best["key"], best["strategy"])
        for r in table:
            r["current"] = (r["key"], r["strategy"]) == prev
            r["chosen"] = (r["key"], r["strategy"]) == cur
        settings, strategy = names[cur]
        shown = [r for i, r in enumerate(table) if i < 5 or r["chosen"] or r["current"]]
        out.append(dict(label=lab, year=year, known=known, kind=kind, shown=shown, candidates=len(table),
                        weekends=max((r["weekends"] for r in table), default=0),
                        table=table, chosen=label(settings, strategy), chosen_settings=settings,
                        chosen_strategy=strategy, switched=cur != prev))
    return out


# ---------------------------------------------------------------------------
# One account's story: bankroll, track record, phases, and what drove them
# ---------------------------------------------------------------------------

def track_record(conn, uid, venue="polymarket", sport=None, sports=False, hide=()):
    """Every weekend with signals or positions: event, strategy run, trades taken (taker) / fills (maker),
    positions, paper P&L (settled, else marked to the market), backtest replay or live. venue: 'polymarket'
    (the default: the strategy's own record), 'private', any other exchange code ('kalshi', 'og': the profile's replay
    on that exchange's tape, e.g. `f1 demo-history --venue kalshi`) or 'all'. A private-book event
    (pipelines/live_dh.py) has positions but no signals: it joins the history on the day the book ran, as its
    own "Private book" run. sports (RACINGLINES_SPORT_PAPER=1, pipelines/sport_paper.py): the NASCAR / MotoGP demo
    replays join the record on their race dates, whatever their venue, flagged `demo` (never a buy_all row). hide: sport
    codes left out (the web app's [sport] audience, web/audience.py)."""
    import pandas as pd
    from sqlalchemy import text

    from racinglines.pipelines import sport_paper as SP
    demo = " AND detail ? 'sport' AND strategy <> 'buy_all' AND coalesce(detail->>'mode', 'update') <> 'buy_all'"
    q = text("""
        WITH s AS (SELECT event_key, min(profile) AS profile, min(strategy) AS strategy, count(*) AS signals,""" + (
                          " bool_or(detail ? 'sport') AS demo," if sports else "") + """
                          count(*) FILTER (WHERE action IN ('buy', 'sell') AND coalesce(detail->>'followed', 'true') = 'true') AS taken,
                          count(*) FILTER (WHERE action = 'fill') AS fills,
                          bool_or(detail->>'backfill' = 'true') AS backfill
                   FROM strategy_signals WHERE user_id = :u
                     AND (:v IN ('private', 'all') OR coalesce(detail->>'venue', 'polymarket') = :v""" + (
                          " OR (TRUE" + demo + ")" if sports else "") + """)
                   GROUP BY event_key),
             p AS (SELECT event_key, count(*) FILTER (WHERE abs(yes_shares) + abs(no_shares) > 1e-9) AS positions,
                          bool_and(outcome IS NOT NULL OR abs(yes_shares) + abs(no_shares) + abs(cash) < 1e-9) AS settled,
                          sum(cash + yes_shares * coalesce(outcome::int, mark) + no_shares * (1 - coalesce(outcome::int, mark))) AS pnl,
                          bool_and(venue = 'private') AS private, max(updated_at) AS updated
                   FROM paper_positions WHERE user_id = :u AND (:v = 'all' OR venue = :v""" + (
                          " OR event_key IN (" + SP.SPORT_KEYS + ")" if sports else "") + """) GROUP BY event_key)
        SELECT event_key, s.profile, s.strategy, coalesce(s.signals, 0) AS signals, coalesce(s.taken, 0) AS taken,
               coalesce(s.fills, 0) AS fills, s.backfill, coalesce(p.private, false) AS private,""" + (
               " coalesce(s.demo, false) AS demo," if sports else "") + """
               coalesce(p.positions, 0) AS positions, coalesce(p.settled, true) AS settled, coalesce(p.pnl, 0) AS pnl,
               coalesce(ra.format->>'event_name', e.name, le.title) AS event_name,
               CASE WHEN p.private THEN 'private' ELSE sp.code END AS sport,
               coalesce(e.start_date, CASE WHEN p.private THEN coalesce((le.opened_at AT TIME ZONE 'UTC')::date,
                                                                        (p.updated AT TIME ZONE 'UTC')::date) END) AS start_date
        FROM s FULL JOIN p USING (event_key)
        LEFT JOIN events e ON e.source_key = event_key
        LEFT JOIN seasons se ON se.id = e.season_id
        LEFT JOIN competitions co ON co.id = se.competition_id
        LEFT JOIN sports sp ON sp.id = co.sport_id
        LEFT JOIN races ra ON ra.event_id = e.id
        LEFT JOIN (SELECT DISTINCT ON (event_key) event_key, title, opened_at FROM live_events
                   ORDER BY event_key, opened_at) le USING (event_key)
        WHERE (:v <> 'private' AND s.event_key IS NOT NULL) OR (:v <> 'polymarket' AND p.private) ORDER BY event_key""")
    rows = [dict(r) for r in conn.execute(q, dict(u=uid, v=venue)).mappings()]
    rows = [r for r in rows if r.get("sport") not in hide]
    if sport:
        sport = sport.lower()
        rows = [r for r in rows if (r.get("sport") or "").lower() == sport]
    from racinglines.pipelines.live import event_name
    for r in rows:
        r["pnl"] = float(r["pnl"] or 0.0)
        if r["private"] and r["profile"] is None:
            r.update(profile="Private book", strategy="private", fills=r["positions"], taken=r["positions"],
                     event_name=r["event_name"] or event_name(r["event_key"]))
        r["date"] = pd.Timestamp(r["start_date"]) if r["start_date"] is not None else None
    if sports:                                          # F1 and the other sports interleaved by date
        rows.sort(key=lambda r: (r["date"] is None, r["date"] or pd.Timestamp.min, r["event_key"]))
    return rows


TAKER_PREFIXES = ("update", "hold", "last", "early")      # the taker modes (weekend_sweep.TAKER_MODES) as strategy names


def mix(record):
    """The record split by sport and by kind of strategy (maker: quotes both sides and earns the spread; taker: takes
    the model's edge at the market's price), for the Strategy page: [dict(sport, family, weekends, up, pnl, demo)],
    biggest P&L first. A Pro account runs both: the F1 maker on Polymarket, the taker on the other sports."""
    out = {}
    for r in record:
        fam = "private book" if r.get("private") else ("taker" if (r.get("strategy") or "").startswith(TAKER_PREFIXES)
                                                        else "maker")
        k = out.setdefault((r.get("sport") or "", fam), dict(sport=r.get("sport") or "", family=fam, weekends=0, up=0,
                                                           pnl=0.0, demo=False))
        k["weekends"] += 1
        k["up"] += r["pnl"] > 0
        k["pnl"] += r["pnl"]
        k["demo"] = k["demo"] or bool(r.get("demo"))
    return sorted(out.values(), key=lambda k: -abs(k["pnl"]))


def deployed(conn, uid):
    """{event_key: peak capital in use during the weekend}: taker, the cost of open positions it took;
    maker, the worst-case loss of its inventory (the maker replay's capital measure)."""
    from sqlalchemy import text
    rows = conn.execute(text("""SELECT event_key, market_key, action, side, shares, limit_price,
                                       detail->>'followed' AS followed, detail->>'maker_side' AS mside
                                FROM strategy_signals WHERE user_id = :u AND action IN ('buy', 'sell', 'fill')
                                ORDER BY event_key, signal_ts, id"""), dict(u=uid)).all()
    out, state = {}, {}
    for ev, mk, action, side, shares, px, followed, mside in rows:
        st = state.setdefault(ev, dict(out=0.0, cash={}, inv={}, peak=0.0))
        if action in ("buy", "sell"):
            if followed == "false":
                continue
            st["out"] += (1 if action == "buy" else -1) * shares * px
            st["peak"] = max(st["peak"], st["out"])
        else:
            sg = 1 if mside == "buy" else -1
            st["cash"][mk] = st["cash"].get(mk, 0.0) - sg * shares * px
            st["inv"][mk] = st["inv"].get(mk, 0.0) + sg * shares
            used = -sum(min(0.0, min(c, c + st["inv"][k])) for k, c in st["cash"].items())
            st["peak"] = max(st["peak"], used)
        out[ev] = st["peak"]
    return out


def _season_pnl(conn, settings, strategy):
    """{event_key: P&L} of one setup from its saved season sweeps (every season that has one)."""
    from sqlalchemy import text

    from racinglines.web import edge
    key, out = SS.Settings.from_dict(settings).key, {}
    for year in (2025, 2026):
        cfg = edge.configs(conn, year).get(key)
        if cfg is None:
            continue
        for w in conn.execute(text("SELECT metrics->'weekends' FROM model_runs WHERE id = :i"),
                              dict(i=cfg["run_id"])).scalar() or []:
            out[w["event_key"]] = float(w.get(f"{strategy}_pnl") or 0.0)
    return out


def taker_detail(conn, uid, profile):
    """What the taker was offered and took, by heat; and every recommendation taken, for comparison."""
    from sqlalchemy import text
    firsts = conn.execute(text("""
        SELECT DISTINCT ON (market_key) market_key, event_key, heat, detail->>'followed' = 'true' AS taken
        FROM strategy_signals WHERE user_id = :u AND action = 'buy' ORDER BY market_key, signal_ts, id"""),
        dict(u=uid)).all()
    pnl = dict(conn.execute(text("""SELECT market_key, cash + yes_shares * coalesce(outcome::int, mark)
                                          + no_shares * (1 - coalesce(outcome::int, mark))
                                    FROM paper_positions WHERE user_id = :u"""), dict(u=uid)).all())
    by = {}
    for mk, ev, h, taken in firsts:
        r = by.setdefault(h or 1, dict(heat=h or 1, offered=0, taken=0, pnl=0.0))
        r["offered"] += 1
        if taken:
            r["taken"] += 1
            r["pnl"] += float(pnl.get(mk) or 0.0)
    all_a = _season_pnl(conn, profile["settings"], profile["strategy"]) if profile else {}
    return dict(by_heat=[by[h] for h in sorted(by)], all_pnl=all_a)


def phases(record):
    """Consecutive weekends run with the same strategy: [dict(profile, first, last, weekends, pnl, up)]."""
    out = []
    for r in record:
        if out and out[-1]["profile"] == r["profile"]:
            p = out[-1]
        else:
            p = dict(profile=r["profile"], first=r, weekends=0, pnl=0.0, up=0)
            out.append(p)
        p["last"], p["weekends"], p["pnl"], p["up"] = r, p["weekends"] + 1, p["pnl"] + r["pnl"], p["up"] + (r["pnl"] > 0)
    return out


SEASON_WEEKENDS = 24            # Sharpe per season, as in the params-4h report: mean / s.d. of weekend P&L x sqrt(24)


def account(conn, uid, profile, maker, markers=True, venue="polymarket", sport=None, sports=False, hide=()):
    """Everything the Strategy page tells about one account: KPIs (with the worst drawdown and the Sharpe
    ratio over the full history), bankroll curve (markers: dashed lines at strategy switches), phases,
    track record (with running balance), and the maker's decisions or the taker's detail. venue: as in
    track_record (the Strategy page: Polymarket only). sports: as in track_record (RACINGLINES_SPORT_PAPER=1)."""
    from racinglines.web.viz import line_chart
    record = track_record(conn, uid, venue, sport=sport, sports=sports, hide=hide)
    bank = (profile or {}).get("bankroll") or {}
    start = float(bank.get("start") or 0.0)
    peak = deployed(conn, uid)
    cum = hi = 0.0
    max_dd = 0.0
    for r in record:
        cum += r["pnl"]
        hi = max(hi, cum)
        max_dd = min(max_dd, cum - hi)
        r["cum"], r["balance"], r["deployed"] = cum, start + cum, peak.get(r["event_key"], 0.0)
    ph = phases([r for r in record if not r.get("demo")]) if sports else phases(record)     # F1's strategy story
    dated = [r for r in record if r["date"] is not None]
    series = {"pnl": [(r["date"], r["cum"]) for r in dated]}             # cumulative P&L from $0, not a bankroll
    labels = {"pnl": "cumulative paper P&L"}
    detail = None
    if not maker and profile and venue != "private":
        detail = taker_detail(conn, uid, profile)
        run, pts = 0.0, []
        for r in dated:
            run += detail["all_pnl"].get(r["event_key"], 0.0)
            pts.append((r["date"], run))
        if detail["all_pnl"]:
            series["all"] = pts
            labels["all"] = "if it had taken every recommendation"
            detail["all_total"] = sum(detail["all_pnl"].get(r["event_key"], 0.0) for r in record)
    marks = [(p["first"]["date"], p["profile"].split(" ")[0]) for p in ph[1:] if p["first"]["date"] is not None] \
        if markers else []
    chart = line_chart(series, labels, markers=marks)
    pnls = np.array([r["pnl"] for r in record])
    sharpe = float(pnls.mean() / pnls.std(ddof=1) * np.sqrt(SEASON_WEEKENDS)) if len(pnls) > 1 and pnls.std(ddof=1) > 0 else None
    seasons = {}
    for r in record:
        y = seasons.setdefault(r["event_key"][:4], dict(year=r["event_key"][:4], pnl=0.0, weekends=0, up=0, rows=[]))
        y["pnl"] += r["pnl"]
        y["weekends"] += 1
        y["up"] += r["pnl"] > 0
        y["rows"].append(r)
    kpis = dict(start=start or None, since=bank.get("since"), balance=start + cum if start else None, total=cum,
                ret=cum / start if start else None, max_dd=max_dd, max_dd_pct=max_dd / start if start else None,
                peak_deployed=max(peak.values(), default=0.0), weekends=len(record), sharpe=sharpe,
                up=sum(r["pnl"] > 0 for r in record))
    return dict(record=record, seasons=list(seasons.values()), phases=ph, chart=chart, kpis=kpis,
                decisions=decisions(conn) if maker else
                _taker_decisions(conn, {r["profile"] for r in record}) if venue == "polymarket" else None,
                taker=detail)


def _taker_decisions(conn, ran):
    """The demo taker's decisions (the walk-forward rule on TAKER_POOL) for an account whose track record ran
    the story's profiles (profiles.HISTORY["taker"]; `ran`: the profile names in its record); else None."""
    from racinglines.pipelines import profiles as PF
    story = {PF.HISTORY_PROFILES[c]["name"] for c, *_ in PF.HISTORY.get("taker", []) if c in PF.HISTORY_PROFILES}
    return decisions(conn, taker=True) if story & set(ran) else None
