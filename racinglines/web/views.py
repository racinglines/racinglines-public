"""
The core maker views, all built on venues.event_matrix / season_matrix:

    /                 Markets: every sport, next races, season, recent results
    /race/{race_id}   one race: our fair vs every venue, my quotes, results, chart
    /season/{code}    a competition's season-long markets, same layout
    /book             my positions across venues
    /lab              launch backtests / scenario forecasts / diagnostics; compare runs
"""

from datetime import timedelta

import pandas as pd
from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from sqlalchemy import text

from racinglines import sports as SP
from racinglines.db.config import get_session
from racinglines.pipelines import sport_paper as SPP

from racinglines.web import audience as A
from racinglines.web import board as B
from racinglines.web import diag
from racinglines.markets import private_book as house
from racinglines.web import jobs
from racinglines.web import layout as LAYOUT
from racinglines.markets import venues as V
from racinglines.web import roles as R
from racinglines.web.app import ANY, PRO, allow, app, audit, check_csrf, conn, data, render, rows
from racinglines.web.viz import price_chart

KIND_ORDER = ["race_win", "race_podium", "race_top5", "race_top10", "race_make_final", "race_h2h", "race_constructor_top",
              "champion", "constructors_champion", "season_wins_ge", "standings_h2h"]


def _maker(user):
    return house.ALL if user["role"] == "admin" else user["id"]


def _groups(df):
    if not len(df):
        return []
    kinds = sorted(df["kind"].unique(), key=lambda k: KIND_ORDER.index(k) if k in KIND_ORDER else 99)
    return [(k, V.KIND_LABEL.get(k, k), rows(df[df["kind"] == k])) for k in kinds]


def _basic_acct(acct):
    """story.account as a basic account sees it: every track-record row, season and phase under roles.BASIC_NAME
    (never which strategy ran a weekend)."""
    rec = {id(r): R.basic_row(r) for r in acct["record"]}
    one = lambda r: rec.get(id(r), r)                          # noqa: E731
    return dict(acct, record=[one(r) for r in acct["record"]],
                seasons=[dict(y, rows=[one(r) for r in y["rows"]]) for y in acct["seasons"]],
                phases=[dict(R.basic_row(p), first=one(p["first"]), last=one(p["last"])) for p in acct["phases"]],
                decisions=None)                              # the story's decisions name every setup


def _sport_options(rows, sport="", user=None):
    """Every modeled sport in the app, plus any sport we have live rows for. This keeps the dropdown populated
    even when a sport has sparse or no current track-record data, instead of dropping it entirely. Sports `user`
    doesn't see ([sport] audience) are left out."""
    seen = set()
    for row in rows:
        code = row.get("sport") if isinstance(row, dict) else row
        if not code:
            continue
        if code in V.SPORT_NAME and code not in SP.SPORT_CODES:
            code = next((s["sport"]["code"] for s in map(SP.load, SP.SPORT_CODES)
                         if s["competition"]["code"] == code), code)
        seen.add(code)
    seen |= set(SP.SPORT_CODES)
    if sport:
        seen.add(sport)
    order = {code: i for i, code in enumerate(SP.SPORT_CODES)}
    seen = {code for code in seen if A.shown(user, code)}
    return sorted((code for code in seen if code), key=lambda code: (order.get(code, 99), str(code)))


def _live_venues():
    """The venue codes whose rows the pages show: every live venue in the registry (Kalshi with its switch on, each
    schema exchange with its own switch, the private book)."""
    return [v.code for v in V.VENUES if v.status == "live"]


def _other_exchanges():
    """Every exchange code but Polymarket (whether its switch is on or not): rows on these are another venue's record,
    kept out of the Polymarket view."""
    from racinglines import exchanges as EX
    return sorted({"kalshi", *EX.CODES} - {"polymarket"})


def book_total(events, maker=True):
    """The private book's P&L across events, as [(time, P&L)]: each event's own running P&L, carried forward to every
    poll time of any event, then summed. Sorting the events' curves together would interleave their separate running
    totals (a sawtooth, and a false drawdown at each event's start)."""
    from racinglines.pipelines import live as LV
    series = []
    for ev in events:
        pts = LV.book_curve(ev, maker)
        if pts:
            s = pd.Series([v for _, v in pts], index=pd.DatetimeIndex([t for t, _ in pts]))
            series.append(s[~s.index.duplicated(keep="last")].sort_index())
    if not series:
        return []
    idx = sorted(set().union(*[set(s.index) for s in series]))
    total = sum(s.reindex(idx, method="ffill").fillna(0.0) for s in series)     # before an event's first poll: 0
    return list(zip(idx, total.tolist()))


def _venue_names():
    """Display name by venue code, from the registry."""
    return {v.code: v.name for v in V.VENUES}


def _venue_order(codes):
    """The given venue codes in the registry's order (Polymarket first), unknown codes after, the private book last."""
    order = {v.code: i for i, v in enumerate(V.VENUES) if v.code != "private"}
    return sorted(codes, key=lambda c: (c == "private", order.get(c, len(order)), c))


def _sport_names():
    """Display names by competition code (V.SPORT_NAME) and by sport code (what the record and positions carry)."""
    by_sport = {s["sport"]["code"]: s["competition"].get("display_name", s["sport"]["name"])
                for s in map(SP.load, SP.SPORT_CODES)}
    return {**V.SPORT_NAME, **by_sport, "private": "Private book"}


# ---------------------------------------------------------------------------
# Markets (home)
# ---------------------------------------------------------------------------

@app.get("/markets", response_class=HTMLResponse)
def board_page(request: Request, msg: str = "", c=Depends(conn)):
    """Markets. Makers and admins: every sport's board (fair prices vs the venues), each in a collapsible
    section with its own exchange breakdown, and a calendar of every event across every sport, filterable by
    sport and exchange. Takers: every open Polymarket market with their strategy's calls (book_routes.bet_markets)."""
    user = request.state.user
    if R.is_basic(user):
        from racinglines.web.book_routes import bet_markets
        return bet_markets(request, msg=msg, c=c)
    from racinglines.markets import disagree as D
    hide = A.hidden_comps(user)                                     # sports this user doesn't see ([sport] audience)
    hide |= SP.off_board_competitions()                             # and sports off the Markets page ([sport] markets_board)
    hv = A.hidden_venues(user)                                      # and exchanges that are admin-only ([exchange] audience)
    calendar = [r for r in V.calendar_rows(c, *([hv] if hv else [])) if r["sport"] not in hide]
    cal_sports = sorted({(r["sport"], r["sport_name"]) for r in calendar}, key=lambda x: V.SPORT_ORDER.get(x[0], 9))
    cal_exchanges = sorted({x for r in calendar for x in r["exchanges"]})
    cal_events = [dict(sport=r["sport"], title=r["title"], status=r["status"], exchanges=r["exchanges"], url=r["url"],
                       date=None if r["date"] is None or pd.isna(r["date"]) else pd.Timestamp(r["date"]).isoformat())
                 for r in calendar]
    from racinglines.web import sport_status as SS                  # RACINGLINES_SPORT_STATUS=1: every sport's status
    return render(request, "board.html", sports=B.board(c, _maker(user), detail=False, hide=hide, hide_venues=hv), h=B.headline(c, _maker(user)), kalshi=V.KALSHI_VENUE,
                  disagree=D.panel(c) if D.ON["on"] else None,
                  sport_status=SS.status(c, hide, hv) if SS.enabled() else None, show_paper=True,
                  recorders=B.recorder_status(c, [v.code for v in A.venues(user, V.EXCHANGES) if v.code != "kalshi" or V.KALSHI_VENUE]),
                  calendar=calendar, cal_sports=cal_sports, cal_exchanges=cal_exchanges, cal_events=cal_events)


@app.get("/markets/sport/{code}", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def board_sport(request: Request, code: str, c=Depends(conn)):
    """One sport's body on the Markets board (race cards, season markets, exchange data), loaded when its collapsed
    section is opened, so the page itself only reads each sport's header."""
    user = request.state.user
    if R.is_basic(user):
        raise HTTPException(403)
    A.guard(user, code)
    got = B.board(c, _maker(user), only=code)
    if not got:
        raise HTTPException(404, "no such sport on the board")
    return render(request, "_board_sport.html", s=got[0], kalshi=V.KALSHI_VENUE)


@app.get("/markets/sport/{code}/recent", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def board_recent(request: Request, code: str, c=Depends(conn)):
    """A sport's recent results on the Markets board (the winner, our pre-race price, the venues'), loaded when the
    section is opened: each one re-reads a past race's whole market, the slowest part of the board."""
    if R.is_basic(request.state.user):
        raise HTTPException(403)
    A.guard(request.state.user, code)
    comp = c.execute(text("SELECT id FROM competitions WHERE code = :c"), dict(c=code)).scalar()
    if comp is None:
        raise HTTPException(404, "no such sport")
    return render(request, "_board_recent.html", recent=B.recent_results(c, int(comp)), kalshi=V.KALSHI_VENUE, code=code)


# ---------------------------------------------------------------------------
# Race / season pages
# ---------------------------------------------------------------------------

def _race_chart(c, info, pricing, df, exchange="polymarket"):
    wins = df[(df["kind"] == "race_win")] if len(df) else df
    tops = [r for r in wins.to_dict("records") if (r["venues"] or {}).get(exchange)][:5]
    if not tops:
        return None
    now = pd.Timestamp.now(tz="UTC")
    start = info["race_start"] if info["race_start"] is not None and not pd.isna(info["race_start"]) else None
    start = pd.Timestamp(start).tz_localize("UTC") if start is not None else None
    if info["status"] == "completed" and start is not None:
        t0, t1 = start - timedelta(hours=72), start + timedelta(hours=1)
    else:
        t0, t1 = now - timedelta(hours=72), now
    toks = {r["venues"][exchange]["token"]: r for r in tops}
    series = V.price_series(c, list(toks), t0.to_pydatetime(), t1.to_pydatetime())
    markers = []
    sess = data.q(c, """SELECT kind, (extra->>'session_date')::timestamp AS ts FROM rounds
                        WHERE race_id = :r AND extra->>'session_date' IS NOT NULL""", r=info["race_id"])
    for k, ts in zip(sess["kind"], sess["ts"]):
        markers.append((pd.Timestamp(ts).tz_localize("UTC"), {"qual": "quali", "race": "race", "final": "final"}.get(k, k)))
    if pricing.get("as_of") is not None:
        markers.append((pd.Timestamp(pricing["as_of"]).tz_localize("UTC"), "our price"))
    out = price_chart(series, {t: r["subject"] for t, r in toks.items()}, {t: r["fair"] for t, r in toks.items()},
                      markers)
    if out and exchange != "polymarket":
        out["venue"] = next(v.name for v in V.EXCHANGES if v.code == exchange)
    return out


def _diagnostic_job(competition):
    """The Lab's event-diagnostic job for a competition's sport (e.g. f1_diagnostic), None when it has none."""
    try:
        sport = SP.by_competition(competition)["sport"]["code"]
    except StopIteration:
        return None
    return next((j.code for j in jobs.CATALOG.values() if j.sport == sport and j.code.endswith("_diagnostic")), None)


def _calibration(competition):
    """The sport schema's [sport] calibration for a competition code (e.g. "baseline"), None when unset or unknown."""
    try:
        return SP.by_competition(competition)["sport"].get("calibration")
    except StopIteration:
        return None


@app.get("/races/{race_id}", response_class=HTMLResponse)
def race_page(request: Request, race_id: int, msg: str = "", c=Depends(conn)):
    user = request.state.user
    if R.is_basic(user):
        return RedirectResponse("/markets", status_code=303)
    hv = A.hidden_venues(user)
    info, pricing, df = V.event_matrix(c, race_id, _maker(user), hv)
    if info is None:
        raise HTTPException(404)
    A.guard(user, info["competition"])
    completed = info["status"] == "completed"
    wins = df[df["kind"] == "race_win"] if len(df) else df
    fav = rows(wins.head(1))[0] if len(wins) else None
    winner, results = None, []
    if completed:
        w = wins[wins["result"] == True] if len(wins) else wins  # noqa: E712
        winner = rows(w)[0] if len(w) else None
        res = house.race_outcomes(c, race_id).sort_values("position", na_position="last")
        names = data.q(c, "SELECT id AS athlete_id, display_name AS athlete FROM athletes WHERE id = ANY(:i)",
                       i=res["athlete_id"].astype(int).tolist())
        res = res.merge(names, on="athlete_id", how="left")
        fair = {(r["kind"], r["athlete_id"]): r["fair"] for r in df.to_dict("records")} if len(df) else {}
        results = [dict(r, win=fair.get(("race_win", r["athlete_id"])), podium=fair.get(("race_podium", r["athlete_id"])))
                   for r in rows(res)]
    diag_runs = rows(data.q(c, """SELECT id, left(params->>'cutoff', 16) AS cutoff FROM model_runs
                                  WHERE kind = 'diagnostic' AND params->>'event_key' = :k ORDER BY id""",
                            k=info["source_key"]))
    charts = [(v, ch) for v in A.venues(user, V.EXCHANGES) if v.status == "live" for ch in [_race_chart(c, info, pricing, df, v.code)] if ch]
    return render(request, "race.html", info=info, pricing=pricing, groups=_groups(df), fav=fav, winner=winner,
                  results=results, completed=completed, season=False, venue_sum=V.venue_summary(df, hv), mine=B._mine(df),
                  exchanges=A.venues(user, V.EXCHANGES), has_pm=bool(len(df) and df["pm_mid"].notna().any()),
                  charts=charts, countdown=B._countdown(info["start_date"]),
                  kalshi=V.KALSHI_VENUE,
                  diag_runs=diag_runs, msg=msg, kind_label=V.KIND_LABEL, placeholders=V.placeholders(c, race_id),
                  quote_kinds=list(V.STANDARD_KINDS.get(info["competition"], ("race_win", "race_podium"))),
                  calibration=_calibration(info["competition"]), diag_job=_diagnostic_job(info["competition"]))


@app.get("/seasons/{code}", response_class=HTMLResponse, dependencies=[allow(*PRO)])
def season_page(request: Request, code: str, c=Depends(conn)):
    A.guard(request.state.user, code)
    hv = A.hidden_venues(request.state.user)
    info, pricing, df = V.season_matrix(c, code, _maker(request.state.user), hv)
    if info is None:
        raise HTTPException(404)
    info = dict(info, status=None, start_date=None, name="Season-long markets", event_id=None, race_id=0, source_key=None)
    champs = df[df["kind"] == "champion"] if len(df) else df
    return render(request, "race.html", info=info, pricing=pricing, groups=_groups(df),
                  fav=rows(champs.head(1))[0] if len(champs) else None, winner=None, results=[], completed=False,
                  season=True, strategy=latest_season_strategy(c, code), venue_sum=V.venue_summary(df, hv), mine=B._mine(df),
                  exchanges=A.venues(request.state.user, V.EXCHANGES), kalshi=V.KALSHI_VENUE, charts=[],
                  has_pm=bool(len(df) and df["pm_mid"].notna().any()), countdown="", diag_runs=[],
                  msg="", kind_label=V.KIND_LABEL, quote_kinds=[], calibration=_calibration(code))


# ---------------------------------------------------------------------------
# My book
# ---------------------------------------------------------------------------

@app.get("/book", response_class=HTMLResponse)
def book_page(request: Request, maker: str = "", c=Depends(conn), user=allow(*PRO)):
    maker_id = user["id"] if user["role"] == "pro" else house.ALL
    if user["role"] == "admin" and maker:
        mk = data.q(c, "SELECT id FROM users WHERE username = :u", u=maker)
        maker_id = int(mk["id"].iloc[0]) if len(mk) else -1
    bk = house.book(c, maker_id=maker_id)
    groups = []
    if len(bk):
        bk["race_key"] = bk["race_id"].fillna(0).astype(int)
        meta = data.q(c, """SELECT ra.id AS race_key, v.name AS venue, e.start_date, e.status, co.code AS competition
                            FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
                            JOIN competitions co ON co.id = s.competition_id LEFT JOIN venues v ON v.id = e.venue_id
                            WHERE ra.id = ANY(:r)""", r=[int(x) for x in bk["race_key"].unique() if x])
        meta = {int(r["race_key"]): r for r in meta.to_dict("records")}
        season_codes = {}                      # season-long markets: the competition of the model run that priced them
        if (bk["race_key"] == 0).any():
            runs = [int(x) for x in bk.loc[bk["race_key"] == 0, "model_run_id"].dropna().unique()]
            if runs:
                season_codes = {int(r["id"]): r["code"] for r in data.q(c, """SELECT mr.id, co.code FROM model_runs mr
                    JOIN competitions co ON co.id = mr.competition_id WHERE mr.id = ANY(:r)""", r=runs).to_dict("records")}
        for rk, g in bk.groupby("race_key", sort=False):
            mt = meta.get(rk, {})
            title = "Season-long markets" if rk == 0 else V.race_title(mt.get("competition"), None, mt.get("venue"), None)
            code = None
            if rk == 0:
                ids = [int(x) for x in g["model_run_id"].dropna()]
                code = next((season_codes[i] for i in ids if i in season_codes), None) or "f1_wdc"
            groups.append(dict(race_id=rk, season_code=code, title=title, sport=V.SPORT_NAME.get(mt.get("competition"), ""),
                               date=mt.get("start_date"), status=mt.get("status", "open"),
                               markets=len(g), open=int((g["status"] == "open").sum()), bets=int(g["bets"].sum()),
                               staked=float(g["staked"].sum()), ev=float(g.loc[g["status"] == "open", "ev"].sum()),
                               worst=float(g.loc[g["status"] == "open", "worst"].sum()),
                               settled=float(g["settled_pnl"].dropna().sum()),
                               top=rows(g.sort_values("staked", ascending=False).head(5))))
        groups.sort(key=lambda x: (x["status"] == "completed", str(x["date"] or "9999")))
    tot = dict(markets=len(bk), open=int((bk["status"] == "open").sum()) if len(bk) else 0,
               bets=int(bk["bets"].sum()) if len(bk) else 0, staked=float(bk["staked"].sum()) if len(bk) else 0.0,
               ev=float(bk.loc[bk["status"] == "open", "ev"].sum()) if len(bk) else 0.0,
               worst=float(bk.loc[bk["status"] == "open", "worst"].sum()) if len(bk) else 0.0,
               settled=float(bk["settled_pnl"].dropna().sum()) if len(bk) else 0.0)
    makers = rows(data.q(c, "SELECT username FROM users WHERE role IN ('pro', 'maker', 'admin') ORDER BY username"))
    orders = data.q(c, "SELECT status, count(*) AS n FROM orders GROUP BY status")
    return render(request, "book.html", kalshi=V.KALSHI_VENUE, groups=groups, tot=tot, makers=makers, maker=maker,
                  orders=dict(zip(orders["status"], orders["n"])) if len(orders) else {})


# ---------------------------------------------------------------------------
# Lab: launch runs, compare them, diagnostics
# ---------------------------------------------------------------------------

def backtest_runs(c, limit=12, hide=()):
    df = data.q(c, """
        SELECT mr.id, mr.created_at, co.code AS competition, mr.params, mr.metrics->'summary' AS summary,
               jsonb_array_length(coalesce(mr.metrics->'events', '[]'::jsonb)) AS rows_
        FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
        WHERE mr.kind = 'backtest' AND NOT co.code = ANY(CAST(:h AS text[])) ORDER BY mr.id DESC LIMIT :n""",
                n=limit, h=list(hide))
    out = []
    for r in df.to_dict("records"):
        p, s = r["params"] or {}, r["summary"] or {}
        for track in ("True", "False"):
            pre = s.get(f"pre_race|track={track}")
            if not pre:
                continue
            out.append(dict(id=r["id"], created_at=r["created_at"], sport=V.SPORT_NAME.get(r["competition"]),
                            variant=p.get("variant", "baseline"), f1=r["competition"] == "f1_wdc",
                            races=p.get("races") or "all", half_life=p.get("half_life_days", 120), sims=p.get("sims"),
                            track="on" if track == "True" else "off",
                            win=pre.get("brier_win"), win_grid=pre.get("brier_win_grid"),
                            podium=pre.get("brier_podium"), podium_grid=pre.get("brier_podium_grid"),
                            top10=pre.get("brier_top10"), top10_grid=pre.get("brier_top10_grid"),
                            pre_quali_win=(s.get(f"pre_quali|track={track}") or {}).get("brier_win"),
                            pre_weekend_win=(s.get(f"pre_weekend|track={track}") or {}).get("brier_win")))
    return out


def scenarios(c, hide=()):
    df = data.q(c, """
        SELECT mr.id, mr.created_at, mr.kind, co.code AS competition, mr.competition_id, mr.category_id, mr.params
        FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
        WHERE (mr.kind = 'scenario' OR (mr.kind = 'forecast' AND mr.params ? 'label')) AND NOT co.code = ANY(CAST(:h AS text[]))
        ORDER BY mr.id DESC LIMIT 12""", h=list(hide))
    out = []
    for r in df.to_dict("records"):
        live, _ = data.latest_forecast_run(c, r["competition_id"], r["category_id"])
        nxt = data.q(c, """SELECT rp.race_id, a.display_name, rp.win_prob FROM race_predictions rp
                           JOIN athletes a ON a.id = rp.athlete_id
                           WHERE rp.model_run_id = :m AND rp.race_id = (SELECT min(race_id) FROM race_predictions
                                                                        WHERE model_run_id = :m AND race_id IS NOT NULL)
                           ORDER BY rp.win_prob DESC LIMIT 3""", m=r["id"])
        diff = None
        if live and len(nxt):
            lv = data.q(c, "SELECT athlete_id, win_prob FROM race_predictions WHERE model_run_id = :m AND race_id = :r",
                        m=live, r=int(nxt["race_id"].iloc[0]))
            sc = data.q(c, "SELECT athlete_id, win_prob FROM race_predictions WHERE model_run_id = :m AND race_id = :r",
                        m=r["id"], r=int(nxt["race_id"].iloc[0]))
            j = sc.merge(lv, on="athlete_id", suffixes=("", "_live"))
            diff = float((j["win_prob"] - j["win_prob_live"]).abs().max()) if len(j) else None
        p = r["params"] or {}
        out.append(dict(id=r["id"], created_at=r["created_at"], sport=V.SPORT_NAME.get(r["competition"]),
                        label=p.get("label"), half_life=p.get("half_life_days"), sims=p.get("sims"),
                        track=p.get("track_features"), live=r["id"] == live, kind=r["kind"],
                        top=[(n, w) for n, w in zip(nxt["display_name"], nxt["win_prob"])], max_diff=diff))
    return out


def latest_season_strategy(c, competition="f1_wdc"):
    """The latest saved default season strategy (kind='season_strategy') for a competition."""
    df = data.q(c, """SELECT mr.id, mr.created_at, mr.params, mr.metrics FROM model_runs mr
                      JOIN competitions co ON co.id = mr.competition_id
                      WHERE mr.kind = 'season_strategy' AND co.code = :c
                        AND coalesce(mr.params->>'variant', 'baseline') = 'baseline' ORDER BY mr.id DESC LIMIT 1""", c=competition)
    if not len(df):
        return None
    r = df.iloc[0].to_dict()
    m = r["metrics"] or {}
    eq = pd.DataFrame(m.get("equity", []))
    chart = None
    if len(eq):
        eq["t"] = pd.to_datetime(eq["t"], utc=True)
        from racinglines.web.viz import line_chart
        series = {"strategy": list(zip(eq["t"], eq["equity"].astype(float)))}
        if "hold" in eq:
            series["hold"] = list(zip(eq["t"], eq["hold"].astype(float)))
        chart = line_chart(series, {"strategy": "update after every race", "hold": "enter pre-season & hold"})
    now = m.get("now", [])
    actions = [n for n in now if n.get("action") not in ("hold", "no position")]
    return dict(id=int(r["id"]), created_at=r["created_at"], params=r["params"] or {}, summary=m.get("summary", {}),
                hold=m.get("hold", {}), positions=m.get("positions", []), trades=m.get("trades", []), now=now,
                actions=actions, decisions=m.get("decisions", []), chart=chart)


def _events_for_diagnostic(c):
    return rows(data.q(c, """
        SELECT e.source_key AS key, v.name AS venue, e.start_date,
               (SELECT min((ro.extra->>'session_date')::timestamp) FROM rounds ro WHERE ro.race_id = ra.id
                 AND ro.kind = 'race') AS race_start,
               EXISTS (SELECT 1 FROM market_links ml WHERE ml.race_id = ra.id AND ml.prediction <> 'unmodeled'
                       AND ml.closed) AS tape
        FROM events e JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
        JOIN races ra ON ra.event_id = e.id LEFT JOIN venues v ON v.id = e.venue_id
        WHERE co.code = 'f1_wdc' AND e.status = 'completed' ORDER BY e.start_date DESC LIMIT 40"""))


LAB_SECTIONS = [("run", "Run"), ("jobs", "Jobs"), ("models", "Model variants"), ("scenarios", "Scenarios"),
                ("backtests", "Backtests"), ("diagnostics", "Event diagnostics"), ("runs", "All runs")]


def _recent_jobs(c, user_id=None, hide=()):
    """The 15 most recent jobs; only this user's when user_id is given. No jobs of a sport in hide ([sport] audience):
    their log tails carry that sport's P&L."""
    return rows(data.q(c, """SELECT j.*, u.username FROM jobs j LEFT JOIN users u ON u.id = j.user_id
                             WHERE (CAST(:u AS integer) IS NULL OR j.user_id = CAST(:u AS integer))
                               AND NOT coalesce(j.sport, '') = ANY(CAST(:h AS text[]))
                             ORDER BY j.id DESC LIMIT 15""", u=user_id, h=list(hide)))


def _edge_ctx(c, user):
    from racinglines.web import edge
    cs = edge.combos(c, user)
    year = edge.year(c, user)
    sport, venue = edge.scope(c, user)
    sports, venues = edge.scopes(c, year)
    sports = [x for x in sports if A.shown(user, x)]       # a sport this user doesn't see has no Edge Finder rows
    sport = sport if A.shown(user, sport) else "f1"
    ef = edge.build(c, cs, year, sport, venue)
    return dict(ef=ef, combos=[list(x) for x in cs], year=year, years=edge.YEARS, candidates=edge.candidates(c),
                sport=sport, venue=venue, sports=sports, venues=venues, sport_names=_sport_names(),
                venue_names=_venue_names(),
                demo_pools=edge.demo_pools(c) if sport == "f1" and venue in ("", "polymarket") else [],
                sweep_job=f"{sport}_sweep" if f"{sport}_sweep" in jobs.CATALOG else "",
                models=[(cf["ref"], cf["label"]) for cf in ef["configs"]] or [("baseline", "baseline")],
                strategies=[(k, edge.label(k)) for k in edge.STRATEGY_KEYS])


def _model_brier(c):
    from racinglines.web import board
    win, grid = board.model_brier(c)
    return dict(win=win, grid=grid)


@app.get("/lab", response_class=HTMLResponse)
def lab_page(request: Request, job: str = "", event: str = "", variant: str = "", candidate: int = 0, cfg: str = "",
             sport: str = "", venue: str = "", msg: str = "", user=allow(*ANY), c=Depends(conn)):
    """The Edge Finder leads (saved runs only, nothing is simulated on a visit); the other sections
    load on demand (/lab/section/{key}). Edge Finder combos and job knobs come from the user's prefs
    (database); which sections are open is a browser view setting. Under the alpha layout the Lab tab is everyone's
    (web/layout.py): an account without Pro opens its strategy's performance instead."""
    from urllib.parse import urlencode

    if user["role"] not in PRO:
        if LAYOUT.alpha():
            return RedirectResponse("/strategy", status_code=303)
        raise HTTPException(403)

    active = any(j["status"] in ("queued", "running") for j in _recent_jobs(c, user["id"], A.hidden(user)))
    open_now = (["run"] if job or event or variant or candidate or cfg else []) + (["jobs"] if active or msg else [])
    q = {k: v for k, v in dict(job=job, event=event, variant=variant, candidate=candidate or "", cfg=cfg, sport=sport,
                               venue=venue).items() if v}
    return render(request, "lab.html", sections=LAB_SECTIONS, open_now=open_now, query="?" + urlencode(q) if q else "",
                  active=active, msg=msg, brier=_model_brier(c), **_edge_ctx(c, user))


@app.get("/lab/section/{key}", response_class=HTMLResponse)
def lab_section(request: Request, key: str, job: str = "", event: str = "", variant: str = "", scope: str = "",
                candidate: int = 0, cfg: str = "", sport: str = "", venue: str = "", user=allow(*PRO), c=Depends(conn)):
    from racinglines.web import edge
    from racinglines.web import prefs as P
    if key not in dict(LAB_SECTIONS):
        raise HTTPException(404)
    pr = P.get(c, user)
    hide = A.hidden_comps(user)
    ctx = dict(combos=[list(x) for x in edge.combos(c, user)])
    if key == "run":
        evs = _events_for_diagnostic(c)
        for e in evs:
            rs = e["race_start"]
            e["default_cutoff"] = ((pd.Timestamp(rs) - timedelta(days=1)).strftime("%Y-%m-%dT23:59") if rs is not None
                                   else f"{pd.Timestamp(e['start_date']) - timedelta(days=1):%Y-%m-%d}T23:59")
        sel_event = event or (evs[0]["key"] if evs else "")
        sel = next((e for e in evs if e["key"] == sel_event), None)
        from racinglines.pipelines import sweep_settings as SS
        knobs = {code: dict(v) for code, v in (pr.get("job_knobs") or {}).items()}     # last used, per job type
        cands = edge.candidates(c)
        # the sweep form starts from: a candidate, a saved configuration, a variant, or the user's last run
        start = dict(knobs.get("f1_sweep", {}).get("settings") or {})
        source = ""
        if candidate and (cd := next((x for x in cands if x["id"] == candidate), None)):
            start, source = cd["settings"], f"candidate {cd['name']}"
        elif cfg and (cf := edge.configs(c, edge.year(c, user)).get(edge.ref_key(cfg))):
            start, source = cf["settings"].to_json(), f"run #{cf['run_id']}"
        elif variant:
            start, source = {"variant": variant}, variant
        sel_job = job if job in jobs.CATALOG else ("f1_sweep" if variant or candidate or cfg else "")
        # a launcher link (?job=…&sport=…&venue=…) presets the chosen job's sport and exchange knobs
        preset = {k: v for k, v in dict(sport=sport, venue=venue).items()
                  if v and any(kn.name == k and v in kn.choices for kn in jobs.CATALOG[sel_job].knobs)} if sel_job else {}
        if preset:
            knobs[sel_job] = dict(knobs.get(sel_job, {}), **preset)
        if sel_job == "f1_sweep" and venue in SS.VENUES:
            start, source = dict(start, venue=venue), source or venue
        sweep_forms = {}
        from racinglines.pipelines import season_sweep as SW
        shown = [code for code in jobs.modeled_sports() if A.shown(user, code)]   # [sport] audience
        for code in shown:
            code_job = f"{code}_sweep"
            if code_job not in jobs.CATALOG:
                continue
            settings_cls = SW.settings_class(code)
            last = knobs.get(code_job, {})
            sport_start = dict(last.get("settings") or {})
            if sel_job == code_job:
                if cfg:
                    saved = edge.configs(c, edge.year(c, user), code, venue or None)
                    chosen = saved.get(edge.ref_key(cfg)) or next((x for x in saved.values() if x["ref"] == cfg), None)
                    if chosen:
                        sport_start = chosen["settings"].to_json()
                if venue:
                    sport_start["venue"] = venue
            sweep_forms[code_job] = dict(
                groups=[(g, dict(SS.GROUPS).get(g, g.replace("_", " ").title()),
                         [x for x in settings_cls.SPEC if x.group == g])
                        for g in dict.fromkeys(x.group for x in settings_cls.SPEC)],
                defaults=settings_cls.from_dict().to_json(),
                start=settings_cls.from_dict(sport_start, strict=False).to_json(),
                year=last.get("year", "2026"))
        ctx.update(sports=[(code, name, jts) for code, name, jts in
                           [(jobs.ANY_SPORT, "Any sport", [j for j in jobs.CATALOG.values() if j.sport == jobs.ANY_SPORT])]
                           + [(code, jobs.sport_name(code), [j for j in jobs.CATALOG.values() if j.sport == code])
                              for code in shown] if jts],
                   launch=jobs.launcher(), venue_name=jobs.venue_name,
                   sel_job=sel_job, events=evs, sel_event=sel_event,
                   sel_cutoff=sel["default_cutoff"] if sel else "", knobs=knobs, candidates=cands,
                   sweep_start=SS.Settings.from_dict(start, strict=False).to_json(), sweep_source=source,
                   sweep_groups=[(g, lab, [x for x in SS.SETTINGS if x.group == g]) for g, lab in SS.GROUPS],
                   sweep_defaults=SS.Settings.from_dict().to_json(), model_choices=jobs.MODEL_CHOICES,
                   variant_switches=jobs.variant_switches(), exclusive_switches=jobs.EXCLUSIVE_SWITCHES,
                   sel_candidate=candidate, sweep_forms=sweep_forms, hidden_sports=A.hidden(user))
    elif key == "jobs":
        scope = scope if scope in ("mine", "all") else "mine"
        js = _recent_jobs(c, user["id"] if scope == "mine" else None, A.hidden(user))
        ctx.update(jobs=js, scope=scope, active=any(j["status"] in ("queued", "running") for j in js))
    elif key == "models":
        from racinglines.models import evaluate as EV
        ctx.update(pnl=EV.strategy_pnl(c))
    elif key == "scenarios":
        fc = data.latest_forecasts(c)
        ctx.update(scenarios=scenarios(c, hide), forecasts=rows(fc[~fc["competition"].isin(hide)] if len(fc) else fc))
    elif key == "backtests":
        ctx.update(backtests=backtest_runs(c, hide=hide), swept=set(edge.swept_variants(c)))
    elif key == "diagnostics":
        ctx.update(event_diags=diag.event_summaries(c, hide), kalshi=V.KALSHI_VENUE)
    elif key == "runs":
        ctx.update(recent=rows(diag.recent_runs(c, hide=hide)))
    return render(request, f"lab_{key}.html", **ctx)


@app.post("/lab/edge", response_class=HTMLResponse, dependencies=[Depends(check_csrf)])
async def lab_edge(request: Request, user=allow(*PRO), c=Depends(conn)):
    """Edit the user's Edge Finder combos; returns the refreshed Edge Finder."""
    from racinglines.web import edge
    form = await request.form()
    if form.get("action") == "year":
        from racinglines.web import prefs as P
        y = int(form.get("year") or 0)
        if y not in edge.YEARS:
            raise HTTPException(400, "unknown season")
        P.put(c, user, "edge_year", y)
        return render(request, "lab_edge.html", **_edge_ctx(c, user))
    if form.get("action") == "scope":                  # the sport (one at a time) / venue ("" for all) filter
        from racinglines.web import prefs as P
        sport, venue = edge.scope(c, user)
        if "sport" in form:
            sport = form.get("sport") or "f1"
        if "venue" in form:
            venue = form.get("venue") or None
        sports, venues = edge.scopes(c, edge.year(c, user))
        if (sport != "f1" and sport not in sports) or not A.shown(user, sport) or (venue and venue not in venues):
            raise HTTPException(400, "no sweep for that sport or venue")
        P.put(c, user, "edge_scope", dict(sport=sport, venue=venue))
        return render(request, "lab_edge.html", **_edge_ctx(c, user))
    try:
        cs = edge.apply(edge.combos(c, user), form.get("action", ""), form.get("variant", ""),
                        form.get("strategy", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    edge.save(c, user, cs)
    return render(request, "lab_edge.html", **_edge_ctx(c, user))


@app.post("/lab/candidate", response_class=HTMLResponse, dependencies=[Depends(check_csrf)])
async def lab_candidate(request: Request, user=allow(*PRO), c=Depends(conn)):
    """Star a shown combo as a Lab candidate (action=add: variant=<config ref>, strategy, name, why), or
    delete one (action=delete, id). Returns the refreshed Edge Finder."""
    from sqlalchemy import text

    from racinglines.pipelines import search as SR
    from racinglines.web import edge
    form = await request.form()
    if form.get("action") == "delete":
        cid = int(form.get("id") or 0)
        c.execute(text("DELETE FROM model_runs WHERE id = :i AND kind = 'candidate'"), dict(i=cid))
        c.commit()
        audit(request, "candidate_delete", id=cid)
    else:
        year = edge.year(c, user)
        cf = edge.configs(c, year).get(edge.ref_key(form.get("variant", "")))
        strategy = form.get("strategy", "")
        if cf is None or strategy not in edge.STRATEGY_LABEL:
            raise HTTPException(400, "no saved full-season run for that configuration")
        name = (form.get("name") or "").strip()[:80] or f"{cf['label']} · {edge.label(strategy)}"
        cid = SR.add_candidate(c, name, cf["settings"], year, strategy, (form.get("why") or "").strip()[:300],
                               cf["run_id"], f"lab:{user['username']}")
        c.commit()
        audit(request, "candidate_add", id=cid, name=name)
    return render(request, "lab_edge.html", **_edge_ctx(c, user))


def _remember_knobs(user, code, params):
    """Last-used knobs per job type, to prefill the Run forms next time (not the event or cutoff)."""
    from racinglines.db.config import get_engine
    from racinglines.web import prefs as P
    with get_engine().connect() as c:
        k = dict(P.get(c, user).get("job_knobs") or {})
        k[code] = {n: v for n, v in params.items() if n not in ("event", "cutoff")}
        P.put(c, user, "job_knobs", k)


@app.post("/lab/run", dependencies=[Depends(check_csrf)])
async def lab_run(request: Request, user=allow(*PRO)):
    form = await request.form()
    jt = jobs.CATALOG.get(form.get("job"))
    if jt is None or not A.shown(user, jt.sport):
        raise HTTPException(400, "unknown job")
    try:
        params = jobs.parse(jt, form)
    except (ValueError, TypeError) as e:
        return RedirectResponse(f"/lab?job={jt.code}&msg=Error: {e}#run", status_code=303)
    if not A.shown(user, params.get("sport")):          # the shared core's sport knob: [sport] audience
        raise HTTPException(400, "unknown sport")
    job_id = jobs.submit(jt, params, user["id"])
    audit(request, "job_submit", job_id=job_id, kind=jt.code, params=params)
    _remember_knobs(user, jt.code, params)
    return RedirectResponse(f"/lab?msg=Job {job_id} queued: {jt.label}#jobs", status_code=303)


@app.post("/lab/promote/{run_id}", dependencies=[Depends(check_csrf)])
def lab_promote(request: Request, run_id: int, user=allow(*PRO)):
    try:
        with get_session() as s:
            jobs.promote(s, run_id)
    except ValueError as e:
        return RedirectResponse(f"/lab?msg=Error: {e}", status_code=303)
    audit(request, "promote_forecast", run_id=run_id)
    return RedirectResponse(f"/lab?msg=Run {run_id} is now the live forecast#scenarios", status_code=303)


@app.get("/lab/diagnostics", dependencies=[allow(*PRO)])
def diag_redirect():
    return RedirectResponse("/lab#diagnostics", status_code=303)


@app.get("/lab/runs", dependencies=[allow(*PRO)])
def runs_redirect():
    return RedirectResponse("/lab", status_code=303)


@app.get("/positions", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def positions_page(request: Request, event: str = "", venue: str = "", sort: str = "", sport: str = "", c=Depends(conn)):
    """Positions: the account's ledger. Every paper position its strategy took on Polymarket (open, settled
    at the race, closed before it), by market type and by weekend; with a weekend picked, the executions
    behind them (a taker's trades taken, a maker's paper fills). A taker's in-app bets, if any, below.
    venue splits the Polymarket history (paper, mostly replayed weekends) from the maker's private book (live,
    one day); sort orders the positions by P&L (pnl: best first, -pnl: worst first) instead of by weekend."""
    from racinglines.pipelines import live as LV
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import weekend_sweep as WS
    user = request.state.user
    profile = PF.of_user(c, user["id"])
    maker = bool(profile and profile.get("strategy") not in WS.TAKER_MODES)
    sp = SPP.enabled()                                  # RACINGLINES_SPORT_PAPER=1: NASCAR / MotoGP demo rows too
    live = _live_venues()                               # the venue registry decides which venues' rows show
    if venue and venue not in live and not sp:
        venue = ""
    hide_private = LAYOUT.alpha()                       # the alpha layout archives the private book (web/layout.py)
    if hide_private and venue == "private":
        venue = ""
    # the account's signals are read once and grouped (per weekend, per market), not looked up again for each
    # position: the signals table has no (user, market) index, so a lookup per position scanned them all
    pos = rows(data.q(c, """
        WITH s AS (SELECT event_key, market_key, action, strategy, detail FROM strategy_signals WHERE user_id = :u),
             bf AS (SELECT event_key, bool_or(detail->>'backfill' = 'true') AS backfill FROM s GROUP BY 1),
             tr AS (SELECT market_key,
                           count(*) FILTER (WHERE action = 'fill' OR (action IN ('buy', 'sell')
                                            AND coalesce(detail->>'followed', 'true') = 'true')) AS trades,
                           bool_or(strategy = 'buy_all' OR detail->>'mode' = 'buy_all') AS buy_all
                    FROM s GROUP BY 1)
        SELECT p.*, coalesce(ra.format->>'event_name', e.name) AS event_name, e.start_date,
               coalesce(sp.code, 'private') AS sport, bf.backfill, coalesce(tr.trades, 0) AS trades
        FROM paper_positions p
        LEFT JOIN bf ON bf.event_key = p.event_key
        LEFT JOIN tr ON tr.market_key = p.market_key
        LEFT JOIN events e ON e.source_key = p.event_key
        LEFT JOIN seasons se ON se.id = e.season_id
        LEFT JOIN competitions co ON co.id = se.competition_id
        LEFT JOIN sports sp ON sp.id = co.sport_id
        LEFT JOIN LATERAL (SELECT format FROM races WHERE event_id = e.id ORDER BY id LIMIT 1) ra ON true
        WHERE p.user_id = :u
          AND NOT coalesce(tr.buy_all, false)                       -- the debug mode: never shown
          AND (p.venue = ANY(:live)"""
        + (" OR p.event_key IN (" + SPP.SPORT_KEYS + ")" if sp else "") + """)
        ORDER BY """ + ("e.start_date DESC NULLS LAST, " if sp else "") + """p.event_key DESC, p.kind, p.subject""",
        u=user["id"], live=live))
    pos = [p for p in pos if A.shown(user, p.get("sport"))]         # [sport] audience: admin-only sports left out
    if sport:
        sport = sport.lower()
        pos = [p for p in pos if (p.get("sport") or "").lower() == sport]
    demo_keys = set(c.execute(text(SPP.SPORT_KEYS), dict(u=user["id"])).scalars()) if sp else set()
    for p in pos:
        p["demo_sport"] = p["event_key"] in demo_keys                   # a NASCAR / MotoGP demo replay (in-sample)
    pos = [p for p in pos if p["trades"] or p["venue"] == "private"]      # markets the account actually traded
    if hide_private:
        pos = [p for p in pos if p["venue"] != "private"]
    basic = R.is_basic(user)                        # basic: never which strategy made a pick (roles.basic_*)
    if basic:
        pos = [R.basic_position(p) for p in pos]
    for p in pos:
        if p["venue"] == "private" and not p["event_name"]:
            p["event_name"] = LV.event_name(p["event_key"])
    venues = {}
    for p in pos:
        y = float(p["outcome"]) if p["outcome"] is not None else p["mark"]
        v = venues.setdefault(p["venue"], dict(venue=p["venue"], n=0, open=0, pnl=0.0, won=0, events=set(), days=set()))
        pnl = 0.0 if y is None else p["cash"] + p["yes_shares"] * y + p["no_shares"] * (1 - y)
        v["n"] += 1
        v["pnl"] += pnl
        v["open"] += p["outcome"] is None
        v["won"] += p["outcome"] is not None and pnl > 0
        v["events"].add(p["event_key"])
        d = p["start_date"] if p["venue"] != "private" else p["updated_at"]     # the private book ran on one day
        if d is not None:
            v["days"].add(d.date() if hasattr(d, "date") else d)
    priv_events = sorted(venues.get("private", {}).get("events", ()))
    for v in venues.values():
        v.update(events=len(v["events"]), first=min(v["days"], default=None), last=max(v["days"], default=None))
        del v["days"]
    if venue:
        pos = [p for p in pos if p["venue"] == venue]
    for p in pos:
        held = abs(p["yes_shares"]) + abs(p["no_shares"]) > 1e-9
        if maker:
            p["side"] = "long YES" if p["yes_shares"] > 1e-9 else ("short YES" if p["yes_shares"] < -1e-9 else None)
            p["shares"] = abs(p["yes_shares"])
        else:
            p["side"] = "YES" if p["yes_shares"] > 1e-9 else ("NO" if p["no_shares"] > 1e-9 else None)
            p["shares"] = p["yes_shares"] if p["side"] == "YES" else p["no_shares"]
        y = float(p["outcome"]) if p["outcome"] is not None else p["mark"]
        p["pnl"] = None if y is None else p["cash"] + p["yes_shares"] * y + p["no_shares"] * (1 - y)
        p["state"] = "settled" if p["outcome"] is not None and held else ("closed" if not held else "open")
        p["won"] = p["state"] == "settled" and (p["pnl"] or 0) > 0
    weekends = []
    for p in pos:
        if not weekends or weekends[-1]["event_key"] != p["event_key"]:
            weekends.append(dict(event_key=p["event_key"], name=p["event_name"] or p["event_key"], n=0, pnl=0.0))
        weekends[-1]["n"] += 1
        weekends[-1]["pnl"] += p["pnl"] or 0.0
    shown = [p for p in pos if p["event_key"] == event] if event else pos
    if sort in ("pnl", "-pnl"):                         # unpriced positions last either way
        shown = sorted(shown, key=lambda p: (p["pnl"] is None, -(p["pnl"] or 0) if sort == "pnl" else (p["pnl"] or 0)))
    else:
        sort = ""

    def link(**kw):
        """This page's URL with the current filters, some replaced (None or "" drops one)."""
        from urllib.parse import urlencode
        q = {k: v for k, v in {**dict(venue=venue, event=event, sort=sort, sport=sport), **kw}.items() if v}
        return "/positions" + ("?" + urlencode(q) if q else "")
    open_ = [p for p in pos if p["state"] == "open"]
    done = [p for p in pos if p["state"] != "open"]
    by_kind = {}
    for p in pos:
        k = by_kind.setdefault(p["kind"], dict(kind=p["kind"], n=0, open=0, pnl=0.0, won=0, settled=0))
        k["n"] += 1
        k["open"] += p["state"] == "open"
        k["pnl"] += p["pnl"] or 0.0
        k["settled"] += p["state"] == "settled"
        k["won"] += p["won"]
    paper = dict(n=len(pos), open=len(open_), settled_pnl=sum(p["pnl"] or 0 for p in done),
                 open_value=sum(p["pnl"] or 0 for p in open_), won=sum(p["won"] for p in done),
                 held=sum(1 for p in done if p["state"] == "settled"))
    trades = []
    if event:
        trades = rows(data.q(c, """SELECT * FROM strategy_signals WHERE user_id = :u AND event_key = :e
                                     AND (action = 'fill' OR (action IN ('buy', 'sell')
                                          AND coalesce(detail->>'followed', 'true') = 'true'))
                                   ORDER BY signal_ts, id""", u=user["id"], e=event))
        # a replay on another exchange flags its signals detail.venue (e.g. 'kalshi'): live venues only, one at a time
        tv = lambda t: (t["detail"] or {}).get("venue") or "polymarket"  # noqa: E731
        trades = [t for t in trades if (tv(t) in live or (sp and event in demo_keys))
                  and (not venue or venue == "private" or tv(t) == venue)]
        if basic:
            trades = [R.basic_signal(t, profile) for t in trades]
    my_bets = house.taker_bets(c, user["id"]) if R.is_basic(user) and not hide_private else pd.DataFrame()
    summary = dict(bets=len(my_bets), staked=float(my_bets["stake"].sum()), open=int((my_bets["status"] == "open").sum()),
                   pnl=float(my_bets["pnl"].sum())) if len(my_bets) else None
    from racinglines.pipelines import story
    from racinglines.web.book_routes import polymarket_calls
    # the two venues are never plotted together: the Polymarket history (the strategy's record) and the
    # private book's P&L through its day(s), from the live snapshots; the page switches between them
    acct = story.account(c, user["id"], profile, maker, markers=False, sport=sport or None, sports=False,
                         hide=A.hidden(user)) if profile else None
    if acct and basic:
        acct = _basic_acct(acct)
    book = None
    if priv_events:
        from racinglines.web.viz import line_chart
        curve = book_total(priv_events, maker)
        cum = [v for _, v in curve]
        book = dict(chart=line_chart({"pnl": curve}, {"pnl": "private book P&L, marked to fair"}), polls=len(curve),
                    max_dd=min((v - max(cum[:i + 1]) for i, v in enumerate(cum)), default=0.0))
    plot = "private" if venue == "private" or (book and not acct) else "polymarket"
    vaccts = {}                                         # the record on every other exchange the account has rows on
    for code in _venue_order(venues):
        if code not in ("polymarket", "private") and profile:
            a = story.account(c, user["id"], profile, maker, markers=False, venue=code, sport=sport or None,
                              hide=A.hidden(user))
            vaccts[code] = _basic_acct(a) if a and basic else a
    if venue in vaccts:
        plot = venue
    coming = polymarket_calls(c, profile, n_races=2) if profile else None
    kcoming = polymarket_calls(c, profile, n_races=2, exchange="kalshi") if profile and V.KALSHI_VENUE else None
    sport_options = _sport_options(pos, sport, user)
    return render(request, "positions.html", profile=R.basic_view_profile(profile) if basic else profile, maker=maker,
                  paper=paper, shown=shown,
                  weekends=weekends, event=event, by_kind=sorted(by_kind.values(), key=lambda k: -k["n"]),
                  trades=trades, my_bets=rows(my_bets), summary=summary, acct=acct, coming=coming,
                  venues={v["venue"]: v for v in venues.values()}, venue=venue, sort=sort, sport=sport,
                  sport_options=sport_options, sport_names=_sport_names(), link=link, book=book, plot=plot,
                  vtotal=sum(v["pnl"] for v in venues.values()), open_pos=open_,
                  cur=next((w for w in weekends if w["event_key"] == event), None),
                  vaccts=vaccts, venue_order=_venue_order(venues), venue_names=_venue_names(), kcoming=kcoming,
                  sport_paper=bool(sp and demo_keys), show_backtests=bool(profile and not basic),
                  hide_private=hide_private,
                  note_sports=[sport] if sport else sorted({p["sport"] for p in pos if isinstance(p.get("sport"), str)}))   # source note


@app.get("/positions/backtests", response_class=HTMLResponse, dependencies=[allow(*PRO)])
def positions_backtests(request: Request, year: int | None = None, sport: str = "f1", venue: str = "polymarket",
                        config: str = "", strategy: str = "", c=Depends(conn)):
    """A selectable, saved full-season Edge Finder P&L curve, kept separate from an account's paper record."""
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import sweep_settings as SS
    from racinglines.web import edge as E
    from racinglines.web.viz import line_chart

    if year is not None and year not in E.YEARS:
        raise HTTPException(400, f"year must be one of {', '.join(map(str, E.YEARS))}")
    if strategy and strategy not in E.STRATEGY_LABEL:
        raise HTTPException(400, "unknown saved-sweep strategy")

    t1 = PF.TAKER_PROFILES["T1"]
    t1_key = SS.Settings.from_dict(t1["settings"]).key
    available_years = [y for y in E.YEARS if E.scopes(c, y)[0]]
    if year is None:
        for y in available_years:
            if t1_key in E.configs(c, y, "f1", "polymarket"):
                year = y
                break
        year = year or (available_years[0] if available_years else E.YEARS[0])

    sports, _ = E.scopes(c, year)
    sports = [x for x in sports if A.shown(request.state.user, x)]     # [sport] audience
    sport = sport if sport in sports else (sports[0] if sports else "f1")
    _, venues = E.scopes(c, year)
    venues = [v for v in venues if E.configs(c, year, sport, v)]
    venue = venue if venue in venues else (venues[0] if venues else "polymarket")
    cfgs = E.configs(c, year, sport, venue)

    selection_note = ""
    if config and config not in cfgs:
        selection_note = "That saved configuration is no longer available for this season, sport, and venue."
    preferred = t1_key if sport == "f1" and venue == "polymarket" and t1_key in cfgs else None
    key = config if config in cfgs else preferred or (SS.Settings.from_dict().key if SS.Settings.from_dict().key in cfgs
                                                       else next(iter(cfgs), ""))
    cfg = cfgs.get(key)
    strategy = strategy if strategy in E.STRATEGY_KEYS else ("update" if key == t1_key else "maker")
    curve = None
    total = None
    traded = 0
    run_id = None
    missing_dates = []
    if cfg:
        run = c.execute(text("SELECT metrics FROM model_runs WHERE id = :id"), dict(id=cfg["run_id"])).scalar_one()
        run_id = cfg["run_id"]
        weekends = (run or {}).get("weekends") or []
        traded_rows = [w for w in weekends if w.get(f"{strategy}_pnl") is not None]
        traded = len(traded_rows)
        if traded_rows:
            event_keys = [w["event_key"] for w in traded_rows]
            comp = SP.load(sport)["competition"]["code"]
            dates = dict(c.execute(text("""SELECT e.source_key, e.start_date
                                        FROM events e JOIN seasons se ON se.id = e.season_id
                                        JOIN competitions co ON co.id = se.competition_id
                                        WHERE co.code = :comp AND se.year = :year
                                          AND e.source_key = ANY(:keys)"""),
                                   dict(comp=comp, year=year, keys=event_keys)).all())
            missing_dates = [w["event_key"] for w in traded_rows if w["event_key"] not in dates]
            if not missing_dates:
                points, total = E.cumulative_curve(weekends, strategy, dates)
                curve = line_chart({"saved": points}, {"saved": "saved backtest, cumulative P&L"})
                run_id = cfg["run_id"]
    if sport == "f1" and venue == "polymarket" and t1_key not in cfgs:
        selection_note = (selection_note + " " if selection_note else "") + \
            "T1/A has an aggregate candidate score, but no complete-season saved sweep is available for this selection."
    cfg_options = sorted(cfgs.values(), key=lambda x: (x["key"] != t1_key, x["label"]))
    strategy_name = E.label(strategy) if strategy in E.STRATEGY_LABEL else strategy
    cfg_name = t1["name"] if cfg and key == t1_key else (cfg["label"] if cfg else "")
    sweep_job = f"{sport}_sweep" if f"{sport}_sweep" in jobs.CATALOG else ""
    return render(request, "positions_backtests.html", years=available_years, year=year, sports=sports, sport=sport,
                  venues=venues, venue=venue, configs=cfg_options, config=key, strategy=strategy,
                  strategies=E.STRATEGY_KEYS, strategy_labels=E.STRATEGY_LABEL, backtest_chart=curve, total=total,
                  traded=traded, run_id=run_id, config_name=cfg_name, strategy_name=strategy_name,
                  missing_dates=missing_dates, selection_note=selection_note, t1_key=t1_key, sweep_job=sweep_job,
                  sport_names=_sport_names(), venue_names=_venue_names())


# ---------------------------------------------------------------------------
# Signals: the live paper signals of the user's strategy profile (pipelines/signals.py)
# ---------------------------------------------------------------------------

def _demo_user(me):
    from racinglines.web import demo
    return demo.is_demo(me)


@app.get("/strategy", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def signals_page(request: Request, user: str = "", event: str = "", venue: str = "", sport: str = "", msg: str = "",
                 c=Depends(conn)):
    """The account's track record (every weekend: profile, trades, paper P&L), then one weekend's signals by
    stage and its paper positions (default: the latest). Takers see what to do and how hot it is (the
    modelled EV, graded), never our fair value or edge. Backfilled weekends are backtest replays and are
    labelled so. Admins can view any user."""
    from sqlalchemy import text as T

    from racinglines.markets.alerts import signal_line
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import weekend_sweep as WS
    from racinglines.pipelines.signals import HEAT_LABEL
    me = request.state.user
    uid = me["id"]
    if user and me["role"] == "admin":
        uid = c.execute(T("SELECT id FROM users WHERE username = :u"), dict(u=user)).scalar() or uid
    viewer = c.execute(T("SELECT id, username, role FROM users WHERE id = :u"), dict(u=uid)).mappings().first()
    profile = PF.of_user(c, uid)
    show_fair = not R.is_basic(me)
    basic = not show_fair                            # basic: never which strategy made a pick (roles.basic_*)
    # what the running strategy is, from its profile's `why` (pro and admin only: it names the strategy)
    profile_why = next((pr.get("why") for pr in {**PF.PROFILES, **PF.HISTORY_PROFILES}.values()
                        if profile and pr.get("name") == profile.get("name")), None) if show_fair else None
    from racinglines.pipelines import story
    is_maker = bool(profile and profile.get("strategy", "update") not in WS.TAKER_MODES)
    # venue=<exchange> (a live venue in the registry other than Polymarket, e.g. kalshi or og): the same profile's
    # record on that exchange's tape, where the account has rows there
    other = [v for v in _live_venues() if v not in ("polymarket", "private")]
    has = set(c.execute(T("SELECT DISTINCT venue FROM paper_positions WHERE user_id = :u"), dict(u=uid)).scalars())
    venue_tabs = [v for v in other if v in has]
    venue = venue if venue in venue_tabs else ""
    sp = SPP.enabled() and not venue                # RACINGLINES_SPORT_PAPER=1: the NASCAR / MotoGP demo rows join the record
    acct = story.account(c, uid, profile, is_maker, venue=venue or "polymarket", sport=sport or None, sports=sp,
                         hide=A.hidden(me), **({"markers": False} if basic else {}))
    if basic:
        acct = _basic_acct(acct)
    record, seasons, total = acct["record"], acct["seasons"], acct["kpis"]["total"]
    if event and event not in {r["event_key"] for r in record}:
        event = ""                                  # not in this account's record (or a sport this viewer doesn't see)
    ev = event or (record[-1]["event_key"] if record else None)
    cur = next((r for r in record if r["event_key"] == ev), None)
    if venue:
        sig = data.q(c, """SELECT * FROM strategy_signals WHERE user_id = :u AND event_key = :e AND detail->>'venue' = :v
                           ORDER BY signal_ts DESC NULLS LAST, id DESC""", u=uid, e=ev, v=venue) if ev else pd.DataFrame()
        pos = data.q(c, """SELECT * FROM paper_positions WHERE user_id = :u AND event_key = :e AND venue = :v
                           ORDER BY kind, subject""", u=uid, e=ev, v=venue) if ev else pd.DataFrame()
    else:
        sig = data.q(c, """SELECT * FROM strategy_signals WHERE user_id = :u AND event_key = :e
                           AND (NOT coalesce(detail->>'venue', 'polymarket') = ANY(:x)""" + (
                           " OR event_key IN (" + SPP.SPORT_KEYS + ")" if sp else "") + """)
                           ORDER BY signal_ts DESC NULLS LAST, id DESC""", u=uid, e=ev, x=_other_exchanges()) if ev else pd.DataFrame()
        pos = data.q(c, """SELECT * FROM paper_positions WHERE user_id = :u AND event_key = :e AND (NOT venue = ANY(:x)""" + (
                           " OR event_key IN (" + SPP.SPORT_KEYS + ")" if sp else "") + """)
                           ORDER BY kind, subject""", u=uid, e=ev, x=_other_exchanges()) if ev else pd.DataFrame()
    stages = []
    for lab, g in (sig.groupby("stage", sort=False) if len(sig) else []):
        items = rows(g)
        if basic:
            items = [R.basic_signal(s, profile) for s in items]
        for s in items:
            s["line"] = signal_line(s)
            s["heat_label"] = HEAT_LABEL.get(s["heat"]) if s["heat"] else None
            s["followed"] = (s["detail"] or {}).get("followed")
        stages.append(dict(stage=lab, signals=items))
    positions = [R.basic_position(p) for p in rows(pos)] if basic else rows(pos)
    for p in positions:
        y = float(p["outcome"]) if p["outcome"] is not None else p["mark"]
        p["value"] = None if y is None else p["cash"] + p["yes_shares"] * y + p["no_shares"] * (1 - y)
        p["settled"] = p["outcome"] is not None
    if uid == me["id"]:
        from racinglines.web import demo
        if demo.is_demo(me) and me.get("sid"):                   # a demo session: seen for this session only
            demo.prefs(me["sid"])["signals_seen_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        else:
            with get_session() as s:
                s.execute(T("UPDATE strategy_signals SET seen_at = now() WHERE user_id = :u AND seen_at IS NULL"), dict(u=uid))
                s.commit()
    users = c.execute(T("""SELECT username FROM users WHERE prefs ? 'strategy_profile' ORDER BY id""")).scalars().all() \
        if me["role"] == "admin" else []
    maker = False if basic else bool(cur and cur["strategy"] not in WS.TAKER_MODES) if cur else is_maker
    sport_options = _sport_options(record, sport, me)
    return render(request, "strategy.html", viewer=viewer, profile=R.basic_view_profile(profile) if basic else profile,
                  show_fair=show_fair, profile_why=profile_why, stages=stages,
                  positions=positions, cur=cur, event_key=ev, users=users, maker=maker, record=record,
                  seasons=seasons, total=total, acct=acct, is_maker=is_maker, heat_label=HEAT_LABEL,
                  venue=venue, sport=sport, sport_options=sport_options, sport_names=_sport_names(),
                  venue_tabs=venue_tabs, venue_names=_venue_names(), mix=None if basic else story.mix(record),
                  msg=msg, can_pick=uid == me["id"] and not _demo_user(me))


# ---------------------------------------------------------------------------
# Live: a private-book event as it runs (pipelines/live.py and the sport's adapter), for the demo maker and taker
# ---------------------------------------------------------------------------

@app.get("/live", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def live_page(request: Request, partial: int = 0, t: str = "", event: str = ""):
    """The live event: the shared shell (status, replay bar, polling) over the registry's event (event: a run
    or event key; default the most recently updated), with the sport's body (templates/live_<sport>.html,
    built by its adapter's view()). Downhill: leaderboard, on course, up next; F1: the weekend's stages and
    every market's quotes. Makers also see the model and the book; takers see their picks' P&L. Once the
    event is over it is a replay: the page as it was at any saved snapshot (t, e.g. 20260927T223221; default
    the end), stepped or played through on a timeline."""
    ctx = live_context(event, t, not R.is_basic(request.state.user), partial, A.hidden(request.state.user),
                       A.hidden_venues(request.state.user))
    return render(request, "live_partial.html" if partial else "live.html", **ctx)


@app.get("/live/f1", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def live_f1_page(request: Request):
    """The Live tab on the newest F1 event (its live-timing button where RACINGLINES_F1_STREAM is on)."""
    from racinglines.pipelines import live as LV
    f1 = next((e for e in LV.events() if e["sport"] == "f1"), None)
    return live_page(request, event=f1["run"] if f1 else "")




def live_context(event="", t="", maker=True, partial=0, hide=(), hide_venues=()):
    """The Live tab's template context: the registry's event, live or as of a replay snapshot, with the
    sport's body from its adapter (no database: the run folder only). hide: sport codes left out ([sport] audience);
    hide_venues: exchange codes whose prices are stripped from the snapshot ([exchange] audience)."""
    from urllib.parse import quote

    from racinglines.pipelines import live as LV
    from racinglines.web import f1_live as FL
    evs = [e for e in LV.events() if e["sport"] not in hide]
    ev = next((e for e in evs if event in (e["run"], e["event_key"])), None) if event else (evs[0] if evs else None)
    mode = LV.state(ev["run"]) if ev else None
    times = LV.snap_times(ev["run"]) if ev else []
    if ev and t and t in times:
        mode = "replay"
        snap, picks, hist = LV.load_at(ev["run"], t)
    elif mode == "replay" and times:
        t = t if t else times[-1]
        snap, picks, hist = LV.load_at(ev["run"], t)
    else:
        snap, picks, hist = LV.load(ev["run"]) if ev else (None, [], [])
    if snap and hide_venues:        # snapshots are written for every venue; an admin-only one is dropped at read time
        snap = dict(snap, markets=[dict(m, exchange_prices=[x for x in m.get("exchange_prices") or []
                                                            if x.get("code") not in hide_venues])
                                   for m in snap.get("markets") or []])
    if snap and ev and ev["sport"] == "f1":
        snap = dict(snap, title=ev["title"])            # snapshots keep the name they were written with
    selected = next((x for x in reversed(times) if x <= t), times[0]) if times and t else (times[-1] if times else None)
    if selected is not None and selected not in times:
        selected = None
    ctx = dict(snap=snap, maker=maker, partial=partial, mode=mode, times=times,
               t=selected, current_run=(ev or {}).get("run"),
               sport=(ev or {}).get("sport") or "mtb_dh",
               evq=f"event={quote(ev['run'], safe='')}&" if event and ev else "",   # the registry's name, URL-quoted
               events=evs, f1_stream=FL.enabled(), stream_event=(ev or {}).get("event_key"))
    if snap and ev:
        ctx.update(LV.adapter(ctx["sport"]).view(ev["run"], snap, picks, hist, mode, maker))
    return ctx
