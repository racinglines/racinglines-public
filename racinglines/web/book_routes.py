"""
The private book: markets quoted by pro accounts (admin = house), bets placed by basic accounts, and the
taker's view of the open exchange markets with their strategy's calls.
Split out of web/app.py, which imports this module to register the routes.
"""

import os

import pandas as pd
from fastapi import Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from racinglines.db import models as m
from racinglines.db.config import get_session
from racinglines.markets import private_book as house
from racinglines.web import audience as A
from racinglines.web import roles as R
from racinglines.web import users as U
from racinglines.web.app import ANY, PRO, allow, app, audit, check_csrf, conn, data, render, rows  # noqa: F401


MAX_STAKE = float(os.environ.get("MAX_STAKE", "100"))   # per-bet cap for taker bets (demo)


def _races_for_house(c, hide=()):
    """Races to quote: upcoming ones and any with markets, none of a competition in hide ([sport] audience)."""
    season = data.q(c, "SELECT count(*) AS n FROM house_markets WHERE race_id IS NULL AND status = 'open'")
    pseudo = [dict(race_id=0, venue="Season-long markets (championships, props)", start_date=None, category="",
                   status="open", markets=int(season["n"].iloc[0]))] if int(season["n"].iloc[0]) else []
    return pseudo + rows(data.q(c, """
        SELECT ra.id AS race_id, v.name AS venue, e.start_date, c.code AS category, e.status,
               (SELECT count(*) FROM house_markets hm WHERE hm.race_id = ra.id) AS markets
        FROM races ra JOIN events e ON e.id = ra.event_id JOIN categories c ON c.id = ra.category_id
        JOIN seasons se ON se.id = e.season_id JOIN competitions co ON co.id = se.competition_id
        LEFT JOIN venues v ON v.id = e.venue_id
        WHERE (e.status <> 'completed' OR EXISTS (SELECT 1 FROM house_markets hm WHERE hm.race_id = ra.id))
          AND NOT co.code = ANY(CAST(:h AS text[]))
        ORDER BY e.start_date, c.code""", h=list(hide)))


def _guard_race(c, user, race_id):
    """404 for a race of a competition this user doesn't see; race 0 is the season-long pseudo race."""
    if race_id:
        comp = data.q(c, """SELECT co.code FROM races ra JOIN events e ON e.id = ra.event_id
                            JOIN seasons se ON se.id = e.season_id JOIN competitions co ON co.id = se.competition_id
                            WHERE ra.id = :r""", r=race_id)
        A.guard(user, comp["code"].iloc[0] if len(comp) else "")


def _own_market(user, market_id):
    """Load a market; makers may only touch their own, admin any."""
    with get_session() as s:
        mk = s.get(m.HouseMarket, market_id)
        if mk is None:
            raise HTTPException(404)
        if user["role"] != "admin" and mk.maker_id != user["id"]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not your market.")
        return mk


@app.get("/book/quotes", response_class=HTMLResponse)
def house_book(request: Request, race_id: int | None = None, maker: str = "", msg: str = "", c=Depends(conn),
               user=allow(*PRO)):
    races = _races_for_house(c, A.hidden_comps(user))
    if race_id is None and races:
        race_id = races[0]["race_id"]
    _guard_race(c, user, race_id)
    if user["role"] == "pro":
        maker_filter = user["id"]
    elif maker == "house":
        maker_filter = None
    elif maker:
        with get_session() as s:
            mu = U.get_user(s, username=maker)
            maker_filter = mu.id if mu else -1
    else:
        maker_filter = house.ALL
    bk = house.book(c, race_id, maker_id=maker_filter)
    totals = dict(markets=len(bk), open=int((bk["status"] == "open").sum()), bets=int(bk["bets"].sum()),
                  staked=float(bk["staked"].sum()), ev=float(bk["ev"].sum()), worst=float(bk["worst"].sum()),
                  settled=float(bk["settled_pnl"].dropna().sum())) if len(bk) else None
    groups = [(kind, rows(g)) for kind, g in bk.groupby("kind", sort=False)] if len(bk) else []
    makers = rows(data.q(c, "SELECT username FROM users WHERE role IN ('pro', 'maker', 'admin') ORDER BY username"))
    return render(request, "house.html", races=races, race_id=race_id, groups=groups, totals=totals, msg=msg,
                  kinds=list(house.KINDS) + ["race_top10"], makers=makers, maker=maker)


@app.post("/book/quotes/generate", dependencies=[Depends(check_csrf)])
async def house_generate(request: Request, c=Depends(conn), user=allow(*PRO)):
    form = await request.form()
    race_id = int(form["race_id"])
    _guard_race(c, user, race_id)
    kinds = [k for k in form.getlist("kinds") if k in house.PROB_FIELD]
    top_n, spread = int(form.get("top_n", 15)), float(form.get("spread_pct", 6)) / 100
    try:
        with get_session() as s:
            created, repriced = house.generate(s, c, race_id, kinds, top_n, spread, maker_id=user["id"])
        msg = f"Created {created} markets, repriced {repriced}."
        audit(request, "markets_generate", race_id=race_id, kinds=kinds, top_n=top_n, spread=spread,
              created=created, repriced=repriced)
    except ValueError as e:
        msg = f"Error: {e}"
    nxt = form.get("next") or ""
    if nxt.startswith("/races/"):
        return RedirectResponse(f"{nxt}?msg={msg}#k-race_win", status_code=303)
    return RedirectResponse(f"/book/quotes?race_id={race_id}&msg={msg}", status_code=303)


@app.post("/book/quotes/settle-auto", dependencies=[Depends(check_csrf), allow("admin")])
def house_settle_auto(request: Request, race_id: int = Form(...), c=Depends(conn)):
    with get_session() as s:
        done = house.auto_settle(s, c, race_id)
    audit(request, "settle_auto", race_id=race_id, settled=[d[0] for d in done])
    msg = f"Auto-settled {len(done)} markets." if done else "Nothing to settle yet (Final not in the data)."
    return RedirectResponse(f"/book/quotes?race_id={race_id}&msg={msg}", status_code=303)


@app.get("/book/quotes/sheet", response_class=HTMLResponse)
def house_sheet(request: Request, race_id: int, c=Depends(conn), user=allow(*ANY)):
    _guard_race(c, user, race_id)
    bk = house.book(c, race_id, status="open")
    with get_session() as s:
        venue, cat = house.race_label(s, race_id)
    groups = [(kind, rows(g.sort_values("fair_prob", ascending=False))) for kind, g in bk.groupby("kind", sort=False)]
    return render(request, "house_sheet.html", groups=groups, venue=venue, cat=cat,
                  now=pd.Timestamp.now(tz="America/Vancouver").strftime("%a %d %b %H:%M"))


@app.get("/book/markets/{market_id}", response_class=HTMLResponse)
def house_market(request: Request, market_id: int, msg: str = "", c=Depends(conn), user=allow(*PRO)):
    _own_market(user, market_id)
    bk = house.book(c)
    mk = bk[bk["id"] == market_id]
    return render(request, "house_market.html", mk=rows(mk)[0], bets=rows(house.bets(c, market_id)), msg=msg)


@app.post("/book/markets/{market_id}/bet", dependencies=[Depends(check_csrf), allow("admin")])
def house_bet(request: Request, market_id: int, counterparty: str = Form(...), side: str = Form(...),
              stake: float = Form(...), price: str = Form(""), note: str = Form("")):
    """Admin records an offline bet for a counterparty without an account."""
    try:
        with get_session() as s:
            b = house.record_bet(s, market_id, counterparty, side, stake, float(price) if price else None, note)
        audit(request, "bet_record_offline", market_id=market_id, counterparty=counterparty, side=side,
              stake=stake, price=b.price, bet_id=b.id)
        msg = f"Recorded: {b.counterparty} {b.side} ${b.stake:.2f} at {b.price:.2f} (pays ${b.payout:.2f})."
    except ValueError as e:
        msg = f"Error: {e}"
    return RedirectResponse(f"/book/markets/{market_id}?msg={msg}", status_code=303)


@app.post("/book/markets/{market_id}/price", dependencies=[Depends(check_csrf)])
def house_price(request: Request, market_id: int, fair_pct: float = Form(...), spread_pct: float = Form(...),
                user=allow(*PRO)):
    _own_market(user, market_id)
    with get_session() as s:
        house.set_price(s, market_id, fair_pct / 100, spread_pct / 100)
    audit(request, "market_price", market_id=market_id, fair=fair_pct / 100, spread=spread_pct / 100)
    return RedirectResponse(f"/book/markets/{market_id}?msg=Repriced (manual fair value).", status_code=303)


@app.post("/book/markets/{market_id}/status", dependencies=[Depends(check_csrf)])
def house_status(request: Request, market_id: int, status_: str = Form(..., alias="status"),
                 user=allow(*PRO)):
    if status_ not in ("open", "closed"):
        raise HTTPException(400)
    _own_market(user, market_id)
    with get_session() as s:
        s.get(m.HouseMarket, market_id).status = status_
        s.commit()
    audit(request, "market_status", market_id=market_id, status=status_)
    return RedirectResponse(f"/book/markets/{market_id}", status_code=303)


@app.post("/book/markets/{market_id}/settle", dependencies=[Depends(check_csrf), allow("admin")])
def house_settle(request: Request, market_id: int, outcome: str = Form(...), note: str = Form(""),
                 confirm: str = Form("")):
    if confirm != "yes":
        raise HTTPException(400, "settlement not confirmed")
    value = {"yes": True, "no": False, "void": None}[outcome]
    with get_session() as s:
        house.settle(s, market_id, value, f"manual: {note}" if note else "manual")
    audit(request, "settle_manual", market_id=market_id, outcome=outcome, note=note)
    return RedirectResponse(f"/book/markets/{market_id}?msg=Settled {outcome.upper()}.", status_code=303)


# ---------------------------------------------------------------------------
# Takers: browse open markets from every maker, place bets, see own bets
# ---------------------------------------------------------------------------

def bet_markets(request: Request, msg: str = "", c=None):          # served at /markets for takers (views.board_page)
    """Every open Polymarket F1 market, with the account's strategy profile's current call on each (side,
    size, most to pay, heat), never our fair value or edge. Upcoming races first (listed or not yet), then
    season markets, then everything else. There is no maker here: takers trade on Polymarket."""
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import weekend_sweep as WS
    user = request.state.user
    profile = PF.of_user(c, user["id"])
    if profile is None or profile["strategy"] not in WS.TAKER_MODES:
        try:
            profile = PF.load(c, "A")
        except ValueError:
            profile = None
    view = R.basic_view_profile(profile) if R.is_basic(user) else profile       # basic: "Your picks", no strategy name
    from racinglines.web import audience as A
    from racinglines.web import sport_status as SS                  # RACINGLINES_SPORT_STATUS=1: every sport's status
    return render(request, "bet.html", msg=msg, profile=view, **polymarket_calls(c, profile),
                  sport_status=SS.status(c, A.hidden_comps(user)) if SS.enabled() else None, show_paper=False)


def polymarket_calls(c, profile, n_races=3, exchange="polymarket"):
    """Every open Polymarket F1 market (or Kalshi's, `exchange="kalshi"`) with `profile`'s current call: a taker's
    side / size / limit / heat (signals.call) or a maker's quotes (signals.maker_call). -> dict(races (the next n,
    listed or not), season, other, synced, n_markets). Never exposes fair values."""
    from datetime import timedelta as _td

    from racinglines.markets import store as MS
    from racinglines.pipelines import signals as SG
    from racinglines.pipelines import weekend_sweep as WS
    maker = bool(profile and profile["strategy"] not in WS.TAKER_MODES)
    call = SG.maker_call if maker else SG.call
    links = data.q(c, """
        SELECT ml.*, a.display_name AS athlete, e.source_key AS event_key, e.start_date, e.status AS event_status,
               coalesce(ra.format->>'event_name', e.name) AS race_name
        FROM market_links ml LEFT JOIN athletes a ON a.id = ml.athlete_id
        LEFT JOIN races ra ON ra.id = ml.race_id LEFT JOIN events e ON e.id = ra.event_id
        WHERE ml.exchange = :x AND NOT ml.closed
        ORDER BY ml.end_date NULLS LAST, ml.event_title, ml.last_price DESC NULLS LAST""", x=exchange)
    now = pd.Timestamp.now(tz="UTC")
    vol = {}
    # 24 h volume per market: by condition on Polymarket; on Kalshi a condition is the whole event, so by ticker
    vkey = "token_id" if exchange == "kalshi" else "condition_id"
    if len(links):
        tr = MS.read(c, "trades", **{"tokens" if exchange == "kalshi" else "conditions": links[vkey].dropna().unique().tolist()},
                     start=now - _td(hours=24), end=now, root=MS.root_for(exchange))
        if len(tr):
            vol = (tr["price"] * tr["size"]).groupby(tr[vkey]).sum().to_dict()
    runs, cache = {}, {}
    rows_ = []
    num = lambda v: None if v is None or pd.isna(v) else float(v)                # noqa: E731
    for link in links.to_dict("records"):
        link.update(last_bid=num(link["last_bid"]), last_ask=num(link["last_ask"]), volume=num(link["volume"]))
        if (link["last_bid"] or 0) <= 0 and (link["last_ask"] is None or link["last_ask"] >= 1):
            continue                                     # an empty book (e.g. a placeholder "Driver A"): not a bet
        kind = link["prediction"]
        ek = link["event_key"] if isinstance(link["event_key"], str) else None
        fair = None
        if profile and ek and kind.startswith("race_"):
            if ek not in runs:
                runs[ek] = SG.latest_run(c, profile, ek)
            if runs[ek]:
                fair = data.model_prob(c, link, cache, run_id=runs[ek])[0]
        mid = link["last_price"] if link["last_price"] is not None and not pd.isna(link["last_price"]) else None
        # 24 h volume from the recorded tape (race-weekend markets); None where no tape is recorded
        v24 = vol.get(link[vkey], 0.0) if kind.startswith("race_") else None
        cl = call(profile, kind, fair, mid, v24) if profile else dict(action=None, why="")
        subject = link["athlete"] or (link["params"].get("team") if isinstance(link["params"], dict) else None) \
            or link["group_title"] or link["outcome"]
        if kind == "race_h2h":
            subject = f"{link['outcome']} ({link['question'].split(': ')[-1]})"
        rows_.append(dict(link, subject=subject, mid=mid, call=cl, volume_24h=vol.get(link[vkey], 0.0),
                          heat_label=SG.HEAT_LABEL.get(cl.get("heat")) if cl.get("heat") else None))
    by_event = {}
    for r in rows_:
        by_event.setdefault(r["event_slug"], []).append(r)
    events = [dict(slug=k, title=v[0]["event_title"], end_date=v[0]["end_date"], event_key=v[0]["event_key"],
                   kind=v[0]["prediction"], rows=v, calls=sum(1 for r in v if r["call"].get("action")))
              for k, v in by_event.items()]
    sched = WS.schedule(now.year)
    upcoming = [w for _, w in sorted(sched.items()) if w["race_start"].tz_localize("UTC") > now][:n_races]
    try:
        from racinglines.models.inputs.laps_pricing import upcoming_schedule
        loc = dict(upcoming_schedule(now.year)[["round", "location"]].itertuples(index=False))
    except Exception:                                   # noqa: BLE001  (offline: names only)
        loc = {}
    races = []
    for w in upcoming:
        evs = [e for e in events if e["event_key"] == w["event_key"]]
        races.append(dict(name=w["name"], location=loc.get(int(w["event_key"].split("-")[1])), event_key=w["event_key"], start=w["race_start"], events=evs,
                          priced=bool(profile and SG.latest_run(c, profile, w["event_key"]))))
    shown = {e["slug"] for r in races for e in r["events"]}
    season = [e for e in events if e["slug"] not in shown and e["kind"] in
              ("champion", "constructors_champion", "season_wins_ge", "standings_h2h")]
    other = [e for e in events if e["slug"] not in shown and e not in season]
    synced = data.q(c, "SELECT max(synced_at) AS t FROM market_links WHERE exchange = :x", x=exchange)["t"].iloc[0]
    from racinglines.web import board as B
    recorders = B.recorder_status(c, [exchange]) if exchange != "polymarket" else None    # the Kalshi / schema recorders
    return dict(races=races, season=season, other=other, synced=synced, n_markets=len(rows_), maker=maker, exchange=exchange)


@app.post("/book/markets/{market_id}/take", dependencies=[Depends(check_csrf)])
def place_bet(request: Request, market_id: int, side: str = Form(...), stake: float = Form(...),
              quoted: float = Form(...), user=allow(*R.BASIC)):
    """Taker bets at the current quote. If the maker repriced since the page was
    loaded (quote changed), the bet is refused so the taker sees the new price."""
    race_id = None
    try:
        if side not in ("YES", "NO"):
            raise ValueError("side must be YES or NO")
        if not (0 < stake <= MAX_STAKE):
            raise ValueError(f"stake must be between $0 and ${MAX_STAKE:.0f}")
        with get_session() as s:
            mk = s.get(m.HouseMarket, market_id)
            if mk is None:
                raise ValueError("no such market")
            race_id = mk.race_id
            current = mk.yes_price if side == "YES" else mk.no_price
            if current is None or abs(current - quoted) > 1e-9:
                raise ValueError(f"the price moved (now {current}); check the new quote")
            b = house.record_bet(s, market_id, user["username"], side, stake, current, taker_id=user["id"])
        audit(request, "bet_place", market_id=market_id, side=side, stake=stake, price=b.price, bet_id=b.id,
              title=mk.title)
        msg = f"Bet placed: {side} ${stake:.2f} at {b.price:.2f} on \"{mk.title}\" (pays ${b.payout:.2f})."
    except ValueError as e:
        audit(request, "bet_rejected", market_id=market_id, side=side, stake=stake, reason=str(e))
        msg = f"Error: {e}"
    dest = f"/races/{race_id}" if race_id and not R.is_basic(user) else "/markets"
    return RedirectResponse(f"{dest}?msg={msg}", status_code=303)
