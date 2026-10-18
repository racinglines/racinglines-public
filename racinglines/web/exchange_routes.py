"""
Exchange boards: every listed event on Polymarket, Kalshi and the schema venues (exchanges/<code>.toml), our
fair values, mirror into my book, sync; and the tape-only sports' market data.
Split out of web/app.py, which imports this module to register the routes.
"""

import math
from datetime import date
from types import SimpleNamespace

import pandas as pd
from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from racinglines import exchanges, sports
from racinglines.db.config import get_engine, get_session
from racinglines.markets import private_book as house
from racinglines.markets import venues as V
from racinglines.web import audience as A
from racinglines.web.app import ANY, PRO, allow, app, audit, check_csrf, conn, data, render, rows  # noqa: F401


@app.get("/markets/polymarket", response_class=HTMLResponse)
def pm_board(request: Request, event: str = "", show: str = "modeled", closed: int = 0, spread_pct: float = 4.0,
             msg: str = "", sport: str = "", c=Depends(conn), user=allow(*PRO)):
    return _exchange_board(request, c, user, "polymarket", event, show, closed, spread_pct, msg, sport)


def _exchange_board(request, c, user, exchange, event, show, closed, spread_pct, msg, sport=""):
    """Every listed event of one exchange (market_links.exchange), our fair values and a quote at ± spread/2.
    `sport`: a competition code (market_links.competition_id, exact match, no text/fuzzy matching) to narrow the
    list to one sport, e.g. from a Markets page "N new" link."""
    links = data.q(c, """
        SELECT ml.*, a.display_name AS athlete, co.code AS note_competition FROM market_links ml LEFT JOIN athletes a ON a.id = ml.athlete_id
        LEFT JOIN competitions co ON co.id = ml.competition_id
        WHERE ml.exchange = :x AND (CAST(:closed AS int) = 1 OR NOT ml.closed)
          AND (CAST(:ev AS text) IS NULL OR ml.event_slug = CAST(:ev AS text))
          AND (CAST(:sport AS text) IS NULL OR co.code = CAST(:sport AS text))
          AND NOT coalesce(co.code, '') = ANY(CAST(:h AS text[]))       -- sports this user doesn't see ([sport] audience)
        ORDER BY ml.end_date NULLS LAST, ml.event_title, ml.last_price DESC NULLS LAST""", closed=closed, ev=event or None,
                   sport=sport or None, x=exchange, h=list(A.hidden_comps(user)))
    mine = set(data.q(c, "SELECT market_link_id FROM house_markets WHERE maker_id = :u AND market_link_id IS NOT NULL",
                      u=user["id"])["market_link_id"].dropna().astype(int))
    from racinglines.markets import alerts
    fresh = alerts.new_links(c)
    cache, out = {}, []
    spread = spread_pct / 100
    schema = exchange in exchanges.CODES                  # a schema venue: the edge is net of its taker fee, per side
    fee = V.schema_fee(exchange) if schema else 0.0
    for link in links.to_dict("records"):
        fair, _ = data.model_prob(c, link, cache)
        mid = link["last_price"] if link["last_price"] is not None and not pd.isna(link["last_price"]) else None
        tick = link["tick_size"] or 0.01
        if schema:
            bid = None if link["last_bid"] is None or pd.isna(link["last_bid"]) else float(link["last_bid"])
            ask = None if link["last_ask"] is None or pd.isna(link["last_ask"]) else float(link["last_ask"])
            link.update(V.net_edge(fee, fair, bid, ask) if fair is not None else dict(edge_yes=None, edge_no=None, best=None, call=""))
        link.update(fair=fair, edge=(fair - mid) if fair is not None and mid is not None else None,
                    q_bid=(math.floor(round((fair - spread / 2) / tick, 6)) * tick) if fair is not None else None,
                    q_ask=(math.ceil(round((fair + spread / 2) / tick, 6)) * tick) if fair is not None else None,
                    mirrored=int(link["id"]) in mine, new=link["token_id"] in fresh,
                    subject=link["athlete"] or (link["params"].get("team") if isinstance(link["params"], dict) else None)
                    or link["group_title"])
        if show == "modeled" and fair is None:
            continue
        out.append(link)
    events = []
    for slug, g in pd.DataFrame(out).groupby("event_slug", sort=False) if out else []:
        vol = g.drop_duplicates("market_slug")["volume"].fillna(0).sum()
        events.append(dict(slug=slug, title=g["event_title"].iloc[0], end_date=g["end_date"].iloc[0], volume=vol,
                           modeled=int(g["fair"].notna().sum()), new=int(g["new"].sum()), rows=rows(g.sort_values("last_price", ascending=False,
                                                                                         na_position="last"))))
    synced = data.q(c, "SELECT max(synced_at) AS t FROM market_links WHERE exchange = :x", x=exchange)["t"].iloc[0]
    from racinglines.web import board as B
    recorders = B.recorder_status(c, [exchange]) if exchange != "polymarket" else None    # the Kalshi / schema recorders
    ctx = dict(events=events, show=show, closed=closed, spread_pct=spread_pct, msg=msg, synced=synced, event=event,
               sport=sport, sport_name=V.SPORT_NAME.get(sport, sport) if sport else "", recorders=recorders,
               note_sports=sorted({x["note_competition"] for x in out if isinstance(x.get("note_competition"), str)}))   # for the source note
    if schema:
        venue = next(v for v in V.SCHEMA_EXCHANGES if v.code == exchange)
        return render(request, "exchange.html", mode="schema", venue=venue, fee=fee, **ctx,
                      sports=[V.SPORT_NAME.get(sports.load(x)["competition"]["code"], x) for x in exchanges.sports(exchange)
                              if A.shown(user, x)])
    if exchange == "kalshi":
        return render(request, "exchange.html", mode="kalshi", venue=next(v for v in V.EXCHANGES if v.code == "kalshi"), **ctx)
    return render(request, "exchange.html", mode="pm", venue=SimpleNamespace(name="Polymarket", code="polymarket", url=""), **ctx)


@app.post("/markets/polymarket/mirror", dependencies=[Depends(check_csrf)])
def pm_mirror(request: Request, event_slug: str = Form(...), spread_pct: float = Form(4.0), c=Depends(conn),
              user=allow(*PRO)):
    with get_session() as s:
        created, repriced, skipped = house.mirror_event(s, c, event_slug, user["id"], spread_pct / 100)
    audit(request, "pm_mirror", event_slug=event_slug, spread=spread_pct / 100, created=created, repriced=repriced,
          skipped=skipped)
    msg = f"Mirrored into your book: {created} new, {repriced} repriced, {skipped} without a model price."
    return RedirectResponse(f"/markets/polymarket?spread_pct={spread_pct}&msg={msg}", status_code=303)


@app.post("/markets/polymarket/sync", dependencies=[Depends(check_csrf)])
def pm_sync(request: Request, c=Depends(conn), user=allow(*PRO)):
    from racinglines.markets import alerts
    with get_session() as s:
        stats, groups, _ = alerts.sync_and_alert(s, c, date.today().year)
    settled = []
    if user["role"] == "admin":
        with get_session() as s, get_engine().connect() as c2:
            settled = house.settle_from_exchange(s, c2)
    audit(request, "pm_sync", **stats, settled=settled)
    msg = (f"Synced {stats['events']} events / {stats['links']} outcomes ({stats['modeled']} priced by the model)."
           + (f" New: {', '.join(g['event_title'] for g in groups)}." if groups else "")
           + (f" Settled {len(settled)} mirrored markets from Polymarket." if settled else ""))
    return RedirectResponse(f"/markets/polymarket?msg={msg}", status_code=303)



def _kalshi_on():
    if not V.KALSHI_VENUE:
        raise HTTPException(404)


@app.get("/markets/kalshi", response_class=HTMLResponse, dependencies=[Depends(_kalshi_on)])
def kalshi_board(request: Request, event: str = "", show: str = "modeled", closed: int = 0, spread_pct: float = 4.0,
                 msg: str = "", sport: str = "", c=Depends(conn), user=allow(*PRO)):
    return _exchange_board(request, c, user, "kalshi", event, show, closed, spread_pct, msg, sport)


@app.post("/markets/kalshi/mirror", dependencies=[Depends(_kalshi_on), Depends(check_csrf)])
def kalshi_mirror(request: Request, event_slug: str = Form(...), spread_pct: float = Form(4.0), c=Depends(conn),
                  user=allow(*PRO)):
    with get_session() as s:
        created, repriced, skipped = house.mirror_event(s, c, event_slug, user["id"], spread_pct / 100)
    audit(request, "kalshi_mirror", event_slug=event_slug, spread=spread_pct / 100, created=created, repriced=repriced,
          skipped=skipped)
    msg = f"Mirrored into your book: {created} new, {repriced} repriced, {skipped} without a model price."
    return RedirectResponse(f"/markets/kalshi?spread_pct={spread_pct}&msg={msg}", status_code=303)


@app.post("/markets/kalshi/sync", dependencies=[Depends(_kalshi_on), Depends(check_csrf)])
def kalshi_sync(request: Request, c=Depends(conn), user=allow(*PRO)):
    from racinglines.markets.kalshi import sync as KS
    try:
        with get_session() as s:
            stats = KS.sync(s, c, date.today().year)
        msg = f"Synced {stats['events']} events / {stats['links']} outcomes ({stats['modeled']} priced by the model)."
    except Exception as e:  # noqa: BLE001  (Kalshi unreachable from this machine)
        stats, msg = None, f"Error: couldn't reach Kalshi ({type(e).__name__})."
    audit(request, "kalshi_sync", stats=str(stats))
    return RedirectResponse(f"/markets/kalshi?msg={msg}", status_code=303)


# Exchanges defined as schemas (exchanges/<code>.toml, e.g. OG.com at /markets/og): each with its own switch
# (RACINGLINES_OG_VENUE=1, off by default). The same list and mirror as Kalshi's, read-only, plus the fair-price
# indicator net of the fee (`racinglines markets --exchange <code> fair`). One route per schema code, registered at
# import (no wildcard, so /markets/linked, /markets/tapes and the legacy /markets/{id} redirect keep their paths);
# with every switch off each is a 404 and nothing else changes.
def _schema_routes(code):
    def on():
        if not exchanges.enabled(code):
            raise HTTPException(404)

    @app.get(f"/markets/{code}", response_class=HTMLResponse, dependencies=[Depends(on)], name=f"{code}_board")
    def board(request: Request, event: str = "", show: str = "modeled", closed: int = 0, spread_pct: float = 4.0,
              msg: str = "", sport: str = "", c=Depends(conn), user=allow(*PRO)):
        return _exchange_board(request, c, user, code, event, show, closed, spread_pct, msg, sport)

    @app.post(f"/markets/{code}/mirror", dependencies=[Depends(on), Depends(check_csrf)], name=f"{code}_mirror")
    def mirror(request: Request, event_slug: str = Form(...), spread_pct: float = Form(4.0), c=Depends(conn),
               user=allow(*PRO)):
        with get_session() as s:
            created, repriced, skipped = house.mirror_event(s, c, event_slug, user["id"], spread_pct / 100)
        audit(request, f"{code}_mirror", event_slug=event_slug, spread=spread_pct / 100, created=created, repriced=repriced,
              skipped=skipped)
        msg = f"Mirrored into your book: {created} new, {repriced} repriced, {skipped} without a model price."
        return RedirectResponse(f"/markets/{code}?spread_pct={spread_pct}&msg={msg}", status_code=303)

    @app.post(f"/markets/{code}/sync", dependencies=[Depends(on), Depends(check_csrf)], name=f"{code}_sync")
    def sync(request: Request, c=Depends(conn), user=allow(*PRO)):
        """Every sport the schema lists (exchanges/<code>.toml [sports.*]), the way the CLI syncs one at a time."""
        from racinglines.markets import exchange_driver as D
        stats, parts = {}, []
        try:
            with get_session() as s:
                for sport in exchanges.sports(code):
                    stats[sport] = D.sync(s, c, code, sport, date.today().year)
                    parts.append(f"{sport}: {stats[sport].get('links', 0)} markets ({stats[sport].get('modeled', 0)} priced by the model)")
            msg = "Synced " + "; ".join(parts) + "."
        except Exception as e:  # noqa: BLE001  (the exchange unreachable from this machine)
            stats, msg = None, f"Error: couldn't reach {code} ({type(e).__name__})."
        audit(request, f"{code}_sync", stats=str(stats))
        return RedirectResponse(f"/markets/{code}?msg={msg}", status_code=303)


for _code in exchanges.CODES:
    _schema_routes(_code)


# Tapes (RACINGLINES_TAPES=1, off by default): the tape-only sports' markets as market data, no model and no P&L
def _tapes_on():
    if not V.TAPES:
        raise HTTPException(404)


@app.get("/markets/tapes", response_class=HTMLResponse, dependencies=[Depends(_tapes_on)])
def tapes_page(request: Request, c=Depends(conn), user=allow(*PRO)):
    return render(request, "tapes.html", blocks=V.tape_summary(c), sports=[s["competition"].get("display_name", s["sport"]["name"])
                                                                            for s in V.tape_sports()])
