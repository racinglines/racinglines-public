"""
Linked exchange markets and Polymarket orders (admin only): link, look up, preview, place, cancel.
Split out of web/app.py, which imports this module to register the routes.
"""

import math

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from racinglines.db import models as m
from racinglines.db.config import get_session
from racinglines.markets.polymarket import trade as polymarket
from racinglines.web.app import ANY, PRO, allow, app, audit, check_csrf, conn, data, render, rows  # noqa: F401


@app.get("/markets/linked", response_class=HTMLResponse, dependencies=[allow("admin")])
def markets(request: Request, c=Depends(conn)):
    links = rows(data.market_links(c))
    for link in links:
        link["model_prob"], _ = data.model_prob(c, link)
    return render(request, "markets.html", links=links)


@app.get("/markets/linked/lookup", response_class=HTMLResponse, dependencies=[allow("admin")])
def market_lookup(request: Request, slug: str = "", query: str = "", c=Depends(conn)):
    found, results, error = [], [], None
    try:
        if slug:
            found = polymarket.lookup(slug)
        elif query:
            results = polymarket.search(query)
    except Exception as e:  # network / API errors are shown, not raised
        error = str(e)
    return render(request, "market_lookup.html", slug=slug, query=query, markets=found, results=results,
                  error=error, races=rows(data.upcoming_races(c)),
                  comps=rows(data.q(c, "SELECT id, code FROM competitions ORDER BY code")),
                  cats=rows(data.q(c, "SELECT id, competition_id, code FROM categories ORDER BY code")),
                  predictions=data.PREDICTION_KINDS)


@app.post("/markets/linked", dependencies=[Depends(check_csrf), allow("admin")])
def create_link(request: Request, market_slug: str = Form(""), question: str = Form(...), condition_id: str = Form(""),
                token_id: str = Form(...), outcome: str = Form(...), neg_risk: str = Form("false"),
                tick_size: float = Form(0.01), min_size: float = Form(5), athlete_id: int = Form(...),
                prediction: str = Form(...), race_id: str = Form(""), competition_id: int = Form(...),
                category_id: str = Form(""), invert: str = Form(""), note: str = Form("")):
    if prediction not in data.PREDICTION_COLUMNS:
        raise HTTPException(400, "unknown prediction kind")
    with get_session() as s:
        if s.get(m.Athlete, athlete_id) is None:
            raise HTTPException(400, f"no athlete with id {athlete_id}")
        link = m.MarketLink(
            market_slug=market_slug or None, question=question, condition_id=condition_id or None,
            token_id=token_id, outcome=outcome, neg_risk=neg_risk.lower() == "true", tick_size=tick_size,
            min_size=min_size, competition_id=competition_id, category_id=int(category_id) if category_id else None,
            athlete_id=athlete_id, race_id=int(race_id) if race_id else None, prediction=prediction,
            invert=bool(invert), note=note or None)
        s.add(link)
        s.commit()
        audit(request, "polymarket_link", link_id=link.id, question=question, outcome=outcome, athlete_id=athlete_id,
              prediction=prediction)
        return RedirectResponse(f"/markets/linked/{link.id}", status_code=303)


@app.post("/markets/linked/{link_id}/toggle", dependencies=[Depends(check_csrf), allow("admin")])
def toggle_link(link_id: int):
    with get_session() as s:
        link = s.get(m.MarketLink, link_id)
        link.active = not link.active
        s.commit()
    return RedirectResponse("/markets/linked", status_code=303)


def _link_context(c, link_id):
    links = data.market_links(c)
    link = links[links["id"] == link_id]
    if not len(link):
        raise HTTPException(404)
    link = link.iloc[0].to_dict()
    prob, run_id = data.model_prob(c, link)
    try:
        bk, book_error = polymarket.book(link["token_id"]), None
    except Exception as e:
        bk, book_error = None, str(e)
    suggestion = None
    if bk and prob is not None:
        tick = bk["tick_size"]
        buy = math.floor((prob - 2 * tick) / tick) * tick  # quote under fair value
        if bk["best_ask"] is not None:
            buy = min(buy, bk["best_ask"] - tick)
        sell = math.ceil((prob + 2 * tick) / tick) * tick
        if bk["best_bid"] is not None:
            sell = max(sell, bk["best_bid"] + tick)
        suggestion = dict(buy=round(buy, 4) if buy >= tick else None, sell=round(sell, 4) if sell <= 1 - tick else None)
    return link, prob, run_id, bk, book_error, suggestion


@app.get("/markets/linked/{link_id}", response_class=HTMLResponse, dependencies=[allow("admin")])
def market_detail(request: Request, link_id: int, c=Depends(conn)):
    link, prob, run_id, bk, book_error, suggestion = _link_context(c, link_id)
    past = data.orders(c)
    return render(request, "market.html", link=link, prob=prob, run_id=run_id, book=bk, book_error=book_error,
                  suggestion=suggestion, orders=rows(past[past["market_link_id"] == link_id]))


@app.post("/markets/linked/{link_id}/preview", response_class=HTMLResponse, dependencies=[Depends(check_csrf), allow("admin")])
def order_preview(request: Request, link_id: int, side: str = Form(...), price: float = Form(...),
                  size: float = Form(...), c=Depends(conn)):
    link, prob, run_id, bk, book_error, _ = _link_context(c, link_id)
    cfg = polymarket.TradingConfig.from_env()
    error, notional = None, None
    if bk is None:
        error = f"could not load the order book: {book_error}"
    else:
        try:
            notional = polymarket.check_order(cfg, side, price, size, bk)
        except polymarket.OrderRejected as e:
            error = str(e)
    edge = None
    if prob is not None:
        edge = (prob - price) if side == "BUY" else (price - prob)
    return render(request, "order_preview.html", link=link, prob=prob, book=bk, side=side, price=price, size=size,
                  notional=notional, edge=edge, error=error, run_id=run_id)


@app.post("/markets/linked/{link_id}/order", dependencies=[Depends(check_csrf), allow("admin")])
def order_submit(request: Request, link_id: int, side: str = Form(...), price: float = Form(...), size: float = Form(...),
                 confirm: str = Form(""), c=Depends(conn)):
    if confirm != "yes":
        raise HTTPException(400, "order not confirmed")
    link, prob, run_id, bk, book_error, _ = _link_context(c, link_id)
    cfg = polymarket.TradingConfig.from_env()
    rec = m.Order(market_link_id=link_id, token_id=link["token_id"], side=side, price=price, size=size,
                  order_type="GTC", post_only=True, model_prob=prob, model_run_id=run_id,
                  best_bid=bk["best_bid"] if bk else None, best_ask=bk["best_ask"] if bk else None, status="error")
    try:
        if bk is None:
            raise polymarket.OrderRejected(f"could not load the order book: {book_error}")
        rec.status, rec.exchange_order_id, resp = polymarket.place_maker_order(cfg, link["token_id"], side, price,
                                                                                size, bk)
        rec.response = _jsonable(resp)
    except polymarket.OrderRejected as e:
        rec.status, rec.error = "rejected", str(e)
    except Exception as e:
        rec.status, rec.error = "error", f"{type(e).__name__}: {e}"
    with get_session() as s:
        s.add(rec)
        s.commit()
        audit(request, "polymarket_order", order_id=rec.id, link_id=link_id, side=side, price=price, size=size,
              status=rec.status, error=rec.error)
        return RedirectResponse(f"/orders?highlight={rec.id}", status_code=303)


def _jsonable(obj):
    import json
    return json.loads(json.dumps(obj, default=str))


@app.get("/orders", response_class=HTMLResponse, dependencies=[allow("admin")])
def orders(request: Request, highlight: int | None = None, c=Depends(conn)):
    cfg = polymarket.TradingConfig.from_env()
    live, live_error = [], None
    try:
        live = polymarket.open_orders(cfg)
    except Exception as e:
        live_error = str(e)
    return render(request, "orders.html", orders=rows(data.orders(c)), live=live, live_error=live_error,
                  highlight=highlight)


@app.post("/orders/{order_id}/cancel", dependencies=[Depends(check_csrf), allow("admin")])
def order_cancel(request: Request, order_id: int):
    cfg = polymarket.TradingConfig.from_env()
    with get_session() as s:
        rec = s.get(m.Order, order_id)
        if rec is None or not rec.exchange_order_id:
            raise HTTPException(400, "no exchange order to cancel")
        try:
            resp = polymarket.cancel(cfg, rec.exchange_order_id)
            rec.status, rec.response = "cancelled", _jsonable(dict(original=rec.response, cancel=resp))
        except Exception as e:
            rec.error = f"cancel failed: {e}"
        s.commit()
        audit(request, "polymarket_cancel", order_id=order_id, status=rec.status, error=rec.error)
    return RedirectResponse(f"/orders?highlight={order_id}", status_code=303)
