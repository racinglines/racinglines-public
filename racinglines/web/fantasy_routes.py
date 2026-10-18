"""
The fantasy pages (owner, 2026-10-12 08:40Z, staging only): an account's F$ ("1k balance they can see"), trading by
hand ("first-cut manual trading") next to following a strategy, and the leaderboard.

    /trade                 the tradable F1 markets with their recorded best prices; ?link=<id> opens one market's
                           book and the order form (the alpha's Trade tab), with what the form's order would do if
                           placed now (fantasy/exchange.quote; ?shares=&limit_price= requote it)
    POST /fantasy/orders   one IOC paper order (fantasy/exchange.place), then back to the market with the result;
                           with preview=1 nothing is placed: back to the form, requoted with its values
    /history               the alpha's History tab: your F$, every trade (by hand, open and settled, and by your
                           strategy), F$ through the run, and the leaderboard (ROI x sqrt(trades), fantasy/ledger.ranked)
    /fantasy, /fantasy/trade, /leaderboard   the first cut's addresses, redirected to the two above

All of it is off unless fantasy/ledger.on() (RACINGLINES_FANTASY=1, or staging with the switch unset): 404 otherwise.
Before `racinglines users setup` has created the trading tables the pages say so and nothing is written. F$ only:
nothing here reaches an exchange (fantasy/exchange.py). Split out of web/app.py, which imports this module.
"""

import logging
import uuid
from urllib.parse import quote, urlencode

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import text

from racinglines.fantasy import exchange as FX
from racinglines.fantasy import ledger as FL
from racinglines.web import demo
from racinglines.web.app import ANY, allow, app, audit, check_csrf, get_engine, render

log = logging.getLogger(__name__)
HIDDEN = ("polymarket-takers",)          # the system account (web/accounts.RESERVED); demo accounts are added at call time


def _gate():
    if not FL.on():
        raise HTTPException(404)


def _ready():
    from racinglines.web import accounts as ACC
    with get_engine().connect() as c:
        return ACC.ready(c) and FL.ready(c)


def _books():
    return FX.books_engine(get_engine())


def _settle():
    """Pay out resolved hand trades before a page shows balances (FX.settle; staging has no timers to do it). A
    failure is logged and the page still loads with the positions as they were."""
    try:
        FX.settle(get_engine(), _books())
    except Exception:                                    # noqa: BLE001 — never block the page on settlement
        log.exception("fantasy settle failed")


def _strategy_trades(c, user_id):
    """The markets the followed strategy traded for this account (paper_positions, the alpha leaves the private book
    out), newest weekend first, with each one's P&L by the positions page's rule. -> [dict]"""
    rs = c.execute(text("""
        SELECT p.event_key, p.kind, p.subject, p.market_key, p.yes_shares, p.no_shares, p.cash, p.mark, p.outcome,
               p.updated_at, coalesce(ra.format->>'event_name', e.name) AS event_name
        FROM paper_positions p LEFT JOIN races ra ON ra.id = p.race_id LEFT JOIN events e ON e.id = ra.event_id
        WHERE p.user_id = :u AND p.venue <> 'private' AND (p.yes_shares > 0 OR p.no_shares > 0 OR p.cash <> 0)
        ORDER BY p.event_key DESC, p.updated_at DESC LIMIT 200"""), dict(u=user_id)).mappings().all()
    out = []
    for r in rs:
        r = dict(r)
        y = (1.0 if r["outcome"] else 0.0) if r["outcome"] is not None else r["mark"]
        r["pnl"] = None if y is None else round(r["cash"] + r["yes_shares"] * y + r["no_shares"] * (1 - y), 2)
        r["side"] = "yes" if r["yes_shares"] > 0 else ("no" if r["no_shares"] > 0 else None)
        r["state"] = "settled" if r["outcome"] is not None else ("open" if r["side"] else "closed")
        out.append(r)
    return out


@app.get("/history", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def history_page(request: Request, msg: str = ""):
    _gate()
    me = request.state.user
    if not _ready():
        return render(request, "history.html", ready=False, msg=msg, rows=[])
    _settle()
    from racinglines.pipelines import profiles as PF
    from racinglines.web.viz import line_chart
    with get_engine().connect() as c:
        q = FX.quotes(_books(), FL.open_tokens(c, me["id"]))
        s = FL.summary(c, me["id"], quotes=q)
        every = FL.positions(c, me["id"], open_only=False, quotes=q)
        pos = [p for p in every if p["status"] == "open" and p["shares"] > 0]
        settled = [dict(p, pnl=round(p["payout"] - p["cost"], 2) if p["payout"] is not None else None)
                   for p in every if p["status"] != "open"]             # paid out by exchange.settle (outcome, payout)
        recent = FL.orders(c, me["id"], limit=100)
        strat = _strategy_trades(c, me["id"])
        curve = FL.manual_curve(c, me["id"], quotes=q)
        profile = PF.of_user(c, me["id"])
        board = FL.ranked(c, hide=(*HIDDEN, *demo.DEMO_USERS), quotes=FX.quotes(_books(), FL.open_tokens(c)))
    mine = next((r for r in board if r["username"] == me["username"]), None)
    chart = (line_chart({"hand": curve}, labels={"hand": "F$ by hand"}, include_zero=False,
                        markers=[(curve[-1][0], "now")]) if len(curve) >= 2 else None)   # the last point is at mark
    return render(request, "history.html", ready=True, s=s, positions=pos, settled=settled, orders=recent, strat=strat, chart=chart,
                  profile=profile, mine=mine, rows=board, min_trades=FL.MIN_TRADES, msg=msg,
                  is_demo_user=demo.is_demo(me), reasons=FX.TEXT, note_sport=FX.SPORT, note_venues=list(FX.EXCHANGES))


@app.get("/fantasy", dependencies=[allow(*ANY)])
def fantasy_home(request: Request):
    _gate()
    return RedirectResponse("/history", status_code=303)


@app.get("/fantasy/trade", dependencies=[allow(*ANY)])
def fantasy_trade_old(request: Request):
    _gate()
    qs = request.url.query
    return RedirectResponse("/trade" + (f"?{qs}" if qs else ""), status_code=303)


@app.get("/trade", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def fantasy_trade(request: Request, link: int = 0, q: str = "", msg: str = "", action: str = "buy", side: str = "yes",
                  shares: str = "", limit_price: str = ""):
    _gate()
    me = request.state.user
    if not _ready():
        return render(request, "fantasy_trade.html", ready=False, msg=msg)
    _settle()
    eng = get_engine()
    q = q.strip()[:60]
    with eng.connect() as c:
        markets = FX.markets(c, q)
        quotes = FX.quotes(_books(), [m["token_id"] for m in markets])
        for m in markets:
            b, a, ts = quotes.get(m["token_id"], (None, None, None))
            m.update(bid=b, ask=a, quoted=ts)
        markets = [m for m in markets if m["quoted"] is not None]          # recorded in the last day
        sel = book = pos = quote = None
        action = action if action in ("buy", "sell") else "buy"
        side = side if side in ("yes", "no") else "yes"
        if link:
            sel = c.execute(text("""SELECT id, exchange, token_id, prediction AS kind, question, group_title, outcome,
                                           event_title FROM market_links WHERE id = :i"""), dict(i=link)).mappings().first()
            if sel is not None:
                sel = dict(sel)
                full = FX._link(c, sel["id"])
                sel["sport_ok"], sel["sport"] = FX.sport_ok(full), full["sport"]
                sel["closes_at"] = FX.closes_for(c, [full])[full["id"]]
                sel["season"] = full["prediction"] in FX.SEASON_KINDS
            if sel is not None and sel["sport_ok"]:
                book = FX.latest_book(_books(), sel["token_id"])
                if book:
                    nb, na = FX.yes_to_no(book["bids"], book["asks"])
                    book.update(no_bids=nb, no_asks=na)
                held = [p for p in FL.positions(c, me["id"], open_only=True) if p["market_link_id"] == sel["id"]]
                pos = held[0] if held else None
            if book and not demo.is_demo(me):
                # the form's values (the page's defaults unless the query names them) and what that order would do now
                if not shares.strip():
                    shares = f"{pos['shares']:g}" if pos and action == "sell" else "10"
                if not limit_price.strip():
                    lv = (book["asks"] if side == "yes" else book["no_asks"]) if action == "buy" else (book["bids"] if side == "yes" else book["no_bids"])
                    limit_price = f"{lv[0][0]:.2f}" if lv else "0.50"
                quote = FX.quote(c, me, sel["id"], side, action, shares, limit_price, get_book=lambda tok: book)
        cash = FL.available(c, me["id"])
    events = []
    for m in markets:
        if not events or events[-1]["title"] != (m["event_title"] or "Season"):
            events.append(dict(title=m["event_title"] or "Season", start=m["start_date"], markets=[]))
        events[-1]["markets"].append(m)
    return render(request, "fantasy_trade.html", ready=True, events=events, n=len(markets), sel=sel, book=book,
                  pos=pos, cash=cash, q=q, msg=msg, action=action, side=side, shares=shares.strip()[:12],
                  limit_price=limit_price.strip()[:8], quote=quote, client_key=uuid.uuid4().hex, max_stake=FX.MAX_STAKE,
                  max_age=FX.MAX_AGE_MIN, is_demo_user=demo.is_demo(me), note_sport=FX.SPORT, note_venues=list(FX.EXCHANGES))


@app.post("/fantasy/orders", dependencies=[Depends(check_csrf), allow(*ANY)])
def fantasy_order(request: Request, market_link_id: str = Form(""), side: str = Form(""), action: str = Form(""),
                  shares: str = Form(""), limit_price: str = Form(""), client_key: str = Form(""), preview: str = Form("")):
    _gate()
    me = request.state.user
    back = f"/trade?link={market_link_id}" if market_link_id.isdigit() else "/trade"
    if preview and market_link_id.isdigit():                    # the Preview button: requote the form, place nothing
        return RedirectResponse(f"{back}&{urlencode(dict(action=action, side=side, shares=shares.strip()[:12], limit_price=limit_price.strip()[:8]))}",
                                status_code=303)
    if not _ready():
        msg = "Error: fantasy trading isn't set up yet."
    elif not FX.rate_ok(me["id"]):
        audit(request, "fantasy_order_rejected", reason="rate_limited", link=market_link_id)
        msg = "Error: " + FX.TEXT["rate_limited"]
    else:
        r = FX.place(get_engine(), me, market_link_id, side, action, shares, limit_price, client_key=client_key,
                     demo=demo.is_demo(me))
        if not r["ok"]:
            msg = "Error: " + r["message"]
        else:
            got = sum(s for _, s in r["fills"])
            avg = sum(p * s for p, s in r["fills"]) / got
            msg = (f"{'Bought' if action == 'buy' else 'Sold'} {got:g} {side.upper()} at an average of {avg:.2f}"
                   + (f", fee F${r['fee']:.2f}" if r["fee"] else "")
                   + (" (part of your order: the rest wasn't on the book at your limit)" if r["status"] == "partial" else "")
                   + (" (already placed)" if r.get("repeat") else "") + ".")
    sep = "&" if "?" in back else "?"
    return RedirectResponse(f"{back}{sep}msg={quote(msg)}", status_code=303)


@app.get("/leaderboard", dependencies=[allow(*ANY)])
def leaderboard_page(request: Request):
    _gate()
    return RedirectResponse("/history#leaderboard", status_code=303)
