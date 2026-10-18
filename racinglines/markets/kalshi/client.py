"""
Kalshi's Trade API v2: the public market data a connector needs (no credentials), and one signed request
helper for the portfolio endpoints (markets/kalshi/trade.py). Built from Kalshi's public API documentation,
then checked against the live read-only API on 2026-09-28 (market data only; no order has been sent).
F1 roadmap, F1-9.

    series(category)                 GET /series                                      the F1 series' tickers
    events(series_ticker, status)    GET /events?with_nested_markets=true             events with their markets
    historical_markets(event)        GET /historical/markets?event_ticker=            markets settled before the cutoff
    market(ticker)                   GET /markets/{ticker}
    orderbook(ticker, depth)         GET /markets/{ticker}/orderbook                  YES bids and NO bids
    trades(ticker, min_ts)           GET /markets/trades + /historical/trades         the tape (paged by cursor)
    candlesticks(series, ticker, …)  GET /series/{s}/markets/{t}/candlesticks         price history
                                     (or /historical/markets/{t}/candlesticks)

Historical data: markets settled before Kalshi's cutoff (GET /historical/cutoff; about two months back)
leave the live endpoints: their events come back with no nested markets, /markets/{ticker} is a 404, and
their markets, trades and candlesticks are served under /historical instead.

Prices: Kalshi quoted in cents (yes_bid 56); current responses quote dollars as strings, in `<field>_dollars`
("0.5600") and, in historical candlesticks, in the plain field ("close": "0.0700"). `price()` reads any of
these and returns a probability in [0, 1].

Environment
    KALSHI_API_HOST    default https://api.elections.kalshi.com (the API lives under /trade-api/v2)
"""

import os

import httpx

from racinglines.sources import http

HOST = os.environ.get("KALSHI_API_HOST", "https://api.elections.kalshi.com")
PREFIX = "/trade-api/v2"
TIMEOUT = 20
PAGE = 200


def price(obj, field):
    """A price field as a probability: `<field>_dollars` ("0.5600") if present, else `<field>` in cents; None
    when missing or zero-sized (Kalshi reports an empty side as 0 bid / 100 ask)."""
    d = obj.get(f"{field}_dollars")
    if d not in (None, ""):
        v = float(d)
    elif isinstance(obj.get(field), str) and "." in obj[field]:     # dollars as a string ("0.0700")
        v = float(obj[field])
    elif obj.get(field) not in (None, ""):
        v = float(obj[field]) / 100
    else:
        return None
    return v


class Client:
    """Read-only Kalshi client. `transport` (an httpx transport) replaces the network, e.g. in tests."""

    def __init__(self, host=HOST, transport=None):
        self.http = httpx.Client(base_url=host.rstrip("/") + PREFIX, timeout=TIMEOUT, transport=transport)

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def get(self, path, **params):
        r = http.get(self.http, path, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()

    def _paged(self, path, key, **params):
        cursor, out = None, []
        while True:
            body = self.get(path, cursor=cursor, **params)
            out += body.get(key) or []
            cursor = body.get("cursor")
            if not cursor:
                return out

    def series(self, category=None):
        return self.get("/series", category=category).get("series") or []

    def events(self, series_ticker=None, status=None):
        return self._paged("/events", "events", series_ticker=series_ticker, status=status,
                           with_nested_markets="true", limit=PAGE)

    def historical_markets(self, event_ticker):
        return self._paged("/historical/markets", "markets", event_ticker=event_ticker, limit=PAGE)

    def market(self, ticker):
        return self.get(f"/markets/{ticker}").get("market")

    def orderbook(self, ticker, depth=10):
        body = self.get(f"/markets/{ticker}/orderbook", depth=depth)
        return body.get("orderbook_fp") or body.get("orderbook") or {}      # orderbook_fp: yes_dollars / no_dollars

    def trades(self, ticker, min_ts=None):
        return (self._paged("/markets/trades", "trades", ticker=ticker, min_ts=min_ts, limit=1000)
                + self._paged("/historical/trades", "trades", ticker=ticker, min_ts=min_ts, limit=1000))

    def candlesticks(self, series_ticker, ticker, start_ts, end_ts, period=60):
        params = dict(start_ts=int(start_ts), end_ts=int(end_ts), period_interval=period)
        try:
            body = self.get(f"/series/{series_ticker}/markets/{ticker}/candlesticks", **params)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            body = self.get(f"/historical/markets/{ticker}/candlesticks", **params)
        return body.get("candlesticks") or []
