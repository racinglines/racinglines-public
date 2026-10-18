"""
Kalshi orders: maker-only, post-only limit orders on a market's YES contract, gated like Polymarket's
(markets/polymarket/trade.py). Built from Kalshi's public API documentation and tested on mocked responses
only; no order has ever been sent.

Every order is:
  - a limit order with post_only, so Kalshi rejects anything that would take liquidity;
  - checked against the current book (BUY below the best YES ask, SELL above the best YES bid) and Kalshi's
    1-cent grid (prices 0.01-0.99);
  - capped at KALSHI_MAX_ORDER_USD notional (price x contracts);
  - only sent when KALSHI_TRADING_ENABLED=true and credentials are set. Otherwise it's a dry run: the request
    is built (and signed, when a key is given) and returned, never sent.

Requests to /portfolio are signed: headers KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP (ms) and
KALSHI-ACCESS-SIGNATURE = base64(RSA-PSS-SHA256(timestamp + METHOD + path without the query)).

Environment
    KALSHI_API_KEY_ID          the API key's id
    KALSHI_PRIVATE_KEY_PATH    its RSA private key (PEM)
    KALSHI_TRADING_ENABLED     "true" to actually send orders (default: off -> dry run)
    KALSHI_MAX_ORDER_USD       per-order notional cap (default 25)

Using Kalshi is subject to its terms and to the rules where you are; make sure the account is eligible
before enabling trading.
"""

import base64
import os
import time
import uuid
from dataclasses import dataclass

from racinglines.markets.kalshi import client as K
from racinglines.markets.kalshi.sync import book_row


def trading_enabled():
    return os.environ.get("KALSHI_TRADING_ENABLED", "").lower() == "true"


def max_order_usd():
    return float(os.environ.get("KALSHI_MAX_ORDER_USD", "25"))


def sign(private_key_pem, ts_ms, method, path):
    """The KALSHI-ACCESS-SIGNATURE for one request (RSA-PSS, SHA-256, salt = digest length)."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key = serialization.load_pem_private_key(private_key_pem, password=None)
    msg = f"{ts_ms}{method.upper()}{path.split('?')[0]}".encode()
    sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                   hashes.SHA256())
    return base64.b64encode(sig).decode()


def headers(key_id, private_key_pem, method, path, ts_ms=None):
    ts_ms = int(time.time() * 1000) if ts_ms is None else ts_ms
    return {"KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
            "KALSHI-ACCESS-SIGNATURE": sign(private_key_pem, ts_ms, method, path)}


@dataclass
class Order:
    ticker: str
    action: str          # buy | sell (of YES)
    price: float         # YES price, 0.01-0.99
    count: int           # contracts

    def check(self, book):
        """Refuse an order that breaks a rule (ValueError). book: a book_row dict."""
        if self.action not in ("buy", "sell"):
            raise ValueError(f"action must be buy or sell, not {self.action!r}")
        cents = round(self.price * 100, 6)
        if cents != int(cents) or not 1 <= cents <= 99:
            raise ValueError(f"price {self.price} is off Kalshi's 1-cent grid (0.01-0.99)")
        if self.count < 1 or int(self.count) != self.count:
            raise ValueError("count must be a whole number of contracts, at least 1")
        if self.price * self.count > max_order_usd() + 1e-9:
            raise ValueError(f"notional ${self.price * self.count:.2f} is over the ${max_order_usd():.2f} cap")
        if self.action == "buy" and book.get("best_ask") is not None and self.price >= book["best_ask"]:
            raise ValueError(f"buy at {self.price} would cross the best ask {book['best_ask']} (maker only)")
        if self.action == "sell" and book.get("best_bid") is not None and self.price <= book["best_bid"]:
            raise ValueError(f"sell at {self.price} would cross the best bid {book['best_bid']} (maker only)")

    def body(self):
        return dict(ticker=self.ticker, client_order_id=str(uuid.uuid4()), side="yes", action=self.action,
                    count=int(self.count), type="limit", yes_price=int(round(self.price * 100)), post_only=True,
                    time_in_force="good_till_canceled")


def place(order, kc=None, key_id=None, private_key_pem=None):
    """Check `order` against the live book and either send it (trading enabled and credentials set) or return
    the dry run: dict(sent, request=dict(method, path, headers, json), response)."""
    kc = kc or K.Client()
    book = book_row(order.ticker, kc.orderbook(order.ticker), None)
    order.check(book)
    key_id = key_id or os.environ.get("KALSHI_API_KEY_ID")
    if private_key_pem is None and os.environ.get("KALSHI_PRIVATE_KEY_PATH"):
        with open(os.environ["KALSHI_PRIVATE_KEY_PATH"], "rb") as f:
            private_key_pem = f.read()
    path = K.PREFIX + "/portfolio/orders"
    req = dict(method="POST", path=path, json=order.body(),
               headers=headers(key_id, private_key_pem, "POST", path) if key_id and private_key_pem else {})
    if not trading_enabled():
        return dict(sent=False, request=req, response=None)
    if not req["headers"]:
        raise RuntimeError("KALSHI_TRADING_ENABLED is set but KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH are not")
    r = kc.http.post("/portfolio/orders", json=req["json"], headers=req["headers"])
    r.raise_for_status()
    return dict(sent=True, request=req, response=r.json())
