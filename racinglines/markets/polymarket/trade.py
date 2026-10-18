"""
Polymarket trading client (maker-only, post-only orders; gated by env settings).

Public data (no credentials): market lookup (Gamma API) and order books (CLOB).
Trading (credentials from environment variables) goes through Polymarket's
official CLOB V2 client (py-clob-client-v2; V1-signed orders stopped working on
2026-04-28), and every order is:

  - maker-side only: GTC limit orders posted with post_only=True, so the
    exchange rejects anything that would cross the book and take liquidity;
  - checked against the current book before signing (BUY below best ask,
    SELL above best bid), the market's tick size and minimum size;
  - capped at POLYMARKET_MAX_ORDER_USD notional (price x size);
  - only sent when POLYMARKET_TRADING_ENABLED=true. Otherwise orders are built
    and signed locally as a dry run (V2 order struct and EIP-712 domain, no
    network call) and never leave this machine.

Environment variables
    POLYMARKET_PRIVATE_KEY        signer key (hex). Required for dry runs and trading.
    POLYMARKET_FUNDER             address holding the funds (proxy wallet), if not the signer
    POLYMARKET_SIGNATURE_TYPE     0 = EOA, 1 = email/Magic proxy, 2 = browser-wallet proxy (default 0)
    POLYMARKET_API_KEY / POLYMARKET_API_SECRET / POLYMARKET_API_PASSPHRASE
                                  L2 API credentials; derived from the key if unset
    POLYMARKET_TRADING_ENABLED    "true" to actually post orders (default: off -> dry run)
    POLYMARKET_MAX_ORDER_USD      per-order notional cap (default 25)
    POLYMARKET_CLOB_HOST          default https://clob.polymarket.com
    POLYMARKET_GAMMA_HOST         default https://gamma-api.polymarket.com
    POLYMARKET_CHAIN_ID           default 137 (Polygon)
    POLYMARKET_ORDER_VERSION      exchange order version for dry runs (default 2); live
                                  orders use the version the CLOB reports

Using Polymarket is subject to its terms and to the rules where you are; make
sure your account and jurisdiction are eligible before enabling trading.
"""

import json
import os
from dataclasses import dataclass
from decimal import Decimal

import httpx

from racinglines.sources import http

CLOB_HOST = os.environ.get("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
GAMMA_HOST = os.environ.get("POLYMARKET_GAMMA_HOST", "https://gamma-api.polymarket.com")
TIMEOUT = 15


# ---------------------------------------------------------------------------
# Public data
# ---------------------------------------------------------------------------

def _parse_market(m):
    outcomes = json.loads(m.get("outcomes") or "[]")
    tokens = json.loads(m.get("clobTokenIds") or "[]")
    prices = json.loads(m.get("outcomePrices") or "[]")
    return dict(
        question=m.get("question"), slug=m.get("slug"), condition_id=m.get("conditionId"),
        end_date=m.get("endDate"), neg_risk=bool(m.get("negRisk")),
        tick_size=float(m.get("orderPriceMinTickSize") or 0.01), min_size=float(m.get("orderMinSize") or 5),
        active=m.get("active"), closed=m.get("closed"),
        outcomes=[dict(outcome=o, token_id=t, price=float(p) if p not in (None, "") else None)
                  for o, t, p in zip(outcomes, tokens, prices + [None] * len(outcomes))],
    )


def lookup(slug_or_url):
    """Markets for a Polymarket market or event slug (a full polymarket.com URL works too)."""
    slug = slug_or_url.strip().rstrip("/").split("/")[-1].split("?")[0]
    with httpx.Client(base_url=GAMMA_HOST, timeout=TIMEOUT) as c:
        markets = http.get(c, "/markets", params={"slug": slug}).json()
        if not markets:
            events = http.get(c, "/events", params={"slug": slug}).json()
            markets = events[0].get("markets", []) if events else []
    return [_parse_market(m) for m in markets]


def search(query, limit=10):
    """Active events matching a text query (Gamma public search)."""
    with httpx.Client(base_url=GAMMA_HOST, timeout=TIMEOUT) as c:
        r = http.get(c, "/public-search", params={"q": query, "limit_per_type": limit, "events_status": "active"})
        r.raise_for_status()
    return [dict(title=e.get("title"), slug=e.get("slug"), end_date=e.get("endDate"))
            for e in (r.json().get("events") or [])]


def book(token_id):
    """Order book summary: best bid/ask, depth (top 5 each side), tick and min size."""
    with httpx.Client(base_url=CLOB_HOST, timeout=TIMEOUT) as c:
        r = http.get(c, "/book", params={"token_id": token_id})
        r.raise_for_status()
        b = r.json()
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", [])), reverse=True)
    asks = sorted((float(x["price"]), float(x["size"])) for x in b.get("asks", []))
    return dict(
        best_bid=bids[0][0] if bids else None, best_ask=asks[0][0] if asks else None,
        bids=bids[:5], asks=asks[:5],
        tick_size=float(b.get("tick_size") or 0.01), min_size=float(b.get("min_order_size") or 5),
        neg_risk=bool(b.get("neg_risk")), last_trade=float(b["last_trade_price"]) if b.get("last_trade_price") else None,
    )


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

@dataclass
class TradingConfig:
    private_key: str | None
    funder: str | None
    signature_type: int
    api_key: str | None
    api_secret: str | None
    api_passphrase: str | None
    enabled: bool
    max_order_usd: float
    chain_id: int
    order_version: int = 2

    @classmethod
    def from_env(cls):
        e = os.environ.get
        return cls(
            private_key=e("POLYMARKET_PRIVATE_KEY") or None,
            funder=e("POLYMARKET_FUNDER") or None,
            signature_type=int(e("POLYMARKET_SIGNATURE_TYPE", "0")),
            api_key=e("POLYMARKET_API_KEY") or None,
            api_secret=e("POLYMARKET_API_SECRET") or None,
            api_passphrase=e("POLYMARKET_API_PASSPHRASE") or None,
            enabled=e("POLYMARKET_TRADING_ENABLED", "").lower() in ("1", "true", "yes"),
            max_order_usd=float(e("POLYMARKET_MAX_ORDER_USD", "25")),
            chain_id=int(e("POLYMARKET_CHAIN_ID", "137")),
            order_version=int(e("POLYMARKET_ORDER_VERSION", "2")),
        )

    @property
    def has_key(self):
        return bool(self.private_key)

    def status(self):
        if not self.has_key:
            return "no credentials (set POLYMARKET_PRIVATE_KEY): lookup and books only"
        if not self.enabled:
            return f"dry run: orders are signed locally but not sent (max ${self.max_order_usd:.0f}/order)"
        return f"LIVE: post-only orders are sent to Polymarket (max ${self.max_order_usd:.0f}/order)"


def _client(cfg, level2):
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import ApiCreds

    client = ClobClient(CLOB_HOST, chain_id=cfg.chain_id, key=cfg.private_key,
                        signature_type=cfg.signature_type, funder=cfg.funder)
    if level2:
        if cfg.api_key and cfg.api_secret and cfg.api_passphrase:
            client.set_api_creds(ApiCreds(cfg.api_key, cfg.api_secret, cfg.api_passphrase))
        else:
            client.set_api_creds(client.create_or_derive_api_key())
    return client


def sign_order(client, token_id, side, price, size, bk, version):
    """Build and EIP-712-sign a limit order locally (no network call): the tick size and
    neg-risk flag come from the book we already checked against."""
    from py_clob_client_v2.clob_types import CreateOrderOptions, OrderArgsV2

    options = CreateOrderOptions(tick_size=_tick(bk["tick_size"]), neg_risk=bool(bk["neg_risk"]))
    return client.builder.build_order(OrderArgsV2(token_id=token_id, price=price, size=size, side=side),
                                      options, version=version)


def _tick(tick_size):
    """The client's tick-size literal ("0.01", "0.001", ...) for a float tick."""
    return format(Decimal(str(tick_size)).normalize(), "f")


def order_json(signed, owner=""):
    """The signed order as the CLOB's post-order payload (for dry-run records)."""
    from py_clob_client_v2.order_utils.model.order_data_v2 import order_to_json_v2

    return order_to_json_v2(signed, owner, "GTC", post_only=True)


class OrderRejected(Exception):
    pass


def check_order(cfg, side, price, size, bk):
    """Raise OrderRejected unless the order is a valid maker-side order within limits."""
    if side not in ("BUY", "SELL"):
        raise OrderRejected("side must be BUY or SELL")
    tick = Decimal(str(bk["tick_size"]))
    p = Decimal(str(price))
    if not (tick <= p <= 1 - tick):
        raise OrderRejected(f"price must be between {tick} and {1 - tick}")
    if p % tick != 0:
        raise OrderRejected(f"price must be a multiple of the tick size {tick}")
    if size < bk["min_size"]:
        raise OrderRejected(f"size must be at least {bk['min_size']:g} shares")
    notional = float(p) * size
    if notional > cfg.max_order_usd:
        raise OrderRejected(f"notional ${notional:.2f} exceeds POLYMARKET_MAX_ORDER_USD (${cfg.max_order_usd:.2f})")
    if side == "BUY" and bk["best_ask"] is not None and price >= bk["best_ask"]:
        raise OrderRejected(f"BUY at {price} would cross the best ask {bk['best_ask']} (not maker-side)")
    if side == "SELL" and bk["best_bid"] is not None and price <= bk["best_bid"]:
        raise OrderRejected(f"SELL at {price} would cross the best bid {bk['best_bid']} (not maker-side)")
    return notional


def place_maker_order(cfg, token_id, side, price, size, bk):
    """Validate, sign and (only if trading is enabled) post a post-only GTC order.
    Returns (status, exchange_order_id, response_dict)."""
    from py_clob_client_v2.clob_types import OrderType

    check_order(cfg, side, price, size, bk)
    if not cfg.has_key:
        raise OrderRejected("POLYMARKET_PRIVATE_KEY is not set")
    client = _client(cfg, level2=cfg.enabled)
    if not cfg.enabled:
        signed = sign_order(client, token_id, side, price, size, bk, cfg.order_version)
        return "dry_run", None, dict(signed_order=order_json(signed), order_version=cfg.order_version)
    signed = sign_order(client, token_id, side, price, size, bk, client.get_version())
    resp = client.post_order(signed, OrderType.GTC, post_only=True)
    ok = bool(resp.get("success")) if isinstance(resp, dict) else False
    return ("submitted" if ok else "rejected"), (resp or {}).get("orderID"), resp


def open_orders(cfg):
    if not (cfg.has_key and cfg.enabled):
        return []
    return _client(cfg, level2=True).get_open_orders()


def cancel(cfg, exchange_order_id):
    from py_clob_client_v2.clob_types import OrderPayload

    if not (cfg.has_key and cfg.enabled):
        raise OrderRejected("trading is not enabled")
    return _client(cfg, level2=True).cancel_order(OrderPayload(orderID=exchange_order_id))
