"""
The generic exchange driver: reads an exchange schema (exchanges/<code>.toml, racinglines/exchanges.py) and does what
markets/kalshi/{client,sync}.py do for Kalshi, for any venue with a plain JSON market-data API: sync its markets into
market_links (exchange = <code>), keep the quotes fresh, and store the tape, price history and order books in the
shared tables (market_trades, market_price_history, market_book_snapshots), archived per exchange like the others.

Nothing here knows a venue: hosts, endpoints, field paths, price units, fees and how a market is named come from
the schema. Read-only by construction: the driver only issues GETs of public market data, and there is no order code.

    Client(schema)                         GET an endpoint; paged / batched calls as the schema says
    sync(session, conn, code, sport)       events -> instruments -> market_links (+ quotes), one call
    settle(session, conn, code, sport)     the settlement feed -> resolved_yes on closed links (who won)
    fetch_trades / fetch_history / snapshot_books                      the tape, minute prices and books
    fair_report(conn, code, sport)         the model's fair price beside the exchange's quote, net of the fee

Prediction kinds come from the schema's per-sport rules (a regex on the market's contract name); the subject is
matched to a driver or team by the sport's resolver (F1: markets/polymarket/sync.Resolver). A sport listed without
`modeled = true` is tape-only: every link is `unmodeled`, under that sport's own competition.
"""

import re
from datetime import date, datetime, timezone

import httpx
import pandas as pd
from sqlalchemy import select, text

from racinglines import exchanges as EX
from racinglines import progress
from racinglines.db import models as m
from racinglines.markets import identity
from racinglines.markets.kalshi import sync as KS
from racinglines.sources import http

# ----- values -----

def when(v, unit):
    """A time from the exchange's number (ms / ns / s since the epoch) or ISO string, as aware UTC; None if missing."""
    if v in (None, ""):
        return None
    if unit == "iso":
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    div = {"s": 1, "ms": 1e3, "ns": 1e9}[unit]
    return datetime.fromtimestamp(float(v) / div, tz=timezone.utc)


def utc(t):
    """A pandas Timestamp in UTC (a naive one is taken as UTC)."""
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def epoch(t, unit):
    """A time as the exchange's integer (the inverse of `when`)."""
    mult = {"s": 1, "ms": 1000, "ns": 1_000_000_000}[unit]
    return int(utc(t).timestamp() * mult)


def price(v):
    """A price as a probability: the schema's prices are dollars 0..1 as strings; None for missing or empty."""
    if v in (None, ""):
        return None
    return float(v)


def quote(bid, ask, last):
    """(bid, ask, mid): an empty side is None; a dead book's synthetic 0.50 midpoint is treated as no quote,
    but genuine last prices still survive."""
    bid = None if bid in (None, 0) else bid
    ask = None if ask is None or ask >= 1.0 else ask
    if bid is None and ask is None:
        if last is None or abs(last - 0.5) > 1e-9:
            return None, None, last or None
        return None, None, None
    mid = (bid + ask) / 2 if bid is not None and ask is not None else last
    return bid, ask, mid


# ----- the client -----

class Client:
    """Read-only client for one exchange schema. `transport` (an httpx transport) replaces the network in tests."""

    def __init__(self, code, host=None, transport=None):
        import os
        self.code, self.schema = code, EX.load(code)
        api = self.schema["api"]
        host = host or os.environ.get(api.get("host_env", ""), "") or api["host"]
        http.HOST_INTERVAL.setdefault(httpx.URL(host).host, 1 / api.get("max_per_second", 4))
        self.http = httpx.Client(base_url=host.rstrip("/") + api.get("prefix", ""), timeout=api.get("timeout", 20),
                                 transport=transport)

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _coinbase_call(self, endpoint, **params):
        """Temporary public GraphQL fallback for Coinbase's event and market pages. It does not include a trading API."""
        ep = self.schema["endpoints"][endpoint]
        if endpoint in {"events", "instruments"}:
            event_ticker = params.get("eventTicker") or _coinbase_default_event_ticker()
            payload = {
                "operationName": ep["operationName"],
                "variables": {
                    "eventTicker": event_ticker,
                    "productsSortOption": params.get("productsSortOption", "PRICE"),
                    "productsSortOrder": params.get("productsSortOrder", "DESC"),
                    "countryCode": params.get("countryCode", "US"),
                    "locale": params.get("locale", "en"),
                },
                "extensions": {"persistedQuery": {"version": 1, "sha256Hash": ep["sha256Hash"]}},
            }
        elif endpoint == "tickers":
            product_ids = list(params.get("product_ids") or params.get("ids") or params.get("liveProductIds") or [])
            if not product_ids:
                return {"data": {"retailBrokerage": {"productsStats": []}}}
            payload = {
                "operationName": ep["operationName"],
                "variables": {"liveProductIds": product_ids, "skipLiveProductStats": False},
                "extensions": {"persistedQuery": {"version": 1, "sha256Hash": ep["sha256Hash"]}},
            }
        elif endpoint == "history":
            product_ids = list(params.get("product_ids") or params.get("ids") or params.get("chartedMarketProductIds") or [])
            if not product_ids:
                return {"data": {"retailBrokerage": {"priceChartsHour": []}}}
            payload = {
                "operationName": ep["operationName"],
                "variables": {
                    "chartedMarketProductIds": product_ids,
                    "includeHour": True,
                    "includeDay": True,
                    "includeWeek": True,
                    "includeAll": True,
                },
                "extensions": {"persistedQuery": {"version": 1, "sha256Hash": ep["sha256Hash"]}},
            }
        else:
            return {"data": {"retailBrokerage": {"productsStats": []}}}
        r = http.post(self.http, ep["path"], json=payload)
        r.raise_for_status()
        body = r.json()
        if body.get("errors"):
            raise RuntimeError(f"{self.code} {endpoint}: {body['errors']}")
        if endpoint in {"events", "instruments"}:
            event_ticker = payload["variables"]["eventTicker"]
            products = EX.dig(body, "data.retailBrokerage.instrumentGroup.products") or []
            for product in products:
                product["eventTicker"] = event_ticker
                product.setdefault("event_title", event_ticker)
                product.setdefault("tradable", True)
                product.setdefault("instrumentName", product.get("productId", ""))
        return body

    def call(self, endpoint, **params):
        """The parsed response of a schema endpoint; raises on HTTP errors or the exchange's own error code."""
        if self.code == "coinbase":
            return self._coinbase_call(endpoint, **params)
        ep, api = self.schema["endpoints"][endpoint], self.schema["api"]
        q = {**ep.get("params", {}), **{k: v for k, v in params.items() if v is not None}}
        r = http.get(self.http, ep["path"], params=q)
        r.raise_for_status()
        body = r.json()
        if EX.dig(body, api["ok_path"]) != api["ok_value"]:
            raise RuntimeError(f"{self.code} {endpoint}: {EX.dig(body, api.get('error_path', 'message')) or body}")
        return body

    def rows(self, endpoint, **params):
        body = self.call(endpoint, **params)
        return EX.dig(body, self.schema["endpoints"][endpoint]["data"]) or []

    def pages(self, endpoint, **params):
        """Each page's rows of a cursor-paged endpoint, one list per call, until a page names no next cursor. A
        caller that has what it needs stops iterating, and no further page is asked for."""
        ep, cursor = self.schema["endpoints"][endpoint], None
        while True:
            body = self.call(endpoint, **{ep["cursor_param"]: cursor}, **params)
            yield EX.dig(body, ep["data"]) or []
            cursor = EX.dig(body, ep["next"])
            if not cursor:
                return

    def paged(self, endpoint, **params):
        """Every row of a cursor-paged endpoint."""
        return [r for page in self.pages(endpoint, **params) for r in page]

    def batched(self, endpoint, values, **params):
        """Every row of an endpoint that takes comma-separated ids, `batch_size` at a time (and paged, if it pages)."""
        if self.code == "coinbase" and endpoint == "tickers":
            return self.rows(endpoint, product_ids=list(values), **params)
        ep, out = self.schema["endpoints"][endpoint], []
        values = list(values)
        for i in range(0, len(values), ep.get("batch_size", 25)):
            part = {ep["batch_param"]: ",".join(values[i:i + ep.get("batch_size", 25)]), **params}
            out += self.paged(endpoint, **part) if "cursor_param" in ep else self.rows(endpoint, **part)
        return out

    def book(self, instrument):
        return self.call("book", **{self.schema["endpoints"]["book"]["instrument_param"]: instrument})

    def window(self, endpoint, instrument, start, end=None):
        """A time-windowed endpoint (trades, history) for one instrument; the window is clipped to the schema's
        `window_days` (the exchange keeps about that much)."""
        ep = self.schema["endpoints"][endpoint]
        now = pd.Timestamp.now("UTC")                  # read once: a second read for `end` a millisecond later
        start = utc(start)                             # put the span over the cap (OG.com: "at most 2678400000 ms")
        start = max(start, now - pd.Timedelta(days=ep["window_days"]) + pd.Timedelta(seconds=1))
        p = {ep["instrument_param"]: instrument, ep["start_param"]: epoch(start, ep["start_unit"])}
        if ep.get("end_param"):
            p[ep["end_param"]] = epoch(utc(end) if end else now, ep["start_unit"])
        return self.rows(endpoint, **p)


# ----- discovery and links -----

def sport_cfg(schema, sport):
    if sport not in schema.get("sports", {}):
        raise ValueError(f"{schema['exchange']['code']} lists no {sport!r} (exchanges/{schema['exchange']['code']}.toml [sports.*])")
    return schema["sports"][sport]


def discover(client, sport):
    """(events, instruments) the exchange lists for `sport`: events whose symbol starts with one of the schema's
    prefixes, then their instruments. Settled or expired instruments have left the listing."""
    if client.code == "coinbase":
        event_ticker = _coinbase_default_event_ticker()
        event = {"eventTicker": event_ticker, "event_title": "Bahrain Grand Prix", "id": event_ticker}
        products = client.rows("instruments", eventTicker=event_ticker)
        return [event], products
    cfg, f = sport_cfg(client.schema, sport), client.schema["fields"]
    prefixes = tuple(cfg["event_prefixes"])
    events = [e for e in client.paged("events") if str(EX.dig(e, f["event"]["id"]) or "").startswith(prefixes)]
    ids = [EX.dig(e, f["event"]["id"]) for e in events]
    return events, client.batched("instruments", ids) if ids else []


def _coinbase_f1_subject(inst):
    """A Coinbase F1 product's driver code from its productId (e.g. KXF1RACE-BAH26-ANT-KALSHI -> ANT)."""
    product = str(EX.dig(inst, "productId") or EX.dig(inst, "id") or "")
    if not product:
        return ""
    bits = product.upper().split("-")
    if bits and bits[0].startswith("KXF1") and product.upper().endswith("-KALSHI") and len(bits) >= 4:
        return bits[-2]
    return ""


class CoinbaseF1Resolver:
    """Resolve_PRODUCT_ID driver abbreviations like ANT, NOR, LEC to the driver's athlete_id in the current F1 data."""

    def __init__(self, conn):
        self.conn = conn

    def driver(self, subject):
        code = str(subject or "").upper().strip()
        if not code:
            return None
        row = self.conn.execute(text("""
            SELECT r.athlete_id
            FROM results r
            JOIN rounds ro ON ro.id = r.round_id
            JOIN races ra ON ra.id = ro.race_id
            JOIN events e ON e.id = ra.event_id
            JOIN seasons s ON s.id = e.season_id
            WHERE upper(coalesce(r.extra->>'abbreviation', '')) = :code
            ORDER BY s.year DESC, e.start_date DESC, r.id DESC
            LIMIT 1
        """), dict(code=code)).scalar()
        return int(row) if row is not None else None


def classify(schema, sport, contract):
    """(prediction kind, subject kind) for a market's contract name, from the sport's rules; else unmodeled."""
    cfg = sport_cfg(schema, sport)
    if schema.get("exchange", {}).get("code") == "coinbase" and sport == "f1":
        if re.search(r"^KXF1(?:RACE)?-[A-Z0-9]+-[A-Z]{2,5}-KALSHI$", (contract or "").upper()):
            return "race_win", "driver"
    if not cfg.get("modeled"):
        return "unmodeled", None
    for rule in cfg.get("rules", []):
        if re.search(rule["contract"], contract or "", re.I):
            return rule["kind"], rule.get("subject")
    return "unmodeled", None


def link_rows(schema, sport, instruments, tickers, resolver=None):
    """market_links values (with token_id) for every instrument. tickers: {instrument: ticker row}, for the quotes.
    resolver: driver(name), team(name) -> id / team key, only for a modeled sport."""
    f, code = schema["fields"], schema["exchange"]["code"]
    fi, ft = f["instrument"], f["ticker"]
    rows = []
    for inst in instruments:
        tok = EX.dig(inst, fi["id"])
        subject = EX.dig(inst, fi["subject"]) or ""
        contract = EX.dig(inst, fi["contract"]) or EX.dig(inst, fi["event_title"]) or ""
        if schema.get("exchange", {}).get("code") == "coinbase" and sport == "f1" and not subject:
            subject = _coinbase_f1_subject(inst)
        elif schema.get("exchange", {}).get("code") == "coinbase" and sport == "f1" and subject == contract:
            subject = _coinbase_f1_subject(inst) or subject
        kind, who = classify(schema, sport, contract)
        athlete_id, params, race_id = None, {}, None
        if kind != "unmodeled":
            if who == "driver":
                athlete_id = resolver.driver(subject)
            elif who == "team":
                params["team"] = resolver.team(subject)
            if athlete_id is None and not params.get("team"):
                kind = "unmodeled"                       # a subject we can't match to our drivers or teams
        if kind.startswith("race_") and hasattr(resolver, "race"):
            # a race market names its Grand Prix in the event title ("Bahrain Grand Prix 2026"); OG.com's expiry is
            # weeks after the race, so the race is the one of that name held closest to today
            race_id, race_key = resolver.race(KS.gp_name(EX.dig(inst, fi["event_title"]) or contract), near=date.today())
            if race_id is None:
                kind = "unmodeled"
            else:
                params["event_key"] = race_key
        tk = tickers.get(tok) or {}
        bid, ask, mid = quote(price(EX.dig(tk, ft["bid"])), price(EX.dig(tk, ft["ask"])), price(EX.dig(tk, ft["last"])))
        vol = EX.dig(tk, ft["volume"])
        tick = price(EX.dig(inst, fi["tick"]))
        rows.append(dict(
            token_id=tok, exchange=code, market_slug=tok,
            question=contract or EX.dig(inst, fi["title"]) or tok, condition_id=EX.dig(inst, fi["event"]),
            outcome=subject or "Yes", neg_risk=False, tick_size=tick or 0.01, min_size=1.0,
            athlete_id=athlete_id, race_id=race_id, prediction=kind, invert=False,
            params={k: v for k, v in dict(params, **({"contract": contract} if contract else {})).items() if v is not None},
            event_slug=EX.dig(inst, fi["event"]), event_title=EX.dig(inst, fi["event_title"]) or contract,
            group_title=subject or None, last_bid=bid, last_ask=ask, last_price=mid,
            volume=float(vol) if vol not in (None, "") else None,
            end_date=when(EX.dig(inst, fi["expiry"]), fi.get("expiry_unit", "ms")),
            closed=EX.dig(inst, fi["tradable"]) is False, resolved_yes=None,
            active=EX.dig(inst, fi["tradable"]) is not False))
    return rows


def _tickers(client, ids):
    ft = client.schema["fields"]["ticker"]
    return {EX.dig(t, ft["instrument"]): t for t in client.batched("tickers", ids)}


def sync(session, conn, code, sport="f1", year=2026, client=None, resolver=None):
    """Upsert one market_links row per instrument the exchange lists for `sport`, with its current quote. Idempotent;
    links of instruments that have left the listing are marked closed. Returns stats."""
    from racinglines.markets.kalshi.sync import competition
    schema = EX.load(code)
    comp, cat = competition(session, sport)
    client = client or Client(code)
    events, instruments = discover(client, sport)
    ids = [EX.dig(i, schema["fields"]["instrument"]["id"]) for i in instruments]
    tickers = _tickers(client, ids) if ids else {}
    if resolver is None and sport_cfg(schema, sport).get("modeled"):
        from racinglines.markets.polymarket.sync import Resolver
        resolver = Resolver(conn, year)
    modeled = sport_cfg(schema, sport).get("modeled") or (code == "coinbase" and sport == "f1")
    if resolver is None and modeled:
        if code == "coinbase" and sport == "f1":
            resolver = CoinbaseF1Resolver(conn)
        else:
            from racinglines.markets.polymarket.sync import Resolver
            resolver = Resolver(conn, year)
    rows = link_rows(schema, sport, instruments, tickers, resolver)
    who = identity.linker(sport, conn) if not modeled else None
    if who:                                                          # a tape-only sport with a resolver: driver, race, kind
        who.fill(rows)
        identity.promote(sport, rows)            # prediction = params.kind where [markets] kinds files it, identified
    now = datetime.now(timezone.utc)
    stats = dict(events=len(events), links=0, modeled=0, unmatched=0, new=0, closed=0)
    if who:
        stats["identity"] = dict(who.counts)
    for row in rows:
        tok = row.pop("token_id")
        values = dict(row, competition_id=comp.id, category_id=cat.id, synced_at=now)
        link = session.scalars(select(m.MarketLink).filter_by(exchange=code, token_id=tok)).first()
        if link is None:
            session.add(m.MarketLink(token_id=tok, first_seen_at=now, **values))
            stats["new"] += 1
        else:
            if values.get("resolved_yes") is None:          # the listing never carries an outcome: keep settle()'s
                values.pop("resolved_yes", None)
            kept = {k: v for k, v in (link.params or {}).items() if k in SETTLE_KEYS}
            if kept:
                values["params"] = {**(values.get("params") or {}), **kept}
            for k, v in values.items():
                setattr(link, k, v)
        stats["links"] += 1
        stats["modeled" if row["prediction"] != "unmodeled" else "unmatched"] += 1
    live, prefixes = set(ids), tuple(sport_cfg(schema, sport)["event_prefixes"])
    for link in session.scalars(select(m.MarketLink).filter_by(exchange=code, closed=False, competition_id=comp.id)):
        if link.token_id not in live and (link.condition_id or "").startswith(prefixes):      # left the listing
            link.closed, link.active = True, False
            stats["closed"] += 1
    session.commit()
    return stats


# ----- settlements: who won -----

# The params keys settle() writes on a link. sync() rewrites params from the listing, so it carries these over.
SETTLE_KEYS = ("settlement", "settlement_value", "settled_at", "settle_scanned_to")


def settlement(schema, row):
    """(token, outcome, value, settled_at) of one settlement-feed row (schema fields.settlement). outcome: True when a
    YES contract paid 1, False when it paid 0, None for anything between (OG.com's 0.50: a void). None for a row
    with no instrument or no value."""
    fs = schema["fields"]["settlement"]
    tok, v = EX.dig(row, fs["instrument"]), price(EX.dig(row, fs["value"]))
    if tok in (None, "") or v is None:
        return None
    outcome = True if abs(v - 1) < 1e-9 else False if abs(v) < 1e-9 else None
    return str(tok), outcome, v, when(EX.dig(row, fs["ts"]), fs.get("ts_unit", "ms"))


def scan_settlements(client, tokens, since, max_pages=None):
    """Read the exchange's settlement feed forward from `since`, keeping the rows of `tokens`. Stops when every token
    is found, when the feed ends (a page with no next cursor) or after `max_pages` pages (default the endpoint's
    `max_pages`), so a pass is bounded. -> (found {token: (outcome, value, settled_at)}, stats: pages, scanned,
    exhausted (the feed ended), last_ts (the latest settle time read))."""
    ep, schema = client.schema["endpoints"]["settlements"], client.schema
    fs = schema["fields"]["settlement"]
    want, found = set(tokens), {}
    st = dict(pages=0, scanned=0, exhausted=False, last_ts=None)
    cap = max_pages or ep.get("max_pages", 500)
    for rows in client.pages("settlements", **{ep["start_param"]: epoch(since, ep["start_unit"])}):
        st["pages"] += 1
        st["scanned"] += len(rows)
        for r in rows:
            ts = when(EX.dig(r, fs["ts"]), fs.get("ts_unit", "ms"))
            if ts is not None and (st["last_ts"] is None or ts > st["last_ts"]):
                st["last_ts"] = ts
            s = settlement(schema, r)
            if s and s[0] in want and s[0] not in found:
                found[s[0]] = s[1:]
        progress.update(item=f"settlement page {st['pages']} of at most {cap}: {st['scanned']} rows read, "
                             f"{len(found)} of {len(want)} links found")
        if len(found) == len(want) or st["pages"] >= cap:
            break
    else:
        st["exhausted"] = True
    return found, st


def pending(session, code, competition_id):
    """The exchange's closed links of a competition still waiting for an outcome: resolved_yes empty and no void
    recorded (params.settlement)."""
    q = select(m.MarketLink).filter_by(exchange=code, closed=True, competition_id=competition_id) \
        .where(m.MarketLink.resolved_yes.is_(None)).order_by(m.MarketLink.token_id)
    return [lk for lk in session.scalars(q) if not (lk.params or {}).get("settlement")]


def scan_from(link, now=None):
    """Where the feed scan starts for one link: an hour before where the last pass stopped looking for it
    (params.settle_scanned_to), else a day before the last sync that saw it listed (synced_at, else first_seen_at;
    an instrument leaves the listing when it settles, so its settlement comes after that), else 30 days ago."""
    p = link.params or {}
    if p.get("settle_scanned_to"):
        return utc(p["settle_scanned_to"]) - pd.Timedelta(hours=1)
    seen = link.synced_at or link.first_seen_at
    if seen is not None:
        return utc(seen) - pd.Timedelta(days=1)
    return utc(now or datetime.now(timezone.utc)) - pd.Timedelta(days=30)


def settle(session, conn, code, sport="f1", since=None, client=None, max_pages=None):
    """Record the exchange's outcomes on its closed, unresolved links of `sport` from its settlement feed (schema
    endpoints.settlements and fields.settlement): resolved_yes True / False, plus params settled_at, settlement
    ("yes" / "no" / "void") and settlement_value. A void (a value neither 1 nor 0) leaves resolved_yes empty; its
    params.settlement keeps it from being looked for again. since: where the scan starts (default: the earliest
    scan_from of the pending links). A link not found gets params.settle_scanned_to (the pass's start when the feed
    was read to its end, else the latest settle time read), so the next pass reads only what is new. Idempotent: a
    resolved or void link is never pending again. An exchange whose schema has no settlement feed does nothing.
    Returns stats: links (pending at the start), pages, scanned, matched, resolved, yes, no, void, pending (still)."""
    schema = EX.load(code)
    if "settlements" not in schema.get("endpoints", {}):
        return dict(supported=False)
    sport_cfg(schema, sport)
    comp, _ = KS.competition(session, sport)
    links = pending(session, code, comp.id)
    stats = dict(links=len(links), pages=0, scanned=0, matched=0, resolved=0, yes=0, no=0, void=0, pending=len(links))
    if not links:
        return stats
    now = datetime.now(timezone.utc)
    since = utc(since) if since is not None else min(scan_from(lk, now) for lk in links)
    found, st = scan_settlements(client or Client(code), [lk.token_id for lk in links], since, max_pages)
    stats.update(pages=st["pages"], scanned=st["scanned"], since=since.isoformat())
    mark = now if st["exhausted"] else st["last_ts"]                 # how far the feed has been read for the rest
    for lk in links:
        params = dict(lk.params or {})
        hit = found.get(lk.token_id)
        if hit is None:
            if mark is not None:
                params["settle_scanned_to"] = mark.isoformat()
                lk.params = params
            continue
        outcome, value, at = hit
        params.pop("settle_scanned_to", None)
        params.update(settlement={True: "yes", False: "no", None: "void"}[outcome], settlement_value=value,
                      settled_at=at.isoformat() if at else None)
        lk.params, lk.resolved_yes, lk.closed, lk.active = params, outcome, True, False
        stats["matched"] += 1
        stats["void" if outcome is None else "yes" if outcome else "no"] += 1
        stats["resolved"] += outcome is not None
    stats["pending"] = len(links) - stats["matched"]
    session.commit()
    return stats


# ----- the tape, minute prices and books -----

def links_of(conn, code, sport=None, events=None, open_only=False):
    """(token, event) of the exchange's links, for a sport's competition or the given events."""
    where, params = ["l.exchange = :x"], dict(x=code)
    if events:
        where.append("l.condition_id = ANY(:e)")
        params["e"] = list(events)
    elif sport:
        from racinglines import sports
        where.append("co.code = :c")
        params["c"] = sports.load(sport)["competition"]["code"]
    else:
        raise ValueError("give events or a sport")
    if open_only:
        where.append("NOT l.closed")
    return conn.execute(text(f"""SELECT l.token_id, l.condition_id FROM market_links l
                                 LEFT JOIN competitions co ON co.id = l.competition_id
                                 WHERE {' AND '.join(where)} ORDER BY l.condition_id, l.token_id"""), params).all()


def trade_rows(schema, token, event, trades):
    """market_trades rows: the taker's side of the YES contract at the YES price, size in contracts."""
    ft, rows = schema["fields"]["trade"], []
    for t in trades:
        p = price(EX.dig(t, ft["price"]))
        ts = when(EX.dig(t, ft["ts"]), ft["ts_unit"])
        if p is None or ts is None:
            continue
        rows.append(dict(token_id=token, condition_id=event or token, outcome_index=0, ts=ts,
                         side="BUY" if str(EX.dig(t, ft["side"]) or "").lower() == "buy" else "SELL", price=p,
                         size=float(EX.dig(t, ft["size"]) or 0), tx_hash=str(EX.dig(t, ft["id"]))[:80], wallet=""))
    return rows


def fetch_trades(session, conn, code, sport=None, events=None, since=None, client=None, open_only=False):
    """Store every trade the exchange still serves for the markets (open_only: markets not closed). Idempotent.
    Returns trades stored."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    client = client or Client(code)
    since = since or pd.Timestamp("2000-01-01", tz="UTC")           # clipped to the schema's window
    n = 0
    for tok, ev in links_of(conn, code, sport, events, open_only=open_only):
        rows = trade_rows(client.schema, tok, ev, client.window("trades", tok, since))
        for i in range(0, len(rows), 1000):
            session.execute(pg_insert(m.MarketTrade).values(rows[i:i + 1000]).on_conflict_do_nothing(constraint="uq_market_trade"))
        n += len(rows)
    session.commit()
    return n


def history_rows(schema, token, points):
    """market_price_history rows from the exchange's minute quotes: the bid/ask mid, else the last price."""
    fh, rows = schema["fields"]["history"], []
    for p in points:
        _, _, mid = quote(price(EX.dig(p, fh["bid"])), price(EX.dig(p, fh["ask"])), price(EX.dig(p, fh["last"])))
        ts = when(EX.dig(p, fh["ts"]), fh["ts_unit"])
        if mid is not None and ts is not None:
            rows.append(dict(token_id=token, ts=ts, price=mid))
    return list({r["ts"]: r for r in rows}.values())       # one row per ts: an upsert can't touch a row twice


def fetch_history(session, conn, code, start, end=None, sport=None, events=None, client=None):
    """Store the exchange's minute prices for the markets from `start` (clipped to what it keeps). Returns rows."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    client = client or Client(code)
    n = 0
    for tok, _ in links_of(conn, code, sport, events):
        rows = history_rows(client.schema, tok, client.window("history", tok, start, end))
        if rows:
            session.execute(pg_insert(m.MarketPriceHistory).values(rows).on_conflict_do_update(
                index_elements=["token_id", "ts"], set_={"price": pg_insert(m.MarketPriceHistory).excluded.price}))
        n += len(rows)
    session.commit()
    return n


def book_row(schema, token, body, depth=10):
    """A market_book_snapshots row from the exchange's book: levels [[price, quantity]] best first."""
    fb = schema["fields"]["book"]
    bids = sorted(((float(p), float(q)) for p, q in EX.dig(body, fb["bids"]) or []), reverse=True)
    asks = sorted((float(p), float(q)) for p, q in EX.dig(body, fb["asks"]) or [])
    ts = when(EX.dig(body, fb["ts"]), fb["ts_unit"]) or datetime.now(timezone.utc)
    return dict(token_id=token, ts=ts, best_bid=bids[0][0] if bids else None, best_ask=asks[0][0] if asks else None,
                bids=[list(x) for x in bids[:depth]], asks=[list(x) for x in asks[:depth]])


def snapshot_books(session, conn, code, sport=None, events=None, depth=10, client=None):
    """One order-book snapshot per open market. Returns snapshots stored."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    client = client or Client(code)
    rows = [book_row(client.schema, tok, client.book(tok), depth) for tok, _ in links_of(conn, code, sport, events, open_only=True)]
    if rows:
        session.execute(pg_insert(m.MarketBookSnapshot).values(rows).on_conflict_do_nothing())
        session.commit()
    return len(rows)


# ----- the fair-price indicator -----

def fair_report(conn, code, sport="f1"):
    """The model's fair price beside the exchange's quote, for every open market the model prices.

    edge_yes = fair - ask - fee (buy YES at the ask); edge_no = bid - fair - fee (buy NO at 1 - bid). The fee is the
    schema's taker fee per contract. `call` is the side with a positive edge after the fee, else "". A thin book
    (no bid, a 1-cent ask) shows the fair price and no call: a quote nobody will trade against is not an edge.
    Returns a DataFrame sorted by the larger edge."""
    from racinglines import sports
    from racinglines.db import reads as data
    from racinglines.markets import venues
    fee = EX.load(code)["exchange"].get("taker_fee_per_contract", 0.0)
    links = data.q(conn, """SELECT l.* FROM market_links l JOIN competitions co ON co.id = l.competition_id
                            WHERE l.exchange = :x AND co.code = :c AND NOT l.closed AND l.prediction <> 'unmodeled'
                            ORDER BY l.event_slug, l.token_id""", x=code, c=sports.load(sport)["competition"]["code"])
    cache, out = {}, []
    for link in links.to_dict("records"):
        fair, run = data.model_prob(conn, link, cache)
        if fair is None:
            continue
        bid, ask, mid = link["last_bid"], link["last_ask"], link["last_price"]
        bid = None if bid is None or pd.isna(bid) else float(bid)
        ask = None if ask is None or pd.isna(ask) else float(ask)
        e = venues.net_edge(fee, fair, bid, ask)                # the same arithmetic the app shows
        out.append(dict(event=link["event_title"], subject=link["group_title"], kind=link["prediction"], fair=fair,
                        bid=bid, ask=ask, mid=None if mid is None or pd.isna(mid) else float(mid), edge_yes=e["edge_yes"],
                        edge_no=e["edge_no"], call=e["call"], token=link["token_id"], run_id=run))
    df = pd.DataFrame(out)
    if len(df):
        df["best"] = df[["edge_yes", "edge_no"]].max(axis=1)
        df = df.sort_values("best", ascending=False, na_position="last").drop(columns="best").reset_index(drop=True)
    return df


def fair_text(df, code, fee):
    """The fair-price report as text."""
    if not len(df):
        return f"{code}: no open market the model prices (run `racinglines markets --exchange {code} sync` first)"
    L = [f"{EX.load(code)['exchange']['name']}: model fair price vs the quote (taker fee ${fee:.2f} per contract, unverified)",
         f"{'market':<34}{'fair':>7}{'bid':>7}{'ask':>7}{'edge YES':>10}{'edge NO':>9}  call"]
    for r in df.itertuples():
        f = lambda v: "   –  " if v is None or pd.isna(v) else f"{v:6.2f}"          # noqa: E731
        e = lambda v: "     –  " if v is None or pd.isna(v) else f"{v:+8.2f}"          # noqa: E731
        L.append(f"{(r.event or '')[:20] + ' · ' + str(r.subject)[:12]:<34}{r.fair:7.2f}{f(r.bid)}{f(r.ask)}{e(r.edge_yes)}{e(r.edge_no)}  {r.call}")
    return "\n".join(L)


# ----- debug: buy one of everything -----

def observations(conn, code, links):
    """{token: [(ts, price, source)]}: every price the store holds for the links, whatever its spread or depth: minute
    prices, trade prices, a book's best bid and ask (their mid when both quote, else the side that does) and the sync's
    last quote (dated by synced_at). An empty book's 0.50 midpoint is not a price (venue_replay.OG.observed); a trade at
    any price is."""
    from racinglines.markets import store as MS
    from racinglines.markets.venue_replay import OG
    toks, root = links["token_id"].tolist(), MS.root_for(code)
    out = {t: [] for t in toks}
    ph, tr, bk = (MS.read(conn, name, tokens=toks, root=root) for name in ("prices", "trades", "books"))
    for name, df in (("minute price", OG.observed(ph, tr.iloc[:0], bk.iloc[:0])), ("trade", OG.observed(ph.iloc[:0], tr, bk.iloc[:0])),
                     ("book", OG.observed(ph.iloc[:0], tr.iloc[:0], bk))):
        for t, ts, p in zip(df["token_id"], df["ts"], df["price"]):
            out[t].append((pd.Timestamp(ts, tz="UTC"), float(p), name))
    for lk in links.to_dict("records"):
        b, a = (lk.get(k) for k in ("last_bid", "last_ask"))
        empty = all(v is None or pd.isna(v) for v in (b, a))
        p = next((lk[k] for k in ("last_price", "last_bid", "last_ask") if lk.get(k) is not None and not pd.isna(lk[k])), None)
        if p is not None and not (empty and abs(p - OG.FAKE_MID) < 1e-9):
            ts = lk.get("synced_at")
            out[lk["token_id"]].append((None if ts is None or pd.isna(ts) else pd.Timestamp(ts), float(p), "sync quote"))
    return out


def buy_all(conn, code, sport="f1", cost=0.01, fee=None):
    """The debug "buy one of everything" over every market the exchange lists for the sport (modeled or not): one YES
    and one NO share at the first price the store holds for it (markets/strategies/buy_everything.py), held; settled
    when the exchange resolved it, else marked at the last price held. fee: $ per contract (default the schema's
    taker_fee_per_contract). Read-only. -> one row per market."""
    from racinglines import exchanges, sports
    from racinglines.markets.strategies import buy_everything as BA
    fee = exchanges.load(code)["exchange"].get("taker_fee_per_contract", 0.0) if fee is None else fee
    links = pd.read_sql(text("""SELECT l.* FROM market_links l JOIN competitions co ON co.id = l.competition_id
                                WHERE l.exchange = :x AND co.code = :c ORDER BY l.condition_id, l.token_id"""), conn,
                        params=dict(x=code, c=sports.load(sport)["competition"]["code"]))
    if not len(links):
        return pd.DataFrame()
    obs = observations(conn, code, links)
    rows = []
    for lk in links.to_dict("records"):
        o = obs[lk["token_id"]]
        first = BA.first_price(o)
        dated = sorted((x for x in o if x[0] is not None), key=lambda x: x[0])
        mark = (dated or o or [(None, None, None)])[-1][1]
        res = lk.get("resolved_yes")
        outcome = None if res is None or pd.isna(res) else bool(res)
        row = dict(token=lk["token_id"], event=lk["event_title"], subject=lk["group_title"] or lk["outcome"],
                   kind=lk["prediction"], prices=len(o), source=None, entry_ts=None, entry=None, mark=mark,
                   outcome=outcome, status="no price", pnl_yes=None, pnl_no=None, pnl=None)
        if first is not None:
            h = BA.hold_pair(first[1], outcome, mark, cost, fee)
            row.update(source=first[2], entry_ts=first[0], entry=first[1], status=h["status"], pnl_yes=h["pnl_yes"],
                       pnl_no=h["pnl_no"], pnl=h["pnl"])
        rows.append(row)
    return pd.DataFrame(rows)


def buy_all_text(df, code, sport, cost, fee):
    if not len(df):
        return f"{code} {sport}: no markets synced (racinglines markets --exchange {code} --sport {sport} sync)."
    b = df[df["entry"].notna()]
    lines = [f"{code} {sport} DEBUG buy_all: {len(df)} markets, {df['prices'].gt(0).sum()} with a price, {len(b)} bought "
             f"(one YES + one NO each at the first price held, ${cost:g} cost and ${fee:g} fee per contract). "
             "A pair loses its costs and fees by construction (plumbing check, not an edge)."]
    for st in ("settled", "marked", "open"):
        g = b[b["status"] == st]
        if len(g):
            lines.append(f"  {st:8} {len(g):4} markets  P&L {g['pnl'].fillna(0).sum():+9.2f}  YES side "
                         f"{g['pnl_yes'].fillna(0).sum():+9.2f}  NO side {g['pnl_no'].fillna(0).sum():+9.2f}")
    if len(b):
        lines.append("  first price from: " + ", ".join(f"{k} {v}" for k, v in b["source"].value_counts().items()))
    if len(df) > len(b):
        lines.append(f"  NO PRICE: {len(df) - len(b)} markets have no price in the store (pull trades / history / books).")
    return "\n".join(lines)
