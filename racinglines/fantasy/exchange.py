"""
First-cut manual trading: immediate-or-cancel paper orders against the exchanges' recorded order books (owner,
2026-10-12 08:40Z: "enable first-cut manual trading, staging only"; F$ only, "NO REAL TRADES OR MONEY").

The design is docs/fantasy-trading.md "Exchange paper trading for takers (EXCH-1)", lines 100-259. What this first cut
changes, and why:

- **Prices come from the recorder, not a live venue call.** The design fetches a live book per order through a new
  markets/books.py (lines 102-122), which needs the read-only Mac probe first (EXCH-0, lines 357-420; CLAUDE.md: never
  guess a live API's behaviour from the cloud). The VM's recorder already stores every linked F1 market's book every
  5 minutes in market_book_snapshots (bids/asks [[price, size], ...], best first). An order walks the latest snapshot,
  refused when it is older than MAX_AGE_MIN (15; the design's 30 s assumes a live fetch). So nothing here calls a
  venue, and nothing imports an order path.
- **Staging has no recorder** (docs/vm-deploy.md "Staging": "no recorder or signals"), so its own books are as old as
  its last copy of production. RACINGLINES_FANTASY_BOOKS_URL, when set, is the database the books are read from
  (production's, read-only SELECTs); unset, the app's own database.
- **No seasons yet.** Every active account with a ledger may trade (owner: "all registered users"); the balance check
  locks the account's `users` row instead of a season_members row (design step 5, line 136).
- **Closing times come from the database** (pipelines/clv.closes: the deciding round's stored start, else race day from
  the schema), not the FastF1 parquet (lines 67-98), which staging's empty data folder doesn't have. A race link with
  no close is refused (`no_schedule`, fails closed). Season futures have no close; the design's pause while a session
  runs (DEC-14, line 98) is not built.

The fill and fee rules are the design's (lines 146-208): buys walk the asks from the best up to the limit, sells walk
the bids down to it and never short, one side per position, Kalshi's fee is ceil_cent(0.07 x sum C x P x (1 - P))
once per order (markets/venue_replay.py Kalshi.TAKER_FEE), Polymarket's 0. Both fees are coded assumptions the owner
has not confirmed yet (lines 204-208).
"""

import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import create_engine, text

from racinglines.fantasy import ledger as L
from racinglines.markets import venue_replay as VR

MAX_STAKE = 100.0                  # F$ per order: shares x limit (DEC-5, docs/fantasy-trading.md line 214)
MAX_AGE_MIN = float(os.environ.get("RACINGLINES_FANTASY_BOOK_MAX_AGE_MIN", "15"))   # recorded book older: refused
DEPTH = 10                         # visible levels walked (design line 153)
RATE_MINUTE, RATE_DAY = 10, 200    # orders per account (design line 217); in memory, one web process
SPORT = "f1"
RACE_KINDS = ("race_win", "race_podium", "race_top5", "race_top10", "race_pole", "race_h2h", "race_constructor_top")
SPRINT_KINDS = ("race_sprint_win", "race_sprint_pole")
SEASON_KINDS = ("champion", "constructors_champion")
EXCHANGES = ("kalshi", "polymarket")
REASONS = ("bad_input", "not_a_member", "sport_not_tradable", "kind_not_tradable", "market_closed", "session_started", "no_schedule",
           "book_unavailable", "stale_book", "no_liquidity", "over_order_cap", "insufficient_balance",
           "opposite_side_held", "sell_exceeds_position")
TEXT = {"bad_input": "Check the side, the shares (a whole number) and the limit (0.01 to 0.99).",
        "not_a_member": "This account can't trade fantasy markets.",
        "sport_not_tradable": "Fantasy trading is Formula 1 only for now: this market can't be traded.",
        "kind_not_tradable": "This market can't be traded here yet.",
        "market_closed": "This market is closed or already settled.",
        "session_started": "This market closed when its session started.",
        "no_schedule": "This market has no known closing time yet, so it can't be traded.",
        "book_unavailable": "The venue's order book couldn't be read just now (or none is recorded yet): try again in a moment.",
        "stale_book": f"The last recorded order book is more than {MAX_AGE_MIN:.0f} minutes old.",
        "no_liquidity": "Nothing on the book at your limit or better.",
        "over_order_cap": f"Orders are capped at F${MAX_STAKE:.0f} (shares x limit).",
        "insufficient_balance": "Not enough F$ to cover this order at your limit, fee included.",
        "opposite_side_held": "You hold the other side of this market: sell it first.",
        "sell_exceeds_position": "You can't sell more shares than you hold.",
        "rate_limited": "Too many orders: wait a minute."}


class Rejected(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def tradable():
    """{exchange: kinds} a player may trade: F1 race kinds (sprint kinds unless RACINGLINES_KALSHI_SPRINTS=0) and the
    two championships, on Kalshi and Polymarket (docs/fantasy-trading.md "What is tradable", lines 30-50)."""
    from racinglines.markets.kalshi.sync import sprints_enabled
    kinds = (*RACE_KINDS, *(SPRINT_KINDS if sprints_enabled() else ()), *SEASON_KINDS)
    return {x: kinds for x in EXCHANGES}


# --- pure: the book, the walk, the fee -------------------------------------------------------------------------

def levels(raw):
    """Recorded levels ([[price, size], ...] or their JSON text) as [(price, size)], dropping bad rows."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    out = []
    for lv in raw or ():
        try:
            p, s = float(lv[0]), float(lv[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 < p < 1 and s > 0:
            out.append((p, s))
    return out


def yes_to_no(bids, asks):
    """The NO side's (bids, asks) from the YES book: a NO ask is 1 - a YES bid, a NO bid is 1 - a YES ask (design lines
    179-185), best first. YES bids 0.39 x 150, 0.38 x 400 -> NO asks 0.61 x 150, 0.62 x 400."""
    return ([(round(1 - p, 4), s) for p, s in asks], [(round(1 - p, 4), s) for p, s in bids])


def walk(book_levels, shares, limit, buy=True):
    """[(price, shares taken)] from best-first levels: take min(level size, shares still wanted) while the price is at or
    better than the limit (at or below for a buy, at or above for a sell); whole shares; at most DEPTH levels."""
    out, want = [], int(shares)
    for p, s in book_levels[:DEPTH]:
        if want <= 0 or (p > limit + 1e-9 if buy else p < limit - 1e-9):
            break
        take = min(int(math.floor(s + 1e-9)), want)
        if take > 0:
            out.append((p, take))
            want -= take
    return out


def ceil_cent(x):
    """Up to the next cent, as venue_replay.Kalshi.taker_fee rounds (float noise below 1e-6 cent ignored)."""
    return math.ceil(round(x * 100, 6)) / 100


def fee(exchange, fills):
    """The order's fee in F$: the venue's TAKER_FEE x sum over fills of C x P x (1 - P), rounded up to the cent once per
    order (design lines 194-202). Polymarket's rate is 0."""
    rate = VR.EXCHANGES[exchange].TAKER_FEE
    return ceil_cent(rate * sum(c * p * (1 - p) for p, c in fills)) if rate else 0.0


def worst_fee(exchange, shares, limit):
    """The most a buy of `shares` at `limit` can pay in fees: P x (1 - P) peaks at 0.5."""
    p = min(limit, 0.5)
    return fee(exchange, [(p, shares)])


def cents(x):
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# --- the recorded books ------------------------------------------------------------------------------------------

_BOOKS = {}


def books_engine(default):
    """The engine books are read from: RACINGLINES_FANTASY_BOOKS_URL (staging: production's database), else `default`.
    That engine is read-only whatever the URL says: every session starts with default_transaction_read_only on, so
    a staging bug can't write to production."""
    url = os.environ.get("RACINGLINES_FANTASY_BOOKS_URL", "").strip()
    if not url:
        return default
    if url not in _BOOKS:
        _BOOKS[url] = create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=2,
                                    connect_args={"options": "-c default_transaction_read_only=on"})
    return _BOOKS[url]


def latest_book(engine, token_id):
    """The latest recorded book for a token: dict(ts, bids, asks) with YES levels best first, or None."""
    with engine.connect() as c:
        r = c.execute(text("""SELECT ts, bids, asks FROM market_book_snapshots WHERE token_id = :t
                              ORDER BY ts DESC LIMIT 1"""), dict(t=token_id)).first()
    if r is None:
        return None
    return dict(ts=r[0], bids=levels(r[1]), asks=levels(r[2]))


LIVE_TIMEOUT = 5.0                 # s per live book request; a slow venue refuses the order (book_unavailable)


def books_mode():
    """RACINGLINES_FANTASY_BOOKS: "live" (default) checks every order against the venue's own book at that moment
    (Kalshi GET /markets/{ticker}/orderbook, Polymarket CLOB GET /book; the public read-only endpoints the recorder
    already reads every 5 minutes); "recorded" walks the latest recorded snapshot instead."""
    return "recorded" if os.environ.get("RACINGLINES_FANTASY_BOOKS", "live").strip().lower() == "recorded" else "live"


def live_book(exchange, token_id, transport=None):
    """The venue's book now: dict(ts, bids, asks) in latest_book's shape (YES levels, best first), or None when the
    request fails. transport (an httpx transport) replaces the network in tests."""
    import httpx
    try:
        if exchange == "kalshi":
            from racinglines.markets.kalshi import client as K
            from racinglines.markets.kalshi import sync as KS
            with K.Client(transport=transport) as kc:
                kc.http.timeout = httpx.Timeout(LIVE_TIMEOUT)
                r = kc.http.get(f"/markets/{token_id}/orderbook", params=dict(depth=DEPTH))
                r.raise_for_status()
                body = r.json()
            row = KS.book_row(token_id, body.get("orderbook_fp") or body.get("orderbook") or {},
                              datetime.now(timezone.utc), DEPTH)
            return dict(ts=row["ts"], bids=levels(row["bids"]), asks=levels(row["asks"]))
        if exchange == "polymarket":
            from racinglines.markets.polymarket import sync as PS
            with httpx.Client(base_url=PS.CLOB, timeout=LIVE_TIMEOUT, transport=transport) as c:
                r = c.get("/book", params={"token_id": token_id})
                r.raise_for_status()
                b = r.json()
            ts = datetime.fromtimestamp(int(b["timestamp"]) / 1000, tz=timezone.utc) if b.get("timestamp") \
                else datetime.now(timezone.utc)
            return dict(ts=ts, bids=levels(PS._levels(b.get("bids"), True)), asks=levels(PS._levels(b.get("asks"), False)))
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None
    return None


def book_getter(engine, link):
    """place / quote's default get_book for this link: the venue's live book (books_mode "live"), else the latest
    recorded one."""
    if books_mode() == "live" and link is not None and link.get("exchange") in EXCHANGES:
        return lambda tok: live_book(link["exchange"], tok)
    return lambda tok: latest_book(books_engine(engine), tok)


def quotes(engine, token_ids):
    """{token_id: (best YES bid, best YES ask, ts)} from each token's latest recorded book in the last day."""
    ids = sorted({t for t in token_ids if t})
    if not ids:
        return {}
    with engine.connect() as c:
        rs = c.execute(text("""SELECT DISTINCT ON (token_id) token_id, best_bid, best_ask, ts FROM market_book_snapshots
                               WHERE token_id = ANY(:t) AND ts > now() - interval '1 day'
                               ORDER BY token_id, ts DESC"""), dict(t=ids)).all()
    return {t: (b, a, ts) for t, b, a, ts in rs}


# --- the rate limit (design line 217: in memory; refusals write an audit row, no order row) -------------------------

_HITS = defaultdict(deque)
_HITS_LOCK = threading.Lock()


def rate_ok(user_id, now=None):
    now = time.time() if now is None else now
    with _HITS_LOCK:
        q = _HITS[user_id]
        while q and q[0] < now - 86400:
            q.popleft()
        if sum(t > now - 60 for t in q) >= RATE_MINUTE or len(q) >= RATE_DAY:
            return False
        q.append(now)
        return True


def markets(c, search=""):
    """The links a player may trade now, newest events first: F1, Kalshi and Polymarket, open and unresolved, in the
    tradable kinds, each with its close (`closes_at`, closes_for: None for a season future or an unknown one). Closing
    times are enforced when an order is placed. -> [dict]"""
    kinds = sorted({k for ks in tradable().values() for k in ks})
    rs = c.execute(text("""
        SELECT ml.id, ml.exchange, ml.token_id, ml.prediction AS kind, ml.prediction, ml.race_id, ml.question,
               ml.group_title, ml.outcome, ml.event_title, ml.last_bid, ml.last_ask, e.start_date, co.code AS competition
        FROM market_links ml JOIN competitions co ON co.id = ml.competition_id JOIN sports sp ON sp.id = co.sport_id
        LEFT JOIN races ra ON ra.id = ml.race_id LEFT JOIN events e ON e.id = ra.event_id
        WHERE sp.code = :sport AND ml.exchange = ANY(:x) AND ml.prediction = ANY(:k) AND ml.active AND NOT ml.closed
          AND ml.resolved_yes IS NULL AND (e.start_date IS NULL OR e.start_date >= current_date - 1)
          AND (:q = '' OR ml.question ILIKE :like OR ml.group_title ILIKE :like OR ml.event_title ILIKE :like)
        ORDER BY e.start_date NULLS LAST, ml.event_title, ml.exchange, ml.prediction, ml.group_title, ml.id"""),
        dict(sport=SPORT, x=list(EXCHANGES), k=kinds, q=search, like=f"%{search}%")).mappings().all()
    out = [dict(r) for r in rs]
    closes = closes_for(c, out)
    for m in out:
        m["closes_at"] = closes[m["id"]]
    return out


# --- the order ---------------------------------------------------------------------------------------------------

_LINK_SQL = """
        SELECT ml.id, ml.exchange, ml.token_id, ml.condition_id, ml.prediction, ml.params, ml.race_id, ml.closed,
               ml.active, ml.resolved_yes, ml.end_date, co.code AS competition, sp.code AS sport, e.start_date
        FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
        LEFT JOIN sports sp ON sp.id = co.sport_id
        LEFT JOIN races ra ON ra.id = ml.race_id LEFT JOIN events e ON e.id = ra.event_id"""


def _link(conn, link_id):
    return conn.execute(text(_LINK_SQL + " WHERE ml.id = :i"), dict(i=link_id)).mappings().first()


def closes_at(conn, link):
    """A link's close as aware UTC, None for a season future (no close), or Rejected('no_schedule') for a race link
    whose deciding round has no stored start (pipelines/clv.closes)."""
    if link["prediction"] in SEASON_KINDS:
        return None
    t = _close(conn, link)
    if t is None and link.get("token_id"):
        # staging's own rounds carry no session starts (no schedule ingest there): ask the books database (production),
        # matching the same token exactly, read-only
        books = books_engine(conn.engine)
        if books is not conn.engine:
            with books.connect() as b:
                there = b.execute(text(_LINK_SQL + " WHERE ml.token_id = :t ORDER BY ml.id LIMIT 1"),
                                  dict(t=link["token_id"])).mappings().first()
                t = _close(b, there) if there is not None else None
    if t is None:
        raise Rejected("no_schedule")
    return _aware(t)


def _aware(t):
    """A stored close (pandas, naive = UTC) as an aware UTC datetime; None stays None."""
    if t is None:
        return None
    return t.tz_localize("UTC").to_pydatetime() if getattr(t, "tzinfo", None) is None else t.to_pydatetime()


def closes_for(conn, links):
    """closes_at for many links at once, for a list or a page: {link id: close as aware UTC, or None}. Season futures
    are None (no close); race links get the deciding round's stored start from one rounds query (pipelines/clv.closes),
    then, for any still without one, the books database's link with the same token (staging, as closes_at); a race
    link with no close anywhere is None here, where closes_at refuses it (an order on it is refused, the list still
    shows it)."""
    from racinglines.pipelines import clv
    rows = [dict(lk) for lk in links]
    race = [r for r in rows if r["prediction"] not in SEASON_KINDS and r.get("race_id") is not None]
    local = clv.closes(conn, race) if race else {}
    out = {}
    for r in rows:
        out[r["id"]] = (None if r["prediction"] in SEASON_KINDS or r.get("race_id") is None
                        else local.get((int(r["race_id"]), r["prediction"]), (None, ""))[0])
    missing = [r for r in rows if out[r["id"]] is None and r["prediction"] not in SEASON_KINDS and r.get("token_id")]
    books = books_engine(conn.engine)
    if missing and books is not conn.engine:
        with books.connect() as b:
            there = {}
            for t in b.execute(text(_LINK_SQL + " WHERE ml.token_id = ANY(:t) ORDER BY ml.id"),
                               dict(t=[r["token_id"] for r in missing])).mappings():
                there.setdefault(t["token_id"], dict(t))
            found = clv.closes(b, [t for t in there.values() if t["race_id"] is not None])
        for r in missing:
            t = there.get(r["token_id"])
            if t is not None and t["race_id"] is not None:
                out[r["id"]] = found.get((int(t["race_id"]), t["prediction"]), (None, ""))[0]
    return {k: _aware(v) for k, v in out.items()}


def _close(conn, link):
    """The deciding round's stored start for a race link in this database (pipelines/clv.closes), or None."""
    if link["race_id"] is None:
        return None
    from racinglines.pipelines import clv
    return clv.closes(conn, [dict(link)]).get((int(link["race_id"]), link["prediction"]), (None, ""))[0]


def sport_ok(link):
    """True when a link's sport is the one fantasy trading covers (F1 only to start: owner, 2026-10-12 23:07Z)."""
    return link is not None and link["sport"] == SPORT


def check_link(conn, link, now):
    """Refuse a link a player may not trade now (kind, venue, sport, closed, resolved, past its close)."""
    if link is None:
        raise Rejected("bad_input")
    if link["sport"] != SPORT:
        raise Rejected("sport_not_tradable")              # owner, 2026-10-12: F1 only to start
    if link["exchange"] not in EXCHANGES or link["prediction"] not in tradable()[link["exchange"]]:
        raise Rejected("kind_not_tradable")
    if link["exchange"] == "polymarket" and link["prediction"] == "race_h2h" and link["condition_id"]:
        first = conn.execute(text("SELECT min(id) FROM market_links WHERE condition_id = :c AND exchange = 'polymarket'"),
                             dict(c=link["condition_id"])).scalar()
        if first != link["id"]:                    # the outcome-1 token: trade NO on the outcome-0 link (lines 186-188)
            raise Rejected("kind_not_tradable")
    if link["closed"] or not link["active"] or link["resolved_yes"] is not None:
        raise Rejected("market_closed")
    if link["end_date"] is not None and link["end_date"] <= now:
        raise Rejected("market_closed")
    close = closes_at(conn, link)
    if close is not None and close <= now:
        raise Rejected("session_started")


def _reject(conn, user, link_id, link, side, action, shares, limit, reason, client_key, book=None):
    oid = conn.execute(text("""
        INSERT INTO accounts.fantasy_orders (user_id, market_link_id, token_id, exchange, side, action, shares_req,
                                             limit_price, status, reject_reason, book, client_key)
        VALUES (:u, :l, :t, :x, :s, :a, :n, :p, 'rejected', :r, CAST(:b AS jsonb), :k) RETURNING id"""),
        dict(u=user["id"], l=link_id if isinstance(link_id, int) else None, t=(link or {}).get("token_id"),
             x=(link or {}).get("exchange"), s=side if side in ("yes", "no") else None,
             a=action if action in ("buy", "sell") else None, n=_num(shares, 12), p=_num(limit, 6), r=reason,
             b=json.dumps(book) if book else None, k=client_key)).scalar()
    return dict(ok=False, order_id=oid, status="rejected", reason=reason, message=TEXT[reason], fills=[], fee=0.0)


def _num(v, width):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and abs(f) < 10 ** (width - 4) else None


def _audit(conn, user, what, **detail):
    if user.get("via"):                                  # "mcp" for an order placed through the MCP server
        detail["via"] = user["via"]
    conn.execute(text("""INSERT INTO activity_log (user_id, username, role, action, detail)
                         VALUES (:u, :n, :r, :a, CAST(:d AS jsonb))"""),
                 dict(u=user["id"], n=user.get("username"), r=user.get("role"), a=what,
                      d=json.dumps(detail, default=str)))


def _existing(conn, user_id, client_key):
    """The result of an order this account already placed with this client key (a retry), or None."""
    if not client_key:
        return None
    o = conn.execute(text("""SELECT id, status, reject_reason FROM accounts.fantasy_orders
                             WHERE user_id = :u AND client_key = :k"""), dict(u=user_id, k=client_key)).first()
    if o is None:
        return None
    fills = conn.execute(text("SELECT price, shares, fee FROM accounts.fantasy_fills WHERE order_id = :o ORDER BY id"),
                         dict(o=o[0])).all()
    return dict(ok=o[1] != "rejected", order_id=o[0], status=o[1], reason=o[2], repeat=True,
                message=TEXT.get(o[2], "") if o[2] else "", fills=[(float(p), float(n)) for p, n, _ in fills],
                fee=float(sum(f for _, _, f in fills)))


def _validate(c, user, link, side, action, shares, limit, now, demo):
    """Step 1 of an order, no lock held: the input, the account's grant, the link and its close, the order cap.
    -> (shares as int, limit rounded) or Rejected."""
    n = _num(shares, 12)
    p = _num(limit, 6)
    if (side not in ("yes", "no") or action not in ("buy", "sell") or n is None or n != int(n) or n < 1
            or p is None or not 0.01 <= p <= 0.99 or link is None):
        raise Rejected("bad_input")
    n, p = int(n), round(p, 4)
    if demo or not L.has_grant(c, user["id"]):
        raise Rejected("not_a_member")
    check_link(c, link, now)
    if action == "buy" and n * p > MAX_STAKE + 1e-9:
        raise Rejected("over_order_cap")
    return n, p


def _levels(get_book, link, side, action, now):
    """Step 2, no database lock held: the latest recorded book, refused when missing or older than MAX_AGE_MIN; the
    side's levels the order walks (NO through yes_to_no), best first, and the snapshot the order row keeps.
    -> (levels, snap) or Rejected."""
    book = get_book(link["token_id"])
    if book is None:
        raise Rejected("book_unavailable")
    ts = book["ts"] if book["ts"].tzinfo else book["ts"].replace(tzinfo=timezone.utc)
    age = (now - ts).total_seconds()
    if age > MAX_AGE_MIN * 60:
        raise Rejected("stale_book")
    bids, asks = book["bids"], book["asks"]
    if side == "no":
        bids, asks = yes_to_no(bids, asks)
    lv = asks if action == "buy" else bids
    snap = dict(ts=ts.isoformat(), age_s=round(age), side_walked=f"{side} {'asks' if action == 'buy' else 'bids'}",
                levels=[list(x) for x in lv[:DEPTH]], best_bid=bids[0][0] if bids else None,
                best_ask=asks[0][0] if asks else None)
    return lv, snap


def _fills(c, user, link, lid, side, action, n, p, lv, lock):
    """Step 3's arithmetic: the account's position on the market (locked for an order, read for a quote), one side
    per market, never short, the walk, the fee and the balance check. -> (fills, fee, held, cost) or Rejected."""
    pos = c.execute(text(f"""SELECT yes_shares, no_shares, cost, status FROM accounts.fantasy_positions
                            WHERE user_id = :u AND market_link_id = :l {"FOR UPDATE" if lock else ""}"""),
                    dict(u=user["id"], l=lid)).first()
    if pos is not None and pos[3] != "open":
        raise Rejected("market_closed")                 # settled (settle below): paid out, nothing to trade
    held_yes, held_no, cost = (float(pos[0]), float(pos[1]), float(pos[2])) if pos else (0.0, 0.0, 0.0)
    held = held_yes if side == "yes" else held_no
    if action == "buy" and (held_no if side == "yes" else held_yes) > 0:
        raise Rejected("opposite_side_held")
    if action == "sell" and n > held + 1e-9:
        raise Rejected("sell_exceeds_position")
    fills = walk(lv, n, p, buy=action == "buy")
    if not fills:
        raise Rejected("no_liquidity")
    f = fee(link["exchange"], fills)
    if action == "buy" and n * p + worst_fee(link["exchange"], n, p) > L.available(c, user["id"]) + 1e-9:
        raise Rejected("insufficient_balance")
    return fills, f, held, cost


def _lid(link_id):
    try:
        return int(link_id)
    except (TypeError, ValueError):
        return None


def quote(conn, user, link_id, side, action, shares, limit, now=None, get_book=None, demo=False):
    """What place would do with this order now, writing nothing: the same three steps in the same order (input,
    account, link and close, cap; the book and its age; the position's rules, the walk, the fee, the balance), read
    without the account's lock, so a quote that fills can still be refused a moment later. -> dict(ok, reason, message,
    status filled|partial, shares, limit, fills [(price, shares)], filled, avg_price, notional, fee, total (what a buy
    costs, fee included, or a sale brings after it), book_ts, age_s, cash)."""
    now = now or datetime.now(timezone.utc)
    get_book = get_book or (lambda tok: latest_book(books_engine(conn.engine), tok))
    side, action = str(side or "").lower(), str(action or "").lower()
    lid = _lid(link_id)
    out = dict(ok=False, reason=None, message="", status=None, shares=None, limit=None, fills=[], filled=0, avg_price=None,
               notional=0.0, fee=0.0, total=0.0, book_ts=None, age_s=None, cash=None)
    try:
        link = _link(conn, lid) if lid is not None else None
        n, p = _validate(conn, user, link, side, action, shares, limit, now, demo)
        out.update(shares=n, limit=p, cash=round(L.available(conn, user["id"]), 2))
        lv, snap = _levels(get_book, link, side, action, now)
        out.update(book_ts=snap["ts"], age_s=snap["age_s"])
        fills, f, _, _ = _fills(conn, user, link, lid, side, action, n, p, lv, lock=False)
    except Rejected as ex:
        out.update(reason=ex.reason, message=TEXT[ex.reason])
        return out
    got = sum(s for _, s in fills)
    notional = sum(s * pr for pr, s in fills)
    out.update(ok=True, status="filled" if got == n else "partial", fills=fills, filled=got,
               avg_price=round(notional / got, 4), notional=round(notional, 2), fee=f,
               total=round(notional + f if action == "buy" else notional - f, 2))
    return out


def place(engine, user, link_id, side, action, shares, limit, client_key=None, now=None, get_book=None, demo=False):
    """Place one IOC paper order for `user` (a dict with id, username, role). -> dict(ok, order_id, status, reason,
    message, fills [(price, shares)], fee, position). Every refusal after the rate limit writes a rejected
    fantasy_orders row and a `fantasy_order_rejected` audit row; a fill writes the order, one fantasy_fills row per
    level (the fee on the first), the position and the ledger rows (one `trade` per fill, one `fee` per order), all in
    one transaction holding the account's row lock. get_book(token_id) -> dict(ts, bids, asks) is injectable (tests);
    by default book_getter: the venue's live book, or the latest recorded one with RACINGLINES_FANTASY_BOOKS=recorded."""
    now = now or datetime.now(timezone.utc)
    client_key = (client_key or "").strip()[:64] or None
    side, action = str(side or "").lower(), str(action or "").lower()
    with engine.begin() as c:
        prior = _existing(c, user["id"], client_key)
        if prior is not None:
            return prior
    # 1. validate without a lock: input, account, link, close (_validate, shared with quote)
    link, lid = None, _lid(link_id)
    with engine.begin() as c:
        try:
            if lid is not None:
                link = _link(c, lid)
            n, p = _validate(c, user, link, side, action, shares, limit, now, demo)
        except Rejected as ex:
            out = _reject(c, user, lid, link, side, action, shares, limit, ex.reason, client_key)
            _audit(c, user, "fantasy_order_rejected", order_id=out["order_id"], reason=ex.reason, link=lid)
            return out
    # 2. the book, with no database lock held (_levels): the venue's own, live, unless injected or books_mode()
    try:
        lv, snap = _levels(get_book or book_getter(engine, link), link, side, action, now)
    except Rejected as ex:
        with engine.begin() as c:
            out = _reject(c, user, lid, link, side, action, n, p, ex.reason, client_key)
            _audit(c, user, "fantasy_order_rejected", order_id=out["order_id"], reason=ex.reason, link=lid)
        return out
    # 3. one transaction: lock the account, re-check, walk (_fills, under the position's lock), write
    with engine.begin() as c:
        c.execute(text("SELECT id FROM users WHERE id = :u FOR UPDATE"), dict(u=user["id"]))
        prior = _existing(c, user["id"], client_key)           # a retry that raced the first attempt
        if prior is not None:
            return prior
        try:
            link = _link(c, lid)
            check_link(c, link, now)
            fills, f, held, cost = _fills(c, user, link, lid, side, action, n, p, lv, lock=True)
        except Rejected as ex:
            out = _reject(c, user, lid, link, side, action, n, p, ex.reason, client_key, book=snap)
            _audit(c, user, "fantasy_order_rejected", order_id=out["order_id"], reason=ex.reason, link=lid)
            return out
        got = sum(s for _, s in fills)
        status = "filled" if got == n else "partial"
        oid = c.execute(text("""
            INSERT INTO accounts.fantasy_orders (user_id, market_link_id, token_id, exchange, side, action, shares_req,
                                                 limit_price, status, book, client_key)
            VALUES (:u, :l, :t, :x, :s, :a, :n, :p, :st, CAST(:b AS jsonb), :k) RETURNING id"""),
            dict(u=user["id"], l=lid, t=link["token_id"], x=link["exchange"], s=side, a=action, n=n, p=p, st=status,
                 b=json.dumps(snap), k=client_key)).scalar()
        sign = -1 if action == "buy" else 1
        for i, (price, sh) in enumerate(fills):
            fid = c.execute(text("""
                INSERT INTO accounts.fantasy_fills (order_id, user_id, market_link_id, side, action, shares, price, fee, ts)
                VALUES (:o, :u, :l, :s, :a, :n, :p, :f, :ts) RETURNING id"""),
                dict(o=oid, u=user["id"], l=lid, s=side, a=action, n=sh, p=price, f=cents(f) if i == 0 else 0, ts=now)).scalar()
            c.execute(text("""INSERT INTO accounts.fantasy_ledger (user_id, amount, reason, note, ref_table, ref_id)
                              VALUES (:u, :a, 'trade', :n, 'fantasy_fills', :r)"""),
                      dict(u=user["id"], a=cents(sign * sh * price), r=fid,
                           n=f"{action} {sh:g} {side.upper()} at {price:.2f}"))
        if f:
            c.execute(text("""INSERT INTO accounts.fantasy_ledger (user_id, amount, reason, note, ref_table, ref_id)
                              VALUES (:u, :a, 'fee', :n, 'fantasy_orders', :r)"""),
                      dict(u=user["id"], a=-cents(f), r=oid, n=f"{link['exchange']} taker fee"))
        notional = sum(s * pr for pr, s in fills)
        if action == "buy":
            new_held, new_cost = held + got, cost + notional
        else:
            avg = cost / held if held else 0.0
            new_held, new_cost = held - got, max(cost - got * avg, 0.0) if held - got > 0 else 0.0
        col = "yes_shares" if side == "yes" else "no_shares"
        c.execute(text(f"""
            INSERT INTO accounts.fantasy_positions (user_id, market_link_id, exchange, {col}, cost, fees)
            VALUES (:u, :l, :x, :h, :c, :f)
            ON CONFLICT (user_id, market_link_id) DO UPDATE SET {col} = :h, cost = :c,
                fees = accounts.fantasy_positions.fees + :f, updated_at = now()"""),
            dict(u=user["id"], l=lid, x=link["exchange"], h=new_held, c=cents(new_cost), f=cents(f)))
        _audit(c, user, "fantasy_order", order_id=oid, link=lid, side=side, order=action, fills=fills, fee=f)
    return dict(ok=True, order_id=oid, status=status, reason=None, fills=fills, fee=f,
                message="", position=dict(side=side, shares=new_held, cost=round(new_cost, 2)))


# --- settlement ---------------------------------------------------------------------------------------------------

def settle(engine, books=None):
    """Pay out every open hand-trade position whose market has resolved: YES shares pay F$1 each on a YES outcome, NO
    shares on a NO, the position becomes 'settled' (outcome, payout, settled_at) and one ledger row (reason 'settle',
    ref fantasy_positions) credits the payout. Idempotent: a position settles once (status check under the row lock,
    and the ledger's unique (ref_table, ref_id, reason) index).

    Resolution is read by token id from `books` (books_engine: production's market_links on staging, whose own links
    are not synced), exact keys only. Called when the Trade and History pages load, so staging, which has no timers,
    settles too. -> number of positions settled"""
    books = books or books_engine(engine)
    with engine.connect() as c:
        if not L.ready(c):
            return 0
        rows = c.execute(text("""
            SELECT p.id, ml.token_id FROM accounts.fantasy_positions p JOIN market_links ml ON ml.id = p.market_link_id
            WHERE p.status = 'open' AND (p.yes_shares > 0 OR p.no_shares > 0) AND ml.token_id IS NOT NULL""")).all()
    if not rows:
        return 0
    with books.connect() as b:
        done = dict(b.execute(text("""SELECT DISTINCT ON (token_id) token_id, resolved_yes FROM market_links
                                      WHERE token_id = ANY(:t) AND resolved_yes IS NOT NULL ORDER BY token_id, id"""),
                              dict(t=list({t for _, t in rows}))).all())
    n = 0
    for pid, token in rows:
        if token not in done:
            continue
        yes = bool(done[token])
        with engine.begin() as c:
            r = c.execute(text("""
                UPDATE accounts.fantasy_positions SET status = 'settled', outcome = :o, settled_at = now(),
                    updated_at = now(), payout = CASE WHEN :o THEN yes_shares ELSE no_shares END
                WHERE id = :i AND status = 'open' RETURNING user_id, payout, yes_shares, no_shares"""),
                dict(i=pid, o=yes)).first()
            if r is None:
                continue
            if r.payout > 0:
                c.execute(text("""INSERT INTO accounts.fantasy_ledger (user_id, amount, reason, note, ref_table, ref_id)
                                  VALUES (:u, :a, 'settle', :n, 'fantasy_positions', :i)
                                  ON CONFLICT DO NOTHING"""),
                          dict(u=r.user_id, a=r.payout, i=pid,
                               n=f"settled {'YES' if yes else 'NO'}: {r.payout:g} shares paid F$1"))
            n += 1
    return n
