"""
The fantasy-trading tables, balances, positions and the leaderboard (owner, 2026-10-12 08:40Z: "1k balance they can
see, NO REAL TRADES OR MONEY ... manual trading as well as following one of our strategies"; leaderboard "staging only").

The ledger is accounts.fantasy_ledger (web/accounts.py), the one that already holds each account's 1,000 sign-up
grant: the owner's "the new ledger IS our fantasy ledger" (2026-10-11 21:38Z). A manual trade adds one `trade` row per
fill and one `fee` row per order, each pointing at the row that caused it (ref_table, ref_id), so a retry can never
write it twice (UNIQUE(ref_table, ref_id, reason)). Cash = SUM(amount).

The design (docs/fantasy-trading.md "Tables owned here", lines 261-302) puts the trades in Alembic tables next to
seasons and a season ledger (bankroll_entries, docs/fantasy-accounts.md lines 192-245). This first cut has no seasons
(every registered account plays) and keeps the tables in the `accounts` schema, created by `racinglines users setup`
like the ledger itself (web/accounts.py: a deploy never changes the VM's database), so merging the code writes
nothing until setup runs, by hand, after a backup. Columns follow the design; market links are stored by id and token
without a foreign key, so an exchange sync that rebuilds links is never blocked by a player's order.

Money is numeric(12,2) in F$, rounded to cents as it is written; prices numeric(6,4).
"""

import os

from sqlalchemy import text

DDL = (
    # the ledger gains a pointer to what caused each row (NULL for grants and the reset)
    "ALTER TABLE accounts.fantasy_ledger ADD COLUMN IF NOT EXISTS ref_table varchar(40)",
    "ALTER TABLE accounts.fantasy_ledger ADD COLUMN IF NOT EXISTS ref_id bigint",
    """CREATE UNIQUE INDEX IF NOT EXISTS fantasy_ledger_ref ON accounts.fantasy_ledger (ref_table, ref_id, reason)
       WHERE ref_table IS NOT NULL""",
    """CREATE TABLE IF NOT EXISTS accounts.fantasy_orders (
        id             bigserial PRIMARY KEY,
        user_id        integer NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
        market_link_id integer,
        token_id       varchar(100),
        exchange       varchar(20),
        side           varchar(3),
        action         varchar(4),
        shares_req     numeric(12, 2),
        limit_price    numeric(6, 4),
        status         varchar(10) NOT NULL CHECK (status IN ('filled', 'partial', 'rejected')),
        reject_reason  varchar(40),
        book           jsonb,
        client_key     varchar(64),
        created_at     timestamptz NOT NULL DEFAULT now(),
        CHECK ((status = 'rejected') = (reject_reason IS NOT NULL)))""",
    "CREATE INDEX IF NOT EXISTS fantasy_orders_user ON accounts.fantasy_orders (user_id, created_at)",
    """CREATE UNIQUE INDEX IF NOT EXISTS fantasy_orders_client_key ON accounts.fantasy_orders (user_id, client_key)
       WHERE client_key IS NOT NULL""",
    """CREATE TABLE IF NOT EXISTS accounts.fantasy_fills (
        id             bigserial PRIMARY KEY,
        order_id       bigint NOT NULL REFERENCES accounts.fantasy_orders (id) ON DELETE CASCADE,
        user_id        integer NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
        market_link_id integer NOT NULL,
        side           varchar(3) NOT NULL,
        action         varchar(4) NOT NULL,
        shares         numeric(12, 2) NOT NULL CHECK (shares > 0),
        price          numeric(6, 4) NOT NULL CHECK (price > 0 AND price < 1),
        fee            numeric(12, 2) NOT NULL DEFAULT 0 CHECK (fee >= 0),
        ts             timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS fantasy_fills_order ON accounts.fantasy_fills (order_id)",
    """CREATE TABLE IF NOT EXISTS accounts.fantasy_positions (
        id             bigserial PRIMARY KEY,
        user_id        integer NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
        market_link_id integer NOT NULL,
        exchange       varchar(20) NOT NULL,
        yes_shares     numeric(12, 2) NOT NULL DEFAULT 0 CHECK (yes_shares >= 0),
        no_shares      numeric(12, 2) NOT NULL DEFAULT 0 CHECK (no_shares >= 0),
        cost           numeric(12, 2) NOT NULL DEFAULT 0,
        fees           numeric(12, 2) NOT NULL DEFAULT 0,
        status         varchar(10) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'settled', 'void')),
        outcome        boolean,
        payout         numeric(12, 2),
        settled_at     timestamptz,
        updated_at     timestamptz NOT NULL DEFAULT now(),
        UNIQUE (user_id, market_link_id),
        CHECK (NOT (yes_shares > 0 AND no_shares > 0)))""",
)
TABLES = ("fantasy_fills", "fantasy_orders", "fantasy_positions")     # in the order a reset deletes them


def on():
    """The fantasy pages and the order route: RACINGLINES_FANTASY=1, or unset on staging (owner, 2026-10-12: "staging
    only"). Production stays off until its env file sets the switch."""
    v = os.environ.get("RACINGLINES_FANTASY", "").strip().lower()
    if v:
        return v in ("1", "true", "yes", "on")
    return os.environ.get("RACINGLINES_ENV", "").strip().lower() == "staging"


def setup(conn):
    """Create the trading tables (idempotent; after web/accounts.DDL, which creates the schema and the ledger)."""
    for stmt in DDL:
        conn.execute(text(stmt))


def ready(conn):
    """True when setup has created the trading tables."""
    return conn.execute(text("SELECT to_regclass('accounts.fantasy_positions') IS NOT NULL")).scalar()


def cash(conn, user_id):
    """The account's F$ cash: the sum of its ledger rows (the grant, minus what its trades paid, plus what they got)."""
    return float(conn.execute(text("SELECT coalesce(sum(amount), 0) FROM accounts.fantasy_ledger WHERE user_id = :u"),
                              dict(u=user_id)).scalar())


# what the followed strategy's trades have done to the shared F$ cash (owner, 2026-10-12 21:04Z: "Only 1k to use for
# both"): what its buys paid and sales got (paper_positions.cash), plus the payout of every settled position.
STRATEGY_CASH = """
    SELECT coalesce(sum(cash + CASE WHEN outcome IS NULL THEN 0
                                    ELSE yes_shares * (CASE WHEN outcome THEN 1 ELSE 0 END)
                                         + no_shares * (CASE WHEN outcome THEN 0 ELSE 1 END) END), 0)
    FROM paper_positions WHERE user_id = :u AND venue <> 'private'"""


def available(conn, user_id):
    """F$ the player can trade with by hand: their cash plus what their strategy's trades have paid or got. Hand trades
    and the followed strategy share one F$1,000 (fantasy/sizing.player_bankroll is the strategy's side)."""
    return cash(conn, user_id) + float(conn.execute(text(STRATEGY_CASH), dict(u=user_id)).scalar())


def has_grant(conn, user_id):
    return conn.execute(text("SELECT 1 FROM accounts.fantasy_ledger WHERE user_id = :u LIMIT 1"),
                        dict(u=user_id)).first() is not None


def positions(conn, user_id, open_only=False, quotes=None):
    """The account's manual positions with their link and a mark: YES shares at the best bid, NO shares at 1 - the
    best ask (what they would sell for now, docs/fantasy-trading.md `liquidation_value`). quotes: {token_id: (bid, ask,
    ts)} from the recorded books (exchange.quotes); a token it lacks uses the link's last synced bid and ask. -> [dict]"""
    rs = conn.execute(text(f"""
        SELECT p.*, ml.question, ml.group_title, ml.outcome AS outcome_label, ml.prediction AS kind, ml.event_title,
               ml.token_id, ml.last_bid, ml.last_ask, ml.resolved_yes
        FROM accounts.fantasy_positions p LEFT JOIN market_links ml ON ml.id = p.market_link_id
        WHERE p.user_id = :u {"AND p.status = 'open' AND (p.yes_shares > 0 OR p.no_shares > 0)" if open_only else ""}
        ORDER BY p.updated_at DESC"""), dict(u=user_id)).mappings().all()
    out = []
    for r in rs:
        r = dict(r)
        q = (quotes or {}).get(r["token_id"])
        if q is not None:
            r["last_bid"], r["last_ask"] = q[0], q[1]
        for k in ("yes_shares", "no_shares", "cost", "fees", "payout"):
            r[k] = None if r[k] is None else float(r[k])
        r["side"] = "yes" if r["yes_shares"] > 0 else ("no" if r["no_shares"] > 0 else None)
        r["shares"] = r["yes_shares"] or r["no_shares"]
        r["value"] = mark(r)
        r["pnl"] = None if r["value"] is None else round(r["value"] - r["cost"], 2)
        out.append(r)
    return out


def mark(p):
    """What a position's held shares are worth now, or None when its side has no price."""
    if p["yes_shares"] > 0:
        return None if p["last_bid"] is None else round(p["yes_shares"] * float(p["last_bid"]), 2)
    if p["no_shares"] > 0:
        return None if p["last_ask"] is None else round(p["no_shares"] * (1 - float(p["last_ask"])), 2)
    return 0.0


def orders(conn, user_id, limit=20):
    """The account's latest orders, rejected ones included (with their reason). -> [dict]"""
    rs = conn.execute(text("""
        SELECT o.id, o.created_at, o.side, o.action, o.shares_req, o.limit_price, o.status, o.reject_reason,
               o.exchange, ml.question, ml.group_title,
               coalesce(sum(f.shares), 0) AS shares_filled,
               CASE WHEN sum(f.shares) > 0 THEN sum(f.shares * f.price) / sum(f.shares) END AS avg_price,
               coalesce(sum(f.fee), 0) AS fee
        FROM accounts.fantasy_orders o LEFT JOIN accounts.fantasy_fills f ON f.order_id = o.id
        LEFT JOIN market_links ml ON ml.id = o.market_link_id
        WHERE o.user_id = :u GROUP BY o.id, ml.id ORDER BY o.created_at DESC, o.id DESC LIMIT :n"""),
        dict(u=user_id, n=limit)).mappings().all()
    return [{k: (float(v) if k in ("shares_req", "limit_price", "shares_filled", "avg_price", "fee") and v is not None
                 else v) for k, v in r.items()} for r in rs]


# the strategy a player follows trades paper_positions (pipelines/signals.py); its P&L is the positions page's rule
# (web/views.py positions_page): cash + YES x y + NO x (1 - y), y = the outcome once settled, else the latest mark.
# The private book (venue 'private', a demo archived for the alpha) is not fantasy money and never counts.
STRATEGY_PNL = """
    SELECT user_id, sum(cash + yes_shares * y + no_shares * (1 - y)) AS pnl, count(*) AS n
    FROM (SELECT user_id, cash, yes_shares, no_shares,
                 coalesce(CASE WHEN outcome IS NULL THEN NULL WHEN outcome THEN 1.0 ELSE 0.0 END, mark) AS y
          FROM paper_positions WHERE venue <> 'private') p
    WHERE y IS NOT NULL GROUP BY user_id"""


def summary(conn, user_id, quotes=None):
    """One account's numbers: cash, open manual positions at their mark, the manual P&L, the followed strategy's P&L
    and the total F$ (the grant plus both P&Ls). -> dict"""
    c = cash(conn, user_id)
    grant = float(conn.execute(text("""SELECT coalesce(sum(amount), 0) FROM accounts.fantasy_ledger
                                       WHERE user_id = :u AND ref_table IS NULL"""), dict(u=user_id)).scalar())
    pos = positions(conn, user_id, open_only=True, quotes=quotes)
    held = sum(p["value"] or 0 for p in pos)
    unpriced = sum(p["value"] is None for p in pos)
    srow = conn.execute(text(f"SELECT pnl, n FROM ({STRATEGY_PNL}) s WHERE user_id = :u"), dict(u=user_id)).first()
    strat = float(srow[0]) if srow and srow[0] is not None else 0.0
    manual = round(c + held - grant, 2)
    return dict(grant=grant, cash=round(available(conn, user_id), 2), held=round(held, 2), open=len(pos), unpriced=unpriced,
                manual_pnl=manual, strategy_pnl=round(strat, 2), strategy_positions=int(srow[1]) if srow else 0,
                total=round(grant + manual + strat, 2))


def open_tokens(conn, user_id=None):
    """The token ids of open manual positions (one account's, or everyone's), for exchange.quotes."""
    return [t for (t,) in conn.execute(text(f"""
        SELECT DISTINCT ml.token_id FROM accounts.fantasy_positions p JOIN market_links ml ON ml.id = p.market_link_id
        WHERE p.status = 'open' AND (p.yes_shares > 0 OR p.no_shares > 0) {"AND p.user_id = :u" if user_id else ""}"""),
        dict(u=user_id))]


def leaderboard(conn, hide=(), quotes=None):
    """Every active account with a ledger, ranked by total F$ (grant + manual P&L + strategy P&L), best first. `hide`:
    usernames left out (the demo and system accounts). Open positions count at their mark (quotes as in positions);
    one with no price counts at nothing. -> [dict(rank, username, total, manual_pnl, strategy_pnl, cash, open)]"""
    rs = conn.execute(text(f"""
        WITH led AS (SELECT user_id, sum(amount) AS cash, sum(amount) FILTER (WHERE ref_table IS NULL) AS grant_
                     FROM accounts.fantasy_ledger GROUP BY user_id),
             st AS ({STRATEGY_PNL})
        SELECT u.id, u.username, led.cash, coalesce(led.grant_, 0) AS grant_, coalesce(st.pnl, 0) AS strategy_pnl
        FROM users u JOIN led ON led.user_id = u.id LEFT JOIN st ON st.user_id = u.id
        WHERE u.active AND NOT (u.username = ANY(:hide))"""), dict(hide=list(hide))).mappings().all()
    held = {}
    for r in conn.execute(text("""
            SELECT p.user_id, p.yes_shares, p.no_shares, ml.token_id, ml.last_bid, ml.last_ask
            FROM accounts.fantasy_positions p LEFT JOIN market_links ml ON ml.id = p.market_link_id
            WHERE p.status = 'open' AND (p.yes_shares > 0 OR p.no_shares > 0)""")).mappings():
        r = dict(r, yes_shares=float(r["yes_shares"]), no_shares=float(r["no_shares"]))
        q = (quotes or {}).get(r["token_id"])
        if q is not None:
            r["last_bid"], r["last_ask"] = q[0], q[1]
        h = held.setdefault(r["user_id"], [0.0, 0])
        h[0] += mark(r) or 0.0
        h[1] += 1
    out = []
    for r in rs:
        v, n = held.get(r["id"], (0.0, 0))
        manual = float(r["cash"]) + v - float(r["grant_"])
        strat = float(r["strategy_pnl"])
        out.append(dict(username=r["username"], cash=round(float(r["cash"]), 2), open=n,
                        manual_pnl=round(manual, 2), strategy_pnl=round(strat, 2),
                        total=round(float(r["grant_"]) + manual + strat, 2)))
    out.sort(key=lambda r: (-r["total"], r["username"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


# The leaderboard's score (owner, 2026-10-12 19:18Z: "roi times square root of total trades or something like that"):
# ROI = P&L / F$ staked, score = ROI x sqrt(trades). A player needs MIN_TRADES to be ranked, so one lucky trade can't
# top the board. Trades and stakes count both sources: an order filled by hand (staked = what its buys paid, fees
# included) and a market the followed strategy traded (staked = what it paid in, net of what it sold back).
MIN_TRADES = 5

TRADE_STATS = """
    WITH m AS (SELECT o.user_id, count(DISTINCT o.id) AS trades,
                      coalesce(sum(f.shares * f.price + f.fee) FILTER (WHERE f.action = 'buy'), 0) AS staked
               FROM accounts.fantasy_orders o JOIN accounts.fantasy_fills f ON f.order_id = o.id GROUP BY o.user_id),
         s AS (SELECT user_id, count(*) AS trades, coalesce(sum(greatest(-cash, 0)), 0) AS staked
               FROM paper_positions WHERE venue <> 'private' AND (yes_shares > 0 OR no_shares > 0 OR cash <> 0)
               GROUP BY user_id)
    SELECT coalesce(m.user_id, s.user_id) AS user_id, coalesce(m.trades, 0) + coalesce(s.trades, 0) AS trades,
           coalesce(m.staked, 0) + coalesce(s.staked, 0) AS staked
    FROM m FULL JOIN s ON s.user_id = m.user_id"""


def score(pnl, staked, trades):
    """(roi, score) for the leaderboard; roi is None with nothing staked, score None below MIN_TRADES."""
    roi = pnl / staked if staked > 0 else None
    return roi, (round(roi * trades ** 0.5, 4) if roi is not None and trades >= MIN_TRADES else None)


def ranked(conn, hide=(), quotes=None):
    """leaderboard() with each player's trades, F$ staked, ROI and score, ranked by score (best first); players below
    MIN_TRADES follow, unranked, by P&L. The page shows P&L and ROI, never a player's balance (owner, 2026-10-12
    22:36Z). -> [dict(..., trades, staked, pnl, roi, score, rank or None)]"""
    rows = leaderboard(conn, hide=hide, quotes=quotes)
    st = {r[0]: (int(r[1]), float(r[2])) for r in conn.execute(text(TRADE_STATS))}
    ids = dict(conn.execute(text("SELECT username, id FROM users WHERE username = ANY(:n)"),
                            dict(n=[r["username"] for r in rows])).all())
    for r in rows:
        r["trades"], r["staked"] = st.get(ids.get(r["username"]), (0, 0.0))
        r["pnl"] = round(r["manual_pnl"] + r["strategy_pnl"], 2)
        r["roi"], r["score"] = score(r["pnl"], r["staked"], r["trades"])
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0), -r["pnl"], r["username"]))
    n = 0
    for r in rows:
        r["rank"] = (n := n + 1) if r["score"] is not None else None
    return rows


def manual_curve(conn, user_id, quotes=None):
    """The account's F$ by hand through the run: after each order, cash plus what its open positions cost (so a trade
    moves it only by its fee until it is sold or settled), then a last point now with open positions at their mark.
    -> [(ts, value)]"""
    rs = conn.execute(text("""
        SELECT o.created_at AS ts, f.action, f.shares * f.price AS amt, f.fee
        FROM accounts.fantasy_orders o JOIN accounts.fantasy_fills f ON f.order_id = o.id
        WHERE o.user_id = :u ORDER BY o.created_at, f.id"""), dict(u=user_id)).mappings().all()
    grant = float(conn.execute(text("""SELECT coalesce(sum(amount), 0) FROM accounts.fantasy_ledger
                                       WHERE user_id = :u AND ref_table IS NULL"""), dict(u=user_id)).scalar())
    if not rs:
        return []
    pts, v = [], grant
    for r in rs:
        v -= float(r["fee"])                       # at cost, a buy or a sell changes the value only by its fee ...
        pts.append((r["ts"], round(v, 2)))
    s = summary(conn, user_id, quotes=quotes)      # ... and the last point marks the open positions (and realised P&L)
    from datetime import datetime, timezone
    pts.append((datetime.now(timezone.utc), round(grant + s["manual_pnl"], 2)))
    return pts
