"""
Beta sign-up and the fantasy-bucks ledger.

Self-serve accounts (RACINGLINES_SIGNUP=1, off by default): /signup takes a username, a password and a tier (pro or
basic) and creates a `users` row (the password as a scrypt hash, web/users.py) plus a SIGNUP_GRANT credit in the
ledger, in one transaction, then signs the person in.

The ledger lives in its own Postgres schema, `accounts`, apart from the market data in `public`: one row per credit
or debit (`amount`, `reason`), never updated, so a balance is a sum and every change has a history. No wallet or
bankroll logic reads it yet; it only has to be right from the first sign-up.

The schema is NOT an alembic migration, because a deploy runs `alembic upgrade head` and deploys never change the
VM's database (owner rule, 2026-10-01). `racinglines users setup` creates it (idempotent; on the VM through
`vm.sh accounts`, which backs up first). Until it exists, sign-up says it isn't open yet and nothing else changes.
"""

import re
import secrets

from sqlalchemy import text

SIGNUP_GRANT = 1000                       # fantasy bucks every new account starts with
GRANT_REASON = "signup_grant"

DDL = (
    "CREATE SCHEMA IF NOT EXISTS accounts",
    """CREATE TABLE IF NOT EXISTS accounts.fantasy_ledger (
        id         bigserial PRIMARY KEY,
        user_id    integer NOT NULL REFERENCES public.users (id) ON DELETE CASCADE,
        amount     numeric(14, 2) NOT NULL,
        reason     varchar(40) NOT NULL,
        note       text,
        created_at timestamptz NOT NULL DEFAULT now())""",
    "CREATE INDEX IF NOT EXISTS fantasy_ledger_user ON accounts.fantasy_ledger (user_id)",
    # one signup grant per account, however often setup or sign-up runs
    f"""CREATE UNIQUE INDEX IF NOT EXISTS fantasy_ledger_one_grant ON accounts.fantasy_ledger (user_id)
        WHERE reason = '{GRANT_REASON}'""",
)

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,29}$")
RESERVED = {"admin", "administrator", "root", "system", "staff", "support", "hello", "help", "racinglines",
            "maker", "taker", "pro", "basic", "demo", "polymarket-takers", "kalshi", "polymarket", "null", "none"}
PASSWORD_MIN, PASSWORD_MAX = 10, 128


def setup(engine, backfill=True):
    """Create the accounts schema and ledger (idempotent). backfill: every active account without a signup grant
    gets one now, so every user starts with SIGNUP_GRANT. Returns the number of grants added."""
    from racinglines.fantasy import ledger as FL
    with engine.begin() as c:
        for stmt in DDL:
            c.execute(text(stmt))
        FL.setup(c)                          # the manual-trading tables (fantasy/ledger.py), same schema
        if not backfill:
            return 0
        return c.execute(text(f"""
            INSERT INTO accounts.fantasy_ledger (user_id, amount, reason, note)
            SELECT u.id, :g, '{GRANT_REASON}', 'backfill at setup' FROM users u
            WHERE u.active AND NOT EXISTS (SELECT 1 FROM accounts.fantasy_ledger l
                                           WHERE l.user_id = u.id AND l.reason = '{GRANT_REASON}')"""),
                         dict(g=SIGNUP_GRANT)).rowcount


def ready(conn):
    """True when the ledger exists (setup has run)."""
    return conn.execute(text("SELECT to_regclass('accounts.fantasy_ledger') IS NOT NULL")).scalar()


def grant(conn, user_id, amount=SIGNUP_GRANT, reason=GRANT_REASON, note=None):
    conn.execute(text("INSERT INTO accounts.fantasy_ledger (user_id, amount, reason, note) VALUES (:u, :a, :r, :n)"),
                 dict(u=user_id, a=amount, r=reason, n=note))


# Per-user trading history the reset clears (owner, 2026-10-11: "seed all users with 1k fantasy bucks, and zero trading
# history"): (table, user column). house_markets a user quotes as a maker are not history and stay. The reset also clears
# every strategy profile (users.prefs strategy_profile*, pipelines/profiles.pref): "no default strategy until they
# select one" (owner, 2026-10-11 22:04Z).
HISTORY = (("strategy_signals", "user_id"), ("paper_positions", "user_id"), ("house_bets", "taker_id"))
KEEP = ("maker", "taker")                 # the demo accounts (web/demo.DEMO_USERS): their history never changes here


PROFILE_KEYS = "SELECT k FROM jsonb_object_keys(coalesce(prefs, '{}'::jsonb)) k WHERE k LIKE 'strategy\\_profile%'"


def reset(conn, keep=KEEP, apply=False, note="reset"):
    """Every active account but `keep` back to one SIGNUP_GRANT, no trading history and no strategy: its fantasy_ledger
    rows and HISTORY rows and manual trades (fantasy/ledger.TABLES) deleted, its strategy profiles removed, one signup grant written (`note`). The ledger is append-only everywhere else; this is the one
    deliberate rewrite, run on the VM after a backup (scripts/vm/reset_fantasy.sh). apply=False (the default) only
    counts. Returns one dict per account: id, username, kept, ledger rows and balance, its profiles (pref keys) and each
    HISTORY table's rows."""
    from racinglines.fantasy import ledger as FL
    # apply: lock the rows first, so a signals pass storing right now (signals.still_assigned, FOR SHARE) either commits
    # before the deletes below or waits and then finds no profile; never positions written back after the reset
    users = conn.execute(text("SELECT id, username FROM users WHERE active ORDER BY id"
                              + (" FOR UPDATE" if apply else ""))).all()
    out = []
    for uid, name in users:
        row = dict(id=uid, username=name, kept=name in keep)
        led = conn.execute(text("SELECT count(*), coalesce(sum(amount), 0) FROM accounts.fantasy_ledger WHERE user_id = :u"),
                           dict(u=uid)).one()
        row.update(ledger_rows=int(led[0]), balance=float(led[1]),
                   profiles=conn.execute(text(f"SELECT count(*) FROM users, LATERAL ({PROFILE_KEYS}) p WHERE id = :u"),
                                         dict(u=uid)).scalar())
        for table, col in HISTORY:
            row[table] = conn.execute(text(f"SELECT count(*) FROM {table} WHERE {col} = :u"), dict(u=uid)).scalar()
        out.append(row)
        trading = FL.ready(conn)                 # manual trades (fantasy/ledger.py), once setup has created them
        row["manual_orders"] = conn.execute(text("SELECT count(*) FROM accounts.fantasy_orders WHERE user_id = :u"),
                                            dict(u=uid)).scalar() if trading else 0
        if apply and not row["kept"]:
            for table, col in HISTORY:
                conn.execute(text(f"DELETE FROM {table} WHERE {col} = :u"), dict(u=uid))
            for table in FL.TABLES if trading else ():
                conn.execute(text(f"DELETE FROM accounts.{table} WHERE user_id = :u"), dict(u=uid))
            conn.execute(text("DELETE FROM accounts.fantasy_ledger WHERE user_id = :u"), dict(u=uid))
            conn.execute(text(f"UPDATE users SET prefs = coalesce(prefs, '{{}}'::jsonb) - ARRAY({PROFILE_KEYS}) WHERE id = :u"),
                         dict(u=uid))
            grant(conn, uid, note=note)
    return out


def balances(conn):
    """{user_id: balance} for every account with a ledger row; {} before setup."""
    if not ready(conn):
        return {}
    rows = conn.execute(text("SELECT user_id, sum(amount) FROM accounts.fantasy_ledger GROUP BY user_id")).all()
    return {u: float(b) for u, b in rows}


def check_username(conn, username):
    """'' if the (lower-cased) username is free and valid, else the message to show."""
    if not USERNAME_RE.match(username):
        return "Usernames are 3 to 30 characters: lower-case letters, digits, '.', '_' or '-', starting with a letter or digit."
    if username in RESERVED or username.startswith("lab-"):     # lab-<id>: the Lab monitor accounts (profiles.ensure_monitors)
        return "That username is reserved. Please pick another."
    if conn.execute(text("SELECT 1 FROM users WHERE lower(username) = :u"), dict(u=username)).first():
        return "That username is taken. Please pick another."
    return ""


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
EMAIL_MAX = 254


def check_email(conn, email):
    """'' if the (trimmed, lower-cased) email is valid and on no other account, else the message to show. New sign-ups
    need one (owner, 2026-10-08); accounts made before then have none and keep working without it. Stored, like the
    settings page's, as users.prefs.email."""
    if not email or len(email) > EMAIL_MAX or not EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if conn.execute(text("SELECT 1 FROM users WHERE lower(prefs->>'email') = :e"), dict(e=email)).first():
        return "That email is already on an account. Sign in, or use another address."
    return ""


def check_password(password, confirm, username):
    if len(password) < PASSWORD_MIN:
        return f"Passwords need at least {PASSWORD_MIN} characters."
    if len(password) > PASSWORD_MAX:
        return f"Passwords can be at most {PASSWORD_MAX} characters."
    if username and username in password.lower():
        return "Your password can't contain your username."
    if password != confirm:
        return "The two passwords don't match."
    return ""


def temp_password():
    """A one-time password for an admin reset: shown to the admin once, never stored or logged in clear."""
    return secrets.token_urlsafe(12)
