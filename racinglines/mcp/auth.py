"""
Who may reach the hosted server (`racinglines mcp --http`): real web-app accounts, each with its own bearer token.

    racinglines mcp token admin            # issue (or replace) the admin account's token; prints it once
    racinglines mcp token admin --revoke

A token is `rl_` + 48 hex characters; only its SHA-256 is stored, in users.prefs["mcp"] (issued_at, token_sha256),
so the users table keeps no secret that could be replayed. A request is accepted when its token matches an active,
non-demo account whose role is in RACINGLINES_MCP_ROLES (default `admin,pro`; set `admin` for admin-only access).
Pro callers can access only their own account records and jobs. Stdio mode has no tokens: whoever can run the
command on that machine is the owner.
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timezone

from sqlalchemy import text

from racinglines.web import roles as R

PREFIX = "rl_"
ROLES = tuple(r.strip() for r in os.environ.get("RACINGLINES_MCP_ROLES", "admin,pro").split(",") if r.strip())


def _demo_users():
    from racinglines.web import demo
    return demo.DEMO_USERS


def _sha(token):
    return hashlib.sha256(token.encode()).hexdigest()


def new_token(engine, username):
    """Issue a fresh token for `username` (replacing any earlier one) and return it: the only time it is shown."""
    with engine.begin() as c:
        row = c.execute(text("SELECT id, role, active FROM users WHERE username = :u"), dict(u=username)).fetchone()
        if row is None:
            raise ValueError(f"no account {username!r}")
        if username in _demo_users():
            raise ValueError(f"{username!r} is a demo account: MCP access is for real accounts")
        if not row.active:
            raise ValueError(f"{username!r} is inactive")
        if R.canonical(row.role) not in ROLES:
            raise ValueError(f"{username!r} has role {row.role!r}; RACINGLINES_MCP_ROLES allows {', '.join(ROLES)}")
        token = PREFIX + secrets.token_hex(24)
        meta = dict(token_sha256=_sha(token), issued_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        c.execute(text("""UPDATE users SET prefs = coalesce(prefs, '{}'::jsonb) || jsonb_build_object('mcp', CAST(:m AS jsonb))
                          WHERE id = :i"""), dict(m=__import__("json").dumps(meta), i=row.id))
    return token


def revoke(engine, username):
    with engine.begin() as c:
        n = c.execute(text("UPDATE users SET prefs = coalesce(prefs, '{}'::jsonb) - 'mcp' WHERE username = :u AND prefs ? 'mcp'"),
                      dict(u=username)).rowcount
    return bool(n)


def lookup(engine, token):
    """The account a bearer token belongs to: dict(id, username, role), or None."""
    if not token or not token.startswith(PREFIX):
        return None
    want = _sha(token)
    with engine.connect() as c:
        rows = c.execute(text("""SELECT id, username, role, prefs->'mcp'->>'token_sha256' AS h FROM users
                                 WHERE active AND prefs ? 'mcp'""")).fetchall()
    for r in rows:
        if r.h and hmac.compare_digest(r.h, want) and R.canonical(r.role) in ROLES and r.username not in _demo_users():
            return dict(id=int(r.id), username=r.username, role=R.canonical(r.role))
    return None


def holders(engine):
    """Accounts with a token: [(username, role, issued_at)]."""
    with engine.connect() as c:
        return [tuple(r) for r in c.execute(text("""SELECT username, role, prefs->'mcp'->>'issued_at' FROM users
                                                    WHERE prefs ? 'mcp' ORDER BY username""")).fetchall()]
