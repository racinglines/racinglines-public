"""
Accounts, roles and the activity log for the web app.

Roles (racinglines/web/roles.py: admin, pro, basic; `maker` / `taker` rows are read as pro / basic):
    admin   everything, plus the activity log, user management and the database explorer
    pro     predictions/model runs, their own quoted markets (fair value +- spread), every strategy, the Lab
    basic   open markets, a limited set of taker strategies, their own bets and P&L

Passwords are hashed with scrypt (standard library): "scrypt$n$r$p$salt$hash".
"""

import hashlib
import hmac
import ipaddress
import json
import os
import secrets

from sqlalchemy import insert, select

from racinglines.db import models as m
from racinglines.web.roles import ROLES, canonical  # noqa: F401  (ROLES: the values create_user accepts)
_N, _R, _P = 2 ** 14, 8, 1


def hash_password(password):
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${h.hex()}"


def verify_password(password, stored):
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(got.hex(), h)
    except (ValueError, AttributeError):
        return False


def get_user(session, user_id=None, username=None):
    q = select(m.User).filter_by(id=user_id) if user_id is not None else select(m.User).filter_by(username=username)
    return session.scalars(q).first()


def authenticate(session, username, password):
    u = get_user(session, username=username)
    return u if u and u.active and verify_password(password, u.password_hash) else None


# The replay counterparty: Polymarket's takers, whose real trades filled a replayed maker (web/diag.py).
# A system account, not a person: it can't sign in, and it is NOT the demo `taker` login (a basic account).
REPLAY_TAKER = "polymarket-takers"


def ensure_replay_taker(session):
    """The system account that records Polymarket takers' fills of a replayed maker (inactive: no login)."""
    u = get_user(session, username=REPLAY_TAKER)
    if u is None:
        u = m.User(username=REPLAY_TAKER, password_hash="!no-login", role="basic", active=False,
                   display_name="Polymarket takers (replay counterparty)")
        session.add(u)
        session.commit()
    return u


def create_user(session, username, password, role, display_name=None):
    role = canonical(role)
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if get_user(session, username=username):
        raise ValueError(f"user {username!r} already exists")
    u = m.User(username=username.strip(), password_hash=hash_password(password), role=role,
               display_name=display_name or username)
    session.add(u)
    session.commit()
    return u


def ensure_admin(session):
    """Create or update the admin account from ADMIN_USERNAME / ADMIN_PASSWORD.
    If no password is set and the admin doesn't exist yet, generate one and print it."""
    username = os.environ.get("ADMIN_USERNAME", "admin")
    password = os.environ.get("ADMIN_PASSWORD")
    u = get_user(session, username=username)
    if u is None:
        if not password:
            password = secrets.token_urlsafe(16)
            print(f"Created admin account: {username} / {password}", flush=True)
        session.add(m.User(username=username, password_hash=hash_password(password), role="admin",
                           display_name="Admin"))
    elif password and not verify_password(password, u.password_hash):
        u.password_hash = hash_password(password)
        u.role, u.active = "admin", True
    session.commit()


DEMO_PASSWORD = "password"          # the sign-in page's "Try the demo" button posts maker / this (signin.html, login.html)
DEMO_ROLES = {"maker": "pro", "taker": "basic"}


def ensure_demo_users(session):
    """The demo accounts (web/demo.DEMO_USERS) for `racinglines web --demo`, so "Try the demo" works on a fresh
    install: each one missing is created with DEMO_PASSWORD (maker pro, taker basic; any other name basic). An
    existing account is left as it is. Returns the names created."""
    from racinglines.web import demo
    made = []
    for name in sorted(demo.DEMO_USERS):
        if get_user(session, username=name) is None:
            session.add(m.User(username=name, password_hash=hash_password(DEMO_PASSWORD),
                               role=DEMO_ROLES.get(name, "basic"), display_name=name.capitalize()))
            made.append(name)
    session.commit()
    return made


def log(engine, user, action, request=None, **detail):
    """Append to activity_log. `user` may be None (e.g. failed login)."""
    with engine.begin() as c:
        c.execute(insert(m.ActivityLog).values(
            user_id=user["id"] if user else None, username=(user or {}).get("username") or detail.pop("username", None),
            role=(user or {}).get("role"), action=action,
            detail=json.loads(json.dumps(detail, default=str)) if detail else None,
            ip=_ip(request) if request is not None else None,
            path=str(request.url.path) if request is not None else None))


def _trusted_proxy(peer):
    """Whether the direct peer may set the client address: loopback (cloudflared on the VM forwards racinglines.bet
    to 127.0.0.1:8000, docs/vm-deploy.md), or any peer with RACINGLINES_TRUST_PROXY=1 (another reverse proxy).
    A self-hosted app reached directly ignores the headers, so a client can't rotate its address past the limits."""
    if os.environ.get("RACINGLINES_TRUST_PROXY") == "1":
        return True
    try:
        a = ipaddress.ip_address(peer or "")
    except ValueError:
        return False
    return a.is_loopback or bool(getattr(a, "ipv4_mapped", None) and a.ipv4_mapped.is_loopback)


def _ip(request):
    """The client address: the proxy's cf-connecting-ip / x-forwarded-for from a trusted peer, else the peer."""
    peer = request.client.host if request.client else None
    if _trusted_proxy(peer):
        fwd = (request.headers.get("cf-connecting-ip")
               or request.headers.get("x-forwarded-for", "").split(",")[0].strip())
        if fwd:
            return fwd
    return peer
