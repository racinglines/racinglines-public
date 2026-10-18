"""
Demo accounts are disposable per browser session. Every sign-in to a demo account (DEMO_USERS: the
one-click `maker` / `taker` logins) gets a fresh session id, and within that session:

- **View settings** (Edge Finder combos and season, job settings, which signals were seen) live in a
  server-side overlay for that session only, on top of the account's saved baseline, which a demo session
  never changes. A new sign-in starts from the baseline again; two visitors on the same demo account never
  see each other's changes. Browser storage is namespaced by the session id (base.html, lab.html).
- **Anything that would create or change data** (Lab jobs, scenario promotion, candidates, the in-app book,
  bets, Polymarket sync / mirror, recorded replays) is refused with a short message. Only ALLOWED posts
  (view settings) go through, into the overlay.
- **Every interaction is logged** to activity_log (action demo_view / demo_post / demo_blocked) with the
  session id, method, path and query, so a visit can be replayed later (Admin > Activity).

    RACINGLINES_DEMO_USERS=maker,taker     (default) the accounts treated as demo accounts
"""

import os
import secrets
import threading
import time

from fastapi import HTTPException, Request, status

DEMO_USERS = {u.strip() for u in os.environ.get("RACINGLINES_DEMO_USERS", "maker,taker").split(",") if u.strip()}
ALLOWED_POSTS = ("/lab/edge", "/logout")          # view settings only; they land in the session overlay
TTL = 12 * 3600                                    # seconds: the session cookie's lifetime
QUIET = ("/static", "/favicon")                    # never logged

_lock = threading.Lock()
_sessions = {}                                     # sid -> dict(t=last seen, prefs={...})


def is_demo(user):
    return bool(user) and user.get("username") in DEMO_USERS


def new_sid():
    return secrets.token_hex(8)


def _state(sid):
    now = time.time()
    with _lock:
        for k in [k for k, v in _sessions.items() if now - v["t"] > TTL]:
            del _sessions[k]
        st = _sessions.setdefault(sid, dict(t=now, prefs={}))
        st["t"] = now
        return st


def reset(sid):
    with _lock:
        _sessions.pop(sid, None)


def prefs(sid):
    """This session's overlay of preferences (a live dict)."""
    return _state(sid)["prefs"]


def guard(request: Request):
    """App-wide dependency (after authentication): log a demo session's every request and refuse its writes."""
    user = getattr(request.state, "user", None)
    if not is_demo(user) or request.url.path.startswith(QUIET):
        return
    from racinglines.web import users as U
    from racinglines.db.config import get_engine
    sid = user.get("sid") or "no-session"
    detail = dict(sid=sid, method=request.method, query=str(request.url.query) or None)
    if request.method in ("GET", "HEAD"):
        U.log(get_engine(), user, "demo_view", request, **detail)
        return
    if request.url.path.startswith(ALLOWED_POSTS):
        U.log(get_engine(), user, "demo_post", request, **detail)
        return
    U.log(get_engine(), user, "demo_blocked", request, **detail)
    msg = "That isn't available in the demo: demo accounts can look around and change views, but not create or change data."
    if request.headers.get("sec-fetch-mode", "navigate") == "navigate":       # a form post: back to the page, with the message
        back = request.headers.get("referer") or "/markets"
        sep = "&" if "?" in back else "?"
        raise HTTPException(status.HTTP_303_SEE_OTHER, headers={"Location": f"{back.split('#')[0]}{sep}msg=Error: {msg}"})
    raise HTTPException(status.HTTP_403_FORBIDDEN, msg)                        # a fetch (the Lab's buttons)
