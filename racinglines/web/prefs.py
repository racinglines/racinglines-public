"""Per-user settings that should follow the account to any device (users.prefs, JSONB): Edge Finder
combos and last-used knobs per job type. Pure view state (open sections, filters) stays in the
browser; history (jobs, runs, bets) lives in its own tables.

`user` is the request's user dict (or a bare user id). For a demo account's session (web/demo.py) reads
layer the session's overlay on the saved baseline and writes go to the overlay only, so a demo visitor's
changes never persist beyond the session."""

import json

from sqlalchemy import text


def _uid(user):
    return user.get("id") if isinstance(user, dict) else user


def _overlay(user):
    from racinglines.web import demo
    if isinstance(user, dict) and demo.is_demo(user) and user.get("sid"):
        return demo.prefs(user["sid"])
    return None


def get(conn, user):
    uid = _uid(user)
    if not uid:
        return {}
    base = conn.execute(text("SELECT prefs FROM users WHERE id = :u"), dict(u=uid)).scalar() or {}
    ov = _overlay(user)
    return {**base, **ov} if ov else base


def put(conn, user, key, value):
    """Set one top-level key (the others are kept)."""
    uid = _uid(user)
    if not uid:
        return
    ov = _overlay(user)
    if ov is not None:
        ov[key] = json.loads(json.dumps(value))
        return
    conn.execute(text("""UPDATE users SET prefs = coalesce(prefs, '{}'::jsonb) || jsonb_build_object(CAST(:k AS text), CAST(:v AS jsonb))
                         WHERE id = :u"""), dict(k=key, v=json.dumps(value), u=uid))
    conn.commit()
