"""
Which sports a signed-in user sees ([sport] audience in sports/<code>.toml): the web app's side of the one gate,
racinglines.sports.is_public, which the MCP tools call too. An "admin" sport is left out of every page for everyone but
admins, and its pages answer 404. Only user-facing readers ask: recording, pricing, settlement and the CLI carry on,
and no data is deleted. A signed-out visitor (user None here) is a user, not an internal caller.

    hidden(user)            the sport codes this user doesn't see (none for an admin)
    hidden_comps(user)      their competition codes
    shown(user, code)       False when a sport or competition code is hidden from this user
    guard(user, code)       404 when it is (a race, event, season or run page of a hidden sport)
    guard_athlete(conn, user, athlete_id)   404 for an athlete who only raced (or was priced) in hidden sports
"""

from fastapi import HTTPException

from racinglines import sports as SP


def _who(user):
    return user if isinstance(user, dict) else {}             # signed out: a visitor, never an internal caller


def hidden(user):
    return frozenset(c for c in SP.SPORT_CODES if not SP.is_public(c, _who(user)))


def hidden_comps(user):
    return frozenset(SP.load(c)["competition"]["code"] for c in hidden(user))


def shown(user, code):
    return SP.is_public(code, _who(user))


def guard(user, code):
    if not shown(user, code):
        raise HTTPException(404)


def guard_athlete(conn, user, athlete_id):
    hide = hidden_comps(user)
    if not hide:
        return
    from racinglines.db import reads
    comps = reads.athlete_competitions(conn, athlete_id)
    if comps and comps <= hide:
        raise HTTPException(404)
