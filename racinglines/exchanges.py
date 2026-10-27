"""
Exchange schemas: an exchange as data (exchanges/<code>.toml), the way sports are (sports/<code>.toml).

A venue with a plain JSON API is described by one file: its host and endpoints, where each field lives in a
response, its price units and fee, and how its markets map onto our prediction kinds per sport. One generic driver
(racinglines/markets/exchange_driver.py) reads it, so a new exchange is a new file, not new code. Polymarket and
Kalshi predate this and keep their own packages; they can move onto schemas later.

    load("og")        the parsed schema
    CODES             every exchange with a schema
    enabled("og")     whether its switch (an env var, off by default) is on
    sports("og")      the sports it lists (sports/<code>.toml codes)
    audience("og")    who the app shows it to ([exchange] audience): "all" (the default) or "admin"
    admin_only()      the exchange codes whose audience is "admin"
    is_public(code, user)   the one gate (web pages and MCP tools): may this user see the exchange's quotes?
    dig(obj, path)    obj["a"]["b"][0] for "a.b.0", None when any step is missing

An endpoint can state the exchange's own caps as `limits = { max_count = 150, max_batch = 10, max_window_days = 31 }`.
`load` checks the endpoint's `params.count`, `batch_size` and `window_days` against them, so a schema that asks for
more than the exchange allows fails when it is loaded, not on the first live call.
"""

import os
import tomllib
from functools import cache

from racinglines.paths import ROOT

SCHEMAS = ROOT / "exchanges"
CODES = tuple(sorted(p.stem for p in SCHEMAS.glob("*.toml")))


# limit name -> (where the endpoint asks for it, what it means)
LIMITS = {"max_count": ("params.count", "rows per call"), "max_batch": ("batch_size", "ids per call"),
          "max_window_days": ("window_days", "days per window")}


def check_limits(code, schema):
    """Raise ValueError naming every endpoint whose request exceeds a cap its own `limits` states."""
    bad = []
    for name, ep in schema.get("endpoints", {}).items():
        for limit, cap in ep.get("limits", {}).items():
            if limit not in LIMITS:
                bad.append(f"{code} endpoints.{name}: unknown limit {limit!r} (known: {', '.join(LIMITS)})")
                continue
            where, what = LIMITS[limit]
            asked = dig(ep, where)
            if asked is not None and asked > cap:
                bad.append(f"{code} endpoints.{name}: asks for {asked} {what}, the exchange allows {cap} ({limit})")
    if bad:
        raise ValueError("; ".join(bad))


@cache
def load(code):
    schema = tomllib.loads((SCHEMAS / f"{code}.toml").read_text())
    check_limits(code, schema)
    return schema


def enabled(code):
    """The exchange's switch env var (schema [exchange] switch) is 1 / true / yes. Off unless set."""
    return os.environ.get(load(code)["exchange"].get("switch", ""), "").lower() in ("1", "true", "yes")


AUDIENCES = ("all", "admin")


def audience(code):
    """Who the app shows the exchange to ([exchange] audience), as [sport] audience does for a sport (sports.audience):
    "all" (the default) or "admin" (owner, 2026-10-21, ProphetX for testing). Only what users see changes: the exchange is
    still recorded, and the CLI and admins see all of it."""
    got = load(code)["exchange"].get("audience", "all")
    if got not in AUDIENCES:
        raise ValueError(f"exchanges/{code}.toml [exchange] audience: {got!r} not in {AUDIENCES}")
    return got


def admin_only():
    """The exchange codes the app shows to admins only ([exchange] audience = "admin")."""
    return frozenset(c for c in CODES if audience(c) == "admin")


def is_public(code, user=None):
    """The exchange gate every user-facing reader calls (sports.is_public's twin): True for an admin, for None (an
    internal caller: the CLI, the recorder, the stdio MCP server) and for a code without a schema (Polymarket, Kalshi);
    else whether the exchange's audience is "all". A signed-out web visitor is {} (web/audience.py), not None."""
    if user is None or code not in CODES or (isinstance(user, dict) and user.get("role") == "admin"):
        return True
    return audience(code) == "all"


def sports(code):
    """The sports the exchange's schema maps ([sports.<code>]), keeping only those whose sports/<code>.toml is
    installed: the public release ships some sports only, and a page that loads every listed sport would fail."""
    return tuple(s for s in load(code).get("sports", {}) if (ROOT / "sports" / f"{s}.toml").exists())


def dig(obj, path):
    """A dotted path into nested dicts and lists: "a.b.0" -> obj["a"]["b"][0]; None when a step is missing."""
    for step in str(path).split("."):
        if isinstance(obj, dict):
            obj = obj.get(step)
        elif isinstance(obj, (list, tuple)) and step.lstrip("-").isdigit() and -len(obj) <= int(step) < len(obj):
            obj = obj[int(step)]
        else:
            return None
        if obj is None:
            return None
    return obj
