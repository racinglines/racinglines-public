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


def sports(code):
    return tuple(load(code).get("sports", {}))


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
