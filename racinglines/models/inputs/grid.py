"""
Confirmed grid penalties (sports/f1/grid_penalties.toml), applied to a race's grid before it is simulated.

    load(path=None)                          every [[penalty]] row, validated ([] when the file or table is missing)
    for_event(event_key, session, cutoff)    the rows for one event ("2026-17") and session ("race", "sprint")
    apply(grid, entrants, penalties)         the grid with the penalised drivers moved, renumbered 1..n

The pricer (pricing.price_race / price_stages) hands apply to model.simulate_race as its `grid_adjust`, which runs it
on the known or the drawn grid after every draw, so no random draw is added or moved and a weekend without a
penalty prices byte-identically. Pure numpy, no RNG.

Kinds: "back" and "pit_lane" (priced as a back-of-grid start) go behind everyone else, in their qualifying order;
"places" drops a driver N places, the drivers behind him closing up, never behind the back-of-grid group.
"""

import re
import tomllib
import unicodedata
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines import sports
from racinglines.paths import ROOT

PATH = ROOT / "sports" / sports.declaring("grid") / "grid_penalties.toml"   # the sport declaring the grid input
KINDS = ("back", "pit_lane", "places")
REQUIRED = ("event", "driver", "session", "kind", "source")
EVENT_KEY = re.compile(r"^\d{4}-\d{2}$")


class PenaltyError(ValueError):
    pass


def normalise(name):
    """The exact-match key for a driver: lowercase, accents and punctuation dropped, single spaces."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _timestamp(v):
    ts = pd.Timestamp(v)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def _validate(i, row):
    from racinglines.models.inputs import laps as M
    where = f"grid penalty {i + 1} ({row.get('event')}, {row.get('driver')})"
    missing = [k for k in REQUIRED if not row.get(k)]
    if missing:
        raise PenaltyError(f"{where}: missing {missing}")
    if not EVENT_KEY.match(str(row["event"])):
        raise PenaltyError(f"{where}: event must be YYYY-RR, got {row['event']!r}")
    if row["session"] not in M.SIM_SESSIONS:
        raise PenaltyError(f"{where}: session must be one of {list(M.SIM_SESSIONS)}, got {row['session']!r}")
    if row["kind"] not in KINDS:
        raise PenaltyError(f"{where}: kind must be one of {list(KINDS)}, got {row['kind']!r}")
    places = row.get("places")
    if row["kind"] == "places" and (not isinstance(places, int) or isinstance(places, bool) or places < 1):
        raise PenaltyError(f"{where}: kind 'places' needs places = N (a positive integer), got {places!r}")
    if row["kind"] != "places" and places is not None:
        raise PenaltyError(f"{where}: places is only for kind 'places'")
    out = dict(row, driver_key=normalise(row["driver"]))
    if "announced" in row:
        out["announced"] = _timestamp(row["announced"])
    return out


@cache
def _load(path, mtime_ns):
    with open(path, "rb") as f:
        rows = tomllib.load(f).get("penalty", [])
    return tuple(_validate(i, dict(r)) for i, r in enumerate(rows))


def load(path=None):
    """Every penalty row of the file (default sports/f1/grid_penalties.toml), validated; [] when it is missing."""
    path = Path(PATH if path is None else path)
    if not path.is_file():
        return []
    mtime = path.stat().st_mtime_ns
    return [dict(r) for r in _load(path, mtime)]


def for_event(event_key, session, cutoff=None, path=None):
    """The penalties for one event and session. cutoff: leave out rows announced after it (the as-of rule)."""
    if event_key is None:
        return []
    cut = None if cutoff is None else _timestamp(cutoff)
    return [r for r in load(path) if r["event"] == event_key and r["session"] == session
            and (cut is None or "announced" not in r or r["announced"] <= cut)]


def driver_index(entrants, penalty):
    """The entrant row a penalty names: an exact match of its normalised driver key on the entrant's full name or
    surname (the last word of `driver`). No match, or a surname two entrants share, raises PenaltyError."""
    names = entrants["driver"].map(normalise).tolist()
    key = penalty["driver_key"]
    hits = [i for i, n in enumerate(names) if n == key]
    if not hits:
        hits = [i for i, n in enumerate(names) if n.split(" ")[-1:] == [key]]
    if len(hits) != 1:
        what = "matches no entrant" if not hits else f"matches {len(hits)} entrants"
        raise PenaltyError(f"grid penalty for {penalty['driver']!r} ({penalty['event']} {penalty['session']}) {what}; "
                           f"the entrants are {sorted(entrants['driver'].astype(str))}")
    return hits[0]


def apply(grid, entrants, penalties):
    """grid: 1-based starting positions, one per entrant row, shape (n,) or (n_sims, n). Returns a new float array
    with each penalty applied and every row renumbered 1..n (ties in the input broken by entrant order)."""
    g = np.asarray(grid, float)
    one = g.ndim == 1
    g2 = np.atleast_2d(g)
    n = g2.shape[1]
    order = np.argsort(g2, axis=1, kind="stable")
    rank = np.empty_like(g2)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(1, n + 1, dtype=float), g2.shape), axis=1)
    key = rank.copy()
    for p in penalties:
        i = driver_index(entrants, p)
        if p["kind"] in ("back", "pit_lane"):
            key[:, i] = 2 * n + rank[:, i]
        else:
            key[:, i] = np.minimum(rank[:, i] + p["places"], n) + 0.5
    new_order = np.lexsort((rank, key))
    out = np.empty_like(g2)
    np.put_along_axis(out, new_order, np.broadcast_to(np.arange(1, n + 1, dtype=float), g2.shape), axis=1)
    return out[0] if one else out


def describe(penalties):
    """What the audit records for each applied penalty."""
    keep = ("event", "session", "driver", "kind", "places", "source")
    return [{k: p[k] for k in keep if k in p} for p in penalties]
