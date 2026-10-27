"""
Calibration power for simulated race outcomes (C48: favourites vs long shots).

The simulation's win chances p_i become p_i^a / sum_j p_j^a, and every simulated race is resampled with weight
q(winner) / p(winner) so the other markets priced from the same draws (podium, top 10, head-to-head, constructor)
move with the win price instead of drifting apart from it. a > 1 sharpens the favourites; a = 1 changes nothing.

The power comes from the `calib_power` setting (pipelines/sweep_settings.py). F1's default is DEFAULT: "walk_forward"
on the hosted service since 2026-10-21 (docs/f1-roadmap.md decision log), "off" in the PyPI package; "off" is the uncalibrated price and keeps the keys it had before
(saved runs without the setting read as off). A number is that power at every stage, and "walk_forward" reads sports/<sport>/calibration.toml, where each [[fit]] holds the
power fitted on the races finished before its `from` date, separately for stages with the starting grid known
(`grid`) and before it (`nogrid`). A cutoff before the first fit gets a = 1.

    power(setting, sport, cutoff, grid_known)   the power to use, or None for off
    apply(sim, a, rng)                          the resampled sims (a new dict; arrays indexed by sim on axis 0)
    win_power(p, a)                             p^a renormalised (the target win chances)
"""

import importlib
import importlib.util
import tomllib
from datetime import date, datetime
from functools import cache

import numpy as np

from racinglines.paths import ROOT

WALK_FORWARD = "walk_forward"
OFF = "off"
# F1's default (owner, 2026-10-21): walk_forward on the hosted service (racinglines/hosted.py, private only), off in
# the PyPI package, which leaves that module out (oss/manifest.toml)
DEFAULT = (importlib.import_module("racinglines.hosted").CALIB_POWER
           if importlib.util.find_spec("racinglines.hosted") else OFF)


@cache
def table(sport):
    path = ROOT / "sports" / sport / "calibration.toml"
    if not path.exists():
        return ()
    fits = tomllib.loads(path.read_text()).get("fit", [])
    return tuple(sorted(fits, key=lambda f: f["from"]))


def parse(value):
    """The setting's canonical text: None (blank: the setting's default), "off", "walk_forward", or a power such as
    "2.0" (1 to 10)."""
    v = str(value).strip().lower().replace("-", "_")
    if v == "":
        return None
    if v in ("none", "off"):          # explicit off: kept apart from blank, so it survives the walk_forward default
        return OFF
    if v == WALK_FORWARD:
        return v
    try:
        a = float(v)
    except ValueError:
        raise ValueError(f"calib_power: {value!r} is not a number or {WALK_FORWARD}") from None
    if not 1.0 <= a <= 10.0:
        raise ValueError(f"calib_power: {a} is outside 1 to 10 (1 = off, above 1 sharpens favourites)")
    return str(a)


def table_hash(sport="f1"):
    """Short hash of `sport`'s fitted calibration rows (None: every sport's): part of the model key with walk_forward,
    so a refit table prices afresh instead of reusing runs cached under the old one, and another sport's table
    leaves the key alone."""
    import hashlib
    import json
    h = hashlib.sha1()
    pattern = f"{sport}/calibration.toml" if sport else "*/calibration.toml"
    for path in sorted((ROOT / "sports").glob(pattern)):                    # the fitted rows, not the comments
        h.update(path.parent.name.encode())
        h.update(json.dumps(tomllib.loads(path.read_text()).get("fit", []), sort_keys=True, default=str).encode())
    return h.hexdigest()[:8]


def _day(cutoff):
    if cutoff is None:
        return None
    if isinstance(cutoff, datetime):
        return cutoff.date()
    if isinstance(cutoff, date):
        return cutoff
    return datetime.fromisoformat(str(cutoff)[:19]).date()


def power(setting, sport, cutoff, grid_known):
    """The calibration power for a price at `cutoff`, or None when calibration is off (setting unset or a = 1)."""
    if setting in (None, "", OFF):
        return None
    if str(setting) != WALK_FORWARD:
        a = float(setting)
        return None if a == 1.0 else a
    day = _day(cutoff)
    fit = None
    for f in table(sport):
        if day is not None and f["from"] <= day:
            fit = f
    if fit is None:
        return None
    a = float(fit["grid" if grid_known else "nogrid"])
    return None if a == 1.0 else a


def win_power(p, a):
    p = np.clip(np.asarray(p, float), 0, None)
    q = np.where(p > 0, p, 0) ** a
    return q / q.sum() if q.sum() > 0 else p


def apply(sim, a, rng):
    """Resample the simulated races so each driver's win share is p^a renormalised. sim: simulate_race's dict
    (pos is (n_sims, n), finishing positions with 1 = winner). Arrays whose first axis is the sim are reindexed,
    other entries (nested stage dicts) are kept as they are. Draws only from `rng`."""
    if a is None:
        return sim
    pos = sim["pos"]
    n_sims, n = pos.shape
    idx = resample_index(np.argmax(pos == 1, axis=1), n, a, rng)
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_sims else v)
            for k, v in sim.items()}


def resample_index(winner, n, a, rng):
    """The sims to keep (with repeats) so each entrant's win share becomes p^a renormalised. winner: the index of
    each sim's winner (0..n-1). Draws only from `rng`."""
    n_sims = len(winner)
    p = np.bincount(winner, minlength=n) / n_sims
    q = win_power(p, a)
    w = np.where(p[winner] > 0, q[winner] / np.where(p[winner] > 0, p[winner], 1), 0)
    return rng.choice(n_sims, size=n_sims, replace=True, p=w / w.sum())


# --- fitting (walk-forward: each fit sees only races finished before it) ----------------------------------------

POWERS = tuple(round(1.0 + 0.1 * i, 1) for i in range(31))     # 1.0 .. 4.0
MIN_RACES = 8                                                    # no fit (a = 1) before this many races
SETTLE_DAYS = 2                                                  # a fit applies this many days after its last race


def _race_nll(df, a):
    """Winner log loss per (group, event) at power a. df: event_key, group, run, p, won (one row per driver per
    priced stage; several stages of one event in a group are summed)."""
    lp = np.log(np.clip(df["p"].to_numpy(float), 1e-4, None))
    z = df.assign(x=a * lp)
    lse = z.groupby("run")["x"].transform(lambda x: np.log(np.exp(x).sum()))
    z = z.assign(nll=-(z["x"] - lse))
    return z[z["won"] == 1].groupby(["group", "event_key"])["nll"].sum()


def fit_walk_forward(df, powers=POWERS, min_races=MIN_RACES):
    """The [[fit]] rows of sports/<sport>/calibration.toml from stored as-of predictions. df columns: event_key,
    race_date (the day the race ran), group ("grid" / "nogrid"), run (one priced stage), p (win chance), won (0/1).
    Row k is fitted on the races strictly before race k and applies from two days after race k-1's date (a night race
    can end after midnight UTC), so no stage of race k sees its own result; the last row (all races) applies from two
    days after the last race."""
    import pandas as pd
    df = df[df.groupby("run")["won"].transform("max") == 1]          # stages whose race has a recorded winner
    nll = pd.concat({a: _race_nll(df, a) for a in powers}, axis=1)   # (group, event) x power
    races = (df.drop_duplicates("event_key").sort_values(["race_date", "event_key"])[["event_key", "race_date"]]
             .to_records(index=False))
    fits = []
    for k in range(min_races, len(races) + 1):
        before = [e for e, _ in races[:k]]
        row = {"from": (pd.Timestamp(races[k - 1][1]) + pd.Timedelta(days=SETTLE_DAYS)).date(), "races": k}
        for g in ("grid", "nogrid"):
            sub = nll[nll.index.get_level_values("group") == g]
            sub = sub[sub.index.get_level_values("event_key").isin(before)]
            row[g] = float(sub.sum().idxmin()) if len(sub) else 1.0
        if k < len(races):
            row["next_event"] = str(races[k][0])
        fits.append(row)
    return fits


def to_toml(fits, header=""):
    out = [header.rstrip() + "\n"] if header else []
    for f in fits:
        lines = [f"from = {f['from'].isoformat()}", f"grid = {f['grid']}", f"nogrid = {f['nogrid']}",
                 f"races = {f['races']}"] + ([f'next_event = "{f["next_event"]}"'] if "next_event" in f else [])
        out.append("[[fit]]\n" + "\n".join(lines) + "\n")
    return "\n".join(out)
