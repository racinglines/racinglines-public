"""
In-race win probability from live timing (the Live tab's win chart while the race runs).

A deliberately simple bridge from the pre-race price to the running order, not a lap simulator: each running
driver's remaining-race score blends where he is now with where the pre-race price put him, weighted by the share
of the race still to run, plus noise that shrinks as the laps run out:

    score = (1 - f) * position_now + f * rank_pre_race + SIGMA * sqrt(f) * N(0, 1)        f = laps left / race laps

The lowest score wins. At the start (f = 1) it is the pre-race order with noise; at the flag (f = 0) the leader wins.
Retired drivers get 0. Drivers are matched to markets by exact full name; an unmatched driver is left out.
"""

import numpy as np

SIGMA = 3.0          # positions of noise over a full race distance
SIMS = 20000


def retired(d):
    """True when the live-timing status says the driver is out (FastF1 results Status)."""
    s = str(d.get("status") or "").strip()
    if not s or s in ("None", "nan", "Finished") or s.startswith("+") or "Lap" in s:
        return False
    return True


def win_probs(drivers, prior, race_laps, sigma=SIGMA, sims=SIMS, seed=0):
    """drivers: the live-timing snapshot's rows (name, pos, laps, status); prior: {name: pre-race win probability};
    race_laps: the race distance in laps. Returns ({name: P(win)}, laps_done), or (None, laps_done) when the
    running order isn't usable yet (fewer than 3 positions)."""
    laps_done = max((int(d.get("laps") or 0) for d in drivers), default=0)
    run = [d for d in drivers if d.get("name") in prior and d.get("pos") is not None and not retired(d)]
    if len(run) < 3 or laps_done <= 0:
        return None, laps_done
    f = min(1.0, max(0.0, (race_laps - laps_done) / race_laps))
    pos = np.array([float(d["pos"]) for d in run])
    order = np.argsort([-prior[d["name"]] for d in run], kind="stable")
    rank = np.empty(len(run))
    rank[order] = np.arange(1, len(run) + 1)
    rng = np.random.default_rng(seed)
    score = (1 - f) * pos + f * rank + sigma * np.sqrt(f) * rng.standard_normal((sims, len(run)))
    wins = np.bincount(score.argmin(1), minlength=len(run)) / sims
    out = {name: 0.0 for name in prior}
    out.update({d["name"]: float(w) for d, w in zip(run, wins)})
    return out, laps_done
