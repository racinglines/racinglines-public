"""
Combo (same-game parlay) prices for one F1 event: the Lab job `f1_combo` (racinglines/web/jobs.py), run as
`racinglines model combo f1 --variant <v> --event 2026-17 --legs '<json>' [--cutoff ...] --save`.

    parse_combos(text)                   the --legs JSON as [(name, [leg, ...])]
    resolve(legs, entrants)              names -> athlete ids and team keys, exact keys only (never fuzzy)
    checks(sims)                         the simulation's own P(pole-sitter wins), P(winner sets the fastest lap)
    combo_metrics(combos, sims, ...)     what the run stores in model_runs.metrics
    price_event(meas, hist, key, ...)    one pricing of the event as of the cutoff (the live stage path's inputs)

The event is priced once (pricing.price_race, as of the cutoff, the same as-of rule as every diagnostic) and every
combo is read from the same simulations (markets/combos.py), with each leg's marginal beside the combo's fair value.

Calibration flags (owner, 2026-10-07). Two correlations the combos lean on are checked against real races and written
next to the prices. A combo with a pole leg priced before qualifying (the grid not known yet) carries
`calibrated: false` and the gap between the simulation's P(pole-sitter wins) and the historical rate: the simulation's
qualifying-to-race link is too weak (about 20 % against 59 %), so win + pole is underpriced until that is fixed (a
follow-up, not this change). After qualifying the pole leg is decided and win + pole is the win given that grid. A
combo with a fastest-lap leg carries the same flag when the run's P(winner sets the fastest lap) is far from history
(the `fastlap` draw; `flpos` lands near it).
"""

import json

import numpy as np
import pandas as pd

from racinglines.markets import combos as C
from racinglines.markets import kinds as K

# Historical rates (VM database, F1 Grands Prix; verified 2026-10-07): the pole-sitter wins, the winner sets the
# fastest lap (sports/f1/fastest_lap.toml)
HISTORY = {
    "pole_sitter_wins": {"2022-2026": [64, 108], "2025-2026": [27, 40]},
    "winner_sets_fastest_lap": {"2022-2026": [36, 108], "2025-2026": [17, 40]},
}
REFERENCE = "2022-2026"
TOLERANCE = 0.10                      # |simulated - historical| above this: the dependent legs are not calibrated
POLE_KINDS = ("race_pole",)           # legs whose correlation with the race runs through the qualifying draw
FL_KINDS = ("race_fastest_lap",)


def parse_combos(text):
    """The --legs argument: JSON, one combo as a list of legs, several as a list of lists, or named as an object
    {name: [legs]}. Returns [(name, legs)]; the legs are checked when they are resolved."""
    try:
        d = json.loads(text) if isinstance(text, str) else text
    except json.JSONDecodeError as e:
        raise ValueError(f"legs: not JSON ({e})") from None
    if isinstance(d, dict):
        items = list(d.items())
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        items = [("combo 1", d)]
    elif isinstance(d, list) and d and all(isinstance(x, list) for x in d):
        items = [(f"combo {i + 1}", x) for i, x in enumerate(d)]
    else:
        raise ValueError("legs: a list of legs, a list of combos (lists of legs), or {name: [legs]}")
    if not items:
        raise ValueError("legs: no combo given")
    return [(str(n), list(legs)) for n, legs in items]


def _athlete(value, entrants, where):
    """An athlete id: an id of the entry list, or a driver's full name or surname matched exactly after normalising
    (penalties.normalise: lowercase, no accents or punctuation); a surname two entrants share is an error."""
    from racinglines.models.inputs.grid import normalise
    ids = entrants["athlete_id"].tolist()
    if not isinstance(value, str) and value in ids:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) in ids:
        return int(value)
    key = normalise(value)
    names = entrants["driver"].map(normalise).tolist()
    hits = [i for i, n in enumerate(names) if n == key] or [i for i, n in enumerate(names) if n.split(" ")[-1:] == [key]]
    if len(hits) != 1:
        what = "matches no entrant" if not hits else f"matches {len(hits)} entrants"
        raise ValueError(f"{where}: {value!r} {what}; the entrants are {sorted(entrants['driver'].astype(str))}")
    return ids[hits[0]]


def _team(value, entrants, where):
    from racinglines.models.inputs.grid import normalise
    keys = sorted(set(entrants["team_key"].astype(str)))
    hit = [k for k in keys if normalise(k) == normalise(value) or k == value]
    if len(hit) != 1:
        raise ValueError(f"{where}: team {value!r} is not one of {keys}")
    return hit[0]


def resolve(legs, entrants):
    """The legs with every driver name as an athlete id (`driver` or `athlete` -> `athlete`, `pair` entries) and team
    names as the model's team keys, then checked (combos.check_legs). Exact keys only."""
    out = []
    for i, leg in enumerate(legs):
        where = f"legs[{i}]"
        if not isinstance(leg, dict):
            raise ValueError(f"{where}: must be a table with a kind")
        r = {k: v for k, v in leg.items() if k != "driver"}
        if "driver" in leg or "athlete" in leg:
            r["athlete"] = _athlete(leg.get("athlete", leg.get("driver")), entrants, where)
        if "pair" in leg:
            if not isinstance(leg["pair"], (list, tuple)) or len(leg["pair"]) != 2:
                raise ValueError(f"{where}.pair: two drivers")
            r["pair"] = [_athlete(a, entrants, where) for a in leg["pair"]]
        if "team" in leg:
            r["team"] = _team(leg["team"], entrants, where)
        out.append(r)
    return C.check_legs(out)


def checks(sims):
    """The simulation's own rates for the correlations combos lean on: P(the pole-sitter wins the race) and, when the
    run draws the fastest lap, P(the winner sets it), each over the simulations where it is defined."""
    win = (sims.rank == 1) & sims.finished
    out = {}
    q = sims.stage_rank.get("qual")
    if q is not None:
        pole = q == 1
        out["pole_sitter_wins"] = float((pole & win).any(axis=1).mean())
    fl = sims.indicators.get("race_fastest_lap")
    if fl is not None:
        has = win.any(axis=1)
        out["winner_sets_fastest_lap"] = float((fl & win).any(axis=1)[has].mean()) if has.any() else None
    return out


def _flags(legs, sim_rates, grid_known):
    """[(stat, gap)] for the dependent legs of one combo whose simulated correlation is off history."""
    kinds = {leg["kind"] for leg in legs}
    out = []
    for stat, needs, active in (("pole_sitter_wins", POLE_KINDS, not grid_known),
                                ("winner_sets_fastest_lap", FL_KINDS, True)):
        if not active or not kinds & set(needs) or not kinds - set(needs) or sim_rates.get(stat) is None:
            continue
        n, d = HISTORY[stat][REFERENCE]
        gap = sim_rates[stat] - n / d
        if abs(gap) > TOLERANCE:
            out.append((stat, round(gap, 4)))
    return out


def _label(leg, names):
    k = K.KINDS[leg["kind"]]
    who = (" vs ".join(names.get(a, str(a)) for a in leg["pair"]) if "pair" in leg
           else names.get(leg.get("athlete"), leg.get("team") or (f"line {leg['line']}" if "line" in leg else "")))
    side = leg.get("side")
    return f"{k.label}: {who}" + (f" ({side})" if side else "")


def combo_metrics(combos, sims, grid_known, names=None):
    """{"combos": [...], "checks": {...}} for model_runs.metrics: per combo its fair value, standard error, each leg's
    marginal, the product, the lift and `calibrated` (False with `flags` when a dependent leg's correlation is off)."""
    names = names or {}
    rates = checks(sims)
    out = []
    for name, legs in combos:
        p = C.price(legs, sims)
        flags = _flags(legs, rates, grid_known)
        row = dict(name=name, fair=round(p["fair"], 6), se=round(p["se"], 6), independent=round(p["independent"], 6),
                   lift=None if p["lift"] is None else round(p["lift"], 4), n_sims=p["n_sims"],
                   legs=[dict(leg, label=_label(leg, names), marginal=round(m, 6)) for leg, m in zip(legs, p["legs"])],
                   calibrated=not flags)
        if flags:
            row["flags"] = [dict(check=s, gap=g, simulated=round(rates[s], 4),
                                 historical=round(HISTORY[s][REFERENCE][0] / HISTORY[s][REFERENCE][1], 4),
                                 note=("before qualifying the simulation links pole to the win too weakly: "
                                       "win + pole is underpriced" if s == "pole_sitter_wins" else
                                       "the run's fastest-lap draw is off history (price with the flpos variant)"))
                            for s, g in flags]
        out.append(row)
    hist = {s: {w: dict(n=n, races=d, rate=round(n / d, 4)) for w, (n, d) in v.items()} for s, v in HISTORY.items()}
    return dict(combos=out, checks=dict(simulated={k: None if v is None else round(v, 4) for k, v in rates.items()},
                                        historical=hist, grid_known=bool(grid_known)))


def price_event(meas, hist, event_key, cutoff, n_sims=10000, seed=42, use_track=True):
    """(sims, entrants, summary, extras) for one event as of `cutoff`: a raced event through pricing.diagnostic, an
    upcoming one through price_race with the live path's entrants, venue and side stages (pipelines/signals.py)."""
    from racinglines.models import outcomes as O
    from racinglines.models.inputs import laps_pricing as run
    from racinglines.pipelines import signals as SG
    year, rnd = (int(x) for x in event_key.split("-"))
    key = f"{year}-{rnd:02d}"
    ev = meas.res[(meas.res["year"] == year) & (meas.res["series_round"] == rnd)]
    event_id = int(ev["event_id"].iloc[0]) if len(ev) else None
    if event_id is not None and (ev["round"] == "race").any():
        _, summ, ex, _ = run.diagnostic(meas, hist, key, cutoff, n_sims=n_sims, seed=seed, use_track=use_track)
    else:
        from racinglines.pipelines import weekend_sweep as WS
        try:
            w = WS.schedule(year, rounds=[rnd]).get(rnd) or dict(event_key=key)
        except Exception as e:  # noqa: BLE001  no schedule (offline): no side stages, venue from the event's rows
            print(f"combo: no schedule for {key} ({e}); pricing without side stages", flush=True)
            w = dict(event_key=key)
        summ, ex = run.price_race(meas, hist, cutoff, event_id, n_sims=n_sims, rng=np.random.default_rng(seed),
                                  use_track=use_track, entrants=SG._entrants(meas, event_id, cutoff),
                                  venue=SG._venue(meas, event_id, w), stages=SG._stages(w), event_key=key)
    return O.from_position_sim(ex["entrants"], ex["sim"]), ex["entrants"], summ, ex


def names_of(entrants):
    return dict(zip(entrants["athlete_id"], entrants["driver"].astype(str))) if "driver" in entrants else {}


def run(meas, hist, event_key, cutoff, legs_text, n_sims=10000, use_track=True, seed=42):
    """Price the event once and every combo of `legs_text`. Returns (metrics, extras)."""
    combos = parse_combos(legs_text)
    sims, entrants, summ, ex = price_event(meas, hist, event_key, pd.Timestamp(cutoff), n_sims=n_sims,
                                           use_track=use_track, seed=seed)
    resolved = [(name, resolve(legs, entrants)) for name, legs in combos]
    grid_known = ex["audit"].get("grid") == "qualifying order"
    return combo_metrics(resolved, sims, grid_known, names_of(entrants)), ex
