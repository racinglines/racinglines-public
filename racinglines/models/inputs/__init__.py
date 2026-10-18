"""The one engine's input layers (racinglines/models/model_global.py, docs/backtest-core.md).

A sport's schema declares the data its model reads, and nothing else picks the code:

    [model]
    inputs = ["results", "qualifying", "grid", "practice", "laps", "sectors"]     # F1
    inputs = ["run_times", "practice"]                                          # downhill
    inputs = ["results", "qualifying", "fastest_lap"]                           # NASCAR

INPUTS names every layer (one module each, named by its input). A core is the part that turns the declared layers
into simulations; `core_for(inputs)` picks the one whose required inputs the schema declares, the most specific one
(most required inputs) when several fit, and refuses a declared input that core doesn't read, so nothing declared is
silently ignored. High-level data (results only) and granular data (laps, sectors, run times) both work: a sport
declares what it has, and the engine prices at that level. An optional input left undeclared turns off the setting it
gates (`gates`), so a sport without, say, practice data never reads it.

The cores keep the algorithms they had as separate models (results: model_global's probit strength; laps: F1's
sector simulation; run_times: downhill's timed runs), so every price is unchanged (decision log 2026-10-11 "One
engine", provisional, staging only). Blending them (laps into NASCAR and MotoGP, one strength model for every core) is
phase 2, after validation.
"""

from dataclasses import dataclass
from importlib import import_module

INPUTS = {
    "results": "finishing positions of past races (results, round kind 'race'), status and team",
    "qualifying": "the qualifying classification (round 'qual'): the grid's order (laps core), or a side "
                  "classification drawn for the pole (results core, [model] sessions)",
    "grid": "the starting grid: the qualifying order moved by confirmed grid penalties (inputs/grid.py)",
    "practice": "practice sessions: the practice pace prior (laps core, inputs/practice.py), or timed training runs "
                "at their run weight (run-times core)",
    "laps": "lap times: race pace and qualifying best laps per driver and car (laps table, inputs/laps.py)",
    "sectors": "sector times and the track's sector profile: the car's fast/slow-sector fit (inputs/laps.py)",
    "fastest_lap": "each past race's order of best laps (laps table), drawn as a side classification for the race's "
                   "fastest lap (results core, [model] best_lap)",
    "run_times": "timed runs: every run's time per round, the rider pace and noise model (inputs/run_times/)",
}


@dataclass(frozen=True)
class Core:
    key: str                     # the core's name (GlobalModel.for_sport(sport, core=key))
    ref: str                     # "module:Class" with the pricing-model contract and a `bind` classmethod
    requires: frozenset          # inputs the core can't price without
    optional: frozenset = frozenset()
    gates: tuple = ()            # (input, setting): the setting defaults to False when the input isn't declared

    def load(self):
        mod, _, cls = self.ref.partition(":")
        return getattr(import_module(mod), cls)

    def reads(self):
        return self.requires | self.optional


CORES = (
    Core("results", "racinglines.models.inputs.results:ResultsCore", frozenset({"results"}),
         frozenset({"qualifying", "fastest_lap"})),
    Core("laps", "racinglines.models.inputs.laps_core:LapsCore", frozenset({"results", "qualifying", "grid", "laps"}),
         frozenset({"sectors", "practice"}), gates=(("sectors", "track_features"), ("practice", "practice_prior"))),
    Core("run_times", "racinglines.models.inputs.run_times.core:RunTimesCore", frozenset({"run_times"}),
         frozenset({"practice"})),
)
BY_KEY = {c.key: c for c in CORES}


def declared(model, where="[model]"):
    """The inputs a schema's [model] table declares, validated. A schema without `inputs` (written before the one
    engine) declares results, plus qualifying and fastest_lap where its results hooks name them."""
    if "inputs" in model:
        inputs = list(model["inputs"])
    else:
        inputs = ["results"] + (["qualifying"] if model.get("sessions") else []) \
                 + (["fastest_lap"] if model.get("best_lap") else [])
    unknown = [i for i in inputs if i not in INPUTS]
    if unknown or not inputs or len(set(inputs)) != len(inputs):
        raise ValueError(f"{where} inputs {inputs}: each one of {sorted(INPUTS)}, at least one, no repeats")
    return tuple(inputs)


def core_for(inputs, key=None, where="[model]"):
    """The core that prices these inputs: `key`'s, or the most specific one whose required inputs are all declared.
    A declared input the default core doesn't read is an error; with `key` (a level below the schema's, e.g. the
    results level of a sport with laps) the inputs it doesn't read are left out."""
    have = set(inputs)
    if key is not None:
        if key not in BY_KEY:
            raise ValueError(f"no engine core {key!r}; choose from {sorted(BY_KEY)}")
        core = BY_KEY[key]
        if not core.requires <= have:
            raise ValueError(f"{where} inputs {list(inputs)} lack what the {key} core needs: "
                             f"{sorted(core.requires - have)}")
        return core
    fits = [c for c in CORES if c.requires <= have]
    if not fits:
        raise ValueError(f"{where} inputs {list(inputs)}: no engine core prices them; each needs one of "
                         + "; ".join(f"{c.key}: {sorted(c.requires)}" for c in CORES))
    most = max(len(c.requires) for c in fits)
    best = [c for c in fits if len(c.requires) == most]
    if len(best) > 1:
        raise ValueError(f"{where} inputs {list(inputs)} fit several cores equally: {[c.key for c in best]}")
    unread = sorted(have - best[0].reads())
    if unread:
        raise ValueError(f"{where} inputs {unread} aren't read by the {best[0].key} core "
                         f"(it reads {sorted(best[0].reads())})")
    return best[0]
