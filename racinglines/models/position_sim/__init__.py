"""DEPRECATED import path, kept for one version (decision log 2026-10-11 "One engine"). The F1 code moved to generic,
input-named modules of the one engine (racinglines/models/model_global.py, docs/backtest-core.md):

    position_sim.model      -> racinglines.models.inputs.laps          (lap and sector pace, grid, finishing, Monte Carlo)
    position_sim.pricing    -> racinglines.models.inputs.laps_pricing  (Measurements, price_race, backtest, forecast)
    position_sim.practice   -> racinglines.models.inputs.practice
    position_sim.penalties  -> racinglines.models.inputs.grid
    position_sim.props, variants, evaluate, combo_job, dnf_check, sprint_check, inrace -> racinglines.models.<same>

Each old name IS the new module (a sys.modules alias), so a switch set through one (variants, Settings.applied) is
seen through the other. Nothing in the repo imports this package; import the new paths. It goes in the next version.
"""

import importlib
import sys

MOVED = {
    "model": "racinglines.models.inputs.laps",
    "pricing": "racinglines.models.inputs.laps_pricing",
    "practice": "racinglines.models.inputs.practice",
    "penalties": "racinglines.models.inputs.grid",
    **{n: f"racinglines.models.{n}" for n in ("props", "variants", "evaluate", "combo_job", "dnf_check",
                                              "sprint_check", "inrace")},
}

for _old, _new in MOVED.items():
    globals()[_old] = sys.modules[f"{__name__}.{_old}"] = importlib.import_module(_new)
