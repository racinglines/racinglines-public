"""DEPRECATED import path, kept for one version (decision log 2026-10-11 "One engine"). The downhill code moved to the
one engine's run-times input, racinglines.models.inputs.run_times (same submodules: model, points, season,
settings). Each old name IS the new module (a sys.modules alias). Nothing in the repo imports this package; it goes in
the next version.
"""

import importlib
import sys

from racinglines.models.inputs.run_times import *  # noqa: F401,F403
from racinglines.models.inputs.run_times import RNG, _spearman, _training_rows  # noqa: F401

for _sub in ("model", "points", "season", "settings"):
    globals()[_sub] = sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(
        f"racinglines.models.inputs.run_times.{_sub}")
