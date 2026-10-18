"""
Prediction records on disk (docs/backtest-core.md, "Prediction records"; docs/engine-roadmap.md phase E4a):
the long-format rows of a saved model run, beside `race_predictions`, and the archive of its simulations.

    data/runs/records/<model_run_id>/predictions.parquet    one row per event x stage x kind x subject
    data/runs/records/<model_run_id>/sims.npz               the OutcomeSims (backtests and forecasts only)

Written only when RACINGLINES_PREDICTION_RECORDS is set (default off: nothing is written and every existing
output is unchanged). RACINGLINES_RECORDS_DIR moves the root (tests use a temporary folder). A run that
simulated several events keeps one archive per event, `sims-<key>.npz`, keyed as given to `write`.
"""

import os
from pathlib import Path

import pandas as pd

SWITCH = "RACINGLINES_PREDICTION_RECORDS"
ROOT_ENV = "RACINGLINES_RECORDS_DIR"


def enabled():
    return os.environ.get(SWITCH, "").strip().lower() in ("1", "true", "yes", "on")


def records_dir():
    if os.environ.get(ROOT_ENV):
        return Path(os.environ[ROOT_ENV])
    from racinglines import paths
    return paths.runs("records", mkdir=False)


def _sims_name(key=None):
    return "sims.npz" if key is None else f"sims-{key}.npz"


def write(run_id, df, sims=None):
    """Write a run's records; `sims` is an OutcomeSims (sims.npz), or {key: OutcomeSims} (sims-<key>.npz each).
    Returns the run's folder."""
    from racinglines.models import outcomes as O
    d = records_dir() / str(int(run_id))
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "predictions.parquet", index=False)
    if sims is not None:
        for key, s in ({None: sims} if isinstance(sims, O.OutcomeSims) else dict(sims)).items():
            O.save_sims(s, d / _sims_name(key))
    return d


def read(run_id):
    return pd.read_parquet(records_dir() / str(int(run_id)) / "predictions.parquet")


def load_sims(run_id, key=None):
    from racinglines.models import outcomes as O
    return O.load_sims(records_dir() / str(int(run_id)) / _sims_name(key))


def concat(frames):
    frames = [f for f in frames if f is not None and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else None
