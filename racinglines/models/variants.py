"""
Named model variants: combinations of the challenger switches in model.py
(docs/f1-roadmap.md, F1-2 / F1-3). "baseline" is today's default model.

    with variants.use("grid+pretrain"):
        hist = pricing.history(meas)          # build history INSIDE the variant: features differ
        ...

Names combine with "+". Each run that prices with a variant records its name
(params.variant), so stored sweep stages and backtests are never mixed.
"""

from contextlib import contextmanager

from racinglines.models.inputs import laps as M

SWITCHES = {
    "baseline": {},
    "grid": dict(GRID_TERMS=True),                 # F1-2 step 1: front-of-grid term in the ridge
    "gridq": dict(GRID_TERMS="known"),             # the same term, only once the real grid is known
    "pretrain": dict(PRE_PRACTICE_TRAIN=True),     # F1-2 step 2: no-practice training for pre-FP1 pricing
    "gbm": dict(FINISH_MODEL="gbm"),               # F1-2 step 3: gradient-boosted finishing model
    "tail": dict(CHAOS=True, TEAM_DNF_CORR=True),  # F1-3: disrupted-race mixture + correlated retirements
    "reset": dict(REG_RESET=True),                 # earlier seasons' car pace discounted in a new-regulations year
    "rookie": dict(ROOKIE_CARRY=0.25),             # F1-2: a finished rookie season's teammate gaps count a quarter
    "fastlap": dict(FASTEST_LAP=True),             # adds the fastest lap (fl_prob); every other price unchanged
    "flpos": dict(FASTEST_LAP=True, FL_FROM="position"),   # the fastest lap from the simulated finishing order
    "practicefast": dict(PRACTICE_FASTEST=True),   # adds each practice session's fastest lap; every other price unchanged
}
DESCRIPTION = {
    "baseline": "current model (shared car, practice prior, ridge finishing model)",
    "grid": "adds a front-of-grid term, log(grid)/log(n), to the finishing model",
    "gridq": "the front-of-grid term only once the real grid is known (after qualifying)",
    "pretrain": "before any practice, uses a finishing model trained on no-practice paces",
    "gbm": "gradient-boosted finishing model (monotone in grid and pace), out-of-time noise",
    "tail": "some simulated races are disrupted (more noise and retirements); teammates' retirements correlated",
    "reset": "in a season with new technical regulations (2022, 2026), earlier seasons' car pace counts a quarter",
    "rookie": "once a driver's rookie season is over, its teammate comparisons count a quarter in the driver offsets",
    "fastlap": "also draws who sets the race's fastest lap (race pace plus noise, among classified cars); "
               "the other prices are unchanged",
    "flpos": "also draws who sets the race's fastest lap from the simulated finishing order, with per-position "
             "weights from 2022-26 races (sports/f1/fastest_lap.toml); the other prices are unchanged",
    "practicefast": "also draws each practice session's order of best laps (qualifying pace plus practice noise; the "
                    "real order once the session has run), for the practice-fastest markets; the other prices are "
                    "unchanged",
}


def switches(name):
    out = {}
    for part in name.split("+"):
        if part not in SWITCHES:
            raise ValueError(f"unknown variant {part!r}; known: {', '.join(SWITCHES)}")
        out.update(SWITCHES[part])
    return out


def describe(name):
    return "; ".join(DESCRIPTION[p] for p in name.split("+"))


@contextmanager
def use(name):
    """Set the variant's switches on the model module; restore the defaults afterwards."""
    sw = switches(name)
    old = {k: getattr(M, k) for k in sw}
    for k, v in sw.items():
        setattr(M, k, v)
    try:
        yield sw
    finally:
        for k, v in old.items():
            setattr(M, k, v)


# switches that only act in some seasons: outside them the variant prices exactly like the one without
SEASONAL = {"reset": lambda year: year in M.RESET_YEARS}


def for_season(name, year):
    """The variant as it prices in `year`: switches inert that season dropped (e.g. "reset" outside a
    new-regulations season), so a held-out season's run of the plainer variant is the same model."""
    parts = [p for p in name.split("+") if p not in SEASONAL or SEASONAL[p](year)]
    return "+".join(parts) or "baseline"
