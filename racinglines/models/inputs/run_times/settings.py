"""
Every setting of a downhill walk-forward, in the same schema as the F1 sweep's
(racinglines/pipelines/sweep_settings.py): the same keys, labels, flags and search queue handling.
Defaults are today's model (models/inputs/run_times/model.py), so the default key is the baseline.

    s = DHSettings.from_dict({"half_life_days": 120, "prior_n": 1.5})
    with s.applied():            # module-level model switches (EPS_DF) set for the duration
        ...
"""

from contextlib import contextmanager

from racinglines.pipelines import sweep_settings as SS

from . import model as TM

CATEGORIES = ("ME", "WE", "MJ", "WJ")

SETTINGS = [
    SS.Setting("category", "model", "Category", "choice", "ME", choices=CATEGORIES,
               help="The field predicted and scored."),
    SS.Setting("sims", "model", "Simulations per event", "int", 5000, 100, 100000),
    SS.Setting("half_life_days", "model", "Recency half-life (days)", "float", TM.HALF_LIFE_DAYS, 20, 2000),
    SS.Setting("prior_n", "model", "Pace shrinkage (runs)", "float", TM.PRIOR_N, 0, 20,
               help="Shrinkage of rider pace toward the field median, in runs' weight."),
    SS.Setting("junior_weight", "model", "Junior runs' training weight", "float", TM.CATEGORY_WEIGHTS["MJ"], 0, 2),
    SS.Setting("train_scope", "model", "Training rows", "choice", "all", choices=("all", "season"),
               help="all = every season and category before the event; season = the target season only."),
    SS.Setting("eps_df", "model", "Run noise t degrees of freedom", "float", None, 2.01, 100,
               help="Empty = normal run noise."),
    SS.Setting("seed", "model", "Monte Carlo seed", "int", None, 0, 2**31 - 1,
               help="Empty = the fixed seed (42). Set different seeds for independent noise draws in a search."),
]


class DHSettings(SS.Settings):
    SPEC = SETTINGS
    BY = {s.name: s for s in SETTINGS}
    MODEL = [s.name for s in SETTINGS]
    ALIASES = {"n_sims": "sims"}

    def _validate(self):
        pass

    def fit_kw(self):
        """Keyword arguments of fit_season_model / walk_forward_season (as `racinglines model backtest mtb_dh`)."""
        return dict(train_scope=self["train_scope"], half_life_days=self["half_life_days"],
                    category_weights={"MJ": self["junior_weight"]}, prior_n=self["prior_n"])

    def label(self):
        ch = self.changed()
        return ", ".join(f"{k}={v}" for k, v in ch.items()) or "baseline"

    @contextmanager
    def applied(self):
        old = TM.EPS_DF
        try:
            TM.EPS_DF = self["eps_df"]
            yield self
        finally:
            TM.EPS_DF = old
