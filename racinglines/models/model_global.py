"""The one engine: the single pricing model of every modelled sport (owner, 2026-10-11: "Single global model ... For
ALL sports, we will want to be able to handle high-level and/or granular data"; decision log 2026-10-11 "One engine",
provisional, staging only). There are no sport-specific models, only schemas that declare the model's inputs and
defaults:

    [sport]
    pricing_model = "racinglines.models.model_global:GlobalModel"     # every modelled sport (tests/test_one_engine.py)
    [model]
    inputs = ["results", "qualifying", "grid", "practice", "laps", "sectors"]     # the layers it reads
    name = "f1_sector_sim"          # model_runs.model of its runs (optional; default: the core's)
    [model.defaults]                # the per-sport defaults of its settings (decision-log entries)
    half_life_days = 120.0

`GlobalModel.for_sport(code)` (called by race_model.model_class) reads that and binds the engine: the declared inputs
pick the core that prices them (models/inputs/__init__.py core_for: results only, lap-level, or timed runs), the core
reads only the declared layers, and the settings take the schema's defaults. Nothing here or in the cores names a
sport. The bound class is a GlobalModel (issubclass) whose contract methods (race_model.py) are the core's.

Unbound, GlobalModel is the results core (models/inputs/results.py: the probit strength model this module held before
the one engine), so `GlobalModel().load(data=...)` prices any results frame, as before.

`for_sport(code, core="results")` is the engine at a lower level than the schema's, e.g. F1 from finishing positions
only (`racinglines backtest walk-forward f1 --model global`); it reads `[model.<core>.defaults]` instead of
`[model.defaults]`.
"""

from racinglines import sports
from racinglines.models import inputs as INP
from racinglines.models.inputs.results import (  # noqa: F401  (the results core's names, kept importable here)
    DNF_PRIOR_N,
    MIN_FINISHERS,
    REQUIRED,
    SETTINGS,
    SIDE_SEED,
    GlobalSettings,
    ResultsCore,
    _frame,
    _settings_class,
    best_lap_order,
)


class GlobalModel(ResultsCore):
    """The one engine. Unbound: the results core. Bound to a sport by `for_sport`: the core its schema's inputs pick."""

    inputs = ("results",)                  # the declared inputs it reads (for_sport)
    core = "results"                       # the core's key (models/inputs CORES)

    @classmethod
    def for_sport(cls, sport, core=None):
        schema = sports.load(sport)
        m = schema.get("model", {})
        where = f"sports/{sport}.toml [model]"
        declared = INP.declared(m, where)
        own = INP.core_for(declared, m.get("core"), where)    # the schema's own level ([model] core pins it when several fit)
        spec = own if core in (None, own.key) else INP.core_for(declared, core, where)
        used = tuple(i for i in declared if i in spec.reads())
        defaults = dict(m.get("defaults", {}) if spec is own else m.get(spec.key, {}).get("defaults", {}))
        for inp, setting in spec.gates:                  # an optional input left undeclared: its setting off
            if inp not in used:
                if defaults.get(setting):
                    raise ValueError(f"{where}: defaults turn {setting} on, but inputs don't declare {inp!r}")
                defaults[setting] = False
        impl = spec.load()
        attrs = impl.bind(sport, schema, used, defaults)
        name = m.get("name") if spec is own and m.get("name") else impl.name
        bases = (cls,) if issubclass(cls, impl) else (impl, cls)     # the core's methods first
        return type(f"GlobalModel[{sport}]", bases,
                    dict(sport=sport, inputs=used, core=spec.key, name=name,
                         noise_unit=getattr(impl, "noise_unit", None), **attrs))

    def at_stage(self, ev, label):
        """`ev` unchanged: the engine (every core: results, laps, timed runs) prices an event once, as of its cutoff,
        the same at every stage label, so `backtest walk-forward --stage LABEL` runs for it and its records carry the
        label, pairing with a stage-priced challenger's run in `eval compare` (Core v2 S3b). The CLI's --stage, the
        walk_forward job's stage and the MCP run_job param reach it; it is a method, not a setting (no settings key)."""
        return ev
