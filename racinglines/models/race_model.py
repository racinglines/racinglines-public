"""
The pricing-model contract (docs/backtest-core.md): what the one engine (models/model_global.py GlobalModel, every
modelled sport's [sport] pricing_model) gives the shared backtest engine (racinglines/core/walk_forward.py), the
search and, later, the venues.

    m = race_model.get("mtb_dh")                  # GlobalModel bound to sports/mtb_dh.toml
    st = m.Settings.from_dict({...})              # the engine's settings, the schema's [model.defaults] as defaults
    data = m.load(engine_url)                     # everything the declared inputs read, once
    for ev in m.events(data, st, seasons):        # what to price, in time order, with its cutoff
        hist = m.history(data, st)                # (once) what the model learns from
        sims = m.price(hist, ev, st, rng)         # OutcomeSims from data strictly before ev.cutoff
        res = m.results(data, ev)                 # the official result, in markets/kinds.settle's shape

`price` must only read data from before `ev.cutoff`; `results` is read only after pricing, to settle.
Adding a sport means its schema (sports/<code>.toml: [model] inputs, defaults): no engine changes.

F1's and downhill's engines before the one engine (PositionSim, TimedRuns) were retired on 2026-10-19: they were the
laps and run-times cores the engine already uses for those inputs, so they priced the same (decision log 2026-10-19).
"""

from dataclasses import dataclass, field



@dataclass(frozen=True)
class Event:
    id: object                  # the sport's own event id
    season: int
    cutoff: object              # price with data strictly before this
    name: str = ""
    info: dict = field(default_factory=dict, compare=False)   # model-specific (e.g. the start list, format)


def bind(cls, sport):
    """The engine (models/model_global) is bound to the sport's schema by its `for_sport`; any other class (a
    challenger, a test's toy model) is returned as it is."""
    return cls.for_sport(sport) if hasattr(cls, "for_sport") else cls


def model_class(sport):
    """The sport's pricing model: [sport] pricing_model in sports/<code>.toml ("module:Class"; the one engine,
    models/model_global.py, for every modelled sport), bound to the sport's schema."""
    from importlib import import_module

    from racinglines import sports
    try:
        ref = sports.load(sport)["sport"].get("pricing_model")
    except FileNotFoundError:
        ref = None
    if ref:
        mod, _, cls = ref.partition(":")
        return bind(getattr(import_module(mod), cls), sport)
    raise ValueError(f"no pricing model for sport {sport!r}: set [sport] pricing_model in sports/{sport}.toml")


def challengers(sport):
    """{name: "module:Class"}: the sport's challenger models ([sport] challengers in sports/<code>.toml), earlier or
    alternative pricing models kept runnable beside [sport] pricing_model (`backtest walk-forward --model NAME`)."""
    from racinglines import sports
    try:
        return dict(sports.load(sport)["sport"].get("challengers") or {})
    except FileNotFoundError:
        return {}


def challenger(sport, name):
    """The challenger model class `name` of `sport`, bound to the sport's schema as the pricing model is."""
    from importlib import import_module
    refs = challengers(sport)
    if name not in refs:
        raise ValueError(f"sports/{sport}.toml [sport] challengers has no {name!r}; choose from {sorted(refs)}")
    mod, _, cls = refs[name].partition(":")
    return bind(getattr(import_module(mod), cls), sport)


def get(sport):
    return model_class(sport)()
