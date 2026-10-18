"""
Every setting of a season sweep, in one schema: the model, entry timing, taker and maker strategy
parameters, and which markets to trade. The same schema drives the `racinglines strategy sweep f1` flags, the
Lab's Edge Finder sweep form, the search queue (sweeps/*.toml) and each saved sweep's params, so any
model x strategy combo found anywhere can be recreated exactly (the simulations are seeded).

Defaults are today's baseline. A combo's identity is `settings_key` (all settings); the stage-pricing
cache is keyed by `model_key` (only the settings that change prices), plus the data each stage saw.
Optional settings (default None, "not set") added later are left out of both keys while unset, so
every key saved before they existed stays the same.

    s = Settings.from_dict({"variant": "gridq+pretrain", "half_life_days": 90})
    with s.applied():            # model globals set for the duration (restored afterwards)
        ...
"""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass

STAGES = ("pre-weekend", "after FP1", "after FP2", "after FP3", "after SQ", "after Sprint", "after Quali")
KINDS = ("race_win", "race_podium", "race_h2h", "race_constructor_top", "race_pole")
OPT_KINDS = ("race_top10",)          # opt-in market kinds: selectable in `market_kinds`, never in its default
DEFAULT_SEED = 42                # pricing.diagnostic's and the signal engine's seed
VENUES = ("polymarket", "kalshi", "og")   # exchanges a sweep / the signal engine can trade (markets/venue_replay.py)
DEFAULT_VENUE = "polymarket"
TAPE_SOURCES = ("ours", "probalytics")   # whose recorded tape a backtest venue reads (markets/probalytics.py)
STAGE_MODES = ("sessions", "weekend")   # where a sweep's stages come from (pipelines/season_sweep.py); F1: sessions


@dataclass(frozen=True)
class Setting:
    name: str
    group: str                   # model | timing | taker | maker | markets
    label: str
    type: str                    # float | int | bool | choice | multi | text
    default: object
    min: float | None = None
    max: float | None = None
    choices: tuple = ()
    help: str = ""
    target: str = ""             # "module.ATTR" set while applied (model settings)
    unset: object = None         # a value that means "not set" (stored as None, so it stays out of the keys)


SETTINGS = [
    # --- model: these change prices (model_key) ---------------------------------------------------
    Setting("variant", "model", "Model variant", "text", "baseline",
            help="Switch combination joined with +, e.g. gridq+pretrain+reset (see Formula 1 > Model variants)."),
    Setting("sims", "model", "Simulations per stage", "int", 4000, 1000, 50000),
    Setting("half_life_days", "model", "Recency half-life (days)", "float", 120.0, 20, 720,
            help="How fast old results stop counting in the pace models.", target="model.HALF_LIFE_DAYS"),
    Setting("track_features", "model", "Track/sector features", "bool", True,
            help="Car's fast-vs-slow sector profile matched to each track."),
    Setting("practice_prior", "model", "Practice pace prior", "bool", True,
            help="Use this weekend's practice laps once they exist.", target="practice.USE_PRACTICE"),
    Setting("ridge_team", "model", "Team pace shrinkage (events)", "float", 2.0, 0, 20,
            help="Prior weight pulling each team's base pace to the field.", target="model.RIDGE_A"),
    Setting("ridge_slope", "model", "Track-slope shrinkage (events)", "float", 6.0, 0, 50,
            help="Prior weight pulling each team's fast/slow-sector slope to 0.", target="model.RIDGE_B"),
    Setting("driver_prior_n", "model", "Driver offset shrinkage", "float", 3.0, 0, 30,
            help="How many races before a driver's gap to the teammate is trusted.", target="model.DRIVER_PRIOR_N"),
    Setting("teammate_corr", "model", "Teammates share noise", "bool", True, target="model.TEAMMATE_CORR"),
    Setting("finish_rho_scale", "model", "Teammate finish correlation scale", "float", 1.0, 0, 2,
            target="model.FINISH_RHO_SCALE"),
    Setting("calib_power", "model", "Win-price calibration power", "text", None,
            help="Sharpen (above 1) the race-win prices as p^a, resampling the sims so every market follows "
                 "(C48). A number, or walk_forward: the power fitted on earlier races only "
                 "(sports/f1/calibration.toml). Empty = off.", target="model.CALIB_POWER"),
    Setting("reset_weight", "model", "Regulation-reset carry-over", "float", 0.25, 0, 1,
            help="With the reset switch: weight of earlier seasons' car data.", target="model.REG_RESET_WEIGHT"),
    Setting("seed", "model", "Monte Carlo seed", "int", None, 0, 2**31 - 1,
            help="Empty = today's fixed seed (42). Set different seeds for independent noise draws in a search."),
    # --- entry timing ---------------------------------------------------------------------------------
    Setting("stages", "timing", "Stage mode", "choice", None, choices=STAGE_MODES, unset="sessions",
            help="Where the stages come from: sessions (the session schedule, one pricing per stage; F1's) or weekend "
                 "(the schema's fixed weekend stages, one pricing per race; NASCAR's and MotoGP's). Unset = the sport's "
                 "default (sports/<code>.toml [sweep] stages), so earlier results stay reproducible."),
    Setting("taker_stages", "timing", "Stages takers may trade", "multi", STAGES, choices=STAGES,
            help="Applies to every taker strategy (update, hold, after quali, stage-aware)."),
    Setting("late_stages", "timing", "Stage-aware taker: stop at", "multi", ("after FP3", "after Quali"), choices=STAGES,
            help="Stages the stage-aware taker skips."),
    # --- taker ----------------------------------------------------------------------------------------
    Setting("min_edge", "taker", "Min edge to act (prob.)", "float", 0.05, 0.005, 0.5),
    Setting("min_edge_h2h", "taker", "Min edge for head-to-head markets", "float", None, 0.005, 0.5,
            help="Empty = same as min edge."),
    Setting("min_edge_by_kind", "taker", "Min edge per market kind", "map", None, 0.005, 0.5, choices=KINDS,
            help="kind=edge pairs, e.g. race_h2h=0.05,race_podium=0.08; each overrides min edge (and the "
                 "head-to-head one) for its kind. Empty = none."),
    Setting("stake_per_edge", "taker", "Stake per unit edge ($)", "float", 250.0, 10, 5000,
            help="Target cost = this x edge, e.g. 250 x 0.10 = $25."),
    Setting("max_stake", "taker", "Max stake per market ($)", "float", 50.0, 1, 5000),
    Setting("cost", "taker", "Cost per share per trade ($)", "float", 0.01, 0, 0.1),
    Setting("thin_edge_mult", "taker", "Trade thin markets at edge x (min edge)", "float", None, 1, 20,
            help="A market under the volume floor is still traded when the edge is at least this many times "
                 "the minimum and a recorded order book (at most 10 min old) shows size at the touch; the "
                 "stake is capped at that size and only buys are made there. Empty = thin markets are skipped."),
    Setting("bankroll", "taker", "Starting bankroll ($)", "float", None, 10, 1e7,
            help="Bankroll-aware sizing: stakes scale with the balance after earlier weekends. Empty = fixed sizing."),
    Setting("bankroll_growth", "taker", "Bankroll growth rule", "choice", None, choices=("linear", "sqrt"),
            unset="linear",
            help="With a bankroll: how stakes follow the balance. linear = balance / bankroll; sqrt = the square root "
                 "of that (square-root staking: +300% -> 2x, -75% -> 0.5x, 0 once it's gone). Empty = linear."),
    Setting("kelly", "taker", "Kelly fraction", "float", None, 0.05, 1.0,
            help="With a bankroll: stake = this x the Kelly fraction x the current balance, at the touch, still capped "
                 "at the max stake per market (0.5 = half Kelly). Empty = linear sizing (stake per unit edge)."),
    Setting("root_unit", "taker", "Root unit ($)", "float", None, 1, 5000,
            help="The root unit system (Circles Off ep. 181): stake = this / sqrt(net odds) = this x sqrt(c / (1 - c)) "
                 "for a contract costing c, instead of stake per unit edge: 1 unit at evens, 0.5 at 4/1, 2 at 1/4. Still "
                 "capped at the max stake; scales with the bankroll like the linear rule. Not with Kelly. Empty = off."),
    Setting("max_deployed", "taker", "Max capital deployed per weekend ($)", "float", None, 1, 1e7,
            help="Across all markets; buys over the cap are cut to fit. Empty = no cap."),
    # --- maker ----------------------------------------------------------------------------------------
    Setting("half_spread", "maker", "Quote half-spread ($)", "float", 0.02, 0.005, 0.2),
    Setting("size", "maker", "Shares per quote", "float", 50.0, 1, 1000),
    Setting("max_pos", "maker", "Max inventory per market (shares)", "float", 250.0, 10, 5000),
    Setting("skew", "maker", "Inventory skew", "float", 1.0, 0, 5),
    Setting("max_disagree", "maker", "Don't quote beyond |fair - market|", "float", 0.15, 0.01, 1),
    Setting("maker_min_volume_24h", "maker", "Maker: min $ traded in prior 24 h", "float", None, 0, 100000,
            help="Per market, before quoting it. Empty = the replay's own $100 (the Markets filter is the takers')."),
    Setting("fill", "maker", "Fill rule", "choice", "through", choices=("through", "touch", "queue"),
            help="through = a trade must cross our price (conservative); touch = at our price; "
                 "queue = at our price once the recorded book's queue ahead of us is served."),
    Setting("info_skew", "maker", "Info-timed skew (maker_skew / maker_all)", "float", 2.0, 0, 10),
    Setting("widen", "maker", "Widen factor on bad markouts (maker_widen / maker_all)", "float", 1.5, 1, 5),
    # --- markets --------------------------------------------------------------------------------------
    Setting("market_kinds", "markets", "Market kinds traded", "multi", KINDS, choices=KINDS + OPT_KINDS,
            help="race_top10 (Kalshi's top-10 finishers) is opt-in: it is priced by the model but not in the default set."),
    Setting("coherence_tol_by_kind", "markets", "Group coherence tolerance per kind", "map", None, 0.01, 2, choices=KINDS + OPT_KINDS,
            help="kind=tolerance pairs, e.g. race_podium=0.35: how far a multi-outcome group's prices may sum "
                 "from its target (as a share of it) before its markets are skipped. Empty = 0.25 for every kind."),
    Setting("min_volume_24h", "markets", "Min $ traded in prior 24 h", "float", 50.0, 0, 100000),
    Setting("venue", "markets", "Venue", "choice", None, choices=VENUES, unset=DEFAULT_VENUE,
            help="Whose markets and recorded tape the strategies trade: polymarket (the default), kalshi "
                 "(its links, its tape read per market ticker, its maker fee) or og (OG.com: its links and tape, "
                 "its flat fee per contract on the taker's cost). Unset = polymarket."),
    Setting("taker_fill", "markets", "Taker fill price", "choice", None, choices=("price", "touch", "book"), unset="price",
            help="price = Polymarket stages trade at the venue's price (our 5-minute history, or Probalytics' book mid), "
                 "as before; touch = a venue that has a book quote gives every stage one, so the taker buys at the ask "
                 "and sells at the bid (Kalshi always does). Our own Polymarket tape has no quote, so it stays on price. "
                 "book = touch, and every order is walked level by level through Probalytics' full book at that "
                 "moment (cut to what the book holds, no cost per share: the walk is the spread and the slippage); "
                 "needs tape_source = probalytics. Unset = price. Not tuned: no default change without a decision-log row."),
    Setting("sell_floor", "markets", "Taker sale floor", "choice", None, choices=("none", "fair", "edge"), unset="none",
            help="The lowest price a taker sale (a rebalance down) may fill at, net of cost and fee: none = no floor "
                 "(one-price fills sell at any bid; taker_fill = book sells the touch level only), fair = at or above "
                 "our fair, edge = at or above fair - min_edge (each side its own fair). Applies to every taker_fill. "
                 "Unset = none. Not tuned: no default change without a decision-log row."),
    Setting("max_spread", "markets", "Max quoted spread ($)", "float", None, 0.01, 1.0,
            help="A stage whose quote is wider than this (ask - bid) is not tradeable: a 0.10 / 0.89 head-to-head book "
                 "has no market to fill at (backtest fill-error probe, 2026-10-14). Applies where stages carry a quote "
                 "(Kalshi; Polymarket with taker_fill = touch on Probalytics' tape). Empty = no limit."),
    Setting("tape_source", "markets", "Tape source", "choice", None, choices=TAPE_SOURCES, unset=TAPE_SOURCES[0],
            help="Whose tape the venue is read from: ours (the recorder and archive, the default) or probalytics "
                 "(Probalytics' books and fills under data/probalytics/, by the sport's [markets.probalytics] tag; "
                 "markets/probalytics.py). RACINGLINES_TAPE_SOURCE=probalytics still switches it on too. Unset = ours."),
]
BY_NAME = {s.name: s for s in SETTINGS}
# The split (docs/backtest-core.md): the pricing model's own group, and the groups any sport's backtest shares
# (entry timing, taker, maker, markets and their guards). Another model family's settings are its own model
# group plus these once it trades a venue (models/inputs/run_times/settings.py is model-only so far).
MODEL_SETTINGS = [s for s in SETTINGS if s.group == "model"]
SHARED_SETTINGS = [s for s in SETTINGS if s.group != "model"]
GROUPS = (("model", "Model"), ("timing", "Entry timing"), ("taker", "Taker strategies"),
          ("maker", "Maker strategies"), ("markets", "Markets"))
MODEL_NAMES = [s.name for s in SETTINGS if s.group == "model"]
ALIASES = {"n_sims": "sims"}            # older saved params


def _coerce(s, v):
    if v is None or (s.default is None and isinstance(v, str) and not v.strip()):
        return s.default
    if s.unset is not None and v == s.unset:
        return s.default
    if s.name == "calib_power":                      # off / walk_forward / a power (models/calibration.py)
        from racinglines.models import calibration as CAL
        return CAL.parse(v)
    if s.type == "float":
        return float(v)
    if s.type == "int":
        return int(float(v))
    if s.type == "bool":
        return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    if s.type == "multi":
        items = [x.strip() for x in v.split(",")] if isinstance(v, str) else list(v)
        bad = [x for x in items if x and x not in s.choices]
        if bad:
            raise ValueError(f"{s.label}: unknown {bad}; choose from {list(s.choices)}")
        return tuple(x for x in s.choices if x in items)         # canonical order
    if s.type == "map":                                      # "kind=value,..." -> canonical text (sorted, floats)
        pairs = {}
        for item in (v.split(",") if isinstance(v, str) else [f"{k}={x}" for k, x in dict(v).items()]):
            if not item.strip():
                continue
            k, _, x = item.partition("=")
            k = k.strip()
            if k not in s.choices:
                raise ValueError(f"{s.label}: unknown {k!r}; choose from {list(s.choices)}")
            x = float(x)
            if not s.min <= x <= s.max:
                raise ValueError(f"{s.label}: {k}={x} is outside {s.min}-{s.max}")
            pairs[k] = x
        return ",".join(f"{k}={pairs[k]:g}" for k in sorted(pairs)) or s.default
    if s.type == "choice":
        if v not in s.choices:
            raise ValueError(f"{s.label}: pick one of {list(s.choices)}")
        return v
    return str(v).strip()


class Settings(dict):
    """A full, validated set of sweep settings (missing ones take their defaults).

    The schema is a class attribute, so another model family defines its own settings by subclassing
    (e.g. racinglines/models/inputs/run_times/settings.py) and keeps the same keys, labels and flags."""

    SPEC = SETTINGS
    BY = BY_NAME
    MODEL = MODEL_NAMES
    ALIASES = ALIASES
    DEFAULT_SEED = DEFAULT_SEED
    DEFAULT_VENUE = DEFAULT_VENUE    # the venue while `venue` is unset (a sport's sweep settings: its first venue)

    @classmethod
    def from_dict(cls, d=None, strict=True):
        d = {cls.ALIASES.get(k, k): v for k, v in (d or {}).items()}
        unknown = [k for k in d if k not in cls.BY]
        if unknown and strict:
            raise ValueError(f"unknown settings {unknown}")
        out = cls()
        for s in cls.SPEC:
            v = _coerce(s, d.get(s.name))
            if s.type in ("float", "int") and v is not None and s.min is not None and not s.min <= v <= s.max:
                raise ValueError(f"{s.label}: {v} is outside {s.min}-{s.max}")
            out[s.name] = v
        out._validate()
        return out

    def _validate(self):
        from racinglines.models import variants as V
        V.switches(self["variant"])                                  # raises on an unknown switch

    @classmethod
    def from_run_params(cls, params):
        """Settings of a saved sweep, including ones saved before this schema existed."""
        p = dict(params or {})
        if isinstance(p.get("settings"), dict):
            return cls.from_dict(p["settings"], strict=False)
        return cls.from_dict({k: v for k, v in p.items() if cls.ALIASES.get(k, k) in cls.BY}, strict=False)

    def changed(self):
        """{name: value} of the settings that differ from the defaults."""
        return {k: v for k, v in self.items() if v != self.BY[k].default}

    def _key(self, names):
        names = [k for k in names if not (self.BY[k].default is None and self[k] is None)]   # unset optionals
        vals = {k: self[k] for k in names}
        if vals.get("calib_power") == "walk_forward":         # the fitted table is part of the pricing
            from racinglines.models import calibration as CAL
            vals["calib_power"] = f"walk_forward@{CAL.table_hash(getattr(self, 'CALIB_SPORT', 'f1'))}"
        blob = json.dumps(vals, sort_keys=True, default=list)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @property
    def key(self):
        """Identity of the whole combo (every setting)."""
        return self._key(sorted(self))

    @property
    def rng_seed(self):
        """The Monte Carlo seed: the `seed` setting, or today's fixed one when it's unset."""
        return self.DEFAULT_SEED if self["seed"] is None else self["seed"]

    @property
    def model_key(self):
        """Identity of the pricing: only the settings that change the model's prices."""
        return self._key(self.MODEL)

    def label(self):
        ch = {k: v for k, v in self.changed().items() if k != "variant"}
        extra = ", ".join(f"{k}={','.join(v) if isinstance(v, tuple) else v}" for k, v in ch.items())
        return self["variant"] + (f" · {extra}" if extra else "")

    def to_json(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.items()}

    def argv(self):
        """`racinglines f1 ... sweep` flags for the non-default settings (the variant is a global flag)."""
        out = []
        for k, v in self.changed().items():
            if k == "variant":
                continue
            flag = "--" + k.replace("_", "-")
            out += [flag, ",".join(v) if isinstance(v, tuple) else str(v).lower() if isinstance(v, bool) else str(v)]
        return out

    @contextmanager
    def applied(self):
        """Model settings (and the variant's switches) in force for the duration."""
        from racinglines.models.inputs import laps as M
        from racinglines.models.inputs import practice as PR
        from racinglines.models import variants as V
        mods = {"model": M, "practice": PR}
        old = []
        try:
            for s in SETTINGS:
                if s.target:
                    mod, attr = s.target.split(".")
                    old.append((mods[mod], attr, getattr(mods[mod], attr)))
                    setattr(mods[mod], attr, self[s.name])
            with V.use(self["variant"]):
                yield self
        finally:
            for mod, attr, v in reversed(old):
                setattr(mod, attr, v)


def with_defaults(cls, defaults, where="[model.defaults]"):
    """`cls` with a sport's per-sport defaults (sports/<code>.toml [model.defaults], read by the one engine,
    models/model_global.py). Unknown names are an error; a default equal to the code's leaves every key unchanged,
    since the keys hash the values (Settings._key)."""
    from dataclasses import replace
    defaults = dict(defaults or {})
    unknown = sorted(set(defaults) - set(cls.BY))
    if unknown:
        raise ValueError(f"{where} names unknown settings {unknown}; choose from {sorted(cls.BY)}")
    if not defaults:
        return cls
    spec = [replace(s, default=_coerce(s, defaults[s.name])) if s.name in defaults else s for s in cls.SPEC]
    return type(cls.__name__, (cls,), dict(SPEC=spec, BY={s.name: s for s in spec}, MODEL=list(cls.MODEL)))


def venue_of(settings):
    """The exchange a settings set trades: its `venue`, or its class's default (polymarket) while unset."""
    return settings.get("venue") or getattr(settings, "DEFAULT_VENUE", DEFAULT_VENUE)


def parse_map(v):
    """A map setting's canonical text -> {key: float} ({} when unset)."""
    return {k: float(x) for k, _, x in (i.partition("=") for i in v.split(","))} if v else {}


def add_arguments(parser, cls=None, skip=()):
    """One flag per setting (the variant is the f1 group's global --variant). skip: settings whose flag the command
    already has with another meaning (walk-forward's --venue scores exchanges; the sweep's venue picks the tape)."""
    for s in (cls or Settings).SPEC:
        if s.name == "variant" or s.name in skip:
            continue
        dflt = ",".join(s.default) if s.type == "multi" else "not set" if s.default is None else s.default
        # argparse formats help with %, so a literal % (bankroll_growth's "+300%") must be doubled
        hlp = f"{s.label} (default {dflt})." + (f" {s.help}" if s.help else "")
        kw = dict(default=None, help=hlp.replace("%", "%%"))
        parser.add_argument("--" + s.name.replace("_", "-"), **kw)


def from_args(args, cls=None, skip=()):
    cls = cls or Settings
    d = {s.name: getattr(args, s.name) for s in cls.SPEC
         if s.name != "variant" and s.name not in skip and getattr(args, s.name, None) is not None}
    if "variant" in cls.BY:
        d["variant"] = getattr(args, "variant", "baseline") or "baseline"
    return cls.from_dict(d)


def data_key(view):
    """Fingerprint of the data a stage was priced from: row counts and column sums of the as-of view,
    leaving out database ids (they differ between databases built from the same data), so a cached
    stage is reused only when it saw exactly the same data, on any machine."""
    parts = []
    for name in ("res", "prof", "drivers", "sectors", "practice", "races"):
        df = getattr(view, name, None)
        if df is None:
            parts.append(f"{name}:-")
            continue
        num = df.select_dtypes("number")
        num = num[[c for c in num.columns if not (c == "id" or str(c).endswith("_id"))]]
        parts.append(f"{name}:{len(df)}:{float(num.fillna(0).to_numpy().sum()) if len(num.columns) else 0:.8g}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]
