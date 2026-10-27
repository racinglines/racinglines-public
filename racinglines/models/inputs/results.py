"""The `results` input layer of the one engine (racinglines/models/model_global.py, docs/backtest-core.md): the
engine's core for a sport whose schema declares finishing positions and nothing more granular ([model] inputs =
["results", ...]). It is the "simple / fuzzy-data" core: it needs no laps, no sectors and no fixed schedule, so it
prices a sport where entrants miss events, fields differ in size and nobody races every round (road cycling), and
it is the level every sport with price tapes can be backtested at (`--model global`). Moved here unchanged from
models/model_global.py (decision log 2026-10-11 "One engine").

A sport's schema sets its inputs and hooks, and nothing here names a sport:

    [sport]
    pricing_model = "racinglines.models.model_global:GlobalModel"
    [model]
    inputs = ["results", "qualifying", "fastest_lap"]   # the layers it reads (models/inputs/__init__.py)
    source = "motogp_api"           # events.source whose race results are the history
    group = "team"                  # results column / extra key naming the entrant's team (group prior)
    prior = "uci_points"            # results.extra key: a pre-race rating of the entrant (rating prior)
    sessions = ["qual"]             # input "qualifying": other classifications of the weekend, by rounds.kind
    best_lap = "race_fastest_lap"   # input "fastest_lap": an indicator kind drawn from each race's best-lap order
    history = "category"            # each category its own history: a race is priced from its category's races only
    [model.defaults]                # per-sport defaults for the Settings below (decision-log entries)
    noise = 0.7

How it prices a race (every number from results on days strictly before the event):

1. Each finish becomes a field-size-free strength z = -probit((position - 0.5) / starters), so 2nd of 150 and
   2nd of 5 are not the same evidence.
2. An entrant's strength is the recency-weighted mean of their z (weight 0.5 ** (age / half_life_days)),
   shrunk toward a prior by `prior_weight` pseudo-starts, so a rider with two starts is mostly the prior and
   one with thirty is mostly their results. The prior is 0 (the field average), plus the team's form
   (`group_weight`) and the entrant's rating rank in the field (`rating_weight`) where the schema names them.
3. The race is a draw of strength + noise, with extra spread for entrants with few starts (`uncertainty`),
   and a retirement draw from the entrant's shrunk DNF rate (`dnf`), retirements classified last.

The start list is `ev.info["field"]` (walk-forward builds it from the event's non-DNS rows; the replay passes the
race's results); without one, every entrant with a start in the last half-life is priced.

Side classifications (package C12b; docs/f1-roadmap.md decision log 2026-10-07 "NASCAR pole and fastest lap",
provisional). The schema may name more orders to draw for the same field, each the race's structure on its own
history (steps 1 and 2 on that classification's past results, then strength + noise, no retirements, no team or
rating prior; the same half-life, prior weight, noise and uncertainty as the race, none tuned):

    sessions = ["qual"]     each a rounds.kind: its order lands in stage_rank[kind], so the stage kinds price it
                            (race_pole: stage_rank["qual"] == 1) and settle on the results' qual_position
    best_lap = "<kind>"     the order of each past race's fastest laps (each car's best lap time in the race round,
                            laps table): the car drawn first sets the fastest lap, sims.indicators[<kind>]; settled
                            from the race's laps when every lap of the race distance is stored (laps_complete)

A side classification is drawn only when its history has a result before the cutoff (else that kind stays unpriced),
on a generator seeded from the shared one without advancing it, so the race prices, and every later race's, are the
same with or without it.

Calibration power (C48 for the results level, docs/f1-calibration.md "Other sports"): `calib_power` (off by default)
makes the race's win chances p^a / sum p^a by resampling the drawn races (models/calibration.py), so every market
priced from them (podium, top n, head-to-head, team) follows the win price. walk_forward reads
sports/<sport>/calibration.toml, fitted on earlier races only (scripts/models/fit_calibration.py --sport). The side
classifications are not resampled. The resample draws from a side generator, so unset leaves every price unchanged.
"""

from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pandas as pd

from racinglines.markets import payoffs as P
from racinglines.models import calibration as CAL
from racinglines.models import outcomes as O
from racinglines.models.race_model import Event
from racinglines.pipelines import sweep_settings as SS

SETTINGS = [
    SS.Setting("sims", "model", "Simulations per race", "int", 2000, 100, 100000),
    SS.Setting("half_life_days", "model", "Recency half-life (days)", "float", 240.0, 10, 3650,
               help="Age at which a finish counts half."),
    SS.Setting("prior_weight", "model", "Prior weight (pseudo-starts)", "float", 3.0, 0, 100,
               help="How many starts of evidence it takes to outweigh the prior."),
    SS.Setting("noise", "model", "Race-day noise (strength units)", "float", 0.8, 0.05, 3,
               help="Spread of one race around an entrant's strength; a finish's strength is ~N(0,1) across the field."),
    SS.Setting("uncertainty", "model", "Extra spread for few starts", "float", 0.5, 0, 3,
               help="Added variance tau^2 / (decayed starts + 1)."),
    SS.Setting("group_weight", "model", "Team-form prior weight", "float", 0.0, 0, 2,
               help="Needs [model] group in the schema. 0 = off."),
    SS.Setting("rating_weight", "model", "Rating prior weight", "float", 0.0, 0, 2,
               help="Needs [model] prior in the schema. 0 = off."),
    SS.Setting("dnf", "model", "Model retirements", "bool", True,
               help="Draw retirements from each entrant's shrunk DNF rate; off = everyone finishes."),
    SS.Setting("calib_power", "model", "Win-price calibration power", "text", None,
               help="Sharpen (above 1) the race-win prices as p^a, resampling the sims so every market follows "
                    "(C48). A number, or walk_forward: the power fitted on earlier races only "
                    "(sports/<sport>/calibration.toml). Empty = off."),
    SS.Setting("seed", "model", "Monte Carlo seed", "int", None, 0, 2**31 - 1),
]

MIN_FINISHERS = 4          # an event with fewer classified finishers is neither history nor priced
SIDE_SEED = 7_401          # salt of the side classifications' generator (sessions, best lap)
CALIB_SEED = 7_501         # salt of the calibration resample's generator
DNF_PRIOR_N = 5.0          # pseudo-starts shrinking an entrant's DNF rate toward the field's
REQUIRED = ("season", "race", "date", "athlete_id", "position", "status")
CATEGORY = "category"      # the results frame's categories.code column, read when [model] history = "category"


class ByCategory(dict):
    """{category code: history}: a model's history split by category ([model] history = "category"); price() reads
    the one of the event's category (Event.info["category"])."""


def by_category(model, data):
    """Whether `model` keeps a history per category for `data` (the schema asked and the frame carries the codes)."""
    return (model.cfg or {}).get("history") == CATEGORY and CATEGORY in data


def split_history(model, data, settings):
    """ByCategory of model.history on each category's rows (the column dropped, so each is an ordinary history)."""
    return ByCategory({str(c): model.history(g.drop(columns=CATEGORY), settings)
                       for c, g in data.groupby(CATEGORY, sort=False)})


def event_history(hist, ev):
    """The history `ev` is priced from: hist itself, or with a ByCategory the event's category's (None: none yet)."""
    if not isinstance(hist, ByCategory):
        return hist
    return hist.get(str((ev.info or {}).get(CATEGORY)))
_N = NormalDist()


class GlobalSettings(SS.Settings):
    SPEC = SETTINGS
    BY = {s.name: s for s in SETTINGS}
    MODEL = [s.name for s in SETTINGS]
    ALIASES = {}

    def _validate(self):
        pass

    def label(self):
        return ", ".join(f"{k}={v}" for k, v in self.changed().items()) or "baseline"


def _settings_class(defaults):
    """GlobalSettings with a sport's [model.defaults] as the defaults (unknown names are an error)."""
    return SS.with_defaults(GlobalSettings, defaults)


def _probit(q):
    return np.fromiter((_N.inv_cdf(float(x)) for x in np.ravel(q)), float, np.size(q)).reshape(np.shape(q))


def _day(ts):
    """A timestamp as whole days since 1970 (a date, a naive or an aware timestamp)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.normalize().value // 86_400_000_000_000)


def _side_rng(rng, salt):
    """A generator seeded from `rng`'s current state, read and not advanced (inputs.laps._side_rng's rule):
    draws from it leave `rng`, and everything drawn from it next, unchanged."""
    import hashlib
    import json
    state = json.dumps(rng.bit_generator.state, sort_keys=True, default=str)
    return np.random.default_rng([salt, int(hashlib.sha256(state.encode()).hexdigest()[:16], 16)])


def best_lap_order(laps):
    """Each race's order of best laps from (race, athlete_id, lap_time_ms) rows, one per lap or per car: position =
    the competition rank of the car's fastest lap time in the race (ties share it; times <= 0 or missing are no lap),
    status OK. Other columns of the first row per (race, athlete_id) are kept."""
    d = laps[pd.to_numeric(laps["lap_time_ms"], errors="coerce") > 0].copy()
    if d.empty:
        return d.assign(position=pd.Series(dtype=float), status=pd.Series(dtype=str))
    d["lap_time_ms"] = d["lap_time_ms"].astype(float)
    best = d.groupby(["race", "athlete_id"], sort=False)["lap_time_ms"].transform("min")
    d = d.assign(lap_time_ms=best).drop_duplicates(["race", "athlete_id"])
    d["position"] = d.groupby("race")["lap_time_ms"].rank(method="min")
    d["status"] = "OK"
    return d.reset_index(drop=True)


def _frame(df):
    """The model's results frame: season, race (event key), date, athlete_id, position, status, and optionally
    name, team, prior, points. One row per entrant per race."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"results frame lacks {missing}")
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["athlete_id"] = d["athlete_id"].astype(int)
    d["position"] = pd.to_numeric(d["position"], errors="coerce")
    d["status"] = d["status"].fillna("OK").astype(str).str.upper()
    return d.sort_values(["date", "race", "position"], kind="stable").reset_index(drop=True)


def _strengths(d):
    """Per row of the non-DNS rows: z (NaN unless the row is a classified finish) and fin (that finish)."""
    d = d[d["status"] != "DNS"]
    starters = d.groupby("race")["athlete_id"].transform("size").to_numpy(float)
    fin = ((d["status"] == "OK") & d["position"].notna() & (d["position"] >= 1)).to_numpy()
    q = np.clip((d["position"].fillna(1).to_numpy(float) - 0.5) / starters, 1e-6, 1 - 1e-6)
    return d, np.where(fin, -_probit(q), np.nan), fin


def dnf_rate(w, fin, Wall, Wf):
    """Each entrant's retirement chance: their decayed non-finishes over decayed starts (Wall - Wf over Wall), shrunk
    toward the field's decayed rate by DNF_PRIOR_N pseudo-starts. w, fin: the history rows' weights and finished flags."""
    rbar = float(w[~fin].sum() / w.sum())
    return (Wall - Wf + DNF_PRIOR_N * rbar) / (Wall + DNF_PRIOR_N)


def _groups(hist, n, j, inf, field, ev):
    """Per entrant of `field`: its team (the group the team markets name), from the event's entry list
    (`ev.info["groups"]`) or else the entrant's latest team in the history before the event; an entrant with neither
    is a group of its own. None when there are no teams at all. Draws nothing, so the prices are unchanged."""
    known = dict((ev.info or {}).get("groups") or {})
    if not known and hist.team is None:
        return None
    last = {}
    if hist.team is not None:
        team = hist.team[:n]
        has = inf & pd.notna(team)
        last = pd.Series(team[has], index=j[has]).groupby(level=0).last().to_dict()
    out = []
    for i, a in enumerate(field):
        g = known.get(int(a), last.get(i))
        out.append(str(g) if g is not None and pd.notna(g) else f"entrant:{int(a)}")
    return out


class ResultsCore:
    """The results core. The engine (model_global.GlobalModel) binds it to a sport's schema through `bind`; unbound
    it prices whatever frame `load(data=...)` is given."""

    sport = None
    name = "global_results"                # model_runs.model
    Settings = GlobalSettings
    cfg = None                             # {competition, source, group, prior}: the schema's data hooks
    noise_unit = "strength"                # `noise` is in strength units, not finishing places

    @classmethod
    def bind(cls, sport, schema, inputs, defaults):
        """The class attributes of the engine bound to `sport`: its data hooks (cfg) and Settings. The side
        classifications are read only for the inputs the schema declares: [model] sessions with "qualifying",
        [model] best_lap with "fastest_lap"."""
        m = schema.get("model", {})
        source = m.get("source") or schema.get("replay", {}).get("source")
        if not source:
            raise ValueError(f"sports/{sport}.toml: the results input needs [model] source (or [replay] source)")
        sessions = list(m.get("sessions") or [])
        best_lap = m.get("best_lap")
        if any(not isinstance(k, str) or not k for k in sessions) or (best_lap is not None and not isinstance(best_lap, str)):
            raise ValueError(f"sports/{sport}.toml: [model] sessions is a list of rounds.kind names, best_lap a kind")
        for hook, value, need in (("sessions", sessions, "qualifying"), ("best_lap", best_lap, "fastest_lap")):
            if value and need not in inputs:
                raise ValueError(f"sports/{sport}.toml: [model] {hook} is set but [model] inputs doesn't declare {need!r}")
        cfg = dict(competition=schema["competition"]["code"], source=source, group=m.get("group"), prior=m.get("prior"))
        cfg.update({k: v for k, v in dict(sessions=sessions, best_lap=best_lap).items() if v})   # only when named
        if m.get("history") not in (None, CATEGORY):
            raise ValueError(f"sports/{sport}.toml: [model] history is {CATEGORY!r} or unset, not {m['history']!r}")
        if m.get("history"):                 # a history per category: races keyed by event and category
            cfg.update(history=CATEGORY, race_key="category")
        S = _settings_class(defaults)        # CALIB_SPORT: whose calibration table a walk_forward model key hashes
        return dict(cfg=cfg, Settings=type(S.__name__, (S,), dict(CALIB_SPORT=sport)))

    sides = None                           # {name: results frame} of the side classifications, set by load()

    # --- data ---------------------------------------------------------------------------------------
    def load(self, engine_url=None, data=None, sides=None):
        """The race results frame. The side classifications the schema names ([model] sessions, best_lap) are read
        too and kept on the model (self.sides: {stage or kind: frame in the race frame's shape; the best-lap frame
        adds laps_complete}); with `data`, `sides` gives them (default: none)."""
        if data is not None:
            self.sides = {k: _frame(v) for k, v in (sides or {}).items()}
            return _frame(data)
        if self.cfg is None:
            raise ValueError("an unbound GlobalModel needs data=; bind it with for_sport(code) to read the database")
        from sqlalchemy import text

        from racinglines.db.config import get_engine

        group, prior = self.cfg["group"], self.cfg["prior"]
        params = dict(c=self.cfg["competition"], src=self.cfg["source"])
        # the schema names a results column ("team") or a results.extra key, as a bound parameter
        gsel = "r.team AS team" if group == "team" else "r.extra->>:gk AS team" if group else "NULL AS team"
        psel = ("CASE WHEN r.extra->>:pk ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN (r.extra->>:pk)::float END AS prior"
                if prior else "NULL::float AS prior")
        if group and group != "team":
            params["gk"] = group
        if prior:
            params["pk"] = prior
        with get_engine(engine_url).connect() as conn:
            df = pd.read_sql(text(self._kind_sql(f", {gsel}, {psel}")), conn,
                             params=dict(params, k=self.cfg.get("kind", "race")))
            self.sides = self._load_sides(conn, params)
        return _frame(df)

    def _kind_sql(self, cols=""):
        """The results of one round kind (:k) of the competition (:c) and events.source (:src), one row per entrant
        per race. Optional hooks (cfg, set by a model that needs them; unset gives the engine's query): race_key
        "category" keys a race by event and category (events.source_key + "_" + categories.code, a sport with several
        categories per event), times adds results.time_ms, source None reads every source of the competition."""
        cfg = self.cfg or {}
        by_cat = cfg.get("race_key") == "category" or cfg.get("history") == CATEGORY
        key = ("ev.source_key || '_' || ca.code AS race, ev.name || ' ' || ca.code AS name" if by_cat
               else "ev.id AS race, ev.name")
        if cfg.get("history") == CATEGORY:   # history "category": the code, so history() can split by it
            key += ", ca.code AS category"
        cols = (", r.time_ms" if cfg.get("times") else "") + cols
        return f"""
                SELECT s.year AS season, {key}, ev.start_date::date AS date, r.athlete_id,
                       r.position, r.status{cols}
                FROM results r JOIN rounds ro ON ro.id = r.round_id AND ro.kind = :k
                JOIN races ra ON ra.id = ro.race_id JOIN events ev ON ev.id = ra.event_id
                {"JOIN categories ca ON ca.id = ra.category_id " if by_cat else ""}JOIN seasons s ON s.id = ev.season_id
                JOIN competitions co ON co.id = s.competition_id
                WHERE co.code = :c {"AND ev.source = :src " if cfg.get("source") else ""}AND r.athlete_id IS NOT NULL
                ORDER BY ev.start_date, {"race" if by_cat else "ev.id"}, r.position NULLS LAST"""

    def _load_kind(self, conn, params, kind):
        """One round kind's results as a frame in the race frame's shape."""
        from sqlalchemy import text
        return _frame(pd.read_sql(text(self._kind_sql()), conn, params=dict(params, k=kind)))

    def _load_sides(self, conn, params):
        """{name: frame}: each [model] sessions round kind's results, and the [model] best_lap order (best_lap_order
        of the race rounds' laps, with laps_complete from the race round)."""
        from sqlalchemy import text
        out = {}
        for kind in (self.cfg or {}).get("sessions") or []:
            out[kind] = self._load_kind(conn, params, kind)
        kind = (self.cfg or {}).get("best_lap")
        if kind:
            df = pd.read_sql(text("""
                SELECT s.year AS season, ev.id AS race, ev.name, ev.start_date::date AS date, r.athlete_id,
                       min(l.lap_time_ms) AS lap_time_ms,
                       coalesce((ro.extra->>'laps_complete')::boolean, true) AS laps_complete
                FROM laps l JOIN results r ON r.id = l.result_id
                JOIN rounds ro ON ro.id = r.round_id AND ro.kind = 'race'
                JOIN races ra ON ra.id = ro.race_id JOIN events ev ON ev.id = ra.event_id
                JOIN seasons s ON s.id = ev.season_id JOIN competitions co ON co.id = s.competition_id
                WHERE co.code = :c AND ev.source = :src AND r.athlete_id IS NOT NULL AND l.lap_time_ms > 0
                GROUP BY s.year, ev.id, ev.name, ev.start_date, r.athlete_id, ro.extra
                ORDER BY ev.start_date, ev.id"""), conn, params=params)
            out[kind] = _frame(best_lap_order(df))
        return out

    @staticmethod
    def data_through(data):
        return data["date"].max()

    def seasons(self, data, settings):
        return sorted(int(s) for s in data["season"].dropna().unique())

    def events(self, data, settings, seasons=None):
        out = []
        for race, g in data.groupby("race", sort=False):
            started = g[g["status"] != "DNS"]
            if int(((started["status"] == "OK") & started["position"].notna()).sum()) < MIN_FINISHERS:
                continue
            season = int(g["season"].iloc[0])
            if seasons and season not in seasons:
                continue
            name = str(g["name"].iloc[0]) if "name" in g and pd.notna(g["name"].iloc[0]) else str(race)
            info = {"field": sorted(started["athlete_id"].unique().tolist())}
            if CATEGORY in g and pd.notna(g[CATEGORY].iloc[0]):          # history "category": the history it reads
                info[CATEGORY] = str(g[CATEGORY].iloc[0])
            if "team" in started and started["team"].notna().any():      # the entry list's teams (group markets)
                t = started.dropna(subset=["team"]).drop_duplicates("athlete_id", keep="last")
                info["groups"] = {int(a): str(x) for a, x in zip(t["athlete_id"], t["team"])}
            out.append(Event(id=race, season=season, cutoff=g["date"].iloc[0], name=name, info=info))
        return sorted(out, key=lambda e: (e.cutoff, str(e.id)))

    def history(self, data, settings):
        if by_category(self, data):
            return split_history(self, data, settings)
        d, z, fin = _strengths(data[data["race"].isin(self._complete_races(data))])
        return SimpleNamespace(
            t=(d["date"].to_numpy("datetime64[D]").astype(np.int64)), a=d["athlete_id"].to_numpy(), z=z, fin=fin,
            team=d["team"].to_numpy() if "team" in d and d["team"].notna().any() else None,
            rating=d["prior"].to_numpy(float) if "prior" in d and d["prior"].notna().any() else None,
            sides={name: self._side_history(f) for name, f in (self.sides or {}).items() if len(f)})

    def _side_history(self, f):
        """A side classification's history: t, a, z, fin as for the race (its events with MIN_FINISHERS placed)."""
        d, z, fin = _strengths(f[f["race"].isin(self._complete_races(f))])
        return SimpleNamespace(t=d["date"].to_numpy("datetime64[D]").astype(np.int64), a=d["athlete_id"].to_numpy(),
                               z=z, fin=fin)

    @staticmethod
    def _complete_races(data):
        started = data[data["status"] != "DNS"]
        ok = ((started["status"] == "OK") & started["position"].notna()).groupby(started["race"]).sum()
        return ok.index[ok >= MIN_FINISHERS]

    # --- pricing ------------------------------------------------------------------------------------
    def price(self, hist, ev, settings, rng):
        """OutcomeSims from finishes on days before ev.cutoff; None when there are none."""
        hist = event_history(hist, ev)
        if hist is None:
            return None
        cut = _day(ev.cutoff)
        n = int(np.searchsorted(hist.t, cut, side="left"))
        if n == 0:
            return None
        t, a, z, fin = hist.t[:n], hist.a[:n], hist.z[:n], hist.fin[:n]
        w = 0.5 ** ((cut - t) / settings["half_life_days"])
        field = list((ev.info or {}).get("field") or [])
        if not field:
            active = pd.Series(w).groupby(a).sum()
            field = sorted(int(x) for x in active.index[active >= 0.5])
            if not field:
                return None
        k = len(field)
        j = pd.Index(field).get_indexer(a)
        inf = j >= 0

        def acc(x, mask):
            return np.bincount(j[mask], weights=x[mask], minlength=k)
        f = fin & inf
        Wf, Wz, Wall = acc(w, f), acc(w * np.where(fin, z, 0.0), f), acc(w, inf)

        prior = np.zeros(k)
        if settings["group_weight"] > 0 and hist.team is not None:
            code, _ = pd.factorize(hist.team[:n])
            ok = fin & (code >= 0)
            tw = np.bincount(code[ok], weights=w[ok], minlength=code.max() + 1 if len(code) else 1)
            tz = np.bincount(code[ok], weights=(w * np.where(fin, z, 0.0))[ok], minlength=len(tw))
            team_mu = tz / (tw + max(settings["prior_weight"], 1e-9))
            last = pd.Series(code[inf & (code >= 0)], index=j[inf & (code >= 0)]).groupby(level=0).last()
            prior[last.index.to_numpy()] += settings["group_weight"] * team_mu[last.to_numpy()]
        if settings["rating_weight"] > 0 and hist.rating is not None:
            r = hist.rating[:n]
            has = inf & ~np.isnan(r)
            last = pd.Series(r[has], index=j[has]).groupby(level=0).last()
            if len(last) >= 3:
                rank = last.rank(method="average").to_numpy()
                prior[last.index.to_numpy()] += settings["rating_weight"] * _probit((rank - 0.5) / len(last))

        den = Wf + settings["prior_weight"]
        mu = np.divide(Wz + settings["prior_weight"] * prior, den, out=prior.copy(), where=den > 0)
        sd = np.sqrt(settings["noise"] ** 2 + settings["uncertainty"] ** 2 / (Wf + 1.0))
        perf = mu + sd * rng.standard_normal((settings["sims"], k))
        if settings["dnf"]:
            dnf = rng.random((settings["sims"], k)) < dnf_rate(w, fin, Wall, Wf)
        else:
            dnf = np.zeros((settings["sims"], k), bool)
        rank = np.argsort(np.argsort(-(perf - 1e3 * dnf), axis=1), axis=1) + 1
        stage_rank, indicators = self._price_sides(getattr(hist, "sides", None) or {}, cut, field, settings, rng)
        a = CAL.power(settings.get("calib_power"), self.sport, ev.cutoff, False)
        if a is not None:                  # on a side generator: unset (or a = 1) leaves every draw unchanged
            idx = CAL.resample_index(np.argmax(rank == 1, axis=1), k, a, _side_rng(rng, CALIB_SEED))
            rank, dnf = rank[idx], dnf[idx]
        return O.OutcomeSims(entrants=[int(x) for x in field], rank=rank, finished=~dnf,
                             groups=_groups(hist, n, j, inf, field, ev), stage_rank=stage_rank, indicators=indicators)

    def _price_sides(self, sides, cut, field, settings, rng):
        """({session: rank}, {best-lap kind: drawn first}) for the side classifications with history before `cut`:
        strength from that history (recency-weighted, shrunk to 0 by prior_weight) + noise, no retirements; every
        draw on a side generator (the race's and later draws unchanged)."""
        stage_rank, indicators = {}, {}
        best = (self.cfg or {}).get("best_lap")
        k = len(field)
        for i, (name, h) in enumerate(sorted(sides.items())):
            n = int(np.searchsorted(h.t, cut, side="left"))
            if n == 0:
                continue
            w = 0.5 ** ((cut - h.t[:n]) / settings["half_life_days"])
            j = pd.Index(field).get_indexer(h.a[:n])
            f = h.fin[:n] & (j >= 0)
            Wf = np.bincount(j[f], weights=w[f], minlength=k)
            Wz = np.bincount(j[f], weights=(w * np.where(h.fin[:n], h.z[:n], 0.0))[f], minlength=k)
            den = Wf + settings["prior_weight"]
            mu = np.divide(Wz, den, out=np.zeros(k), where=den > 0)
            sd = np.sqrt(settings["noise"] ** 2 + settings["uncertainty"] ** 2 / (Wf + 1.0))
            g = _side_rng(rng, SIDE_SEED + i)
            perf = mu + sd * g.standard_normal((settings["sims"], k))
            order = np.argsort(np.argsort(-perf, axis=1), axis=1) + 1
            if name == best:
                indicators[name] = order == 1
            else:
                stage_rank[name] = order
        return stage_rank, indicators

    def results(self, data, ev):
        """The classification in markets/kinds.settle's shape; unclassified entrants follow the finishers."""
        r = data[data["race"] == ev.id]
        pos = r["position"].to_numpy(float)
        missing = np.isnan(pos)
        pos[missing] = (np.nanmax(pos) if (~missing).any() else 0) + 1 + np.arange(missing.sum())
        out = pd.DataFrame(dict(
            athlete_id=r["athlete_id"].to_numpy(), position=pos, status=r["status"].to_numpy(),
            qual_position=np.nan, team_id=r["team"].to_numpy() if "team" in r else None,
            points=r["points"].fillna(0.0).to_numpy(float) if "points" in r else 0.0))
        best = (self.cfg or {}).get("best_lap")
        for name, f in (self.sides or {}).items():
            side = f[f["race"] == ev.id]
            if side.empty:
                continue
            if name == best:
                # settled only when every lap of the race distance is stored (else a missing lap could be faster)
                if "laps_complete" in side and not side["laps_complete"].fillna(True).astype(bool).all():
                    continue
                first = set(side.loc[side["position"] == side["position"].min(), "athlete_id"])
                out[name] = out["athlete_id"].isin(first)
            else:
                col = P.stage_col(name)
                out[col] = out["athlete_id"].map(side.drop_duplicates("athlete_id").set_index("athlete_id")["position"]
                                                 .astype(float))
        return out
