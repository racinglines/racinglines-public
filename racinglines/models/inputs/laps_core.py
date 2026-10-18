"""The laps core of the one engine (racinglines/models/model_global.py, docs/backtest-core.md): the pricing core of a
sport whose schema declares lap-level data ([model] inputs with "results", "qualifying", "grid" and "laps"; "sectors"
and "practice" optional). It is the F1 engine that was race_model.PositionSim, moved here unchanged (decision log
2026-10-11 "One engine"): one event per race, priced as of one minute before the start (qualifying known) through
laps_pricing.price_race, the as-of view of Measurements being the only data it sees. The season sweep prices every
stage through the same price_race.

Its layers, each a module named by its input:

    laps        inputs/laps.py          race pace and qualifying best laps, finishing model, Monte Carlo
    sectors     inputs/laps.py          the car's fast/slow-sector fit to the track (setting track_features)
    practice    inputs/practice.py      the practice pace prior (setting practice_prior)
    grid        inputs/grid.py          confirmed grid penalties applied to the qualifying order
    qualifying  inputs/laps_pricing.py  the qualifying order as the grid, once qualifying has run

An optional input the schema doesn't declare turns its setting off by default (inputs.CORES gates).
"""

import pandas as pd

from racinglines.models import outcomes as O


class LapsCore:
    sport = None
    name = "lap_sim"                     # model_runs.model; a schema names its own ([model] name, e.g. f1_sector_sim)

    from racinglines.pipelines.sweep_settings import Settings

    @classmethod
    def bind(cls, sport, schema, inputs, defaults):
        """The class attributes of the engine bound to `sport`: Settings with the schema's [model.defaults]."""
        from racinglines.pipelines import sweep_settings as SS
        return dict(Settings=SS.with_defaults(cls.Settings, defaults))

    def load(self, engine_url=None, data=None):
        from racinglines.models.inputs import laps_pricing as run
        if data is None:
            from racinglines.db.config import get_engine
            data = run.Measurements.load(get_engine(engine_url))
        return data

    @staticmethod
    def data_through(data):
        return data.drivers["r_ts"].max()

    def seasons(self, data, settings):
        return sorted(int(y) for y in data.drivers["year"].dropna().unique())

    def events(self, data, settings, seasons=None):
        from racinglines.models.inputs import laps_pricing as run
        from racinglines.models.race_model import Event
        d = data.drivers.drop_duplicates("event_id").sort_values("r_ts")
        if seasons:
            d = d[d["year"].isin(seasons)]
        return [Event(id=r.event_id, season=int(r.year), cutoff=r.r_ts - run.ONE_MIN, name=str(r.venue))
                for r in d.itertuples()]

    def history(self, data, settings):
        from types import SimpleNamespace

        from racinglines.models.inputs import laps_pricing as run
        with settings.applied():
            return SimpleNamespace(meas=data, hist=run.history(data, settings["track_features"]))

    def price(self, hist, ev, settings, rng):
        from racinglines.models.inputs import laps_pricing as run
        with settings.applied():
            _, ex = run.price_race(hist.meas, hist.hist, ev.cutoff, ev.id, n_sims=settings["sims"], rng=rng,
                                   use_track=settings["track_features"])
        return O.from_position_sim(ex["entrants"], ex["sim"])

    def results(self, data, ev):
        from racinglines.models.inputs import practice as PR
        o = data.drivers[data.drivers["event_id"] == ev.id]
        q = data.res[(data.res["event_id"] == ev.id) & (data.res["round"] == "qual")].set_index("athlete_id")["position"]
        out = pd.DataFrame(dict(athlete_id=o["athlete_id"].to_numpy(), position=o["position"].to_numpy(float),
                                status=o["status"].to_numpy(), qual_position=o["athlete_id"].map(q).to_numpy(float),
                                team_id=o["team_key"].to_numpy() if "team_key" in o else None,
                                points=o["points"].to_numpy(float) if "points" in o else 0.0))
        # each practice session's order of best laps (fp1_position, ...: the race_fp<n>_fastest kinds), where stored
        return PR.with_positions(out, PR.classification(data.practice, ev.id))
