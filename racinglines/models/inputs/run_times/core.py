"""The run-times core of the one engine (racinglines/models/model_global.py, docs/backtest-core.md): the pricing core
of a sport whose schema declares timed runs ([model] inputs with "run_times"; "practice" optional: the timed training
runs, at the schema's [rounds] run_weights practice weight). It is the downhill engine before the one engine,
moved here unchanged (decision log 2026-10-11 "One engine"): one event per race weekend, priced from every run before
it with its actual start list and format, as walk_forward_season does.
"""

import numpy as np
import pandas as pd

from racinglines.models import outcomes as O


class RunTimesCore:
    sport = None
    name = "timed_runs"                  # model_runs.model
    inputs = ("run_times", "practice")   # the engine sets the schema's; practice runs are left out when undeclared
    competition = None                   # the schema's [competition] code (bind)

    from racinglines.models.inputs.run_times.settings import DHSettings as Settings

    cfg = None                           # the schema's constants (model.cfg_from_schema); None: the import-time ones
    round_map = {}                       # data round names -> canonical (qual, final, ...), [rounds] map
    season_col = None                    # "season": the schema's seasons.year; None: the event date's year

    @classmethod
    def bind(cls, sport, schema, inputs, defaults):
        """The class attributes of the engine bound to `sport`: its competition, its [points] / [rounds] constants
        (cfg), the round-name map, the season column and Settings with [model.defaults] and the sport's categories."""
        from dataclasses import replace

        from racinglines.pipelines import sweep_settings as SS
        from racinglines.models.inputs.run_times import model as TM
        cfg = TM.cfg_from_schema(schema)
        rounds = schema.get("rounds", {})
        cats = tuple(schema["competition"]["categories"])
        spec = [replace(s, choices=cats, default=defaults.get("category", s.default if s.default in cats else cats[0]))
                if s.name == "category" else s for s in cls.Settings.SPEC]
        settings = type(cls.Settings.__name__, (cls.Settings,), dict(SPEC=spec, BY={s.name: s for s in spec}, MODEL=list(cls.Settings.MODEL)))
        return dict(competition=schema["competition"]["code"], Settings=SS.with_defaults(settings, defaults), cfg=cfg,
                    round_map=dict(rounds.get("map", {})), season_col="season" if rounds.get("season") == "schema" else None)

    def _cfg(self):
        from racinglines.models.inputs.run_times import model as TM
        return self.cfg or TM.default_cfg()

    def _rounds(self):
        return [r for r in self._cfg()["run_weights"] if r != "practice" or "practice" in self.inputs]

    def load(self, engine_url=None, competition=None, data=None):
        """The tidy runs frame (database, or `data`: a frame already loaded)."""
        if data is None:
            from racinglines.db.config import get_engine
            from racinglines.db.queries import load_tidy
            data = load_tidy(get_engine(engine_url), competition=competition or self.competition, with_splits=False)
        if self.round_map:
            data = data.assign(round=data["round"].map(lambda r: self.round_map.get(r, r)))
        return data[data["round"].isin(self._rounds())]

    @staticmethod
    def data_through(data):
        return data["event_date"].max()

    def seasons(self, data, settings, min_events=3):
        """Seasons of the category with min_events or more completed events (as `mtb_dh backtest`)."""
        from racinglines.models.inputs.run_times import completed_events, select_target
        rows = data[data["category"] == settings["category"]]
        years = rows["season"].astype(str) if self.season_col == "season" else rows["event_date"].astype(str).str[:4]
        out = []
        for y in sorted(years.unique()):
            t = select_target(data, y, settings["category"], self.season_col)
            if len(completed_events(t, self._cfg())) >= min_events:
                out.append(int(y))
        return out

    def events(self, data, settings, seasons=None):
        from racinglines.core import stages as STG
        from racinglines.models.inputs.run_times import completed_events, event_format, event_order, select_target
        from racinglines.models.inputs.run_times.season import _event_date
        from racinglines.models.race_model import Event
        stage = STG.spec(self.sport)["pre_label"]          # priced once, before the event starts
        cat = settings["category"]
        out = []
        for season in seasons or self.seasons(data, settings):
            target = select_target(data, season, cat, self.season_col)
            target = target[target["event_id"].isin(completed_events(target, self._cfg()))]
            for e in event_order(target):
                fmt = event_format(target, e, self._cfg())
                cutoff = _event_date(target, e)
                out.append(Event(id=e, season=int(season), cutoff=cutoff,
                                 name=str(target.loc[target["event_id"] == e, "venue"].iloc[0]) if "venue" in target else str(e),
                                 info=dict(format=fmt, target=target, stage=stage)))
        return out

    def history(self, data, settings):
        return data

    def price(self, hist, ev, settings, rng):
        """None when the event can't be priced (nothing to learn from yet, or no final results)."""
        from racinglines.models.inputs.run_times import event_starters, fit_season_model, simulate_weekend
        from racinglines.models.inputs.run_times.season import _training_rows
        target, fmt = ev.info["target"], ev.info["format"]
        kw = settings.fit_kw()
        train = _training_rows(hist, target, ev.cutoff, kw.pop("train_scope"))
        if not (train["category"] == settings["category"]).any():
            return None
        if not fmt.get("to_final", fmt.get("q1_to_final")):
            return None
        model = fit_season_model(train, category=settings["category"], cfg=self._cfg(), **kw)
        field = event_starters(target, ev.id, self._cfg())
        with settings.applied():
            sim = simulate_weekend(model, field, n_sims=settings["sims"], rng=rng, fmt=fmt, cfg=self._cfg())
        return O.from_timed_runs(field, sim)

    def results(self, data, ev):
        """Final classification (position = final rank, status OK when ranked in the Final), the points
        qualifier's rank, points and who reached the Final."""
        from racinglines.models.inputs.run_times import actual_event_points
        target = ev.info["target"]
        cfg = self._cfg()
        pts = actual_event_points(target, cfg)
        pts = pts[pts["event_id"] == ev.id]
        fin = target[(target["event_id"] == ev.id) & (target["round"] == "final")]
        if cfg["starters_only"]:
            fin = fin[fin["status"].isin(("OK", "DNF", "DSQ"))]
        made = set(fin["rider_id"])
        ids = sorted(set(pts["rider_id"]) | made)
        p = pts.set_index("rider_id").reindex(ids)
        return pd.DataFrame(dict(athlete_id=ids, position=p["final_rank"].to_numpy(),
                                 status=np.where(p["final_rank"].notna(), "OK", "DNF"),
                                 qual_position=p["qual_rank"].to_numpy(), team_id=None,
                                 points=p["points"].fillna(0.0).to_numpy(), reached_final=[r in made for r in ids]))
