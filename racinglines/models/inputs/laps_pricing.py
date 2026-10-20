"""
F1 pricing, backtesting and forecasting, with a hard as-of boundary.

ONE pricing function, used everywhere:

    price_race(meas, hist, cutoff, event_id)   fair prices for one race, using only
                                               data from sessions that STARTED before `cutoff`

and three drivers of it that never mix:

    backtest(...)      historical: cutoff = just before each past race's qualifying
                       ("pre_quali") or just before its race ("pre_race"); scored
                       against outcomes read separately after pricing.
    diagnostic(...)    one past event at a chosen cutoff (e.g. "yesterday"), scored
                       against the outcome and compared with exchange prices *at that cutoff*.
    forecast(...)      live: cutoff = now; prices upcoming races + the championship.

Leakage guards:
  - Measurements.view(cutoff) is the only way pricing code gets data; it filters
    every table by session start time and asserts nothing at/after the cutoff
    remains (LeakageError otherwise). Track profiles (which use race results)
    are only usable after all of that event's sessions have run.
  - Per-race measurements (qualifying gaps, race pace, sector gaps) depend only
    on that race's own session, so computing them once is safe; they are
    time-stamped and filtered by the view.
  - The finishing model is trained only on races whose results were known at
    the cutoff, each row with features computed as of just before that race.
  - The grid is the QUALIFYING order (in the view), not the official grid
    stored with race results, moved by the confirmed grid penalties in
    sports/f1/grid_penalties.toml (penalties.py; a penalty announced after the
    cutoff is left out). Pole is priced from the qualifying order.
  - Exchange (Polymarket) prices never enter fitting or pricing. They are only
    compared with our prices after the fact.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from racinglines.models import calibration as CAL
from racinglines.models import outcomes as O
from racinglines.models.inputs import laps as M
from racinglines.models.inputs import grid as GP
from racinglines.core.stats import brier, ranks

CALIB_SEED = 20261014   # side stream of the calibration resample (models/calibration.py)

ONE_MIN = pd.Timedelta(minutes=1)
PRE_WEEKEND, PRE_QUALI, PRE_RACE = "pre_weekend", "pre_quali", "pre_race"


class LeakageError(AssertionError):
    pass


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


@dataclass
class Measurements:
    res: pd.DataFrame        # results rows (all sessions), with session_ts
    prof: pd.DataFrame       # track profiles, with ready_ts
    drivers: pd.DataFrame    # per race x driver: outcome, q_def, r_def, r_ts, q_ts
    sectors: pd.DataFrame    # per qualifying x team x sector: deficit, trap speed, q_ts
    practice: pd.DataFrame = None   # per practice session x driver: best_def, long_def, session_ts
    cutoff: pd.Timestamp | None = None
    xmap: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    races: pd.DataFrame = None      # per past race: safety car / red flag / rain, disrupted (model.race_disruption)

    @classmethod
    def load(cls, engine):
        return cls.from_frames(*M.load_frames(engine))

    @classmethod
    def from_frames(cls, res, laps, prof):
        """From the raw SQL frames (see inputs.laps.load_frames), e.g. pinned test fixtures."""
        res, laps, prof = M.prepare(res, laps, prof)
        drivers, sectors = M.event_measurements(res, laps, prof)
        return cls(res, prof, drivers, sectors, M.practice_measurements(laps), None, M.event_x(prof),
                   races=M.race_disruption(laps, prof))

    def view(self, cutoff):
        """Everything knowable strictly before `cutoff` (session start times, UTC)."""
        cutoff = _naive(cutoff)
        prof = self.prof[self.prof["ready_ts"] + M.RACE_DONE < cutoff]
        v = Measurements(
            res=self.res[M.session_end(self.res["round"], self.res["session_ts"]) < cutoff],
            prof=prof,
            drivers=self.drivers[self.drivers["r_ts"] + M.RACE_DONE < cutoff],
            sectors=self.sectors[self.sectors["q_ts"] + M.QUAL_DONE < cutoff],
            practice=self.practice[M.session_end(self.practice["round"], self.practice["session_ts"]) < cutoff],
            cutoff=cutoff,
            xmap={k: x for k, x in self.xmap.items() if k in set(prof["event_id"])},
            races=None if self.races is None else self.races[self.races["r_ts"] + M.RACE_DONE < cutoff])
        v.audit = assert_no_leak(v, cutoff)
        return v

    def sessions(self, event_id):
        s = self.res[self.res["event_id"] == event_id].groupby("round")["session_ts"].first()
        return {k: s.get(k) for k in ("fp1", "fp2", "fp3", "sprint_qual", "qual", "sprint", "race")}


def assert_no_leak(v, cutoff):
    """Every session in the view must have ENDED before the cutoff. Reports the latest session starts."""
    ends = {"results": M.session_end(v.res["round"], v.res["session_ts"]).max() if len(v.res) else None,
            "race measurements": v.drivers["r_ts"].max() + M.RACE_DONE if len(v.drivers) else None,
            "qualifying sectors": v.sectors["q_ts"].max() + M.QUAL_DONE if len(v.sectors) else None,
            "track profiles": v.prof["ready_ts"].max() + M.RACE_DONE if len(v.prof) else None,
            "practice": M.session_end(v.practice["round"], v.practice["session_ts"]).max()
            if v.practice is not None and len(v.practice) else None}
    if v.races is not None and len(v.races):
        ends["race disruption"] = v.races["r_ts"].max() + M.RACE_DONE
    for what, ts in ends.items():
        if ts is not None and pd.notna(ts) and ts >= cutoff:
            raise LeakageError(f"{what}: a session ending {ts} is not over before the cutoff {cutoff}")
    latest = {"results": v.res["session_ts"].max(), "race measurements": v.drivers["r_ts"].max(),
              "qualifying sectors": v.sectors["q_ts"].max(), "track profiles": v.prof["ready_ts"].max(),
              "practice": v.practice["session_ts"].max() if v.practice is not None and len(v.practice) else None}
    return {k: (None if t is None or pd.isna(t) else str(t)) for k, t in latest.items()}


# ---------------------------------------------------------------------------
# Training rows for the finishing model
# ---------------------------------------------------------------------------

def entry_list(meas, event_id):
    """Drivers entered for an event (from the qualifying classification, which lists
    every entrant). Only identities and teams are used, never results."""
    q = meas.res[(meas.res["event_id"] == event_id) & (meas.res["round"] == "qual")]
    if q.empty:
        q = meas.res[(meas.res["event_id"] == event_id) & (meas.res["round"] == "race")]
    return q[["athlete_id", "driver", "team_key"]].drop_duplicates("athlete_id").reset_index(drop=True)


def qualifying_grid(view, event_id, entrants):
    """Grid = qualifying classification from the as-of view (None if qualifying hasn't run)."""
    q = view.res[(view.res["event_id"] == event_id) & (view.res["round"] == "qual")]
    if q.empty:
        return None
    pos = q.set_index("athlete_id")["position"].astype(float)
    return entrants["athlete_id"].map(pos).fillna(len(entrants)).to_numpy(float)


def event_key_of(meas, event_id):
    """The event's key YYYY-RR (season, championship round) from its rows; None when it has none."""
    rows = meas.res[meas.res["event_id"] == event_id] if event_id is not None else meas.res.iloc[:0]
    return f"{int(rows['year'].iloc[0])}-{int(rows['series_round'].iloc[0]):02d}" if len(rows) else None


def grid_adjust(key, session, entrants, cutoff):
    """(simulate_race's grid_adjust, the penalties) for a session's confirmed grid penalties known at `cutoff`;
    (None, []) when it has none, so a weekend without one runs exactly the code it ran before."""
    pens = GP.for_event(key, session, cutoff)
    if not pens:
        return None, []
    for p in pens:                             # a typo fails here, not silently in the draws
        GP.driver_index(entrants, p)
    return (lambda g: GP.apply(g, entrants, pens)), pens


def side_stages(meas, event_id, stages=None):
    """The race-like sessions besides the Grand Prix this weekend has (sports/f1.toml [sessions.sim], schema order):
    those named in `stages` when the caller knows the weekend's format (the forecast: its schedule's sprint flag),
    else those the event's rows name (the session or its grid session: which sessions a weekend has is schedule,
    not result, so the full frame is read; their results are read only from the as-of view). A stage's grid session
    counts only in its default form (grid_from): a per-year source such as 2021's qual is on every weekend."""
    side = [k for k in M.SIM_SESSIONS if k != M.MAIN_STAGE]
    if stages is not None:
        return [k for k in side if k in set(stages)]
    if event_id is None:
        return []
    have = set(meas.res.loc[meas.res["event_id"] == event_id, "round"])
    return [k for k in side if k in have or M.SIM_SESSIONS[k]["grid_from"] in have]


def stage_grid(view, event_id, entrants, grid_from):
    """(grid, source) for a side stage from the as-of view: the grid session's laps ranked by each driver's best
    valid lap (Sprint Qualifying is stored without positions: FastF1 gives none), else its classification positions
    if it has any, else (None, "simulated from qualifying pace"). A driver without a lap or a position starts last."""
    p = view.practice
    laps = p[(p["event_id"] == event_id) & (p["round"] == grid_from)] if p is not None and len(p) else None
    if laps is not None and len(laps):
        order = laps.drop_duplicates("athlete_id").set_index("athlete_id")["best_def"].rank(method="first")
        return entrants["athlete_id"].map(order).fillna(len(entrants)).to_numpy(float), f"{grid_from} best laps"
    c = view.res[(view.res["event_id"] == event_id) & (view.res["round"] == grid_from)]
    if len(c) and c["position"].notna().any():
        pos = c.drop_duplicates("athlete_id").set_index("athlete_id")["position"].astype(float)
        return entrants["athlete_id"].map(pos).fillna(len(entrants)).to_numpy(float), f"{grid_from} classification"
    return None, "simulated from qualifying pace"


def stage_result(view, event_id, entrants, stage, n_sims, grid):
    """A side stage's actual result once it has run (in the as-of view), as simulation arrays repeated n_sims times
    (as the Grand Prix uses the actual qualifying order): finishers in classification order, then the rest; None
    before it has run. grid: the stage's grid (else the result rows' own starting grid)."""
    r = view.res[(view.res["event_id"] == event_id) & (view.res["round"] == stage)]
    if r.empty:
        return None
    by = r.drop_duplicates("athlete_id").set_index("athlete_id")
    ids = entrants["athlete_id"]
    ok = ids.map(by["status"]).eq("OK").to_numpy()
    pos = ids.map(by["position"]).astype(float).fillna(np.inf).to_numpy()
    rank = np.empty(len(ids))
    rank[np.lexsort((pos, ~ok))] = np.arange(1, len(ids) + 1)
    pts = ids.map(by["points"]).astype(float).fillna(0.0).to_numpy()
    if grid is None:
        grid = ids.map(by["grid"]).astype(float).replace(0, np.nan).fillna(len(ids)).to_numpy()
    tile = lambda a: np.tile(np.asarray(a), (n_sims, 1))      # noqa: E731
    return dict(pos=tile(rank), dnf=tile(~ok), points=tile(pts), grid=tile(grid))


def price_stages(meas, v, event_id, e, fm, tf, rng, n_sims, chaos_p, stages=None, key=None):
    """The weekend's side stages (the sprint), from the same paces and finishing model as the Grand Prix: the
    schema's points table and retirement scale, the grid from stage_grid, the actual result once the stage has run.
    Every draw is on a side stream seeded from `rng` without advancing it, so the Grand Prix's prices, and whatever
    the caller draws next, are byte-identical with or without these stages. A stage's confirmed grid penalties
    (key: the event's YYYY-RR) move its simulated starting grid; its pole stays the qualifying order. Returns
    ({stage: sim}, {stage: audit}, [applied penalties])."""
    sims, audit, applied = {}, {}, []
    rows = meas.res[meas.res["event_id"] == event_id] if event_id is not None else meas.res.iloc[:0]
    year = int(rows["year"].iloc[0]) if len(rows) else v.cutoff.year
    for i, stage in enumerate(side_stages(meas, event_id, stages)):
        cfg = M.SIM_SESSIONS[stage]
        grid_from = M.grid_source(stage, year)
        grid, src = stage_grid(v, event_id, e, grid_from) if event_id is not None else (None, "simulated")
        s = stage_result(v, event_id, e, stage, n_sims, grid) if event_id is not None else None
        if s is None:
            es = e.assign(p_dnf=np.clip(e["p_dnf"].to_numpy() * float(cfg.get("dnf_scale", 1.0)), 0.0, 1.0))
            if grid is not None:
                es["grid"] = grid
            adjust, pens = grid_adjust(key, stage, es, v.cutoff)
            applied += pens
            s = M.simulate_race(fm, es, tf, n_sims=n_sims, rng=M._side_rng(rng, M.STAGE_SEED + i),
                                grid_known=grid is not None, points=M.points_table(cfg["points"], year), chaos_p=chaos_p,
                                grid_adjust=adjust)
            s.pop("fl", None)
        sims[stage] = dict(pos=s["pos"], dnf=s["dnf"], points=s["points"], grid=s.get("qual", s["grid"]),
                           grid_from=grid_from)
        audit[stage] = dict(grid=src, result="actual" if stage in set(v.res.loc[v.res["event_id"] == event_id, "round"])
                            else "simulated", dnf_scale=float(cfg.get("dnf_scale", 1.0)), points=cfg["points"])
    return sims, audit, applied


def history(meas, use_track=True):
    """One row per entrant per completed race, with features computed as of one
    minute before that race's start (qualifying known) and its outcome."""
    from racinglines.models.inputs import practice as PR
    ptrain = PR.training_rows(meas, use_track) if PR.USE_PRACTICE else pd.DataFrame()
    rows = []
    events = meas.drivers.drop_duplicates("event_id").sort_values("r_ts")
    for ev in events.itertuples():
        cutoff = ev.r_ts - ONE_MIN
        v = meas.view(cutoff)
        snap = M.snapshot(v.drivers, v.sectors, v.xmap if use_track else {}, cutoff, use_track)
        if not snap.team_q:
            continue
        tf = M.venue_track_features(v.prof, ev.venue, cutoff)
        ent = entry_list(meas, ev.event_id)
        e = M.predict_paces(snap, ent, tf)
        e["qp_nopr"] = e["qp"]                    # before the practice prior (for the no-practice grid noise)
        e_nopr = e.copy()
        e, _ = PR.apply(e, v.practice, ev.event_id, PR.fit(ptrain, cutoff))
        grid = qualifying_grid(v, ev.event_id, ent)
        adjust, _ = grid_adjust(f"{int(ev.year)}-{int(ev.series_round):02d}", M.MAIN_STAGE, ent, cutoff)
        if grid is not None and adjust is not None:   # the starting grid the outcome was raced from
            grid = adjust(grid)
        e["grid_used"] = grid if grid is not None else np.nan
        if M.PRE_PRACTICE_TRAIN:                  # the same features from paces before the practice prior
            nopr = M.design(e_nopr.assign(grid_used=e["grid_used"].to_numpy()), tf, use_track)
            e = pd.concat([e, nopr.add_suffix("_nopr")], axis=1)
        outcome_rows = meas.drivers[meas.drivers["event_id"] == ev.event_id].drop(columns=["team_key", "driver"])
        e = e.merge(outcome_rows, on="athlete_id", how="left")
        e["status"] = e["status"].fillna("DNS")
        e[["event_id", "r_ts", "start_date", "year", "venue"]] = e[["event_id", "r_ts", "start_date", "year", "venue"]].ffill().bfill()
        e = pd.concat([e, M.design(e, tf, use_track)], axis=1)
        for k, val in tf.items():
            e[f"tf_{k}"] = val
        e["cutoff"] = cutoff
        if meas.races is not None:
            e["disrupted"] = bool(meas.races.set_index("event_id")["disrupted"].get(ev.event_id, False))
        rows.append(e)
    if not rows:
        raise ValueError("no completed F1 race with qualifying in the database to train on: load them with "
                         "`racinglines data fetch f1` then `racinglines data ingest f1`, or add --demo for the demo store")
    out = pd.concat(rows, ignore_index=True)
    out.attrs["practice_train"] = ptrain
    return out


# ---------------------------------------------------------------------------
# The pricing function
# ---------------------------------------------------------------------------

def price_race(meas, hist, cutoff, event_id, n_sims=10000, rng=None, use_track=True, entrants=None,
               venue=None, points=M.RACE_POINTS, stages=None, event_key=None):
    """Fair prices for one race using only data from before `cutoff`.

    Returns (summary, extras): summary has per-driver win/podium/top10/pole/DNF
    probabilities, expected points and head-to-head probabilities; extras has the
    per-team top-scorer probabilities, the leakage audit, track features and model.
    On a weekend with side stages (the sprint, see price_stages; `stages` names them when the
    event's rows can't, e.g. a future weekend), the summary adds <stage>_win_prob and
    <stage>_pole_prob, extras["sim"]["stages"] their arrays and the audit their sources. The event's
    confirmed grid penalties (sports/f1/grid_penalties.toml, looked up by `event_key` YYYY-RR, else the
    event's rows) move the starting grid of the race and stages they name, recorded in
    extras["audit"]["grid_penalties"]."""
    rng = rng if rng is not None else np.random.default_rng(0)
    v = meas.view(cutoff)
    key = event_key if event_key is not None else event_key_of(meas, event_id)
    if entrants is None:
        entrants = entry_list(meas, event_id)
    if venue is None:
        venue = meas.res.loc[meas.res["event_id"] == event_id, "venue"].iloc[0]
    snap = M.snapshot(v.drivers, v.sectors, v.xmap if use_track else {}, v.cutoff, use_track)
    tf = M.venue_track_features(v.prof, venue, v.cutoff)
    e = M.predict_paces(snap, entrants, tf).reset_index(drop=True)
    from racinglines.models.inputs import practice as PR
    e, sigma_q = PR.apply(e, v.practice, event_id, PR.fit(hist.attrs.get("practice_train"), v.cutoff))
    grid = qualifying_grid(v, event_id, e) if event_id is not None else None
    fm = M.fit_finish(hist, v.cutoff, use_track, no_practice=M.PRE_PRACTICE_TRAIN and sigma_q is None,
                      grid_known=grid is not None)
    if sigma_q is not None:
        from dataclasses import replace as _replace
        fm = _replace(fm, sigma_q=sigma_q)
    if grid is not None:
        e["grid"] = grid
    chaos_p = M.chaos_prob(hist, venue, v.cutoff, fm.chaos["p"]) if fm.chaos else None
    adjust, applied = grid_adjust(key, M.MAIN_STAGE, e, v.cutoff)
    sim = M.simulate_race(fm, e, tf, n_sims=n_sims, rng=rng, grid_known=grid is not None, points=points,
                          chaos_p=chaos_p, grid_adjust=adjust)
    calib = CAL.power(M.CALIB_POWER, M.SPORT, v.cutoff, grid is not None)
    if calib is not None:            # resampled on a side stream, so every other draw is unchanged
        sim = CAL.apply(sim, calib, M._side_rng(rng, CALIB_SEED))
    stage_sims, stage_audit, stage_applied = price_stages(meas, v, event_id, e, fm, tf, rng, n_sims, chaos_p, stages,
                                                          key)
    applied += stage_applied
    summ = M.summarize(e, sim)
    for stage, s in stage_sims.items():        # race_sprint_win / race_sprint_pole (markets/kinds.py, db/reads.py)
        summ[f"{stage}_win_prob"] = summ["athlete_id"].map(dict(zip(e["athlete_id"], (s["pos"] == 1).mean(0))))
        summ[f"{stage}_pole_prob"] = summ["athlete_id"].map(dict(zip(e["athlete_id"], (s["grid"] == 1).mean(0))))
    if stage_sims:
        sim["stages"] = stage_sims
    if M.PRACTICE_FASTEST and event_id is not None:   # practice sessions' best-lap orders, on a side stream
        sim["practice"] = PR.simulate(meas, v, event_id, e, fm, rng, n_sims)
    ids = e["athlete_id"].tolist()
    h2h = (sim["pos"][:, :, None] < sim["pos"][:, None, :]).mean(0)
    summ["h2h"] = summ["athlete_id"].map({a: {str(b): round(float(h2h[i, j]), 4) for j, b in enumerate(ids) if b != a}
                                         for i, a in enumerate(ids)})
    teams = sorted(set(e["team_key"]))
    tpts = np.zeros((n_sims, len(teams)))
    np.add.at(tpts.T, np.array([teams.index(t) for t in e["team_key"]]), sim["points"].T)
    top = np.argmax(tpts + rng.random(tpts.shape) * 1e-3, axis=1)
    sessions = meas.sessions(event_id) if event_id is not None else {}
    extras = dict(constructor_top={t: float((top == i).mean()) for i, t in enumerate(teams)},
                  car_gaps=M.car_gaps(snap, teams, tf),
                  audit=dict(cutoff=str(v.cutoff), latest_data=v.audit,
                             event_sessions={k: (None if ts is None or pd.isna(ts) else str(ts)) for k, ts in sessions.items()},
                             sessions_used=[k for k, ts in sessions.items() if ts is not None and pd.notna(ts) and ts < v.cutoff],
                             grid="qualifying order" if grid is not None else "simulated from qualifying pace",
                             practice_prior=sigma_q is not None,
                             training_races=int(hist.loc[hist["r_ts"] + M.RACE_DONE < v.cutoff, "event_id"].nunique())),
                  track=tf, sim=sim, entrants=e, model=fm)
    if calib is not None:
        extras["audit"]["calib_power"] = calib
    if stage_audit:
        extras["audit"]["stages"] = stage_audit
    if applied:
        extras["audit"]["grid_penalties"] = GP.describe(applied)
    return summ, extras


# ---------------------------------------------------------------------------
# Backtest (historical only)
# ---------------------------------------------------------------------------

def _logloss(p, y, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def grid_baseline(hist, before, max_grid=22, prior=2.0):
    """Empirical P(win / podium / top 10 | qualifying position) from races known before `before`."""
    h = hist[(hist["r_ts"] + M.RACE_DONE < before) & hist["grid_used"].notna()]
    g = h["grid_used"].astype(int).clip(1, max_grid)
    ok = h["status"] == "OK"
    out = {}
    for k, lim in (("win", 1), ("podium", 3), ("top10", 10)):
        y = (ok & (h["position"] <= lim)).astype(float)
        base = y.mean()
        stats = y.groupby(g).agg(["sum", "count"])
        out[k] = ((stats["sum"] + prior * base) / (stats["count"] + prior)).reindex(range(1, max_grid + 1)).fillna(base)
    return out


def outcome(meas, event_id, athlete_ids):
    """Actual result for scoring (read only after pricing)."""
    o = meas.drivers[meas.drivers["event_id"] == event_id].set_index("athlete_id")
    status = pd.Series(athlete_ids).map(o["status"]).fillna("DNS").to_numpy()
    pos = pd.Series(athlete_ids).map(o["position"]).to_numpy(float)
    ok = status == "OK"
    return dict(status=status, position=pos, win=ok & (pos <= 1), podium=ok & (pos <= 3), top10=ok & (pos <= 10))


def backtest(meas, hist, start_year=2021, n_sims=4000, use_track=True, modes=(PRE_WEEKEND, PRE_QUALI, PRE_RACE), seed=0,
             echo=None, last_n=None, keep_probs=False):
    """One row per race x mode with Brier / log-loss scores. keep_probs: also keep each
    driver's win / podium / top-10 probability and outcome (column `pred`), for
    reliability curves (models/evaluate.py)."""
    rng = np.random.default_rng(seed)
    rows = []
    events = meas.drivers[meas.drivers["year"] >= start_year].drop_duplicates("event_id").sort_values("r_ts")
    if last_n:
        events = events.tail(last_n)
    for i, ev in enumerate(events.itertuples(), 1):
        sessions = meas.sessions(ev.event_id)
        for mode in modes:
            if mode == PRE_WEEKEND:     # before the first practice session (events with practice data only)
                pr = [sessions.get(k) for k in ("fp1", "fp2", "fp3", "sprint_qual")]
                pr = [t for t in pr if t is not None and pd.notna(t)]
                anchor = min(pr) if pr else None
            else:
                anchor = sessions.get("qual") if mode == PRE_QUALI else sessions.get("race")
            if anchor is None or pd.isna(anchor):
                continue
            cutoff = anchor - ONE_MIN
            if not (hist["r_ts"] + M.RACE_DONE < cutoff).any():
                continue
            summ, ex = price_race(meas, hist, cutoff, ev.event_id, n_sims=n_sims, rng=rng, use_track=use_track)
            y = outcome(meas, ev.event_id, summ["athlete_id"].tolist())       # scored AFTER pricing
            n = len(summ)
            ok = y["status"] == "OK"
            row = dict(event_id=ev.event_id, year=ev.year, round=ev.series_round, venue=ev.venue, mode=mode,
                       cutoff=str(cutoff), n=n, training_races=ex["audit"]["training_races"],
                       winner=summ.loc[y["win"], "driver"].squeeze() if y["win"].any() else None,
                       winner_prob=float(summ["win_prob"].to_numpy()[y["win"]].sum()),
                       favourite=summ.loc[summ["win_prob"].idxmax(), "driver"],
                       spearman=float(pd.Series(summ["exp_position"].to_numpy()[ok]).rank().corr(
                           pd.Series(y["position"][ok]).rank())))
            row.update(_teammate_scores(meas, ev.event_id, summ, ex))
            if keep_probs:
                row["pred"] = {**{k: summ[f"{k}_prob"].round(5).tolist() for k in ("win", "podium", "top10")},
                               **{f"y_{k}": y[k].astype(int).tolist() for k in ("win", "podium", "top10")}}
            gb = grid_baseline(hist, cutoff) if mode == PRE_RACE else None
            for k, share in (("win", 1), ("podium", 3), ("top10", 10)):
                row[f"brier_{k}"] = brier(summ[f"{k}_prob"], y[k])
                row[f"logloss_{k}"] = _logloss(summ[f"{k}_prob"], y[k])
                row[f"brier_{k}_uniform"] = brier(np.full(n, share / n), y[k])
                if gb is not None:
                    g = pd.Series(summ["athlete_id"]).map(dict(zip(ex["entrants"]["athlete_id"], ex["entrants"]["grid"])))
                    row[f"brier_{k}_grid"] = brier(g.fillna(n).clip(1, 22).astype(int).map(gb[k]).to_numpy(), y[k])
            rows.append(row)
        if echo:
            echo(f"progress {i}/{len(events)} {ev.year} R{ev.series_round:02d} {ev.venue}")
    return pd.DataFrame(rows)


def _teammate_scores(meas, event_id, summ, ex):
    """Brier of teammate head-to-heads (who finishes ahead; a DNF is behind) and of the
    top-scoring constructor, both scored after pricing."""
    o = meas.drivers[meas.drivers["event_id"] == event_id].set_index("athlete_id")
    out = {}
    pairs = []
    for team, g in summ.groupby("team_key"):
        if len(g) != 2:
            continue
        a, b = g["athlete_id"].tolist()
        sa, sb = o["status"].get(a), o["status"].get(b)
        if sa is None or sb is None or (sa != "OK" and sb != "OK"):
            continue
        pa = o["position"].get(a) if sa == "OK" else 99
        pb = o["position"].get(b) if sb == "OK" else 99
        p = (summ.loc[summ["athlete_id"] == a, "h2h"].iloc[0] or {}).get(str(b))
        if p is not None and pa != pb:
            pairs.append((p, float(pa < pb)))
    if pairs:
        p, y = np.array(pairs).T
        out["brier_teammate_h2h"] = float(np.mean((p - y) ** 2))
        out["teammate_pairs"] = len(pairs)
    pts = o.groupby("team_key")["points"].sum() if "points" in o else None
    ct = ex.get("constructor_top") or {}
    if pts is not None and len(pts) and ct and (pts == pts.max()).sum() == 1:
        top = pts.idxmax()
        out["brier_constructor_top"] = float(np.mean([(ct.get(t, 0.0) - float(t == top)) ** 2 for t in ct]))
    return out


def summarize_backtest(bt):
    cols = [c for c in bt.columns if c.startswith(("brier_", "logloss_"))] + ["winner_prob", "spearman"]
    return bt.groupby("mode")[cols].mean()


# ---------------------------------------------------------------------------
# Diagnostic: one past event priced at a chosen cutoff
# ---------------------------------------------------------------------------

def diagnostic(meas, hist, event_key, cutoff, n_sims=10000, seed=42, use_track=True, stages=None):
    """Price one completed event as of `cutoff`, then (separately) read its result."""
    year, rnd = (int(x) for x in event_key.split("-"))
    ev = meas.res[(meas.res["year"] == year) & (meas.res["series_round"] == rnd)]
    if ev.empty:
        raise ValueError(f"no event {event_key}")
    event_id = int(ev["event_id"].iloc[0])
    summ, ex = price_race(meas, hist, cutoff, event_id, n_sims=n_sims, rng=np.random.default_rng(seed),
                          use_track=use_track, stages=stages, event_key=event_key)
    y = outcome(meas, event_id, summ["athlete_id"].tolist())
    result = pd.DataFrame(dict(athlete_id=summ["athlete_id"], status=y["status"], position=y["position"]))
    return event_id, summ, ex, result


# ---------------------------------------------------------------------------
# Live forecast (cutoff = now)
# ---------------------------------------------------------------------------

def upcoming_schedule(year):
    """Season schedule from FastF1 (dates/venues/sprint flags only, no results): sources/fastf1/fetch.event_schedule,
    which reads the saved copy offline (RACINGLINES_OFFLINE=1) or when FastF1 can't be reached. A copy saved before
    it kept the venue and race day falls back to the event name and the last session's day."""
    from racinglines.sources.fastf1.fetch import SPRINT_FORMATS, event_schedule
    s = event_schedule(year)
    last = s[[f"Session{i}DateUtc" for i in range(5, 0, -1) if f"Session{i}DateUtc" in s]].bfill(axis=1).iloc[:, 0]
    date = pd.to_datetime(s["EventDate"]) if "EventDate" in s else pd.to_datetime(last).dt.normalize()
    return pd.DataFrame(dict(round=s["RoundNumber"].astype(int), name=s["EventName"],
                             official=s["OfficialEventName"] if "OfficialEventName" in s else s["EventName"],
                             location=s["Location"] if "Location" in s else s["EventName"], date=date,
                             sprint=s["EventFormat"].astype(str).isin(SPRINT_FORMATS)))


def estimate_drift(hist, cutoff):
    """Season-level pace drift (fraction of lap) from races known before `cutoff`:
    how much a team's / driver's race-pace prediction error persists into the next race."""
    h = hist[(hist["r_ts"] + M.RACE_DONE < cutoff) & hist["r_def"].notna()]
    t = h.groupby(["event_id", "r_ts", "team_key"]).agg(pred=("rp", "mean"), act=("r_def", "mean")).reset_index()
    t["err"] = t["act"] - t["pred"]
    t = t.sort_values(["team_key", "r_ts"])
    t["nxt"] = t.groupby("team_key")["err"].shift(-1)
    t = t.dropna()
    team_sd = float(np.sqrt(max(t["err"].corr(t["nxt"]), 0) * t["err"].var())) if len(t) > 20 else M.TEAM_DRIFT_SD
    d = h.assign(err=h["r_def"] - h["rp"])
    d["drv"] = d["err"] - d.groupby(["event_id", "team_key"])["err"].transform("mean")
    d = d.sort_values(["athlete_id", "r_ts"])
    d["nxt"] = d.groupby("athlete_id")["drv"].shift(-1)
    d = d.dropna(subset=["nxt"])
    drv_sd = float(np.sqrt(max(d["drv"].corr(d["nxt"]), 0) * d["drv"].var())) if len(d) > 20 else M.DRIVER_DRIFT_SD
    return team_sd, drv_sd


def forecast(meas, hist, year, cutoff=None, n_sims=10000, seed=42, schedule=None, use_track=True, field=None,
             race_prices=True):
    """Live pricing: every not-yet-raced round of `year` and the championships, using
    data before `cutoff` (default now, UTC). Per-race prices come from the same
    price_race used in backtests (no drift); season totals add an as-of drift.
    field: the entrants to simulate (default: the last raced event's classification in
    the as-of view); race_prices=False skips per-race prices (championships only)."""
    rng = np.random.default_rng(seed)
    cutoff = _naive(cutoff if cutoff is not None else pd.Timestamp.now(tz="UTC"))
    v = meas.view(cutoff)
    season_res = v.res[v.res["year"] == year]
    raced = set(season_res.loc[season_res["round"] == "race", "series_round"])
    schedule = schedule if schedule is not None else upcoming_schedule(year)
    remaining = schedule[~schedule["round"].isin(raced)].sort_values("round")
    last = v.drivers.sort_values("r_ts")["event_id"].iloc[-1]
    field_ = field if field is not None else entry_list(v, last)
    team_sd, drv_sd = estimate_drift(hist, cutoff)

    pts_now = season_res.groupby("athlete_id")["points"].sum().astype(float)
    wins_now = season_res[(season_res["round"] == "race") & (season_res["status"] == "OK")
                          & (season_res["position"] == 1)].groupby("athlete_id").size()
    team_now = season_res.groupby("team_key")["points"].sum().astype(float)
    ids = list(dict.fromkeys(list(pts_now.index) + list(field_["athlete_id"])))
    col = {a: i for i, a in enumerate(ids)}
    teams = sorted(set(team_now.index) | set(field_["team_key"]))
    tcol = {t: i for i, t in enumerate(teams)}
    total = np.tile(pts_now.reindex(ids).fillna(0).to_numpy(), (n_sims, 1))
    wins = np.tile(wins_now.reindex(ids).fillna(0).to_numpy(float), (n_sims, 1))
    team_total = np.tile(team_now.reindex(teams).fillna(0).to_numpy(), (n_sims, 1))
    drift = M.season_drift(field_, n_sims, rng, team_sd, drv_sd)
    fm = M.fit_finish(hist, cutoff, no_practice=M.PRE_PRACTICE_TRAIN, grid_known=False)   # future races: no practice, no grid
    snap = M.snapshot(v.drivers, v.sectors, v.xmap, cutoff)
    per_event, race_constructor_top, car_gaps = [], {}, {}
    from racinglines.db.ingest import _slugify
    for ev in remaining.itertuples():
        venue = _slugify(ev.location)
        ev_rows = meas.res[(meas.res["year"] == year) & (meas.res["series_round"] == ev.round)]
        event_id = int(ev_rows["event_id"].iloc[0]) if len(ev_rows) else None
        entrants = entry_list(v, event_id) if event_id is not None and len(entry_list(v, event_id)) else field_
        if race_prices:
            # published race prices: the backtested single-race model (no season drift)
            summ, ex = price_race(meas, hist, cutoff, event_id, n_sims=n_sims, rng=rng, entrants=entrants, venue=venue,
                                  use_track=use_track, stages=[k for k in M.SIM_SESSIONS if bool(getattr(ev, k, False))],
                                  event_key=f"{year}-{ev.round:02d}")
            race_constructor_top[f"{year}-{ev.round:02d}"] = ex["constructor_top"]
            car_gaps[f"{year}-{ev.round:02d}"] = ex["car_gaps"]
            tf = ex["track"]
        else:
            summ, tf = None, M.venue_track_features(v.prof, venue, v.cutoff)
        # season totals: same model plus a season-long pace drift shared across the remaining races
        e = M.predict_paces(snap, field_, tf).reset_index(drop=True)
        idx = [col[a] for a in e["athlete_id"]]
        tidx = np.array([tcol[t] for t in e["team_key"]])
        for pts in [M.RACE_POINTS] + ([M.SPRINT_POINTS.get(year, M.SPRINT_POINTS_DEFAULT)] if ev.sprint else []):
            s = M.simulate_race(fm, e, tf, n_sims=n_sims, rng=rng, grid_known=False, points=pts, pace_shock=drift,
                                chaos_p=M.chaos_prob(hist, venue, cutoff, fm.chaos["p"]) if fm.chaos else None)
            total[:, idx] += s["points"]
            if pts is M.RACE_POINTS:
                wins[:, idx] += (s["pos"] == 1) & ~s["dnf"]
            tp = np.zeros((n_sims, len(teams)))
            np.add.at(tp.T, tidx, s["points"].T)
            team_total += tp
        if race_prices:
            per_event.append(dict(round=ev.round, name=ev.name, official=ev.official, location=ev.location, venue=venue,
                                  date=ev.date, sprint=bool(ev.sprint), grid_known=ex["audit"]["grid"] == "qualifying order",
                                  track=tf, summary=summ))
            if records_enabled():         # prediction records: keep the event's simulations for save_forecast
                from racinglines.core import stages as STG
                per_event[-1]["outcome"] = dict(sims=O.from_position_sim(ex["entrants"], ex["sim"]),
                                                stage=STG.spec(M.SPORT)["pre_label"], cutoff=cutoff, archive=True)

    rank = ranks(-(total + rng.random(total.shape) * 1e-3))
    ahead = (rank[:, :, None] < rank[:, None, :]).mean(0)
    names = meas.res.drop_duplicates("athlete_id", keep="last").set_index("athlete_id")["driver"]
    standings = pd.DataFrame(dict(
        athlete_id=ids, driver=[names.get(a, a) for a in ids], current_points=pts_now.reindex(ids).fillna(0).to_numpy(),
        exp_points=total.mean(0), points_p10=np.percentile(total, 10, 0), points_p90=np.percentile(total, 90, 0),
        champion_prob=(rank == 1).mean(0), top3_prob=(rank <= 3).mean(0), top10_prob=(rank <= 10).mean(0),
        exp_rank=rank.mean(0)))
    standings["extra"] = [dict(
        wins_now=int(wins_now.get(a, 0)), exp_wins=round(float(wins[:, i].mean()), 3),
        wins_ge={str(k): round(float((wins[:, i] >= k).mean()), 4) for k in range(1, 11)},
        ahead_of={str(b): round(float(ahead[i, j]), 4) for j, b in enumerate(ids) if b != a})
        for i, a in enumerate(ids)]
    standings = standings.sort_values("exp_points", ascending=False).reset_index(drop=True)
    standings["current_rank"] = standings["current_points"].rank(ascending=False, method="min")
    trank = ranks(-(team_total + rng.random(team_total.shape) * 1e-3))
    team_names = meas.res.sort_values("start_date").drop_duplicates("team_key", keep="last").set_index("team_key")["team"]
    constructors = pd.DataFrame(dict(
        team_key=teams, team=[team_names.get(t, t) for t in teams], current_points=team_now.reindex(teams).fillna(0).to_numpy(),
        exp_points=team_total.mean(0), champion_prob=(trank == 1).mean(0), top3_prob=(trank <= 3).mean(0),
        exp_rank=trank.mean(0))).sort_values("exp_points", ascending=False).reset_index(drop=True)
    extras = dict(constructors=constructors, race_constructor_top=race_constructor_top, car_gaps=car_gaps,
                  cutoff=str(cutoff), latest_data=v.audit, drift=dict(team_sd=team_sd, driver_sd=drv_sd), model=fm)
    return per_event, standings, extras


def save_forecast(engine_url, year, per_event, standings, params, metrics, kind="forecast"):
    """Create scheduled events/races for upcoming rounds (if missing) and store the run."""
    from datetime import date as _date

    from sqlalchemy import select

    from racinglines.db import models as m
    from racinglines.db.config import get_session
    from racinglines.db.ingest import _upsert, resolve_venue
    from racinglines.db.queries import save_model_run

    preds, event_ids = [], {}
    with get_session(engine_url) as s:
        comp = s.scalars(select(m.Competition).filter_by(code=M.COMPETITION)).one()
        cat = s.scalars(select(m.Category).filter_by(competition_id=comp.id, code="DRV")).one()
        season = _upsert(s, m.Season, dict(competition_id=comp.id, year=year))
        for ev in per_event:
            key = f"{year}-{ev['round']:02d}"
            event = s.scalars(select(m.Event).filter_by(season_id=season.id, source="f1timing", source_key=key)).first()
            if event is None:
                venue = resolve_venue(s, ev["location"])
                event = m.Event(season_id=season.id, source="f1timing", source_key=key, name=ev["official"] or ev["name"],
                                start_date=pd.Timestamp(ev["date"]).date(), venue_id=venue.id, series_round=ev["round"],
                                status="scheduled")
                s.add(event)
                s.flush()
            event_ids[key] = event.id
            race = _upsert(s, m.Race, dict(event_id=event.id, category_id=cat.id))
            if race.format is None:
                race.format = dict(kind=M.SPORT, sprint=ev["sprint"], event_name=ev["name"])
            elif ev["name"] and race.format.get("event_name") != ev["name"]:     # repair a stale name
                race.format = dict(race.format, event_name=ev["name"])
            s.commit()
            preds.append(ev["summary"].assign(rider_id=lambda d: "ath:" + d["athlete_id"].astype(str),
                                              target=ev.get("target", f"event:{key}"), race_id=race.id,
                                              make_final_prob=None))
    with get_session(engine_url) as s:
        run_id = save_model_run(
            s, competition=M.COMPETITION, season=year, category="DRV", model=M.RUN_NAME, kind=kind,
            data_through=_date.today(), params=params, metrics=metrics,
            race_predictions=pd.concat(preds, ignore_index=True) if preds else None,
            standings=standings.assign(rider_id=lambda d: "ath:" + d["athlete_id"].astype(str)) if standings is not None else None)
    if records_enabled():
        _write_records(run_id, year, per_event, event_ids)
    return run_id


def records_enabled():
    from racinglines.db import records as REC
    return REC.enabled()


def _write_records(run_id, year, per_event, event_ids):
    """Prediction records beside race_predictions (racinglines/db/records.py): per event that kept its
    simulations. Only forecasts and scenarios archive the sims (D3); a stage (diagnostic) run stays light."""
    from racinglines.db import records as REC
    frames, archive = [], {}
    for ev in per_event:
        oc = ev.get("outcome")
        if not oc:
            continue
        key = f"{year}-{ev['round']:02d}"
        frames.append(oc["sims"].to_records(run_id, M.SPORT, M.RUN_NAME, year, event_ids.get(key), key, oc["stage"],
                                            oc["cutoff"]))
        if oc.get("archive"):
            archive[key] = oc["sims"]
    df = REC.concat(frames)
    if df is not None:
        REC.write(run_id, df, sims=archive or None)


def save_diagnostic(engine_url, event_key, cutoff, summ, ex, sims, **extra_params):
    """Store one as-of pricing of a past event as kind='diagnostic'."""
    year, rnd = (int(x) for x in event_key.split("-"))
    params = dict(event_key=event_key, cutoff=str(cutoff), sims=sims, mode="as-of diagnostic",
                  coef=[float(c) for c in ex["model"].coef], sigma=ex["model"].sigma,
                  half_life_days=M.HALF_LIFE_DAYS, **extra_params)
    metrics = dict(audit=ex["audit"], track={k: float(v) for k, v in ex["track"].items()},
                   race_constructor_top={event_key: ex["constructor_top"]}, car_gaps={event_key: ex["car_gaps"]})
    per_event = [dict(round=rnd, name=None, official=None, location=None, date=None, sprint=False, summary=summ,
                      target=f"asof:{event_key}")]
    if records_enabled():             # stage runs write records but never archive the sims (D3)
        per_event[0]["outcome"] = dict(sims=O.from_position_sim(ex["entrants"], ex["sim"]), archive=False,
                                       stage=extra_params.get("sweep_stage", "asof"), cutoff=cutoff)
    return save_forecast(engine_url, year, per_event, None, params, metrics, kind="diagnostic")
