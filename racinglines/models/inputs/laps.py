"""
F1 race model: sector-aware car pace, driver offsets, grid/overtaking, Monte Carlo.

Everything is built in date order, so the features for a race only use data
from before that race (no look-ahead in backtests).

1. Measurements per past event (from the database)
   - qualifying: each driver's best clean lap, and each TEAM's best time in each
     of the 3 sectors, as a % deficit to the fastest in that sector;
   - race pace: each driver's median clean race lap (green flag, accurate timing,
     no pit in/out laps, not lap 1, within 107% of the field median), as a %
     deficit to the fastest driver's median;
   - track profile (track_profiles): sector time shares and speed-trap speeds.

2. Sector-type car model (the track/sector part)
   For each team, fit  sector_deficit = a + b * x  over its recent qualifying
   sectors, where x = (sector trap speed - 250) / 50. Positive b = the car loses
   more in fast sectors; negative b = it's relatively stronger there. Weighted by
   recency (half-life HALF_LIFE_DAYS), ridge-shrunk toward the field. At a
   target track the predicted team deficit is  a + b * x_track, with x_track the
   time-weighted sector speed of that venue (its earlier dry profiles). The same
   is done for race pace against the event-level x.

   The car is shared by both drivers: a team's sector deficit at an event is the
   average of its drivers' deficits (CAR_PACE = "mean"), so the car is not
   credited with the faster driver's pace and then again through that driver's
   offset.

3. Driver offsets vs teammate (qualifying and race pace), shrunk.

4. Finishing model (ridge regression, fit on earlier races): normalized finishing
   position from grid, predicted race pace, predicted qualifying pace, and their
   interactions with the track's overtaking ease (mean places gained at that
   venue) and street-circuit flag.

5. Simulation: grid (actual if qualifying is done, else simulated from qualifying
   pace + noise) -> finishing score + noise -> DNFs (team/driver reliability) ->
   positions and official points (race 25-18-..., sprint 8-7-...). Teammates share
   part of the noise (a car's good or bad weekend): the qualifying and finishing
   noise are split into a team part and a driver part, with the teammate
   correlation measured on earlier races (TEAMMATE_CORR). With FASTEST_LAP (variant
   "fastlap", off by default), also the fastest lap: the quickest classified car on
   race pace + noise, drawn on a side stream so nothing else changes (fastest_lap); with variant "flpos" instead
   from the simulated finishing order and race pace (fastest_lap_from_position, sports/f1/fastest_lap.toml).
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines import sports
from racinglines.db.registry import STREET_CIRCUITS

SPORT = sports.declaring("laps")      # the sport whose [model] inputs declare laps (sports/f1.toml)
SCHEMA = sports.load(SPORT)
COMPETITION = SCHEMA["competition"]["code"]
RUN_NAME = SCHEMA.get("model", {}).get("name", "lap_sim")   # model_runs.model of its runs (f1_sector_sim)
from racinglines.core.stats import ranks
from racinglines.markets.kinds import biggest_mover
from racinglines.models import calibration as CAL

HALF_LIFE_DAYS = 120.0
RIDGE_A = 2.0          # prior weight (in "events") pulling team base pace to the field median
RIDGE_B = 6.0          # prior weight pulling the high-speed slope b to 0
DRIVER_PRIOR_N = 3.0   # driver-vs-teammate offset shrinkage
CAR_PACE = "mean"      # team sector pace from both drivers ("mean") or the team's best sectors ("best", old)
TEAMMATE_CORR = True   # share part of the qualifying/finishing noise between teammates
FINISH_RHO_SCALE = 1.0 # fraction of the measured teammate finishing correlation used in the simulation
CALIB_POWER = CAL.DEFAULT       # calibration power on the race-win prices (models/calibration.py, C48); "off" or None = off
X_CENTER, X_SCALE = 250.0, 50.0

# Challenger switches (docs/f1-roadmap.md, F1-2 / F1-3). The defaults reproduce the
# baseline exactly; models/variants.py names combinations for comparisons.
FINISH_MODEL = "ridge"       # "ridge" | "gbm": gradient-boosted finishing model, monotone in grid and pace gaps
GRID_TERMS = False           # front-of-grid term log(grid)/log(n): True = always, "known" = only once the grid is known
PRE_PRACTICE_TRAIN = False   # before any practice at the event, use a finishing model trained on no-practice paces
CHAOS = False                # race-level mixture: some simulated races are disrupted (more noise, more DNFs)
TEAM_DNF_CORR = False        # teammates' retirements correlated at the measured rate
GRID_NOISE = False           # finishing noise per starting-grid band (GRID_NOISE_BANDS), fitted on earlier races'
                             # residuals, instead of one sigma for every car (variant "gridnoise")
GRID_NOISE_BANDS = (1, 2, 4, 6, 10)   # band upper edges: P1, P2, P3-4, P5-6, P7-10, P11 and back (a priori)
GRID_NOISE_PRIOR = 20.0      # each band's variance shrunk toward the field's with this many weighted rows (a priori)
REG_RESET = False            # in a regulation-reset season, car data from earlier seasons count REG_RESET_WEIGHT
REG_RESET_WEIGHT = 0.25      # set a priori (= two half-lives older), not fitted
ROOKIE_CARRY = None          # driver offsets: once a driver's rookie season is over, that season's teammate
                             # comparisons (the rookie's and the teammate's) count this much; None = off
FASTEST_LAP = False          # draw the race's fastest lap in each simulation (sim["fl"], summary fl_prob)
# Fastest-lap noise, a fraction of a lap like rp (a priori, not fitted): the race-pace surprise (team part at
# least TEAM_DRIFT_SD's 0.3 %, about 0.4 % at a lag-1 persistence of one half) and a best-lap-vs-median term
# of the same size (tyre age, fuel, traffic), added in quadrature: 0.6 % (~0.5 s on a 90 s lap)
FL_SIGMA = 0.006
FL_RHO = 0.3                 # teammates' share of it: half the variance is race pace, whose teammate
                             # surprises correlate +0.63 (docs/f1.md), so 0.63 / 2
FL_SEED = 20261004           # salt of the fastest lap's own random stream (see _side_rng)
FL_FROM = "pace"             # how FASTEST_LAP draws it: "pace" (fastest_lap: race pace + noise) or "position" (variant
                             # "flpos": per finishing position weights from past races, fastest_lap_from_position)
FL_TABLE = sports.SCHEMAS / SPORT / "fastest_lap.toml"
PRACTICE_FASTEST = False     # draw each practice session's order of best laps (sim["practice"], practice.simulate):
                             # the race_fp<n>_fastest kinds; every other price unchanged (variant "practicefast")
# Practice best-lap deficit around the qualifying deficit, a fraction of a lap (a priori, not fitted against a score):
# 0.7 %, the within-session spread of (practice best-lap deficit - qualifying deficit) on the 10 stored fixture
# weekends (tests/fixtures/f1, 538 driver-sessions: FP1 0.73 %, FP2 0.73 %, FP3 0.57 %). Added to the qualifying draw
# (qp + sigma_q noise), independent between teammates (programmes differ)
PRACTICE_SIGMA = 0.007
PRACTICE_SEED = 20261007     # salt of the practice draws' own random stream (+ the session's index)
RESET_YEARS = frozenset(SCHEMA["regulations"]["resets"])       # sports/f1.toml
# a past race counts as disrupted if it had a red flag, >= 10% of laps behind the safety car, or rain (set a priori)
DISRUPTED_SC_SHARE, DISRUPTED_RAIN_SHARE, CHAOS_PRIOR_N = 0.10, 0.25, 4.0
RACE_POINTS = list(SCHEMA["points"]["race"])                       # sports/f1.toml
SPRINT_POINTS = sports.int_keys(SCHEMA["points"]["sprint_by_year"])
SPRINT_POINTS_DEFAULT = list(SCHEMA["points"]["sprint"])
# The race-like sessions simulated per weekend (sports/f1.toml [sessions.sim]): MAIN_STAGE is the Grand Prix, the
# others are side stages (the sprint), each drawn on its own stream (_side_rng, salt STAGE_SEED + its index)
SIM_SESSIONS = {k: dict(v) for k, v in SCHEMA["sessions"]["sim"].items()}
MAIN_STAGE = "race"
STAGE_SEED = 20261006


def points_table(name, year):
    """The [points] table `name` for a season (its <name>_by_year entry when the season has one)."""
    pts = SCHEMA["points"]
    return list(sports.int_keys(pts.get(f"{name}_by_year", {})).get(int(year), pts[name]))


def grid_source(stage, year):
    """The session whose order sets a side stage's grid in a season: the stage's grid_from_by_year entry when the
    season has one (2021: Friday's qualifying set the sprint grid, there was no Sprint Qualifying), else grid_from."""
    cfg = SIM_SESSIONS[stage]
    return sports.int_keys(cfg.get("grid_from_by_year", {})).get(int(year), cfg["grid_from"])

# team lineage across renames, so history carries over
TEAM_ALIASES = {"racing_point": "aston_martin", "renault": "alpine", "toro_rosso": "rb", "alphatauri": "rb",
                "alfa": "sauber", "alfa_romeo": "sauber", "kick_sauber": "sauber", "audi": "sauber"}


def team_key(team_id):
    t = str(team_id or "").lower()
    return TEAM_ALIASES.get(t, t)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

RESULTS_SQL = f"""
SELECT e.id AS event_id, e.start_date, s.year, e.series_round, e.status AS event_status, v.slug AS venue,
       ra.id AS race_id, ro.kind AS round, r.athlete_id, a.display_name AS driver, r.position, r.status,
       r.time_ms, r.team, r.extra, ro.extra->>'session_date' AS session_ts
FROM results r JOIN rounds ro ON ro.id = r.round_id JOIN races ra ON ra.id = ro.race_id
JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
JOIN competitions co ON co.id = s.competition_id LEFT JOIN venues v ON v.id = e.venue_id
JOIN athletes a ON a.id = r.athlete_id
WHERE co.code = '{COMPETITION}'
"""

LAPS_SQL = f"""
SELECT ra.event_id, ro.kind AS round, r.athlete_id, l.lap, l.lap_time_ms, l.s1_ms, l.s2_ms, l.s3_ms,
       l.pit_in, l.pit_out, l.track_status, l.is_accurate, l.deleted, l.stint, ro.extra->>'session_date' AS session_ts
FROM laps l JOIN results r ON r.id = l.result_id JOIN rounds ro ON ro.id = r.round_id
JOIN races ra ON ra.id = ro.race_id JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
JOIN competitions co ON co.id = s.competition_id
WHERE co.code = '{COMPETITION}' AND ro.kind IN ('qual', 'race', 'fp1', 'fp2', 'fp3', 'sprint_qual')
"""

PROFILE_SQL = """
SELECT tp.event_id, e.start_date, v.slug AS venue, tp.features,
       (SELECT max(ro.extra->>'session_date') FROM rounds ro JOIN races ra ON ra.id = ro.race_id
         WHERE ra.event_id = tp.event_id) AS ready_ts   -- profile uses race results: usable once all sessions ran
FROM track_profiles tp JOIN events e ON e.id = tp.event_id LEFT JOIN venues v ON v.id = tp.venue_id
"""


def load_frames(engine):
    """The raw tables the model reads (results, laps, track profiles), straight from SQL."""
    with engine.connect() as c:
        return (pd.read_sql(text(RESULTS_SQL), c), pd.read_sql(text(LAPS_SQL), c), pd.read_sql(text(PROFILE_SQL), c))


def load(engine):
    return prepare(*load_frames(engine))


def prepare(res, laps, prof):
    """Typed, derived columns on the raw frames (team keys, timestamps, profile features)."""
    res, laps, prof = res.copy(), laps.copy(), prof.copy()
    extra = pd.json_normalize(res.pop("extra").fillna({}).tolist())
    for col in ("grid", "points", "team_id", "abbreviation"):
        res[col] = extra[col].to_numpy() if col in extra else None
    res["team_key"] = res["team_id"].map(team_key)
    # a few results lack a team id ("nan" in the source): recover it from the team name
    bad = res["team_key"].isin(["", "nan", "none"])
    if bad.any():
        by_name = res[~bad].groupby("team")["team_key"].agg(lambda k: k.mode().iloc[0])
        res.loc[bad, "team_key"] = res.loc[bad, "team"].map(by_name)
    res["start_date"] = pd.to_datetime(res["start_date"])
    res["session_ts"] = pd.to_datetime(res["session_ts"])
    laps["session_ts"] = pd.to_datetime(laps["session_ts"])
    return res, laps, prepare_profiles(prof)


def prepare_profiles(prof):
    """The track profiles (PROFILE_SQL) with typed timestamps and their features as columns (also read by a
    challenger's track features)."""
    prof = prof.copy()
    prof["start_date"] = pd.to_datetime(prof["start_date"])
    prof["ready_ts"] = pd.to_datetime(prof["ready_ts"])
    return pd.concat([prof.drop(columns="features"), pd.json_normalize(prof["features"].tolist()).drop(
        columns=["venue"], errors="ignore")], axis=1)


# ---------------------------------------------------------------------------
# Per-event measurements
# ---------------------------------------------------------------------------

def _deleted(laps):
    """The laps' `deleted` flag as bools, a missing flag counting as not deleted. The column is object dtype (None, True,
    False); fillna on it no longer downcasts silently in pandas 2.2+, so opt in to that behaviour here: same values,
    no FutureWarning."""
    with pd.option_context("future.no_silent_downcasting", True):
        return laps["deleted"].fillna(False).astype(bool)


def event_measurements(res, laps, prof):
    """One row per (event, driver) with qualifying/race-pace deficits, plus one
    row per (event, team, sector) with sector deficits and trap speeds."""
    q = laps[(laps["round"] == "qual") & laps["lap_time_ms"].notna()].copy()
    q = q[~_deleted(q)]
    best = q.groupby(["event_id", "athlete_id"])["lap_time_ms"].min().rename("q_best").reset_index()
    best["q_def"] = best["q_best"] / best.groupby("event_id")["q_best"].transform("min") - 1
    secs = q.groupby(["event_id", "athlete_id"])[["s1_ms", "s2_ms", "s3_ms"]].min().reset_index()

    r = laps[(laps["round"] == "race") & laps["lap_time_ms"].notna()].copy()
    clean = (r["track_status"] == "1") & r["is_accurate"].fillna(False).astype(bool) & ~r["pit_in"].fillna(False) \
        & ~r["pit_out"].fillna(False) & (r["lap"] > 1)
    r = r[clean]
    r = r[r["lap_time_ms"] <= 1.07 * r.groupby("event_id")["lap_time_ms"].transform("median")]
    rp = r.groupby(["event_id", "athlete_id"])["lap_time_ms"].agg(["median", "count"]).reset_index()
    rp = rp[rp["count"] >= 8]
    rp["r_def"] = rp["median"] / rp.groupby("event_id")["median"].transform("min") - 1

    drivers = res[res["round"] == "race"][["event_id", "start_date", "year", "series_round", "venue", "race_id",
                                           "athlete_id", "driver", "team_key", "position", "status", "grid",
                                           "points", "session_ts"]].rename(columns={"session_ts": "r_ts"}).copy()
    q_ts = res[res["round"] == "qual"].groupby("event_id")["session_ts"].first()
    drivers["q_ts"] = drivers["event_id"].map(q_ts)
    drivers = drivers.merge(best[["event_id", "athlete_id", "q_def"]], on=["event_id", "athlete_id"], how="left")
    drivers = drivers.merge(rp[["event_id", "athlete_id", "r_def"]], on=["event_id", "athlete_id"], how="left")

    # team sector deficits (qualifying) + that event's trap speeds. The car is shared:
    # its deficit is the mean of its drivers' deficits (each vs the fastest driver
    # in that sector), matching driver offsets that are measured vs the team mean.
    sd = secs.merge(res[res["round"] == "qual"][["event_id", "athlete_id", "team_key"]], on=["event_id", "athlete_id"])
    rows = []
    speed_cols = {"s1_ms": "v_i1", "s2_ms": "v_i2", "s3_ms": "v_fl"}
    pmap = prof.set_index("event_id")
    for k, vcol in speed_cols.items():
        dd = sd[["event_id", "team_key", k]].dropna().copy()
        dd["def"] = dd[k] / dd.groupby("event_id")[k].transform("min") - 1
        dd = dd[dd["def"] < 0.08]                                   # aborted / red-flag laps
        t = dd.groupby(["event_id", "team_key"])["def"].agg("mean" if CAR_PACE == "mean" else "min").reset_index()
        if CAR_PACE != "mean":
            t["def"] = t["def"] - t.groupby("event_id")["def"].transform("min")
        t["v"] = t["event_id"].map(pmap[vcol]) if vcol in pmap else np.nan
        t["sector"] = k[:2]
        rows.append(t[["event_id", "team_key", "sector", "def", "v"]])
    sectors = pd.concat(rows, ignore_index=True).dropna()
    sectors = sectors[sectors["def"] < 0.08]  # drop red-flag / aborted-lap outliers
    sectors["q_ts"] = sectors["event_id"].map(q_ts)                 # usable once that qualifying has run
    sectors["start_date"] = sectors["q_ts"]
    return drivers, sectors


def race_disruption(laps, prof):
    """Per past race: safety-car lap share, red flag, rain, and whether it counts as
    disrupted (see DISRUPTED_*). Known once the race has run (r_ts = race start)."""
    r = laps[laps["round"] == "race"]
    if r.empty:
        return pd.DataFrame(columns=["event_id", "r_ts", "sc_share", "red_flag", "rain", "disrupted"])
    st = r["track_status"].fillna("").astype(str)
    g = r.assign(sc=st.str.contains("4"), red=st.str.contains("5")).groupby("event_id")
    out = pd.DataFrame(dict(r_ts=g["session_ts"].first(), sc_share=g["sc"].mean(), red_flag=g["red"].any())).reset_index()
    rain = prof.set_index("event_id")["race_rain_share"] if "race_rain_share" in prof else pd.Series(dtype=float)
    out["rain"] = out["event_id"].map(rain).fillna(0.0).astype(float)
    out["disrupted"] = (out["red_flag"] | (out["sc_share"] >= DISRUPTED_SC_SHARE)
                        | (out["rain"] >= DISRUPTED_RAIN_SHARE))
    return out


def venue_track_features(prof, venue, before):
    """Track features for a venue from its dry profiles before a date (median),
    falling back to the field median if the venue is new."""
    p = prof[(prof["ready_ts"] + RACE_DONE < before)]   # only profiles whose sessions (incl. the race) had all run
    dry = p[(p.get("qual_rain_share", pd.Series(0, index=p.index)).fillna(0) < 0.1)]
    v = dry[dry["venue"] == venue]
    base = dry if len(dry) else p
    def med(col, frame):
        return float(frame[col].median()) if col in frame and frame[col].notna().any() else np.nan
    x_parts = []
    src = v if len(v) else base
    for share, speed in (("s1_share", "v_i1"), ("s2_share", "v_i2"), ("s3_share", "v_fl")):
        x_parts.append(med(share, src) * (med(speed, src) - X_CENTER) / X_SCALE)
    x_track = float(np.nansum(x_parts)) if not all(np.isnan(x_parts)) else 0.0
    mpg = med("mean_places_gained", v) if len(v) else np.nan
    if np.isnan(mpg):
        mpg = med("mean_places_gained", base)
    return dict(x_track=x_track, ease=(mpg if not np.isnan(mpg) else 2.8) / 2.8,
                street=float(venue in STREET_CIRCUITS),
                venue_known=bool(len(v)))


# ---------------------------------------------------------------------------
# Rolling (walk-forward) estimates
# ---------------------------------------------------------------------------

PRACTICE_KINDS = tuple(SCHEMA["sessions"]["practice"])
# the practice sessions classified on their own (race_fp<n>_fastest): every practice session but a side stage's grid
# session (Sprint Qualifying sets the sprint grid, priced as the sprint pole)
PRACTICE_CLASSIFIED = tuple(k for k in PRACTICE_KINDS if k not in {c["grid_from"] for c in SIM_SESSIONS.values()})
# a session's data is usable only once it has ended (a cutoff inside a session sees nothing of it)
SESSION_MINUTES = dict(SCHEMA["sessions"]["minutes"])
QUAL_DONE = pd.Timedelta(minutes=SESSION_MINUTES["qual"])
RACE_DONE = pd.Timedelta(minutes=SESSION_MINUTES["race"])


def session_end(round_kind, start):
    return start + pd.to_timedelta(pd.Series(round_kind).map(SESSION_MINUTES).fillna(150).to_numpy(), unit="m")
LONG_RUN_LAPS = 5


def practice_measurements(laps):
    """Per (event, practice session, driver): one-lap pace (best clean lap, % to the session's
    fastest) and long-run pace (median lap of stints with >= LONG_RUN_LAPS green laps, % to the
    best driver's). Sprint qualifying counts as a practice session here (it's before qualifying)."""
    p = laps[laps["round"].isin(PRACTICE_KINDS) & laps["lap_time_ms"].notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=["event_id", "round", "athlete_id", "session_ts", "best_def", "long_def"])
    p = p[~_deleted(p)]
    p = p[p["lap_time_ms"] <= 1.07 * p.groupby(["event_id", "round"])["lap_time_ms"].transform("min")]
    key = ["event_id", "round", "athlete_id"]
    best = p.groupby(key).agg(best=("lap_time_ms", "min"), session_ts=("session_ts", "first")).reset_index()
    best["best_def"] = best["best"] / best.groupby(["event_id", "round"])["best"].transform("min") - 1
    g = p[(p["track_status"] == "1") & ~p["pit_in"].fillna(False).astype(bool) & ~p["pit_out"].fillna(False).astype(bool)]
    g = g[g.groupby(key + ["stint"])["lap"].transform("count") >= LONG_RUN_LAPS]
    lr = g.groupby(key)["lap_time_ms"].median().rename("long").reset_index()
    lr["long_def"] = lr["long"] / lr.groupby(["event_id", "round"])["long"].transform("min") - 1
    out = best.merge(lr[key + ["long_def"]], on=key, how="left")
    return out[["event_id", "round", "athlete_id", "session_ts", "best_def", "long_def"]]


def _weights(dates, now):
    age = (now - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    return 0.5 ** (age / HALF_LIFE_DAYS)


def _car_weights(dates, now):
    """Recency weights for car (team) pace; with REG_RESET, earlier seasons' data are
    discounted once `now` is in a season with new technical regulations."""
    w = _weights(dates, now)
    if REG_RESET and pd.Timestamp(now).year in RESET_YEARS:
        w = np.where(pd.to_datetime(dates).dt.year.to_numpy() < pd.Timestamp(now).year, w * REG_RESET_WEIGHT, w)
    return w


def fit_team_sector(sectors, now, use_track=True):
    """{team: (a, b)} from qualifying sector deficits before `now`."""
    s = sectors[sectors["q_ts"] + QUAL_DONE < now]
    if s.empty:
        return {}, 0.02
    w = _car_weights(s["start_date"], now)
    s = s.assign(w=w, x=(s["v"] - X_CENTER) / X_SCALE)
    a0 = float(np.average(s["def"], weights=s["w"]))
    out = {}
    for team, g in s.groupby("team_key"):
        W, x, y = g["w"].to_numpy(), g["x"].to_numpy(), g["def"].to_numpy()
        if not use_track:
            a = (np.sum(W * y) + RIDGE_A * a0) / (np.sum(W) + RIDGE_A)
            out[team] = (a, 0.0)
            continue
        # weighted ridge on [a, b] with priors a0, 0 (2x2 normal equations)
        A = np.array([[np.sum(W) + RIDGE_A, np.sum(W * x)], [np.sum(W * x), np.sum(W * x * x) + RIDGE_B]])
        rhs = np.array([np.sum(W * y) + RIDGE_A * a0, np.sum(W * x * y)])
        try:
            a, b = np.linalg.solve(A, rhs)
            if np.isnan(a) or np.isnan(b):
                a = (np.sum(W * y) + RIDGE_A * a0) / (np.sum(W) + RIDGE_A)
                b = 0.0
        except np.linalg.LinAlgError:
            a = (np.sum(W * y) + RIDGE_A * a0) / (np.sum(W) + RIDGE_A)
            b = 0.0
        out[team] = (float(a), float(b))
    return out, a0


def fit_team_race(drivers, xmap, now, use_track=True):
    """{team: (c, d)}: race-pace deficit vs the event's track x (event-level)."""
    d = drivers[(drivers["r_ts"] + RACE_DONE < now) & drivers["r_def"].notna()]
    if d.empty:
        return {}, 0.01
    t = d.groupby(["event_id", "team_key", "start_date"])["r_def"].mean().reset_index()
    t["x"] = t["event_id"].map(xmap).fillna(0.0)
    t["w"] = _car_weights(t["start_date"], now)
    c0 = float(np.average(t["r_def"], weights=t["w"]))
    out = {}
    for team, g in t.groupby("team_key"):
        W, x, y = g["w"].to_numpy(), g["x"].to_numpy(), g["r_def"].to_numpy()
        if not use_track:
            out[team] = ((np.sum(W * y) + RIDGE_A * c0) / (np.sum(W) + RIDGE_A), 0.0)
            continue
        A = np.array([[np.sum(W) + RIDGE_A, np.sum(W * x)], [np.sum(W * x), np.sum(W * x * x) + RIDGE_B]])
        rhs = np.array([np.sum(W * y) + RIDGE_A * c0, np.sum(W * x * y)])
        try:
            c, dd = np.linalg.solve(A, rhs)
            if np.isnan(c) or np.isnan(dd):
                c = (np.sum(W * y) + RIDGE_A * c0) / (np.sum(W) + RIDGE_A)
                dd = 0.0
        except np.linalg.LinAlgError:
            c = (np.sum(W * y) + RIDGE_A * c0) / (np.sum(W) + RIDGE_A)
            dd = 0.0
        out[team] = (float(c), float(dd))
    return out, c0


def driver_offsets(drivers, col, now):
    """Driver deficit minus teammates' mean, recency-weighted and shrunk to 0."""
    ts, done = ("q_ts", QUAL_DONE) if col == "q_def" else ("r_ts", RACE_DONE)   # usable once that session has ended
    d = drivers[(drivers[ts] + done < now) & drivers[col].notna()].copy()
    if d.empty:
        return {}
    d["off"] = d[col] - d.groupby(["event_id", "team_key"])[col].transform("mean")
    d["w"] = _weights(d["start_date"], now)
    if ROOKIE_CARRY is not None:
        d["w"] *= _rookie_carry(d, now)
    g = d.assign(wo=d["w"] * d["off"]).groupby("athlete_id")[["wo", "w"]].sum()
    return (g["wo"] / (g["w"] + DRIVER_PRIOR_N)).to_dict()


def _rookie_carry(d, now):
    """ROOKIE_CARRY for rows from a team-event with a driver in their finished rookie season, else 1.
    A rookie season is a driver's first season in the as-of data, unless that is the data's first
    season (a driver already racing then isn't known to be a rookie)."""
    first = d.groupby("athlete_id")["year"].transform("min")
    rookie = (d["year"] == first) & (first > d["year"].min()) & (d["year"] < now.year)
    team_event = rookie.groupby([d["event_id"], d["team_key"]]).transform("any")
    return np.where(team_event, ROOKIE_CARRY, 1.0)


def dnf_rates(drivers, now, prior_n=10.0):
    d = drivers[drivers["r_ts"] + RACE_DONE < now]
    if d.empty:
        return {}, {}, 0.1
    d = d.assign(dnf=d["status"].isin(["DNF", "DSQ"]).astype(float), w=_weights(d["start_date"], now))
    p0 = float(np.average(d["dnf"], weights=d["w"]))
    tg = d.assign(wd=d["w"] * d["dnf"]).groupby("team_key")[["wd", "w"]].sum()
    team = ((tg["wd"] + prior_n * p0) / (tg["w"] + prior_n)).to_dict()
    return team, {}, p0


@dataclass
class Snapshot:
    """Everything known just before a date."""
    team_q: dict
    q0: float
    team_r: dict
    r0: float
    drv_q: dict
    drv_r: dict
    dnf_team: dict
    dnf0: float
    use_track: bool = True


def snapshot(drivers, sectors, xmap, now, use_track=True):
    team_q, q0 = fit_team_sector(sectors, now, use_track)
    team_r, r0 = fit_team_race(drivers, xmap, now, use_track)
    dnf_team, _, dnf0 = dnf_rates(drivers, now)
    return Snapshot(team_q, q0, team_r, r0, driver_offsets(drivers, "q_def", now),
                    driver_offsets(drivers, "r_def", now), dnf_team, dnf0, use_track)


def predict_paces(snap, entrants, tf):
    """entrants: DataFrame athlete_id, team_key. Adds qp (qualifying deficit),
    rp (race-pace deficit), p_dnf. Unknown teams get a slow default."""
    x = tf["x_track"] if snap.use_track else 0.0
    slow_q = max(v[0] for v in snap.team_q.values()) if snap.team_q else snap.q0
    slow_r = max(v[0] for v in snap.team_r.values()) if snap.team_r else snap.r0
    e = entrants.copy()
    e["qp"] = [snap.team_q.get(t, (slow_q, 0.0))[0] + snap.team_q.get(t, (slow_q, 0.0))[1] * x
               + snap.drv_q.get(a, 0.0) for a, t in zip(e["athlete_id"], e["team_key"])]
    e["rp"] = [snap.team_r.get(t, (slow_r, 0.0))[0] + snap.team_r.get(t, (slow_r, 0.0))[1] * x
               + snap.drv_r.get(a, 0.0) for a, t in zip(e["athlete_id"], e["team_key"])]
    e["p_dnf"] = [snap.dnf_team.get(t, snap.dnf0) for t in e["team_key"]]
    return e


def car_gaps(snap, teams, tf):
    """{team: {"race_pct", "qual_pct"}}: each car's predicted race-pace and qualifying deficit at this track,
    as a % gap to the fastest of `teams` (0.0 = fastest). The car term only (a + b * x_track), no driver
    offsets, so it is the car without its drivers' teammate-relative pace; unknown teams get the slow default."""
    x = tf["x_track"] if snap.use_track else 0.0
    out = {}
    for name, table, base in (("race_pct", snap.team_r, snap.r0), ("qual_pct", snap.team_q, snap.q0)):
        slow = max(v[0] for v in table.values()) if table else base
        d = {t: table.get(t, (slow, 0.0))[0] + table.get(t, (slow, 0.0))[1] * x for t in teams}
        best = min(d.values()) if d else 0.0
        for t, v in d.items():
            out.setdefault(t, {})[name] = round(100 * (v - best), 3)
    return dict(sorted(out.items(), key=lambda kv: kv[1]["race_pct"]))


# ---------------------------------------------------------------------------
# Walk-forward feature table + finishing model
# ---------------------------------------------------------------------------

FEATURES = ["g", "g_ease", "rp_rel", "rp_ease", "qp_rel", "street_g"]
MONOTONE = {"g": 1, "g_log": 1, "rp_rel": 1, "qp_rel": 1}   # GBM: a worse grid or pace never improves the finish


def features(grid_known=None):
    """The finishing model's inputs under the current switches (grid_known: whether the
    race being priced has its real grid; matters for GRID_TERMS = "known")."""
    use = GRID_TERMS is True or (GRID_TERMS == "known" and grid_known is not False)
    return FEATURES + (["g_log"] if use else [])


def design(df, tf, use_track=True):
    n = len(df)
    g = df["grid_used"].fillna(n).clip(lower=1)
    g = g.where(g > 0, n)
    out = pd.DataFrame(index=df.index)
    out["g"] = (g - 1) / max(n - 1, 1)
    if GRID_TERMS:
        out["g_log"] = np.log(g.clip(upper=n)) / np.log(max(n, 2))
    out["rp_rel"] = (df["rp"] - df["rp"].min()) * 100
    out["qp_rel"] = (df["qp"] - df["qp"].min()) * 100
    ease = tf["ease"] if use_track else 1.0
    street = tf["street"] if use_track else 0.0
    out["g_ease"] = out["g"] * (ease - 1)
    out["rp_ease"] = out["rp_rel"] * (ease - 1)
    out["street_g"] = out["g"] * street
    return out[features()]


def event_x(prof):
    """Track x for each past event from its own profile (known once it has run)."""
    x = sum(prof[share].fillna(1 / 3) * (prof[speed] - X_CENTER) / X_SCALE
            for share, speed in (("s1_share", "v_i1"), ("s2_share", "v_i2"), ("s3_share", "v_fl"))
            if share in prof and speed in prof)
    return dict(zip(prof["event_id"], pd.Series(x).fillna(0.0)))


@dataclass
class FinishModel:
    coef: np.ndarray
    intercept: float
    sigma: float
    sigma_q: float
    use_track: bool = True
    rho_q: float = 0.0     # teammate correlation of qualifying noise
    rho_f: float = 0.0     # teammate correlation of finishing noise
    features: tuple = tuple(FEATURES)
    gbm: object = None     # FINISH_MODEL = "gbm": the fitted regressor (replaces coef/intercept)
    chaos: dict | None = None   # CHAOS: dict(p, sigma_calm, sigma_chaos, dnf_calm, dnf_chaos)
    rho_dnf: float = 0.0   # TEAM_DNF_CORR: latent correlation of teammates' retirements
    noise_mult: tuple | None = None   # GRID_NOISE: noise multiplier per grid band (GRID_NOISE_BANDS), field = 1


def grid_band(grid):
    """Band index (0 = pole) of each starting position under GRID_NOISE_BANDS."""
    return np.searchsorted(np.asarray(GRID_NOISE_BANDS, float), np.asarray(grid, float), side="left")


def _noise_mult(fin, resid, w, sigma):
    """GRID_NOISE: each grid band's residual sd over the field's (sigma), its variance shrunk toward sigma**2 by
    GRID_NOISE_PRIOR weighted rows. fin: the classified finishers the model was fitted on (grid_used), resid and w
    aligned to it. Only earlier races' rows reach here (fit_finish's cutoff), so the multipliers are walk-forward."""
    if not GRID_NOISE or not sigma > 0:
        return None
    grid = fin["grid_used"].to_numpy(float)
    n = fin.groupby("event_id")["athlete_id"].transform("count").to_numpy(float)
    grid = np.where(np.isnan(grid) | (grid <= 0), n, grid)
    band = grid_band(grid)
    r2, wt = np.asarray(resid, float) ** 2, np.asarray(w, float)
    out = []
    for b in range(len(GRID_NOISE_BANDS) + 1):
        m = band == b
        var = (np.sum(wt[m] * r2[m]) + GRID_NOISE_PRIOR * sigma ** 2) / (np.sum(wt[m]) + GRID_NOISE_PRIOR)
        out.append(float(np.sqrt(var) / sigma))
    return tuple(out)


def fit_finish(hist, before, use_track=True, alpha=1.0, no_practice=False, grid_known=None):
    """Ridge (or, with FINISH_MODEL = "gbm", gradient boosting) on earlier races'
    classified finishers: normalized finish ~ features(). no_practice: train on the
    features computed from paces before the practice prior (PRE_PRACTICE_TRAIN)."""
    from sklearn.linear_model import Ridge
    h = hist[(hist["r_ts"] + RACE_DONE < before)]   # only races whose results were known at the cutoff
    fin = h[h["status"] == "OK"].copy()
    n = fin.groupby("event_id")["athlete_id"].transform("count")
    fin["y"] = (fin["position"] - 1) / (n - 1).clip(lower=1)
    w = _weights(fin["start_date"], before) + 0.05
    feats = features(grid_known)
    X = fin[[f"{f}_nopr" for f in feats]].set_axis(feats, axis=1) if no_practice else fin[feats]
    X = X.fillna(0.0)
    if not use_track:
        X = X.assign(g_ease=0.0, rp_ease=0.0, street_g=0.0)
    gbm = None
    if FINISH_MODEL == "gbm":
        gbm, resid = _fit_gbm(fin, X, w, feats)
        m = None
    else:
        m = Ridge(alpha=alpha).fit(X, fin["y"], sample_weight=w)
        resid = fin["y"] - m.predict(X)
    q = h[h["q_def"].notna()]
    qcol = "qp_nopr" if "qp_nopr" in q else "qp"      # grid noise without practice (the prior sets its own)
    sigma_q = float(np.sqrt(np.average((q["q_def"] - q[qcol]) ** 2, weights=_weights(q["start_date"], before) + 0.05)))
    rho_q = rho_f = 0.0
    if TEAMMATE_CORR:
        rho_q = teammate_corr(q.assign(r=q["q_def"] - q[qcol]), "r")
        rho_f = teammate_corr(fin.assign(r=resid), "r") * FINISH_RHO_SCALE
    ws = pd.Series(w, index=fin.index)
    if gbm is not None:
        rw = ws.loc[resid.index]
        sigma = float(np.sqrt(np.average(resid ** 2, weights=rw)))
        return FinishModel(np.zeros(len(feats)), 0.0, sigma, sigma_q, use_track, rho_q, rho_f, tuple(feats), gbm,
                           _chaos(h, fin, resid, ws), _dnf_corr(h),
                           _noise_mult(fin.loc[resid.index], resid, rw, sigma))
    sigma = float(np.sqrt(np.average(resid ** 2, weights=w)))
    if feats == FEATURES and not (CHAOS or TEAM_DNF_CORR or GRID_NOISE):
        return FinishModel(m.coef_, float(m.intercept_), sigma, sigma_q, use_track, rho_q, rho_f)
    return FinishModel(m.coef_, float(m.intercept_), sigma, sigma_q, use_track, rho_q, rho_f, tuple(feats), None,
                       _chaos(h, fin, resid, ws), _dnf_corr(h), _noise_mult(fin, resid, w, sigma))


def _fit_gbm(fin, X, w, feats, holdout=0.2):
    """Gradient boosting with fixed, a-priori settings (no tuning). The noise sigma
    comes from out-of-time residuals: fit on the older races, score the most recent
    `holdout` share, then refit on everything."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    def model():
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                             l2_regularization=1.0, monotonic_cst=[MONOTONE.get(f, 0) for f in feats],
                                             random_state=0)
    X = X.to_numpy(float)                     # simulate_race predicts on plain arrays
    order = fin.groupby("event_id")["r_ts"].first().sort_values()
    recent = set(order.index[int(len(order) * (1 - holdout)):])
    late = fin["event_id"].isin(recent).to_numpy()
    if late.sum() >= 100 and (~late).sum() >= 200:
        early = model().fit(X[~late], fin["y"][~late], sample_weight=w[~late])
        resid = fin["y"][late] - early.predict(X[late])
    else:
        resid = None
    final = model().fit(X, fin["y"], sample_weight=w)
    if resid is None:
        resid = fin["y"] - final.predict(X)
    return final, resid


def _chaos(h, fin, resid, w):
    """CHAOS: noise and DNF rates in disrupted vs calm past races (as of the cutoff)."""
    if not CHAOS or "disrupted" not in h:
        return None
    dis = fin.loc[resid.index, "disrupted"].fillna(False).astype(bool)
    rw = w.loc[resid.index]
    if dis.sum() < 50 or (~dis).sum() < 50:
        return None
    sd = lambda m: float(np.sqrt(np.average(resid[m] ** 2, weights=rw[m])))
    ev = h.drop_duplicates("event_id")
    dnf = h["status"].isin(["DNF", "DSQ"])
    hd = h["disrupted"].fillna(False).astype(bool)
    base = float(dnf.mean())
    return dict(p=float(ev["disrupted"].fillna(False).mean()), sigma_calm=sd(~dis), sigma_chaos=sd(dis),
                dnf_calm=float(dnf[~hd].mean()) / base if base else 1.0,
                dnf_chaos=float(dnf[hd].mean()) / base if base else 1.0)


def chaos_prob(hist, venue, before, field):
    """Share of disrupted races at a venue before a date, shrunk to the field rate."""
    ev = hist[(hist["r_ts"] + RACE_DONE < before) & (hist["venue"] == venue)].drop_duplicates("event_id")
    k = float(ev["disrupted"].fillna(False).sum()) if "disrupted" in ev else 0.0
    return (k + CHAOS_PRIOR_N * field) / (len(ev) + CHAOS_PRIOR_N)


def _dnf_corr(h):
    """TEAM_DNF_CORR: latent correlation that reproduces the measured teammate DNF correlation."""
    if not TEAM_DNF_CORR:
        return 0.0
    d = h[h["status"] != "DNS"].assign(dnf=h["status"].isin(["DNF", "DSQ"]).astype(float))
    phi = teammate_corr(d, "dnf")
    p = float(d["dnf"].mean())
    if phi <= 0 or not 0 < p < 1:
        return 0.0
    from scipy.stats import multivariate_normal, norm
    z = norm.ppf(p)
    target = phi * p * (1 - p) + p * p               # P(both retire)
    lo, hi = 0.0, 0.95
    for _ in range(25):
        mid = (lo + hi) / 2
        both = multivariate_normal.cdf([z, z], cov=[[1, mid], [mid, 1]])
        lo, hi = (mid, hi) if both < target else (lo, mid)
    return float((lo + hi) / 2)


def teammate_corr(df, col, lo=0.0, hi=0.9):
    """Correlation of teammates' residuals in the same event (both cars present)."""
    d = df[df[col].notna()]
    d = d[d.groupby(["event_id", "team_key"])[col].transform("count") == 2].sort_values(["event_id", "team_key", "athlete_id"])
    if len(d) < 40:
        return 0.0
    a = d.groupby(["event_id", "team_key"])[col].agg(["first", "last"])
    r = a["first"].corr(a["last"])
    return float(np.clip(r, lo, hi)) if pd.notna(r) else 0.0


def _noise(rng, sigma, rho, teams, n_sims):
    """(n_sims, n) normal noise with sd `sigma`, correlation `rho` between drivers of the same team."""
    n = len(teams)
    if rho <= 0:
        return rng.normal(0, sigma, (n_sims, n))
    uniq = {t: i for i, t in enumerate(dict.fromkeys(teams))}
    shared = rng.normal(0, 1, (n_sims, len(uniq)))[:, [uniq[t] for t in teams]]
    return sigma * (np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, (n_sims, n)))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

# Season-level pace drift (fraction of lap time), estimated from how much a team's
# race-pace prediction error persists into its next race (lag-1 correlation x variance):
# team ~0.30 %, driver-vs-teammate ~0.06 % (2020-21 + 2025-26 data). Applied as one
# random shift per simulated season, shared by all remaining races.
TEAM_DRIFT_SD = 0.0030
DRIVER_DRIFT_SD = 0.0006


def season_drift(e, n_sims, rng, team_sd=TEAM_DRIFT_SD, driver_sd=DRIVER_DRIFT_SD):
    """(n_sims, n) pace shift per simulated season (same order as e)."""
    teams = e["team_key"].astype(str).to_numpy()
    uniq = {t: i for i, t in enumerate(dict.fromkeys(teams))}
    team_shock = rng.normal(0, team_sd, (n_sims, len(uniq)))[:, [uniq[t] for t in teams]]
    return team_shock + rng.normal(0, driver_sd, (n_sims, len(teams)))


def simulate_race(fm, e, tf, n_sims=10000, rng=None, grid_known=False, points=RACE_POINTS, pace_shock=None,
                  chaos_p=None, grid_adjust=None):
    """e: entrants with qp, rp, p_dnf (and grid if grid_known). pace_shock: optional
    (n_sims, n) shift of both qualifying and race pace (see season_drift). chaos_p:
    probability this race is disrupted (fm.chaos). grid_adjust: optional callable taking the
    (n_sims, n) qualifying order (known or drawn) and returning the starting grid (grid
    penalties, penalties.apply); it runs after the grid's draws and draws nothing itself.
    Returns dict of (n_sims, n) arrays: pos (finishing position, DNFs last), dnf, points,
    grid (the starting grid); with grid_adjust also qual (the qualifying order, for pole);
    with FASTEST_LAP also fl (who set the race's fastest lap, see fastest_lap)."""
    rng = rng or np.random.default_rng(0)
    n = len(e)
    shock = pace_shock if pace_shock is not None else 0.0
    qp = e["qp"].to_numpy()[None, :] + shock
    rp = e["rp"].to_numpy()[None, :] + shock
    teams = e["team_key"].astype(str).to_numpy()
    if grid_known:
        grid = np.tile(e["grid"].fillna(n).replace(0, n).to_numpy(float), (n_sims, 1))
    else:
        q = qp + _noise(rng, fm.sigma_q, fm.rho_q, teams, n_sims)
        grid = ranks(q).astype(float)
    qual = grid
    if grid_adjust is not None:
        grid = np.asarray(grid_adjust(grid), float)
    g = (grid - 1) / max(n - 1, 1)
    rp_rel = (rp - rp.min(axis=1, keepdims=True)) * 100
    qp_rel = (qp - qp.min(axis=1, keepdims=True)) * 100
    ease = (tf["ease"] - 1) if fm.use_track else 0.0
    street = tf["street"] if fm.use_track else 0.0
    feats = {"g": g, "g_ease": g * ease, "rp_rel": rp_rel, "rp_ease": rp_rel * ease, "qp_rel": qp_rel,
             "street_g": g * street}
    if "g_log" in fm.features:
        feats["g_log"] = np.log(np.clip(grid, 1, n)) / np.log(max(n, 2))
    if fm.gbm is not None:
        X = np.stack([np.broadcast_to(feats[f], (n_sims, n)).ravel() for f in fm.features], axis=1)
        score = fm.gbm.predict(X).reshape(n_sims, n)
    else:
        score = fm.intercept + sum(c * feats[f] for c, f in zip(fm.coef, fm.features))
    p_dnf = e["p_dnf"].to_numpy()[None, :]
    # GRID_NOISE: each car's noise scaled by its starting band's multiplier (from the grid it starts from in this
    # simulation); 1.0 otherwise, so the draws and every price are unchanged with the switch off
    mult = np.asarray(fm.noise_mult, float)[grid_band(grid)] if fm.noise_mult else 1.0
    if fm.chaos and chaos_p is not None:
        wild = rng.random((n_sims, 1)) < chaos_p
        sig = np.where(wild, fm.chaos["sigma_chaos"], fm.chaos["sigma_calm"])
        score = score + sig * mult * _noise(rng, 1.0, fm.rho_f, teams, n_sims)
        p_dnf = np.clip(p_dnf * np.where(wild, fm.chaos["dnf_chaos"], fm.chaos["dnf_calm"]), 0, 0.9)
    else:
        score = score + mult * _noise(rng, fm.sigma, fm.rho_f, teams, n_sims)
    if fm.rho_dnf > 0:
        from scipy.stats import norm
        dnf = _noise(rng, 1.0, fm.rho_dnf, teams, n_sims) < norm.ppf(np.clip(p_dnf, 1e-6, 1 - 1e-6))
    else:
        dnf = rng.random((n_sims, n)) < p_dnf
    score = np.where(dnf, 10 + rng.random((n_sims, n)), score)
    pos = ranks(score)
    pts = np.zeros((n_sims, n))
    for p, v in enumerate(points, start=1):
        pts += (pos == p) * v
    pts = np.where(dnf, 0, pts)
    out = dict(pos=pos, dnf=dnf, points=pts, grid=grid)
    if grid_adjust is not None:
        out["qual"] = qual
    if FASTEST_LAP:
        out["fl"] = (fastest_lap_from_position(pos, dnf, rp, rng) if FL_FROM == "position"
                     else fastest_lap(rp, dnf, teams, rng))
    return out


def _side_rng(rng, salt):
    """A generator for draws that must not move `rng`: seeded from rng's current state, which is read, not
    advanced, so every output drawn from `rng` (and whatever the caller draws from it next) is the same with or
    without these draws, and races priced off one shared stream still get different side draws."""
    import hashlib
    import json
    state = json.dumps(rng.bit_generator.state, sort_keys=True, default=str)
    return np.random.default_rng([salt, int(hashlib.sha256(state.encode()).hexdigest()[:16], 16)])


def fastest_lap(rp, dnf, teams, rng):
    """(n_sims, n) bool: who sets the race's fastest lap in each simulation. Among the classified cars, the
    quickest of race pace rp (with the simulation's pace shock) plus FL_SIGMA noise, FL_RHO of it shared by
    teammates (with TEAMMATE_CORR). A retired car never gets it here, though one can in a real race. Drawn on a
    side stream (_side_rng), so the race's other outputs are byte-identical with FASTEST_LAP on or off."""
    n_sims = dnf.shape[0]
    side = _side_rng(rng, FL_SEED)
    lap = rp + _noise(side, FL_SIGMA, FL_RHO if TEAMMATE_CORR else 0.0, teams, n_sims)
    lap = np.where(dnf, np.inf, lap)
    fl = np.zeros(dnf.shape, bool)
    fl[np.arange(n_sims), lap.argmin(axis=1)] = True
    return fl & ~dnf                           # a race where every car retired has no fastest lap


@cache
def fl_table(path=None):
    """sports/f1/fastest_lap.toml as (weight per finishing position 1..99, pace_scale). A position's weight is its
    band's rate, the fastest laps over the classified starts of every position in the band (`bands`, `by_position`);
    a position past the last band takes the last band's."""
    import tomllib
    where = path or FL_TABLE
    with open(where, "rb") as f:
        t = tomllib.load(f)
    rows = {r["position"]: r for r in t["by_position"]}
    rates = []
    for lo, hi in t["bands"]:
        band = [r for p, r in rows.items() if lo <= p <= hi]
        races = sum(r["races"] for r in band)
        if not races:
            raise ValueError(f"{where}: band {lo}-{hi} has no races")
        rates.append((lo, hi, sum(r["fl"] for r in band) / races))
    w = np.array([next((x for lo, hi, x in rates if lo <= p <= hi), rates[-1][2]) for p in range(1, 100)])
    return w, float(t["fit"]["pace_scale"])


def fastest_lap_from_position(pos, dnf, pace, rng, weights=None, pace_scale=None):
    """(n_sims, n) bool: who sets the race's fastest lap in each simulation (variant "flpos"): one classified car,
    drawn with probability proportional to weights[finishing position - 1] * exp(-(pace - the quickest classified
    pace) / pace_scale), from sports/f1/fastest_lap.toml (fl_table: band rates and a pace scale from 2022-26 races). The fastest lap so
    follows the simulated result (the winner sets it about a third of the time) and the car (a midfield winner in a
    slow car less often than a top-car winner). pace: (n_sims, n) or (n,) race pace, a fraction of a lap, lower is
    quicker. A retired car never gets it. Drawn on the same side stream as fastest_lap (_side_rng, FL_SEED), so every
    other output is byte-identical with or without it."""
    n_sims, n = pos.shape
    if weights is None or pace_scale is None:
        w0, s0 = fl_table()
        weights = w0 if weights is None else weights
        pace_scale = s0 if pace_scale is None else pace_scale
    w = np.asarray(weights, float)
    pace = np.broadcast_to(np.asarray(pace, float), (n_sims, n))
    best = np.where(dnf, np.inf, pace).min(axis=1, keepdims=True)
    gap = np.where(dnf, 0.0, pace - np.where(np.isfinite(best), best, 0.0))
    p = np.where(dnf, 0.0, w[np.clip(np.asarray(pos, int), 1, len(w)) - 1] * np.exp(-gap / pace_scale))
    tot = p.sum(axis=1, keepdims=True)
    cum = np.cumsum(p, axis=1) / np.where(tot > 0, tot, 1.0)
    u = _side_rng(rng, FL_SEED).random((n_sims, 1))
    pick = np.minimum((cum <= u).sum(axis=1), n - 1)
    fl = np.zeros(pos.shape, bool)
    fl[np.arange(n_sims), pick] = True
    return fl & (tot > 0) & ~dnf               # a race where every car retired has no fastest lap


def summarize(e, sim):
    pos, dnf = sim["pos"], sim["dnf"]
    out = dict(
        athlete_id=e["athlete_id"].to_numpy(), driver=e["driver"].to_numpy() if "driver" in e else None,
        team_key=e["team_key"].to_numpy(),
        win_prob=((pos == 1) & ~dnf).mean(0), podium_prob=((pos <= 3) & ~dnf).mean(0),
        top10_prob=((pos <= 10) & ~dnf).mean(0), pole_prob=(sim.get("qual", sim["grid"]) == 1).mean(0),
        dnf_prob=dnf.mean(0), exp_points=sim["points"].mean(0),
        exp_position=pos.mean(0),
        qp=e["qp"].to_numpy(), rp=e["rp"].to_numpy(),
        # Kalshi's top 5 and biggest mover (markets.kinds race_top5 / race_biggest_mover), stored in race_predictions.extra:
        # read from the same draws, no new randomness
        top5_prob=((pos <= 5) & ~dnf).mean(0), mover_prob=biggest_mover(sim["grid"], pos, ~dnf).mean(0),
    )
    if "fl" in sim:                            # FASTEST_LAP: stored in race_predictions.extra.fl_prob
        out["fl_prob"] = sim["fl"].mean(0)
    return pd.DataFrame(out).sort_values("win_prob", ascending=False).reset_index(drop=True)
