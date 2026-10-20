"""
F1 frames (L1, docs/frames.md), built from `Measurements`: the data object the F1 engine's `load` returns. Read-only
views over what the model already holds: no database query, and nothing about how the model loads or prices changes.

    entrants          the model's own entry list per event (pricing.entry_list: the qualifying classification, the
                      race when there is none), with the team_id the model keys teams by
    sessions          Measurements.res: one row per (event, session); start from the session timestamp
    classifications   Measurements.res: every session's result rows (practice rows have no position or time; the
                      sprint qualifying rows are all DNS, as the source records them)
    conditions        Measurements.prof: rain share and track temperature of the qualifying and the race
    venue_features    Measurements.prof: the whole track profile, as {feature: value} (nulls left out)
    official_results  the race's classification rows (the sprint stays in classifications, session "sprint")

`available_at` (naive UTC, like everything in position_sim):
  - Session rows (sessions, classifications): the session's end, start + `[sessions] minutes` in sports/f1.toml,
    through inputs.laps.session_end, the function Measurements.view gates `res` with. No lag is added: the
    30 minutes of `[stages] lag_minutes` move a stage's *cutoff* after a session ends, and `view` has none.
  - official_results: the race's end, the same rule.
  - conditions, venue_features: come from `track_profiles`, computed from the race results, so a profile is
    knowable once its race is over: `ready_ts + RACE_DONE`, the gate Measurements.view applies to `prof`. Practice
    weather isn't in the profile, so conditions covers "qual" and "race" only.
  - entrants: the end of the driver's first session of the event, i.e. when our tables first list them. Assumption to
    review in E2: the real entry list is published earlier, but nothing in the tables says when, and this can only be
    late, never early. `price_race` reads the entry list from the ungated `meas.res` ("identities and teams, never
    results"), so pricing before the first session still sees entrants: E2 has to carry that exception over.

Another E2 assumption, the sessions themselves: the `sessions` frame dates a session's `start` and `end` at the
session's end (available_at = end), but the model reads the weekend schedule ungated (Measurements.sessions: the
backtest's anchors, price_race's event_sessions and sessions_used, testing/checks.py; the pre_weekend golden depends
on it). So E2 reads `start` from the ungated sessions frame, like entrants, and gates only what a session produced.

Not built: `laps`. Measurements keeps measures derived from the laps (qualifying and race deficits, sector deficits,
practice pace, race disruption), not the laps, so no lap-level frame can come from it. `[data] frames` in
sports/f1.toml doesn't list it; a later task decides how the raw laps reach the adapter.

A row whose session has no timestamp gets a null `available_at`, and `validate` rejects it: nothing is dropped silently.
"""

import numpy as np
import pandas as pd

from racinglines.frames._util import empty, gap_to_leader, shape
from racinglines.models.inputs import laps as M
from racinglines.models.inputs.laps_pricing import entry_list

WEATHER = {"qual": ("qual_rain_share", "qual_track_temp"), "race": ("race_rain_share", "race_track_temp")}
PROFILE_META = ("event_id", "start_date", "venue", "ready_ts")      # the profile's own columns; the rest are features


def _res(meas):
    """The results rows, each with the end of its session."""
    res = meas.res
    return res.assign(end=M.session_end(res["round"], res["session_ts"]))


def _profile_ready(meas):
    return meas.prof["ready_ts"] + M.RACE_DONE


def entrants(meas):
    """The model's entry list of each event (`pricing.entry_list`), with the team name, the race and the end of the
    driver's first session of the event."""
    res = _res(meas)
    seen = res.groupby(["event_id", "athlete_id"], as_index=False).agg(
        race_id=("race_id", "first"), team=("team", "first"), available_at=("end", "min"))
    listed = [entry_list(meas, e).assign(event_id=e) for e in res["event_id"].drop_duplicates()]
    if not listed:
        return empty("entrants")
    g = pd.concat(listed, ignore_index=True).rename(columns={"driver": "name", "team_key": "team_id"}).merge(
        seen, on=["event_id", "athlete_id"], how="left")
    g["bib"] = None                                   # the source has no start numbers
    return shape("entrants", g)


def sessions(meas):
    g = _res(meas).groupby(["event_id", "round"], as_index=False).agg(start=("session_ts", "min"), end=("end", "min"))
    g = g.rename(columns={"round": "session"})
    g["available_at"] = g["end"]
    return shape("sessions", g)


def classifications(meas):
    r = _res(meas)
    c = pd.DataFrame({
        "event_id": r["event_id"], "session": r["round"], "athlete_id": r["athlete_id"],
        "position": r["position"].astype(float), "status": r["status"], "time_ms": r["time_ms"].astype(float),
        "grid": pd.to_numeric(r["grid"], errors="coerce"), "points": pd.to_numeric(r["points"], errors="coerce"),
        "available_at": r["end"]})
    c["gap_ms"] = gap_to_leader(c)
    return shape("classifications", c)


def conditions(meas):
    p = meas.prof
    parts = []
    for session, (rain, temp) in WEATHER.items():
        d = pd.DataFrame({
            "event_id": p["event_id"], "session": session,
            "rain_share": pd.to_numeric(p[rain], errors="coerce") if rain in p else np.nan,
            "track_temp": pd.to_numeric(p[temp], errors="coerce") if temp in p else np.nan,
            "available_at": _profile_ready(meas)})
        parts.append(d[d["rain_share"].notna() | d["track_temp"].notna()])
    return shape("conditions", pd.concat(parts, ignore_index=True))


def venue_features(meas):
    p = meas.prof
    cols = [c for c in p.columns if c not in PROFILE_META]
    feats = [{k: v for k, v in row.items() if pd.notna(v)} for row in p[cols].to_dict("records")]
    out = pd.DataFrame({"event_id": p["event_id"].to_numpy(), "venue": p["venue"].to_numpy(), "features": feats,
                        "available_at": _profile_ready(meas).to_numpy()})
    return shape("venue_features", out)


def official_results(meas):
    c = classifications(meas)
    out = c.loc[c["session"] == "race", ["event_id", "athlete_id", "position", "status", "points", "available_at"]].copy()
    out["rounds_reached"] = None                      # no elimination rounds in F1
    return shape("official_results", out)


BUILDERS = {"entrants": entrants, "sessions": sessions, "classifications": classifications,
            "conditions": conditions, "venue_features": venue_features, "official_results": official_results}
