"""
Downhill frames (L1, docs/frames.md), built from the tidy runs frame the downhill engine's `load` returns. Read-only
views over
what the model already holds: no database query, and nothing about how the model loads or prices changes. Only the
FINISH rows are used (one per rider per round; a frame loaded with splits also has S1, S2 ... rows).

    entrants          one row per (event, rider), everyone with a row in the event, DNS riders included
    sessions          one row per (event, round), in the rounds' own names ([rounds] in sports/mtb_dh.toml)
    classifications   one row per (event, round, rider): rank, status and time of each run; gap_ms is to the round's
                      fastest run; no grid or points at round level (the source has neither)
    official_results  one row per rider of every completed event (the model's own `completed_events`: the final has
                      results): the Final's rank, the event's points, and the race rounds the rider
                      reached (start-list entries, DNS included)

`event_id` already names the category ("20260821_mtb_ME"), so an event is one category's race weekend.
`official_results` reuses the model's own `actual_event_points`, with the rules the engine's `results` applies: position =
the Final rank, status OK when ranked in the Final and DNF otherwise, points 0 when none. `rounds_reached` lists the
race rounds (`[rounds] race`) in which the rider has a row (a start-list entry, so a DNS in the Final still
reached it, as `results()`' `reached_final` counts it), in running order (`[rounds] order`).

`available_at` (naive UTC): the tidy frame carries an event *date* (the weekend's first day at best, not even always
that), not session times, and mtb_dh.toml has no session minutes. The tables have no end date either, so the end of
the weekend is parsed from the event name ("... Les Gets, August 21-23, FRA": the last day of the range, a range
across two months such as "May 30-Jun 1" included); when that fails, or lands outside the two weeks after the event
date, it is the event date plus two days (a three-day weekend). Every row of an event is knowable from noon UTC of the
day after that last day (midnight in the far west, UTC-7, is 07:00 UTC, so the last run has finished by then). A
session's `start` is the event date's midnight and its `end` that instant: the bounds of the weekend, not of the round.
The frame agrees with the model's day rule (`_training_rows`: the rows of an earlier event, `event_date < cutoff`) at
every event date; it is later than that rule inside a weekend (the rule has an event's rows from the day after its
date, the frame from the day after its last day), and the tables have no end date, so the end is parsed from the event
name. Assumption to review in E2, where an `events.end_date` set at ingest replaces the parse.
Entrants and official_results use the same date: the model reads the actual start list of the event it prices
(`event_starters`), before that date, so E2 has to carry that read over as an exception ("identities, never results").

Not built: `laps` (downhill has splits instead), `conditions` (the frame's track_condition is a constant
"unknown"), `venue_features` (no venue history in the frame).
"""

import re

import numpy as np
import pandas as pd

from racinglines import sports
from racinglines.frames._util import empty, gap_to_leader, shape
from racinglines.frames.schema import FrameError
from racinglines.models.inputs.run_times import actual_event_points, completed_events

DAY = pd.Timedelta(days=1)
AFTER_LAST_DAY = DAY + pd.Timedelta(hours=12)      # noon UTC of the day after the weekend's last day
FALLBACK_LAST_DAY = pd.Timedelta(days=2)           # a three-day weekend, when the name has no date range
MAX_WEEKEND = pd.Timedelta(days=14)                # a parsed end further out than this is a misparse
ROUNDS = sports.load("mtb_dh")["rounds"]           # sports/mtb_dh.toml [rounds]
NEED = ("event_id", "athlete_id", "race_id", "rider_id", "round", "event_date", "rank_at_split", "status",
        "cum_time_s", "bib", "rider_name", "team")

_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_MONTH = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_RANGE = re.compile(rf"\b{_MONTH}\.?\s+(\d{{1,2}})\s*-\s*(?:{_MONTH}\.?\s+)?(\d{{1,2}})\b")


def _last_day(date, name):
    """The last day of the weekend: the end of the date range in the event name ("August 21-23", "May 30-Jun 1"),
    or the event date plus two days when the name has none (or a range that doesn't fit the event date)."""
    start = pd.Timestamp(date).normalize()
    m = _RANGE.search(name) if isinstance(name, str) else None
    if m:
        month = _MONTHS[(m.group(3) or m.group(1))[:3].lower()]
        for year in (start.year, start.year + 1):
            try:
                end = pd.Timestamp(year=year, month=month, day=int(m.group(4)))
            except ValueError:
                continue
            if start <= end <= start + MAX_WEEKEND:
                return end
    return start + FALLBACK_LAST_DAY


def _runs(data):
    """The FINISH rows, with the start of the day of their event and the end of its weekend (the last day of the
    range in the event name, plus a day and 12 hours)."""
    missing = [c for c in NEED if c not in data.columns]
    if missing:
        raise FrameError(f"the downhill frames are built from the tidy runs frame (racinglines.db.queries.load_tidy), "
                         f"which has {missing} too")
    d = data[data["sector_id"] == "FINISH"] if "sector_id" in data else data
    day = pd.to_datetime(d["event_date"])
    names = d["event_name"] if "event_name" in d else pd.Series(None, index=d.index, dtype=object)
    pairs = pd.DataFrame({"day": day, "name": names}).drop_duplicates()
    last = {(r.day, r.name): _last_day(r.day, r.name) for r in pairs.itertuples()}
    end = pd.Series([last[k] for k in zip(day, names)], index=d.index, dtype="datetime64[ns]") + AFTER_LAST_DAY
    return d.assign(_start=day, _end=end)


def _text(s):
    """A str (or None) per value: start numbers can arrive as numbers (12.0 is "12")."""
    def one(v):
        if pd.isna(v):
            return v
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)
    return s.map(one).astype(object)


def entrants(data):
    d = _runs(data)
    g = d.groupby(["event_id", "athlete_id"], as_index=False).agg(
        race_id=("race_id", "first"), name=("rider_name", "first"), team=("team", "first"), bib=("bib", "first"),
        available_at=("_end", "min"))
    g["team_id"] = None                                # the tidy frame has team names only
    g["bib"] = _text(g["bib"])
    return shape("entrants", g)


def sessions(data):
    g = _runs(data).groupby(["event_id", "round"], as_index=False).agg(start=("_start", "min"), end=("_end", "min"))
    g = g.rename(columns={"round": "session"})
    g["available_at"] = g["end"]
    return shape("sessions", g)


def classifications(data):
    d = _runs(data)
    c = pd.DataFrame({
        "event_id": d["event_id"], "session": d["round"], "athlete_id": d["athlete_id"],
        "position": d["rank_at_split"].astype(float), "status": d["status"],
        "time_ms": np.rint(d["cum_time_s"].astype(float) * 1000),
        "grid": np.nan, "points": np.nan, "available_at": d["_end"]})
    c["gap_ms"] = gap_to_leader(c)
    return shape("classifications", c)


def official_results(data):
    d = _runs(data)
    done = d[d["event_id"].isin(completed_events(d))]
    if done.empty:
        return empty("official_results")
    pts = actual_event_points(done)
    made = done.loc[done["round"] == "final", ["event_id", "rider_id"]]
    out = pd.concat([pts[["event_id", "rider_id"]], made]).drop_duplicates().merge(
        pts[["event_id", "rider_id", "final_rank", "points"]], on=["event_id", "rider_id"], how="left")
    out["athlete_id"] = out["rider_id"].map(done.drop_duplicates("rider_id").set_index("rider_id")["athlete_id"])
    out["position"] = out["final_rank"].astype(float)
    out["status"] = np.where(out["final_rank"].notna(), "OK", "DNF")
    out["points"] = out["points"].fillna(0.0)
    out["available_at"] = out["event_id"].map(done.groupby("event_id")["_end"].max())
    listed = done[done["round"].isin(ROUNDS["race"])]
    reached = (listed.assign(_order=listed["round"].map(ROUNDS["order"])).sort_values("_order", kind="stable")
               .groupby(["event_id", "rider_id"])["round"].agg(tuple).rename("rounds_reached"))
    out = out.join(reached, on=["event_id", "rider_id"])
    out["rounds_reached"] = [t if isinstance(t, tuple) else () for t in out["rounds_reached"]]
    return shape("official_results", out)


BUILDERS = {"entrants": entrants, "sessions": sessions, "classifications": classifications,
            "official_results": official_results}
