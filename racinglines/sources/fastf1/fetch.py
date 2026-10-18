"""
Download Formula 1 sessions from the official F1 live-timing archive (via
FastF1) into data/raw/f1/fastf1/<year>/, one set of files per session:

    <round>_<session>.results.parquet  classification: driver, team, grid, position, status, points, Q1-Q3
    <round>_<session>.laps.parquet     every lap: lap/sector times, speed traps, tyre, pit, track status
    <round>_<session>.meta.json        event name, circuit, date, format, weather summary

Sessions: Q (qualifying), S (sprint), R (race) by default; practice FP1-FP3 and
SQ (sprint qualifying / shootout) with --sessions. Only sessions the weekend
actually had are fetched. Already-downloaded sessions are skipped, so this is
safe to re-run (e.g. after each race weekend).
FastF1 keeps its own HTTP cache in data/cache/fastf1 (cleared after each fetch).
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]      # the repository
from racinglines import paths  # noqa: E402

OUT = paths.F1_RAW
CACHE = paths.cache("fastf1")
SESSIONS = ("Q", "S", "R")
PRACTICE = ("FP1", "FP2", "FP3", "SQ")
SCHEDULE_NAME = {"FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3", "SQ": ("Sprint Qualifying", "Sprint Shootout"),
                 "Q": "Qualifying", "S": "Sprint", "R": "Race"}
SPRINT_FORMATS = ("sprint", "sprint_shootout", "sprint_qualifying")

TIME_COLS_RESULTS = ["Q1", "Q2", "Q3", "Time"]
TIME_COLS_LAPS = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "PitInTime", "PitOutTime",
                  "LapStartTime", "Time"]
LAP_COLS = ["Driver", "DriverNumber", "Team", "LapNumber", "Stint", "LapTime", "Sector1Time", "Sector2Time",
            "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "Compound", "TyreLife", "FreshTyre",
            "PitInTime", "PitOutTime", "TrackStatus", "Position", "Deleted", "IsAccurate", "LapStartTime", "Time"]
RESULT_COLS = ["DriverNumber", "Abbreviation", "DriverId", "FullName", "CountryCode", "TeamName", "TeamId",
               "Position", "ClassifiedPosition", "GridPosition", "Status", "Points", "Laps", "Q1", "Q2", "Q3", "Time"]


# the event schedule's columns kept in <year>/schedule.parquet (event_schedule): the stages (sessions and their UTC
# starts) and what the forecast's calendar reads (round, names, location, race day, format)
SCHEDULE_COLS = ["RoundNumber", "Country", "Location", "OfficialEventName", "EventDate", "EventName", "EventFormat"] + \
    [f"Session{i}{x}" for i in range(1, 6) for x in ("", "DateUtc")]


def offline():
    """RACINGLINES_OFFLINE=1: never call FastF1's servers, read only what is already under data/raw/f1/fastf1 (a
    machine without access to them, e.g. a cloud session, or a store built from files, e.g. the demo's)."""
    return os.environ.get("RACINGLINES_OFFLINE", "").strip().lower() not in ("", "0", "false", "no")


def event_schedule(year, save=True):
    """FastF1's event schedule for a season (SCHEDULE_COLS), saved next to the raw files
    (data/raw/f1/fastf1/<year>/schedule.parquet) when that folder exists and `save`. Offline (offline()) the saved copy
    is read without trying FastF1; online, the saved copy is the fallback when FastF1 can't be reached."""
    snap = OUT / str(year) / "schedule.parquet"
    if offline():
        if snap.is_file():
            return pd.read_parquet(snap)
        raise FileNotFoundError(f"offline (RACINGLINES_OFFLINE): no saved schedule at {snap}")
    try:
        import fastf1
        logging.getLogger("fastf1").setLevel(logging.ERROR)
        CACHE.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE))
        sch = pd.DataFrame(fastf1.get_event_schedule(year, include_testing=False))
        sch = sch[[c for c in SCHEDULE_COLS if c in sch]]
        if save and snap.parent.is_dir():
            sch.to_parquet(snap, index=False)
        return sch
    except Exception:  # noqa: BLE001  (unreachable: fall back to the saved copy)
        if snap.is_file():
            return pd.read_parquet(snap)
        raise


def _seconds(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_timedelta(df[c]).dt.total_seconds()
    return df


def session_started(event, kind, now):
    """Has this weekend's session `kind` started by `now` (naive UTC)? EventDate is the race day, so a weekend in
    progress is judged per session: practice and qualifying are fetched before the race. A session the schedule
    doesn't date falls back to the race day; one the event doesn't have is never started."""
    names = SCHEDULE_NAME[kind] if isinstance(SCHEDULE_NAME[kind], tuple) else (SCHEDULE_NAME[kind],)
    for i in range(1, 6):
        if str(event.get(f"Session{i}")) in names:
            t = event.get(f"Session{i}DateUtc")
            if t is None or pd.isna(t):
                return pd.Timestamp(event["EventDate"]).date() <= now.date()
            return pd.Timestamp(t).tz_localize(None) <= now
    return False


# How long after its start a session's data is final: the session's length plus a margin (a race is capped at 3 h
# including suspensions). A session fetched while it runs comes back partial and would then be kept as "cached".
SESSION_LENGTH = {"FP1": pd.Timedelta("75min"), "FP2": pd.Timedelta("75min"), "FP3": pd.Timedelta("75min"),
                  "SQ": pd.Timedelta("60min"), "S": pd.Timedelta("75min"), "Q": pd.Timedelta("75min"),
                  "R": pd.Timedelta("210min")}


def session_ended(event, kind, now):
    """Is this weekend's session `kind` over by `now` (naive UTC): started at least SESSION_LENGTH[kind] ago?"""
    return session_started(event, kind, now - SESSION_LENGTH.get(kind, pd.Timedelta("75min")))


def fetch_session(fastf1, year, event, kind, force=False):
    rnd = int(event["RoundNumber"])
    base = OUT / str(year) / f"{rnd:02d}_{kind}"
    if base.with_suffix(".meta.json").exists() and base.with_suffix(".laps.parquet").exists() and not force:
        return "cached"
    session = fastf1.get_session(year, rnd, kind)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    results = _seconds(session.results[[c for c in RESULT_COLS if c in session.results]].copy(), TIME_COLS_RESULTS)
    laps = _seconds(pd.DataFrame(session.laps)[[c for c in LAP_COLS if c in session.laps]].copy(), TIME_COLS_LAPS)
    if results.empty:
        return "no results"
    for df in (results, laps):  # mixed-type object columns (e.g. ClassifiedPosition "1"/"R") -> str
        for c in df.columns[df.dtypes == object]:
            df[c] = df[c].astype("string")
    base.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(base.with_suffix(".results.parquet"), index=False)
    laps.to_parquet(base.with_suffix(".laps.parquet"), index=False)
    w = session.weather_data
    meta = dict(
        year=year, round=rnd, session=kind, event_name=event["EventName"], official_name=event.get("OfficialEventName"),
        country=event.get("Country"), location=event.get("Location"), event_date=str(pd.Timestamp(event["EventDate"]).date()),
        session_date=str(session.date) if getattr(session, "date", None) is not None else None,
        event_format=event.get("EventFormat"), total_laps=getattr(session, "total_laps", None),
        weather=None if w is None or w.empty else dict(
            air_temp=float(w["AirTemp"].mean()), track_temp=float(w["TrackTemp"].mean()),
            rain_share=float(w["Rainfall"].astype(float).mean()), humidity=float(w["Humidity"].mean()),
            wind_speed=float(w["WindSpeed"].mean())),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    base.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return f"{len(results)} drivers, {len(laps)} laps"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2026,2025,2024,2023,2022,2021,2020",
                    help="Years in the order to fetch, e.g. 2026,2025 or 2020-2026. Newest first gets the data "
                         "the forecast needs soonest (FastF1 allows ~500 API calls/hour, about 65 sessions).")
    ap.add_argument("--sprints-from", type=int, default=2026,
                    help="Only fetch sprint sessions from this season on (older sprints aren't used by the model; "
                         "the current season's are needed for championship points).")
    ap.add_argument("--force", action="store_true", help="Re-download sessions that are already on disk.")
    ap.add_argument("--sessions", default="Q,S,R", help="Session codes: Q,S,R and/or FP1,FP2,FP3,SQ.")
    ap.add_argument("--rounds", default=None, help="Only these rounds, e.g. 6-15 or 13,15 (default: all raced).")
    ap.add_argument("--keep-cache", action="store_true",
                    help="Keep FastF1's HTTP cache (data/cache/fastf1, ~1 GB for 2020-2025) after fetching. By default it's "
                         "cleared: every session is saved as Parquet and re-fetches skip sessions already on disk.")
    args = ap.parse_args()
    import fastf1

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    if "-" in args.years:
        a, _, b = args.years.partition("-")
        years = list(range(int(a), int(b) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    rounds = None
    if args.rounds:
        rounds = set()
        for part in args.rounds.split(","):
            lo, _, hi = part.partition("-")
            rounds |= set(range(int(lo), int(hi or lo) + 1))
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            if rounds and int(event["RoundNumber"]) not in rounds:
                continue
            wanted = [k.strip() for k in args.sessions.split(",") if k.strip()]
            kinds = [k for k in wanted if session_ended(event, k, now) and (k != "S" or year >= args.sprints_from)]
            for kind in kinds:
                t = time.time()
                for attempt in range(13):
                    try:
                        status = fetch_session(fastf1, year, event, kind, force=args.force)
                        break
                    except Exception as e:  # keep going; a missing session shouldn't stop the backfill
                        status = f"ERROR {type(e).__name__}: {e}"
                        if "RateLimit" not in type(e).__name__:
                            break
                        # FastF1 caps API calls at 500/hour; wait for the window to roll over
                        print(f"rate limit reached; waiting 5 min (attempt {attempt + 1})", flush=True)
                        time.sleep(300)
                print(f"{year} R{int(event['RoundNumber']):02d} {kind:<2} {event['EventName']:<32} {status} "
                      f"({time.time() - t:.0f}s)", flush=True)
    if not args.keep_cache:
        clear_cache()


def clear_cache():
    """Remove FastF1's HTTP cache; the Parquet under data/raw/f1/fastf1/<year>/ is the record."""
    import shutil
    cache = CACHE
    if cache.exists():
        size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
        shutil.rmtree(cache)
        print(f"cleared FastF1 cache ({size / 1e6:,.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
