"""
The forecast frame (docs/weather.md): what a model reads about the weather, whoever forecast it.

One row per (event_key, session, valid_utc):

    event_key    the sportsbook event key, e.g. "2026-17"
    session      "race", "qual", "sprint" (an hour inside that session's window), "day" (a daily forecast row) or
                 "hour" (an hourly row outside every session window the caller gave)
    valid_utc    the time the row forecasts (naive UTC: an hour's start, or the provider's own time for a day)
    issued_utc   when the forecast was issued (naive UTC; the fetch time when the provider doesn't say)
    precip_prob  chance of precipitation, 0 to 1
    precip_mm    expected precipitation, mm (the hour's, or the day's total)
    temp_c       air temperature, C (a day: the mean of its max and min)
    humidity     relative humidity, 0 to 100
    wind_kph     wind speed, km/h
    condition    the provider's own text ("Showers", "Partly Cloudy" ...)
    source       the provider code ("open_meteo", "wunderground")
    lat, lon     where the forecast is for

    validate(df)            df itself, or FrameError listing every problem, each naming its column
    summary(df, session)    {precip_prob: max, precip_mm: sum, temp_c: mean, wind_kph: max, issued_utc, source}
                            over the session's rows, or None when it has none
    save(df) / load(key)    one file per issue: <data>/weather/<event_key>/<source>-<issued>.parquet; load returns
                            the latest issue (of one source, with source=)

Times are naive UTC, the convention of position_sim and frames/. The data root is $RACINGLINES_DATA, read when
save / load run (so a test can point it at a tmp dir), else paths.DATA.
"""

import os
from pathlib import Path

import pandas as pd
from pandas.api import types as T

from racinglines.frames.schema import FrameError

ROOT_ENV = "RACINGLINES_DATA"
SESSIONS = ("race", "qual", "sprint", "day", "hour")
KEY = ("event_key", "session", "valid_utc")

# column: (dtype family, nullable); the families are frames/schema.py's
COLUMNS = {
    "event_key": ("str", False),
    "session": ("str", False),
    "valid_utc": ("datetime", False),
    "issued_utc": ("datetime", False),
    "precip_prob": ("number", True),
    "precip_mm": ("number", True),
    "temp_c": ("number", True),
    "humidity": ("number", True),
    "wind_kph": ("number", True),
    "condition": ("str", True),
    "source": ("str", False),
    "lat": ("number", False),
    "lon": ("number", False),
}
RANGES = {"precip_prob": (0, 1), "precip_mm": (0, None), "humidity": (0, 100), "wind_kph": (0, None),
          "lat": (-90, 90), "lon": (-180, 180)}
STAMP = "%Y%m%dT%H%M%SZ"         # issued_utc in a file name


def _family_ok(s, family):
    if family == "datetime":
        return T.is_datetime64_dtype(s) and getattr(s.dt, "tz", None) is None
    if family == "number":
        return T.is_numeric_dtype(s) and not T.is_bool_dtype(s)
    if family == "str":
        return (T.is_string_dtype(s) or T.is_object_dtype(s)) and bool(s.dropna().map(lambda v: isinstance(v, str)).all())
    raise ValueError(family)


def problems(df):
    """Every way df differs from the forecast frame, one line each, naming the column. [] when it matches."""
    out = [f"missing column {c!r}" for c in COLUMNS if c not in df.columns]
    for c, (family, nullable) in COLUMNS.items():
        if c not in df.columns:
            continue
        s = df[c]
        if not _family_ok(s, family):
            out.append(f"column {c!r}: dtype {s.dtype} is not {family}"
                       + (" (naive UTC)" if family == "datetime" else ""))
            continue
        if not nullable and s.isna().any():
            out.append(f"column {c!r}: {int(s.isna().sum())} null value(s)")
        if c in RANGES:
            lo, hi = RANGES[c]
            bad = pd.Series(False, index=s.index)
            if lo is not None:
                bad |= s.notna() & (s < lo)
            if hi is not None:
                bad |= s.notna() & (s > hi)
            if bad.any():
                out.append(f"column {c!r}: {int(bad.sum())} value(s) outside [{lo}, {hi if hi is not None else ''}]"
                           f", e.g. {s[bad].iloc[0]!r}")
    if "session" in df.columns:
        odd = sorted(set(df["session"].dropna()) - set(SESSIONS))
        if odd:
            out.append(f"column 'session': unknown value(s) {odd}; expected one of {list(SESSIONS)}")
    if all(k in df.columns for k in KEY) and len(df):
        dup = df.duplicated(list(KEY))
        if dup.any():
            out.append(f"key {list(KEY)}: {int(dup.sum())} duplicate row(s)")
    return out


def validate(df):
    """df itself, or FrameError whose .problems lists each mismatch (problems(df))."""
    p = problems(df)
    if p:
        raise FrameError("forecast frame: " + "; ".join(p), p)
    return df


def summary(df, session):
    """The session's forecast in one dict: precip_prob max, precip_mm sum, temp_c mean, wind_kph max (NaN when the
    provider gave none), the latest issued_utc and the source(s). None when df has no rows for the session."""
    rows = df[df["session"] == session]
    if rows.empty:
        return None
    return {"precip_prob": float(rows["precip_prob"].max()), "precip_mm": float(rows["precip_mm"].sum(min_count=1)),
            "temp_c": float(rows["temp_c"].mean()), "wind_kph": float(rows["wind_kph"].max()),
            "issued_utc": rows["issued_utc"].max(), "source": ",".join(sorted(rows["source"].unique()))}


def root():
    if os.environ.get(ROOT_ENV):
        return Path(os.environ[ROOT_ENV])
    from racinglines import paths
    return paths.DATA


def event_dir(event_key):
    return root() / "weather" / str(event_key)


def save(df):
    """Validate df and write it as one issue: <data>/weather/<event_key>/<source>-<issued>.parquet. One event, one
    source and one issued_utc per file. Returns the path."""
    validate(df)
    for c in ("event_key", "source", "issued_utc"):
        if df[c].nunique() != 1:
            raise FrameError(f"forecast frame: save needs one {c} per file, got {df[c].nunique()}",
                             [f"column {c!r}: {df[c].nunique()} distinct values"])
    key, source, issued = df["event_key"].iloc[0], df["source"].iloc[0], df["issued_utc"].iloc[0]
    d = event_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{source}-{pd.Timestamp(issued).strftime(STAMP)}.parquet"
    df[list(COLUMNS)].to_parquet(p, index=False)
    return p


def issues(event_key, source=None):
    """[(issued_utc, source, path)] of the saved issues for the event, oldest first."""
    out = []
    for p in sorted(event_dir(event_key).glob("*.parquet")):
        src, _, stamp = p.stem.rpartition("-")
        try:
            t = pd.to_datetime(stamp, format=STAMP)
        except ValueError:
            continue
        if source is None or src == source:
            out.append((t, src, p))
    return sorted(out)


def load(event_key, source=None):
    """The latest saved issue for the event (of `source`, when given), validated. FileNotFoundError when none."""
    found = issues(event_key, source)
    if not found:
        raise FileNotFoundError(f"no saved forecast for {event_key!r}"
                                + (f" from {source!r}" if source else "") + f" in {event_dir(event_key)}")
    return validate(pd.read_parquet(found[-1][2]))
