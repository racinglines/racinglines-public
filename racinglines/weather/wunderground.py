"""
Weather Underground forecasts (docs/weather.md), an optional provider of the forecast frame (weather/schema.py);
Open-Meteo (weather/open_meteo.py) is the default. Never probed: the owner has no PWS key.

Weather Underground's forecasts are served by The Weather Company's API (api.weather.com, the "v3" endpoints), called
with an `apiKey` query parameter: the key a Weather Underground PWS owner gets, read from $WUNDERGROUND_API_KEY.

UNVERIFIED IN THE CLOUD. The cloud has no network to any weather host, so every endpoint path, query parameter and
response field name below is written from memory of the public API documentation, not from a response we captured.
Each constant carries the same mark. To verify them, on the Mac:

    racinglines weather probe --lat <lat> --lon <lon>

writes the untouched responses to data/raw/weather/wunderground/ and lists every declared field the response doesn't
have. That file, trimmed, replaces tests/fixtures/weather/wunderground-sample.json (hand-written to the declared shape
today). Also unverified: whether a PWS key may call the hourly endpoint at all (`fetch_raw` keeps the daily forecast
and records the hourly error when it may not), the key's rate limits, and the units under units=m (C, km/h, mm).

    fetch_raw(lat, lon, key, hourly=True)                  {"daily": body, "hourly": body | None, ...}, the envelope
    parse(raw, event_key, lat, lon, issued_utc, sessions)  the forecast frame; a declared field the response lacks
                                                           raises WundergroundFieldError naming it (never NaN columns)
    probe(lat, lon, key, out_path)                         fetch_raw, written untouched to out_path; returns
                                                           (path, missing fields)
"""

import json
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.weather import schema as S

SOURCE = "wunderground"
KEY_ENV = "WUNDERGROUND_API_KEY"

# unverified in the cloud: confirm with `racinglines weather probe` on the Mac
BASE_URL = "https://api.weather.com"
# unverified in the cloud: confirm with `racinglines weather probe` on the Mac
DAILY_PATH = "/v3/wx/forecast/daily/5day"
# unverified in the cloud: confirm with `racinglines weather probe` on the Mac (and whether a PWS key may call it)
HOURLY_PATH = "/v3/wx/forecast/hourly/2day"
# unverified in the cloud: confirm with `racinglines weather probe` on the Mac (geocode "lat,lon", metric units)
PARAMS = {"format": "json", "units": "m", "language": "en-US"}
GEOCODE_PARAM = "geocode"           # unverified in the cloud: confirm with `racinglines weather probe` on the Mac
KEY_PARAM = "apiKey"                # unverified in the cloud: confirm with `racinglines weather probe` on the Mac

# Daily body: parallel arrays, one entry per day, plus "daypart": [ {parallel arrays, two entries per day: day, night} ]
D_TIME = "validTimeUtc"             # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (epoch s)
D_TMAX = "temperatureMax"           # unverified in the cloud: confirm with `racinglines weather probe` on the Mac
D_TMIN = "temperatureMin"           # unverified in the cloud: confirm with `racinglines weather probe` on the Mac
D_QPF = "qpf"                       # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (mm)
D_NARRATIVE = "narrative"           # unverified in the cloud: confirm with `racinglines weather probe` on the Mac
D_DAYPART = "daypart"               # unverified in the cloud: confirm with `racinglines weather probe` on the Mac
DP_PRECIP_CHANCE = "precipChance"   # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (%)
DP_HUMIDITY = "relativeHumidity"    # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (%)
DP_WIND = "windSpeed"               # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (km/h)

# Hourly body: parallel arrays, one entry per hour
H_TIME = "validTimeUtc"             # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (epoch s)
H_TEMP = "temperature"              # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (C)
H_PRECIP_CHANCE = "precipChance"    # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (%)
H_QPF = "qpf"                       # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (mm)
H_HUMIDITY = "relativeHumidity"     # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (%)
H_WIND = "windSpeed"                # unverified in the cloud: confirm with `racinglines weather probe` on the Mac (km/h)
H_PHRASE = "wxPhraseLong"           # unverified in the cloud: confirm with `racinglines weather probe` on the Mac

DAILY_FIELDS = (D_TIME, D_TMAX, D_TMIN, D_QPF, D_NARRATIVE, D_DAYPART)
DAYPART_FIELDS = (DP_PRECIP_CHANCE, DP_HUMIDITY, DP_WIND)
HOURLY_FIELDS = (H_TIME, H_TEMP, H_PRECIP_CHANCE, H_QPF, H_HUMIDITY, H_WIND, H_PHRASE)
NAMED_SESSIONS = ("race", "qual", "sprint")


class WundergroundFieldError(ValueError):
    """A declared field the response doesn't have (or doesn't hold as the declared shape). The message names it."""


def _redact(text, key):
    return text.replace(key, "<key>") if key else text


def _get(session, path, lat, lon, key):
    from racinglines.sources import http
    params = {GEOCODE_PARAM: f"{lat},{lon}", **PARAMS, KEY_PARAM: key}
    r = http.get(session, BASE_URL + path, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(_redact(f"wunderground {path}: HTTP {r.status_code}: {r.text[:300]}", key))
    return r.json()


def fetch_raw(lat, lon, key, hourly=True):
    """The daily forecast and (hourly=True) the hourly one, as the envelope parse() reads:
    {"provider", "fetched_utc", "lat", "lon", "daily": body, "hourly": body | None, "errors": {...}}. The bodies are the
    responses' JSON untouched. A failed daily request raises; a failed hourly one is recorded under errors["hourly"]
    (a PWS key may not cover it: unverified) and the daily forecast is kept. The key never lands in the envelope."""
    import requests
    if not key:
        raise SystemExit(f"no Weather Underground key: set {KEY_ENV}")
    out = {"provider": SOURCE, "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "lat": lat, "lon": lon, "daily": None, "hourly": None, "errors": {}}
    with requests.Session() as s:
        out["daily"] = _get(s, DAILY_PATH, lat, lon, key)
        if hourly:
            try:
                out["hourly"] = _get(s, HOURLY_PATH, lat, lon, key)
            except (RuntimeError, ValueError) as ex:
                out["errors"]["hourly"] = _redact(str(ex), key)
    return out


def _field(body, name, where, n=None):
    if not isinstance(body, dict):
        raise WundergroundFieldError(f"wunderground {where}: expected a JSON object, got {type(body).__name__}")
    if name not in body:
        raise WundergroundFieldError(
            f"wunderground {where}: no field {name!r} (declared in weather/wunderground.py, unverified); the response "
            f"has {sorted(body)[:40]}")
    v = body[name]
    if n is not None and (not isinstance(v, list) or len(v) != n):
        got = len(v) if isinstance(v, list) else type(v).__name__
        raise WundergroundFieldError(f"wunderground {where}: field {name!r} should be a list of {n}, got {got}")
    return v


def _num(values):
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _utc(epochs):
    return pd.to_datetime(_num(epochs), unit="s")       # naive UTC


def _daily(body):
    t = _field(body, D_TIME, "daily")
    if not isinstance(t, list):
        raise WundergroundFieldError(f"wunderground daily: field {D_TIME!r} should be a list")
    n = len(t)
    tmax, tmin = _num(_field(body, D_TMAX, "daily", n)), _num(_field(body, D_TMIN, "daily", n))
    dp = _field(body, D_DAYPART, "daily")
    if not isinstance(dp, list) or not dp:
        raise WundergroundFieldError(f"wunderground daily: field {D_DAYPART!r} should be a list of one object")
    # two entries per day, day then night, null for a half already past (unverified: the probe shows the order)
    parts = {f: _num(_field(dp[0], f, "daily.daypart[0]", 2 * n)).reshape(n, 2) for f in DAYPART_FIELDS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows (a past day part) stay NaN
        chance = np.nanmax(parts[DP_PRECIP_CHANCE], axis=1) / 100 if n else np.array([])
        wind = np.nanmax(parts[DP_WIND], axis=1) if n else np.array([])
        hum = np.nanmean(parts[DP_HUMIDITY], axis=1) if n else np.array([])
        temp = np.nanmean(np.vstack([tmax, tmin]), axis=0) if n else np.array([])
    return pd.DataFrame({
        "session": "day", "valid_utc": _utc(t), "precip_prob": chance,
        "precip_mm": _num(_field(body, D_QPF, "daily", n)), "temp_c": temp, "humidity": hum, "wind_kph": wind,
        "condition": [None if v is None else str(v) for v in _field(body, D_NARRATIVE, "daily", n)]})


def _hourly(body):
    t = _field(body, H_TIME, "hourly")
    if not isinstance(t, list):
        raise WundergroundFieldError(f"wunderground hourly: field {H_TIME!r} should be a list")
    n = len(t)
    return pd.DataFrame({
        "session": "hour", "valid_utc": _utc(t),
        "precip_prob": _num(_field(body, H_PRECIP_CHANCE, "hourly", n)) / 100,
        "precip_mm": _num(_field(body, H_QPF, "hourly", n)), "temp_c": _num(_field(body, H_TEMP, "hourly", n)),
        "humidity": _num(_field(body, H_HUMIDITY, "hourly", n)), "wind_kph": _num(_field(body, H_WIND, "hourly", n)),
        "condition": [None if v is None else str(v) for v in _field(body, H_PHRASE, "hourly", n)]})


def _naive_utc(t):
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def parse(raw_json, event_key, lat, lon, issued_utc, sessions=None):
    """The forecast frame (schema.COLUMNS) from fetch_raw's envelope: one "day" row per daily entry and one row per
    hourly entry, labelled with the session whose window holds it (`sessions`: {"race" | "qual" | "sprint":
    (start, end)}, naive UTC, end exclusive) or "hour" outside every window. A day's precip_prob and wind_kph are the
    larger of its day and night halves, humidity their mean, temp_c the mean of its max and min. Validated before it is
    returned. A declared field the response doesn't have raises WundergroundFieldError naming it."""
    if not isinstance(raw_json, dict) or "daily" not in raw_json:
        raise WundergroundFieldError("wunderground: expected fetch_raw's envelope with a 'daily' body")
    parts = [_daily(raw_json["daily"])]
    if raw_json.get("hourly") is not None:
        h = _hourly(raw_json["hourly"])
        for name, (start, end) in (sessions or {}).items():
            if name not in NAMED_SESSIONS:
                raise ValueError(f"unknown session {name!r}; expected one of {NAMED_SESSIONS}")
            inside = (h["valid_utc"] >= _naive_utc(start)) & (h["valid_utc"] < _naive_utc(end))
            h.loc[inside & (h["session"] == "hour"), "session"] = name
        parts.append(h)
    df = pd.concat(parts, ignore_index=True)
    df = df.assign(event_key=str(event_key), issued_utc=_naive_utc(issued_utc), source=SOURCE, lat=float(lat), lon=float(lon))
    df["valid_utc"] = df["valid_utc"].astype("datetime64[ns]")
    df["issued_utc"] = df["issued_utc"].astype("datetime64[ns]")
    df["condition"] = df["condition"].astype(object)
    return S.validate(df[list(S.COLUMNS)].sort_values(["valid_utc", "session"], ignore_index=True))


def missing_fields(raw_json):
    """Every declared field the envelope's bodies don't have, as "daily.<name>" etc. [] when all are there."""
    out = []
    daily, hourly = (raw_json or {}).get("daily"), (raw_json or {}).get("hourly")
    if isinstance(daily, dict):
        out += [f"daily.{f}" for f in DAILY_FIELDS if f not in daily]
        dp = daily.get(D_DAYPART)
        first = dp[0] if isinstance(dp, list) and dp and isinstance(dp[0], dict) else {}
        out += [f"daily.{D_DAYPART}[0].{f}" for f in DAYPART_FIELDS if f not in first]
    else:
        out.append("daily (no body)")
    if isinstance(hourly, dict):
        out += [f"hourly.{f}" for f in HOURLY_FIELDS if f not in hourly]
    return out


def probe(lat, lon, key, out_path):
    """Fetch the daily and hourly forecasts for (lat, lon) and write the envelope untouched (bodies as the API sent
    them, the key left out) to out_path, so the owner can trim it into the test fixture. Returns (path, missing),
    `missing` the declared fields the response doesn't have (missing_fields)."""
    raw = fetch_raw(lat, lon, key, hourly=True)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(raw, indent=1))
    return p, missing_fields(raw)


def api_key():
    return os.environ.get(KEY_ENV, "").strip()
