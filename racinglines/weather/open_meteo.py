"""
Open-Meteo forecasts (docs/weather.md), the default provider of the forecast frame (weather/schema.py).

Open-Meteo (open-meteo.com) serves global hourly and daily forecasts with no API key. The endpoint, query parameters
and every response field below are verified by the probe of 2026-10-06 (Marina Bay, data/raw/weather/open_meteo/
probe-2026-17-singapore.json; trimmed into tests/fixtures/weather/open_meteo-sample.json). The free API is for
non-commercial use: commercial use needs Open-Meteo's paid plan (an open item for the owner, docs/weather.md).

The response is parallel arrays per block: {"hourly": {"time": [...], "<variable>": [...]}, "daily": {...}}, with
`timeformat=unixtime` so every time is a UTC epoch second. `timezone=auto` makes the daily rows the venue's local
calendar days (each `time` is local midnight, as a UTC epoch); hourly rows are unaffected. Units are the API's
defaults, which the response states in hourly_units / daily_units: C, %, mm, km/h, WMO weather code.

    fetch_raw(lat, lon, days=16)                           {"hourly": body, "daily": body, ...}, the envelope
    parse(raw, event_key, lat, lon, issued_utc, sessions)  the forecast frame; a field the response lacks raises
                                                           OpenMeteoFieldError naming it (never NaN columns)
    probe(lat, lon, out_path, days=16)                     fetch_raw, written untouched to out_path; returns
                                                           (path, missing fields)
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.weather import schema as S

SOURCE = "open_meteo"
URL = "https://api.open-meteo.com/v1/forecast"
MAX_DAYS = 16                      # the forecast endpoint's horizon

H_TIME = "time"
H_TEMP = "temperature_2m"
H_PRECIP_CHANCE = "precipitation_probability"
H_PRECIP = "precipitation"
H_HUMIDITY = "relative_humidity_2m"
H_WIND = "wind_speed_10m"
H_CODE = "weather_code"

D_TIME = "time"
D_TMAX = "temperature_2m_max"
D_TMIN = "temperature_2m_min"
D_PRECIP_CHANCE = "precipitation_probability_max"
D_PRECIP = "precipitation_sum"
D_HUMIDITY = "relative_humidity_2m_mean"
D_WIND = "wind_speed_10m_max"
D_CODE = "weather_code"

HOURLY_FIELDS = (H_TIME, H_TEMP, H_PRECIP_CHANCE, H_PRECIP, H_HUMIDITY, H_WIND, H_CODE)
DAILY_FIELDS = (D_TIME, D_TMAX, D_TMIN, D_PRECIP_CHANCE, D_PRECIP, D_HUMIDITY, D_WIND, D_CODE)
NAMED_SESSIONS = ("race", "qual", "sprint")

# WMO weather interpretation codes, as Open-Meteo's documentation lists them
WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle", 56: "Light freezing drizzle",
    57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}


class OpenMeteoFieldError(ValueError):
    """A field the response doesn't have (or doesn't hold as a list of the block's length). The message names it."""


def params(lat, lon, days=MAX_DAYS):
    return {"latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY_FIELDS[1:]),
            "daily": ",".join(DAILY_FIELDS[1:]), "timezone": "auto", "timeformat": "unixtime",
            "forecast_days": int(days)}


def fetch_raw(lat, lon, days=MAX_DAYS):
    """The hourly and daily forecasts for (lat, lon), `days` days ahead (1 to 16), as the envelope parse() reads:
    {"provider", "fetched_utc", "lat", "lon", "hourly": body, "daily": body, "meta": the response's other fields}.
    The bodies are the response's blocks untouched. A failed request raises."""
    import requests

    from racinglines.sources import http
    if not 1 <= int(days) <= MAX_DAYS:
        raise ValueError(f"days must be 1 to {MAX_DAYS}, got {days}")
    fetched = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with requests.Session() as s:
        r = http.get(s, URL, params=params(lat, lon, days), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"open_meteo: HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    return {"provider": SOURCE, "fetched_utc": fetched, "lat": lat, "lon": lon,
            "hourly": body.get("hourly"), "daily": body.get("daily"),
            "meta": {k: v for k, v in body.items() if k not in ("hourly", "daily")}}


def _field(body, name, where, n=None):
    if not isinstance(body, dict):
        raise OpenMeteoFieldError(f"open_meteo {where}: expected a JSON object, got {type(body).__name__}")
    if name not in body:
        raise OpenMeteoFieldError(f"open_meteo {where}: no field {name!r}; the response has {sorted(body)}")
    v = body[name]
    if not isinstance(v, list) or (n is not None and len(v) != n):
        got = len(v) if isinstance(v, list) else type(v).__name__
        raise OpenMeteoFieldError(f"open_meteo {where}: field {name!r} should be a list of {n}, got {got}")
    return v


def _num(values):
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _utc(epochs):
    return pd.to_datetime(_num(epochs), unit="s")       # naive UTC


def _conditions(codes):
    return [None if c is None else WMO.get(int(c), f"WMO {int(c)}") for c in codes]


def _hourly(body):
    n = len(_field(body, H_TIME, "hourly"))
    f = lambda name: _field(body, name, "hourly", n)     # noqa: E731
    return pd.DataFrame({
        "session": "hour", "valid_utc": _utc(f(H_TIME)), "precip_prob": _num(f(H_PRECIP_CHANCE)) / 100,
        "precip_mm": _num(f(H_PRECIP)), "temp_c": _num(f(H_TEMP)), "humidity": _num(f(H_HUMIDITY)),
        "wind_kph": _num(f(H_WIND)), "condition": _conditions(f(H_CODE))})


def _daily(body):
    n = len(_field(body, D_TIME, "daily"))
    f = lambda name: _field(body, name, "daily", n)      # noqa: E731
    return pd.DataFrame({
        "session": "day", "valid_utc": _utc(f(D_TIME)), "precip_prob": _num(f(D_PRECIP_CHANCE)) / 100,
        "precip_mm": _num(f(D_PRECIP)), "temp_c": (_num(f(D_TMAX)) + _num(f(D_TMIN))) / 2,
        "humidity": _num(f(D_HUMIDITY)), "wind_kph": _num(f(D_WIND)), "condition": _conditions(f(D_CODE))})


def _naive_utc(t):
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def parse(raw_json, event_key, lat, lon, issued_utc, sessions=None):
    """The forecast frame (schema.COLUMNS) from fetch_raw's envelope: one "day" row per daily entry (the venue's local
    day; temp_c the mean of its max and min) and one row per hourly entry, labelled with the session whose window holds
    it (`sessions`: {"race" | "qual" | "sprint": (start, end)}, naive UTC, end exclusive) or "hour" outside every
    window. Validated before it is returned. A field the response doesn't have raises OpenMeteoFieldError naming it."""
    if not isinstance(raw_json, dict) or "hourly" not in raw_json:
        raise OpenMeteoFieldError("open_meteo: expected fetch_raw's envelope with an 'hourly' body")
    h = _hourly(raw_json["hourly"])
    for name, (start, end) in (sessions or {}).items():
        if name not in NAMED_SESSIONS:
            raise ValueError(f"unknown session {name!r}; expected one of {NAMED_SESSIONS}")
        inside = (h["valid_utc"] >= _naive_utc(start)) & (h["valid_utc"] < _naive_utc(end))
        h.loc[inside & (h["session"] == "hour"), "session"] = name
    parts = [h] if raw_json.get("daily") is None else [_daily(raw_json["daily"]), h]
    df = pd.concat(parts, ignore_index=True)
    df = df.assign(event_key=str(event_key), issued_utc=_naive_utc(issued_utc), source=SOURCE, lat=float(lat),
                   lon=float(lon))
    df["valid_utc"] = df["valid_utc"].astype("datetime64[ns]")
    df["issued_utc"] = df["issued_utc"].astype("datetime64[ns]")
    df["condition"] = df["condition"].astype(object)
    return S.validate(df[list(S.COLUMNS)].sort_values(["valid_utc", "session"], ignore_index=True))


def missing_fields(raw_json):
    """Every requested field the envelope's bodies don't have, as "hourly.<name>" etc. [] when all are there."""
    out = []
    for block, fields in (("hourly", HOURLY_FIELDS), ("daily", DAILY_FIELDS)):
        body = (raw_json or {}).get(block)
        if isinstance(body, dict):
            out += [f"{block}.{f}" for f in fields if f not in body]
        else:
            out.append(f"{block} (no body)")
    return out


def probe(lat, lon, out_path, days=MAX_DAYS):
    """Fetch the forecast for (lat, lon) and write the envelope untouched to out_path. Returns (path, missing),
    `missing` the requested fields the response doesn't have (missing_fields)."""
    raw = fetch_raw(lat, lon, days)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(raw, indent=1))
    return p, missing_fields(raw)


# ---------------------------------------------------------------------------
# Forecasts issued N days before a past window (the backtest's input; docs/weather.md#forecasts-of-past-races)
# ---------------------------------------------------------------------------

PREVIOUS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"      # leads 1-7, from about February 2024
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"   # lead 0 (each run's first hours)
# global models with archived runs (probe 2026-10-06: meteofrance_seamless and ukmo_seamless return none)
MODELS = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless", "jma_seamless", "cma_grapes_global",
          "bom_access_global")
LEADS = (1, 2, 3, 4, 5, 6, 7)
LEAD_VARS = ("precipitation", "temperature_2m", "wind_speed_10m", "weather_code")
RACE_H = 2                     # the race window, start to start + 2 h: precip_mm, temp_c, wind_kph, weather_code
VOTE_H = (-1, 3)               # the wet vote's window (weather/wet.py): hours around the start
LEAD_COLUMNS = ("race_id", "event_key", "venue_slug", "lat", "lon", "race_start_utc", "window_end_utc", "lead_days",
                "precip_prob", "precip_mm", "temp_c", "wind_kph", "weather_code", "max_hour_mm", "source")


def _get_json(url, query):
    import requests

    from racinglines.sources import http
    with requests.Session() as s:
        r = http.get(s, url, params=query, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"open_meteo {url}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def window_values(hourly, start, suffix, model):
    """One model's values for a race starting at `start` (naive UTC) from an hourly block keyed
    "<variable><suffix>_<model>": precip_mm (sum), temp_c (mean), wind_kph (max), weather_code (max) over the race
    window, and max_hour_mm (the wettest hour in the vote window). None when the model has no value in the window."""
    start = pd.Timestamp(start)
    t = pd.to_datetime(np.asarray(hourly["time"], dtype=float), unit="s")
    race = (t >= start) & (t < start + pd.Timedelta(hours=RACE_H))
    vote = (t >= start + pd.Timedelta(hours=VOTE_H[0])) & (t < start + pd.Timedelta(hours=VOTE_H[1]))

    def col(var):
        v = hourly.get(f"{var}{suffix}_{model}")
        return None if v is None else _num(v)
    p = col("precipitation")
    if p is None or np.isnan(p[race]).all():
        return None
    temp, wind, code = col("temperature_2m"), col("wind_speed_10m"), col("weather_code")
    agg = lambda a, f: None if a is None or np.isnan(a[race]).all() else float(f(a[race]))   # noqa: E731
    return dict(precip_mm=float(np.nansum(p[race])), temp_c=agg(temp, np.nanmean), wind_kph=agg(wind, np.nanmax),
                weather_code=agg(code, np.nanmax), max_hour_mm=float(np.nanmax(p[vote])))


def fetch_leads(lat, lon, start, leads=LEADS, models=MODELS):
    """The archived forecasts for the days around a race start: {"lead0": the Historical Forecast API body (lead 0),
    "previous": the Previous Runs API body (leads 1-7)}, untouched."""
    start = pd.Timestamp(start)
    lo, hi = start + pd.Timedelta(hours=VOTE_H[0]), start + pd.Timedelta(hours=VOTE_H[1])
    common = {"latitude": lat, "longitude": lon, "models": ",".join(models), "start_date": f"{lo:%Y-%m-%d}",
              "end_date": f"{hi:%Y-%m-%d}", "timezone": "GMT", "timeformat": "unixtime"}
    lead0 = _get_json(HISTORICAL_URL, {**common, "hourly": ",".join(LEAD_VARS)})
    prev = _get_json(PREVIOUS_URL, {**common, "hourly": ",".join(f"{v}_previous_day{n}" for v in LEAD_VARS
                                                                 for n in leads)})
    return {"lead0": lead0, "previous": prev}


def lead_rows(race, bodies, leads=LEADS, models=MODELS):
    """LEAD_COLUMNS rows for one race (a mapping with race_id, event_key, venue_slug, lat, lon, race_start_utc) from
    fetch_leads' bodies: one row per lead and model with a value (lead 0 from the Historical Forecast API). source is
    "open_meteo:<model>"; precip_prob is None (not archived at any lead)."""
    start = pd.Timestamp(race["race_start_utc"])
    out = []
    for lead in (0, *leads):
        body = bodies["lead0"] if lead == 0 else bodies["previous"]
        hourly = (body or {}).get("hourly")
        if not hourly:
            continue
        for m in models:
            v = window_values(hourly, start, "" if lead == 0 else f"_previous_day{lead}", m)
            if v is None:
                continue
            out.append(dict(race_id=race["race_id"], event_key=race["event_key"], venue_slug=race["venue_slug"],
                            lat=race["lat"], lon=race["lon"], race_start_utc=f"{start:%Y-%m-%d %H:%M:%S}",
                            window_end_utc=f"{start + pd.Timedelta(hours=RACE_H):%Y-%m-%d %H:%M:%S}",
                            lead_days=lead, precip_prob=None, **v, source=f"{SOURCE}:{m}"))
    return out
