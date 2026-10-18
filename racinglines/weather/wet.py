"""
Will the race be wet? A probability of rain for an F1 race window from Open-Meteo's global models, the same way live
and in the backtest (docs/weather.md#wet-race-forecast), and its walk-forward backtest against the race props.

The signal is a vote: each of MODELS (seven global models) votes "wet" when it forecasts at least WET_MM in any hour of
the race window (WINDOW_H around the start, UTC). p_wet shrinks the vote share to the circuit's climatology (its past
wet share, props.rate(hist, venue, "wet")):

    p_wet = (wet_votes + PRIOR_VOTES * climatology) / (models_with_data + PRIOR_VOTES)

Open-Meteo archives each model's earlier runs (the Previous Runs API, `<variable>_previous_dayN`: the run issued N
days before, leads 1-7, all seven models from about February 2024; before that JMA alone: leads 1-4 in 2020, 1-7 in
2021-23), but not its precipitation probability, so the vote is what can be backtested. `racinglines weather leads` pulls them per race
into open_meteo.LEAD_COLUMNS (committed: tests/fixtures/weather/open_meteo-leads-f1.csv). Live, `fetch_live` asks the
forecast API for the same models and `racinglines weather fetch --session race=...` saves the vote next to the issue
(wet-<issued>.json); the live book's props read it (pipelines/live_f1.py).

    venues()                               weather/venues.toml: DB venue slug -> name, lat, lon (Jolpica, 2026-10-06)
    vote(hourly, start)                    {"models", "wet_votes", "max_mm"} for the race window (forecast API block)
    p_wet(votes, models, climatology)      the shrunk probability
    fetch_live(lat, lon, start)            today's vote for a coming race, from the forecast API
    save_vote / load_vote(event_key, asof) one vote per issue, <data>/weather/<event_key>/wet-<issued>.json
    races(conn)                            props.history() plus each race's UTC start and venue slug
    backtest(hist, leads_df=None)          per race from February 2024: the realised wet / red / sc / n_dnf and the
                                           vote at each lead (from the leads CSV; default the committed fixture)
    votes(leads_df)                        the leads CSV's per-model rows as one vote per race and lead
    p_wet_series(hist, leads_df, lead)     walk-forward p_wet per race_id at one lead: the -WX checks' forecast
                                           (racinglines model props f1 --check / --dnf-check --forecast CSV --lead N)
    score(rows, hist, lead)                Brier and log loss of rain, red flag and safety car per method, and the
                                           DNF count (MAE, Poisson deviance) per method; the forecast-aware method is
                                           its base's name + "-WX" (wx_name: circuit-WX, climatology-WX)
    leads_table(rows, hist)                the rain and red-flag scores at every lead
    correlations(hist)                     wet vs dry: red flag, safety car and DNF rates, 2020 on

Settings (decision log 2026-10-06, provisional): WET_MM, WINDOW_H, PRIOR_VOTES, MODELS are fixed a priori, not tuned
on the backtest; score() reports PRIOR_VOTES's sensitivity for information only.
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.models import props as P
from racinglines.weather import open_meteo as OM
from racinglines.weather import schema as S

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MODELS = OM.MODELS            # seven global models (weather/open_meteo.py)
LEADS = OM.LEADS
WET_MM = 0.1                  # mm in one hour: a model's "rain" (Open-Meteo's resolution)
WINDOW_H = OM.VOTE_H          # hours around the start: a race runs about two, plus slack for timing errors
WX = "-WX"                    # owner, 2026-10-06: a weather-aware method, variant or strategy is its base name + WX
PRIOR_VOTES = 3.0             # climatology's weight, in model votes (the seven models are correlated)
FIRST_ARCHIVED = pd.Timestamp("2024-02-15")
LEADS_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "weather" / "open_meteo-leads-f1.csv"


def venues():
    """weather/venues.toml: {DB venue slug: {"name", "circuit", "lat", "lon", "source", ...}} (Jolpica, 2026-10-06)."""
    import tomllib
    return tomllib.loads((Path(__file__).with_name("venues.toml")).read_text())


def _window(start):
    start = pd.Timestamp(start)
    return start + timedelta(hours=WINDOW_H[0]), start + timedelta(hours=WINDOW_H[1])


def vote(hourly, start, models=MODELS):
    """The race window's vote from a forecast-API hourly block keyed "precipitation_<model>" (the API's multi-model
    naming): {"models": models with a value in the window, "wet_votes": those forecasting >= WET_MM in some hour,
    "max_mm": the mean over models of their wettest hour}."""
    lo, hi = _window(start)
    t = pd.to_datetime(np.asarray(hourly["time"], dtype=float), unit="s")
    inside = (t >= lo) & (t < hi)
    n, wet, mx = 0, 0, []
    for m in models:
        vals = hourly.get(f"precipitation_{m}")
        if vals is None:
            continue
        v = np.array([np.nan if x is None else float(x) for x in vals])[inside]
        v = v[~np.isnan(v)]
        if not len(v):
            continue
        n += 1
        wet += int(v.max() >= WET_MM)
        mx.append(float(v.max()))
    return {"models": n, "wet_votes": wet, "max_mm": float(np.mean(mx)) if mx else np.nan}


def wx_name(base):
    """The weather-aware name of a method, model variant or strategy: base + "-WX" (owner, 2026-10-06)."""
    return base if base.endswith(WX) else f"{base}{WX}"


def p_wet(wet_votes, models, climatology, prior=PRIOR_VOTES):
    """The vote share shrunk to the circuit's climatology; climatology alone with no model."""
    return float((wet_votes + prior * climatology) / (models + prior))


def fetch_live(lat, lon, start, models=MODELS):
    """Today's vote for a race starting at `start` (naive UTC, within the forecast horizon) from the forecast API, and
    the lead in hours. Raises when no model covers the window."""
    body = OM._get_json(FORECAST_URL, {"latitude": lat, "longitude": lon, "hourly": "precipitation",
                                       "models": ",".join(models), "forecast_days": 16, "timezone": "GMT",
                                       "timeformat": "unixtime"})
    out = vote(body["hourly"], start, models)
    if not out["models"]:
        raise RuntimeError(f"open_meteo: no model covers the race window from {start}")
    return out


def races(conn):
    """props.history() with each race's UTC start (rounds.extra.session_date, the FastF1 session start in UTC) and the
    venue's slug."""
    from sqlalchemy import text
    h = P.history(conn)
    x = pd.read_sql(text("""
        SELECT ra.id AS race_id, v.slug, ro.extra->>'session_date' AS start_utc
        FROM rounds ro JOIN races ra ON ra.id = ro.race_id JOIN events e ON e.id = ra.event_id
        JOIN venues v ON v.id = e.venue_id WHERE ro.kind = 'race' AND e.source = 'f1timing'"""), conn)
    return h.merge(x, on="race_id", how="left")


def backtest(hist, leads_df=None, leads=LEADS, since=FIRST_ARCHIVED):
    """One row per race from `since` in both `hist` (races()) and `leads_df` (open_meteo.lead_rows' CSV; default
    the committed fixture): event_key, slug, start_utc, wet, red, sc, n_dnf, and per lead N: models_N, votes_N,
    max_mm_N (the vote of the runs issued N days before the race: models with max_hour_mm >= WET_MM)."""
    v = votes(pd.read_csv(LEADS_FIXTURE) if leads_df is None else leads_df, leads)
    h = P.prepare(hist)
    h = h[pd.to_datetime(h["start"]) >= since]
    rows = []
    for r in h.itertuples():
        vr = v[v["race_id"] == r.race_id]
        if vr.empty:
            continue
        row = dict(race_id=r.race_id, event_key=r.event_key, slug=r.slug, start_utc=pd.Timestamp(r.start_utc),
                   wet=bool(r.wet), red=bool(r.red), sc=bool(r.sc), n_dnf=int(r.n_dnf))
        for n in leads:
            x = vr[vr["lead_days"] == n]
            row.update({f"models_{n}": int(x["models"].sum()), f"votes_{n}": int(x["votes"].sum()),
                        f"max_mm_{n}": float(x["max_mm"].iloc[0]) if len(x) else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def votes(leads_df, leads=LEADS):
    """The leads CSV's per-model rows as one vote per race and lead: race_id, lead_days, models (rows), votes (models
    with max_hour_mm >= WET_MM), max_mm (their mean wettest hour). backtest() and p_wet_series() share it."""
    ld = leads_df[leads_df["lead_days"].isin(leads)]
    g = ld.assign(wet=ld["max_hour_mm"] >= WET_MM).groupby(["race_id", "lead_days"])
    return pd.DataFrame({"models": g.size(), "votes": g["wet"].sum(), "max_mm": g["max_hour_mm"].mean()}).reset_index()


def p_wet_series(hist, leads_df, lead, prior=PRIOR_VOTES, prior_n=P.PRIOR_N, since=FIRST_ARCHIVED):
    """p_wet per race_id at `lead` days, walk-forward: the race's vote (votes(), as backtest() counts it) shrunk to
    climatology = props.rate(past, venue_id, "wet", prior_n), past = `hist`'s races that started before it (hist:
    props.history()'s columns, through props.prepare). Races from `since` (default FIRST_ARCHIVED, as backtest(): the
    seven-model archive; None: every race, JMA alone before 2024) in both frames, with a vote and a past race. This is
    the forecast the -WX checks take: props.check(..., forecast=) and dnf_check.check(..., forecast=).
    -> Series indexed by race_id, named p_wet."""
    v = votes(leads_df, (int(lead),)).set_index("race_id")
    h = P.prepare(hist)
    scored = h if since is None else h[pd.to_datetime(h["start"]) >= pd.Timestamp(since)]
    out = {}
    for r in scored[scored["race_id"].isin(v.index)].itertuples():
        x = v.loc[r.race_id]
        clim = P.rate(h[h["start"] < r.start], r.venue_id, "wet", prior_n)
        if clim is not None and int(x["models"]):
            out[r.race_id] = p_wet(int(x["votes"]), int(x["models"]), clim, prior)
    return pd.Series(out, dtype=float, name="p_wet")


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score(rows, hist, lead=5, prior=PRIOR_VOTES):
    """Walk-forward: each race priced from the races before it (all of `hist`, 2020 on) and its vote at `lead` days.
    -> (per-race frame, summary frame). Methods per prop: rain: circuit (its own past rate) and circuit-WX (p_wet);
    red and sc: climatology and climatology-WX (props.rate_wx with the circuit's wet share or with p_wet) and
    wet_oracle (the realised flag, the ceiling); dnf (count): field (the past mean), and climatology /
    climatology-WX / wet_oracle (the past wet and dry means mixed by the same p_wet)."""
    hist = P.prepare(hist)
    per = []
    for r in rows.itertuples():
        m, v = getattr(r, f"models_{lead}"), getattr(r, f"votes_{lead}")
        if not m:
            continue
        past = hist[pd.to_datetime(hist["start"]) < r.start_utc.normalize()]
        venue = hist.loc[hist["race_id"] == r.race_id, "venue_id"].iloc[0]
        clim = P.rate(past, venue, "wet")
        pw = p_wet(v, m, clim, prior)
        wet_dnf, dry_dnf = past.loc[past["wet"], "n_dnf"].mean(), past.loc[~past["wet"], "n_dnf"].mean()
        mix = lambda p: p * wet_dnf + (1 - p) * dry_dnf      # noqa: E731
        per.append(dict(
            event_key=r.event_key, slug=r.slug, wet=r.wet, red=r.red, sc=r.sc, n_dnf=r.n_dnf, votes=v, models=m,
            climatology=clim, p_wet=pw,
            rain_circuit=P.rate(past, venue, "rain"), rain_forecast=pw,
            red_climatology=P.rate_wx(past, venue, "red"), red_forecast=P.rate_wx(past, venue, "red", pw),
            red_wet_oracle=P.rate_wx(past, venue, "red", float(r.wet)),
            sc_climatology=P.rate_wx(past, venue, "sc"), sc_forecast=P.rate_wx(past, venue, "sc", pw),
            sc_wet_oracle=P.rate_wx(past, venue, "sc", float(r.wet)),
            dnf_field=past["n_dnf"].mean(), dnf_climatology=mix(clim), dnf_forecast=mix(pw),
            dnf_wet_oracle=mix(float(r.wet))))
    per = pd.DataFrame(per)
    out = []
    for prop, col, meths in (("rain", "rain", ("circuit", "forecast")),
                             ("red_flag", "red", ("climatology", "forecast", "wet_oracle")),
                             ("safety_car", "sc", ("climatology", "forecast", "wet_oracle"))):
        y = per["wet" if col == "rain" else col].astype(float).to_numpy()
        for m in meths:
            p = per[f"{col}_{m}"].to_numpy(float)
            out.append(dict(prop=prop, method=wx_name(meths[0]) if m == "forecast" else m, races=len(per), yes_rate=y.mean(), mean_price=p.mean(),
                            brier=float(((p - y) ** 2).mean()), log_loss=_ll(p, y)))
    y = per["n_dnf"].to_numpy(float)
    for m in ("field", "climatology", "forecast", "wet_oracle"):
        mu = np.clip(per[f"dnf_{m}"].to_numpy(float), 1e-6, None)
        dev = 2 * np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - 2 * (y - mu)
        out.append(dict(prop="dnf_count", method=wx_name("climatology") if m == "forecast" else m, races=len(per), yes_rate=y.mean(), mean_price=mu.mean(),
                        mae=float(np.abs(y - mu).mean()), poisson_dev=float(dev.mean())))
    return per, pd.DataFrame(out)


def leads_table(rows, hist, leads=LEADS, prior=PRIOR_VOTES):
    """score()'s rain and red-flag forecast rows at every lead, to see how the signal fades with lead time."""
    out = []
    for n in leads:
        _, s = score(rows, hist, n, prior)
        s = s.set_index(["prop", "method"])
        out.append(dict(lead_days=n, rain_brier=s.loc[("rain", wx_name("circuit")), "brier"],
                        rain_brier_circuit=s.loc[("rain", "circuit"), "brier"],
                        red_ll=s.loc[("red_flag", wx_name("climatology")), "log_loss"],
                        red_ll_clim=s.loc[("red_flag", "climatology"), "log_loss"]))
    return pd.DataFrame(out)


def correlations(hist, since="2020-01-01"):
    """Wet vs dry races from `since`: count, red-flag rate, safety-car rate, mean DNFs (DNF + DSQ) per race."""
    h = P.prepare(hist)
    h = h[pd.to_datetime(h["start"]) >= since]
    g = h.groupby(h["wet"].map({True: "wet", False: "dry"}))
    return pd.DataFrame({"races": g.size(), "red_flag": g["red"].mean(), "safety_car": g["sc"].mean(),
                         "dnf_per_race": g["n_dnf"].mean()})


def save_vote(event_key, race_start, lat, lon, v, issued):
    """Write one vote as <data>/weather/<event_key>/wet-<issued>.json (naive UTC times); returns the path."""
    issued = pd.Timestamp(issued)
    p = S.root() / "weather" / str(event_key) / f"wet-{issued:{S.STAMP}}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "event_key": str(event_key), "race_start_utc": f"{pd.Timestamp(race_start):%Y-%m-%dT%H:%M}",
        "issued_utc": f"{issued:%Y-%m-%dT%H:%M:%S}", "lat": lat, "lon": lon, "models": v["models"],
        "wet_votes": v["wet_votes"], "max_mm": None if np.isnan(v["max_mm"]) else round(v["max_mm"], 3),
        "lead_hours": round((pd.Timestamp(race_start) - issued).total_seconds() / 3600, 1),
        "settings": {"model_list": list(MODELS), "wet_mm": WET_MM, "window_h": list(WINDOW_H)}}, indent=1))
    return p


def load_vote(event_key, asof=None):
    """The event's latest saved vote issued at or before `asof` (naive UTC; None: now), or None."""
    asof = pd.Timestamp.now(tz="UTC").tz_localize(None) if asof is None else pd.Timestamp(asof)
    d = S.root() / "weather" / str(event_key)
    best = None
    for p in sorted(d.glob("wet-*.json")) if d.exists() else ():
        v = json.loads(p.read_text())
        if pd.Timestamp(v["issued_utc"]) <= asof and (best is None or v["issued_utc"] > best["issued_utc"]):
            best = v
    return best
