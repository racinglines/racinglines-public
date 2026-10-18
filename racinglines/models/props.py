"""
F1 race props (F1 roadmap F1-3): safety car, red flag, rain and fastest lap, priced from the race history in
the database. Off by default: nothing prices them unless a live event lists their kinds ([live.markets] kinds
in sports/f1.toml or a launch spec).

    race_safety_car            a safety car is deployed in the race (track status 4 on any lap; a VSC alone is NO)
    race_red_flag              the race is red-flagged (track status 5 on any lap)
    race_rain                  rain falls during the race (any weather sample with rainfall: rounds.extra.weather)
    race_fastest_lap:<athlete> the driver sets the race's fastest lap (the quickest timed lap, deleted laps left out)

The three yes/no props are per-circuit rates shrunk to the field's rate: the field rate counts as PRIOR_N races
against a circuit's own record, over every race finished before the event starts. Rain uses the history only:
no weather forecast. Measured walk-forward on 2022-2026 (107 races, `--check`): a circuit's own record doesn't
beat the field rate within the noise (lighter shrinkage, e.g. 4, was worse), so PRIOR_N is a heavy 16.

Wet or dry (decision log 2026-10-06, provisional): a race is `wet` when any weather sample had rain (rain_share > 0,
the `rain` rule). `rate_given` is a prop's rate among a circuit's wet (or dry) races, shrunk to the field's wet (or
dry) rate, itself shrunk to the overall field rate; `rate_wx` mixes the two with a probability of rain:
p_wet * rate_given(wet) + (1 - p_wet) * rate_given(dry), and with no p_wet it uses the circuit's own shrunk wet
share (climatology). Red flags are far likelier in the wet (2020-2026: 10 of 35 wet races, 12 of 112 dry). The live
book prices with `rate()` unless a probability of rain is passed: `market_set(..., p_wet=...)` prices race_red_flag
with `rate_wx` and race_rain at p_wet itself (the live book passes the saved wet vote of racinglines/weather/wet.py,
shrunk to climatology; its 5-day backtest beat the circuit history for rain, decision log 2026-10-06), and the
safety car stays on `rate()` because conditioning it was worse in the walk-forward (WX_KINDS). `check()` scores both: `climatology` (p_wet = the circuit's past wet share) and
`wet_oracle` (p_wet = the race's realised wet flag, the ceiling a perfect forecast would reach).

Weather-aware (-WX) variants. Which kinds a probability of rain moves is schema, not code: each kind's `wx`
(markets/kinds.py; `wx = "..."` in markets/kinds.toml). "rate" kinds (WX_KINDS: race_red_flag) are priced with
rate_wx; "dnf" kinds (WX_DNF_KINDS: the classification and retirement kinds) from the run's dnf_prob times
wx_scale(hist, venue_id, p_wet), the circuit's retirement rate (dnf_rate = n_dnf / n per race) at that p_wet over the
same at its climatological wet share. `check(..., forecast=)` adds `climatology-WX` (rate_wx with a real forecast
issued days before, weather/wet.p_wet_series, instead of the circuit's wet share) and its paired Brier difference
against `climatology`; dnf_check.py's `model-WX` is the retirement twin.

Fastest lap: from history, how often the fastest lap goes to a driver finishing 1st, 2nd-3rd, 4th-10th, lower,
or not finishing (per driver in that bucket); a driver's price is his bucket probabilities from the stage run
(win, podium, top 10, DNF) times those rates, scaled so the field sums to 1.

    racinglines model props f1 2026-16                  # the prices for an event
    racinglines model props f1 --check --from 2022      # walk-forward calibration of the yes/no props
    racinglines model props f1 --check --history h.csv  # the same on a history CSV (history()'s columns), no database
    racinglines model props f1 --dnf-check dnf.csv      # the position simulation's DNF calibration (dnf_check.py)
    racinglines model props f1 --check --history h.csv --forecast leads.csv --lead 5     # adds climatology-WX
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.markets import kinds as K

PROP_KINDS = ("race_safety_car", "race_red_flag", "race_rain", "race_fastest_lap")
BINARY = {"race_safety_car": "sc", "race_red_flag": "red", "race_rain": "rain"}
LABEL = {"race_safety_car": "Safety car", "race_red_flag": "Red flag", "race_rain": "Rain",
         "race_fastest_lap": "Fastest lap"}
PRIOR_N = 16.0                # the field rate counts as this many races against a circuit's own record
FL_BUCKETS = ("p1", "p2_3", "p4_10", "p11", "dnf")
FL_SMOOTH = 0.5               # fastest laps added to each bucket (and spread over its driver slots) before rates


def history(conn, before=None):
    """Per finished race (with timed laps) before `before` (a date; None: all): race_id, event_key, venue_id,
    start (date), sc, red, rain, wet (bools), rain_share, fl_athlete, fl_bucket, n (result rows), n_ok, n_dnf,
    dnf_rate."""
    df = pd.read_sql(text("""
        WITH rl AS (
            SELECT ro.race_id, r.athlete_id, r.position, r.status, l.lap_time_ms, l.deleted, l.track_status
            FROM laps l JOIN results r ON r.id = l.result_id JOIN rounds ro ON ro.id = r.round_id
            WHERE ro.kind = 'race'),
        st AS (SELECT race_id, bool_or(coalesce(track_status, '') LIKE '%4%') AS sc,
                      bool_or(coalesce(track_status, '') LIKE '%5%') AS red FROM rl GROUP BY race_id),
        fl AS (SELECT DISTINCT ON (race_id) race_id, athlete_id AS fl_athlete, position AS fl_pos, status AS fl_status
               FROM rl WHERE lap_time_ms IS NOT NULL AND NOT coalesce(deleted, false)
               ORDER BY race_id, lap_time_ms, athlete_id),
        cl AS (SELECT ro.race_id, count(*) AS n, count(*) FILTER (WHERE r.status = 'OK') AS n_ok,
                      count(*) FILTER (WHERE r.status IN ('DNF', 'DSQ')) AS n_dnf,
                      max((ro.extra->'weather'->>'rain_share')::float) AS rain_share
               FROM results r JOIN rounds ro ON ro.id = r.round_id WHERE ro.kind = 'race' GROUP BY ro.race_id)
        SELECT st.race_id, e.source_key AS event_key, e.venue_id, e.start_date AS start, st.sc, st.red,
               coalesce(cl.rain_share, 0) AS rain_share, fl.fl_athlete, fl.fl_pos, fl.fl_status, cl.n, cl.n_ok, cl.n_dnf
        FROM st JOIN races ra ON ra.id = st.race_id JOIN events e ON e.id = ra.event_id
        JOIN cl ON cl.race_id = st.race_id LEFT JOIN fl ON fl.race_id = st.race_id
        WHERE e.source = 'f1timing' AND (CAST(:b AS date) IS NULL OR e.start_date < CAST(:b AS date))
        ORDER BY e.start_date, st.race_id"""), conn, params=dict(b=None if before is None else str(pd.Timestamp(before).date())))
    df = prepare(df)
    df["fl_bucket"] = [_bucket(p, s) for p, s in zip(df["fl_pos"], df["fl_status"])]
    return df


def prepare(df):
    """history()'s derived columns on a frame with at least start, venue_id, sc, red and rain_share (e.g. a CSV
    export): start as a date, rain and wet (rain_share > 0; two names, one rule: `rain` is the prop, `wet` the
    condition), sc and red as bools, and with n_dnf, dnf_rate = n_dnf / n (the race's share of retirements; n is the
    race's result rows, n_ok + n_dnf when the frame has no n)."""
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"]).dt.date
    df["rain_share"] = pd.to_numeric(df["rain_share"], errors="coerce").fillna(0.0)
    df["rain"] = df["rain_share"] > 0
    df["wet"] = df["rain"]
    for c in ("sc", "red"):
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.lower().isin(("true", "t", "1"))
        df[c] = df[c].astype(bool)
    if "n_dnf" in df:
        n = pd.to_numeric(df["n"] if "n" in df else df["n_ok"] + df["n_dnf"], errors="coerce")
        df["dnf_rate"] = pd.to_numeric(df["n_dnf"], errors="coerce") / n.clip(lower=1)
    return df


def _bucket(pos, status):
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return None
    if status != "OK":
        return "dnf"
    return "p1" if pos == 1 else "p2_3" if pos <= 3 else "p4_10" if pos <= 10 else "p11"


def rate(hist, venue_id, col, prior_n=PRIOR_N):
    """P(YES) for a yes/no prop at a venue: its own record shrunk to the field rate (None with no history). `col` may
    be bool or a per-race share (dnf_rate): the same mean, shrunk the same way."""
    if hist.empty:
        return None
    field = float(hist[col].mean())
    v = hist.loc[hist["venue_id"] == venue_id, col]
    return float((v.sum() + prior_n * field) / (len(v) + prior_n))


def rate_given(hist, venue_id, col, wet, prior_n=PRIOR_N):
    """P(YES) for a yes/no prop at a venue given the race is wet (True) or dry (False): the circuit's rate among
    its wet (dry) races, shrunk to the field's wet (dry) rate, which is itself shrunk to the overall field rate
    with the same prior_n (two levels). The field's wet sample is small (35 of 147 races 2020-2026, 25 of 108 from
    2022), so its rate leans on the overall one; a circuit has 0-4 wet races, so its own wet record barely moves
    the price. None with no history."""
    if hist.empty:
        return None
    field = float(hist[col].mean())
    sub = hist[hist["wet"].astype(bool) == bool(wet)]
    field_c = float((sub[col].sum() + prior_n * field) / (len(sub) + prior_n))
    v = sub.loc[sub["venue_id"] == venue_id, col]
    return float((v.sum() + prior_n * field_c) / (len(v) + prior_n))


def rate_wx(hist, venue_id, col, p_wet=None, prior_n=PRIOR_N):
    """P(YES) given a probability of rain: p_wet * rate_given(wet) + (1 - p_wet) * rate_given(dry). With p_wet
    None, the circuit's own wet share shrunk to the field's (climatology: rate(hist, venue_id, "wet")), so it always
    prices. None with no history."""
    if hist.empty:
        return None
    if p_wet is None:
        p_wet = rate(hist, venue_id, "wet", prior_n)
    p_wet = min(max(float(p_wet), 0.0), 1.0)
    return p_wet * rate_given(hist, venue_id, col, True, prior_n) + \
        (1 - p_wet) * rate_given(hist, venue_id, col, False, prior_n)


def fl_rates(hist):
    """{bucket: P(fastest lap) per driver in that bucket} from history (smoothed)."""
    h = hist.dropna(subset=["fl_bucket"])
    slots = dict(p1=len(h), p2_3=2.0 * len(h), p4_10=7.0 * len(h),
                 p11=float((h["n_ok"] - 10).clip(lower=0).sum()), dnf=float(h["n_dnf"].sum()))
    n = h["fl_bucket"].value_counts()
    base = len(h) / (sum(slots.values()) or 1.0)          # the per-slot rate over every bucket
    if base == 0:
        return {b: 1.0 for b in FL_BUCKETS}
    return {b: float((n.get(b, 0) + FL_SMOOTH) / (slots[b] + FL_SMOOTH / base)) for b in FL_BUCKETS}


def fl_probs(preds, rates):
    """P(driver sets the fastest lap) from each driver's win / podium / top-10 / DNF probabilities; sums to 1."""
    win = preds["win_prob"].fillna(0).to_numpy(float)
    pod = np.maximum(preds["podium_prob"].fillna(0).to_numpy(float), win)
    top = preds["top10_prob"] if "top10_prob" in preds else pd.Series(np.nan, index=preds.index)
    top = np.maximum(top.fillna(pd.Series(np.minimum(1, pod * 10 / 3), index=preds.index)).to_numpy(float), pod)
    dnf = preds["dnf_prob"] if "dnf_prob" in preds else pd.Series(0.0, index=preds.index)
    dnf = np.clip(dnf.fillna(0).to_numpy(float), 0, 1 - top)
    b = dict(p1=win, p2_3=pod - win, p4_10=top - pod, p11=np.clip(1 - top - dnf, 0, 1), dnf=dnf)
    raw = sum(b[k] * rates[k] for k in FL_BUCKETS)
    return raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1 / max(len(raw), 1))


def event_info(conn, event_key):
    """(venue_id, first day) of an F1 event."""
    row = conn.execute(text("SELECT venue_id, start_date FROM events WHERE source = 'f1timing' AND source_key = :k"),
                       dict(k=event_key)).first()
    return (None, None) if row is None else (row[0], row[1])


def run_preds(conn, run_id):
    """A stage run's per-driver probabilities for the fastest-lap and dnf prices: athlete_id, driver, team_key,
    win_prob, podium_prob, top10_prob, dnf_prob."""
    df = pd.read_sql(text("""SELECT athlete_id, win_prob, podium_prob, top10_prob, extra->>'driver' AS driver,
                                    extra->>'team_key' AS team_key, (extra->>'dnf_prob')::float AS dnf_prob
                             FROM race_predictions WHERE model_run_id = :r ORDER BY win_prob DESC, athlete_id"""),
                     conn, params=dict(r=run_id))
    return df


# Priced given a probability of rain when one is passed, read from the kinds' `wx` (markets/kinds.py): "rate" kinds
# with rate_wx (race_red_flag; the safety car is not: conditioning it on wet was worse in the walk-forward, Brier
# +0.0046, se 0.0027, decision log 2026-10-06, so it stays on rate(); check() still scores it), "dnf" kinds (the
# classification and retirement kinds of markets/kinds.toml) from dnf_prob scaled by wx_scale.
# race_rain is the probability of rain itself (RAIN_KIND): the forecast beat the circuit's history 5 days out (Brier
# 0.123 vs 0.167, 63 races 2024-2026, weather/wet.py backtest, decision log 2026-10-06)
WX_KINDS = K.wx_kinds("rate")
WX_DNF_KINDS = K.wx_kinds("dnf")
RAIN_KIND = "race_rain"
WX_CHECKED = ("race_safety_car", "race_red_flag")
DNF_SIMS = 20000              # draws for the dnf kinds priced from a run's dnf_prob (dnf_sims)


def wx_scale(hist, venue_id, p_wet, prior_n=PRIOR_N):
    """How much a probability of rain moves a venue's retirement rate: rate_wx(hist, venue_id, "dnf_rate", p_wet) over
    the same with the circuit's own wet share (rate_wx with p_wet None, climatology). 1 when p_wet is the circuit's
    shrunk wet share, i.e. when the forecast says no more than the history the model was fitted on; above 1 when a
    wetter race than usual is forecast at a circuit whose wet races retire more. 1.0 with no history, no dnf_rate or a
    zero rate. The `-WX` variants multiply dnf_prob by it (market_set, dnf_check)."""
    if hist.empty or "dnf_rate" not in hist or p_wet is None or pd.isna(p_wet):
        return 1.0
    base = rate_wx(hist, venue_id, "dnf_rate", None, prior_n)
    return float(rate_wx(hist, venue_id, "dnf_rate", p_wet, prior_n) / base) if base else 1.0


def dnf_sims(preds, n_sims=DNF_SIMS, seed=0):
    """An OutcomeSims of retirements only from a run's per-driver dnf_prob (preds: athlete_id, dnf_prob, and team_key
    for the team kinds): each driver retires independently with his dnf_prob (the run's teammate correlation is not
    kept). The classification order is the preds' row order (run_preds: by win_prob), so last_classified is the
    classified car lowest in it; the run's own simulations rank better. Prices the wx == "dnf" kinds via kinds.fair."""
    from racinglines.models.outcomes import OutcomeSims
    p = np.clip(preds["dnf_prob"].fillna(0).to_numpy(float), 0, 1)
    dnf = np.random.default_rng(seed).random((n_sims, len(p))) < p[None, :]
    rank = np.where(dnf, np.inf, np.arange(1, len(p) + 1, dtype=float)[None, :])
    groups = preds["team_key"].tolist() if "team_key" in preds else None
    return OutcomeSims(entrants=[int(a) for a in preds["athlete_id"]], rank=rank, finished=~dnf, groups=groups)


def dnf_markets(preds, kinds, lines=None, n_sims=DNF_SIMS, seed=0):
    """The wx == "dnf" kinds among `kinds`, priced by kinds.fair from dnf_sims(preds): one market per driver (key
    kind:athlete_id), per team (kind:team, params team) or per line of the field kinds (kind:line, params line; lines:
    {kind: [line, ...]}, none listed: not priced)."""
    want = [k for k in kinds if k in WX_DNF_KINDS]
    if not want or not len(preds):
        return []
    sims, out = dnf_sims(preds, n_sims, seed), []
    names = dict(zip(preds["athlete_id"].astype(int), preds["driver"] if "driver" in preds else preds["athlete_id"]))
    for kind in want:
        k = K.KINDS[kind]
        subject = k.spec["payoff"]["subject"]
        if subject == "driver":
            p = K.fair(kind, sims)
            out += [dict(key=f"{kind}:{a}", kind=kind, athlete_id=a, params=None, subject=names[a], fair=float(v))
                    for a, v in zip(sims.entrants, p)]
        elif subject == "team":
            out += [dict(key=f"{kind}:{t}", kind=kind, athlete_id=None, params=dict(team=t), subject=t, fair=v)
                    for t, v in K.fair(kind, sims).items()]
        else:
            out += [dict(key=f"{kind}:{x}", kind=kind, athlete_id=None, params=dict(line=x), subject=f"{k.label} {x}",
                         fair=K.fair(kind, sims, line=x)) for x in (lines or {}).get(kind, ())]
    return out


def market_set(preds, venue_id, hist, kinds=PROP_KINDS, prior_n=PRIOR_N, p_wet=None, lines=None):
    """The prop markets and their fair values (live_f1.market_set's shape): [dict(key, kind, athlete_id,
    params, subject, fair)]. p_wet (a probability of rain, None by default): race_rain is priced at p_wet itself
    (RAIN_KIND), and the kinds with a `wx` (markets/kinds.py) move with it: the "rate" kinds (race_red_flag) are
    priced with rate_wx instead of rate, and the "dnf" kinds (markets/kinds.toml's classification and retirement
    kinds, priced here only when listed in `kinds`, from the preds' dnf_prob: dnf_markets) with every dnf_prob times
    wx_scale, clipped to [0, 1]. No p_wet: rate() and the run's dnf_prob as they are."""
    out = []
    for kind, col in BINARY.items():
        if kind in kinds:
            if p_wet is not None and kind == RAIN_KIND:
                fair = min(max(float(p_wet), 0.0), 1.0)
            elif p_wet is not None and kind in WX_KINDS:
                fair = rate_wx(hist, venue_id, col, p_wet, prior_n)
            else:
                fair = rate(hist, venue_id, col, prior_n)
            out.append(dict(key=kind, kind=kind, athlete_id=None, params=None, subject=LABEL[kind], fair=fair))
    if "race_fastest_lap" in kinds and len(preds) and not hist.empty:
        p = fl_probs(preds, fl_rates(hist))
        for r, v in zip(preds.itertuples(), p):
            out.append(dict(key=f"race_fastest_lap:{int(r.athlete_id)}", kind="race_fastest_lap",
                            athlete_id=int(r.athlete_id), params=None, subject=r.driver, fair=float(v)))
    if any(k in WX_DNF_KINDS for k in kinds) and len(preds):
        scale = wx_scale(hist, venue_id, p_wet, prior_n)          # 1.0 with no p_wet
        out += dnf_markets(preds.assign(dnf_prob=(preds["dnf_prob"].fillna(0) * scale).clip(0, 1)), kinds, lines)
    return out


def markets(conn, event_key, run_id, kinds=PROP_KINDS, prior_n=PRIOR_N, p_wet=None, wet_vote=None, lines=None):
    """The event's prop markets from the race history before it and a stage run (fastest lap, the dnf kinds); p_wet,
    lines: see market_set (p_wet None, the default, prices with rate() and the run's dnf_prob). wet_vote (a saved
    weather.wet vote: {"wet_votes", "models"}) gives p_wet as the vote shrunk to the circuit's wet share
    (weather.wet.p_wet) when p_wet is None."""
    venue_id, start = event_info(conn, event_key)
    hist = history(conn, start)
    if p_wet is None and wet_vote and not hist.empty:
        from racinglines.weather import wet as WET
        p_wet = WET.p_wet(wet_vote["wet_votes"], wet_vote["models"], rate(hist, venue_id, "wet", prior_n))
    need = "race_fastest_lap" in kinds or any(k in WX_DNF_KINDS for k in kinds)
    preds = run_preds(conn, run_id) if need and run_id is not None else pd.DataFrame()
    return market_set(preds, venue_id, hist, kinds, prior_n, p_wet, lines)


def outcomes(conn, race_id, mkts):
    """{key: True / False} for the prop markets, from the race's laps and weather; {} until the race has laps."""
    h = history(conn)
    h = h[h["race_id"] == race_id]
    if h.empty:
        return {}
    r = h.iloc[0]
    out = {}
    for m in mkts:
        if m["kind"] in BINARY:
            out[m["key"]] = bool(r[BINARY[m["kind"]]])
        elif m["kind"] == "race_fastest_lap" and pd.notna(r["fl_athlete"]):
            out[m["key"]] = int(r["fl_athlete"]) == m["athlete_id"]
    return out


METHODS = ("circuit", "field", "coin", "climatology", "wet_oracle")
WX_METHODS = ("climatology-WX",)  # scored only when check() is given a forecast
EPS = 1e-4                    # log loss clip


def score(p, y):
    """Brier, log loss and the Brier's standard error of prices p against outcomes y (floats)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    q = np.clip(p, EPS, 1 - EPS)
    se2 = (q - y) ** 2
    return dict(brier=float(se2.mean()), log_loss=float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()),
                se=float(se2.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else float("nan"))


def paired(p_a, p_b, y):
    """Method a minus method b over the rows both price (NaN in either is left out): n, the mean Brier and log loss
    differences, and the Brier difference's standard error (negative: a is better)."""
    p_a, p_b, y = (np.asarray(x, float) for x in (p_a, p_b, y))
    m = ~(np.isnan(p_a) | np.isnan(p_b))
    if not m.any():
        return dict(n=0, brier=float("nan"), log_loss=float("nan"), se=float("nan"))
    a, b = score(p_a[m], y[m]), score(p_b[m], y[m])
    q_a, q_b = np.clip(p_a[m], EPS, 1 - EPS), np.clip(p_b[m], EPS, 1 - EPS)
    d = (q_a - y[m]) ** 2 - (q_b - y[m]) ** 2
    return dict(n=int(m.sum()), brier=a["brier"] - b["brier"], log_loss=a["log_loss"] - b["log_loss"],
                se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"))


def check(conn=None, start_year=2022, prior_n=PRIOR_N, history_df=None, forecast=None):
    """Walk-forward calibration of the yes/no props: each race from `start_year` priced from the races before
    it, against the field rate alone and a coin flip; race_safety_car and race_red_flag (WX_CHECKED) also given wet or dry
    (`climatology`: rate_wx with the circuit's past wet share; `wet_oracle`: rate_wx with the race's realised wet
    flag, a perfect forecast's ceiling; race_rain has neither, its conditioned price would be itself).
    history_df: a history frame (history()'s columns; fl_* not needed) instead of reading `conn`.
    forecast: P(wet) per race_id (weather/wet.p_wet_series at one lead): adds `climatology-WX` for WX_CHECKED,
    rate_wx with the forecast in place of the circuit's wet share (NaN for a race without one, scored on the races
    that have one), and a paired row per kind, method "climatology-WX - climatology": the mean Brier and log loss
    differences over the races both price, with the Brier difference's se (negative: the forecast helps).
    -> (per-race DataFrame, summary DataFrame: Brier and log loss per prop and method, with the mean YES rate and
    mean price; `races` counts the races each method priced)."""
    h = prepare(history_df) if history_df is not None else history(conn)
    h = h.sort_values(["start", "race_id"] if "race_id" in h else ["start"]).reset_index(drop=True)
    fc = {} if forecast is None else {k: float(v) for k, v in pd.Series(forecast).dropna().items()}
    rows = []
    for r in h[pd.to_datetime(h["start"]).dt.year >= start_year].itertuples():
        past = h[h["start"] < r.start]
        if len(past) < 10:
            continue
        p_fc = fc.get(getattr(r, "race_id", None))
        for kind, col in BINARY.items():
            y = bool(getattr(r, col))
            row = dict(event_key=r.event_key, kind=kind, y=y, wet=bool(r.wet),
                       circuit=rate(past, r.venue_id, col, prior_n), field=float(past[col].mean()), coin=0.5)
            if kind in WX_CHECKED:
                row.update(climatology=rate_wx(past, r.venue_id, col, None, prior_n),
                           wet_oracle=rate_wx(past, r.venue_id, col, float(r.wet), prior_n))
                if forecast is not None:
                    row["climatology-WX"] = np.nan if p_fc is None else rate_wx(past, r.venue_id, col, p_fc, prior_n)
            rows.append(row)
    per = pd.DataFrame(rows)
    out = []
    for kind, g in per.groupby("kind", sort=False):
        for meth in METHODS + WX_METHODS:
            if meth not in g or g[meth].isna().all():
                continue
            s = g[g[meth].notna()]
            y = s["y"].astype(float).to_numpy()
            p = np.clip(s[meth].to_numpy(float), EPS, 1 - EPS)
            out.append(dict(kind=kind, method=meth, races=len(s), yes_rate=y.mean(), mean_price=p.mean(), **score(p, y)))
        if "climatology-WX" in g and g["climatology-WX"].notna().any():
            d = paired(g["climatology-WX"], g["climatology"], g["y"].astype(float))
            both = g[g["climatology-WX"].notna()]
            out.append(dict(kind=kind, method="climatology-WX - climatology", races=d["n"], yes_rate=both["y"].mean(),
                            mean_price=float("nan"), brier=d["brier"], log_loss=d["log_loss"], se=d["se"]))
    return per, pd.DataFrame(out)
