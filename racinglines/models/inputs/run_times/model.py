"""
Timed-runs model (UCI downhill): per-rider pace, noise and incidents on log run
times, and a Monte Carlo of a race weekend (qualifying rounds -> Final), plus the
season standings built from simulated weekends.
"""

import numpy as np
import pandas as pd
from racinglines import sports
from racinglines.core.stats import ranks

RNG = np.random.default_rng(42)
_SPORT = sports.declaring("run_times", None)          # the sport whose [model] inputs declare run_times
SCHEMA = sports.load(_SPORT) if _SPORT else {"points": {}, "rounds": {}}   # none installed: empty (public release, F1 only)

# ---------------------------------------------------------------------------
# 7. Season simulation: full race weekend (Q1 -> Q2 -> Final) Monte Carlo
#    + championship points / standings
# ---------------------------------------------------------------------------
#
# Per-run model, on log finish time y of rider i in run r (one run = one
# event+round):   y = run_effect[r] + mu[i] + u[i, event] + eps
#   mu      rider pace (recency-weighted, shrunk toward the field median)
#   u       rider x track effect, shared by all of a rider's runs that weekend
#   eps     run-to-run noise
# plus a per-rider "incident" rate (DNF/DSQ, or a finished run more than
# INCIDENT_THRESHOLD slower than expected); incident runs are simulated as a
# DNF or as a time loss resampled from the observed incident losses.
# Weekend format observed in every 2026 round: Q1 top 20 go to the final,
# everyone else rides Q2, Q2 top 10 fill the final (30 riders).
#
# Training can use every season and category (older elite seasons with their
# qual -> semi -> final format, juniors' qual -> final), but predictions,
# points and backtests are only for the target season + category. Each run
# is weighted by round type, category and age (half-life in days). Runs are
# per event+category, so junior and elite paces are tied together through
# riders who raced both (juniors moving up).

# Sport-specific values come from sports/mtb_dh.toml.
# PLACEHOLDER POINTS: approximate UCI DHI World Cup scales, NOT the official
# tables. Replace them in the schema; everything downstream reads these.
FINAL_POINTS = list(SCHEMA["points"].get("final", ()))
QUAL_POINTS = list(SCHEMA["points"].get("qual", ()))
# Round QUAL_POINTS are paid on, by weekend format (placeholder guesses too)
QUAL_POINTS_ROUND = dict(SCHEMA["points"].get("qual_round", {}))

# 2026 format, used for unraced rounds. Past events use their own format,
# inferred from their results by event_format().
DEFAULT_FORMAT = dict(SCHEMA["rounds"].get("default_format", {}))
# per-category formats for unraced rounds, used only with unraced="last" when a category has no completed event
# this season (sports/mtb_dh.toml [rounds.category_format]; none set: the official field sizes are the owner's)
CATEGORY_FORMAT = {k: dict(v) for k, v in SCHEMA["rounds"].get("category_format", {}).items()}
RACE_ROUNDS = tuple(SCHEMA["rounds"].get("race", ()))
RUN_WEIGHTS = dict(SCHEMA["rounds"].get("run_weights", {}))
# Defaults from the 43-round tuning sweep (docs/model.md, Calibration; owner's OK 2026-09-28).
# Before: MJ 0.5, half-life 120, prior_n 1.5.
CATEGORY_WEIGHTS = {"ME": 1.0, "MJ": 0.25}  # training weight per category (others: 0.5)
HALF_LIFE_DAYS = 240.0
PRIOR_N = 0.5                 # shrinkage of rider pace toward the field median (in runs' weight)
INCIDENT_THRESHOLD = 0.04     # finished >4% slower than expected = incident
EPS_DF = None                 # run noise eps: None = normal; a number = Student-t with these degrees of
                              # freedom, scaled to the same sd (heavier tails; docs/todo.md, Model)


def select_target(raw, season=None, category="ME"):
    """Rows of the season + category being predicted (season defaults to the
    latest year in the data)."""
    year = raw["event_date"].astype(str).str[:4]
    season = str(season or year.max())
    return raw[(year == season) & (raw["category"] == category)]


def points_for(pos, table):
    """Vectorized f(rank) -> points; NaN/inf/out-of-table ranks score 0."""
    pos = np.asarray(pos, dtype=float)
    out = np.zeros(pos.shape)
    ok = np.isfinite(pos) & (pos >= 1) & (pos <= len(table))
    out[ok] = np.asarray(table, dtype=float)[pos[ok].astype(int) - 1]
    return out


def event_order(raw):
    return raw.groupby("event_id")["event_date"].min().sort_values().index.tolist()


def event_format(raw, event_id):
    """Infer an event's weekend format and field sizes from its own results
    (the format is published before the race, so this is fair in backtests).
      q1q2   Q1 top k go straight to the final, the rest ride Q2 (2025-26 elite)
      semi   qualifier -> semi-final -> final                (2023-24 elite)
      single qualifier -> final                   (juniors, elite up to 2022)"""
    ev = raw[(raw["event_id"] == event_id) & (raw["sector_id"] == "FINISH")]
    entered = lambda r: set(ev.loc[ev["round"] == r, "rider_id"])
    final = entered("final")
    if entered("qual1"):
        q2 = entered("qual2")
        return dict(kind="q1q2", q1_to_final=len(final - q2), q2_to_final=len(final & q2))
    if entered("semi"):
        return dict(kind="semi", to_semi=len(entered("semi")), to_final=len(final))
    return dict(kind="single", to_final=len(final))


def unraced_format(target, done, mode="default"):
    """The format an unraced round is simulated in. mode "default" (today): DEFAULT_FORMAT, the 2026 elite men's,
    for every category. "last": the target's latest completed event's own format (event_format; the target is
    one category and season, so women and juniors get their own field sizes), else CATEGORY_FORMAT for the
    category, else DEFAULT_FORMAT."""
    if mode == "default":
        return DEFAULT_FORMAT
    if mode != "last":
        raise ValueError(f"unraced format mode must be 'default' or 'last', not {mode!r}")
    events = [e for e in event_order(target) if e in set(done)]
    if events:
        return event_format(target, events[-1])
    cat = target["category"].iloc[0] if "category" in target and len(target) else None
    return CATEGORY_FORMAT.get(cat, DEFAULT_FORMAT)


def actual_event_points(raw):
    """Championship points actually scored, one row per rider per event."""
    fin = raw[(raw["sector_id"] == "FINISH") & (raw["status"] == "OK")]
    qual_round = {e: QUAL_POINTS_ROUND[event_format(raw, e)["kind"]] for e in fin["event_id"].unique()}
    fin_q = fin[fin["round"] == fin["event_id"].map(qual_round)]
    q = fin_q.set_index(["event_id", "rider_id"])["rank_at_split"]
    f = fin[fin["round"] == "final"].set_index(["event_id", "rider_id"])["rank_at_split"]
    idx = q.index.union(f.index)
    out = pd.DataFrame(index=idx)
    out["qual_rank"] = q.reindex(idx)
    out["final_rank"] = f.reindex(idx)
    out["qual_pts"] = points_for(out["qual_rank"], QUAL_POINTS)
    out["final_pts"] = points_for(out["final_rank"], FINAL_POINTS)
    out["points"] = out["qual_pts"] + out["final_pts"]
    return out.reset_index()


def event_starters(raw, event_id):
    """Riders who actually took the start of the first qualifier (not DNS)."""
    ev = raw[(raw["event_id"] == event_id) & (raw["sector_id"] == "FINISH")]
    first = "qual1" if (ev["round"] == "qual1").any() else "qual"
    q1 = ev[ev["round"] == first]
    return q1.loc[q1["status"] != "DNS", "rider_id"].unique().tolist()


def fit_season_model(raw, category="ME", half_life_days=HALF_LIFE_DAYS, category_weights=None,
                     prior_n=PRIOR_N, incident_prior_n=8.0, n_iter=30):
    """Fit rider pace / noise / incident model on every row of `raw` (any
    season or category). Pooled noise and incident parameters come from
    `category` only, since that's the field being simulated."""
    cat_w = {**CATEGORY_WEIGHTS, **(category_weights or {})}
    fin = raw[(raw["sector_id"] == "FINISH") & raw["round"].isin(RUN_WEIGHTS)].copy()
    dates = pd.to_datetime(fin["event_date"])
    fin["run"] = fin["event_id"] + "|" + fin["round"]
    fin["w"] = (fin["round"].map(RUN_WEIGHTS)
                * fin["category"].map(cat_w).fillna(0.5)
                * 0.5 ** ((dates.max() - dates).dt.days / half_life_days))

    ok = fin[(fin["status"] == "OK") & (fin["cum_time_s"] > 0)].copy()
    ok["y"] = np.log(ok["cum_time_s"])
    riders = ok["rider_id"].unique()
    mu = pd.Series(0.0, index=riders)
    clean = pd.Series(True, index=ok.index)
    for _ in range(n_iter):
        c = ok[clean]
        run_eff = (c["y"] - c["rider_id"].map(mu)).groupby(c["run"]).median()
        ok["resid"] = ok["y"] - ok["run"].map(run_eff)
        c = ok[clean]
        num = (c["resid"] * c["w"]).groupby(c["rider_id"]).sum()
        den = c["w"].groupby(c["rider_id"]).sum()
        mu = (num / (den + prior_n)).reindex(riders).fillna(0.0)  # shrink toward 0 = field median
        mu -= mu.median()
        clean = (ok["resid"] - ok["rider_id"].map(mu)) < INCIDENT_THRESHOLD
    ok["e"] = ok["resid"] - ok["rider_id"].map(mu)

    # noise: split clean race-run residuals into within-weekend (sigma) and
    # rider x weekend (tau) components
    race = ok[clean & ok["round"].isin(RACE_ROUNDS) & (ok["category"] == category)]
    g = race.groupby(["rider_id", "event_id"])["e"]
    mean_ie = g.transform("mean")
    dof = len(race) - g.ngroups
    sigma2 = ((race["e"] - mean_ie) ** 2).sum() / max(dof, 1)
    grp = race.groupby(["rider_id", "event_id"])["e"].agg(["mean", "count"])
    tau2 = max(grp["mean"].var() - (sigma2 / grp["count"]).mean(), 1e-6)

    # incidents over started race runs
    started = fin[fin["round"].isin(RACE_ROUNDS) & fin["status"].isin(["OK", "DNF", "DSQ"])].copy()
    started["e"] = ok["e"].reindex(started.index)
    started["incident"] = started["status"].isin(["DNF", "DSQ"]) | (started["e"] >= INCIDENT_THRESHOLD)
    pool = started[started["category"] == category]
    p0 = pool["incident"].mean()
    # per-rider rate: recency/category-weighted, shrunk toward the pooled rate
    started["w_inc"] = started["w"] * started["incident"]
    inc = started.groupby("rider_id")[["w_inc", "w"]].sum()
    p_inc = (inc["w_inc"] + incident_prior_n * p0) / (inc["w"] + incident_prior_n)
    incidents = pool[pool["incident"]]
    dnf_share = incidents["status"].isin(["DNF", "DSQ"]).mean()
    excess = incidents.loc[incidents["status"] == "OK", "e"].to_numpy()

    names = raw.drop_duplicates("rider_id", keep="last").set_index("rider_id")["rider_name"]
    cat_riders = ok.loc[ok["category"] == category, "rider_id"].unique()
    return dict(mu=mu, mu_new=float(mu.reindex(cat_riders).quantile(0.75)), sigma=float(np.sqrt(sigma2)),
                tau=float(np.sqrt(tau2)), p_inc=p_inc, p0=float(p0),
                dnf_share=float(dnf_share), excess=excess if len(excess) else np.array([0.05]),
                names=names)


def simulate_weekend(model, riders, n_sims=10000, attend_prob=None, rng=RNG, fmt=None, u_prior=None):
    """Simulate a race weekend for `riders` in format `fmt` (see
    event_format; default = 2026 format). u_prior = (mean, sd) arrays for each
    rider's weekend effect, e.g. from this weekend's timed training (see
    weekend_prior); default N(0, tau). Returns dict of (n_sims, n) arrays:
    qual_rank (the points-paying qualifier), final_rank (inf = didn't run or
    finish), made_final, points."""
    fmt = fmt or DEFAULT_FORMAT
    n = len(riders)
    mu = model["mu"].reindex(riders).fillna(model["mu_new"]).to_numpy()
    p_inc = model["p_inc"].reindex(riders).fillna(model["p0"]).to_numpy()
    if attend_prob is None:
        starts = np.ones((n_sims, n), bool)
    else:
        starts = rng.random((n_sims, n)) < np.asarray(attend_prob)
    u_mean, u_sd = u_prior if u_prior is not None else (0.0, model["tau"])
    u = rng.normal(u_mean, u_sd, (n_sims, n))

    def eps():
        if EPS_DF is None:
            return rng.normal(0.0, model["sigma"], (n_sims, n))
        return model["sigma"] * np.sqrt((EPS_DF - 2) / EPS_DF) * rng.standard_t(EPS_DF, (n_sims, n))

    def run(mask):
        t = mu + u + eps()
        inc = rng.random((n_sims, n)) < p_inc
        dnf = inc & (rng.random((n_sims, n)) < model["dnf_share"])
        t = t + np.where(inc & ~dnf, rng.choice(model["excess"], (n_sims, n)), 0.0)
        t[dnf | ~mask] = np.inf
        return t

    if fmt["kind"] == "q1q2":
        qual = ranks(run(starts))
        to_final = qual <= fmt["q1_to_final"]
        q2 = ranks(run(starts & ~to_final))
        to_final |= q2 <= fmt["q2_to_final"]
    elif fmt["kind"] == "semi":
        to_semi = ranks(run(starts)) <= fmt["to_semi"]
        qual = ranks(run(to_semi))           # semi-final ranks
        to_final = qual <= fmt["to_final"]
    else:
        qual = ranks(run(starts))
        to_final = qual <= fmt["to_final"]
    final = ranks(run(to_final))
    points = points_for(qual, QUAL_POINTS) + points_for(final, FINAL_POINTS)
    return dict(qual_rank=qual, final_rank=final, made_final=to_final, points=points)


def summarize_weekend(model, riders, sim):
    fr = sim["final_rank"]
    return pd.DataFrame(dict(
        rider_id=riders,
        rider_name=[model["names"].get(r, r) for r in riders],
        win_prob=(fr == 1).mean(0),
        podium_prob=(fr <= 3).mean(0),
        top10_prob=(fr <= 10).mean(0),
        make_final_prob=sim["made_final"].mean(0),
        exp_points=sim["points"].mean(0),
    )).sort_values("exp_points", ascending=False).reset_index(drop=True)


def simulate_standings(model, current_points, weekends, n_sims=10000, rng=RNG):
    """current_points: Series rider_id -> points so far. weekends: list of
    (riders, attend_prob_or_None, fmt_or_None[, u_prior]). Returns
    (standings_df, per-weekend sims)."""
    all_riders = list(dict.fromkeys(list(current_points.index) + [r for w, *_ in weekends for r in w]))
    col = {r: j for j, r in enumerate(all_riders)}
    total = np.tile(current_points.reindex(all_riders).fillna(0.0).to_numpy(), (n_sims, 1))
    sims = []
    for riders, attend, fmt, *prior in weekends:
        sim = simulate_weekend(model, riders, n_sims=n_sims, attend_prob=attend, rng=rng, fmt=fmt,
                               u_prior=prior[0] if prior else None)
        total[:, [col[r] for r in riders]] += sim["points"]
        sims.append(sim)
    jitter = rng.random(total.shape) * 1e-3   # random tie-break
    rank = ranks(-(total + jitter))
    df = pd.DataFrame(dict(
        rider_id=all_riders,
        rider_name=[model["names"].get(r, r) for r in all_riders],
        current_points=current_points.reindex(all_riders).fillna(0.0).to_numpy(),
        exp_points=total.mean(0),
        points_p10=np.percentile(total, 10, axis=0),
        points_p90=np.percentile(total, 90, axis=0),
        champion_prob=(rank == 1).mean(0),
        top3_prob=(rank <= 3).mean(0),
        top10_prob=(rank <= 10).mean(0),
        exp_rank=rank.mean(0),
    )).sort_values("exp_points", ascending=False).reset_index(drop=True)
    return df, sims


