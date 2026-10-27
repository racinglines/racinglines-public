"""
Season-level uses of the timed-runs model: backtests, walk-forward validation,
the season forecast (with each weekend's timed-training prior) and rank moves.
"""

import numpy as np
import pandas as pd

from racinglines.core.stats import brier

from .model import (
    DEFAULT_FORMAT, INCIDENT_THRESHOLD, RNG, actual_event_points, event_format, event_order, event_starters,
    fit_season_model, simulate_standings, simulate_weekend, summarize_weekend, unraced_format,
)


def _spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def _training_rows(raw, target, before_date=None, train_scope="all"):
    """Rows the model may learn from: every season/category ("all") or only
    the target season + category ("season"), strictly before `before_date`."""
    rows = target if train_scope == "season" else raw
    if before_date is not None:
        rows = rows[rows["event_date"] < before_date]
    return rows


def _event_date(df, event_id):
    return df.loc[df["event_id"] == event_id, "event_date"].min()


def _score_weekend(target, pts, event_id, model, field, sim):
    """Compare one simulated weekend with what actually happened."""
    summ = summarize_weekend(model, field, sim)
    # riders new to the model have no name in its lookup; take it from the event
    names = target.drop_duplicates("rider_id").set_index("rider_id")["rider_name"]
    summ["rider_name"] = summ["rider_id"].map(names).fillna(summ["rider_name"])
    act = pts[pts["event_id"] == event_id].set_index("rider_id")
    summ["actual_final_rank"] = summ["rider_id"].map(act["final_rank"])
    summ["actual_points"] = summ["rider_id"].map(act["points"]).fillna(0.0)
    fr = summ["actual_final_rank"]
    made = summ["rider_id"].isin(
        target.loc[(target["event_id"] == event_id) & (target["round"] == "final"), "rider_id"])
    n = len(summ)
    metrics = dict(
        event_id=event_id,
        venue=target.loc[target["event_id"] == event_id, "venue"].iloc[0] if "venue" in target else event_id,
        n_riders=n,
        spearman_points=_spearman(summ["exp_points"], summ["actual_points"]),
        brier_win=brier(summ["win_prob"], fr == 1),
        brier_win_base=brier(np.full(n, 1 / n), fr == 1),
        brier_podium=brier(summ["podium_prob"], fr <= 3),
        brier_podium_base=brier(np.full(n, 3 / n), fr <= 3),
        brier_final=brier(summ["make_final_prob"], made),
        brier_final_base=brier(np.full(n, made.mean()), made),
        top10_hits=len(set(summ.nlargest(10, "top10_prob")["rider_id"])
                       & set(summ.loc[fr <= 10, "rider_id"])),
        winner=summ.loc[fr == 1, "rider_name"].squeeze() if (fr == 1).any() else None,
        winner_pred_win_prob=float(summ.loc[fr == 1, "win_prob"].sum()),
    )
    return metrics, summ


def backtest_season(raw, target, n_holdout=2, n_sims=10000, rng=RNG, train_scope="all", **fit_kw):
    """Fit on everything before the last n_holdout target events, simulate
    those events with their actual start lists, and score against what
    actually happened (per event, and the standings after them)."""
    category = target["category"].iloc[0]
    events = event_order(target)
    train_ev, test_ev = events[:-n_holdout], events[-n_holdout:]
    train = _training_rows(raw, target, _event_date(target, test_ev[0]), train_scope)
    model = fit_season_model(train, category=category, **fit_kw)
    pts = actual_event_points(target)
    before = pts[pts["event_id"].isin(train_ev)].groupby("rider_id")["points"].sum()
    fields = [event_starters(target, e) for e in test_ev]
    weekends = [(f, None, event_format(target, e)) for f, e in zip(fields, test_ev)]
    standings, sims = simulate_standings(model, before, weekends, n_sims=n_sims, rng=rng)
    event_reports = [_score_weekend(target, pts, e, model, f, sim) for e, f, sim in zip(test_ev, fields, sims)]

    actual_total = pts.groupby("rider_id")["points"].sum()
    standings["actual_points"] = standings["rider_id"].map(actual_total).fillna(0.0)
    standings["actual_rank"] = standings["actual_points"].rank(ascending=False, method="min")
    return model, event_reports, standings, (train_ev, test_ev)


def walk_forward_season(raw, target, min_prior_events=None, n_sims=5000, rng=RNG, train_scope="all",
                        rider_rows=None, **fit_kw):
    """For each target event after the first min_prior_events: fit on
    everything before it, simulate it with its actual start list, score it.
    With train_scope="all" even round 1 has history, so it's included.
    rider_rows: a list to append each event's per-rider predictions and outcomes to
    (for reliability curves, see calibration())."""
    category = target["category"].iloc[0]
    events = event_order(target)
    if min_prior_events is None:
        min_prior_events = 0 if train_scope == "all" else 1
    pts = actual_event_points(target)
    rows = []
    for e in events[min_prior_events:]:
        train = _training_rows(raw, target, _event_date(target, e), train_scope)
        if not (train["category"] == category).any():
            continue  # nothing to learn from yet (first event in the data)
        model = fit_season_model(train, category=category, **fit_kw)
        field = event_starters(target, e)
        fmt = event_format(target, e)
        if not fmt.get("to_final", fmt.get("q1_to_final")):
            continue  # no final results (e.g. PDF-only round)
        sim = simulate_weekend(model, field, n_sims=n_sims, rng=rng, fmt=fmt)
        metrics, summ = _score_weekend(target, pts, e, model, field, sim)
        rows.append({**metrics, "format": fmt["kind"],
                     "n_final": fmt.get("to_final", fmt.get("q1_to_final", 0) + fmt.get("q2_to_final", 0))})
        if rider_rows is not None:
            made = target.loc[(target["event_id"] == e) & (target["round"] == "final"), "rider_id"]
            fr = summ["actual_final_rank"]
            rider_rows.append(summ.assign(event_id=e, venue=metrics["venue"], won=fr == 1, podium=fr <= 3,
                                          top10=fr <= 10, made_final=summ["rider_id"].isin(made)))
    return pd.DataFrame(rows)


# market -> (predicted column, outcome column)
CAL_MARKETS = dict(win=("win_prob", "won"), podium=("podium_prob", "podium"), top10=("top10_prob", "top10"),
                   final=("make_final_prob", "made_final"))
CAL_BINS = (0, .01, .03, .06, .1, .2, .35, .5, .7, .9, 1.0001)


def calibration(riders, bins=CAL_BINS):
    """Reliability table from walk-forward rider rows: per market and probability bin, the
    count, mean predicted and observed frequency (z = (observed - predicted) / its binomial
    SE), plus a 'total' row per market with Brier and log loss."""
    out = []
    for mkt, (p, y) in CAL_MARKETS.items():
        pr, ob = riders[p].to_numpy(float), riders[y].to_numpy(bool).astype(float)
        cut = pd.cut(pr, bins, right=False)
        for b, g in pd.DataFrame(dict(p=pr, y=ob, bin=cut)).groupby("bin", observed=True):
            m = g["p"].mean()
            se = np.sqrt(max(m * (1 - m), 1e-6) / len(g))
            out.append(dict(market=mkt, bin=f"{b.left:.2f}-{min(b.right, 1):.2f}", n=len(g), predicted=m,
                            observed=g["y"].mean(), z=(g["y"].mean() - m) / se))
        q = np.clip(pr, 1e-4, 1 - 1e-4)
        out.append(dict(market=mkt, bin="total", n=len(pr), predicted=pr.mean(), observed=ob.mean(),
                        z=np.nan, brier=float(((pr - ob) ** 2).mean()),
                        logloss=float(-(ob * np.log(q) + (1 - ob) * np.log(1 - q)).mean())))
    return pd.DataFrame(out)


def rank_moves(current_points, riders, sim_points):
    """Championship rank movement over one simulated weekend.

    current_points: Series rider_id -> points before the weekend; riders /
    sim_points: the weekend's field and its (n_sims, n) simulated points. Ranks
    are "min" style (tied riders share the better rank), before and after.
    Returns a DataFrame: rider_id, current_rank, rank_up_prob, rank_down_prob,
    exp_rank_after. Uses the same (placeholder) points tables as everything else."""
    all_riders = list(dict.fromkeys(list(current_points.index) + list(riders)))
    col = {r: j for j, r in enumerate(all_riders)}
    before = current_points.reindex(all_riders).fillna(0.0).to_numpy()
    total = np.tile(before, (sim_points.shape[0], 1))
    total[:, [col[r] for r in riders]] += sim_points
    rank_before = 1 + (before[None, :] > before[:, None]).sum(axis=1)
    neg_sorted = np.sort(-total, axis=1)
    rank_after = np.empty_like(total)
    for i in range(total.shape[0]):
        rank_after[i] = np.searchsorted(neg_sorted[i], -total[i], side="left") + 1
    out = pd.DataFrame(dict(rider_id=all_riders, current_rank=rank_before,
                            rank_up_prob=(rank_after < rank_before).mean(0),
                            rank_down_prob=(rank_after > rank_before).mean(0),
                            exp_rank_after=rank_after.mean(0)))
    return out[out["rider_id"].isin(riders)]


def completed_events(target, cfg=None):
    """Target events whose final has results (upcoming / in-progress ones don't). With a cfg whose sport has
    single-run events (starters_only, model.event_format "single_run"): an event with no final round at all whose
    qualifier has results is completed too (one timed run decided it)."""
    fin = target[(target["round"] == "final") & (target["sector_id"] == "FINISH") & (target["status"] == "OK")]
    done = set(fin["event_id"])
    if cfg and cfg.get("starters_only"):
        has_final = set(target.loc[target["round"] == "final", "event_id"])
        q = target[(target["round"] == "qual") & (target["sector_id"] == "FINISH") & (target["status"] == "OK")]
        done |= set(q["event_id"]) - has_final
    return done


def weekend_prior(model, event_rows, riders, practice_sd_mult=1.5, max_iqr=0.08):
    """Posterior (mean, sd) of each rider's weekend effect u, given this
    weekend's timed training. Training runs are noisier than race runs (riders
    hold back, test lines), so their noise is inflated by practice_sd_mult;
    runs slower than INCIDENT_THRESHOLD are ignored. Riders without a training
    run keep the prior N(0, tau).

    Safety check: if the session's residuals are far more spread out than
    normal run-to-run noise (interquartile range > max_iqr in log-time, e.g.
    riders held on track during a stoppage, as at Whistler 2026), the session
    isn't a pace signal and is ignored entirely."""
    tau2, sig2 = model["tau"] ** 2, (model["sigma"] * practice_sd_mult) ** 2
    tt = event_rows[(event_rows["round"] == "practice") & (event_rows["sector_id"] == "FINISH")
                    & (event_rows["status"] == "OK") & (event_rows["cum_time_s"] > 0)]
    mean = pd.Series(0.0, index=riders)
    sd = pd.Series(model["tau"], index=riders)
    if len(tt) >= 10:
        mu = tt["rider_id"].map(model["mu"]).fillna(model["mu_new"])
        e0 = np.log(tt["cum_time_s"]) - mu
        iqr = float(e0.quantile(0.75) - e0.quantile(0.25))
        if iqr > max_iqr:
            print(f"WARNING: timed training ignored for {event_rows['event_id'].iloc[0]}: residual spread "
                  f"(IQR {iqr:.3f} log-time) is too large to be pace, e.g. riders held on track. "
                  f"Using the unconditioned model for this weekend.")
            return mean.to_numpy(), sd.to_numpy()
        e = pd.Series((e0 - e0.median()).to_numpy(), index=tt["rider_id"].to_numpy())
        e = e[e < INCIDENT_THRESHOLD].groupby(level=0).mean()
        e = e[e.index.isin(riders)]
        mean[e.index] = tau2 * e / (tau2 + sig2)
        sd[e.index] = np.sqrt(tau2 * sig2 / (tau2 + sig2))
    return mean.to_numpy(), sd.to_numpy()


def forecast_season(raw, target, n_remaining=2, n_sims=10000, attend_window=3, rng=RNG,
                    train_scope="all", unraced="default", **fit_kw):
    """Simulate the rest of the target season: n_remaining rounds in total.

    - Rounds already in the data without final results (the weekend in
      progress, e.g. timed training done, Q1 start list published) are
      simulated with their real start list, and each rider's weekend effect is
      conditioned on this weekend's timed training (weekend_prior). The model is
      fit only on data from before that weekend.
    - Remaining rounds not in the data yet: field = riders who started Q1 in
      any of the last attend_window completed events, each attending with
      probability (starts in that window / window).
    - unraced: the format those rounds are simulated in (model.unraced_format): "default" = the 2026 elite men's
      format for every category; "last" = the category's latest completed event's own format.
    Returns (model, upcoming, per_round, standings): upcoming = [(event_id,
    summary)] for in-data rounds; per_round = summary for an unknown round (or
    None)."""
    category = target["category"].iloc[0]
    done = completed_events(target)
    events = [e for e in event_order(target) if e in done]
    upcoming = [e for e in event_order(target) if e not in done]
    cutoff = min(_event_date(target, e) for e in upcoming) if upcoming else None
    model = fit_season_model(_training_rows(raw, target, cutoff, train_scope), category=category, **fit_kw)
    pts = actual_event_points(target[target["event_id"].isin(done)])
    current = pts.groupby("rider_id")["points"].sum()

    fmt = unraced_format(target, done, unraced)
    weekends = []
    for e in upcoming:
        field = event_starters(target, e)
        prior = weekend_prior(model, target[target["event_id"] == e], field)
        weekends.append((field, None, fmt, prior))
    n_unknown = max(n_remaining - len(upcoming), 0)
    recent = events[-attend_window:]
    starts = pd.Series([r for e in recent for r in event_starters(target, e)]).value_counts()
    # ties in rider-id order: value_counts' own tie order comes from an unstable sort that differs by CPU (Apple
    # silicon vs x86), and each rider's attendance draw follows this order, so the forecast moved with the machine
    starts = starts.sort_index(kind="stable").sort_values(ascending=False, kind="stable")
    field = starts.index.tolist()
    weekends += [(field, (starts / len(recent)).to_numpy(), fmt)] * n_unknown

    standings, sims = simulate_standings(model, current, weekends, n_sims=n_sims, rng=rng)
    upcoming_summaries = []
    for e, (f, *_), sim in zip(upcoming, weekends, sims):
        summ = summarize_weekend(model, f, sim)
        prior_mean = dict(zip(f, weekends[upcoming.index(e)][3][0]))
        summ["tt_pace_adj_pct"] = summ["rider_id"].map(prior_mean) * 100  # + = slower than usual in training
        if e == upcoming[0]:  # standings move over the next weekend (later ones depend on it)
            moves = rank_moves(current, f, sim["points"]).set_index("rider_id")
            for c in ("current_rank", "rank_up_prob", "rank_down_prob", "exp_rank_after"):
                summ[c] = summ["rider_id"].map(moves[c])
        upcoming_summaries.append((e, summ))
    per_round = None
    if n_unknown:
        per_round = summarize_weekend(model, field, sims[len(upcoming)])
        per_round["attend_prob"] = per_round["rider_id"].map(starts / len(recent))
    standings["current_rank"] = standings["current_points"].rank(ascending=False, method="min")
    return model, upcoming_summaries, per_round, standings


