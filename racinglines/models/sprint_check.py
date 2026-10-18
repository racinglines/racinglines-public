"""
Walk-forward check of the sprint stage (pricing.price_stages, decision log 2026-10-06, provisional): every stored
sprint weekend priced as of a minute before Sprint Qualifying and a minute before the sprint, through the one
pricing path (price_race, so the as-of view is the only data it sees), then scored against the sprint's result,
read only after pricing:

    sprint stage     extra.sprint_win_prob, the sprint simulated with the schema's points and dnf_scale
    GP stand-in      the Grand Prix win probability at the same cutoff (what db/reads.SPRINT_FALLBACK prices
                     a sprint-winner link with when a run has no sprint stage)

Brier and log loss of the sprint winner per weekend (over the entrants), their paired difference (stage − stand-in,
with its standard error over weekends), and per weekend the retirements in the sprint vs the same weekend's Grand
Prix (DNF + DSQ over the starters, DNS left out) beside the simulated sprint DNF share, the evidence for
[sessions.sim] sprint.dnf_scale in sports/f1.toml.

    racinglines model sprint-check f1 [--from 2021] [--sims 4000] [--out sprint_check.csv]
"""

import numpy as np
import pandas as pd

from racinglines.core.stats import brier
from racinglines.models.inputs import laps as M
from racinglines.models.inputs import laps_pricing as run

STAGE = "sprint"
RETIRED = ("DNF", "DSQ")
# the mode's cutoff is a minute before that session: "pre_sq" the weekend's own grid session (M.grid_source: Sprint
# Qualifying, or in 2021 the Friday qualifying), "pre_sprint" the sprint
MODES = ("pre_sq", "pre_sprint")


def _anchor(mode, year):
    return M.grid_source(STAGE, year) if mode == "pre_sq" else STAGE


def _retired(rows):
    """(retirements, starters) of a classification: DNF + DSQ over everyone who didn't DNS."""
    st = rows["status"].astype(str)
    started = st != "DNS"
    return int(st[started].isin(RETIRED).sum()), int(started.sum())


def check(meas, hist, start_year=2021, n_sims=4000, seed=0, modes=MODES, echo=None):
    """One row per sprint weekend x mode (see the module docstring)."""
    from racinglines import progress as PG
    rng = np.random.default_rng(seed)
    sp = meas.res[(meas.res["round"] == STAGE) & (meas.res["year"] >= start_year)]
    events = sp.groupby("event_id")["session_ts"].first().sort_values()
    rows = []
    for eid in PG.track(list(events.index), "sprint weekend"):
        sessions = meas.sessions(eid)
        year = int(sp.loc[sp["event_id"] == eid, "year"].iloc[0])
        for mode in modes:
            anchor = sessions.get(_anchor(mode, year))
            if anchor is None or pd.isna(anchor):
                continue
            cutoff = anchor - run.ONE_MIN
            if not (hist["r_ts"] + M.RACE_DONE < cutoff).any():
                continue
            summ, ex = run.price_race(meas, hist, cutoff, eid, n_sims=n_sims, rng=rng)
            if f"{STAGE}_win_prob" not in summ:
                continue
            # scored AFTER pricing: the sprint's and the Grand Prix's own results
            res = meas.res[(meas.res["event_id"] == eid) & (meas.res["round"] == STAGE)].drop_duplicates("athlete_id")
            by = res.set_index("athlete_id")
            won = summ["athlete_id"].map(by["position"]).eq(1) & summ["athlete_id"].map(by["status"]).eq("OK")
            y = won.to_numpy(float)
            gp = meas.res[(meas.res["event_id"] == eid) & (meas.res["round"] == M.MAIN_STAGE)].drop_duplicates("athlete_id")
            s_dnf, s_n = _retired(res)
            g_dnf, g_n = _retired(gp)
            row = dict(event_id=int(eid), year=int(res["year"].iloc[0]), round=int(res["series_round"].iloc[0]),
                       venue=res["venue"].iloc[0], mode=mode, cutoff=str(cutoff), n=len(summ),
                       grid=ex["audit"]["stages"][STAGE]["grid"], winner_found=bool(y.sum() == 1),
                       sprint_dnf=s_dnf, sprint_starters=s_n, gp_dnf=g_dnf, gp_starters=g_n,
                       sim_sprint_dnf_share=float(ex["sim"]["stages"][STAGE]["dnf"].mean()),
                       sim_gp_dnf_share=float(ex["sim"]["dnf"].mean()))
            for name, col in (("stage", f"{STAGE}_win_prob"), ("proxy", "win_prob")):
                p = summ[col].to_numpy(float)
                row[f"brier_{name}"] = brier(p, y)
                row[f"logloss_{name}"] = run._logloss(p, y)
                row[f"winner_prob_{name}"] = float(p[y == 1].sum())
            rows.append(row)
        if echo:
            echo(f"progress {len(rows)} rows, event {eid}")
    return pd.DataFrame(rows)


def summarize(df):
    """Per mode: weekends, mean Brier and log loss (stage, stand-in), the paired difference stage − stand-in with its
    standard error over weekends, and the observed and simulated DNF shares (sprint vs Grand Prix)."""
    out = []
    for mode, g in df.groupby("mode", sort=False):
        r = dict(mode=mode, weekends=len(g))
        for m in ("brier", "logloss"):
            d = g[f"{m}_stage"] - g[f"{m}_proxy"]
            r.update({f"{m}_stage": g[f"{m}_stage"].mean(), f"{m}_proxy": g[f"{m}_proxy"].mean(), f"{m}_diff": d.mean(),
                      f"{m}_diff_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")})
        r.update(sprint_dnf_share=g["sprint_dnf"].sum() / max(g["sprint_starters"].sum(), 1),
                 gp_dnf_share=g["gp_dnf"].sum() / max(g["gp_starters"].sum(), 1),
                 sim_sprint_dnf_share=g["sim_sprint_dnf_share"].mean(), sim_gp_dnf_share=g["sim_gp_dnf_share"].mean())
        r["observed_dnf_ratio"] = r["sprint_dnf_share"] / r["gp_dnf_share"] if r["gp_dnf_share"] else float("nan")
        out.append(r)
    return pd.DataFrame(out)


def render(df):
    if df.empty:
        return "No sprint weekend with a sprint result and earlier races to train on."
    cols = ["year", "round", "venue", "mode", "grid", "brier_stage", "brier_proxy", "winner_prob_stage",
            "winner_prob_proxy", "sprint_dnf", "sprint_starters", "gp_dnf", "gp_starters"]
    return ("=== Per weekend ===\n" + df[cols].to_string(index=False, float_format="{:.4f}".format)
            + f"\n\n=== Summary (dnf_scale {M.SIM_SESSIONS[STAGE].get('dnf_scale', 1.0)}) ===\n"
            + summarize(df).to_string(index=False, float_format="{:.4f}".format))
