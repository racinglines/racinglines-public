"""
The walk-forward engine for any sport (docs/backtest-core.md): price every event from data strictly
before it with the sport's pricing model (racinglines/models/race_model.py), settle every market kind
the simulations support from the official result (racinglines/markets/kinds.py), and score the fair
values (racinglines/core/calibration.py).

    out = walk_forward.run(race_model.get("mtb_dh"), data, settings, seasons=[2025])
    out["events"]        one row per event: n, Brier and log loss per kind
    out["rows"]          one row per entrant x kind: fair value and outcome (for reliability curves); kind
                         teammate_h2h = every same-team pair, P(a ahead of b) (teammate_rows below)
    out["teammates"]     the teammate_h2h rows with the team key and each driver's win probability and win
                         outcome (a_win, b_win, a_won, b_won): the within-team win share, the top of the distribution,
                         beside P(ahead) (the driver-vs-car diagnostic reads these)
    out["calibration"]   per kind (and per kind x season): n, Brier, log loss, ECE
    out["reliability"]   per kind: probability bins
    out["cutoffs"]       {event id: the cutoff it was priced at}, to read a venue's prices as of then

Model-only by default; `--venue` on the command line scores a venue's prices beside the model's, on exact
market links (racinglines/pipelines/model_vs_market.py). One seeded generator is shared by every event in
order, so a run is reproducible from its settings.

    run(model, data, settings, stage="after Quali")
                         prices every event at that stage of the schema's [stages] (core/stages.py): the model's
                         `at_stage(ev, label)` gives the event the sessions run by then (Event.info "sessions", as
                         the season sweep's "sessions" mode does) or None (the event has no such stage: not priced).
                         Only a model with at_stage prices by stage (a challenger; the engine, GlobalModel, prices the
                         same at every label and only labels its records); unset, every event is priced once
                         at its cutoff, as before.

A model with `event_params(ev)` (a challenger) adds what it chose for each event to its events row (the temperature
tau with `temperature` set, tau_front with `temperature_front`, each entrant's quasi-SE as JSON with `cov`; nothing
otherwise, so the columns are as before).
"""

import numpy as np
import pandas as pd

from racinglines.core import calibration as CAL
from racinglines.markets import kinds as K
from racinglines.markets import payoffs as P

PER_ENTRANT = ("top_n", "stage_top_n", "reached", "indicator")   # indicator: a drawn yes/no (the fastest lap)


def event_rows(ev, sims, res, kinds=None):
    """(kind, athlete_id, opponent, fair, y) for every market of `kinds` (default: every per-entrant kind
    the simulations support) on one event; y is None where the result can't settle it."""
    summ = K.summary(sims)
    kinds = kinds or [c for c in summ.columns if c != "athlete_id"]
    out = []
    for kind in kinds:
        k = K.KINDS[kind]
        if k.payoff in PER_ENTRANT:
            if kind in summ:
                col = summ[kind]
            elif k.payoff == "top_n":              # a top-n kind off by default (race_top5), priced when named
                col = K.fair(kind, sims)
            else:
                continue
            for a, p in zip(sims.entrants, col):
                out.append(dict(kind=kind, athlete_id=a, opponent=None, fair=float(p),
                                y=K.settle(kind, a, None, res)))
        elif k.payoff == "h2h":
            h = K.h2h_matrix(sims)
            for i, a in enumerate(sims.entrants):
                for j, b in enumerate(sims.entrants[i + 1:], i + 1):
                    out.append(dict(kind=kind, athlete_id=a, opponent=b, fair=float(h[i, j]),
                                    y=K.settle(kind, a, {"opponent_id": b}, res)))
        elif k.payoff == "group_top" and sims.groups is not None and sims.points is not None:
            for g, p in K.group_top(sims).items():
                out.append(dict(kind=kind, athlete_id=None, opponent=g, fair=p,
                                y=K.settle(kind, None, {"team": g}, res)))
    df = pd.DataFrame(out, columns=["kind", "athlete_id", "opponent", "fair", "y"])
    return df.assign(event_id=ev.id, season=ev.season, event=ev.name)


TEAMMATE_KIND = "teammate_h2h"


def teammate_rows(ev, sims, res):
    """(kind teammate_h2h, athlete_id, opponent, fair, y) for every pair of entrants on the same team: P(a classified
    ahead of b) from the simulations (the race_h2h payoff), settled by the race_h2h rule. Teams come from
    sims.groups (the model's team key per entrant) or, failing that, the results frame's team_id. Within-team
    pairs are where the car cancels and driver skill alone is priced (docs/todo.md U-F1-DC, 2026-10-23); the
    walk-forward scores them beside win and podium as `teammate_h2h_*` on every event with 2+ entrants per team."""
    if sims.groups is not None:
        team = dict(zip(sims.entrants, sims.groups))
    elif res is not None and "team_id" in res:
        team = dict(zip(res["athlete_id"], res["team_id"]))
    else:
        team = {}
    team = {a: t for a, t in team.items() if t is not None and not (isinstance(t, float) and np.isnan(t))}
    out, h, win = [], None, None
    for i, a in enumerate(sims.entrants):
        for j, b in enumerate(sims.entrants[i + 1:], i + 1):
            if a in team and b in team and team[a] == team[b]:
                if h is None:
                    h = K.h2h_matrix(sims); win = K.fair("race_win", sims)          # the win share within the team: the top of the distribution
                out.append(dict(kind=TEAMMATE_KIND, athlete_id=a, opponent=b, fair=float(h[i, j]),
                                y=K.settle("race_h2h", a, {"opponent_id": b}, res), team=team[a],
                                a_win=float(win[i]), b_win=float(win[j]),
                                a_won=K.settle("race_win", a, None, res), b_won=K.settle("race_win", b, None, res)))
    df = pd.DataFrame(out, columns=["kind", "athlete_id", "opponent", "fair", "y", "team", "a_win", "b_win", "a_won", "b_won"])
    return df.assign(event_id=ev.id, season=ev.season, event=ev.name)


def run(model, data, settings, seasons=None, kinds=None, echo=print, keep_sims=False, stage=None):
    if stage is not None and not hasattr(model, "at_stage"):
        raise ValueError(f"{model.name} prices each event once (no at_stage): it can't be priced at stage {stage!r}")
    rng = np.random.default_rng(settings.rng_seed)
    hist = model.history(data, settings)
    evs = model.events(data, settings, seasons)
    rows, events, kept, teammates = [], [], [], []
    cutoffs = {}
    for i, ev in enumerate(evs, 1):
        if stage is not None:
            ev = model.at_stage(ev, stage)
            if ev is None:
                echo(f"progress {i}/{len(evs)} {evs[i - 1].season} {evs[i - 1].name}: no {stage!r} stage, not priced")
                continue
        sims = model.price(hist, ev, settings, rng)
        if sims is None:
            continue
        cutoffs[ev.id] = ev.cutoff
        if keep_sims:
            kept.append((ev, sims))
        res = P.mark_sport(model.results(data, ev), getattr(model, "sport", None))
        r = event_rows(ev, sims, res, kinds)
        tm = teammate_rows(ev, sims, res)
        if len(tm):
            r = pd.concat([r, tm[["kind", "athlete_id", "opponent", "fair", "y", "event_id", "season", "event"]]], ignore_index=True); teammates.append(tm)
        rows.append(r)
        e = dict(season=ev.season, event_id=ev.id, event=ev.name, n_entrants=len(sims.entrants))
        if hasattr(model, "event_params"):           # what the model chose for this event (e.g. a temperature)
            e.update(model.event_params(ev))
        for kind, g in r.dropna(subset=["y"]).groupby("kind", sort=False):
            s = CAL.scores(g["fair"], g["y"].astype(float))
            e.update({f"{kind}_n": s["n"], f"{kind}_brier": s["brier"], f"{kind}_logloss": s["logloss"]})
        events.append(e)
        echo(f"progress {i}/{len(evs)} priced {ev.season} {ev.name}")
    rows = pd.concat(rows, ignore_index=True) if rows else \
        pd.DataFrame(columns=["kind", "athlete_id", "opponent", "fair", "y", "event_id", "season", "event"])
    scored = rows.dropna(subset=["y"]).assign(y=lambda d: d["y"].astype(float))
    cal_all, rel = CAL.table(scored, ("fair",), by=("kind",))
    cal_season, _ = CAL.table(scored, ("fair",), by=("kind", "season"))
    out = dict(events=pd.DataFrame(events), rows=rows, reliability=rel, cutoffs=cutoffs,
               calibration=pd.concat([cal_all.assign(season="all"), cal_season], ignore_index=True),
               teammates=pd.concat(teammates, ignore_index=True) if teammates else
               pd.DataFrame(columns=["kind", "athlete_id", "opponent", "fair", "y", "team", "a_win", "b_win", "a_won", "b_won", "event_id", "season", "event"]))
    if keep_sims:                    # opt-in (prediction records): [(event, OutcomeSims), ...] in pricing order
        out["sims"] = kept
    return out


def saved_metrics(out):
    """What a saved walk_forward run keeps (model_runs.metrics): the per-event rows, the calibration, and
    `weekends`: per event, each kind's score = -1000 x log loss (higher is better), the search report's curves."""
    from racinglines.db.queries import records
    ev = out["events"]
    kinds = sorted({c[:-len("_logloss")] for c in ev.columns if c.endswith("_logloss")})
    weekends = [dict(round=i + 1, event=f"{r['season']} {r['event']}", season=r["season"],
                     **{f"{k}_score": -1000 * r[f"{k}_logloss"] for k in kinds if pd.notna(r.get(f"{k}_logloss"))})
                for i, r in ev.reset_index(drop=True).iterrows()]
    return dict(events=records(ev), weekends=weekends, calibration=records(out["calibration"]))
