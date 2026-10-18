"""
What every race model simulates, in one shape (docs/backtest-core.md): the contract between a
sport's model and the shared market, backtest and search code.

A race model's Monte Carlo gives, per simulation, a classification of the entrants. Every market
we price (win, podium, top 10, pole, head-to-head, makes the Final, top team) is a function of it,
so market code reads an `OutcomeSims` and never a model's own arrays. A yes/no the classification
doesn't decide, which a model draws itself (F1's fastest lap), rides along in `indicators`.

    sims = from_position_sim(entrants, sim)     # F1: pricing.price_race's extras["sim"], ["entrants"]
    sims = from_timed_runs(riders, sim)         # downhill: run_times.model.simulate_weekend
    markets.kinds.fair("race_podium", sims)     # per-entrant fair values

Adapters only re-label arrays (no copies of the random draws, no new randomness), so prices built on
an `OutcomeSims` equal the model's own summaries exactly.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutcomeSims:
    entrants: list                   # athlete ids, the columns of every array
    rank: np.ndarray                 # (n_sims, n) classification rank; ties allowed; np.inf = not classified
    finished: np.ndarray             # (n_sims, n) bool: counts for top-n markets (F1: not a DNF; DH: ran the Final)
    stage_rank: dict = field(default_factory=dict)   # earlier deciding rounds' ranks, e.g. {"qual": grid}
    reached: dict = field(default_factory=dict)      # (n_sims, n) bool per round reached, e.g. {"final": made_final}
    points: np.ndarray | None = None                 # (n_sims, n) championship points scored
    groups: list | None = None                       # per entrant: team / nation key (group markets)
    indicators: dict = field(default_factory=dict)   # (n_sims, n) bool per yes/no kind the model draws itself,
                                                     # keyed by kind, e.g. {"race_fastest_lap": fl}
    stage_finished: dict = field(default_factory=dict)   # per stage_rank key that is a race of its own (F1: the
    stage_points: dict = field(default_factory=dict)     # sprint): (n_sims, n) finished and points, see at()

    @property
    def n_sims(self):
        return self.rank.shape[0]

    def index(self, athlete_id):
        return self.entrants.index(athlete_id)

    def at(self, stage):
        """The simulations of one earlier round as a classification of its own (F1: the sprint), so every race
        payoff (top n, head-to-head, top team) prices it unchanged: rank = stage_rank[stage], with that stage's
        finished and points (a stage without them: every entrant finished, no points)."""
        if stage not in self.stage_rank:
            raise ValueError(f"these simulations have no {stage} stage")
        r = self.stage_rank[stage]
        return OutcomeSims(entrants=self.entrants, rank=r, finished=self.stage_finished.get(stage, np.isfinite(r)),
                           points=self.stage_points.get(stage), groups=self.groups)

    def to_records(self, run_id, sport, model_id, season, event_id, event, stage, cutoff, kinds=None):
        """Long-format prediction records (see `to_records` below)."""
        return to_records(self, run_id, sport, model_id, season, event_id, event, stage, cutoff, kinds)


@dataclass(frozen=True)
class SeasonSims:
    """What a season simulation gives (e.g. models/nascar_season.py): per simulation, the final championship
    standings. Season-long markets (markets.kinds.season_fair) read it the way race markets read an OutcomeSims."""
    entrants: list                   # athlete ids, the columns of every array
    rank: np.ndarray                 # (n_sims, n) final standings position, 1 = champion, no ties
    points: np.ndarray               # (n_sims, n) final points
    wins: np.ndarray | None = None   # (n_sims, n) season race wins
    qualified: np.ndarray | None = None   # (n_sims, n) bool: made the playoff / Chase

    @property
    def n_sims(self):
        return self.rank.shape[0]

    def index(self, athlete_id):
        return self.entrants.index(athlete_id)


def from_position_sim(entrants, sim):
    """F1 (position_sim): `pos` is the classification with retirements last, `grid` the starting grid and `qual`
    the qualifying order when they differ (grid penalties; stage_rank["qual"] is always the qualifying order),
    `fl` (with model.FASTEST_LAP) who set the fastest lap; `practice` (with model.PRACTICE_FASTEST) each practice
    session's order of best laps under its name (stage_rank["fp1"], ...); `stages` (sprint weekends, pricing.price_stages) adds
    each side stage's classification, finished and points under its name and its grid under its grid session's
    (stage_rank["sprint"], ["sprint_qual"])."""
    pos = sim["pos"]
    stage_rank, finished, points = {"qual": sim.get("qual", sim["grid"])}, {}, {}
    for stage, s in sim.get("stages", {}).items():
        if s["grid_from"] != "qual":       # 2021's sprint grid is the weekend's own qualifying: never overwrite the GP's
            stage_rank[s["grid_from"]] = s["grid"]
        stage_rank[stage] = s["pos"]
        finished[stage], points[stage] = ~s["dnf"], s["points"]
    for session, r in sim.get("practice", {}).items():     # practice best-lap orders (model.PRACTICE_FASTEST)
        stage_rank.setdefault(session, r)
    return OutcomeSims(entrants=entrants["athlete_id"].tolist(), rank=pos, finished=~sim["dnf"],
                       stage_rank=stage_rank, points=sim["points"],
                       groups=entrants["team_key"].tolist() if "team_key" in entrants else None,
                       indicators={"race_fastest_lap": sim["fl"]} if "fl" in sim else {},
                       stage_finished=finished, stage_points=points)


def from_timed_runs(riders, sim):
    """Downhill (timed_runs): `final_rank` is inf for riders who didn't run or finish the Final."""
    fr = sim["final_rank"]
    return OutcomeSims(entrants=list(riders), rank=fr, finished=np.isfinite(fr),
                       stage_rank={"qual": sim["qual_rank"]}, reached={"final": sim["made_final"]},
                       points=sim["points"])


# --- prediction records (docs/backtest-core.md, "Prediction records") --------------------------------------

RECORD_COLUMNS = ["run_id", "sport", "model_id", "season", "event_id", "event", "stage", "cutoff", "kind", "subject",
                  "params", "fair", "se", "n_sims"]


def _py(v):
    """numpy scalars -> plain Python (JSON, record columns)."""
    return v.item() if isinstance(v, np.generic) else v


def _id_value(v):
    v = _py(v)
    if v is None or isinstance(v, (int, str)):
        return v
    try:
        return int(v) if float(v) == int(v) else str(v)
    except (TypeError, ValueError):
        return str(v)


def _se(p, n):
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def to_records(sims, run_id, sport, model_id, season, event_id, event, stage, cutoff, kinds=None):
    """Long-format prediction records for one event x stage: one row per kind x subject with the fair value,
    its Monte Carlo standard error sqrt(p(1-p)/n_sims) and n_sims. Every kind the simulations support
    (markets.kinds.summary: per-entrant kinds; head-to-heads for each pair once, as walk_forward.event_rows;
    the top team when points and groups exist), or just `kinds`. `subject` is the athlete id (the group key
    for group_top); `params` is JSON: {} | {"opponent_id": b} | {"team": g}."""
    import json

    from racinglines.markets import kinds as K
    n = sims.n_sims
    summ = K.summary(sims)
    want = list(kinds) if kinds is not None else list(K.KINDS)
    rows = []

    def add(kind, subject, params, p):
        p = float(p)
        rows.append((kind, str(_py(subject)), json.dumps(params, sort_keys=True), p, _se(p, n)))

    for kind in want:
        k = K.KINDS.get(kind)
        if k is None or k.payoff == "standings" or (kinds is None and not k.default):
            continue
        s = sims
        if k.session is not None:              # a kind of an earlier round (the sprint): that round's view
            if k.session not in sims.stage_rank:
                continue
            s = sims.at(k.session)
        if k.payoff in ("top_n", "stage_top_n", "reached", "indicator"):
            if kind not in summ:
                continue
            for a, p in zip(sims.entrants, summ[kind]):
                add(kind, a, {}, p)
        elif k.payoff == "h2h":
            h = K.h2h_matrix(s)
            for i, a in enumerate(sims.entrants):
                for j, b in enumerate(sims.entrants[i + 1:], i + 1):
                    add(kind, a, {"opponent_id": _py(b)}, h[i, j])
        elif k.payoff == "group_top" and s.groups is not None and s.points is not None:
            for g, p in K.group_top(s).items():
                add(kind, g, {"team": _py(g)}, p)
    df = pd.DataFrame(rows, columns=["kind", "subject", "params", "fair", "se"])
    ts = pd.Timestamp(cutoff).as_unit("ns") if cutoff is not None else pd.NaT
    head = dict(run_id=int(run_id), sport=sport, model_id=model_id, season=None if season is None else int(season),
                event_id=_id_value(event_id), event=None if event is None else str(event), stage=stage, cutoff=ts)
    for c, v in reversed(list(head.items())):
        df.insert(0, c, v)
    df["n_sims"] = int(n)
    return df[RECORD_COLUMNS]


def save_sims(sims, path):
    """Archive an OutcomeSims to one compressed .npz (the arrays, plus a JSON `meta` string for the entrants,
    groups and the stage_rank / reached / indicators keys); `load_sims` restores it exactly."""
    import json
    arrays = dict(rank=sims.rank, finished=sims.finished)
    if sims.points is not None:
        arrays["points"] = sims.points
    for k, v in sims.stage_rank.items():
        arrays[f"stage_rank__{k}"] = v
    for k, v in sims.reached.items():
        arrays[f"reached__{k}"] = v
    for k, v in sims.indicators.items():
        arrays[f"indicators__{k}"] = v
    for k, v in sims.stage_finished.items():
        arrays[f"stage_finished__{k}"] = v
    for k, v in sims.stage_points.items():
        arrays[f"stage_points__{k}"] = v
    meta = dict(entrants=[_py(a) for a in sims.entrants], stage_rank=list(sims.stage_rank), reached=list(sims.reached),
                groups=None if sims.groups is None else [_py(g) for g in sims.groups],
                has_points=sims.points is not None)
    if sims.indicators:                       # left out when empty, so earlier archives' meta is unchanged
        meta["indicators"] = list(sims.indicators)
    if sims.stage_finished or sims.stage_points:     # the same: only archives of a weekend with side stages
        meta["stage_finished"], meta["stage_points"] = list(sims.stage_finished), list(sims.stage_points)
    with open(path, "wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta)), **arrays)


def load_sims(path):
    import json
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return OutcomeSims(
            entrants=meta["entrants"], rank=z["rank"], finished=z["finished"],
            stage_rank={k: z[f"stage_rank__{k}"] for k in meta["stage_rank"]},
            reached={k: z[f"reached__{k}"] for k in meta["reached"]},
            points=z["points"] if meta["has_points"] else None, groups=meta["groups"],
            indicators={k: z[f"indicators__{k}"] for k in meta.get("indicators", [])},
            stage_finished={k: z[f"stage_finished__{k}"] for k in meta.get("stage_finished", [])},
            stage_points={k: z[f"stage_points__{k}"] for k in meta.get("stage_points", [])})

