"""
Market kinds as payoffs (docs/backtest-core.md): one definition per kind gives both its fair value,
from a model's simulations (racinglines/models/outcomes.py), and its settlement, from the official
result. Any sport whose model returns an `OutcomeSims` gets every kind here without new code.

    KINDS["race_podium"]                 the definition: top_n(3)
    fair("race_podium", sims)            per-entrant fair values, aligned with sims.entrants
    fair("race_h2h", sims, a, b)         one pair: P(a classified ahead of b)
    settle("race_win", athlete_id, params, res)   YES / NO / None (undecidable) from the classification

Settlement reads the result frame of private_book.race_outcomes (athlete_id, position, status,
qual_position, team_id, points) and is what private_book.outcome_for returns.

Every kind is a row of markets/kinds.toml, so a new kind is a table in that file, not a branch here (owner,
2026-10-06: "as few conditional code switches in the model, and more generalized support functions who's inputs are
set by the schemas"). Two kinds of row. The declarative kinds (the sportsbook classification and retirement markets,
the exact place, the team pole and points rank, ...) are specs: subject / predicate / aggregate / compare, priced and
settled by the two generic functions of racinglines/markets/payoffs.py. The legacy kinds name a payoff instead (top_n,
stage_top_n, reached, h2h, group_top, standings, indicator, mover), each one function below whose inputs (n, stage,
session, exact) come from the row: their fair values feed the golden tests and the live book, and moving them onto
specs is a later change under the promotion rule (docs/f1-roadmap.md).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from racinglines.markets import payoffs as P

first_retired = P.first_retired          # re-exported (tests/test_kinds.py)


@dataclass(frozen=True)
class Kind:
    code: str
    payoff: str               # top_n | stage_top_n | h2h | reached | group_top | indicator | mover | standings (not from
                              # sims) | spec (a declarative kind: markets/kinds.toml, priced and settled by payoffs.py)
                              # | prop (a history-rate race prop, PROPS below: not in KINDS)
    n: int | None = None      # top_n / stage_top_n
    stage: str | None = None  # stage_top_n / mover: which earlier round (sims.stage_rank key); reached: which round
    label: str = ""
    default: bool = True      # False: priced only when named (fair(), a model's own summary); left out of summary(),
                              # to_records and the walk-forward default set, so adding one changes no existing output
    spec: dict | None = field(default=None, compare=False)   # payoff "spec": {"payoff": {...}, "settle": {...}}
    wx: str | None = None     # how a probability of rain moves its price (models/props.py): None, not at
                              # all; "rate" a yes/no rate priced with props.rate_wx; "dnf" its drivers' dnf_prob scaled by
                              # props.wx_scale before pricing. Declarative kinds: `wx = "..."` in markets/kinds.toml
    session: str | None = None   # a market on an earlier round raced as its own classification (F1: "sprint"):
                                 # priced on sims.at(session), settled on the result's <session>_ columns
    exact: bool = False       # standings: P(position == n) instead of P(position <= n)
    subject: str = "driver"   # who a selection names: driver | team | field | pair (h2h); links and team settlement


# The legacy payoffs a row may name (payoff = "<name>" in markets/kinds.toml), each with the row fields it reads.
LEGACY_PAYOFFS = {"top_n": ("n",), "stage_top_n": ("n", "stage"), "reached": ("stage",), "h2h": (), "group_top": (),
                  "indicator": (), "mover": ("stage",), "standings": ()}
_SUBJECT = {"h2h": "pair", "group_top": "team"}       # a legacy payoff's subject when the row doesn't name one


def legacy_kind(e, where="kinds.toml"):
    """A Kind from a legacy row of markets/kinds.toml (payoff a name), or ValueError naming the bad field."""
    payoff = e["payoff"]
    if payoff not in LEGACY_PAYOFFS:
        raise ValueError(f"{where}.payoff: {payoff!r} not in {tuple(LEGACY_PAYOFFS)} (or a spec table)")
    for f in LEGACY_PAYOFFS[payoff]:
        if e.get(f) is None:
            raise ValueError(f"{where}.{f}: payoff {payoff} needs it")
    unknown = set(e) - {"code", "label", "payoff", "n", "stage", "session", "default", "exact", "subject"}
    if unknown:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}")
    return Kind(e["code"], payoff, n=e.get("n"), stage=e.get("stage"), label=e.get("label", e["code"]),
                default=e.get("default", True), session=e.get("session"), exact=e.get("exact", False),
                subject=e.get("subject", _SUBJECT.get(payoff, "driver")))


def spec_kind(e):
    """A Kind from a declarative row (payoffs.load's entry)."""
    return Kind(e["code"], "spec", label=e["label"], default=e["default"], spec=e, wx=e["wx"], session=e["session"],
                subject=e["payoff"]["subject"])


def load(path=None):
    """{code: Kind} for every row of markets/kinds.toml (or `path`) in file order: the legacy payoffs, then the
    declarative ones (the file keeps them in that order, so the declarative kinds stay last in the registry)."""
    specs = P.load(path)
    out = {}
    for i, e in enumerate(P.entries(path)):
        where = f"kinds[{i}] ({e['code']})"
        if e["code"] in specs:
            out[e["code"]] = spec_kind(specs[e["code"]])
        elif any(k.spec is not None for k in out.values()):
            raise ValueError(f"{where}: a legacy payoff after a declarative kind (keep them first)")
        else:
            out[e["code"]] = legacy_kind(e, where)
    return out


# Every kind, in the order the readers list prediction kinds (db/reads.PREDICTION_KINDS is derived from it).
# payoff "standings": season-long markets priced from a season simulation (season_fair), not from an OutcomeSims.
KINDS = load()
LEGACY = tuple(k for k in KINDS.values() if k.spec is None)

# The race props priced from the race history (models/props.py), not from simulations: kinds for their
# `wx` only, kept out of KINDS so the prediction kinds (db/reads.PREDICTION_KINDS) and every summary stay as they are.
# The red flag is priced given a probability of rain; the safety car is not (conditioning it was worse in the
# walk-forward, decision log 2026-10-06) and rain is the forecast itself.
PROPS = (
    Kind("race_safety_car", "prop", label="Safety car", default=False),
    Kind("race_red_flag", "prop", label="Red flag", default=False, wx="rate"),
    Kind("race_rain", "prop", label="Rain", default=False),
)


def wx_kinds(how):
    """The codes whose `wx` is `how` ("rate" or "dnf"), props first, then KINDS in registry order."""
    return tuple(k.code for k in PROPS + tuple(KINDS.values()) if k.wx == how)


# --- fair values ---------------------------------------------------------------------------------------

def fair(kind, sims, a=None, b=None, line=None):
    """Fair probability of YES. Per-entrant kinds return an array over sims.entrants (or one value with
    `a` = an athlete id); race_h2h needs a and b; race_constructor_top and the team kinds of markets/kinds.toml
    return {group: probability}; its field kinds with compare over (race_n_classified, race_n_retirements) need
    `line` and return P(over)."""
    k = KINDS[kind]
    if k.session is not None:                  # an earlier round's own classification (the sprint)
        sims = sims.at(k.session)
    if k.spec is not None:
        try:
            return P.fair(k.spec["payoff"], sims, a=a, line=line, b=b)
        except ValueError as e:
            raise ValueError(f"{kind}: {e}") from None
    if k.payoff == "h2h":
        if a is None:
            return h2h_matrix(sims)
        i, j = sims.index(a), sims.index(b)
        return float((sims.rank[:, i] < sims.rank[:, j]).mean())
    if k.payoff == "group_top":
        return group_top(sims)
    if k.payoff == "top_n":
        p = ((sims.rank <= k.n) & sims.finished).mean(0)
    elif k.payoff == "stage_top_n":
        p = (sims.stage_rank[k.stage] <= k.n).mean(0)
    elif k.payoff == "reached":
        p = sims.reached[k.stage].mean(0)
    elif k.payoff == "indicator":
        if kind not in sims.indicators:
            raise ValueError(f"{kind}: these simulations don't draw it")
        p = sims.indicators[kind].mean(0)
    elif k.payoff == "mover":
        if k.stage not in sims.stage_rank:
            raise ValueError(f"{kind}: these simulations have no {k.stage} order")
        p = biggest_mover(sims.stage_rank[k.stage], sims.rank, sims.finished).mean(0)
    elif k.payoff == "standings":
        raise ValueError(f"{kind} is a standings market: not priced from an OutcomeSims")
    else:
        raise ValueError(f"unknown payoff {k.payoff}")
    return p if a is None else float(p[sims.index(a)])


def biggest_mover(grid, rank, finished):
    """(n_sims, n) bool: who has the largest gain from `grid` to `rank` among the cars that finished, in each
    simulation; nobody when no one gained, everyone tied on the largest gain."""
    gain = np.where(finished, np.asarray(grid, float) - rank, -np.inf)
    best = gain.max(axis=1, keepdims=True)
    return (gain == best) & (best > 0)


def h2h_matrix(sims):
    """P(row entrant classified ahead of column entrant): the ranks as simulated (F1: retirements are
    classified behind finishers)."""
    r = sims.rank
    return (r[:, :, None] < r[:, None, :]).mean(0)


def group_top(sims):
    """{group: P(its entrants score the most points)}; a tie counts as a split win."""
    if sims.points is None or sims.groups is None:
        raise ValueError("group markets need points and groups")
    keys = sorted(set(sims.groups))
    tot = np.zeros((sims.n_sims, len(keys)))
    np.add.at(tot.T, np.array([keys.index(g) for g in sims.groups]), sims.points.T)
    best = tot == tot.max(axis=1, keepdims=True)
    share = best / best.sum(axis=1, keepdims=True)
    return {g: float(share[:, i].mean()) for i, g in enumerate(keys)}


def summary(sims):
    """Per entrant: the fair value of every per-entrant kind the simulations support."""
    out = dict(athlete_id=sims.entrants)
    for payoff in ("top_n", "stage_top_n", "reached", "indicator"):     # columns grouped by payoff, in registry order
        for code, k in KINDS.items():
            if k.payoff != payoff or not k.default:
                continue
            if payoff == "indicator":
                has = code in sims.indicators
            else:
                has = payoff == "top_n" or k.stage in (sims.stage_rank if payoff == "stage_top_n" else sims.reached)
            if has:
                out[code] = fair(code, sims)
    return pd.DataFrame(out)


# --- season-long markets, from a season simulation ------------------------------------------------------------

def season_fair(kind, ss, a=None, b=None, n=None):
    """Fair probability of YES for a "standings" kind from an outcomes.SeasonSims (a season simulation; `fair`
    above stays for race markets). A row with n (champion, standings_top3, standings_p2 .. p5, and the constructors'
    on a season simulation whose entrants are the teams): P(position <= n), or == n when the row says exact, per
    entrant (or one value with `a`); standings_h2h: the matrix, or P(a finishes the season ahead of b);
    season_wins_ge: P(at least n wins)."""
    k = KINDS[kind]
    if k.payoff != "standings":
        raise ValueError(f"{kind} is a race market: priced by fair() from an OutcomeSims")
    if k.n is not None:
        p = (ss.rank == k.n).mean(0) if k.exact else standings_position(ss, k.n)
    elif kind == "standings_h2h":
        if a is None:
            r = ss.rank
            return (r[:, :, None] < r[:, None, :]).mean(0)
        return float((ss.rank[:, ss.index(a)] < ss.rank[:, ss.index(b)]).mean())
    elif kind == "season_wins_ge":
        if ss.wins is None or n is None:
            raise ValueError("season_wins_ge needs simulated wins and n")
        p = (ss.wins >= n).mean(0)
    else:
        raise ValueError(f"{kind} is not priced from a driver season simulation")
    return p if a is None else float(p[ss.index(a)])


def standings_position(ss, n):
    """P(finishing the season in the top `n` of the standings), per entrant: the points-position payoff
    (n = 1 is the champion)."""
    return (ss.rank <= n).mean(0)


# --- settlement ----------------------------------------------------------------------------------------

def session_result(res, session):
    """An earlier round's classification out of the race's result frame (private_book.race_outcomes adds
    <session>_position / _status / _points), as a result frame of its own: the drivers who took part in it.
    None when the frame doesn't carry it (that round's results aren't in yet)."""
    cols = {f"{session}_{c}": c for c in ("position", "status", "points")}
    if not set(cols) <= set(res.columns):
        return None
    out = res.drop(columns=[c for c in cols.values() if c in res]).rename(columns=cols)
    out = out[out["status"].notna()]
    return out if len(out) else None


def settle(kind, athlete_id, params, res, group_key=None):
    """YES/NO for a race market from the official classification (None if undecidable).
    group_key: maps a result's team_id to the group key the market names (F1: position_sim team_key).
    The n-th retirement kinds read a `laps_completed` column when the frame has one (the caller adds it; nothing here
    queries the database): the DNF with the fewest laps is first, every tied driver YES."""
    if res.empty:
        return None
    k = KINDS.get(kind)
    if k is not None and k.session is not None:
        res = session_result(res, k.session)
        if res is None:
            return None
    if k is not None and k.spec is not None:
        return P.settle(k.spec["settle"], athlete_id, params, res, group_key=group_key)
    if k is None or k.payoff == "standings":
        return None
    by = res.set_index("athlete_id")
    if k.payoff == "stage_top_n":
        col = "qual_position" if k.stage == "qual" else f"{k.stage}_position"   # sprint rounds: sprint_qual_position, ...
        if col not in res:
            return None
        if athlete_id not in by.index or pd.isna(by.loc[athlete_id, col]):
            return False if athlete_id not in by.index else None
        return bool(by.loc[athlete_id, col] <= k.n)
    if k.payoff == "top_n":
        rows = res["athlete_id"] == athlete_id
        if not rows.any():
            return False
        return bool(P.top_n(res, k.n)[rows].iloc[0])        # the one top-n rule (payoffs.top_n): classified in the top n
    if k.payoff == "h2h":
        b = (params or {}).get("opponent_id")
        if athlete_id not in by.index or b not in by.index:
            return None
        # classification order (retirements are classified behind finishers by laps completed); a driver with no
        # position is behind every driver with one; neither has one: NO (the walk-forward's rule since C9; unverified
        # against the venues: Polymarket's head-to-head rule resolves a tie 50-50, markets/settlement_rules.py)
        pa, pb = (float(by.loc[x, "position"]) if pd.notna(by.loc[x, "position"]) else np.inf for x in (athlete_id, b))
        return bool(pa < pb)
    if k.payoff == "reached":
        col = f"reached_{k.stage}"          # results that record the round (e.g. timed_runs: reached_final)
        if col not in res:
            return None
        return bool(by.loc[athlete_id, col]) if athlete_id in by.index else False
    if k.payoff == "indicator":
        # a yes/no the model draws (the fastest lap): settled from a column of that kind's name when the frame
        # carries one (models/model_global.py adds it from the race's laps), else undecidable
        if kind not in res or res[kind].isna().all():
            return None
        return bool(by.loc[athlete_id, kind]) if athlete_id in by.index else False
    if k.payoff == "mover":
        won, _ = P.biggest_mover(res)        # from the stored starting grid (the frame's `grid`); None when it can't say
        if won is None:
            return None
        rows = res["athlete_id"] == athlete_id
        return bool(won[rows].iloc[0]) if rows.any() else False
    if k.payoff == "group_top":
        keys =res["team_id"].map(group_key) if group_key else res["team_id"]
        pts = res.assign(tk=keys).groupby("tk")["points"].sum()
        return bool(pts.idxmax() == (params or {}).get("team")) if len(pts) else None
    return None
