"""
Declarative market kinds: a kind is a payoff spec read from markets/kinds.toml, and two generic functions turn any
spec into a fair value and a settlement. No per-kind branch: a spec names a predicate, an aggregate and a comparison,
each a small table of named functions below.

    load()                                    {code: entry} of the declarative rows of markets/kinds.toml, checked
    entries()                                 every row of markets/kinds.toml, the legacy payoffs included
    fair(spec, sims, a=None, line=None)       from a model's simulations (racinglines/models/outcomes.py)
    settle(spec, athlete_id, params, res)     YES / NO / None (undecidable) from the results frame
    check(spec)                               the spec, or ValueError naming the bad field

A spec (an entry's `payoff` table; `settle` holds the settlement view, the payoff table with its overrides):

    subject    driver | team | field        driver: per entrant; team: {group: p}; field: one value
    predicate  classified | retired | top | position | stage_top | indicator | points | last_classified |
               nth_retired                  (n_sims, n_cars), bool or float
    n          top / position / stage_top / nth_retired;  stage: stage_top;  of: indicator
    aggregate  any | all | sum | count | total   how a team's or the field's cars combine (none for driver)
    compare    over                         P(aggregate > line)
               rank (place)                 the team's place among the teams by aggregate == place
               ahead                        the team ahead of params.opponent (b when pricing) by aggregate
    absent     no | void                    settlement only: a driver missing from the results (default void)

    fair({"subject": "team", "predicate": "top", "n": 5, "aggregate": "any"}, sims)    a team with a car in the top 5

Owner (2026-10-06): "as few conditional code switches in the model, and more generalized support functions who's
inputs are set by the schemas". The classification, retirement and parity kinds (exact place, team pole, team points
rank, ...) are declared this way. The legacy kinds are rows of the same file too, with a named payoff instead of a
spec (top_n, stage_top_n, reached, h2h, group_top, standings, indicator, mover; racinglines/markets/kinds.py prices
and settles them): their fair values feed the golden tests and the live book, and moving them onto specs is a later
change under the promotion rule (docs/f1-roadmap.md). load() returns the declarative rows only; entries() every row.
"""

import tomllib
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.paths import ROOT

PATH = ROOT / "markets" / "kinds.toml"
RETIRED = ("DNF", "DSQ")      # result statuses that count as a retirement
SUBJECTS = ("driver", "team", "field")
ABSENT = {"no": False, "void": None}
WX = (None, "rate", "dnf")    # a kind's `wx`: how a probability of rain moves its price (markets/kinds.py Kind.wx)


# --- predicates: one car, every simulation or every result row ---------------------------------------------------

def last_classified(rank, finished):
    """(n_sims, n) bool: the classified car with the worst rank in each simulation (everyone tied on it counts);
    nobody when no car is classified."""
    r = np.where(finished, np.asarray(rank, float), -np.inf)
    worst = r.max(axis=1, keepdims=True)
    return (r == worst) & np.isfinite(worst)


def first_retired(finished):
    """(n_sims, n) float: each car's chance of being the first retirement in each simulation, under the uniform
    timing assumption (the simulations draw who retires, not when): 1 / n_dnf for every retiring car, 0 for the
    rest and for every car when nobody retires. Rows sum to 1, or 0 with no retirement."""
    dnf = ~np.asarray(finished, bool)
    n = dnf.sum(axis=1, keepdims=True)
    return np.where(dnf, 1.0 / np.maximum(n, 1), 0.0)


def nth_retired(finished, n):
    """(n_sims, n_cars) float: each car's chance of being the n-th retirement under the uniform timing assumption:
    first_retired's 1 / n_dnf for every retiring car in a simulation with at least n retirements, else 0."""
    dnf = ~np.asarray(finished, bool)
    enough = dnf.sum(axis=1, keepdims=True) >= n
    return np.where(enough, first_retired(finished), 0.0)


# Who counts as classified, by the sport's rule (sports/<code>.toml [results] classified; racinglines.sports):
CLASSIFIED = {
    "status_ok": lambda res: res["status"] == "OK",       # F1 (ClassifiedPosition a number), MotoGP (INSTND)
    "placed": lambda res: res["position"].notna(),        # NASCAR: every car that started is ranked, DNFs by laps run
}
CLASSIFIED_COL = "classified"


def mark_classified(res, rule):
    """The results frame with its `classified` column set by the sport's rule (None: left as it is)."""
    if rule is None or res is None or not len(res):
        return res
    return res.assign(**{CLASSIFIED_COL: CLASSIFIED[rule](res).astype(bool)})


def mark_sport(res, sport):
    """mark_classified with the rule of sports/<sport>.toml (unknown or None sport: left as it is)."""
    from racinglines import sports
    return mark_classified(res, sports.classified_rule(sport) if sport else None)


def classified(res):
    """bool per row of a results frame: classified in the official result. The one classification of every settlement
    (top-n, exact place, last classified, the biggest mover; the walk-forward, the replays and the private book): the
    frame's `classified` column, which the loaders set from the sport's [results] classified rule (mark_sport), else
    status OK (the "status_ok" default) (docs/f1-roadmap.md decision log, 2026-10-07)."""
    if CLASSIFIED_COL in res:
        return res[CLASSIFIED_COL].fillna(False).astype(bool)
    return CLASSIFIED["status_ok"](res)


def top_n(res, n):
    """bool per row: classified (by the sport's rule) and placed in the top `n`. The top-n kinds' settlement
    everywhere (kinds.settle top_n, the "top" predicate here, position_replay and season_sweep through kinds.settle)."""
    return classified(res) & (res["position"] <= n)


GRID = "grid"           # the results frame's starting-grid column (results.extra.grid of the race round; 0 = pit lane)


def biggest_mover(res):
    """(outcome per row, reason): the race's biggest mover from the stored starting grid (docs/f1-roadmap.md decision
    log, 2026-10-04): the classified driver with the largest gain from `grid` to `position`, if anyone gained; every
    driver tied on that gain is YES, everyone else NO (nobody gained: NO for all). The outcome is None, with the
    reason, when the stored data can't decide it:
      * no `grid` column, or a classified driver without a stored grid slot;
      * a classified pit-lane starter (grid 0: FastF1's GridPosition, stored by sources/fastf1/ingest.py) whose gain
        could reach the best one. The venue's rule (Kalshi: "the largest positive differential between their starting
        grid position and their finishing position") doesn't say what a pit-lane start counts as, so the outcome is
        settled only when no reading of it changes the answer: at most the slot behind every car that started (the
        cars with a status other than DNS)."""
    if GRID not in res:
        return None, "no starting grid in the results"
    ok = classified(res)
    grid = pd.to_numeric(res[GRID], errors="coerce")
    if grid[ok].isna().any():
        return None, "a classified driver has no stored starting grid slot"
    pit = ok & (grid <= 0)
    gain = (grid - res["position"].astype(float)).where(ok & ~pit)
    best = gain.max() if gain.notna().any() else -np.inf
    if pit.any():
        last = int((res["status"] != "DNS").sum())
        if ((last - res.loc[pit, "position"].astype(float)) >= max(best, 1)).any():
            return None, "a pit-lane starter may be the biggest mover, depending on the slot the venue counts"
    if not best > 0:
        return pd.Series(False, index=res.index, dtype=object), "nobody gained a place"
    return (gain == best).astype(object), f"largest gain {best:g}"


def stage_col(stage):
    """The results frame's column for an earlier round's order: qual_position, or <stage>_position (sprint_qual, ...)."""
    return "qual_position" if stage == "qual" else f"{stage}_position"


def _stage_rank(sims, s):
    if s["stage"] not in sims.stage_rank:
        raise ValueError(f"these simulations have no {s['stage']} order")
    return sims.stage_rank[s["stage"]]


def _indicator(sims, s):
    if s["of"] not in sims.indicators:
        raise ValueError(f"these simulations don't draw {s['of']}")
    return sims.indicators[s["of"]]


def _points(sims, s):
    if sims.points is None:
        raise ValueError("these simulations have no points")
    return sims.points


def _res_stage_top(res, s):
    col = stage_col(s["stage"])
    if col not in res or res[col].isna().all():
        return None                              # that round hasn't run (or isn't stored)
    return (res[col] <= s["n"]).astype(object)   # a car with no place in it, once others have one: NO


def _res_indicator(res, s):
    """A drawn yes/no (the fastest lap) from the frame's column of that kind's name (models/model_global.py adds it
    from the race's laps); None when the frame has no such column: the venue settles it."""
    if s["of"] not in res or res[s["of"]].isna().all():
        return None
    return res[s["of"]].astype(object)


def _res_retired(res, s):
    v = res["status"].isin(RETIRED).astype(object)
    v[res["status"] == "DNS"] = None             # a non-starter: void, not a retirement
    return v


def _res_last_classified(res, s):
    ok = classified(res)
    if not ok.any():
        return None
    return ok & (res["position"] == res.loc[ok, "position"].max())


def _res_nth_retired(res, s):
    dnf = res["status"] == "DNF"                 # a DSQ is decided after the flag, not a car stopping first
    if not dnf.any():
        return dnf                               # nobody retired: every driver (and team) is a NO
    if "laps_completed" not in res or res.loc[dnf, "laps_completed"].isna().any():
        return None
    place = res.loc[dnf, "laps_completed"].rank(method="min")              # competition ranking: ties share the place
    return dnf & (place.reindex(res.index) == s["n"])


# name -> (on the simulations: (sims, spec) -> (n_sims, n_cars) array,
#          on the results: (res, spec) -> a value per row (True / False / None, or a number for points) or None when
#          the frame can't decide,
#          the fields it needs: n a positive integer, stage / of a name)
PREDICATES = {
    "classified": (lambda sims, s: sims.finished,
                   lambda res, s: classified(res), ()),
    "retired": (lambda sims, s: ~sims.finished,
                _res_retired, ()),
    "top": (lambda sims, s: (sims.rank <= s["n"]) & sims.finished,
            lambda res, s: top_n(res, s["n"]), ("n",)),
    "position": (lambda sims, s: (sims.rank == s["n"]) & sims.finished,
                 lambda res, s: classified(res) & (res["position"] == s["n"]), ("n",)),
    "stage_top": (lambda sims, s: _stage_rank(sims, s) <= s["n"],
                  _res_stage_top, ("n", "stage")),
    "indicator": (_indicator,
                  _res_indicator, ("of",)),     # the results' <of> column when the frame has one, else undecidable
    "points": (_points,
               lambda res, s: res["points"].astype(float).fillna(0.0), ()),
    "last_classified": (lambda sims, s: last_classified(sims.rank, sims.finished),
                        _res_last_classified, ()),
    "nth_retired": (lambda sims, s: nth_retired(sims.finished, s["n"]),
                    _res_nth_retired, ("n",)),
}


# --- aggregates: a team's or the field's cars -> one value per simulation, or one settlement ---------------------

def _vals(vals):
    """True / False / None (undecided) per car."""
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else bool(v) for v in vals]


def _any(vals):
    vals = _vals(vals)
    return True if True in vals else (None if None in vals else False)


def _all(vals):
    vals = _vals(vals)
    return False if False in vals else (None if None in vals else True)


def _count(vals):
    return sum(1 for v in _vals(vals) if v)


def _total(vals):
    return float(sum(0.0 if v is None or np.isnan(float(v)) else float(v) for v in vals))


# name -> (on the simulations: (n_sims, k) -> (n_sims,), on the results: [True / False / None] -> value)
AGGREGATES = {
    "any": (lambda v: v.any(axis=1), _any),
    "all": (lambda v: v.all(axis=1), _all),
    "sum": (lambda v: v.sum(axis=1), _count),       # per-car shares of one outcome (first retiring constructor)
    "count": (lambda v: v.sum(axis=1), _count),
    "total": (lambda v: np.asarray(v, float).sum(axis=1), _total),    # a number per car (points), summed
}
COUNTS = ("sum", "count")
NUMBERS = ("points",)          # predicates that give a number, not a yes/no: aggregated by total only


# name -> (on the simulations: (x, line) -> bool array, on the results: (x, line) -> bool)
COMPARES = {
    "over": (lambda x, line: x > line, lambda x, line: bool(x > float(line))),
}
TEAM_COMPARES = ("rank", "ahead")     # a team against the other teams (team_order), not against a line


def team_order(x, best):
    """(n_sims, k) arrays of each team's aggregate `x` and its best-placed car `best` (lower is better; inf: none
    ranked) -> (above, tied), (n_sims, k, k) bool: above[s, i, j] when team j is ahead of team i (a larger aggregate,
    or the same and a better-placed car: the constructors' countback), tied[s, i, j] when neither is ahead (j == i
    included)."""
    xi, xj = x[:, :, None], x[:, None, :]
    bi, bj = best[:, :, None], best[:, None, :]
    return (xj > xi) | ((xj == xi) & (bj < bi)), (xj == xi) & (bj == bi)


def place_share(above, tied, place):
    """(n_sims, k) float: each team's share of `place` (1 = first) in each simulation. A team with `a` teams ahead and
    `t` tied with it (itself included) holds places a+1 .. a+t, each with share 1/t."""
    a, t = above.sum(axis=2), tied.sum(axis=2)
    return np.where((a < place) & (place <= a + t), 1.0 / t, 0.0)


# --- the spec ------------------------------------------------------------------------------------------------------

def check(spec, where="spec"):
    """The spec unchanged, or ValueError naming the first bad field (`where`.field)."""
    def bad(field, msg):
        raise ValueError(f"{where}.{field}: {msg}")
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: must be a table")
    subject, pred, agg, comp = spec.get("subject"), spec.get("predicate"), spec.get("aggregate"), spec.get("compare")
    if subject not in SUBJECTS:
        bad("subject", f"{subject!r} not in {SUBJECTS}")
    if pred not in PREDICATES:
        bad("predicate", f"{pred!r} not in {tuple(PREDICATES)}")
    for f in PREDICATES[pred][2]:
        if f in ("stage", "of"):
            if not isinstance(spec.get(f), str) or not spec[f]:
                bad(f, f"predicate {pred} needs a name {f}, got {spec.get(f)!r}")
        elif isinstance(spec.get(f), bool) or not isinstance(spec.get(f), int) or spec[f] < 1:
            bad(f, f"predicate {pred} needs a positive integer {f}, got {spec.get(f)!r}")
    if subject == "driver":
        if agg is not None:
            bad("aggregate", "a driver kind has no aggregate")
        if pred in NUMBERS:
            bad("predicate", f"{pred} is a number: a driver kind needs a yes/no predicate")
    elif agg not in AGGREGATES:
        bad("aggregate", f"{agg!r} not in {tuple(AGGREGATES)}")
    elif (agg == "total") != (pred in NUMBERS):
        bad("aggregate", f"{agg!r} on predicate {pred}: total sums a number (points), the others count yes/no values")
    if comp is not None and comp not in COMPARES and comp not in TEAM_COMPARES:
        bad("compare", f"{comp!r} not in {tuple(COMPARES) + TEAM_COMPARES}")
    if comp in COMPARES and agg not in COUNTS:
        bad("compare", f"compares a count: aggregate must be one of {COUNTS}, got {agg!r}")
    if comp in TEAM_COMPARES and subject != "team":
        bad("compare", f"{comp} compares teams: subject must be team, got {subject!r}")
    if comp == "rank" and (isinstance(spec.get("place"), bool) or not isinstance(spec.get("place"), int)
                           or spec["place"] < 1):
        bad("place", f"compare rank needs a positive integer place, got {spec.get('place')!r}")
    if agg == "count" and comp is None:
        bad("compare", "a count needs a compare (e.g. over)")
    if agg == "total" and comp not in TEAM_COMPARES:
        bad("compare", f"a total is compared between teams: compare must be one of {TEAM_COMPARES}")
    if "absent" in spec and spec["absent"] not in ABSENT:
        bad("absent", f"{spec['absent']!r} not in {tuple(ABSENT)}")
    return spec


def load(path=None):
    """{code: {"code", "label", "default", "payoff", "settle", "wx", "session"}} for the declarative rows (payoff a
    table) in file order, every spec checked; `settle` is the payoff spec with the file's [kinds.settle] overrides.
    The legacy rows (payoff a name) are not here: entries() has every row. markets/kinds.toml is read once; another
    `path` each call."""
    return _default() if path is None else _parse(Path(path))


def entries(path=None):
    """Every [[kinds]] row of the file as written (legacy and declarative), in order, each code present and unique."""
    return _entries_default() if path is None else _entries(Path(path))


@cache
def _default():
    return _parse(PATH)


@cache
def _entries_default():
    return _entries(PATH)


def _entries(p):
    with open(p, "rb") as f:
        rows = tomllib.load(f).get("kinds", [])
    seen = set()
    for i, e in enumerate(rows):
        code = e.get("code")
        where = f"{p.name}: kinds[{i}]" + (f" ({code})" if code else "")
        if not isinstance(code, str) or not code:
            raise ValueError(f"{where}.code: missing")
        if code in seen:
            raise ValueError(f"{where}.code: {code} declared twice")
        if len(code) > 30:
            raise ValueError(f"{where}.code: longer than market_links.prediction's 30 characters")
        seen.add(code)
    return rows


def _parse(p):
    out = {}
    for i, e in enumerate(_entries(p)):
        code = e["code"]
        where = f"{p.name}: kinds[{i}] ({code})"
        if isinstance(e.get("payoff"), str):
            continue                             # a legacy payoff, priced and settled by racinglines/markets/kinds.py
        if e.get("default", False) is not False:
            raise ValueError(f"{where}.default: declarative kinds are default = false (summary() and the records "
                             "read only the legacy payoffs)")
        payoff = check(e.get("payoff"), f"{where}.payoff")
        settle = check({**payoff, **e.get("settle", {})}, f"{where}.settle")
        wx = e.get("wx")
        if wx not in WX:
            raise ValueError(f"{where}.wx: {wx!r} not in {WX}")
        session = e.get("session")
        if session is not None and (not isinstance(session, str) or not session):
            raise ValueError(f"{where}.session: {session!r} is not a round's name")
        out[code] = dict(code=code, label=e.get("label", code), default=False, payoff=payoff, settle=settle, wx=wx,
                         session=session)
    return out


# --- pricing -------------------------------------------------------------------------------------------------------

def _compare(s, x, line):
    if s.get("compare") is None:
        return x
    if line is None:
        raise ValueError("needs a line (e.g. 18.5)")
    return COMPARES[s["compare"]][0](x, line)


def _ahead(above, tied, i, j):
    """P(team i ahead of team j) given that they don't tie (a tie leaves the market undecided)."""
    pi, pj = above[:, j, i].mean(), above[:, i, j].mean()
    return float(pi / (pi + pj)) if pi + pj > 0 else 0.5


def fair(spec, sims, a=None, line=None, b=None):
    """Fair probability of YES from an OutcomeSims. driver: an array over sims.entrants (or one value with `a`);
    team: {group: probability} over sims.groups (compare ahead: {group: {opponent: p}}, or one value with `a` and `b`,
    two group keys); field: one probability (P(count > line) for compare over)."""
    s = check(spec)
    v = PREDICATES[s["predicate"]][0](sims, s)
    if s["subject"] == "driver":
        p = v.mean(0)
        return p if a is None else float(p[sims.index(a)])
    agg = AGGREGATES[s["aggregate"]][0]
    if s["subject"] == "field":
        return float(_compare(s, agg(v), line).mean())
    if sims.groups is None:
        raise ValueError("group markets need groups")
    g, keys = np.array(sims.groups), sorted(set(sims.groups))
    if s.get("compare") not in TEAM_COMPARES:
        return {key: float(_compare(s, agg(v[:, g == key]), line).mean()) for key in keys}
    x = np.stack([agg(v[:, g == key]) for key in keys], axis=1)
    best = np.stack([np.asarray(sims.rank, float)[:, g == key].min(axis=1) for key in keys], axis=1)
    above, tied = team_order(x, best)
    if s["compare"] == "rank":
        share = place_share(above, tied, s["place"]).mean(0)
        return {key: float(share[i]) for i, key in enumerate(keys)}
    if a is not None:
        return _ahead(above, tied, keys.index(a), keys.index(b))
    return {ka: {kb: _ahead(above, tied, i, j) for j, kb in enumerate(keys) if j != i} for i, ka in enumerate(keys)}


# --- settlement ----------------------------------------------------------------------------------------------------

def settle(spec, athlete_id, params, res, group_key=None):
    """YES / NO / None (undecidable) from the results frame (athlete_id, position, status, team_id, and
    laps_completed when the frame has one). driver: `athlete_id`; team: params["team"], results' team_id mapped
    through `group_key`; compare over: params["line"]; compare ahead: params["opponent"], another team."""
    s = check(spec)
    if res.empty:
        return None
    v = PREDICATES[s["predicate"]][1](res, s)
    if v is None:
        return None
    params = params or {}
    if s["subject"] == "driver":
        rows = res["athlete_id"] == athlete_id
        if not rows.any():
            return ABSENT[s.get("absent", "void")]
        return _any(list(v[rows]))
    if s["subject"] == "team":
        keys = res["team_id"].map(group_key) if group_key else res["team_id"]
        if s.get("compare") in TEAM_COMPARES:
            return _settle_teams(s, v, keys, res, params)
        rows = keys == params.get("team")
        if not rows.any():
            return None
        vals = list(v[rows])
    else:
        vals = list(v)
    x = AGGREGATES[s["aggregate"]][1](vals)
    if s.get("compare") is not None:
        line = params.get("line")
        return None if line is None else COMPARES[s["compare"]][1](x, line)
    return x if x is None or isinstance(x, bool) else bool(x > 0)


def _settle_teams(s, v, keys, res, params):
    """rank / ahead from the results: every team's aggregate and best-placed car (team_order), then the named team's
    place, or whether it is ahead of params["opponent"]. A tie that the countback doesn't break: undecided."""
    team, names = params.get("team"), sorted(k for k in set(keys) if k is not None and not pd.isna(k))
    if team not in names or (s["compare"] == "ahead" and params.get("opponent") not in names):
        return None
    agg = AGGREGATES[s["aggregate"]][1]
    pos = res["position"].astype(float).fillna(np.inf)
    x = np.array([[agg(list(v[keys == k])) for k in names]], float)
    if np.isnan(x).any():
        return None
    best = np.array([[pos[keys == k].min() for k in names]], float)
    above, tied = team_order(x, best)
    i = names.index(team)
    if s["compare"] == "ahead":
        j = names.index(params["opponent"])
        return None if tied[0, i, j] else bool(above[0, j, i])
    a, t = int(above[0, i].sum()), int(tied[0, i].sum())
    if t > 1 and a < s["place"] <= a + t:
        return None                              # tied for that place
    return a + 1 == s["place"]
