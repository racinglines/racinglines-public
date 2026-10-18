"""
Combos (same-game parlays): one selection made of several legs on the same race, e.g. a sportsbook's "Race Winner and
Fastest Lap". A combo wins when every leg wins, so its fair value is the share of simulations in which all legs hold
at once, read from the same OutcomeSims (racinglines/models/outcomes.py) the single markets are priced from. Legs that
move together (the winner usually sets the fastest lap) price above the product of their marginals.

    leg_hits(sims, leg)                 (n_sims,) per-simulation value of one leg: 1 / 0, or a share in [0, 1]
    combo_fair(legs, sims)              P(every leg holds): the mean over simulations of the legs' product
    price(legs, sims)                   fair, each leg's marginal, their product and the lift (fair / product)
    settle(legs, res, void_leg)         YES / NO / None (void, undecidable or manual) from the results frame
    check_legs(legs)                    the legs, or ValueError naming the bad leg and field

A leg is a table: `kind` (a code of the registry, racinglines/markets/kinds.py KINDS, declarative kinds included), the
subject the kind names, and an optional side:

    {"kind": "race_win", "athlete": 830}                      a driver (athlete id)
    {"kind": "race_constructor_top", "team": "mclaren"}       a team (the model's team_key)
    {"kind": "race_h2h", "pair": [830, 844]}                  a head-to-head: the first finishes ahead of the second
    {"kind": "race_n_classified", "line": 18.5}               a field kind with compare over: P(count > line)
    "side": "yes" | "no" | "over" | "under"                   no / under take the complement (default yes / over)

What a kind names comes from its definition, not from a branch per kind: a declarative kind's spec `subject` (driver,
team, field; a `compare` adds the line), a legacy kind's payoff (SUBJECT below). Each payoff's per-simulation value is
one entry of HITS. A kind that can't be decided per simulation (a season standings market, a history-rate prop such as
the safety car, a yes/no these simulations don't draw) raises ValueError naming the leg.

Per-simulation values are 1 / 0 except two shares: the top-team payoff splits a points tie (a dead heat), and the n-th
retirement predicate is the uniform timing assumption's 1 / n_dnf (markets/payoffs.py). The product over legs is
right for a share as long as its randomness is independent of the other legs; two legs on the n-th retirement share
one timing draw, so a combo may carry at most one such leg (ValueError otherwise).

Settlement (book schema `void_leg`, docs/sportsbook/index.md): any losing leg settles the combo NO; a void or
undecidable leg voids it all ("void_all", the default) or is dropped and the rest decide ("drop_leg"). A leg the
results can't record (the fastest lap: the classification has no lap times) returns None, a manual settlement.
"""

import numpy as np

from racinglines.markets import kinds as K
from racinglines.markets import payoffs as P

SIDES = {"yes": False, "over": False, "no": True, "under": True}     # side -> take the complement
VOID_LEG = ("void_all", "drop_leg")
SHARED_DRAW = ("nth_retired",)        # predicates whose share comes from one extra draw shared by every leg using it
MANUAL = ("indicator",)               # legacy payoffs the results frame can't settle (the fastest lap)

# legacy payoff -> what a leg names ("athlete", "team", "pair"); a declarative kind's comes from its spec
SUBJECT = {"top_n": "athlete", "stage_top_n": "athlete", "reached": "athlete", "indicator": "athlete",
           "mover": "athlete", "h2h": "pair", "group_top": "team"}


def _col(sims, a):
    try:
        return sims.index(a)
    except ValueError:
        raise ValueError(f"athlete {a!r} is not an entrant of these simulations") from None


def _group_share(sims, team):
    """(n_sims,) the team's share of the top-team payoff in each simulation (a points tie splits it)."""
    if sims.points is None or sims.groups is None:
        raise ValueError("group markets need points and groups")
    keys = sorted(set(sims.groups))
    if team not in keys:
        raise ValueError(f"team {team!r} is not a group of these simulations ({keys})")
    tot = np.zeros((sims.n_sims, len(keys)))
    np.add.at(tot.T, np.array([keys.index(g) for g in sims.groups]), sims.points.T)
    best = tot == tot.max(axis=1, keepdims=True)
    return (best / best.sum(axis=1, keepdims=True))[:, keys.index(team)]


def _indicator(sims, k, leg):
    if k.code not in sims.indicators:
        raise ValueError("these simulations don't draw it (F1: price with the fastlap or flpos variant)")
    return sims.indicators[k.code][:, _col(sims, leg["athlete"])]


def _mover(sims, k, leg):
    if k.stage not in sims.stage_rank:
        raise ValueError(f"these simulations have no {k.stage} order")
    return K.biggest_mover(sims.stage_rank[k.stage], sims.rank, sims.finished)[:, _col(sims, leg["athlete"])]


def _stage_top_n(sims, k, leg):
    if k.stage not in sims.stage_rank:
        raise ValueError(f"these simulations have no {k.stage} stage")
    return sims.stage_rank[k.stage][:, _col(sims, leg["athlete"])] <= k.n


def _reached(sims, k, leg):
    if k.stage not in sims.reached:
        raise ValueError(f"these simulations have no {k.stage} round")
    return sims.reached[k.stage][:, _col(sims, leg["athlete"])]


def _h2h(sims, k, leg):
    a, b = leg["pair"]
    return sims.rank[:, _col(sims, a)] < sims.rank[:, _col(sims, b)]


# legacy payoff -> (sims, kind, leg) -> (n_sims,) per-simulation value, the same rule as kinds.fair's
HITS = {
    "top_n": lambda s, k, leg: (s.rank[:, _col(s, leg["athlete"])] <= k.n) & s.finished[:, _col(s, leg["athlete"])],
    "stage_top_n": _stage_top_n,
    "reached": _reached,
    "indicator": _indicator,
    "mover": _mover,
    "h2h": _h2h,
    "group_top": lambda s, k, leg: _group_share(s, leg["team"]),
}


def _spec_hits(sims, spec, leg):
    """A declarative kind's per-simulation value: the spec's predicate (payoffs.PREDICATES), on the leg's driver, or
    aggregated over the leg's team or the field (payoffs.AGGREGATES), compared with the leg's line when it has one."""
    v = P.PREDICATES[spec["predicate"]][0](sims, spec)
    if spec["subject"] == "driver":
        return v[:, _col(sims, leg["athlete"])]
    if spec["subject"] == "team":
        if sims.groups is None:
            raise ValueError("group markets need groups")
        g = np.array(sims.groups)
        if leg["team"] not in set(sims.groups):
            raise ValueError(f"team {leg['team']!r} is not a group of these simulations")
        v = v[:, g == leg["team"]]
    x = P.AGGREGATES[spec["aggregate"]][0](v)
    if spec.get("compare") is not None:
        return P.COMPARES[spec["compare"]][0](x, leg["line"])
    return np.minimum(x, 1.0) if spec["aggregate"] in P.COUNTS else x


def needs(kind):
    """The fields a leg of `kind` must carry besides kind, e.g. ("athlete",) or ("line",). ValueError for a kind a
    combo can't take (unknown, a season standings market, a history-rate prop)."""
    k = K.KINDS.get(kind)
    if k is None:
        raise ValueError(f"{kind!r} is not a kind priced from simulations (kinds.KINDS)")
    if k.spec is not None:
        spec = k.spec["payoff"]
        sub = {"driver": ("athlete",), "team": ("team",), "field": ()}[spec["subject"]]
        return sub + (("line",) if spec.get("compare") is not None else ())
    if k.payoff not in SUBJECT:
        raise ValueError(f"{kind} ({k.payoff}) is not decided per simulation of one race")
    return (SUBJECT[k.payoff],)


def check_legs(legs):
    """The legs unchanged, or ValueError naming the first bad one: at least two, each a table with a known kind, the
    fields that kind names, a known side; at most one leg on a shared extra draw (SHARED_DRAW)."""
    if not isinstance(legs, (list, tuple)) or len(legs) < 2:
        raise ValueError("a combo needs a list of at least two legs")
    shared = 0
    for i, leg in enumerate(legs):
        where = f"legs[{i}]"
        if not isinstance(leg, dict):
            raise ValueError(f"{where}: must be a table with a kind")
        try:
            need = needs(leg.get("kind"))
        except ValueError as e:
            raise ValueError(f"{where}.kind: {e}") from None
        for f in need:
            if leg.get(f) is None:
                raise ValueError(f"{where}: kind {leg['kind']} needs {f}")
        if "pair" in need and (not isinstance(leg["pair"], (list, tuple)) or len(leg["pair"]) != 2
                               or leg["pair"][0] == leg["pair"][1]):
            raise ValueError(f"{where}.pair: two different athletes")
        if "line" in need and (isinstance(leg["line"], bool) or not isinstance(leg["line"], (int, float))):
            raise ValueError(f"{where}.line: {leg['line']!r} is not a number")
        if leg.get("side", "yes") not in SIDES:
            raise ValueError(f"{where}.side: {leg.get('side')!r} not in {tuple(SIDES)}")
        k = K.KINDS[leg["kind"]]
        shared += bool(k.spec is not None and k.spec["payoff"]["predicate"] in SHARED_DRAW)
    if shared > 1:
        raise ValueError(f"at most one leg on {SHARED_DRAW}: their shares come from one shared timing draw")
    return legs


def leg_hits(sims, leg):
    """(n_sims,) float: the leg's value in each simulation, 1 / 0 (or a share, see the module docstring), with the
    leg's side applied. ValueError naming the leg's kind when these simulations can't decide it."""
    try:
        for f in needs(leg.get("kind")):
            if leg.get(f) is None:
                raise ValueError(f"needs {f}")
        k = K.KINDS[leg["kind"]]
        s = sims.at(k.session) if k.session is not None else sims
        v = _spec_hits(s, k.spec["payoff"], leg) if k.spec is not None else HITS[k.payoff](s, k, leg)
    except ValueError as e:
        raise ValueError(f"leg {leg.get('kind')}: {e}") from None
    v = np.asarray(v, float)
    side = leg.get("side", "yes")
    if side not in SIDES:
        raise ValueError(f"leg {leg['kind']}: side {side!r} not in {tuple(SIDES)}")
    return 1.0 - v if SIDES[side] else v


def combo_fair(legs, sims):
    """P(every leg holds) from one OutcomeSims: the mean over simulations of the product of the legs' values."""
    check_legs(legs)
    hit = np.ones(sims.n_sims)
    for leg in legs:
        hit = hit * leg_hits(sims, leg)
    return float(hit.mean())


def price(legs, sims):
    """dict(fair, legs=[each leg's marginal], independent=the product of the marginals, lift=fair / independent,
    se=the Monte Carlo standard error of fair, n_sims)."""
    check_legs(legs)
    vals = [leg_hits(sims, leg) for leg in legs]
    hit = np.prod(vals, axis=0)
    fair = float(hit.mean())
    marg = [float(v.mean()) for v in vals]
    ind = float(np.prod(marg))
    return dict(fair=fair, legs=marg, independent=ind, lift=fair / ind if ind > 0 else None,
                se=float(hit.std(ddof=0) / np.sqrt(sims.n_sims)), n_sims=int(sims.n_sims))


# --- settlement ----------------------------------------------------------------------------------------------------

def _leg_params(leg):
    """(athlete_id, params) the single-market settle() takes for one leg."""
    if "pair" in leg:
        return leg["pair"][0], {"opponent_id": leg["pair"][1]}
    params = {}
    if "team" in leg:
        params["team"] = leg["team"]
    if "line" in leg:
        params["line"] = leg["line"]
    return leg.get("athlete"), params


def settle(legs, res, void_leg="void_all", group_key=None):
    """YES / NO / None from the results frame (private_book.race_outcomes' columns), every leg by the single market's
    own rule (kinds.settle). Any losing leg: NO. Otherwise a leg the results can't settle (MANUAL, the fastest lap):
    None, a manual settlement. A void or undecidable leg: None with void_all; dropped with drop_leg (None if none is left)."""
    check_legs(legs)
    if void_leg not in VOID_LEG:
        raise ValueError(f"void_leg {void_leg!r} not in {VOID_LEG}")
    out, manual = [], False
    for leg in legs:
        k = K.KINDS[leg["kind"]]
        if k.spec is None and k.payoff in MANUAL:
            manual = True
            continue
        a, params = _leg_params(leg)
        v = K.settle(leg["kind"], a, params, res, group_key=group_key)
        if v is not None and SIDES[leg.get("side", "yes")]:
            v = not v
        out.append(v)
    if False in out:
        return False
    if manual:
        return None
    if None in out:
        if void_leg == "void_all":
            return None
        out = [v for v in out if v is not None]
        return True if out else None
    return True
