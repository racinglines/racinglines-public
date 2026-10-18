"""Position sizing shared by the taker strategies."""

import math


def target_shares(fair, price, cost, min_edge, stake_per_edge, max_stake, kelly=None, balance=None, root_unit=None):
    """(yes_shares, no_shares) to hold: nothing unless |fair - price| >= min_edge; otherwise
    a stake of stake_per_edge x |edge| (capped at max_stake) on the cheap side, paying cost
    per share on top of the price (YES at price, NO at 1 - price).

    With `kelly` (a fraction, e.g. 0.5 = half Kelly) and `balance` (the current bankroll), the stake is
    kelly x f* x balance instead, still capped at max_stake: f* = (q - c) / (1 - c) is the Kelly fraction
    for a contract paying 1 that costs c per share (price plus cost) and wins with probability q.

    With `root_unit` ($) instead, the root unit system (Dan Abrams, Circles Off ep. 181): the stake is the unit over
    the square root of the net odds, root_unit x sqrt(c / (1 - c)), still capped at max_stake: a unit at evens
    (c = 0.5), half a unit at 4/1 (c = 0.2), two units at 1/4 (c = 0.8). The edge only decides whether to bet."""
    edge = fair - price
    if abs(edge) < min_edge:
        return 0.0, 0.0
    c = price + cost if edge > 0 else 1 - price + cost
    if kelly is not None and balance is not None:
        q = fair if edge > 0 else 1 - fair
        f = (q - c) / (1 - c) if c < 1 else 0.0
        stake = min(max_stake, kelly * f * balance) if f > 0 else 0.0
    elif root_unit is not None:
        stake = min(max_stake, root_unit * math.sqrt(c / (1 - c))) if c < 1 else 0.0
    else:
        stake = min(max_stake, stake_per_edge * abs(edge))
    return (stake / c, 0.0) if edge > 0 else (0.0, stake / c)
