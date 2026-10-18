"""
The maker's quotes on a private book, shared by every live sport: fair +- a half-spread, leaning against
inventory, capped per market, with no quote on a decided market or outside 1-99 cents. The half-spread
itself (and any widening) is the sport's: downhill widens while a rider is on course, F1 tightens stage by
stage (sports/<code>.toml [live.quoting]).

    quote(0.40, 0.03, inv=-500)      -> (bid, ask), either None when not offered
"""

import math
from dataclasses import asdict, dataclass

LO, HI = 0.005, 0.995           # no quote on a market this close to decided


@dataclass(frozen=True)
class Params:
    half_spread: float = 0.03
    max_pos: float = 2500.0     # the maker's shares per market, either way
    skew: float = 1.0           # quotes lean against inventory: shift = -skew x half-spread x inventory / max_pos
    max_loss: float | None = None   # stop adding to a side once the market's worst case loses this much ($); None: off
    floor_bid: bool = False     # a long shot whose bid rounds below 1c still gets a 1c bid, if that's under fair

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)


def quote(fair, hs, inv=0.0, max_pos=2500.0, skew=1.0, tidy=True):
    """The maker's YES quote on one market: (bid, ask) in cents-rounded dollars, or (None, None). Leans
    against inventory; stops adding to a position at max_pos. tidy rounds away float noise before the
    floor / ceiling (0.40 + 0.03 quotes 0.43, not 0.44); downhill quotes without it, as it always has."""
    if fair is None or fair <= LO or fair >= HI:
        return None, None
    shift = -skew * hs * inv / max_pos
    lo, hi = (fair - hs + shift) * 100, (fair + hs + shift) * 100
    if tidy:
        lo, hi = round(lo, 6), round(hi, 6)
    bid = math.floor(lo) / 100
    ask = math.ceil(hi) / 100
    bid = bid if bid >= 0.01 and inv < max_pos else None
    ask = ask if ask <= 0.99 and inv > -max_pos else None
    return bid, ask


def worst_case(inv, cash):
    """The maker's loss in one market if YES wins and if NO wins ($, positive = a loss): inv YES shares, cash."""
    return -(cash + inv), -cash


def capped(fair, hs, inv=0.0, cash=0.0, p=Params(), tidy=True):
    """quote() with the optional risk switches: a long shot's 1c floor bid (floor_bid: the crowd can sell YES back,
    so the maker isn't only ever short), and a per-market loss cap (max_loss: no more selling YES once losing
    max_loss if YES wins; no more buying once losing it if NO wins). Both off by default: quote() as is."""
    bid, ask = quote(fair, hs, inv, p.max_pos, p.skew, tidy)
    if p.floor_bid and bid is None and fair is not None and LO < fair < HI and inv < p.max_pos \
            and fair - 0.01 >= hs / 2:
        bid = 0.01
    if p.max_loss is not None:
        if_yes, if_no = worst_case(inv, cash)
        if ask is not None and if_yes >= p.max_loss:
            ask = None
        if bid is not None and if_no >= p.max_loss:
            bid = None
    return bid, ask


def half_spread(by_stage, stage, default=0.03):
    """The half-spread for a stage: a number, or {stage label: half-spread} (the last known stage's for a
    label not listed)."""
    if not isinstance(by_stage, dict):
        return float(by_stage)
    return float(by_stage.get(stage, default))
