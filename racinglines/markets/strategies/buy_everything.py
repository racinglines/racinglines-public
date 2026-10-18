"""
Debug strategy "buy one of everything": at the first stage a market is open and priced strictly inside (0, 1), buy
one YES share and one NO share of it, then hold to resolution. No edge, volume, price band or group coherence check;
the model's fair value is not read. The venue's costs still apply (the per-share cost and, where the caller passes
one, the taker fee), so a market's pair loses exactly its costs and fees; what the P&L cells show is that every
market of a sport x exchange reached the replay, was priced and settled (and per side, how YES and NO fared).

Off by default: the replays run it only when asked (`--buy-all` on `racinglines strategy replay nascar|motogp`, or
RACINGLINES_BUY_ALL=1, which the F1 weekend sweep reads too). Backtest and paper only; nothing here places an order.
It is reported as its own mode, `buy_all`, beside the taker's modes, whose results are unchanged.

A stage may carry `open` (False = the market had closed by then) and, from recorded order books, `depth_yes` /
`depth_no` (shares at the touch): a side with no depth there is not bought. Both are optional.
"""

import os

import pandas as pd

from racinglines.markets import settlement_rules as SR
from racinglines.markets.strategies.taker_weekend import _fee

ENV = "RACINGLINES_BUY_ALL"
MODE = "buy_all"


def enabled(flag=None):
    """The switch: `flag` when given, else RACINGLINES_BUY_ALL (1 / true / yes / on); off by default."""
    if flag is not None:
        return bool(flag)
    return os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes", "on")


def entry(stages):
    """Index of the first stage the market is open and priced strictly inside (0, 1), else None."""
    for i, s in enumerate(stages):
        p = s.get("price")
        if s.get("open", True) and p is not None and 0 < p < 1:
            return i
    return None


def sides(stage):
    """The sides that can be bought at this stage: both, unless a recorded book shows no size at a side's touch."""
    return [side for side, key in (("YES", "depth_yes"), ("NO", "depth_no"))
            if stage.get(key) is None or stage[key] > 0]


def run_market(stages, outcome, cost=0.01, taker_fee=0.0, shares=1.0):
    """One market: dict(trades, pnl). outcome: True / False, a payout, SR.VOID, or None / SR.FAIR (unresolved)."""
    i = entry(stages)
    trades = []
    if i is not None:
        s = stages[i]
        for side in sides(s):
            px = s["price"] if side == "YES" else 1 - s["price"]
            trades.append(dict(stage=s["label"], t=s["t"], side=side, shares=shares,
                               price=px + cost + _fee(taker_fee, px, shares), fair=s.get("fair"), mid=s["price"]))
    pnl = None
    if outcome is not None and outcome != SR.FAIR:
        pnl = 0.0
        for tr in trades:
            if outcome == SR.VOID:
                tr["pnl"] = 0.0
            else:
                y = float(outcome)
                tr["pnl"] = tr["shares"] * ((y if tr["side"] == "YES" else 1 - y) - tr["price"])
            pnl += tr["pnl"]
    return dict(trades=trades, pnl=pnl)


def run_weekend(markets, cost=0.01, taker_fee=0.0, shares=1.0):
    """markets: list of dict(key, kind, subject, outcome, stages), as taker_weekend.run_weekend reads them.
    -> (trades df, per-market df) in taker_weekend's shape, so its summarize() and by() read them."""
    all_trades, per = [], []
    for mk in markets:
        r = run_market(mk["stages"], mk["outcome"], cost, taker_fee, shares)
        for tr in r["trades"]:
            all_trades.append(dict(tr, key=mk["key"], kind=mk["kind"], subject=mk["subject"]))
        per.append(dict(key=mk["key"], kind=mk["kind"], subject=mk["subject"], trades=len(r["trades"]),
                        bought=sum(tr["shares"] * tr["price"] for tr in r["trades"]),
                        pnl=r["pnl"], outcome=mk["outcome"]))
    return pd.DataFrame(all_trades), pd.DataFrame(per)


# --- a market with no race stages (an exchange's season futures, e.g. OG.com's) ------------------------------------

def first_price(obs):
    """obs: [(ts, price, source)] from whatever the store holds (minute prices, trades, book sides, the sync's quote).
    -> the earliest (ts, price, source) strictly inside (0, 1), else None. An undated quote (ts None) comes last."""
    ok = [o for o in obs if o[1] is not None and 0 < o[1] < 1]
    dated = sorted((o for o in ok if o[0] is not None), key=lambda o: o[0])
    return (dated or [o for o in ok if o[0] is None] or [None])[0]


def hold_pair(price, outcome=None, mark=None, cost=0.01, fee=0.0, shares=1.0):
    """One YES and one NO share bought at `price` (+ cost + fee per contract, a flat fee as OG.com charges), held.
    Settled on `outcome` when the market resolved, else marked at `mark`. -> dict(yes, no, pnl, status)."""
    yes, no = price + cost + fee, 1 - price + cost + fee
    if outcome is not None:
        y, status = float(outcome), "settled"
    elif mark is not None:
        y, status = float(mark), "marked"
    else:
        return dict(yes=yes, no=no, pnl_yes=None, pnl_no=None, pnl=None, status="open")
    pnl_yes, pnl_no = shares * (y - yes), shares * ((1 - y) - no)
    return dict(yes=yes, no=no, pnl_yes=pnl_yes, pnl_no=pnl_no, pnl=pnl_yes + pnl_no, status=status)
