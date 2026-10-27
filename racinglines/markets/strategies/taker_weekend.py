"""
Taker strategy through a race weekend: at each pricing stage (before any running,
then after each session), compare our fresh fair value with the exchange price at
that moment and rebalance each market to a target position; hold to resolution.

Per market and stage the inputs are only what was known then: our as-of fair value,
the exchange price, and whether the market was tradeable (liquid, not extreme, a
coherent book). The outcome is used only to settle.

Modes (to see whether updating adds value):
    update   rebalance at every stage (enter before any running, re-price after each session)
    hold     trade once, at the first stage the market is tradeable, then hold
    last     trade once, at the last stage (after qualifying)
    early    rebalance at every stage except the late ones (after FP3, after qualifying),
             then hold: the stage-aware taker (docs/f1-roadmap.md, F1-4)

Execution: buy at price + cost, sell at price - cost (cost per share covers spread
and slippage; the Baku books were a cent or two wide). NO is bought at 1 - price.
A stage with a quote (`bid`, `ask`: Kalshi's recorded book or candle, venue_replay.Kalshi.quote)
trades at the touch instead: YES bought at the ask and sold at the bid, NO bought at 1 - bid and
sold at 1 - ask, plus cost (slippage beyond the touch) and the fee; the edge each side needs is
then fair - ask (YES) or bid - fair (NO). Polymarket stages carry no quote, so its results are unchanged.
On a venue with a taker fee (Kalshi: 0.07 x contracts x P x (1 - P) per order, rounded
up to the cent) the fee is added per share on top of the cost, and sizing sees it too;
Polymarket's rate is 0, so its results are unchanged.

Sizing (both off by default, so results match the fixed sizing exactly):
    scale          multiplies the stake per edge and the per-market cap: bankroll-aware sizing
                   passes current bankroll / starting bankroll (weekend_sweep.bankroll_scale), or its
                   square root under bankroll_growth = sqrt (square-root staking)
    max_deployed   a cap on the weekend's capital deployed across all markets (cash spent net of
                   cash received, per market, summed). Markets are then traded in time order
                   and a buy that would go over the cap is cut to fit (sells are always allowed)
"""

import math
from dataclasses import dataclass, replace

import pandas as pd
from racinglines.core.stages import PRE_RACE
from racinglines.markets import settlement_rules as SR
from racinglines.markets.strategies.sizing import target_shares


@dataclass(frozen=True)
class TakerParams:
    min_edge: float = 0.05         # act only when |fair - price| >= 5 pts
    stake_per_edge: float = 250.0  # target cost = 250 x edge (10 pts -> $25) ...
    max_stake: float = 50.0        # ... capped per market
    cost: float = 0.01             # $ per share on every trade
    min_trade: float = 2.0         # skip rebalances smaller than $2
    price_band: tuple = (0.02, 0.98)
    mode: str = "update"
    late_stages: tuple = ("after FP3", "after Quali")    # mode "early" doesn't trade these
    stages: tuple | None = None                          # entry timing: stages any mode may trade (None = all)
    min_edge_h2h: float | None = None                    # head-to-head markets' threshold (None = min_edge)
    min_edge_by_kind: tuple = ()                         # ((kind, threshold), ...): per market kind, over both
    scale: float = 1.0                                   # stake multiplier (bankroll-aware sizing)
    max_deployed: float | None = None                    # $ cap on the weekend's deployed capital (None = no cap)
    thin_edge_mult: float | None = None                  # trade a thin market (a stage with `thin`) when the edge is
                                                         # >= this x min_edge, capped at the book's size at the touch
                                                         # (None = thin markets are skipped)
    kelly: float | None = None                           # Kelly fraction (0.5 = half Kelly): stake = kelly x f* x
                                                         # balance, capped at max_stake x scale (None = linear sizing)
    bankroll: float | None = None                        # starting bankroll for Kelly; balance = bankroll x scale
    root_unit: float | None = None                       # root unit system: stake = root_unit x scale x sqrt(c / (1 - c))
                                                         # for a contract costing c, capped at max_stake x scale
    fill: str = "price"                                  # "price": one price per order; "book": walk the stage's
                                                         # book levels (stages with `asks` / `bids`, else "price");
                                                         # "book_only": walk them, and a stage with no book trades
                                                         # nothing (the live pass, Mexico spec rule 5)
    sell_floor: str | None = None                        # the lowest a sale may fill (both fills): None = no floor
                                                         # (one price: any bid; book: the touch level only), "fair"
                                                         # = net of fee at or above fair, "edge" = at or above
                                                         # fair - min_edge (see _floor_ok)
    taker_fee: float = 0.0                               # the venue's taker fee rate: rate x contracts x P x (1 - P)
                                                         # per order, rounded up to the cent (Kalshi 0.07,
                                                         # Polymarket 0; venue_replay.EXCHANGES)


def walk(levels, shares):
    """(average price, shares filled) of taking `shares` through best-first [(price, size)] levels."""
    left, paid = float(shares), 0.0
    for px, size in levels:
        take = min(left, float(size))
        if take > 0:
            paid += take * float(px)
            left -= take
        if left <= 1e-9:
            break
    filled = float(shares) - max(left, 0.0)
    return (paid / filled if filled > 0 else None), filled


def _book_side(s, side, buy):
    """The stage's levels a trade takes, best first: buy YES = the asks, buy NO = 1 - the bids (highest bid first),
    sell YES = the bids, sell NO = 1 - the asks (lowest ask first). None when the stage has no book."""
    if "asks" not in s or "bids" not in s:
        return None
    if side == "YES":
        return s["asks"] if buy else s["bids"]
    return [(1 - p, q) for p, q in (s["bids"] if buy else s["asks"])]


def _floor_ok(px, fair, min_edge, floor, cost=0.0, fee_rate=0.0):
    """A sale at px clears the sell_floor setting: net of cost and fee, at or above fair ("fair") or fair - min_edge
    ("edge"); always with no floor. fair is the side's own (1 - fair for NO)."""
    if floor is None:
        return True
    bar = fair if floor == "fair" else fair - min_edge
    return px - cost - fee_rate * px * (1 - px) >= bar - 1e-9


def _within(levels, fair, min_edge, fee_rate, buy, floor=None):
    """The levels a book walk may take, best first, up to the first one past its limit. A buy is a limit order at
    the edge: only where fair - price - fee >= min_edge (the bar that sized it), or a thin book's far levels (0.99
    asks) fill the rest of an order sized at the touch. A sale walks down to the sell_floor (_floor_ok); with no
    floor it takes the touch level only, as the one-price fill sells at the bid (never down a thin book to 0.01)."""
    if not buy and floor is None:
        return list(levels[:1])
    out = []
    for px, size in levels:
        ok = (fair - px - fee_rate * px * (1 - px) >= min_edge - 1e-9) if buy else \
            _floor_ok(px, fair, min_edge, floor, fee_rate=fee_rate)
        if not ok:
            break
        out.append((px, size))
    return out


def _fee(rate, px, shares):
    """The venue's taker fee per share for one order of `shares` at `px`: rate x contracts x P x (1 - P),
    rounded up to the cent per order (Kalshi's schedule); 0 when the rate is 0."""
    if not rate:
        return 0.0
    n = abs(shares)
    return math.ceil(round(rate * n * px * (1 - px) * 100, 6)) / 100 / n


class _Market:
    """One market's position through the weekend, one stage at a time."""

    def __init__(self, stages, p: TakerParams):
        self.stages, self.p = stages, p
        self.yes = self.no = self.cash = 0.0
        self.trades, self.marks = [], []
        idx = [i for i, s in enumerate(stages)
               if s["fair"] is not None and s["price"] is not None
               and (s["tradeable"] or (p.thin_edge_mult is not None and s.get("thin")
                                       and abs(s["fair"] - s["price"]) >= p.thin_edge_mult * p.min_edge))]
        if p.stages is not None:                         # the live pre-race stage is open to every player (spec rule 2)
            idx = [i for i in idx if stages[i]["label"] in p.stages or stages[i]["label"] == PRE_RACE]
        if p.mode == "hold":
            idx = idx[:1]
        elif p.mode == "last":
            idx = [len(stages) - 1] if (len(stages) - 1) in idx else []
        elif p.mode == "early":
            idx = [i for i in idx if stages[i]["label"] not in p.late_stages]
        self.idx = set(idx)

    @property
    def deployed(self):
        return max(-self.cash, 0.0)

    def step(self, i, room=None):
        """Stage i: rebalance if it's a trading stage, then mark. room: $ this market may add to its
        deployed capital (None = no cap). Returns the change in deployed capital."""
        p, s = self.p, self.stages[i]
        price = s["price"]
        before = self.deployed
        if i in self.idx:
            thin = not s["tradeable"]            # a thin market: in idx only with edge >= thin_edge_mult x min_edge
            bid, ask = s.get("bid"), s.get("ask")
            kb = (p.kelly, p.bankroll * p.scale) if p.kelly is not None and p.bankroll is not None else (None, None)
            ru = p.root_unit * p.scale if p.root_unit is not None else None
            if bid is not None and ask is not None:
                ty = target_shares(s["fair"], ask, p.cost + p.taker_fee * ask * (1 - ask), p.min_edge,
                                   p.stake_per_edge * p.scale, p.max_stake * p.scale, *kb, root_unit=ru)[0]
                tn = target_shares(s["fair"], bid, p.cost + p.taker_fee * bid * (1 - bid), p.min_edge,
                                   p.stake_per_edge * p.scale, p.max_stake * p.scale, *kb, root_unit=ru)[1]
                buy, sell = {"YES": ask, "NO": 1 - bid}, {"YES": bid, "NO": 1 - ask}
            else:
                ty, tn = target_shares(s["fair"], price, p.cost + p.taker_fee * price * (1 - price), p.min_edge,
                                       p.stake_per_edge * p.scale,
                                       p.max_stake * p.scale, *kb, root_unit=ru)
                buy = sell = {"YES": price, "NO": 1 - price}
            if thin:                             # the book's size at the touch caps what can be bought
                ty, tn = min(ty, s.get("depth_yes") or 0.0), min(tn, s.get("depth_no") or 0.0)
            for side, tgt in (("YES", ty), ("NO", tn)):
                cur = self.yes if side == "YES" else self.no
                d = tgt - cur
                px = buy[side] if d > 0 else sell[side]
                if thin and d <= 0:
                    continue                     # thin: buys only, never a sale into an unseen book
                if abs(d) * px < p.min_trade and tgt != 0:
                    continue
                if abs(d) < 1e-9:
                    continue
                f_side = s["fair"] if side == "YES" else 1 - s["fair"]
                levels = _book_side(s, side, d > 0) if p.fill in ("book", "book_only") else None
                if levels is None and p.fill == "book_only":
                    continue                                 # no recorded book at this stage: no fill
                if levels is not None:                       # a limit: never past the edge or floor (_within)
                    levels = _within(levels, f_side, p.min_edge, p.taker_fee, d > 0, p.sell_floor)
                elif d < 0 and not _floor_ok(px, f_side, p.min_edge, p.sell_floor, p.cost, p.taker_fee):
                    continue                                 # one price: the sale is under the floor
                if levels is not None:                       # walk the book: the spread and the slippage
                    px, got = walk(levels, abs(d))
                    if px is None:
                        continue
                    if got < abs(d) - 1e-9:                  # cut to what the book holds
                        d = got if d > 0 else -got
                        tgt = cur + d
                        if abs(d) * px < p.min_trade and tgt != 0:
                            continue
                cost = 0.0 if levels is not None else p.cost
                fee = _fee(p.taker_fee, px, d)
                exec_px = px + cost + fee if d > 0 else max(px - cost - fee, 0.0)
                if room is not None and d > 0:
                    left = room - (self.deployed - before)
                    if d * exec_px > left:                   # cut the buy to fit under the cap
                        d = max(left, 0.0) / exec_px
                        if d * px < p.min_trade:
                            continue
                        tgt = cur + d
                        if levels is not None:               # a smaller order never walks deeper
                            px = walk(levels, d)[0]
                            exec_px = px + _fee(p.taker_fee, px, d)
                self.cash -= d * exec_px
                if side == "YES":
                    self.yes = tgt
                else:
                    self.no = tgt
                self.trades.append(dict(stage=s["label"], t=s["t"], side=side, shares=d, price=exec_px,
                                        fair=s["fair"], mid=price, bid=s.get("bid"), ask=s.get("ask"),
                                        bid_size=s.get("bid_size"), ask_size=s.get("ask_size")))
        if price is not None:
            self.marks.append(dict(stage=s["label"], value=self.cash + self.yes * price + self.no * (1 - price)))
        return self.deployed - before

    def result(self, outcome):
        """outcome: True / False, or under a cancelled race's venue rules (markets/settlement_rules.py) a payout
        (0.5, or Kalshi's last price) or VOID (every trade refunded at its price: P&L 0); None = unresolved, as
        is FAIR (Kalshi's fair-price settlement) when no price was recorded."""
        pnl = None
        if outcome is not None and outcome != SR.FAIR:
            if outcome == SR.VOID:
                pnl = 0.0
                for tr in self.trades:
                    tr["pnl"] = 0.0
            else:
                y = float(outcome)
                pnl = self.cash + self.yes * y + self.no * (1 - y)
                for tr in self.trades:      # each decision's P&L to resolution (attribution by stage)
                    v = y if tr["side"] == "YES" else 1 - y
                    tr["pnl"] = tr["shares"] * (v - tr["price"])
        return dict(trades=self.trades, marks=self.marks, pnl=pnl, yes=self.yes, no=self.no, cash=self.cash)


def run_market(stages, outcome, p: TakerParams):
    """stages: list of dict(label, t, fair, price, tradeable). outcome: True/False/None.
    Returns dict(trades, marks, pnl, yes, no, cash)."""
    mk = _Market(stages, p)
    for i in range(len(stages)):
        mk.step(i)
    return mk.result(outcome)


def params_for(kind, p: TakerParams):
    """The parameters one market is traded with: a kind's own threshold (min_edge_by_kind), else head-to-head
    markets' (min_edge_h2h), else min_edge."""
    by = dict(p.min_edge_by_kind)
    if kind in by:
        return replace(p, min_edge=by[kind])
    return replace(p, min_edge=p.min_edge_h2h) if kind == "race_h2h" and p.min_edge_h2h is not None else p


def run_weekend(markets, p: TakerParams):
    """markets: list of dict(key, kind, subject, outcome, stages). -> (trades df, per-market df)."""
    results = _capped(markets, p) if p.max_deployed is not None else \
        [run_market(mk["stages"], mk["outcome"], params_for(mk["kind"], p)) for mk in markets]
    all_trades, per = [], []
    for mk, r in zip(markets, results):
        for tr in r["trades"]:
            all_trades.append(dict(tr, key=mk["key"], kind=mk["kind"], subject=mk["subject"],
                                   outcome=mk["outcome"]))
        cost_basis = sum(tr["shares"] * tr["price"] for tr in r["trades"] if tr["shares"] > 0)
        per.append(dict(key=mk["key"], kind=mk["kind"], subject=mk["subject"], trades=len(r["trades"]),
                        bought=cost_basis, pnl=r["pnl"], outcome=mk["outcome"]))
    return pd.DataFrame(all_trades), pd.DataFrame(per)


def _capped(markets, p: TakerParams):
    """Every market's stages in time order (ties in market order), sharing the deployed-capital cap."""
    books = [_Market(mk["stages"], params_for(mk["kind"], p)) for mk in markets]
    order = sorted(((s["t"], j, i) for j, mk in enumerate(markets) for i, s in enumerate(mk["stages"])),
                   key=lambda x: (pd.Timestamp(x[0]) if x[0] is not None else pd.Timestamp.min, x[1], x[2]))
    deployed = 0.0
    for _, j, i in order:
        deployed += books[j].step(i, room=max(p.max_deployed - deployed, 0.0))
    return [b.result(mk["outcome"]) for b, mk in zip(books, markets)]


# touch quote, sizes and outcome on each trade row: for --trades-out (weekend_sweep.trades_frame) only
DUMP_COLS = ("bid", "ask", "bid_size", "ask_size", "outcome")


def core(trades):
    """`trades` without DUMP_COLS, so a sweep's own tables and saved outputs stay as they were."""
    return trades.drop(columns=[c for c in DUMP_COLS if c in trades])


def summarize(trades, per):
    if not len(per):
        return dict(markets=0, traded=0, trades=0, turnover=0.0, bought=0.0, pnl=0.0)
    t = per[per["trades"] > 0]
    turnover = float((trades["shares"].abs() * trades["price"]).sum()) if len(trades) else 0.0
    return dict(markets=len(per), traded=len(t), trades=int(per["trades"].sum()), turnover=turnover,
                bought=float(per["bought"].sum()), pnl=float(per["pnl"].fillna(0).sum()),
                roi=float(per["pnl"].fillna(0).sum() / per["bought"].sum()) if per["bought"].sum() > 0 else None)


def by(trades, col):
    """P&L to resolution of the trades made, grouped (e.g. by stage or kind)."""
    if not len(trades) or "pnl" not in trades:
        return pd.DataFrame(columns=[col, "trades", "pnl"])
    g = trades.groupby(col, sort=False).agg(trades=("pnl", "size"), pnl=("pnl", "sum"))
    return g.reset_index()
