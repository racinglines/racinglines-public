"""
Default season-long strategy for championship markets, replayed against the
exchange's real price history.

Decisions (e.g. pre-season, then after every race) each come with our as-of fair
values. At each decision we rebalance every tradeable market toward a target
position, executing at the exchange's recorded price `exec_delay` after the
decision plus a per-market cost (half-spread + slippage). Positions are held;
a market that resolves (e.g. a driver eliminated from the title fight) settles at
0/1 when it closes; the rest are valued at what they'd fetch if sold now
(price - cost for YES, (1 - price) - cost for NO).

Guard rails: stake per market capped, total capital (cost basis of open positions)
capped, no trading at extreme prices, stale prices (> max_age) are not tradeable.
"""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from racinglines.core.stats import last_at
from racinglines.markets.strategies.sizing import target_shares


@dataclass(frozen=True)
class SeasonParams:
    min_edge: float = 0.03            # act when |fair - price| >= 3 pts
    stake_per_edge: float = 500.0     # target cost = 500 x edge (5 pts -> $25) ...
    max_stake: float = 150.0          # ... capped per market
    capital: float = 1500.0           # cap on the cost basis of all open positions
    price_band: tuple = (0.01, 0.99)
    exec_delay: timedelta = timedelta(hours=1)
    max_age: timedelta = timedelta(hours=48)
    min_trade: float = 5.0            # skip rebalances under $5
    mode: str = "update"              # update (rebalance at every decision) | hold (first decision only)


@dataclass
class SeasonMarket:
    key: str
    kind: str
    subject: str
    ts: np.ndarray                    # price history timestamps (UTC-aware pandas)
    px: np.ndarray
    cost: float                       # $ per share per trade
    outcome: bool | None = None       # resolved YES/NO (None = open)
    closed_at: pd.Timestamp | None = None


def replay(markets, decisions, p: SeasonParams = SeasonParams(), now=None, marks=None):
    """markets: {key: SeasonMarket}; decisions: [dict(t, label, fairs={key: fair})] in time order.
    Returns dict(trades, positions, equity, summary)."""
    pos = {k: [0.0, 0.0] for k in markets}          # yes, no shares
    basis = {k: 0.0 for k in markets}               # cost basis of the open position
    cash, trades, settled = 0.0, [], {}
    now = now or pd.Timestamp.now(tz="UTC")
    events = [("decision", d["t"], d) for d in decisions]
    events += [("settle", mk.closed_at, mk.key) for mk in markets.values()
               if mk.outcome is not None and mk.closed_at is not None]
    events.sort(key=lambda e: (e[1], 0 if e[0] == "settle" else 1))
    first = True
    for kind, t, obj in events:
        if t > now:
            break
        if kind == "settle":
            k = obj
            if k in settled:
                continue
            y = float(markets[k].outcome)
            cash += pos[k][0] * y + pos[k][1] * (1 - y)
            settled[k] = (t, pos[k][0] * y + pos[k][1] * (1 - y) - basis[k])
            pos[k], basis[k] = [0.0, 0.0], 0.0
            continue
        if p.mode == "hold" and not first:
            continue
        first = False
        d = obj
        te = t + p.exec_delay
        plan = []
        for k, mk in markets.items():
            if k in settled or k not in d["fairs"] or d["fairs"][k] is None:
                continue
            price = last_at(mk.ts, mk.px, te, p.max_age)
            if price is None:
                continue
            ty, tn = target_shares(d["fairs"][k], price, mk.cost, p.min_edge, p.stake_per_edge, p.max_stake)
            if not (p.price_band[0] <= price <= p.price_band[1]):
                # outside the band: never open or add, but always allowed to reduce / close
                ty, tn = min(ty, pos[k][0]), min(tn, pos[k][1])
                if (ty, tn) == tuple(pos[k]):
                    continue
            plan.append((k, price, (ty, tn)))
        # capital cap: scale down new exposure if the targets exceed it
        want = sum(ty * (pr + markets[k].cost) + tn * (1 - pr + markets[k].cost) for k, pr, (ty, tn) in plan)
        scale = min(1.0, p.capital / want) if want > 0 else 1.0
        for k, price, (ty, tn) in plan:
            mk = markets[k]
            ty, tn = ty * scale, tn * scale
            for side, idx, tgt, px in (("YES", 0, ty, price), ("NO", 1, tn, 1 - price)):
                dq = tgt - pos[k][idx]
                if abs(dq) * px < p.min_trade and tgt != 0:
                    continue
                if abs(dq) < 1e-9:
                    continue
                exec_px = px + mk.cost if dq > 0 else max(px - mk.cost, 0.0)
                cash -= dq * exec_px
                if dq > 0:
                    basis[k] += dq * exec_px
                else:
                    held = pos[k][idx]
                    basis[k] *= (1 + dq / held) if held > 0 else 0
                pos[k][idx] = tgt
                trades.append(dict(t=te, decision=d["label"], key=k, kind=mk.kind, subject=mk.subject, side=side,
                                   shares=dq, price=exec_px, mid=price, fair=d["fairs"][k]))
    # equity curve: cash + liquidation value of open positions (+ settled), at each mark time
    marks = marks if marks is not None else pd.date_range(decisions[0]["t"].normalize(), now, freq="D", tz=None)
    tr = pd.DataFrame(trades)
    equity = []
    for m in marks:
        m = pd.Timestamp(m)
        m = m.tz_localize("UTC") if m.tzinfo is None else m
        c, val = 0.0, 0.0
        hold = {k: [0.0, 0.0] for k in markets}
        if len(tr):
            for r in tr[tr["t"] <= m].itertuples():
                c -= r.shares * r.price
                hold[r.key][0 if r.side == "YES" else 1] += r.shares
        for k, mk in markets.items():
            y, n = hold[k]
            if not y and not n:
                continue
            if mk.outcome is not None and mk.closed_at is not None and mk.closed_at <= m:
                val += y * float(mk.outcome) + n * (1 - float(mk.outcome))
            else:
                px = last_at(mk.ts, mk.px, m, timedelta(days=30))
                if px is not None:
                    val += y * max(px - mk.cost, 0) + n * max(1 - px - mk.cost, 0)
        equity.append(dict(t=m, equity=c + val))
    equity = pd.DataFrame(equity)
    # final positions
    rows = []
    for k, mk in markets.items():
        y, n = pos[k]
        px = last_at(mk.ts, mk.px, now, timedelta(days=30))
        if k in settled:
            rows.append(dict(key=k, kind=mk.kind, subject=mk.subject, side="settled", shares=0.0, basis=0.0,
                             price=float(mk.outcome), value=None, pnl=settled[k][1], status="settled"))
        elif y or n:
            side, sh = ("YES", y) if y else ("NO", n)
            value = sh * max((px if side == "YES" else 1 - px) - mk.cost, 0) if px is not None else None
            rows.append(dict(key=k, kind=mk.kind, subject=mk.subject, side=side, shares=sh, basis=basis[k], price=px,
                             value=value, pnl=(value - basis[k]) if value is not None else None, status="open"))
    positions = pd.DataFrame(rows)
    bought = float((tr.loc[tr["shares"] > 0, "shares"] * tr.loc[tr["shares"] > 0, "price"]).sum()) if len(tr) else 0.0
    eq = equity["equity"] if len(equity) else pd.Series([0.0])
    summary = dict(pnl=float(eq.iloc[-1]), trades=len(tr), bought=bought,
                   open_positions=int((positions["status"] == "open").sum()) if len(positions) else 0,
                   settled_pnl=float(sum(v[1] for v in settled.values())),
                   max_drawdown=float((eq - eq.cummax()).min()), peak=float(eq.max()),
                   capital_used=float(sum(basis.values())))
    return dict(trades=tr, positions=positions, equity=equity, summary=summary)
