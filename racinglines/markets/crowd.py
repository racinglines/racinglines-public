"""
The private book's crowd: anonymous simulated takers trading against a maker's quotes, shared by every
live sport (pipelines/live.py). Moved out of pipelines/live_dh.py, which imports it back under its old
names; the downhill numbers are unchanged.

A market is identified by its key: a quote's "key", else "<bib>:<market>" (downhill). The book:

    markets   {key: dict(inv, cash, crowd_inv, crowd_cash)}   the maker's side and the crowd's
    crowd     dict(fills, volume, budget_total)
    budget    each taker's event budget ($), fixed once and seeded; left: what each has left
    ids       each taker's anonymous id (a UUID; no accounts)
    takers    {id: {key: [YES shares, cash]}}
    late_left a fresh cap per taker for the late (pre-race) window, once it opens

Two ways to trade:
  - fills(): one poll (downhill: every few seconds), each quoted market hit by each taker with probability
    p x intensity;
  - window(): one batch for a stretch of time (F1: the hours between session-end updates) at a per-hour
    rate, the fills timed across the window and processed in time order.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

PRIVATE = "private"
MAKER_USER, TAKER_USER = "maker", "taker"        # the demo accounts: the book's maker, and the tracked taker


@dataclass(frozen=True)
class Params:
    takers: int = 1000
    p: float = 0.0004                   # per taker, per quoted market, per poll
    budget: tuple = (10.0, 200.0)       # $ each taker may bet over the whole event, log-uniform
    bet: tuple = (0.2, 0.8)             # each bet: this share of what the taker has left (at least min_bet)
    min_bet: float = 2.0
    max_pos: float = 2500.0             # the maker's shares per market, either way
    seed: int = 20260927                # the crowd's budgets and ids
    rate_h: float = 0.0                 # window(): per taker, per quoted market, per hour
    late_cap: float = 100.0             # the late window's fresh cap per taker
    late_pace: float = 50.0             # the late window's pace, x the normal rate
    max_loss: float | None = None       # the maker's per-market loss cap (quoting's): a fill never takes it past it

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        for k in ("budget", "bet"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def mkey(q):
    """A quote's or a fill's market key."""
    return q.get("key") or f"{q['bib']}:{q['market']}"


def new_book(p=Params()):
    import uuid
    rng = np.random.default_rng(p.seed)                        # the crowd's budgets and ids, fixed once (replayable)
    budget = np.exp(rng.uniform(math.log(p.budget[0]), math.log(p.budget[1]), p.takers)).round(2).tolist()
    ids = [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(p.takers)]   # anonymous: no accounts
    return dict(markets={}, crowd=dict(fills=0, volume=0.0, budget_total=round(sum(budget), 2)), seeds=[],
                budget=budget, left=list(budget), ids=ids, takers={})


def load_book(out, p=Params()):
    import json
    f = out / "book.json"
    return json.loads(f.read_text()) if f.exists() else new_book(p)


def open_late(book, p=Params(), at=None):
    """The late window opens: every taker gets a fresh cap (their event budgets are untouched)."""
    book["late"], book["late_at"] = True, at
    book["late_left"] = [p.late_cap] * p.takers


def _fill(q, who, book, rng, p, pot):
    """One taker hits one quote: a random side, part of what the taker has left. Mutates book; the fill or None."""
    if book[pot][who] < p.min_bet:
        return None
    k = mkey(q)
    m = book["markets"].setdefault(k, dict(inv=0.0, cash=0.0, crowd_inv=0.0, crowd_cash=0.0))
    sides = [s for s, px in (("buy", q["ask"]), ("sell", q["bid"])) if px is not None]
    side = sides[int(rng.integers(len(sides)))]
    px = q["ask"] if side == "buy" else q["bid"]
    cost = px if side == "buy" else 1 - px                     # a crowd seller buys NO at 1 - bid
    stake = max(p.min_bet, book[pot][who] * float(rng.uniform(*p.bet)))
    stake = min(stake, book[pot][who])
    shares = round(stake / max(cost, 0.01), 2)
    room = p.max_pos + m["inv"] if side == "buy" else p.max_pos - m["inv"]
    shares = min(shares, max(room, 0.0))
    if p.max_loss is not None:                                  # the maker's worst case in this market stays capped
        if side == "buy":                                       # the maker sells YES: loses (1 - px) a share if YES
            left = p.max_loss + m["cash"] + m["inv"]
            shares = min(shares, round(max(left, 0.0) / max(1 - px, 0.01), 2))
        else:                                                   # the maker buys YES: loses px a share if NO
            left = p.max_loss + m["cash"]
            shares = min(shares, round(max(left, 0.0) / max(px, 0.01), 2))
    if shares <= 0:
        return None
    book[pot][who] = round(book[pot][who] - shares * cost, 2)
    tid = book["ids"][who]
    pos = book["takers"].setdefault(tid, {}).setdefault(k, [0.0, 0.0])
    sg = 1 if side == "buy" else -1                             # crowd's YES shares
    m["inv"] -= sg * shares
    m["cash"] += sg * shares * px
    m["crowd_inv"] += sg * shares
    m["crowd_cash"] -= sg * shares * px
    pos[0] += sg * shares                                       # this taker's YES shares and cash
    pos[1] -= sg * shares * px
    book["crowd"]["fills"] += 1
    book["crowd"]["volume"] += shares * cost
    where = dict(bib=q["bib"], market=q["market"]) if "bib" in q else dict(key=k)
    return dict(where, side=side, price=px, shares=shares, fair=q["fair"], taker=tid)


def fills(quotes, book, rng, p=Params(), intensity=1.0, pot="left"):
    """One poll of the anonymous crowd against the maker's quotes (intensity x the normal hit rate). Mutates
    book; returns the fills."""
    out = []
    for q in quotes:
        if q["bid"] is None and q["ask"] is None:
            continue
        for _ in range(int(rng.binomial(p.takers, min(1.0, p.p * intensity)))):
            who = int(rng.integers(p.takers))                   # an anonymous taker with money left
            f = _fill(q, who, book, rng, p, pot)
            if f is not None:
                out.append(f)
    return out


def window(quotes, book, rng, p, start, end, pace=1.0, pot="left"):
    """One batch for the window start..end (datetimes): each quoted market is hit by each taker with
    probability rate_h x hours x pace, capped at 1; the hits get times spread uniformly across the window and
    are filled in time order at the quotes posted at its start. Mutates book; returns the fills, each with
    its time (ISO)."""
    hours = max((end - start).total_seconds() / 3600, 0.0)
    prob = min(1.0, p.rate_h * hours * pace)
    hits = []
    for i, q in enumerate(quotes):
        if (q["bid"] is None and q["ask"] is None) or prob <= 0:
            continue
        n = int(rng.binomial(p.takers, prob))
        for who, u in zip(rng.integers(p.takers, size=n), rng.random(n)):
            hits.append((float(u), i, int(who)))
    out = []
    for u, i, who in sorted(hits):
        f = _fill(quotes[i], who, book, rng, p, pot)
        if f is not None:
            f["ts"] = (start + (end - start) * u).isoformat(timespec="seconds")
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# P&L, the crowd's results, and the book rebuilt from its fills
# ---------------------------------------------------------------------------

def _value(k, fair, outcomes):
    o = outcomes.get(k)
    return float(o) if o is not None else fair.get(k, 0.0)


def pick_key(p):
    return p.get("key") or f"{p['bib']}:{p['market']}"


def book_pnl(book, fair, outcomes, picks=()):
    """The maker's P&L, marked to fair (settled where the outcome is known): total, vs the crowd, vs the demo
    taker. fair and outcomes by market key."""
    crowd = sum(mk["cash"] + mk["inv"] * _value(k, fair, outcomes) for k, mk in book["markets"].items())
    taker = sum(p["stake"] - p["shares"] * _value(pick_key(p), fair, outcomes) for p in picks)
    return dict(total=crowd + taker, crowd=crowd, taker=taker)


def crowd_results(book, fair, outcomes, spotlight_seed=20260927):
    """Each anonymous taker's P&L (marked to fair; settled where known), summarised as a group, plus one
    randomly chosen taker's bets."""
    pnl = {tid: sum(c + y * _value(k, fair, outcomes) for k, (y, c) in mk.items())
           for tid, mk in book.get("takers", {}).items()}
    if not pnl:
        return None
    v = np.array(list(pnl.values()))
    ids = sorted(pnl)
    pick = ids[int(np.random.default_rng(spotlight_seed).integers(len(ids)))]
    bets = [dict(market=k, yes=y, cash=c, value=_value(k, fair, outcomes), pnl=c + y * _value(k, fair, outcomes))
            for k, (y, c) in book["takers"][pick].items()]
    return dict(bettors=len(v), up=int((v > 0.005).sum()), down=int((v < -0.005).sum()), median=float(np.median(v)),
                best=float(v.max()), worst=float(v.min()), total=float(v.sum()),
                spotlight=dict(taker=pick, pnl=pnl[pick], bets=bets))


def rebuild(polls, ts=None):
    """The maker's book rebuilt from logged crowd polls (crowd.jsonl lines) up to ts (ISO; None = all):
    {key: dict(inv, cash)} and the polls used."""
    polls = [e for e in polls if ts is None or e["ts"] <= ts]
    markets = {}
    for e in polls:
        for f in e["fills"]:
            mk = markets.setdefault(mkey(f), dict(inv=0.0, cash=0.0))
            sg = -1 if f["side"] == "buy" else 1                       # the crowd buys YES: the maker is short
            mk["inv"] += sg * f["shares"]
            mk["cash"] -= sg * f["shares"] * f["price"]
    return markets, polls


def maker_positions(book, picks):
    """{key: dict(yes, cash)}: the demo maker's position per market (the crowd's fills and the demo taker's
    picks)."""
    maker = {k: dict(yes=mk["inv"], cash=mk["cash"]) for k, mk in book["markets"].items()}
    for p in picks:
        m = maker.setdefault(pick_key(p), dict(yes=0.0, cash=0.0))
        m["yes"] -= p["shares"]
        m["cash"] += p["stake"]
    return maker


def sync_positions(event_key, book, picks, describe, fair, outcomes, race_id=None):
    """Write the private book into paper_positions (venue 'private', event_key): the demo maker's position per
    market and the demo taker's picks. describe(key) -> (kind, subject); fair and outcomes by market key.
    Replaced on every update, so the Positions page shows them live and settled."""
    from sqlalchemy import text

    from racinglines.db.config import get_engine
    maker = maker_positions(book, picks)

    def row(uid, k, yes, cash):
        kind, subj = describe(k)
        return dict(u=uid, e=event_key, k=f"{event_key}:{k}", kind=kind, subj=subj, yes=yes, no=0.0,
                    cash=cash, mark=fair.get(k), out=outcomes.get(k), v=PRIVATE, r=race_id)
    with get_engine().begin() as c:
        ids = dict(c.execute(text("SELECT username, id FROM users WHERE username = ANY(:u)"),
                             dict(u=[MAKER_USER, TAKER_USER])).all())
        c.execute(text("DELETE FROM paper_positions WHERE venue = :v AND event_key = :e"), dict(v=PRIVATE, e=event_key))
        rows_ = []
        if MAKER_USER in ids:
            rows_ += [row(ids[MAKER_USER], k, m["yes"], m["cash"]) for k, m in maker.items() if abs(m["yes"]) > 1e-9]
        if TAKER_USER in ids:
            rows_ += [row(ids[TAKER_USER], pick_key(p), p["shares"], -p["stake"]) for p in picks]
        for r in rows_:
            c.execute(text("""INSERT INTO paper_positions (user_id, candidate_id, race_id, event_key, market_key, kind, subject,
                                yes_shares, no_shares, cash, mark, outcome, venue)
                              VALUES (:u, NULL, :r, :e, :k, :kind, :subj, :yes, :no, :cash, :mark, :out, :v)"""), r)
