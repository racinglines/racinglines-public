"""
Venues in a backtest (docs/backtest-core.md, step 4): where a strategy's orders would have met prices,
read only as of each moment. Every venue gives the same four things:

    markets()             the venue's markets on the event: key, kind, subject, athlete / params
    view(market, t)       what was public at t: price (None when stale or absent), 24 h volume, tradeable
    cost / fill           the venue's costs per share and how an order fills (the strategies apply them)
    resolve(market, res)  YES / NO / None from the official result (markets/kinds.py's settlement)

    Polymarket(conn, w, start, end)          the recorded price history and trade tape (the F1 sweep reads it)
    Kalshi(conn, links, start, end)          the same for Kalshi's markets (markets/kalshi/ writes the same tables),
                                             per market ticker
    OG(conn, links, start, end)              OG.com's (exchanges/og.toml): every stored price counts whatever its
                                             spread or depth, except an empty book's 0.50; a flat fee per contract
    EXCHANGES                                code -> class. Each exchange declares its fee schedule (TAKER_FEE,
                                             MAKER_FEE); the sweep's maker, the disagreement log and the signal
                                             engine read the fees from here
    PrivateBook.from_run(run_dir)            a live run's private book: our quotes and the simulated crowd,
                                             replayed from the logged polls and seeds (exactly, by default,
                                             or under another quoting rule)

A venue reads nothing after t in view(), so strategies can't look ahead through it; settlement is a
separate call made after trading. Another exchange is one more class with these methods; the display side
of venues (links, names, the board) is markets/venues.py.
"""

import gzip
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.core.stats import last_at
from racinglines.markets import crowd as C

STALE = timedelta(hours=6)           # a price older than this is no price


class Polymarket:
    """Recorded Polymarket data for one weekend: 5-minute prices per token, the trade tape for liquidity,
    and the multi-outcome coherence check (a group whose prices don't sum near its target isn't traded)."""

    code = "polymarket"
    TAKER_FEE = 0.0                  # the venue's fee schedule: rate x contracts x P x (1 - P), rounded up to the
    MAKER_FEE = 0.0                  # cent (Polymarket charges neither on these markets)
    MARKET_KEY = "condition_id"      # one market per condition (its outcome-0 token); Kalshi / OG.com: per token

    def __init__(self, conn, links, start, end, group_target=None, coherence_tol=0.25, stale=STALE):
        from racinglines.markets import store as MS
        self.links = links
        self.stale = stale
        self.group_target = group_target or {}
        self.coherence_tol = coherence_tol
        a, b = pd.Timestamp(start).tz_localize("UTC"), pd.Timestamp(end).tz_localize("UTC")
        ph = MS.read(conn, "prices", tokens=links["token_id"].tolist(), start=a, end=b)[["token_id", "ts", "price"]]
        tr = MS.read(conn, "trades", conditions=links["condition_id"].tolist(), start=a - timedelta(hours=24), end=b)
        tr = tr.assign(usd=tr["price"] * tr["size"])[["condition_id", "ts", "usd"]]
        for df in (ph, tr):
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        self.prices = {t: g for t, g in ph.groupby("token_id")}
        self.trades = {c: g for c, g in tr.groupby("condition_id")}

    def markets(self):
        return self.links.to_dict("records")

    def price(self, token, t):
        g = self.prices.get(token)
        return None if g is None else last_at(pd.DatetimeIndex(g["ts"]), g["price"].to_numpy(), t, self.stale)

    def volume_24h(self, condition, t):
        g = self.trades.get(condition)
        if g is None:
            return 0.0
        a, b = g["ts"].searchsorted(t - timedelta(hours=24)), g["ts"].searchsorted(t, side="right")
        return float(g["usd"].iloc[a:b].sum())

    def quote(self, token, t):
        """(bid, ask) a taker would have met at t, or None: the venue then trades at price() (Polymarket)."""
        return None

    tol_by_kind = {}                 # {kind: tolerance} over coherence_tol (coherence_tol_by_kind); {} = none
    books = None                     # {token: (ts, bids, asks)} once load_books() ran (thin_edge_mult)

    def coherent(self, kind, t):
        """A multi-outcome group's prices at t sum to within tolerance of its target (stale books skipped);
        True for kinds that aren't groups."""
        if kind not in self.group_target:
            return True
        target = self.group_target[kind]
        tol = self.tol_by_kind.get(kind, self.coherence_tol)
        ps = [self.price(tok, t) for tok in self.links.loc[self.links["prediction"] == kind, "token_id"]]
        ps = [p for p in ps if p is not None]
        return bool(ps) and abs(sum(ps) - target) <= tol * target

    def load_books(self, conn, start, end):
        """Recorded order-book snapshots of the links' YES tokens over [start, end] (markets record writes them;
        none stored = no thin-market exceptions, which is the safe side)."""
        from racinglines.markets import store as MS
        from racinglines.markets.strategies.maker_replay import _levels
        root = None if self.code == "polymarket" else MS.root_for(self.code)
        a, b = pd.Timestamp(start).tz_localize("UTC"), pd.Timestamp(end).tz_localize("UTC")
        bk = MS.read(conn, "books", tokens=self.links["token_id"].tolist(), start=a, end=b, root=root)
        self.books = {}
        for tok, g in bk.groupby("token_id"):
            ts = pd.DatetimeIndex(pd.to_datetime(g["ts"], utc=True)).tz_localize(None)
            self.books[tok] = (ts, [_levels(v) for v in g["bids"]], [_levels(v) for v in g["asks"]])

    def touch_depth(self, token, t, max_age=timedelta(minutes=10)):
        """(shares resting at the best ask, at the best bid) of the token's latest book snapshot at or before t;
        None when there is no snapshot within max_age. Buying YES takes the ask's size, buying NO the bid's."""
        b = (self.books or {}).get(token)
        if b is None:
            return None
        ts, bids, asks = b
        i = ts.searchsorted(pd.Timestamp(t), side="right") - 1
        if i < 0 or pd.Timestamp(t) - ts[i] > max_age:
            return None
        return (float(asks[i][min(asks[i])]) if asks[i] else 0.0, float(bids[i][max(bids[i])]) if bids[i] else 0.0)

    def view(self, market, t, min_volume_24h):
        """(price, volume_24h, tradeable-by-the-venue): priced, strictly inside (0, 1) and liquid enough."""
        p = self.price(market["token_id"], t)
        v = self.volume_24h(market["condition_id"], t)
        return p, v, p is not None and 0 < p < 1 and v >= min_volume_24h

    @staticmethod
    def resolve(market, res):
        from racinglines.markets import private_book as house
        ath = None if pd.isna(market["athlete_id"]) else int(market["athlete_id"])
        return house.outcome_for(market["prediction"], ath, market["params"], res)

    @classmethod
    def links(cls, conn, race_id, kinds):
        """The race's links on this exchange of these kinds (every token; one market per MARKET_KEY is the caller's
        drop_duplicates), in link order."""
        from sqlalchemy import text
        return pd.read_sql(text("""SELECT ml.*, a.display_name AS athlete FROM market_links ml
                                   LEFT JOIN athletes a ON a.id = ml.athlete_id
                                   WHERE ml.race_id = :r AND ml.prediction = ANY(:k) AND ml.exchange = :x
                                   ORDER BY ml.id"""), conn, params=dict(r=race_id, k=list(kinds), x=cls.code))

    @classmethod
    def per_share_fee(cls):
        """A flat fee in dollars per contract traded (OG.com's), added to a taker's cost per share; 0 here."""
        return 0.0

    @classmethod
    def tape_prices(cls, conn, tokens, start, end):
        """(token_id, ts, price) of the tokens over [start, end] (UTC), from this exchange's store: what a maker
        replay reads as the market's price (markets/strategies/maker_replay.py)."""
        from racinglines.markets import store as MS
        return MS.read(conn, "prices", tokens=list(tokens), start=start, end=end, root=MS.root_for(cls.code))


class Kalshi(Polymarket):
    """Recorded Kalshi data for one weekend: markets/kalshi/sync.py stores Kalshi's candlesticks, tape (contracts
    pay $1, so USD volume = price x contracts) and links (exchange 'kalshi', token = the market's YES contract) in
    the tables Polymarket uses, so reading them is the same. What differs is the fee: Kalshi charges takers
    ceil(TAKER_FEE x contracts x P x (1 - P)) in cents per order (Kalshi's published schedule for most markets;
    check the market's own before relying on it), and the taker's price: quote() gives the bid and ask at t, from
    the recorded order book (every 5 minutes on the VM since 2026-09-30) when a snapshot is at most BOOK_AGE old,
    else from the last hourly candle's closing bid and ask (within the staleness window), so a taker buys at the
    ask and sells at the bid instead of at the last trade."""

    code = "kalshi"
    MARKET_KEY = "token_id"          # a Kalshi condition_id is the event ticker, shared by every market of the event
    TAKER_FEE = 0.07
    MAKER_FEE = 0.0175               # on most markets (check the market's own schedule)
    BOOK_AGE = timedelta(minutes=10)  # a book snapshot older than this gives way to the candle's quote

    def __init__(self, conn, links, start, end, group_target=None, coherence_tol=0.25, stale=STALE):
        """As Polymarket's, but a Kalshi link's condition_id is its event ticker (every driver's market of the
        event), so the tape is read and summed per market ticker, from Kalshi's own Parquet archive."""
        from racinglines.markets import store as MS
        self.links = links
        self.stale = stale
        self.group_target = group_target or {}
        self.coherence_tol = coherence_tol
        a, b = pd.Timestamp(start).tz_localize("UTC"), pd.Timestamp(end).tz_localize("UTC")
        root, toks = MS.root_for(self.code), links["token_id"].tolist()
        ph = MS.read(conn, "prices", tokens=toks, start=a, end=b, root=root).reindex(
            columns=["token_id", "ts", "price", "bid", "ask"])
        tr = MS.read(conn, "trades", tokens=toks, start=a - timedelta(hours=24), end=b, root=root)
        tr = tr.assign(usd=tr["price"] * tr["size"])[["token_id", "ts", "usd"]]
        bk = MS.read(conn, "books", tokens=toks, start=a - self.BOOK_AGE, end=b, root=root)
        bk = bk.reindex(columns=["token_id", "ts", "best_bid", "best_ask"]).rename(columns=dict(best_bid="bid", best_ask="ask"))
        for df in (ph, tr, bk):
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        self.prices = {t: g for t, g in ph.groupby("token_id")}
        self.trades = {t: g for t, g in tr.groupby("token_id")}
        self.book_quotes = {t: g.sort_values("ts") for t, g in bk.groupby("token_id")}

    @staticmethod
    def _quote_at(g, t, max_age):
        """(bid, ask) of the last row of g at or before t, if it is at most max_age old and both sides quote."""
        if g is None or not len(g) or "bid" not in g:
            return None
        i = pd.DatetimeIndex(g["ts"]).searchsorted(pd.Timestamp(t), side="right") - 1
        if i < 0 or pd.Timestamp(t) - g["ts"].iloc[i] > max_age:
            return None
        bid, ask = g["bid"].iloc[i], g["ask"].iloc[i]
        if pd.isna(bid) or pd.isna(ask) or not 0 < bid <= ask < 1:
            return None
        return float(bid), float(ask)

    def quote(self, token, t):
        return self._quote_at(getattr(self, "book_quotes", {}).get(token), t, self.BOOK_AGE) or \
            self._quote_at(self.prices.get(token), t, self.stale)

    def view(self, market, t, min_volume_24h):
        p = self.price(market["token_id"], t)
        v = self.volume_24h(market["token_id"], t)
        return p, v, p is not None and 0 < p < 1 and v >= min_volume_24h

    @classmethod
    def taker_fee(cls, price, contracts):
        """Dollars for one taker order of `contracts` at `price` (rounded up to the cent)."""
        return math.ceil(round(cls.TAKER_FEE * contracts * price * (1 - price) * 100, 6)) / 100



# ---------------------------------------------------------------------------
# The private book: our quotes, the simulated crowd
# ---------------------------------------------------------------------------

def _snaps(run_dir):
    out = {}
    for f in sorted((run_dir / "snaps").glob("*.json.gz")):
        with gzip.open(f, "rt") as g:
            s = json.load(g)
        out[s.get("ts")] = s
    return out


def _read_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


class OG(Kalshi):
    """Recorded OG.com data (exchanges/og.toml; markets/exchange_driver.py stores its minute prices, tape and books in
    the shared tables, per instrument, archived under data/archive/markets/og). Every price stored counts, whatever
    its spread, side or depth, with one exception: a stored 0.50 is the midpoint an empty book used to produce, so it is
    not a price unless a trade at that time says so. A trade at any price is one. The price at t is the latest of the
    minute prices (0.50 dropped), the trades and the books' quotes (their mid, or the one side that quotes), within the
    staleness window. Fees: a flat taker fee per contract (the schema's taker_fee_per_contract, $0.02, unverified),
    not Kalshi's P x (1 - P) schedule; the replay adds it per share (fees())."""

    code = "og"
    TAKER_FEE = 0.0                  # no P x (1 - P) rate: OG.com charges per contract (FEE_PER_CONTRACT)
    MAKER_FEE = 0.0
    FAKE_MID = 0.5

    def __init__(self, conn, links, start, end, group_target=None, coherence_tol=0.25, stale=STALE):
        from racinglines.markets import store as MS
        self.links = links
        self.stale = stale
        self.group_target = group_target or {}
        self.coherence_tol = coherence_tol
        a, b = pd.Timestamp(start).tz_localize("UTC"), pd.Timestamp(end).tz_localize("UTC")
        root, toks = MS.root_for(self.code), links["token_id"].tolist()
        ph = MS.read(conn, "prices", tokens=toks, start=a - stale, end=b, root=root)[["token_id", "ts", "price"]]
        tr = MS.read(conn, "trades", tokens=toks, start=a - timedelta(hours=24), end=b, root=root)
        bk = MS.read(conn, "books", tokens=toks, start=a - stale, end=b, root=root)
        self.prices = {t: g for t, g in self.observed(ph, tr, bk).groupby("token_id")}
        tr = tr.assign(usd=tr["price"] * tr["size"])[["token_id", "ts", "usd"]]
        tr["ts"] = pd.to_datetime(tr["ts"], utc=True).dt.tz_localize(None)
        self.trades = {t: g for t, g in tr.groupby("token_id")}

    def quote(self, token, t):
        return None                  # OG trades at its observed price (no recorded quote rule yet)

    @classmethod
    def observed(cls, prices, trades, books):
        """(token_id, ts, price) of every price the store holds, time-ordered: minute prices other than the empty
        book's 0.50, every trade, and each book's mid (or its one quoting side)."""
        ph = prices[(prices["price"] - cls.FAKE_MID).abs() > 1e-9][["token_id", "ts", "price"]]
        tp = trades[["token_id", "ts", "price"]]
        bq = pd.DataFrame(columns=["token_id", "ts", "price"])
        if len(books):
            bid = books["best_bid"].where(books["best_bid"] > 0)
            ask = books["best_ask"].where(books["best_ask"] < 1)
            bq = books.assign(price=((bid + ask) / 2).fillna(bid).fillna(ask))[["token_id", "ts", "price"]]
            bq = bq[bq["price"].notna()]
        df = pd.concat([x for x in (ph, tp, bq) if len(x)], ignore_index=True) if any(len(x) for x in (ph, tp, bq)) \
            else pd.DataFrame(columns=["token_id", "ts", "price"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
        df["price"] = df["price"].astype(float)
        return df.sort_values(["token_id", "ts"], kind="stable").reset_index(drop=True)

    @classmethod
    def fee_per_contract(cls):
        from racinglines import exchanges
        return float(exchanges.load(cls.code)["exchange"].get("taker_fee_per_contract", 0.0))

    @classmethod
    def fees(cls, trades):
        """Dollars of OG.com's flat taker fee on the trades (per contract bought or sold)."""
        return float(trades["shares"].abs().sum() * cls.fee_per_contract()) if len(trades) else 0.0

    @classmethod
    def per_share_fee(cls):
        return cls.fee_per_contract()

    @classmethod
    def tape_prices(cls, conn, tokens, start, end):
        """The observed prices (observed(): the minute prices without the empty book's 0.50, the trades, the books'
        quotes), so a maker replay never reads the empty book's midpoint as a market."""
        from racinglines.markets import store as MS
        root, toks = MS.root_for(cls.code), list(tokens)
        ph = MS.read(conn, "prices", tokens=toks, start=start, end=end, root=root)[["token_id", "ts", "price"]]
        tr = MS.read(conn, "trades", tokens=toks, start=start, end=end, root=root)
        bk = MS.read(conn, "books", tokens=toks, start=start, end=end, root=root)
        return cls.observed(ph, tr, bk)


EXCHANGES = {v.code: v for v in (Polymarket, Kalshi, OG)}  # code -> venue class: its fees, how its tape is read


class PrivateBook:
    """A live run's private book replayed as a backtest. Each poll is what the crowd saw and how it was
    drawn: the quotes, the logged seed, and the rate (downhill: one poll's hits, scaled by the poll interval
    and the late pace; F1: one window's batch). Replaying with the recorded quotes re-derives the live
    book exactly; `quoter` re-quotes every market from its fair value and the replayed inventory instead,
    to backtest another quoting rule against the same crowd draws."""

    code = "private"

    def __init__(self, polls, params, outcomes, fair):
        self.polls, self.params, self.outcomes, self.fair = polls, params, outcomes, fair

    @classmethod
    def from_run(cls, run_dir):
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        latest = json.loads((run_dir / "latest.json").read_text())
        sport = latest.get("sport") or ("mtb_dh" if "slug" in meta else "f1")
        snaps = _snaps(run_dir)
        crowd = _read_jsonl(run_dir / "crowd.jsonl")
        if sport == "mtb_dh":
            fix = run_dir / "replay.json"
            return cls._downhill(meta, latest, snaps, crowd, json.loads(fix.read_text()) if fix.exists() else {},
                                 _read_jsonl(run_dir / "history.jsonl"))
        return cls._windows(latest, snaps, crowd, run_dir)

    @classmethod
    def _downhill(cls, meta, latest, snaps, crowd, fix=None, history=()):
        """Each poll's rate: the logged `intensity` (runs since 2026-09-28), else re-derived with today's rule
        (poll interval / BASE_INTERVAL, x the late pace). A run whose rule changed while it ran says so in its
        replay.json: `unscaled_until` (ISO) = polls before it ran at 1 (x the late pace), whatever their interval
        (Whistler: the interval scaling arrived with the 22:37:58 restart).

        Two polls logged in the same second share one snapshot file (the later overwrites it), so their quotes
        come from history.jsonl instead, the n-th line of that second for the n-th poll; replay.json's
        `intensity` {"<ts>#<n>": rate} gives their rates when they weren't logged (Whistler: two loops overlapped
        for a minute after the 22:37:58 restart)."""
        from racinglines.pipelines import live_dh as L
        p = meta["params"]
        params = C.Params(takers=p["CROWD"], p=p["CROWD_P"], budget=tuple(p["CROWD_BUDGET"]), bet=tuple(p["CROWD_BET"]),
                          min_bet=p["MIN_BET"], max_pos=p["MAX_POS"], seed=L.CROWD_SEED, late_cap=p["LATE_CAP"],
                          late_pace=p["LATE_PACE"])
        polls, seen, hist = [], {}, {}
        for h in history:
            hist.setdefault(h.get("ts"), []).append(h)
        same_second = {t for t in (e["ts"] for e in crowd) if sum(x["ts"] == t for x in crowd) > 1}
        for e in crowd:
            s = snaps.get(e["ts"])
            if s is None:
                raise ValueError(f"no snapshot for the crowd poll at {e['ts']}")
            n = seen[e["ts"]] = seen.get(e["ts"], -1) + 1
            quotes = s["quotes"]
            if e["ts"] in same_second:
                h = hist[e["ts"]][n]
                quotes = [dict(bib=int(k.split(":")[0]), market=k.split(":", 1)[1], fair=h["fair"].get(k), bid=q[0], ask=q[1])
                          for k, q in h["quotes"].items()]
            b = s.get("betting") or {}
            late = bool(b.get("late"))
            if (fix or {}).get("intensity", {}).get(f"{e['ts']}#{n}") is not None:
                rate = fix["intensity"][f"{e['ts']}#{n}"]
            elif e.get("intensity") is not None:
                rate = e["intensity"]
            elif (fix or {}).get("unscaled_until") and e["ts"] < fix["unscaled_until"]:
                rate = params.late_pace if late else 1.0
            else:
                rate = (b.get("interval") or L.BASE_INTERVAL) / L.BASE_INTERVAL * (params.late_pace if late else 1.0)
            polls.append(dict(ts=e["ts"], seed=e["seed"], quotes=quotes, late=late, late_at=b.get("late_at"),
                              done=bool(s.get("done")), mode="poll", intensity=rate))
        outcomes = {f"{o['bib']}:{o['market']}": o["yes"] for o in latest.get("outcomes") or []}
        fair = {C.mkey(q): q["fair"] for q in latest.get("quotes") or []}
        return cls(polls, params, outcomes, fair)

    @classmethod
    def _windows(cls, latest, snaps, crowd, run_dir):
        live = json.loads((run_dir / "meta.json").read_text()).get("settings") or {}
        params = C.Params.from_dict(dict(live.get("crowd") or {}, max_loss=(live.get("quoting") or {}).get("max_loss")))
        order = sorted(snaps)
        polls = []
        for e in crowd:
            start = e["window"][0]
            prev = [ts for ts in order if ts <= start]              # the quotes posted when the window opened
            quotes = [dict(key=m["key"], fair=m["fair"], bid=m.get("bid"), ask=m.get("ask"))
                      for m in (snaps[prev[-1]].get("markets") if prev else [])]
            polls.append(dict(ts=e["ts"], seed=e["seed"], quotes=quotes, late=bool(e.get("late")), done=False,
                              mode="window", window=e["window"]))
        outcomes = {o["key"]: o["yes"] for o in latest.get("outcomes") or []}
        fair = {m["key"]: m["fair"] for m in latest.get("markets") or []}
        return cls(polls, params, outcomes, fair)

    def replay(self, quoter=None):
        """The book after every poll. quoter(quote, inv) -> (bid, ask): None = the recorded quotes.
        Returns dict(book, fills, pnl) (P&L settled where the outcome is known, else marked to fair)."""
        p = self.params
        book = C.new_book(p)
        fills = []
        for e in self.polls:
            if e["late"] and not book.get("late"):
                C.open_late(book, p, e.get("late_at"))
            if e["done"]:
                continue
            quotes = e["quotes"]
            if quoter is not None:
                quotes = []
                for q in e["quotes"]:
                    bid, ask = quoter(q, book["markets"].get(C.mkey(q), {}).get("inv", 0.0))
                    quotes.append(dict(q, bid=bid, ask=ask))
            pot = "late_left" if book.get("late") else "left"
            rng = np.random.default_rng(e["seed"])
            if e["mode"] == "poll":
                f = C.fills(quotes, book, rng, p, intensity=e["intensity"], pot=pot)
            else:
                a, b = (pd.Timestamp(x).to_pydatetime() for x in e["window"])
                f = C.window(quotes, book, rng, p, a, b, pace=p.late_pace if e["late"] else 1.0, pot=pot)
            fills += [dict(x, poll=e["ts"]) for x in f]
        return dict(book=book, fills=fills, pnl=C.book_pnl(book, self.fair, self.outcomes))

    def resolve(self, key):
        return self.outcomes.get(key)


def spread_quoter(half_spread, max_pos=2500.0, skew=1.0, lo=0.01, hi=0.99):
    """A quoting rule for PrivateBook.replay: a fixed half-spread around the recorded fair, leaning against
    inventory (markets/quoting.py). Markets the live maker didn't quote stay unquoted."""
    from racinglines.markets import quoting as Q

    def quoter(q, inv):
        if q.get("bid") is None and q.get("ask") is None:
            return None, None
        f = q.get("fair")
        if f is None or math.isnan(f) or f <= lo or f >= hi:
            return None, None
        return Q.quote(f, half_spread, inv, max_pos, skew, tidy=False)
    return quoter
