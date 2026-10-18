"""Probalytics' tape as a backtest venue: the same interface as markets/venue_replay.py's Polymarket and Kalshi
classes (markets, view, price, quote, volume_24h, coherent, touch_depth, resolve), read from the Parquet files that
scripts/probes/probalytics_pull.py writes, so a sweep can be re-run on their books and fills and compared with ours.

Switched on for a backtest's venues (weekend_sweep's and position_replay's) by the sweep setting tape_source =
"probalytics" or RACINGLINES_TAPE_SOURCE=probalytics (off by default: nothing changes unless one is set). Read from
$RACINGLINES_DATA/probalytics/, by the sport's tag (sports/<code>.toml [markets.probalytics] tag, e.g. F1 or CYC):

    stages/<event_key>_<VENUE>.parquet      the book as of each stage cutoff (the last snapshot at or before it)
    books/<YYYY-MM-DD>_<TAG>_<VENUE>.parquet every snapshot of a day, when pulled (preferred: the full timeline)
    fills/<TAG>_<VENUE>_<YYYY-MM>.parquet   every fill
TAG is the pull's market set: F1 (our linked markets, the default), F1X (F1 markets we have not linked), CYC (cycling).

What differs from our own tape, on purpose:
- price() is the book's mid (best bid + best ask) / 2 where we read Polymarket's 5-minute price history; quote()
  is the book's touch on both venues' Kalshi path, as ours is from our 5-minute Kalshi snapshots.
- Their snapshots are written on change, not on a clock, so a snapshot stays the book until the next one: it is
  stale only after BOOK_STALE (48 hours, the stage files' lookback; a sport's [markets.probalytics]
  book_stale_hours overrides it), not after venue_replay.STALE's 6 hours.
- volume_24h() sums price x size of their fills (both outcomes of a Polymarket condition, one Kalshi ticker).
Keys are exact: a Polymarket token is their outcome platform id, a Kalshi ticker their market platform id (YES book).

Fills at the second of an order: with levels=True a venue keeps every level, book_at(token, t) gives the whole book
as of t, and fantasy_book(venue, t) adapts it to fantasy/exchange.place's get_book, so a backtest prices an order by
walking Probalytics' book at that moment instead of our latest 5-minute snapshot.

From the bucket: with PROBALYTICS_BUCKET=gs://<bucket>/vendor/probalytics, a file missing under the data folder is
fetched with `gcloud storage cp` (the Mac's login, the VM's service account) before it is read, and with
PROBALYTICS_CACHE_EVICT=1 deleted again once it is loaded (the VM's disk holds a weekend, not the 19 GB archive).
"""

from __future__ import annotations

import os
import subprocess
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.markets.venue_replay import Kalshi, Polymarket

ENV = "RACINGLINES_TAPE_SOURCE"
BOOK_STALE = timedelta(hours=48)
LEVEL_COLS = ("bid_px", "bid_sz", "ask_px", "ask_sz")
DEFAULT_TAG = "F1"


def enabled(setting=None):
    """Whether a backtest reads this tape: the sweep setting tape_source, else RACINGLINES_TAPE_SOURCE."""
    return setting == "probalytics" or os.environ.get(ENV, "").strip().lower() == "probalytics"


def tag_of(sport):
    """The sport's name in the tape's file names (sports/<code>.toml [markets.probalytics] tag)."""
    from racinglines import sports
    tag = sports.probalytics(sport).get("tag")
    if not tag:
        raise ValueError(f"sport {sport!r}: no Probalytics tape (sports/{sport}.toml [markets.probalytics] tag)")
    return str(tag)


def book_stale_of(sport):
    """How long a snapshot stays the sport's book: [markets.probalytics] book_stale_hours, else BOOK_STALE."""
    from racinglines import sports
    h = sports.probalytics(sport).get("book_stale_hours")
    return BOOK_STALE if h is None else timedelta(hours=float(h))


def root():
    from racinglines.paths import DATA
    return Path(os.environ.get("PROBALYTICS_DATA", DATA / "probalytics"))


def bucket():
    return os.environ.get("PROBALYTICS_BUCKET", "").strip().rstrip("/")


def _days(a, b):
    d = a.normalize()
    while d <= b:
        yield d
        d += pd.Timedelta(days=1)


def _months(a, b):
    m = a.normalize().replace(day=1)
    while m <= b:
        yield m
        m = (m + pd.Timedelta(days=32)).replace(day=1)


# --- the bucket -----------------------------------------------------------------------------------------------------

_LISTED = {}       # bucket prefix -> the object names under it (one listing per prefix per process)


def _listed(prefix):
    if prefix not in _LISTED:
        r = subprocess.run(["gcloud", "storage", "ls", f"{prefix}/"], capture_output=True, text=True, timeout=600)
        _LISTED[prefix] = {x.rsplit("/", 1)[-1] for x in r.stdout.split()} if r.returncode == 0 else set()
    return _LISTED[prefix]


def fetch(paths, base=None):
    """Copy the files of `paths` that are missing locally but in PROBALYTICS_BUCKET down to the same place under
    base; -> the paths fetched (so the caller can evict them). Nothing happens without the bucket setting."""
    b = bucket()
    if not b:
        return []
    base = Path(base or root())
    want = {}
    for p in map(Path, paths):
        if p.exists():
            continue
        rel = p.relative_to(base)
        prefix = f"{b}/{rel.parent.as_posix()}"
        if rel.name in _listed(prefix):
            want.setdefault(p.parent, []).append(f"{prefix}/{rel.name}")
    got = []
    for d, urls in want.items():
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", *urls, str(d)], check=True, capture_output=True, timeout=3600)
        got += [d / u.rsplit("/", 1)[-1] for u in urls]
    return got


def _evict(paths):
    if os.environ.get("PROBALYTICS_CACHE_EVICT", "").strip() == "1":
        for p in paths:
            Path(p).unlink(missing_ok=True)


def _read_table(paths, cols, filters, base=None):
    """The rows of `paths` (fetched from the bucket first when set) as one Arrow table, or None."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    got = fetch(paths, base)
    try:
        parts = [pq.read_table(p, columns=cols, filters=filters) for p in map(Path, paths) if p.exists()]
    finally:
        _evict(got)
    parts = [x for x in parts if x.num_rows]
    return pa.concat_tables(parts, promote_options="permissive") if parts else None


def _read(paths, cols, filters, base=None):
    t = _read_table(paths, cols, filters, base)
    return t.to_pandas() if t is not None else pd.DataFrame(columns=cols)


# --- reads ----------------------------------------------------------------------------------------------------------

def best_levels(px, sz, highest):
    """(price, size at that price) of each row's best level, vectorized over Arrow list columns (a race weekend is
    5-20M snapshots): the highest price for bids, the lowest for asks, whatever order the levels come in; NaN and 0
    for an empty side. px and sz have the same shape (the pull writes both from one level array)."""
    px = px.combine_chunks() if hasattr(px, "combine_chunks") else px
    sz = sz.combine_chunks() if hasattr(sz, "combine_chunks") else sz
    offs = np.asarray(px.offsets, dtype=np.int64)
    offs = offs - offs[0]
    vals = np.asarray(px.flatten(), dtype=float)
    sizes = np.asarray(sz.flatten(), dtype=float)
    lens = np.diff(offs)
    price, size = np.full(len(lens), np.nan), np.zeros(len(lens))
    ne = lens > 0
    if ne.any():
        starts = offs[:-1][ne]
        price[ne] = (np.maximum if highest else np.minimum).reduceat(vals, starts)
        at_best = vals == np.repeat(price, lens)
        size[ne] = np.add.reduceat(np.where(at_best, sizes, 0.0), starts)
    return price, size


def _touch_frame(t, venue, key_col, levels):
    """One file's Arrow table -> the per-snapshot frame: the touch, plus the levels when asked for."""
    import pyarrow.compute as pc
    if venue == "kalshi":
        t = t.filter(pc.equal(t["outcome_pid"], "yes"))
    if not t.num_rows:
        return None
    bid, bid_size = best_levels(t["bid_px"], t["bid_sz"], highest=True)
    ask, ask_size = best_levels(t["ask_px"], t["ask_sz"], highest=False)
    out = pd.DataFrame(dict(key=t[key_col].to_pandas().astype(str).to_numpy(),
                            ts=pd.to_datetime(t["timestamp"].to_pandas(), utc=True).dt.tz_localize(None).to_numpy(),
                            bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size))
    if levels:
        for c in LEVEL_COLS:
            out[c] = t[c].to_pandas().to_numpy()
    return out


def read_books(venue, event_key, start, end, keys, base=None, tag=DEFAULT_TAG, levels=False, stale=BOOK_STALE,
               day_books=True):
    """One row per snapshot of the keys' books over [start - stale, end]: key, ts (naive UTC), bid, ask,
    bid_size, ask_size (at the touch); with levels=True also bid_px, bid_sz, ask_px, ask_sz (every level, as pulled).
    The day files of the sport's `tag` when any were pulled for the window (locally or in the bucket), else the stage
    file (when there is an event_key). One file is read and reduced at a time and the level lists are dropped unless
    asked for: a race day's lists are gigabytes in memory (the VM's comparison was OOM-killed at Singapore on
    2026-10-14 reading four days at once; per file, all 1,840 Polymarket tokens of those days peak at 2.1 GB)."""
    import pyarrow.parquet as pq
    base = Path(base or root())
    code = venue.upper()
    key_col = "outcome_pid" if venue == "polymarket" else "market_platform_id"
    cols = list(dict.fromkeys([key_col, "outcome_pid", "timestamp", *LEVEL_COLS]))
    out_cols = ["key", "ts", "bid", "ask", "bid_size", "ask_size"] + (list(LEVEL_COLS) if levels else [])
    a = pd.Timestamp(start) - stale
    days = [base / "books" / f"{d:%Y-%m-%d}_{tag}_{code}.parquet" for d in _days(a, pd.Timestamp(end))]
    if not day_books:                 # the stage file only, as for every weekend without day books (sweeps)
        days = []
    filt = [(key_col, "in", list(keys))]
    got = fetch(days, base)
    parts = []
    try:
        paths = [p for p in days if p.exists()]
        if not paths and event_key is not None:
            stage = base / "stages" / f"{event_key}_{code}.parquet"
            got += fetch([stage], base)
            paths = [stage] if stage.exists() else []
        for p in paths:
            f = _touch_frame(pq.read_table(p, columns=cols, filters=filt), venue, key_col, levels)
            if f is not None and len(f):
                parts.append(f)
    finally:
        _evict(got)
    if not parts:
        return pd.DataFrame(columns=out_cols)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["key", "ts"], keep="last").sort_values(["key", "ts"], kind="stable")
    return out.reset_index(drop=True)


def read_fills(venue, start, end, keys, base=None, tag=DEFAULT_TAG):
    """key, ts (naive UTC), usd (price x size) of every fill of the keys (Polymarket condition ids or Kalshi
    tickers) over [start, end], from the sport's `tag` files, one row per fill_id."""
    base = Path(base or root())
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    paths = [base / "fills" / f"{tag}_{venue.upper()}_{m:%Y-%m}.parquet" for m in _months(a, b)]
    cols = ["market_platform_id", "timestamp", "price", "size"]
    got = fetch(paths, base)                     # fetched once here; _read below finds them local
    try:
        import pyarrow.parquet as pq
        have = next((set(pq.read_schema(p).names) for p in map(Path, paths) if p.exists()), set())
        df = _read(paths, cols + (["fill_id"] if "fill_id" in have else []), [("market_platform_id", "in", list(keys))],
                   base)
    finally:
        _evict(got)
    if len(df) and "fill_id" in df:
        # their fills table repeats rows (2026-07 Kalshi F1: 97,432 exact duplicates of 298,487, the British GP twice)
        df = df.drop_duplicates("fill_id")
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None) if len(df) else pd.Series([], dtype="datetime64[ns]")
    out = pd.DataFrame(dict(key=df["market_platform_id"].astype(str) if len(df) else [], ts=ts,
                            usd=(df["price"] * df["size"]).astype(float) if len(df) else []))
    out = out[(out["ts"] >= a) & (out["ts"] <= b)]
    return out.sort_values(["key", "ts"]).reset_index(drop=True)


# --- the venues -----------------------------------------------------------------------------------------------------

class _Tape:
    """The shared reads: books per key, fills per market."""

    book_stale = BOOK_STALE

    def _load(self, venue, event_key, links, start, end, book_key, fill_key, base=None, levels=False, tag=None,
              book_stale=None, day_books=True):
        if book_stale is not None:
            self.book_stale = book_stale
        tag = tag or DEFAULT_TAG
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        bk = read_books(venue, event_key, a, b, links[book_key].dropna().astype(str).unique(), base, tag=tag,
                        levels=levels, stale=self.book_stale, day_books=day_books)
        fl = read_fills(venue, a - timedelta(hours=24), b, links[fill_key].dropna().astype(str).unique(), base, tag)
        self.pb_books = {k: g for k, g in bk.groupby("key")}
        self.pb_fills = {k: g for k, g in fl.groupby("key")}

    def _book_at(self, key, t):
        g = self.pb_books.get(key)
        if g is None:
            return None
        ts = pd.DatetimeIndex(g["ts"])
        i = ts.searchsorted(pd.Timestamp(t), side="right") - 1
        if i < 0 or pd.Timestamp(t) - ts[i] > self.book_stale:
            return None
        return g.iloc[i]

    def _touch(self, key, t):
        r = self._book_at(key, t)
        if r is None or pd.isna(r["bid"]) or pd.isna(r["ask"]) or not 0 < r["bid"] <= r["ask"] < 1:
            return None
        return float(r["bid"]), float(r["ask"])

    def _vol(self, key, t):
        g = self.pb_fills.get(key)
        if g is None:
            return 0.0
        a, b = g["ts"].searchsorted(t - timedelta(hours=24)), g["ts"].searchsorted(t, side="right")
        return float(g["usd"].iloc[a:b].sum())

    def _book_key(self, token):
        return str(token)

    def touch_depth(self, token, t, max_age=None):
        r = self._book_at(self._book_key(token), t)
        if r is None:
            return None
        return float(r.get("ask_size") or 0.0), float(r.get("bid_size") or 0.0)

    def load_books(self, conn, start, end):
        """The books came with the tape (touch_depth reads them)."""

    def book_at(self, token, t):
        """The whole YES book of `token` as of t (naive UTC): dict(ts=the snapshot's time, bids, asks), each a list
        of (price, size) best first (bids high to low, asks low to high), or None when there is no snapshot within
        book_stale. Needs the venue built with levels=True."""
        r = self._book_at(self._book_key(token), t)
        if r is None or "bid_px" not in r:
            return None

        def side(px, sz, desc):
            return sorted(((float(p), float(q)) for p, q in zip(px, sz) if q > 0), key=lambda x: x[0], reverse=desc)
        return dict(ts=pd.Timestamp(r["ts"]).to_pydatetime(), bids=side(r["bid_px"], r["bid_sz"], True),
                    asks=side(r["ask_px"], r["ask_sz"], False))


class ProbalyticsPolymarket(_Tape, Polymarket):
    """Polymarket on Probalytics' tape: price = the book's mid, trades at it (as ours trade at the price history);
    with touch_quotes (the taker_fill = touch setting) its stages carry the book's touch, so the taker buys at the ask
    and sells at the bid as on Kalshi."""

    touch_quotes = False

    def quote(self, token, t):
        return self._touch(str(token), t) if self.touch_quotes else None

    def __init__(self, conn, links, start, end, group_target=None, coherence_tol=0.25, stale=None, event_key=None,
                 base=None, levels=False, tag=None, book_stale=None, day_books=True):
        self.links, self.stale = links, stale
        self.group_target, self.coherence_tol = group_target or {}, coherence_tol
        self._load("polymarket", event_key, links, start, end, "token_id", "condition_id", base, levels=levels, tag=tag,
                   book_stale=book_stale, day_books=day_books)

    def price(self, token, t):
        q = self._touch(str(token), t)
        return None if q is None else (q[0] + q[1]) / 2

    def volume_24h(self, condition, t):
        return self._vol(str(condition), t)


class ProbalyticsKalshi(_Tape, Kalshi):
    """Kalshi on Probalytics' tape: the taker meets the book's touch; price = its mid."""

    def __init__(self, conn, links, start, end, group_target=None, coherence_tol=0.25, stale=None, event_key=None,
                 base=None, levels=False, tag=None, book_stale=None, day_books=True):
        self.links, self.stale = links, stale
        self.group_target, self.coherence_tol = group_target or {}, coherence_tol
        self._load("kalshi", event_key, links, start, end, "token_id", "token_id", base, levels=levels, tag=tag,
                   book_stale=book_stale, day_books=day_books)

    def price(self, token, t):
        q = self._touch(str(token), t)
        return None if q is None else (q[0] + q[1]) / 2

    def quote(self, token, t):
        return self._touch(str(token), t)

    def volume_24h(self, ticker, t):
        return self._vol(str(ticker), t)


EXCHANGES = {"polymarket": ProbalyticsPolymarket, "kalshi": ProbalyticsKalshi}


def fantasy_book(venue, t):
    """fantasy/exchange.place's get_book for a backtest order at time t: Probalytics' book as of t, from a venue built
    with levels=True. Their snapshots are written on change, so a book stands until the next one: ts is reported as t
    (place's 15-minute staleness check is about our recorder's lag, not this tape's) and the snapshot's own time is
    kept as snapshot_ts; a book older than BOOK_STALE is None, which place refuses as book_unavailable."""
    at = pd.Timestamp(t)
    naive = at.tz_convert("UTC").tz_localize(None) if at.tzinfo else at
    aware = (at if at.tzinfo else at.tz_localize("UTC")).to_pydatetime()

    def get(token):
        b = venue.book_at(token, naive)
        if b is None:
            return None
        return dict(ts=aware, snapshot_ts=b["ts"].replace(tzinfo=timezone.utc), bids=b["bids"], asks=b["asks"])
    return get
