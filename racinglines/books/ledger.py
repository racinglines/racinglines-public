"""
A book's bets as ledger rows (docs/sportsbook/slips.md#ledger-export): `racinglines book export FILE --ledger OUT.csv`.

    rows(book, priced, settled)      one dict per staked line, in LEDGER_COLUMNS, from price_book's and settle_book's
                                     results (local or --remote); no database, no file
    ledger_rows(conn, book)          the same, pricing and settling the book on `conn` first
    write_csv(rows, path, force=)    the CSV; refuses an existing file unless force, never appends

A line is a bet when it carries a `stake` (books/schema.py); lines without one are skipped and counted. The columns:

    date_utc       the line's placed_utc, else the book's captured_utc, to the minute (2026-10-03T22:32Z)
    sport, event   from the line's legs as mapped by exact keys: "<season>-<round> <event>" (2026-16 Malaysia GP),
                   joined with " + " across races; an unmapped line gets the book's event key, said in notes
    venue          book.venue                         market, selection   the line's title and selection
    stake          the line's stake, in book.currency (never converted)
    odds_taken     decimal odds (the payout per unit staked); the book's own quote goes in notes when not decimal
    our_fair_prob  the model's probability of the whole bet (price_book), blank when it has none
    closing_odds   blank (no closing line yet)
    result         won / lost / void, else open (pending, manual or unmapped, said in notes)
    payout         the total returned: won = stake x the line's payout per unit, lost = 0, void = the stake, open blank
    follower       blank
    notes          the line id, the original quote, contracts for an exchange quote, "settled: ours" (our own results,
                   settle_book) on a settled line, and the flags and reasons behind the row

The row builder returns dicts so another writer (a database table) can take the same rows.
"""

import csv
import os

from racinglines.books import schema as S

LEDGER_COLUMNS = ("date_utc", "sport", "event", "venue", "market", "selection", "stake", "currency", "odds_taken",
                  "our_fair_prob", "closing_odds", "result", "payout", "follower", "notes")
SETTLED = ("won", "lost", "void")
CONTRACT_FORMATS = ("cents", "dollars")                  # a $1 contract: Kalshi in cents, Polymarket in dollars
CENTS_CURRENCY = "USD"                                    # a Kalshi contract pays $1, so its stake is in dollars


class LedgerError(ValueError):
    pass


def _minute(utc):
    """2026-10-03T22:32:10Z -> 2026-10-03T22:32Z."""
    return utc[:16] + "Z"


def _num(x, nd=4):
    """At least two decimals, up to nd, no float noise: 13 -> 13.00, 1.8474576 -> 1.8475."""
    s = f"{round(float(x), nd):.{nd}f}".rstrip("0")
    whole, _, frac = s.partition(".")
    return f"{whole}.{frac.ljust(2, '0')}"


def _where(line, book):
    """(sport, event, note) of a line from its mapped legs."""
    legs = [lg for lg in line["legs"] if lg.get("race_id") is not None and lg.get("season") is not None]
    if line["status"] != "mapped" or not legs:
        return book.get("sport") or "", book["event"], f"event: the book's key {book['event']} (not mapped to a race)"
    sports, events = [], []
    for lg in legs:
        ev = f"{lg['season']}-{int(lg['round']):02d} {lg['event']}" if lg.get("round") is not None else lg["event"]
        if lg["sport"] not in sports:
            sports.append(lg["sport"])
        if ev not in events:
            events.append(ev)
    return " + ".join(sports), " + ".join(events), None


def check_units(book):
    """A cents (Kalshi) book's stakes are dollars: its currency must be USD, never a count of cents or contracts."""
    b = book["book"]
    if b["odds"] == "cents" and b["currency"] != CENTS_CURRENCY:
        raise LedgerError(f"book.currency {b['currency']!r}: a cents book quotes $1 contracts, so its stakes are in "
                          f"{CENTS_CURRENCY}; write each stake in dollars and currency = \"{CENTS_CURRENCY}\"")


def rows(book, priced, settled):
    """(rows, skipped): a ledger row dict per staked line, and the ids of the lines with no stake."""
    check_units(book)
    b = book["book"]
    fmt = b["odds"]
    out, skipped = [], []
    for ln, p, s in zip(book["lines"], priced["lines"], settled["lines"]):
        if "stake" not in ln:
            skipped.append(s["id"])
            continue
        stake = float(ln["stake"])
        sport, event, where_note = _where(s, b)
        notes = [f"id: {s['id']}"]
        dec = s.get("decimal_odds") or p.get("decimal_odds")
        if dec is None and ln.get("odds") is not None:
            dec = 1.0 / S.to_prob(ln["odds"], fmt)
        if fmt != "decimal" and ln.get("odds") is not None:
            notes.append(f"odds: {ln['odds']} {fmt}")
            if fmt in CONTRACT_FORMATS:
                notes.append(f"contracts: {_num(stake / S.to_prob(ln['odds'], fmt), 2)}")
        if where_note:
            notes.append(where_note)
        result, payout = s.get("result"), None
        if result in SETTLED:
            if result == "void":
                payout = stake
            elif s.get("payout") is not None:
                payout = stake * float(s["payout"])
            notes.append("settled: ours")
            if payout is None:
                notes.append("payout: left to the book (a dropped leg without odds)")
        else:
            notes.append(f"result: {result}" if result else f"unmapped: {s.get('reason')}")
            result = "open"
        notes += [f"flag: {f}" for f in (p.get("flags") or [])]
        out.append({
            "date_utc": _minute(ln.get("placed_utc") or b["captured_utc"]), "sport": sport, "event": event,
            "venue": b["venue"], "market": s["title"], "selection": s["selection"], "stake": _num(stake, 2),
            "currency": b["currency"], "odds_taken": "" if dec is None else _num(dec),
            "our_fair_prob": "" if p.get("model_prob") is None else _num(p["model_prob"]), "closing_odds": "",
            "result": result, "payout": "" if payout is None else _num(payout, 2), "follower": "",
            "notes": "; ".join(notes)})
    return out, skipped


def ledger_rows(conn, book, runs=None, sims=None):
    """rows() of a book priced and settled on `conn`."""
    from racinglines.books import slips as B
    return rows(book, B.price_book(conn, book, runs=runs, sims=sims), B.settle_book(conn, book))


def write_csv(rows_, path, force=False):
    """Write the rows to a new CSV at `path` (the ledger's header, one row per bet). Refuses an existing file unless
    force; never appends."""
    if os.path.exists(path) and not force:
        raise LedgerError(f"{path} exists: give --force to replace it (the exporter never appends)")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        w.writeheader()
        w.writerows(rows_)
