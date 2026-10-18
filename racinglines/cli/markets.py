"""
racinglines markets <command>: exchange data (Polymarket and Kalshi; F1 plus the tape-only sports).

    sync       Sync the exchange's events into market links (live prices, resolutions)
    history    Store price history for events (--fidelity 1 = minute-level)
    trades     Store every taker trade for events (the tape a maker replay fills against)
    record     Record order-book snapshots every --interval seconds (archives to Parquet hourly)
    archive    Move stale prices / trades / books from Postgres to Parquet (--stats: sizes)
    disagree   Cross-venue disagreement log: Polymarket vs Kalshi on every outcome listed on both, per tick
               (--event 2026-15, a round number, or season [--year]; --start/--end UTC, --step minutes, --no-save)
    settle-check  Read-only, any --exchange and --sport: our outcome from the results (markets/kinds.settle) beside
               the exchange's resolved_yes on every resolved link; lists disagreements, exit 1 if any [--out FILE.csv]

Options: --exchange polymarket (default) --sport f1 (default), then the command's own options.

Polymarket for a tape-only sport (--sport nascar|motogp|indycar; sports/<code>.toml [markets.polymarket] tags,
unverified against the live Gamma API; markets/polymarket/sync.py). Additive: rows are upserted, never deleted:
    sync       [--year 2026] [--closed] [--tags SLUG …]   the sport's Polymarket events into market links
    trades     [--events SLUG …]                          the tape of those events' markets
    history    [--events SLUG …] --start --end [--fidelity 60]   price history (minutes per point)
    books      [--events SLUG …]                          one order-book snapshot per open market
(--sport f1 keeps the commands above, except `trades` and `history` without --events, which take every F1 market
the same way; `record` is F1 only.) `trades --open --since-hours H` on any exchange: only markets not closed, only
trades from H hours ago on (the recorder's pass, scripts/vm/record_venues.sh and pm_sync.sh; the one-off backfill,
scripts/vm/backfill_tape.sh, takes --since-hours alone).

Kalshi (--exchange kalshi; markets/kalshi/, built on mocked responses, unverified against the live API):
    sync       [--year 2026] [--closed] [--series TICKER …]   the sport's Kalshi markets into market links
    trades     [--events TICKER …]                       the tape of those events' markets
    history    [--events TICKER …] --start --end [--period 60]   candlesticks (minutes: 1, 60, 1440): price, bid, ask
               [--save-raw DIR | --from-raw DIR]          also keep Kalshi's raw responses / re-import them offline
    books      [--events TICKER …]                       one order-book snapshot per open market
    coverage   [--events TICKER …]                       stored prices with the candle's bid/ask (Parquet + Postgres)

Exchanges defined as schemas (--exchange og; exchanges/<code>.toml, markets/exchange_driver.py), read-only:
    sync       [--year 2026]                             the sport's markets and quotes into market links
    settle     [--since UTC] [--max-pages N]             outcomes of closed links from the settlement feed
               (resolved_yes; a void is recorded in params and left empty); a schema with no feed does nothing
               --check: no scan, read-only: compare our settlement with resolved_yes (= settle-check)
    trades     [--events SYMBOL …]                       the tape the exchange still serves (about a month)
    history    [--events SYMBOL …] --start [--end]       minute prices (clipped to what the exchange keeps)
    books      [--events SYMBOL …]                       one order-book snapshot per open market
    fair       the model's fair price beside the quote, net of the taker fee: a simple indicator, no trades
    buy-all    debug, read-only: one YES and one NO of every market (modeled or not) at the first price stored
               (minute price, trade, book side or the sync's quote, whatever the spread or depth), held; P&L settled
               when resolved, else marked at the last price stored [--cost 0.01] [--fee F] [--out FILE.csv]

--sport names the sport (default f1): the tape-only sports (nascar, motogp, indycar; sports/<code>.toml
[markets.kalshi] / [markets.polymarket]) are synced only when named, every link unmodeled, under their own
competition. Without --events, trades / history / books take every event of --sport's competition on the
exchange (books: open markets).
"""

import argparse
import sys

COMMANDS = {"sync": "pm-sync", "history": "pm-history", "trades": "pm-trades", "record": "pm-record",
            "archive": "pm-archive"}


def _since(args):
    """The trades command's --since-hours as a UTC datetime (None: every trade the exchange serves)."""
    if getattr(args, "since_hours", None) is None:
        return None
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone.utc) - timedelta(hours=args.since_hours)


def kalshi_sports():
    """The sports the Kalshi sync knows: F1 and every schema with a [markets.kalshi] series list."""
    from racinglines import sports
    return ["f1"] + [c for c in sports.SPORT_CODES if c != "f1" and sports.kalshi_series(c)]


def polymarket_sports():
    """The sports the Polymarket sync knows: F1 and every schema with a [markets.polymarket] tags list."""
    from racinglines import sports
    return ["f1"] + [c for c in sports.SPORT_CODES if c != "f1" and sports.polymarket_tags(c)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="racinglines markets", description=__doc__, add_help=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from racinglines import exchanges
    ap.add_argument("--exchange", default="polymarket", choices=["polymarket", "kalshi", *exchanges.CODES])
    ap.add_argument("--sport", default="f1", choices=sorted({*kalshi_sports(), *polymarket_sports(), *(s for c in exchanges.CODES for s in exchanges.sports(c))}))
    ap.add_argument("--db", default=None)
    known, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "settle-check":
        return settle_check(known.exchange, known.db, known.sport, rest[1:])
    if known.exchange in exchanges.CODES:
        return schema_exchange(known.exchange, known.db, rest, known.sport)
    if known.exchange == "kalshi":
        return kalshi(known.db, rest, known.sport)
    if known.sport == "f1" and rest[:1] in (["trades"], ["history"]) and "--events" not in rest:
        return polymarket(known.db, rest, "f1")        # every F1 market, as the other sports (recorder, backfill_tape.sh)
    if known.sport != "f1":
        if known.sport not in polymarket_sports():
            print(f"--sport {known.sport}: only Kalshi lists it (--exchange kalshi)", file=sys.stderr)
            return 2
        return polymarket(known.db, rest, known.sport)
    if rest and rest[0] == "disagree":
        return disagree(known.db, rest[1:])
    if not rest or rest[0] in ("-h", "--help") or rest[0] not in COMMANDS:
        print(__doc__)
        return 0 if rest and rest[0] in ("-h", "--help") else 2
    from . import f1
    return f1.main((["--db", known.db] if known.db else []) + [COMMANDS[rest[0]]] + rest[1:])


def settle_check(exchange, db, sport="f1", argv=()):
    """racinglines markets --exchange X [--sport S] settle-check: read-only, our settlement from the results beside the
    exchange's resolved_yes on every resolved link (markets/settle_check.py). Exit 1 when any disagree."""
    ap = argparse.ArgumentParser(prog=f"racinglines markets --exchange {exchange} --sport {sport} settle-check")
    ap.add_argument("--out", default=None, help="Also write every checked link to this CSV.")
    args = ap.parse_args(list(argv))
    from racinglines.db.config import get_engine
    from racinglines.markets import settle_check as SC
    with get_engine(db).connect() as c:
        df = SC.check(c, exchange, sport)
    print(SC.text_report(df, exchange, sport))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 1 if (df["verdict"] == "disagree").any() else 0


def disagree(db, argv):
    """racinglines markets disagree (markets/disagree.py): backfill the cross-venue log from stored prices and print
    the report. --event YEAR-ROUND / ROUND (a race weekend, up to the race start) or season (the drivers' and
    constructors' champion markets of --year). The window defaults to where both venues have prices, on a --step
    minute grid."""
    ap = argparse.ArgumentParser(prog="racinglines markets disagree")
    ap.add_argument("--event", required=True, help="YEAR-ROUND (e.g. 2026-15), a round number of --year, or 'season'")
    ap.add_argument("--year", type=int, default=None, help="Season (default: this year)")
    ap.add_argument("--competition", default="f1_wdc", help="Competition code (default f1_wdc)")
    ap.add_argument("--start", default=None, help="UTC start, e.g. 2026-09-22T00:00 (default: first tick both venues quote)")
    ap.add_argument("--end", default=None, help="UTC end (default: the last, or the race start)")
    ap.add_argument("--step", type=int, default=60, help="Minutes between ticks (default 60)")
    ap.add_argument("--no-save", action="store_true", help="Print the report without writing market_disagreements")
    args = ap.parse_args(argv)
    from datetime import datetime, timedelta, timezone

    import pandas as pd
    from sqlalchemy import text

    from racinglines.db.config import get_engine
    from racinglines.markets import disagree as D
    year = args.year or datetime.now(timezone.utc).year
    engine = get_engine(db)
    with engine.begin() as c:
        comp = c.execute(text("SELECT id, name FROM competitions WHERE code = :c"), dict(c=args.competition)).fetchone()
        if comp is None:
            print(f"unknown competition {args.competition}", file=sys.stderr)
            return 2
        if args.event == "season":
            prs, title = D.pairs(c, competition_id=int(comp[0]), year=year), f"{comp[1]} {year} championship"
        else:
            key = args.event if "-" in args.event else f"{year}-{args.event}"
            race = c.execute(text("""SELECT ra.id, e.name, (SELECT min((ro.extra->>'session_date')::timestamp) FROM rounds ro
                                                          WHERE ro.race_id = ra.id AND ro.kind = 'race') AS race_start
                                     FROM races ra JOIN events e ON e.id = ra.event_id
                                     JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
                                     WHERE e.source_key = :k AND co.code = :c ORDER BY ra.id LIMIT 1"""),
                             dict(k=key, c=args.competition)).fetchone()
            if race is None:
                print(f"no race for event {key}", file=sys.stderr)
                return 2
            prs, title = D.pairs(c, race_id=int(race[0])), f"{race[1]} ({key})"
        if not len(prs):
            print(f"{title}: no outcome linked on both venues")
            return 0
        start, end = D.coverage(c, prs)
        if args.event != "season" and race[2] is not None and end is not None:
            end = min(end, pd.Timestamp(race[2]).tz_localize(timezone.utc))     # up to the start: after it, settlement
        if args.start:
            start = pd.Timestamp(args.start).tz_localize(timezone.utc)
        if args.end:
            end = pd.Timestamp(args.end).tz_localize(timezone.utc)
        if start is None or end is None:
            print(f"{title}: {len(prs)} outcomes on both venues, but no stored prices for both")
            return 0
        df = D.build(c, prs, start, end, step=timedelta(minutes=args.step))
        n = 0 if args.no_save else D.save(c, df)
    print(D.report(df, title))
    print(f"\n{len(prs)} outcomes on both venues · {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC every {args.step} min · "
          f"{n} rows {'would be ' if args.no_save else ''}written to market_disagreements")
    return 0


def kalshi(db, argv):
    """racinglines markets --exchange kalshi <command> (markets/kalshi/sync.py)."""
    ap = argparse.ArgumentParser(prog="racinglines markets --exchange kalshi")
def kalshi(db, argv, sport="f1"):
    """racinglines markets --exchange kalshi [--sport f1] <command> (markets/kalshi/sync.py)."""
    ap = argparse.ArgumentParser(prog=f"racinglines markets --exchange kalshi --sport {sport}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sync")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--closed", action="store_true", help="Also settled events")
    p.add_argument("--series", nargs="+", default=None, help="Series tickers to sync instead of discovering them")
    for name in ("trades", "history", "books", "coverage"):
        p = sub.add_parser(name)
        p.add_argument("--events", nargs="+", default=None,
                       help=f"Kalshi event tickers (market_links.condition_id); default: every {sport} event")
        if name == "trades":
            p.add_argument("--open", action="store_true", help="Only markets not closed (the recorder's pass)")
            p.add_argument("--since-hours", type=float, default=None, help="Only trades from this many hours ago on")
        if name == "history":
            p.add_argument("--start", required=True, help="UTC start, e.g. 2026-10-01T00:00")
            p.add_argument("--end", required=True, help="UTC end")
            p.add_argument("--period", type=int, default=60, choices=[1, 60, 1440], help="Minutes per candle")
            raw = p.add_mutually_exclusive_group()
            raw.add_argument("--save-raw", metavar="DIR",
                             help="Also save each market's raw candlesticks response to DIR/<series>/<ticker>.json")
            raw.add_argument("--from-raw", metavar="DIR",
                             help="Import from files saved by --save-raw instead of calling Kalshi")
    args = ap.parse_args(argv)
    from datetime import timezone

    import pandas as pd

    from racinglines.db.config import get_engine, get_session
    from racinglines.markets.kalshi import sync as KS
    with get_engine(db).connect() as c, get_session(db) as s:
        if args.cmd == "sync":
            print(KS.sync(s, c, args.year, include_closed=args.closed, sport=sport, series=args.series))
        elif args.cmd == "trades":
            print(f"{KS.fetch_trades(s, c, args.events, since=_since(args), sport=sport, open_only=args.open)} trades stored")
        elif args.cmd == "history":
            t = [pd.Timestamp(x).tz_localize(timezone.utc).to_pydatetime() for x in (args.start, args.end)]
            print(f"{KS.fetch_history(s, c, args.events, t[0], t[1], args.period, sport=sport, save_raw=args.save_raw, from_raw=args.from_raw)} price points stored")
        elif args.cmd == "coverage":
            print(KS.coverage(c, args.events, sport=sport))
        else:
            print(f"{KS.snapshot_books(s, c, args.events, sport=sport)} book snapshots stored")
    return 0


def polymarket(db, argv, sport):
    """racinglines markets --sport <tape-only sport> <command> (markets/polymarket/sync.py). F1 goes through cli/f1.py."""
    ap = argparse.ArgumentParser(prog=f"racinglines markets --sport {sport}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sync")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--closed", action="store_true", help="Also every closed event of that year")
    p.add_argument("--tags", nargs="+", default=None, help="Gamma tag_slug values to page instead of the schema's")
    for name in ("trades", "history", "books"):
        p = sub.add_parser(name)
        p.add_argument("--events", nargs="+", default=None,
                       help=f"Polymarket event slugs (market_links.event_slug); default: every {sport} event")
        if name == "trades":
            p.add_argument("--open", action="store_true", help="Only markets not closed (the recorder's pass)")
            p.add_argument("--since-hours", type=float, default=None, help="Only trades from this many hours ago on")
        if name == "history":
            p.add_argument("--start", required=True, help="UTC start, e.g. 2026-10-01T00:00")
            p.add_argument("--end", required=True, help="UTC end")
            p.add_argument("--fidelity", type=int, default=60, help="Minutes per price point")
    args = ap.parse_args(argv)
    from datetime import timezone

    import pandas as pd

    from racinglines.db.config import get_engine, get_session
    from racinglines.markets.polymarket import sync as PS
    with get_engine(db).connect() as c, get_session(db) as s:
        if args.cmd == "sync":
            print(PS.sync(s, c, args.year, include_closed=args.closed, sport=sport, tags=args.tags))
        elif args.cmd == "trades":
            print(f"{PS.fetch_trades(s, c, args.events, since=_since(args), sport=sport, open_only=args.open)} trades stored")
        elif args.cmd == "history":
            t = [pd.Timestamp(x).tz_localize(timezone.utc).to_pydatetime() for x in (args.start, args.end)]
            print(f"{PS.fetch_history(s, c, args.events, t[0], t[1], args.fidelity, sport=sport)} price points stored")
        else:
            print(f"{PS.snapshot_books(s, c, args.events, sport=sport)} book snapshots stored")
    return 0


def schema_exchange(code, db, argv, sport="f1"):
    """racinglines markets --exchange <code> [--sport f1] <command>: an exchange defined by a schema (exchanges/<code>.toml)."""
    from racinglines import exchanges
    if sport not in exchanges.sports(code):
        print(f"--sport {sport}: {code} lists {', '.join(exchanges.sports(code))} (exchanges/{code}.toml)", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog=f"racinglines markets --exchange {code} --sport {sport}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sync")
    p.add_argument("--year", type=int, default=2026)
    p = sub.add_parser("settle", help="Outcomes of closed links from the exchange's settlement feed (resolved_yes).")
    p.add_argument("--since", default=None, help="UTC time to scan the feed from (default: where each pending link's "
                                                 "last pass stopped, else a day before its last sync)")
    p.add_argument("--max-pages", type=int, default=None, help="Pages to read at most (default the schema's max_pages)")
    p.add_argument("--check", action="store_true", help="Read-only: no feed scan; compare our settlement from the results "
                                                        "with resolved_yes on every resolved link (settle-check)")
    for name in ("trades", "history", "books"):
        p = sub.add_parser(name)
        p.add_argument("--events", nargs="+", default=None, help="Event symbols (market_links.condition_id); default: every event")
        if name == "trades":
            p.add_argument("--open", action="store_true", help="Only markets not closed (the recorder's pass)")
            p.add_argument("--since-hours", type=float, default=None, help="Only trades from this many hours ago on")
        if name == "history":
            p.add_argument("--start", required=True, help="UTC start, e.g. 2026-09-01T00:00 (clipped to what the exchange keeps)")
            p.add_argument("--end", default=None, help="UTC end (default now)")
    sub.add_parser("fair")
    p = sub.add_parser("buy-all", help="Debug, read-only: one YES and one NO of every market at the first price stored, "
                                       "held; P&L settled or marked (markets/strategies/buy_everything.py).")
    p.add_argument("--cost", type=float, default=0.01, help="$ per contract on each side (default 0.01).")
    p.add_argument("--fee", type=float, default=None, help="$ taker fee per contract (default the schema's).")
    p.add_argument("--out", default=None, help="Also write the per-market rows to this CSV.")
    args = ap.parse_args(argv)
    from racinglines.db.config import get_engine, get_session
    from racinglines.markets import exchange_driver as D
    with get_engine(db).connect() as c, get_session(db) as s:
        if args.cmd == "sync":
            print(D.sync(s, c, code, sport, args.year))
        elif args.cmd == "settle" and args.check:
            return settle_check(code, db, sport)
        elif args.cmd == "settle":
            st = D.settle(s, c, code, sport, since=args.since, max_pages=args.max_pages)
            print(st if st.get("supported", True) else f"{code}: no settlement feed in exchanges/{code}.toml, nothing done")
        elif args.cmd == "trades":
            print(f"{D.fetch_trades(s, c, code, sport=sport, events=args.events, since=_since(args), open_only=args.open)} trades stored")
        elif args.cmd == "history":
            print(f"{D.fetch_history(s, c, code, args.start, args.end, sport=sport, events=args.events)} price rows stored")
        elif args.cmd == "books":
            print(f"{D.snapshot_books(s, c, code, sport=sport, events=args.events)} book snapshots stored")
        elif args.cmd == "buy-all":
            fee = exchanges.load(code)["exchange"].get("taker_fee_per_contract", 0.0) if args.fee is None else args.fee
            df = D.buy_all(c, code, sport, cost=args.cost, fee=fee)
            print(D.buy_all_text(df, code, sport, args.cost, fee))
            if args.out:
                df.to_csv(args.out, index=False)
                print(f"Wrote {args.out}")
        else:
            print(D.fair_text(D.fair_report(c, code, sport), code, exchanges.load(code)["exchange"].get("taker_fee_per_contract", 0.0)))
    return 0
