"""
`racinglines data pull SPORT`: one sport's shared data from the hosted server into the local database and data folder
(racinglines/db/pull.py says what a pull holds). Sign-in is `--remote`'s (cli/remote.py: the browser once, or
RACINGLINES_MCP_TOKEN), and the remote-access warning is printed first.

    racinglines data pull f1                   everything for F1 (the first pull wants a fresh database: db init)
    racinglines data pull f1 --since last      a top-up from where this database's last F1 pull stopped
    racinglines data pull f1 --since 2026-09-01 --no-markets
"""

import argparse
import sys


def parser():
    ap = argparse.ArgumentParser(prog="racinglines data pull", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sport", help="a sport code (sports/<code>.toml), e.g. f1, nascar, motogp")
    ap.add_argument("--since", help="'last' (where this database's last pull stopped) or a UTC date: a top-up")
    ap.add_argument("--no-markets", action="store_true", help="race data and model runs only (no exchange history)")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded files (data/pulls/<pull id>/)")
    ap.add_argument("--force", action="store_true", help="a first pull into a database that already has model runs: "
                                                          "replace its race data and the rows that point at it")
    ap.add_argument("--db", help="the local database (default: DATABASE_URL or the CLI defaults)")
    ap.add_argument("--mcp-url", dest="mcp_url", help="the hosted server (default RACINGLINES_MCP_URL or racinglines.bet)")
    ap.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)      # a pull is always remote
    return ap


def main(argv):
    import shutil

    from racinglines import sports
    from racinglines.cli import remote as RM
    from racinglines.db import pull as PL
    from racinglines.db.config import database_url, get_engine
    args = parser().parse_args(argv)
    sport = {"cycling": "road_cycling"}.get(args.sport, args.sport)
    if sport not in sports.SPORT_CODES:
        print(f"data pull: unknown sport {args.sport!r}; one of {', '.join(sports.SPORT_CODES)}", file=sys.stderr)
        return 2
    db = database_url(args.db)
    since = PL.since_arg(args.since, db, sport)
    first = PL.first_into(db)
    try:
        with RM.Session(args.mcp_url) as s:
            m = s.call("pull_manifest", dict(sport=sport, since=since))
            files = [f for f in m["files"] if not (args.no_markets and f["kind"] == "market")]
            print(f"pull {m['pull_id']} from {RM.url(args.mcp_url)}: {sport}, "
                  f"{'since ' + m['since'] if m['since'] else 'everything'}, {len(files)} files", file=sys.stderr)
            out = PL.DIR / m["pull_id"]
            PL.fetch(lambda name, offset: s.call("pull_chunk", dict(pull_id=m["pull_id"], name=name, offset=offset)),
                     dict(m, files=files), out, echo=lambda line: print(line, file=sys.stderr, flush=True))
    except RM.RemoteError as e:
        print(f"data pull {sport}: {e}", file=sys.stderr)
        return 2
    try:
        counts = PL.load(get_engine(db), out, first=first, force=args.force)
    except RuntimeError as e:
        print(f"data pull {sport}: {e}", file=sys.stderr)
        return 2
    PL.record(db, sport, m["as_of"])
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    for name, n in counts.items():
        print(f"{name:<24} {n:>12,}")
    print(f"pulled {sport} as of {m['as_of']} into the local database and data folder"
          f"{' (a fresh-database load)' if first else ''}; next time: racinglines data pull {sport} --since last",
          file=sys.stderr)
    return 0
