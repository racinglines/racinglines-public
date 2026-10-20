"""
racinglines check: quick validation for new setups (about a minute, no data downloaded).

    code       the pipelines on small synthetic data (no network, no database)
    endpoints  one tiny request to each data source (FastF1, ChronoRace, Polymarket)
    database   Postgres reachable and migrated (skipped with --no-db)

Options: --sport f1|mtb_dh (default: both, where mtb_dh is installed) --offline (skip endpoints) --no-db
The full regression suite on real data: docs/testing.md.
"""

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="racinglines check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from racinglines.cli import present
    offered = ["f1"] + (["mtb_dh"] if present("cli.mtb_dh") else [])     # offered when installed
    ap.add_argument("--sport", choices=offered, action="append", help="Default: every one offered.")
    ap.add_argument("--offline", action="store_true", help="Skip the data-endpoint checks.")
    ap.add_argument("--no-db", action="store_true", help="Skip the database check.")
    args = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings("ignore")
    from racinglines.testing import checks as C
    sports = tuple(args.sport or offered)
    results = []
    for title, fn in [("code (synthetic data)", lambda: C.run_code(sports))] + \
            ([] if args.offline else [("data endpoints", lambda: C.run_endpoints(sports))]) + \
            ([] if args.no_db else [("database", C.run_database)]):
        print(f"\n{title}")
        for r in fn():
            results.append(r)
            mark = "✓" if r.ok else "✗"
            print(f"  {mark} {r.group:<22} {r.name:<38} {r.seconds:5.1f}s  {r.detail}", flush=True)
    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (" — see ✗ above" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
