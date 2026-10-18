"""
racinglines db <command>

    init      Create or upgrade the schema to the latest migration (Alembic), then seed reference data.
    seed      Upsert sports / leagues / competitions / categories / venues from registry.py.
    stats     Row counts and coverage per season and category.
    export    Write the tidy frame (same columns as `racinglines data parse mtb_dh`'s CSV) for a competition to CSV.
    snapshot-export  The tables the models read, with ids, to data/archive/db/ (Parquet).
    snapshot-import  Load that snapshot into a fresh database: an exact replica (same ids, same prices).
    merge-athletes   Merge two athletes that are the same person (e.g. a name change found by UCI ID).
    changes   The data change log: ingests that changed anything, merges, notes (--add "why" for a note).
    storage   Where the data lives: VM disk, data folders, database sizes, the data bucket's prefixes (the admin
              Storage page's snapshot; --refresh takes a new one, which lists the bucket and can take minutes).

Connection: $DATABASE_URL, or --db URL (default: docker-compose.yml's database).
"""

import argparse

from sqlalchemy import text  # noqa: E402

from racinglines.db.config import database_url, get_engine, get_session  # noqa: E402


def cmd_init(args):
    from alembic import command
    from alembic.config import Config

    from racinglines.paths import ROOT              # the checkout, or the wheel's bundled migrations (setup.py)
    ini = ROOT / "alembic.ini"                      # a checkout's (its logging); a wheel has none and needs none
    cfg = Config(str(ini)) if ini.exists() else Config()
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", database_url(args.db))
    command.upgrade(cfg, "head")
    cmd_seed(args)


def cmd_seed(args):
    from racinglines.db.ingest import seed

    with get_session(args.db) as s:
        seed(s)
        s.commit()
    print("Reference data seeded.")


STATS_SQL = """
SELECT co.code AS competition, s.year, c.code AS category, count(DISTINCT e.id) AS events,
       count(DISTINCT ro.id) AS rounds, count(r.id) AS results
FROM results r JOIN rounds ro ON ro.id = r.round_id JOIN races ra ON ra.id = ro.race_id
JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
JOIN competitions co ON co.id = s.competition_id JOIN categories c ON c.id = ra.category_id
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
"""


def cmd_stats(args):
    import pandas as pd

    from racinglines.cli import tty
    eng = get_engine(args.db)
    tables = ["sports", "leagues", "competitions", "categories", "venues", "events", "races", "rounds",
              "athletes", "athlete_identifiers", "results", "splits", "source_files", "model_runs",
              "race_predictions", "standings_predictions"]
    with eng.connect() as conn:
        counts = {t: conn.execute(text(f'SELECT count(*) FROM {t}')).scalar() for t in tables}
        cov = pd.read_sql(text(STATS_SQL), conn)
    con = tty.console()
    if con is None:
        for t in tables:
            print(f"{t:<22} {counts[t]:>8}")
        print()
        print(cov.to_string(index=False))
        return
    _stats_rich(con, eng, counts, cov)


STATS_GROUPS = (("📅 calendar", ("sports", "leagues", "competitions", "categories", "venues", "events", "races", "rounds")),
                ("🏁 results", ("athletes", "athlete_identifiers", "results", "splits", "source_files")),
                ("🔮 model", ("model_runs", "race_predictions", "standings_predictions")))


def _stats_rich(con, eng, counts, cov):
    """db stats as a panel: the row counts grouped, then the coverage per competition and season."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    groups = []
    for title, names in STATS_GROUPS:
        t = Table(box=None, show_header=False, title=f"[bold]{title}[/]", title_justify="left", pad_edge=False)
        t.add_column(style="dim")
        t.add_column(justify="right", style="bold cyan")
        for n in names:
            t.add_row(n.replace("_", " "), f"{counts[n]:,}" if counts[n] else "[dim]0[/]")
        groups.append(t)
    db = eng.url.database
    grid = Table.grid(padding=(0, 4))
    grid.add_row(*groups)
    con.print(Panel.fit(grid, title=f"📦 [bold]database {db}[/]", title_align="left",
                        border_style="blue", box=box.ROUNDED))
    t = Table(box=box.SIMPLE_HEAD, title="[bold]coverage[/]", title_justify="left", header_style="bold")
    for c in cov.columns:
        t.add_column(c.replace("_", " "), justify="right" if cov[c].dtype.kind in "iuf" else "left")
    for r in cov.to_dict("records"):
        t.add_row(*[f"{int(v):,}" if c not in ("year", "season") and cov[c].dtype.kind in "iu" else str(v)
                    for c, v in r.items()])
    con.print(t)


def cmd_export(args):
    from racinglines.db.queries import load_tidy

    df = load_tidy(get_engine(args.db), competition=args.competition)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


def cmd_snapshot_export(args):
    from racinglines import paths
    from racinglines.db import snapshot as S
    counts = S.export(get_engine(args.db))
    print(f"exported {sum(counts.values()):,} rows in {len(counts)} tables -> {paths.rel(S.DIR)}")


def cmd_snapshot_import(args):
    from racinglines.db import snapshot as S
    counts = S.import_(get_engine(args.db), force=args.force)
    print(f"imported {sum(counts.values()):,} rows in {len(counts)} tables")


def cmd_merge_athletes(args):
    from racinglines.db.ingest import merge_athletes
    with get_session(args.db) as s:
        moved = merge_athletes(s, args.keep, args.drop, dry_run=args.dry_run)
        if not args.dry_run:
            from racinglines.db import changes
            changes.record(s, "merge-athletes", f"athlete {args.drop} merged into {args.keep}", sport="mtb_dh",
                           detail=dict(keep=args.keep, drop=args.drop, moved=moved))
            s.commit()
    rows = ", ".join(f"{t} {n}" for t, n in moved.items()) or "nothing"
    print(f"{'Would move' if args.dry_run else 'Moved'} athlete {args.drop} into {args.keep}: {rows}")


def cmd_changes(args):
    from racinglines.db import changes
    with get_session(args.db) as s:
        if args.add:
            changes.record(s, "note", args.add, sport=args.sport)
            s.commit()
        for c in changes.recent(s, args.limit, args.sport):
            print(f"{c.at:%Y-%m-%d %H:%M}  {c.sport or '-':7} {c.kind:15} {c.summary}  ({c.by or '?'})")


def cmd_storage(args):
    import json

    from racinglines import storage_status as ST
    if args.refresh:
        with get_engine(args.db).connect() as conn:
            snap = ST.refresh(conn)
    else:
        snap = ST.read()
    if args.json:
        print(json.dumps(snap, indent=1))
        return
    print("\n".join(ST.lines(snap)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="racinglines db", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="Database URL (default: $DATABASE_URL or the docker-compose database).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="Migrate schema to latest and seed reference data.").set_defaults(func=cmd_init)
    sub.add_parser("seed", help="Upsert reference data from registry.py.").set_defaults(func=cmd_seed)
    sub.add_parser("stats", help="Row counts and coverage.").set_defaults(func=cmd_stats)
    p = sub.add_parser("export", help="Write the tidy frame for a competition to CSV.")
    p.add_argument("--competition", default="uci_dhi_wc")
    p.add_argument("--out", default="splits.csv")
    p.set_defaults(func=cmd_export)
    sub.add_parser("snapshot-export", help="Model tables with ids -> data/archive/db/.").set_defaults(func=cmd_snapshot_export)
    p = sub.add_parser("snapshot-import", help="Load data/archive/db/ into a fresh database.")
    p.add_argument("--force", action="store_true", help="Even if the database already holds model runs.")
    p.set_defaults(func=cmd_snapshot_import)
    p = sub.add_parser("merge-athletes", help="Merge athlete DROP into KEEP (same person), then delete DROP.")
    p.add_argument("keep", type=int)
    p.add_argument("drop", type=int)
    p.add_argument("--dry-run", action="store_true", help="Only report what would move.")
    p.set_defaults(func=cmd_merge_athletes)
    p = sub.add_parser("changes", help="The data change log, newest first; --add \"why\" adds a note.")
    p.add_argument("--add", metavar="TEXT", help="Record a note (e.g. why the data was re-downloaded).")
    p.add_argument("--sport", help="Only this sport (mtb_dh, f1); with --add, the note's sport.")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_changes)
    p = sub.add_parser("storage", help="VM disk, data folders, database sizes and the data bucket (last snapshot).")
    p.add_argument("--refresh", action="store_true", help="Take a new snapshot first (lists the bucket: minutes).")
    p.add_argument("--json", action="store_true", help="The snapshot as JSON.")
    p.set_defaults(func=cmd_storage)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
