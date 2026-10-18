"""
Database snapshot: the tables the models and sweeps read, as Parquet files with their ids intact, so a
fresh database (e.g. a Claude cloud session's) becomes an exact replica of this one. Rebuilding from the
raw files instead gives the same data under different ids, and the pricing code orders drivers by id
in places, so seeded simulations would then differ at the level of simulation noise.

    racinglines db snapshot-export     # -> data/archive/db/<table>.parquet
    racinglines db snapshot-import     # fresh database only: replaces these tables, resets id sequences

Market prices and trades aren't here: they live in the Parquet market archive (markets/store.py).
Web-app tables (users, books, bets, jobs) and model runs are never exported.
"""

import json

import pandas as pd
from sqlalchemy import MetaData, text

from racinglines import paths

DIR = paths.DATA / "archive" / "db"
# foreign-key order (parents first)
TABLES = ["sports", "leagues", "competitions", "categories", "seasons", "points_schemes", "venues", "venue_aliases",
          "athletes", "athlete_identifiers", "events", "races", "rounds", "results", "splits", "laps",
          "track_profiles", "source_files", "market_links"]


def _json_cols(conn, table):
    return [r[0] for r in conn.execute(text("""SELECT column_name FROM information_schema.columns
                                               WHERE table_name = :t AND data_type IN ('json', 'jsonb')"""), dict(t=table))]


def export(engine, out=DIR):
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with engine.connect() as c:
        for t in TABLES:
            df = pd.read_sql(text(f"SELECT * FROM {t} ORDER BY 1"), c)
            for col in _json_cols(c, t):                   # JSON as text: Parquet can't hold free-form JSON
                df[col] = df[col].map(lambda v: None if v is None else json.dumps(v))
            df.to_parquet(out / f"{t}.parquet", index=False, compression="zstd")
            counts[t] = len(df)
    (out / "manifest.json").write_text(json.dumps(dict(tables=TABLES, rows=counts), indent=1))
    return counts


def import_(engine, src=DIR, force=False, chunk=5000):
    """Load the snapshot into `engine`'s database (schema already at head, e.g. `racinglines db init`).
    Refuses a database that already holds model runs unless force: this replaces the tables."""
    with engine.begin() as c:
        if not force and c.execute(text("SELECT count(*) FROM model_runs")).scalar():
            raise RuntimeError("this database has model runs: snapshot-import is for a fresh database (--force to override)")
        meta = MetaData()
        meta.reflect(bind=c, only=TABLES)
        c.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
        counts = {}
        for t in TABLES:
            f = src / f"{t}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            for col in _json_cols(c, t):
                df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else json.loads(v))
            df = df.astype(object).where(df.notna(), None)
            rows = df.to_dict("records")
            for i in range(0, len(rows), chunk):
                c.execute(meta.tables[t].insert(), rows[i:i + chunk])
            if "id" in df.columns and len(df):
                c.execute(text(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), (SELECT max(id) FROM {t}))"))
            counts[t] = len(rows)
    return counts
