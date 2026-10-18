"""
Polymarket market links as a file, so a machine that can't reach Polymarket's API (e.g. a cloud session
whose network blocks it) gets exactly the links `racinglines data pm-sync f1` built here.

market_links rows refer to database ids (race, athlete, competition, category, and an h2h opponent in
params), which differ between databases. The export swaps each for a stable key, and the import swaps
them back in the target database:

    race        -> the event's source_key + the race's category code   ("2026-15", "DRV")
    athlete     -> the FastF1 (Ergast) driverId, athlete_identifiers scheme "f1"
    competition -> competitions.code;  category -> categories.code
    params.opponent_id -> params.opponent_key (same driverId)

    racinglines data pm-links-export f1      # -> data/archive/markets/polymarket/links/market_links.parquet
    racinglines data pm-links-import f1      # replaces each token's row; reports rows whose keys don't resolve
    racinglines data pm-links-export f1 --exchange kalshi   # the same for Kalshi's links -> .../kalshi/links/
    racinglines data pm-links-import f1 --exchange kalshi
"""

import json

import pandas as pd
from sqlalchemy import text

from racinglines import paths

PATH = paths.archive_markets("polymarket") / "links" / "market_links.parquet"


def path_for(exchange="polymarket"):
    return paths.archive_markets(exchange) / "links" / "market_links.parquet"
SKIP = ("id", "created_at", "race_id", "athlete_id", "competition_id", "category_id")


def export(engine, path=PATH, exchange="polymarket"):
    with engine.connect() as c:
        df = pd.read_sql(text("""
            SELECT ml.*, e.source_key AS race_key, rc.code AS race_category, ai.value AS athlete_key,
                   co.code AS competition_code, ca.code AS category_code
            FROM market_links ml
            LEFT JOIN races r ON r.id = ml.race_id LEFT JOIN events e ON e.id = r.event_id
            LEFT JOIN categories rc ON rc.id = r.category_id
            LEFT JOIN athlete_identifiers ai ON ai.athlete_id = ml.athlete_id AND ai.scheme = 'f1'
            LEFT JOIN competitions co ON co.id = ml.competition_id
            LEFT JOIN categories ca ON ca.id = ml.category_id
            WHERE ml.exchange = :x"""), c, params=dict(x=exchange))
        opp = dict(c.execute(text("SELECT athlete_id, value FROM athlete_identifiers WHERE scheme = 'f1'")).all())
    missing = df[(df["race_id"].notna() & df["race_key"].isna()) | (df["athlete_id"].notna() & df["athlete_key"].isna())]
    if len(missing):
        raise ValueError(f"{len(missing)} links refer to a race or athlete without a stable key")

    def swap(p):
        p = dict(p or {})
        if p.get("opponent_id") is not None:
            p["opponent_key"] = opp.get(int(p.pop("opponent_id")))
        return json.dumps(p) if p else None
    df["params"] = df["params"].map(swap)
    df = df.drop(columns=[c for c in SKIP if c in df.columns])
    for col in df.columns:                          # keep types Parquet-friendly
        if df[col].dtype == object and col not in ("params",):
            df[col] = df[col].astype("string")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    return len(df)


def _value(v):
    """None for NaN, NA and NaT. psycopg writes a NaT as a year-48113 timestamp, which then fails every read."""
    return None if v is pd.NA or v is pd.NaT or (isinstance(v, float) and pd.isna(v)) else v


def import_(engine, path=PATH):
    """Upsert the file's links into this database (by token_id). Returns dict(rows, unresolved)."""
    df = pd.read_parquet(path)
    with engine.begin() as c:
        races = {(k, cat): rid for rid, k, cat in c.execute(text("""
            SELECT r.id, e.source_key, ca.code FROM races r JOIN events e ON e.id = r.event_id
            JOIN categories ca ON ca.id = r.category_id""")).all()}
        drivers = dict(c.execute(text("SELECT value, athlete_id FROM athlete_identifiers WHERE scheme = 'f1'")).all())
        comps = dict(c.execute(text("SELECT code, id FROM competitions")).all())
        cats = {(co, ca): i for i, co, ca in c.execute(text("""
            SELECT ca.id, co.code, ca.code FROM categories ca JOIN competitions co ON co.id = ca.competition_id""")).all()}
        cols = [r[0] for r in c.execute(text("""SELECT column_name FROM information_schema.columns
                                                WHERE table_name = 'market_links'"""))]
        unresolved = 0
        for rec in df.to_dict("records"):
            rec = {k: _value(v) for k, v in rec.items()}
            race = races.get((rec.pop("race_key"), rec.pop("race_category"))) if rec.get("race_key") else None
            rec.pop("race_key", None), rec.pop("race_category", None)
            ak = rec.pop("athlete_key", None)
            athlete = drivers.get(ak) if ak else None
            comp = comps.get(rec.pop("competition_code", None))
            cat = cats.get((next((k for k, v in comps.items() if v == comp), None), rec.pop("category_code", None)))
            p = json.loads(rec["params"]) if rec.get("params") else {}
            if "opponent_key" in p:
                p["opponent_id"] = drivers.get(p.pop("opponent_key"))
            if (ak and athlete is None) or (p.get("opponent_id", 0) is None):
                unresolved += 1
            row = {k: v for k, v in rec.items() if k in cols}
            row.update(race_id=race, athlete_id=athlete, competition_id=comp, category_id=cat,
                       params=json.dumps(p) if p else None)
            names = list(row)
            c.execute(text("DELETE FROM market_links WHERE token_id = :t"), dict(t=row["token_id"]))   # upsert
            c.execute(text(f"""INSERT INTO market_links ({", ".join(names)})
                               VALUES ({", ".join(f"CAST(:{n} AS jsonb)" if n == "params" else f":{n}" for n in names)})"""),
                      row)
    return dict(rows=len(df), unresolved=unresolved)
