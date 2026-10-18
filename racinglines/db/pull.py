"""
`racinglines data pull SPORT`: one sport's shared data from the hosted server into a local database and data folder
(owner, 2026-10-11 18:13Z, Core C51; the data-terms review cleared raw exchange history for it on 2026-10-14). The
server side builds Parquet files on request; the client side fetches them in chunks over the hosted MCP sign-in
(cli/remote.py) and loads them.

What a pull holds (`files()`):
    registry   sports, leagues, competitions, categories, seasons, points_schemes, venues, venue_aliases, athletes,
               athlete_identifiers: every row (small, and every sport's rows share their ids)
    race data  the sport's events, races, rounds, results, splits, laps, track_profiles, source_files, market_links
    model runs the sport's model_runs (not candidates: those are saved by an account), race_predictions,
               standings_predictions
    markets    exchange prices, trades and books of the sport's linked markets, one file per exchange, store and month,
               from the Parquet archive and Postgres both (markets/store.read)
Never: accounts, profiles, positions, signals, orders, fantasy, jobs, bets, the activity log.

`since` (a UTC date) narrows a top-up: race data of events that start from 14 days before it, model runs made from
it, market rows from it. The registry and market_links always come whole (they change in place).

The local database: the first pull into one is a fresh-database load (snapshot.import_'s rule: it refuses a database
that already holds model runs, unless force), which empties the pulled tables first so the server's ids are kept. Every
later pull upserts by primary key. data/pulls/state.json records, per database, which sports were pulled and when, so
`--since last` knows where the last one stopped.
"""

import base64
import hashlib
import json
import shutil
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import MetaData, text

from racinglines import paths
from racinglines.db import snapshot as SNAP

DIR = paths.DATA / "pulls"
SERVER_DIR = DIR / "server"                # the server's built files, one folder per pull id
STATE = DIR / "state.json"                 # the client's record of its pulls
CHUNK = 4 * 1024 * 1024                    # raw bytes per pull_chunk call (base64 adds a third)
KEEP_HOURS = 24                            # a server pull folder older than this is removed on the next manifest
LEAD_DAYS = 14                             # `since` reaches back this far for the events it re-sends

REGISTRY = ["sports", "leagues", "competitions", "categories", "seasons", "points_schemes", "venues", "venue_aliases",
            "athletes", "athlete_identifiers"]
RACE = ["events", "races", "rounds", "results", "splits", "laps", "track_profiles", "source_files", "market_links"]
RUNS = ["model_runs", "race_predictions", "standings_predictions"]
TABLES = REGISTRY + RACE + RUNS            # foreign-key order, parents first
MARKETS = ("prices", "trades", "books")
assert set(SNAP.TABLES) <= set(TABLES)

# the sport's rows of each scoped table: :c is the sport's competition ids, :since a date or NULL
_EVENTS = """SELECT e.id FROM events e JOIN seasons s ON s.id = e.season_id
             WHERE s.competition_id = ANY(:c) AND (CAST(:since AS date) IS NULL OR e.start_date >= CAST(:since AS date))"""
_RACES = f"SELECT id FROM races WHERE event_id IN ({_EVENTS})"
_RESULTS = f"SELECT r.id FROM results r JOIN rounds ro ON ro.id = r.round_id WHERE ro.race_id IN ({_RACES})"
_RUNS = """SELECT id FROM model_runs WHERE competition_id = ANY(:c) AND kind <> 'candidate'
           AND (CAST(:since_ts AS timestamptz) IS NULL OR created_at >= CAST(:since_ts AS timestamptz))"""
SCOPE = {
    "events": f"id IN ({_EVENTS})",
    "races": f"id IN ({_RACES})",
    "rounds": f"race_id IN ({_RACES})",
    "results": f"id IN ({_RESULTS})",
    "splits": f"result_id IN ({_RESULTS})",
    "laps": f"result_id IN ({_RESULTS})",
    "track_profiles": f"event_id IN ({_EVENTS})",
    "source_files": f"race_id IN ({_RACES})",
    "market_links": "competition_id = ANY(:c)",
    "model_runs": f"id IN ({_RUNS})",
    "race_predictions": f"model_run_id IN ({_RUNS})",
    "standings_predictions": f"model_run_id IN ({_RUNS})",
}


def competition_ids(conn, sport):
    from racinglines import sports
    code = sports.load(sport)["competition"]["code"]
    return [r[0] for r in conn.execute(text("SELECT id FROM competitions WHERE code = :c"), dict(c=code))]


def _params(conn, sport, since):
    s = None if since is None else pd.Timestamp(since).date()
    return dict(c=competition_ids(conn, sport), since=None if s is None else s - timedelta(days=LEAD_DAYS),
                since_ts=None if s is None else datetime(s.year, s.month, s.day, tzinfo=timezone.utc))


# --- server: the manifest, the files, the chunks ------------------------------------------------------------------

def _months(start, end):
    m, out = pd.Timestamp(start).tz_localize(None).to_period("M") if pd.Timestamp(start).tzinfo else pd.Timestamp(start).to_period("M"), []
    while m <= pd.Timestamp(pd.Timestamp(end).strftime("%Y-%m-%d")).to_period("M"):
        out.append(str(m))
        m += 1
    return out


def _archive_months(exchange, name):
    from racinglines.markets import store as MS
    d = MS.root_for(exchange) / name
    return sorted(p.name.split("=", 1)[1] for p in d.glob("month=*") if p.is_dir()) if d.exists() else []


def _exchange_tokens(conn, sport):
    """{exchange: [token_id]} of the sport's market links."""
    rows = conn.execute(text("SELECT coalesce(exchange, 'polymarket'), token_id FROM market_links WHERE competition_id = ANY(:c)"),
                        dict(c=competition_ids(conn, sport))).fetchall()
    out = {}
    for ex, tok in rows:
        out.setdefault(ex, []).append(tok)
    return out


def files(conn, sport, since=None, now=None):
    """[(name, kind, spec)]: what a pull of `sport` holds. kind "table" (spec: the table) or "market" (spec:
    (exchange, store, month))."""
    out = [(f"tables/{t}.parquet", "table", t) for t in TABLES]
    now = now or datetime.now(timezone.utc)
    for ex, toks in sorted(_exchange_tokens(conn, sport).items()):
        if not toks:
            continue
        for name in MARKETS:
            months = set(_archive_months(ex, name))
            months |= set(_months(now - timedelta(days=62), now))        # Postgres keeps the recent rows
            if since is not None:
                months = {m for m in months if m >= pd.Timestamp(since).strftime("%Y-%m")}
            out += [(f"markets/{ex}/{name}/month={m}.parquet", "market", (ex, name, m)) for m in sorted(months)]
    return out


def pull_id(sport, since, now=None):
    now = now or datetime.now(timezone.utc)
    key = f"{sport}|{since or ''}|{now:%Y%m%dT%H}"                     # one build per sport, since and hour
    return f"{sport}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _cleanup(now=None):
    now = now or time.time()
    if not SERVER_DIR.exists():
        return
    for d in SERVER_DIR.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > KEEP_HOURS * 3600:
            shutil.rmtree(d, ignore_errors=True)


def manifest(conn, sport, since=None, now=None):
    """The pull's id and its files (not built yet: pull_chunk builds each one the first time it is asked for)."""
    now = now or datetime.now(timezone.utc)
    _cleanup()
    pid = pull_id(sport, since, now)
    d = SERVER_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    m = dict(pull_id=pid, sport=sport, since=None if since is None else str(pd.Timestamp(since).date()),
             as_of=now.isoformat(timespec="seconds"), chunk=CHUNK,
             files=[dict(name=n, kind=k) for n, k, _ in files(conn, sport, since, now)])
    (d / "manifest.json").write_text(json.dumps(m))
    return m


def _table_frame(conn, table, sport, since):
    if table in SCOPE:
        df = pd.read_sql(text(f"SELECT * FROM {table} WHERE {SCOPE[table]} ORDER BY 1"), conn,
                         params=_params(conn, sport, since))
    else:
        df = pd.read_sql(text(f"SELECT * FROM {table} ORDER BY 1"), conn)
    for col in SNAP._json_cols(conn, table):           # JSON as text: Parquet can't hold free-form JSON
        df[col] = df[col].map(lambda v: None if v is None else json.dumps(v))
    return df


def _market_frame(conn, sport, exchange, name, month, since):
    from racinglines.markets import store as MS
    toks = _exchange_tokens(conn, sport).get(exchange, [])
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1) - pd.Timedelta(microseconds=1)
    if since is not None:
        start = max(start, pd.Timestamp(since, tz="UTC"))
    df = MS.read(conn, name, tokens=toks, start=start, end=end, root=MS.root_for(exchange))
    return df.reindex(columns=MS.STORES[name]["cols"])


def build(conn, pid, name):
    """The file `name` of pull `pid`, built on first use. -> its path."""
    d = SERVER_DIR / pid
    mf = d / "manifest.json"
    if not mf.exists():
        raise ValueError(f"unknown or expired pull {pid!r}: ask for a new manifest")
    m = json.loads(mf.read_text())
    if name not in {f["name"] for f in m["files"]}:
        raise ValueError(f"{name!r} is not a file of pull {pid}")
    out = d / name
    if out.exists():
        return out
    spec = next(s for n, _, s in files(conn, m["sport"], m["since"], pd.Timestamp(m["as_of"]).to_pydatetime())
                if n == name)
    df = (_table_frame(conn, spec, m["sport"], m["since"]) if isinstance(spec, str)
          else _market_frame(conn, m["sport"], *spec, m["since"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out)
    return out


def chunk(conn, pid, name, offset=0):
    """One chunk of a pull file: dict(data=base64, offset, size, sha256 of the whole file, rows when offset is 0)."""
    p = build(conn, pid, name)
    size = p.stat().st_size
    offset = max(0, int(offset))
    with p.open("rb") as f:
        f.seek(offset)
        raw = f.read(CHUNK)
    out = dict(name=name, offset=offset, size=size, data=base64.b64encode(raw).decode(), done=offset + len(raw) >= size)
    if offset == 0:
        out["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    if out["done"]:                 # served: the VM's disk holds one built file at a time (a retry builds it again)
        p.unlink(missing_ok=True)
    return out


# --- client: fetch and load ---------------------------------------------------------------------------------------

def _state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _db_key(url):
    from sqlalchemy.engine import make_url
    u = make_url(url)
    return f"{u.host or ''}:{u.port or ''}/{u.database}"


def last_pull(url, sport):
    return _state().get(_db_key(url), {}).get(sport)


def record(url, sport, as_of):
    s = _state()
    s.setdefault(_db_key(url), {})[sport] = as_of
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def first_into(url):
    """True when no pull has loaded this database yet (then it is a fresh-database load)."""
    return not _state().get(_db_key(url))


def fetch(call, m, out, echo=print):
    """Download every file of manifest `m` into `out` with `call(name, offset) -> chunk dict`. -> [path]."""
    from racinglines import progress as PG
    got = []
    for i, f in enumerate(m["files"], 1):
        p = out / f["name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        PG.update(item=f"file {i} of {len(m['files'])}: {f['name']}")
        offset, sha = 0, None
        with p.open("wb") as fh:
            while True:
                c = call(f["name"], offset)
                fh.write(base64.b64decode(c["data"]))
                sha = c.get("sha256", sha)
                offset = c["offset"] + len(base64.b64decode(c["data"]))
                if c["done"]:
                    break
        if sha and hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise IOError(f"{f['name']}: checksum mismatch after download")
        echo(f"  {f['name']}: {p.stat().st_size:,} bytes")
        got.append(p)
    return got


PG_MAX_PARAMS = 65535                      # one statement's bound parameters (psycopg / Postgres protocol limit)


def rows_per_statement(ncols, chunk_rows=5000):
    """Rows per INSERT: `chunk_rows`, cut so rows x columns stays under the parameter limit (laps has 19 columns)."""
    return max(1, min(chunk_rows, (PG_MAX_PARAMS - 1000) // max(1, ncols)))


def load(engine, folder, first=False, force=False, chunk_rows=5000):
    """Load a downloaded pull: the tables (first: emptied first, keeping the server's ids; else upserted by primary
    key) and the market files (appended to this data folder's archive; readers drop duplicates). -> {name: rows}."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from racinglines.markets import store as MS
    counts = {}
    with engine.begin() as c:
        if first and not force and c.execute(text("SELECT count(*) FROM model_runs")).scalar():
            raise RuntimeError("this database has model runs and no earlier pull: the first pull is for a fresh database "
                               "(`racinglines db init` on an empty one), or --force to replace its race data")
        meta = MetaData()
        meta.reflect(bind=c, only=TABLES)
        if first:
            c.execute(text("TRUNCATE " + ", ".join(TABLES) + " RESTART IDENTITY CASCADE"))
        for t in TABLES:
            f = folder / "tables" / f"{t}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            for col in SNAP._json_cols(c, t):
                df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else json.loads(v))
            df = df.astype(object).where(df.notna(), None)
            rows = df.to_dict("records")
            tbl = meta.tables[t]
            pk = [col.name for col in tbl.primary_key.columns]
            step = rows_per_statement(len(tbl.columns), chunk_rows)
            for i in range(0, len(rows), step):
                stmt = pg_insert(tbl).values(rows[i:i + step])
                upd = {col.name: stmt.excluded[col.name] for col in tbl.columns if col.name not in pk}
                c.execute(stmt.on_conflict_do_update(index_elements=pk, set_=upd) if upd else stmt.on_conflict_do_nothing())
            if "id" in df.columns and len(df):
                c.execute(text(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), (SELECT max(id) FROM {t}))"))
            counts[t] = len(rows)
    for p in sorted((folder / "markets").rglob("*.parquet")) if (folder / "markets").exists() else []:
        ex, name = p.parts[-3], p.parts[-2]
        df = pd.read_parquet(p)
        n = MS._write(name, df, root=MS.root_for(ex))
        counts[f"{ex}/{name}"] = counts.get(f"{ex}/{name}", 0) + n
    return counts


def since_arg(value, url, sport):
    """`--since`: None (everything), "last" (where this database's last pull of the sport stopped) or a date."""
    if value in (None, ""):
        return None
    if value == "last":
        last = last_pull(url, sport)
        if last is None:
            return None
        return str(pd.Timestamp(last).date())
    return str(pd.Timestamp(value).date())
