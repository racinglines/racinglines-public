"""
Staging as a live copy of production (docs/vm-deploy.md, "Staging: the data mirror").

Why: staging's database was a copy of production made once (deploy/vm/staging/setup.sh). Staging runs no recorders,
signals or pricing timers, so it fell behind from the minute it was copied, and the only way to catch up
(`vm.sh staging reset`) dropped the whole database, staging's own accounts and test rows with it. The mirror fixes it
at the source with Postgres logical replication:

    production `racinglines`  --publication-->  staging `racinglines_staging`

Production's rows in the mirrored tables arrive in staging within seconds, for as long as the mirror is on: no
refresh, no copy at deploy. Production is only read: the subscription connects as `racinglines_mirror`, a role with
REPLICATION and SELECT on the mirrored tables and nothing else.

What is mirrored (the scope, chosen by `on --scope`):

    data       race history, athletes, model runs and predictions, market links, the market tape, live events, the
               data change log (DATA)
    accounts   data + users and the fantasy ledger (ACCOUNTS)
    all        every table in `public` and `accounts` that production has, except STAGING_OWN

STAGING_OWN is never mirrored, whatever the scope: the job queue (staging's worker must not run production's jobs)
and alembic_version (staging is a migration ahead). A table only staging has (a migration production hasn't run yet)
stays staging's until production has it too and `on` runs again.

Staging can still write: its own rows in a mirrored table (a test account, a Lab job's model run, a reference row a
staging deploy seeded first) take ids from OFFSET up, so they never collide with production's ids. If production later
gets the same unique key (say staging seeded a new competition first), replication would stop on the duplicate;
`repair` (racinglines-staging-mirror-check.timer, every 5 minutes) makes staging's row production's, points staging's
references at it, and replication carries on. Staging's edits to production's rows last until production changes
that row. Only inserts, updates and deletes are published, never TRUNCATE.

    python -m racinglines.db.mirror plan [--scope S]   # read-only: per table, what `on` would replace
    python -m racinglines.db.mirror on [--scope S]     # production: role, publication, slot; staging: subscription
    python -m racinglines.db.mirror status             # lag, errors, per-table state (exit 1: needs a look)
    python -m racinglines.db.mirror repair             # unblock a stalled subscription (staging rows only)
    python -m racinglines.db.mirror off                # drop it all; staging keeps its rows as of then

Run on the VM with staging's env file (DATABASE_URL = racinglines_staging, RACINGLINES_ENV=staging), through
`bash scripts/deploy/vm.sh mirror plan|on|status|repair|off`. Production needs wal_level=logical first
(`vm.sh postgres logical`).
"""

import argparse
import os
import secrets
import sys
import time

# Production's race and market data, parents first.
DATA = (
    "public.sports", "public.leagues", "public.competitions", "public.categories", "public.seasons", "public.venues",
    "public.venue_aliases", "public.points_schemes", "public.events", "public.races", "public.rounds",
    "public.track_profiles", "public.athletes", "public.athlete_identifiers", "public.results", "public.splits",
    "public.laps", "public.source_files", "public.model_runs", "public.race_predictions",
    "public.standings_predictions", "public.market_links", "public.market_price_history", "public.market_trades",
    "public.market_book_snapshots", "public.market_disagreements", "public.live_events", "public.data_changes",
    "public.fastf1_session_snapshots", "public.fastf1_driver_positions",
)
ACCOUNTS = ("public.users", "public.user_email_prefs", "accounts.fantasy_ledger")
STAGING_OWN = ("public.jobs", "public.alembic_version")
SCHEMAS = ("public", "accounts")
SCOPES = ("data", "accounts", "all")
OFFSET = 1_000_000_000                 # staging's own rows in a mirrored table start here (int4 max is 2,147,483,647)
NAME = "racinglines_staging_mirror"    # the publication, the subscription and the replication slot
ROLE = "racinglines_mirror"
PROD_DB = "racinglines"
STAGING_DB = "racinglines_staging"
# The subscription's worker runs inside the Postgres container, so it reaches production on the container's own port.
PUBLISHER = os.environ.get("MIRROR_PUBLISHER", "host=127.0.0.1 port=5432 dbname=racinglines")


def log(msg):
    print(f"[mirror {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def urls(env=os.environ):
    """(production, staging) libpq URLs from staging's env file; refuses anything but staging's database."""
    url = env.get("DATABASE_URL", "")
    if env.get("RACINGLINES_ENV") != "staging" or not url.endswith("/" + STAGING_DB):
        raise SystemExit(f"refusing: run with staging's env file (DATABASE_URL .../{STAGING_DB}, RACINGLINES_ENV=staging)")
    staging = url.replace("postgresql+psycopg://", "postgresql://", 1)
    return staging[: -len(STAGING_DB)] + PROD_DB, staging


def connect(url, read_only=False):
    import psycopg
    c = psycopg.connect(url, autocommit=True)
    if read_only:
        c.execute("SET default_transaction_read_only = on")
    return c


def q(c, query, args=None):
    return c.execute(query, args).fetchall()


def one(c, query, args=None):
    r = c.execute(query, args).fetchone()
    return r[0] if r else None


def split(name):
    schema, _, table = name.partition(".")
    return schema, table


def ident(name):
    """A schema-qualified table ("public.users") or a bare name, quoted."""
    from psycopg import sql
    return sql.Identifier(*name.split(".")) if "." in name else sql.Identifier(name)


def tables(c):
    return {f"{s}.{t}" for s, t in q(c, "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = ANY(%s)",
                                     (list(SCHEMAS),))}


def columns(c, name):
    s, t = split(name)
    return [r[0] for r in q(c, "SELECT column_name FROM information_schema.columns "
                               "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position", (s, t))]


def has_pk(c, name):
    s, t = split(name)
    return bool(one(c, "SELECT 1 FROM pg_index i JOIN pg_class t ON t.oid = i.indrelid "
                       "JOIN pg_namespace n ON n.oid = t.relnamespace "
                       "WHERE n.nspname = %s AND t.relname = %s AND i.indisprimary", (s, t)))


def wanted(scope, prod):
    """The tables a scope mirrors, parents first where the order is known."""
    if scope == "data":
        return list(DATA)
    if scope == "accounts":
        return list(ACCOUNTS) + list(DATA)
    rest = sorted(t for t in tables(prod) if t not in DATA + ACCOUNTS + STAGING_OWN)
    return list(ACCOUNTS) + list(DATA) + rest


def mirrorable(prod, stg, scope):
    """The scope's tables on both databases with a primary key on production (logical replication needs one for
    updates and deletes), and the ones left out with why."""
    p, s = tables(prod), tables(stg)
    ok, skipped = [], []
    for t in wanted(scope, prod):
        if t in STAGING_OWN:
            continue
        if t not in p:
            skipped.append((t, "not on production yet"))
        elif t not in s:
            skipped.append((t, "not on staging"))
        elif not has_pk(prod, t):
            skipped.append((t, "no primary key on production"))
        else:
            ok.append(t)
    return ok, skipped


def subscription(stg):
    """(oid, enabled) of the subscription in staging's database, or None."""
    r = q(stg, "SELECT s.oid, s.subenabled FROM pg_subscription s JOIN pg_database d ON d.oid = s.subdbid "
               "WHERE s.subname = %s AND d.datname = current_database()", (NAME,))
    return r[0] if r else None


def subscribed(stg):
    return {f"{n}.{c}" for n, c in q(stg, "SELECT ns.nspname, c.relname FROM pg_subscription_rel r "
                                          "JOIN pg_subscription s ON s.oid = r.srsubid JOIN pg_class c ON c.oid = r.srrelid "
                                          "JOIN pg_namespace ns ON ns.oid = c.relnamespace WHERE s.subname = %s", (NAME,))}


def sequences(stg, name):
    """The sequences behind a table's serial or identity columns."""
    s, t = split(name)
    return [r[0] for r in q(stg, "SELECT pg_get_serial_sequence(quote_ident(%s) || '.' || quote_ident(%s), a.attname) "
                                 "FROM pg_attribute a JOIN pg_class t ON t.oid = a.attrelid "
                                 "JOIN pg_namespace n ON n.oid = t.relnamespace "
                                 "WHERE n.nspname = %s AND t.relname = %s AND a.attnum > 0 AND NOT a.attisdropped",
                                 (s, t, s, t)) if r[0]]


def lift_sequences(stg, names):
    """Staging's own rows in a mirrored table take ids from OFFSET up (idempotent; never lowers a sequence)."""
    for t in names:
        for seq in sequences(stg, t):
            if one(stg, f"SELECT last_value FROM {seq}") < OFFSET:
                stg.execute("SELECT setval(%s, %s, false)", (seq, OFFSET))
                log(f"{seq}: staging's own ids start at {OFFSET:,} from now on")


def plan(prod, stg, scope):
    """Read-only: what `on` would do. Staging rows production doesn't have as-is (an id production lacks, or the same
    id with other content) are replaced when a table is first mirrored; they are counted here and kept in the backup."""
    ok, skipped = mirrorable(prod, stg, scope)
    wal = one(prod, "SHOW wal_level")
    sub = subscription(stg)
    done = subscribed(stg) if sub else set()
    log(f"scope {scope}; production wal_level={wal}"
        + ("" if wal == "logical" else " (needs: bash scripts/deploy/vm.sh postgres logical)"))
    log(f"subscription: {'on, ' + str(len(done)) + ' tables' if sub else 'none yet'}")
    print(f"{'table':34} {'production':>12} {'staging':>12} {'replaced*':>10}  action")
    for t in ok:
        np_, ns = one(prod, f"SELECT count(*) FROM {t}"), one(stg, f"SELECT count(*) FROM {t}")
        cols = [c for c in columns(stg, t) if c in columns(prod, t)]
        diff = ""
        if t not in done and "id" in cols and max(np_, ns) < 2_000_000:
            row = "md5(ROW(" + ", ".join(f'"{c}"' for c in cols) + ")::text)"
            pid = dict(q(prod, f"SELECT id, {row} FROM {t}"))
            diff = sum(1 for i, h in q(stg, f"SELECT id, {row} FROM {t} WHERE id < {OFFSET}") if pid.get(i) != h)
        action = "mirrored" if t in done else "replace with production's rows, then follow production"
        print(f"{t:34} {np_:>12,} {ns:>12,} {diff!s:>10}  {action}")
    for t, why in skipped:
        print(f"{t:34} skipped: {why}")
    extra = sorted(done - set(ok))
    if extra:
        print(f"no longer in scope (`on` stops mirroring them; staging keeps their rows): {', '.join(extra)}")
    print("* staging rows production doesn't have as-is; blank for tables over 2M rows or already mirrored")


def publish(prod, names, password):
    """Production's half: the read-only role, the publication (insert/update/delete only) and the slot.
    Catalog changes only; no production row is written."""
    from psycopg import sql
    if one(prod, "SHOW wal_level") != "logical":
        raise SystemExit("production's wal_level isn't logical: bash scripts/deploy/vm.sh postgres logical first")
    verb = "ALTER" if one(prod, "SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,)) else "CREATE"
    prod.execute(sql.SQL(verb + " ROLE {} WITH LOGIN REPLICATION NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}")
                 .format(ident(ROLE), sql.Literal(password)))
    prod.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(ident(ROLE)))
    for s in {split(t)[0] for t in names}:
        prod.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(ident(s), ident(ROLE)))
    tl = sql.SQL(", ").join(ident(t) for t in names)
    prod.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(tl, ident(ROLE)))
    if one(prod, "SELECT 1 FROM pg_publication WHERE pubname = %s", (NAME,)):
        prod.execute(sql.SQL("ALTER PUBLICATION {} SET TABLE {}").format(ident(NAME), tl))
    else:
        prod.execute(sql.SQL("CREATE PUBLICATION {} FOR TABLE {} WITH (publish = 'insert, update, delete')")
                     .format(ident(NAME), tl))
    # same cluster: the slot must exist before CREATE SUBSCRIPTION (which would otherwise wait on itself)
    if not one(prod, "SELECT 1 FROM pg_replication_slots WHERE slot_name = %s", (NAME,)):
        prod.execute("SELECT pg_create_logical_replication_slot(%s, 'pgoutput')", (NAME,))
    log(f"production: publication {NAME} ({len(names)} tables), role {ROLE} (read-only), slot {NAME}")


def clear(stg, name):
    """Delete staging's rows in a table about to be mirrored, without firing foreign keys (staging's rows that point
    at them keep their ids, which production's copy brings back)."""
    with stg.transaction():
        stg.execute("SET LOCAL session_replication_role = replica")
        n = stg.execute(f"DELETE FROM {name}").rowcount
    if n:
        log(f"{name}: cleared {n:,} staging rows (production's copy follows)")


def subscribe(stg, names, password):
    """Staging's half: clear each newly mirrored table, lift its sequences, then subscribe or refresh (the initial
    copy runs in the background; `status` shows each table reach 'ready')."""
    from psycopg import sql
    conninfo = f"{PUBLISHER} user={ROLE} password={password}"
    sub = subscription(stg)
    new = [t for t in names if t not in (subscribed(stg) if sub else set())]
    if sub:
        stg.execute(sql.SQL("ALTER SUBSCRIPTION {} CONNECTION {}").format(ident(NAME), sql.Literal(conninfo)))
    for t in reversed(new):     # children first
        clear(stg, t)
    lift_sequences(stg, names)
    if sub:
        stg.execute(sql.SQL("ALTER SUBSCRIPTION {} REFRESH PUBLICATION WITH (copy_data = true)").format(ident(NAME)))
        if not sub[1]:
            stg.execute(sql.SQL("ALTER SUBSCRIPTION {} ENABLE").format(ident(NAME)))
        log(f"staging: subscription refreshed ({len(new)} new tables copying)")
    else:
        stg.execute(sql.SQL("CREATE SUBSCRIPTION {} CONNECTION {} PUBLICATION {} "
                            "WITH (create_slot = false, slot_name = {}, copy_data = true)")
                    .format(ident(NAME), sql.Literal(conninfo), ident(NAME), sql.Literal(NAME)))
        log(f"staging: subscribed to {len(names)} tables; the initial copy runs in the background")


def turn_on(prod, stg, scope):
    ok, skipped = mirrorable(prod, stg, scope)
    for t, why in skipped:
        log(f"{t}: skipped ({why})")
    password = secrets.token_hex(24)
    publish(prod, ok, password)
    subscribe(stg, ok, password)


def unique_keys(stg, name):
    """(columns, predicate or None) of a table's unique constraints and indexes, the primary key excepted."""
    s, t = split(name)
    return [(list(r[0]), r[1]) for r in q(stg, """
        SELECT array_agg(a.attname ORDER BY k.ord), pg_get_expr(i.indpred, i.indrelid) FROM pg_index i
        JOIN pg_class t ON t.oid = i.indrelid JOIN pg_namespace n ON n.oid = t.relnamespace
        CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ord)
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
        WHERE n.nspname = %s AND t.relname = %s AND i.indisunique AND NOT i.indisprimary
        GROUP BY i.indexrelid, i.indpred, i.indrelid""", (s, t))]


def references(stg, name):
    """(schema.table, column) pairs whose single-column foreign key points at name.id."""
    s, t = split(name)
    return [(f"{a}.{b}", c) for a, b, c in q(stg, """
        SELECT ns.nspname, cl.relname, a.attname FROM pg_constraint c
        JOIN pg_class cl ON cl.oid = c.conrelid JOIN pg_namespace ns ON ns.oid = cl.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
        JOIN pg_class rt ON rt.oid = c.confrelid JOIN pg_namespace rn ON rn.oid = rt.relnamespace
        WHERE c.contype = 'f' AND rn.nspname = %s AND rt.relname = %s AND array_length(c.conkey, 1) = 1""", (s, t))]


def repair(prod, stg):
    """Each staging-own row (id >= OFFSET) in a mirrored table whose unique key production also has becomes
    production's row: staging's references move to production's id and staging's row goes, so the subscription
    carries on. Staging only (production is read). Returns the number of rows moved."""
    from psycopg import sql
    mirrored = subscribed(stg) if subscription(stg) else set()
    moved = 0
    for t in sorted(mirrored):
        if "id" not in columns(stg, t):
            continue
        for key, pred in unique_keys(stg, t):
            cols = sql.SQL(", ").join(ident(k) for k in key)
            where = sql.SQL(" AND ").join(sql.SQL("{} = %s").format(ident(k)) for k in key)
            if pred:
                where = sql.SQL("{} AND ({})").format(where, sql.SQL(pred))
                mine = sql.SQL("SELECT id, {} FROM {} WHERE id >= %s AND ({})").format(cols, ident(t), sql.SQL(pred))
            else:
                mine = sql.SQL("SELECT id, {} FROM {} WHERE id >= %s").format(cols, ident(t))
            for sid, *vals in q(stg, mine, (OFFSET,)):
                if any(v is None for v in vals):
                    continue
                pid = one(prod, sql.SQL("SELECT id FROM {} WHERE {}").format(ident(t), where), vals)
                if pid is None:
                    continue
                with stg.transaction():
                    stg.execute("SET LOCAL session_replication_role = replica")
                    for rt, rc in references(stg, t):
                        stg.execute(sql.SQL("UPDATE {} SET {} = %s WHERE {} = %s").format(ident(rt), ident(rc), ident(rc)),
                                    (pid, sid))
                    stg.execute(sql.SQL("DELETE FROM {} WHERE id = %s").format(ident(t)), (sid,))
                log(f"{t}: staging's row {sid} {dict(zip(key, vals))} is production's {pid} now (references moved)")
                moved += 1
    lift_sequences(stg, mirrored)
    return moved


def status(prod, stg):
    """One line per check; returns False when something needs a look."""
    sub = subscription(stg)
    if not sub:
        log("mirror: off (staging holds its last copy of production)")
        return True
    good = True
    st = q(stg, "SELECT srsubstate, count(*) FROM pg_subscription_rel r JOIN pg_subscription s ON s.oid = r.srsubid "
                "WHERE s.subname = %s GROUP BY 1", (NAME,))
    states = {"r": "ready", "s": "synced", "d": "copying", "f": "copied", "i": "starting"}
    log("tables: " + ", ".join(f"{n} {states.get(k, k)}" for k, n in st))
    errs = q(stg, "SELECT apply_error_count, sync_error_count FROM pg_stat_subscription_stats WHERE subname = %s", (NAME,))
    if errs and (errs[0][0] or errs[0][1]):
        log(f"errors so far (cumulative): apply {errs[0][0]}, initial copy {errs[0][1]} "
            "(`docker compose logs db` names the row; repair fixes staging-own collisions)")
    worker = one(stg, "SELECT bool_or(pid IS NOT NULL) FROM pg_stat_subscription WHERE subname = %s", (NAME,))
    if not sub[1] or not worker:
        log("WARN: the subscription's worker isn't running" + ("" if sub[1] else " (disabled)"))
        good = False
    slot = q(prod, "SELECT active, wal_status, pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn)), "
                   "pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn) FROM pg_replication_slots WHERE slot_name = %s",
                   (NAME,))
    if not slot:
        log("WARN: production has no slot: the mirror is broken; bash scripts/deploy/vm.sh mirror off, then on")
        return False
    active, wal, behind, behind_b = slot[0]
    log(f"production slot: {'active' if active else 'INACTIVE'}, wal {wal}, staging is {behind} behind")
    if wal in ("lost", "unreserved") or (behind_b or 0) > 512 * 2**20:
        log("WARN: staging is far behind production (a stalled apply?): bash scripts/deploy/vm.sh mirror repair; "
            "if it stays, mirror off then on")
        good = False
    own = sorted((tables(prod) & tables(stg)) - subscribed(stg))
    log(f"staging's own (not mirrored): {', '.join(own)}")
    for t in ("public.model_runs", "public.market_trades"):
        if t in subscribed(stg):
            p, s = one(prod, f"SELECT max(id) FROM {t}"), one(stg, f"SELECT max(id) FROM {t} WHERE id < {OFFSET}")
            log(f"{t}: newest id production {p}, staging {s}" + ("" if p == s else "  (catching up)"))
    return good


def turn_off(prod, stg):
    from psycopg import sql
    if subscription(stg):
        stg.execute(sql.SQL("ALTER SUBSCRIPTION {} DISABLE").format(ident(NAME)))
        stg.execute(sql.SQL("ALTER SUBSCRIPTION {} SET (slot_name = NONE)").format(ident(NAME)))
        stg.execute(sql.SQL("DROP SUBSCRIPTION {}").format(ident(NAME)))
        log("staging: subscription dropped (its rows stay, as of now)")
    if one(prod, "SELECT 1 FROM pg_replication_slots WHERE slot_name = %s", (NAME,)):
        prod.execute("SELECT pg_drop_replication_slot(%s)", (NAME,))
    prod.execute(sql.SQL("DROP PUBLICATION IF EXISTS {}").format(ident(NAME)))
    if one(prod, "SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,)):
        for s in SCHEMAS:
            if one(prod, "SELECT 1 FROM pg_namespace WHERE nspname = %s", (s,)):
                prod.execute(sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM {}").format(ident(s), ident(ROLE)))
                prod.execute(sql.SQL("REVOKE ALL ON SCHEMA {} FROM {}").format(ident(s), ident(ROLE)))
        prod.execute(sql.SQL("DROP ROLE {}").format(ident(ROLE)))
    log("production: slot, publication and role dropped")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m racinglines.db.mirror", description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("command", choices=("plan", "on", "status", "repair", "off"))
    ap.add_argument("--scope", choices=SCOPES, default="all", help="what follows production (default: all, the owner's pick 2026-10-12)")
    a = ap.parse_args(argv)
    prod_url, stg_url = urls()
    prod, stg = connect(prod_url, read_only=a.command not in ("on", "off")), connect(stg_url)
    if a.command == "plan":
        plan(prod, stg, a.scope)
    elif a.command == "on":
        turn_on(prod, stg, a.scope)
        status(prod, stg)
    elif a.command == "status":
        return 0 if status(prod, stg) else 1
    elif a.command == "repair":
        n = repair(prod, stg) if subscription(stg) else 0
        if n:
            log(f"repair: {n} staging rows matched to production's")
        return 0 if status(prod, stg) else 1
    elif a.command == "off":
        turn_off(prod, stg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
