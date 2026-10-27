"""
The MCP tools as plain functions: `(conn, ...) -> dict`, where `conn` is a SQLAlchemy connection (a read-only
transaction; see server.py). Every list goes through page.page (row and byte caps); everything reuses the
functions the web app reads from: db/reads.py, markets/venues.py, markets/store.py, web/edge.py, web/diag.py,
pipelines/story.py, web/jobs.py. Nothing here writes to the database except the job tools (queue, cancel), which
do what the Lab's Run form does.
"""

import contextvars
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from racinglines.db import reads as data
from racinglines.mcp import page as P

RUN_KINDS = ("forecast", "scenario", "diagnostic", "backtest", "sweep", "season_strategy", "season_checkpoints",
             "season_asof", "walk_forward", "candidate")
# tables the sql tool never reads (login secrets; exchange credentials and responses)
SQL_HIDDEN = ("users", "orders")
SQL_TIMEOUT = "10s"

# one line per table, for describe_schema (docs/database.md has the long version)
TABLES = {
    "sports": "reference: what is raced (f1, mtb_dh)", "leagues": "reference: who organizes it",
    "competitions": "a league's championship in one sport (f1_wdc, uci_dhi_wc)", "categories": "per competition (DRV, ME, MJ, ...)",
    "seasons": "competition x year", "events": "a race weekend / round: source_key, start_date, venue, status",
    "venues": "circuits and resorts", "venue_aliases": "spellings of a venue in source data",
    "races": "event x category, with the weekend format (JSONB)", "rounds": "practice / qual / final / race ... of a race",
    "results": "one row per athlete per round: position, status, time_ms", "splits": "intermediate times of a result",
    "laps": "F1 laps: lap and sector ms, tyre, stint, position", "track_profiles": "per-event track features the F1 model uses",
    "athletes": "drivers and riders (shared across sports)", "athlete_identifiers": "(scheme, value) ids of an athlete",
    "source_files": "every ingested file with its hash", "points_schemes": "championship points tables",
    "model_runs": "one row per model run: kind (forecast, scenario, diagnostic, backtest, sweep, ...), params, metrics (JSONB)",
    "race_predictions": "per-athlete probabilities of a run for a race (win, podium, top10, make_final, exp_points)",
    "standings_predictions": "per-athlete projected standings of a run",
    "market_links": "an exchange token (Polymarket) or market (Kalshi) linked to a prediction kind, athlete, race; last price, volume, resolution",
    "market_price_history": "recent exchange prices (older rows are in the Parquet archive; use get_market_history)",
    "market_trades": "recent exchange trades (same)", "market_book_snapshots": "recent order books (same)",
    "house_markets": "YES/NO markets quoted in the app (the private book)", "house_bets": "bets against house markets",
    "users": "web-app accounts (hidden from sql)", "activity_log": "web-app audit trail", "jobs": "Lab jobs (CLI subprocesses)",
    "strategy_signals": "paper signals of a user's strategy profile", "paper_positions": "paper positions per user, market, venue",
    "live_events": "settled live private-book events", "data_changes": "the data change log", "orders": "exchange orders (hidden from sql)",
}


def _all_sports_of(conn):
    return {r[0]: r[1] for r in conn.execute(text(
        "SELECT co.code, sp.code FROM competitions co JOIN sports sp ON sp.id = co.sport_id ORDER BY co.code")).fetchall()}


def sports_of(conn):
    """{competition code: sport code}, from the competitions and sports tables: every sport the database holds that the
    caller may see (hidden_sports)."""
    hidden = HIDDEN.get()
    return {c: s for c, s in _all_sports_of(conn).items() if s not in hidden}


# ---------------------------------------------------------------------------------------------------
# sports a caller may not see: sports.is_public ([sport] audience = "admin"), for non-admin accounts
# ---------------------------------------------------------------------------------------------------

# The sport codes this request may not see. server.py sets it per tool call from the caller (hidden_sports); empty for
# internal callers (stdio, the CLI, tests) and admins, so every tool reads as before for them.
HIDDEN = contextvars.ContextVar("racinglines_mcp_hidden_sports", default=frozenset())


def hidden_sports(viewer):
    """The sports `viewer` (server.caller(): dict(id, username, role), None over stdio) may not see: the ones the one
    gate, sports.is_public, turns away ([sport] audience = "admin", as on the web pages). None for internal callers and
    admins."""
    from racinglines import sports as SC
    from racinglines.web import roles as R
    if not viewer or R.canonical(viewer.get("role")) == "admin":
        return frozenset()
    return frozenset(c for c in SC.SPORT_CODES if not SC.is_public(c, viewer))


# The exchange codes this request may not see ([exchange] audience = "admin"), set per tool call like HIDDEN.
HIDDEN_VENUES = contextvars.ContextVar("racinglines_mcp_hidden_venues", default=frozenset())


def _hv():
    h = HIDDEN_VENUES.get()
    return dict(hide_venues=h) if h else {}


def hidden_venues(viewer):
    """The exchanges `viewer` may not see: the ones the one gate, exchanges.is_public, turns away. None for internal
    callers (stdio) and admins. The raw `sql` tool is admin-only, so it needs no filter."""
    from racinglines import exchanges as EX
    from racinglines.web import roles as R
    if not viewer or R.canonical(viewer.get("role")) == "admin":
        return frozenset()
    return frozenset(c for c in EX.CODES if not EX.is_public(c, viewer))


def not_available(what):
    return ValueError(f"{what} is not available")


def _hidden(conn):
    """What the hidden sports cover, as the keys the tools read rows by: dict(sports, competitions, event_ids,
    event_keys, race_ids, run_ids). None when the caller sees everything."""
    sports = HIDDEN.get()
    if not sports:
        return None
    comps = sorted(c for c, s in _all_sports_of(conn).items() if s in sports)
    ids = lambda sql: {r[0] for r in conn.execute(text(sql), dict(c=comps)).fetchall()}        # noqa: E731
    events = conn.execute(text("""SELECT e.id, e.source_key FROM events e JOIN seasons s ON s.id = e.season_id
                                  JOIN competitions co ON co.id = s.competition_id WHERE co.code = ANY(:c)"""),
                          dict(c=comps)).fetchall()
    return dict(sports=sports, competitions=set(comps), event_ids={int(r[0]) for r in events},
                event_keys={r[1] for r in events if r[1]},
                race_ids=ids("""SELECT ra.id FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
                                JOIN competitions co ON co.id = s.competition_id WHERE co.code = ANY(:c)"""),
                run_ids=ids("""SELECT mr.id FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
                               WHERE co.code = ANY(:c)"""))


def _drop_hidden(conn, rows, **cols):
    """`rows` (a DataFrame or a list of dicts) without the ones a hidden sport covers: cols maps a column to the key
    set of _hidden it is checked against (e.g. competition="competitions", event_id="event_ids")."""
    h = _hidden(conn)
    if h is None:
        return rows
    if isinstance(rows, pd.DataFrame):
        keep = pd.Series(True, index=rows.index)
        for col, key in cols.items():
            if col in rows.columns:
                keep &= ~rows[col].isin(h[key])
        return rows[keep]
    return [r for r in rows if not any(r.get(col) in h[key] for col, key in cols.items())]


def _guard(conn, key, value, what=None):
    """ValueError 'not available' when `value` (a competition code, event id, race id, ...) is in _hidden's `key`."""
    h = _hidden(conn)
    if h is not None and value is not None and value in h[key]:
        raise not_available(what or "that")


def _guard_tree(conn, obj):
    """Refuse an output (a mapped or priced book) when any part of it names a hidden sport, competition or race."""
    h = _hidden(conn)
    if h is None:
        return obj
    def walk(o):
        if isinstance(o, dict):
            sport, comp, race = o.get("sport") or o.get("sport_code"), o.get("competition"), o.get("race_id")
            if sport in h["sports"] or comp in h["competitions"] or (race is not None and race in h["race_ids"]):
                raise not_available(sport or comp or "that race")
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return obj


def _sport_filter(conn, sport=None, competition=None):
    """The competition code a tool filters on: `competition` as given, else the one competition of `sport`. A hidden
    sport or competition is 'not available' (never 'no sport', which would say it exists elsewhere)."""
    hidden = HIDDEN.get()
    if competition:
        if hidden and _all_sports_of(conn).get(competition) in hidden:
            raise not_available(competition)
        return competition
    if not sport:
        return None
    if sport in hidden or _all_sports_of(conn).get(sport) in hidden:
        raise not_available(sport)
    of = sports_of(conn)
    if sport in of:                     # a competition code passed as the sport
        return sport
    comps = [c for c, s in of.items() if s == sport]
    if len(comps) == 1:
        return comps[0]
    if comps:
        raise ValueError(f"sport {sport!r} has several competitions ({', '.join(comps)}): pass competition")
    raise ValueError(f"no sport {sport!r}; the database has {', '.join(sorted(set(of.values())))}")


def _sources(sport=None, venues=None, run=None, as_of=None, model=False):
    """The compact source note every answer with event, result or price data carries (racinglines/attribution.py):
    {results: [{name, url}], prices: [venue names], fair: "racinglines model (run #N, as of ...)" or None, text}."""
    from racinglines import attribution
    n = attribution.note(sport, venues, run, as_of, model=model)
    return {k: n[k] for k in ("results", "prices", "fair", "text")}


def _token_sources(conn, toks, run=None):
    """_sources for a list of market tokens: their sports and exchanges, read by exact token id."""
    got = conn.execute(text("""SELECT DISTINCT co.code, ml.exchange FROM market_links ml
                               JOIN competitions co ON co.id = ml.competition_id WHERE ml.token_id = ANY(:t)"""),
                       dict(t=list(toks))).fetchall()
    return _sources(sorted({r[0] for r in got}), sorted({r[1] for r in got}))


def _run_competition(conn, run_id):
    """The competition code of a model run, None when it has none."""
    return conn.execute(text("""SELECT co.code FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
                                WHERE mr.id = :r"""), dict(r=int(run_id))).scalar()


def _usernames(conn):
    return {r[0]: int(r[1]) for r in conn.execute(text("SELECT username, id FROM users")).fetchall()}


def _user_id(conn, user):
    ids = _usernames(conn)
    if user is None:
        raise ValueError(f"which user? one of: {', '.join(sorted(ids))}")
    if user not in ids:
        raise ValueError(f"no user {user!r}; users: {', '.join(sorted(ids))}")
    return ids[user]


def _account_viewer(viewer):
    """Hosted non-admin accounts can access only their own account data; stdio remains owner mode."""
    from racinglines.web import roles as R
    return viewer if viewer and R.canonical(viewer.get("role")) != "admin" else None


def _account_id(conn, user, viewer):
    scoped = _account_viewer(viewer)
    if scoped:
        if user is not None and user != scoped["username"]:
            raise ValueError("MCP access is limited to your own account")
        return scoped["id"], scoped["username"]
    return _user_id(conn, user), user


# ---------------------------------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------------------------------

def overview(conn, viewer=None):
    """What the database holds, so a client knows what to ask for."""
    from racinglines.markets import venues as V
    seasons = data.q(conn, """
        SELECT co.code AS competition, s.year AS season, count(e.id) AS events,
               count(e.id) FILTER (WHERE e.status = 'completed') AS completed, min(e.start_date) AS first, max(e.start_date) AS last
        FROM seasons s JOIN competitions co ON co.id = s.competition_id LEFT JOIN events e ON e.season_id = s.id
        GROUP BY 1, 2 HAVING count(e.id) > 0 ORDER BY 1, 2""")
    h = _hidden(conn)
    no_comps = sorted(h["competitions"]) if h else []
    seasons = _drop_hidden(conn, seasons, competition="competitions")
    runs = data.q(conn, """SELECT mr.kind, count(*) AS runs, max(mr.created_at) AS latest FROM model_runs mr
                           JOIN competitions co ON co.id = mr.competition_id WHERE co.code <> ALL(:c)
                           GROUP BY mr.kind ORDER BY mr.kind""", c=no_comps)
    links = data.q(conn, """SELECT ml.exchange, count(*) AS links, count(*) FILTER (WHERE NOT ml.closed) AS open,
                            count(DISTINCT ml.race_id) AS races FROM market_links ml
                            LEFT JOIN competitions co ON co.id = ml.competition_id
                            WHERE (co.code IS NULL OR co.code <> ALL(:c)) AND ml.exchange <> ALL(:x)
                            GROUP BY ml.exchange ORDER BY ml.exchange""", c=no_comps, x=sorted(HIDDEN_VENUES.get()))
    scoped = _account_viewer(viewer)
    uid = scoped["id"] if scoped else None
    users = data.q(conn, """SELECT username, role, prefs->'strategy_profile'->>'name' AS profile FROM users
                           WHERE active AND (CAST(:u AS int) IS NULL OR id = :u) ORDER BY id""", u=uid)
    counts = {t: int(conn.execute(text(f"SELECT count(*) FROM {t}")).scalar())
              for t in ("events", "results", "athletes", "market_links", "model_runs")}
    for table in ("paper_positions", "strategy_signals", "jobs"):
        counts[table] = int(conn.execute(text(
            f"SELECT count(*) FROM {table} WHERE CAST(:u AS int) IS NULL OR user_id = :u"), dict(u=uid)).scalar())
    upcoming = data.q(conn, """SELECT e.id AS event_id, co.code AS competition, e.name, e.start_date FROM events e
                               JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
                               WHERE e.status <> 'completed' AND e.start_date >= current_date - 7 AND co.code <> ALL(:c)
                               ORDER BY e.start_date LIMIT 6""", c=no_comps)
    return dict(
        sports=sports_of(conn),
        seasons=[P.record(r) for r in seasons.to_dict("records")],
        forecasts=[P.record(r) for r in _drop_hidden(conn, data.latest_forecasts(conn), competition="competitions").to_dict("records")],
        model_runs=[P.record(r) for r in runs.to_dict("records")],
        market_links=[P.record(r) for r in links.to_dict("records")],
        venues=[dict(code=v.code, name=v.name, status=v.status, kind=v.kind) for v in V.shown(V.VENUES, HIDDEN_VENUES.get())],
        users=[P.record(r) for r in users.to_dict("records")],
        upcoming=[P.record(r) for r in _drop_hidden(conn, upcoming, competition="competitions").to_dict("records")],
        row_counts=counts,
        hint="Start with list_events / list_markets / get_forecast; "
             + ("account records and jobs are your own; SQL and audit tools are admin-only; " if scoped else
                "use sql for anything else (read-only); ")
             + "run_job for a simulation and get_job to follow it.",
    )


def describe_schema(conn, table=None):
    """The tables with one-line meanings, or one table's columns."""
    if table is None:
        return dict(tables=[dict(table=t, what=w) for t, w in TABLES.items()])
    if not re.fullmatch(r"[a-z_]+", table):
        raise ValueError("table names are lower-case words")
    cols = data.q(conn, """SELECT column_name AS name, data_type AS type, is_nullable = 'YES' AS nullable, column_default AS "default"
                           FROM information_schema.columns WHERE table_schema = 'public' AND table_name = :t ORDER BY ordinal_position""", t=table)
    if not len(cols):
        raise ValueError(f"no table {table!r}")
    n = int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()) if table not in SQL_HIDDEN else None
    return dict(table=table, what=TABLES.get(table, ""), rows=n, columns=[P.record(r) for r in cols.to_dict("records")])


# ---------------------------------------------------------------------------------------------------
# events, athletes
# ---------------------------------------------------------------------------------------------------

def list_events(conn, sport=None, competition=None, season=None, status=None, limit=None, offset=0):
    df = _drop_hidden(conn, data.events(conn, competition=_sport_filter(conn, sport, competition), season=season),
                      id="event_ids")
    if status:
        df = df[df["status"] == status]
    out = P.page(df, limit=limit, offset=offset)
    out["sources"] = _sources(sorted(set(df["competition"].dropna())) if len(df) else _sport_filter(conn, sport, competition))
    return out


def get_event(conn, event_id=None, source_key=None, include="results", viewer=None):
    """One event: its summary, and with include= any of results, predictions, markets (comma-separated)."""
    if event_id is None and source_key:
        row = data.q(conn, "SELECT id FROM events WHERE source_key = :k ORDER BY id DESC LIMIT 1", k=source_key)
        if not len(row):
            raise ValueError(f"no event with source_key {source_key!r}")
        event_id = int(row["id"].iloc[0])
    ev = data.event(conn, int(event_id))
    if not ev:
        raise ValueError(f"no event {event_id}")
    _guard(conn, "competitions", ev.get("competition"), f"event {event_id}")
    wants = {w.strip() for w in (include or "").split(",") if w.strip()}
    races = data.q(conn, """SELECT ra.id AS race_id, c.code AS category, ra.format FROM races ra
                            JOIN categories c ON c.id = ra.category_id WHERE ra.event_id = :e ORDER BY c.code""", e=event_id)
    out = dict(event=P.record(ev), races=[P.record(r) for r in races.to_dict("records")])
    if "results" in wants:
        res = data.event_results(conn, event_id)
        finals = res[res["round"].isin(("final", "race"))] if len(res) else res
        out["results"] = P.page(finals if len(finals) else res, limit=60, note="final/race rounds only; sql for every round")
        if len(res):
            out["rounds"] = {str(k): int(v) for k, v in res.groupby("round").size().items()}
    if "predictions" in wants:
        pr = data.event_predictions(conn, event_id)
        out["predictions"] = P.page(pr, limit=60, note="newest run first")
    if "markets" in wants:
        out["markets"] = {int(r): list_markets(conn, race_id=int(r), limit=40, viewer=viewer) for r in races["race_id"]}
    shown = [v["venue"] for m in out.get("markets", {}).values() for v in m.get("venues", []) if v.get("listed")]
    out["sources"] = _sources(ev.get("competition"), shown, model="predictions" in wants or "markets" in wants)
    return out


def search_athletes(conn, q=None, limit=None):
    return P.page(data.athletes(conn, q, limit=P.MAX_LIMIT), limit=limit)


def get_athlete(conn, athlete_id, include="results"):
    a = data.athlete(conn, int(athlete_id))
    if not a:
        raise ValueError(f"no athlete {athlete_id}")
    h = _hidden(conn)
    comps = data.athlete_competitions(conn, athlete_id) if h else None
    if comps and comps <= h["competitions"]:            # raced (or priced) only in hidden sports, as the web page
        raise not_available(f"athlete {athlete_id}")
    wants = {w.strip() for w in (include or "").split(",") if w.strip()}
    out = dict(athlete=P.record(a))
    if "results" in wants:
        out["results"] = P.page(_drop_hidden(conn, data.athlete_results(conn, athlete_id), event_id="event_ids"), limit=60)
    if "predictions" in wants:
        out["predictions"] = P.page(_drop_hidden(conn, data.athlete_predictions(conn, athlete_id), run="run_ids"), limit=60)
    comps = data.athlete_competitions(conn, athlete_id) if comps is None else comps
    out["sources"] = _sources(sorted(comps - (h["competitions"] if h else set())), model="predictions" in wants)
    return out


# ---------------------------------------------------------------------------------------------------
# markets: the venues matrix and the exchange time series
# ---------------------------------------------------------------------------------------------------

def _matrix_rows(df):
    rows = []
    for r in df.to_dict("records"):
        row = dict(kind=r["kind"], subject=r["subject"], detail=r["detail"], athlete_id=data._id(r["athlete_id"]), team=r["team"],
                   opponent_id=data._id(r["opponent_id"]), fair=r["fair"], result=r.get("result"))
        for venue, q in (r.get("venues") or {}).items():
            row[f"{venue}_mid"] = q.get("mid")
            row[f"{venue}_bid"], row[f"{venue}_ask"] = q.get("bid"), q.get("ask")
            row[f"{venue}_volume"] = q.get("volume")
            row[f"{venue}_token"] = q.get("token")
            row[f"{venue}_closed"] = q.get("closed")
            if q.get("at_as_of"):
                row[f"{venue}_resolved_mid"] = q.get("resolved_mid")
        p = r.get("private")
        if isinstance(p, dict):
            row.update(private_yes=p.get("yes"), private_no=p.get("no"), private_status=p.get("status"), private_bets=p.get("bets"),
                       private_settled_pnl=p.get("settled_pnl"))
        row["gap_polymarket"] = r.get("gap")
        rows.append(row)
    return rows


def list_markets(conn, race_id=None, event_id=None, competition=None, sport=None, kinds=None, limit=None, offset=0,
                 viewer=None):
    """One row per outcome (kind x subject): our fair value, each venue's quote, the gap and, for a past race, the
    result. Give race_id (or event_id: its first race) for a race weekend, or competition/sport for the season markets."""
    from racinglines.markets import venues as V
    from racinglines.markets import private_book as house
    scoped = _account_viewer(viewer)
    maker_id = scoped["id"] if scoped else house.ALL
    if race_id is None and event_id is not None:
        r = data.q(conn, "SELECT id FROM races WHERE event_id = :e ORDER BY id LIMIT 1", e=int(event_id))
        if not len(r):
            raise ValueError(f"event {event_id} has no races")
        race_id = int(r["id"].iloc[0])
    if race_id is not None:
        _guard(conn, "race_ids", int(race_id), f"race {race_id}")
        info, pricing, df = V.event_matrix(conn, int(race_id), maker_id=maker_id, **_hv())
        if info is None:
            raise ValueError(f"no race {race_id}")
        head = dict(race_id=int(race_id), event_id=info["event_id"], title=info["title"], competition=info["competition"],
                    season=info["season"], status=info["status"], start_date=info["start_date"], race_start=info["race_start"])
    else:
        comp = _sport_filter(conn, sport, competition)
        if not comp:
            raise ValueError("give race_id, event_id, or competition/sport (season markets)")
        info, pricing, df = V.season_matrix(conn, comp, maker_id=maker_id, **_hv())
        if info is None:
            raise ValueError(f"no competition {comp!r}")
        head = dict(competition=comp, title=info["title"], scope="season")
    if kinds:
        want = [k.strip() for k in kinds.split(",")]
        df = df[df["kind"].isin(want)] if len(df) else df
    rows = _matrix_rows(df) if len(df) else []
    by_kind = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    scope = dict(event_id=head["event_id"]) if race_id is not None else dict(competition=head["competition"])
    upcoming = race_id is None or head.get("status") != "completed"
    venues = [dict(venue=v["venue"].code, listed=v["listed"], volume=v["volume"]) for v in V.venue_summary(df, HIDDEN_VENUES.get())] if len(df) else []
    pr = pricing or {}
    return dict(**P.record(head), pricing=P.record(pr), outcomes_by_kind=by_kind,
                freshness=freshness(conn, upcoming=upcoming, **scope),
                venues=venues,
                outcomes=P.page(rows, limit=limit, offset=offset),
                sources=_sources(head["competition"], [v["venue"] for v in venues if v["listed"]], pr.get("run_id"),
                                 pr.get("as_of"), model=pr.get("run_id") is not None or pr.get("as_of") is not None))


def stale_hours():
    """How old a venue's newest market link sync may be before list_markets flags its prices stale:
    RACINGLINES_STALE_HOURS, default 3 (polling runs every 5 minutes, so 3 h means several missed passes)."""
    try:
        return float(os.environ.get("RACINGLINES_STALE_HOURS") or 3)
    except ValueError:
        return 3.0


def freshness(conn, event_id=None, competition=None, upcoming=True, now=None):
    """Per exchange: links, the newest synced_at over the event's races (or the competition's season links, with no
    race) and its age; `stale` when an upcoming event has no link on a live exchange or the newest sync is older than
    stale_hours(). A completed event is never flagged: its prices are history."""
    from racinglines.markets import venues as V
    if event_id is not None:
        df = data.q(conn, """SELECT ml.exchange, count(*) AS links, max(ml.synced_at) AS synced_at FROM market_links ml
                             JOIN races ra ON ra.id = ml.race_id WHERE ra.event_id = :e GROUP BY 1""", e=int(event_id))
    else:
        df = data.q(conn, """SELECT ml.exchange, count(*) AS links, max(ml.synced_at) AS synced_at FROM market_links ml
                             JOIN competitions co ON co.id = ml.competition_id
                             WHERE co.code = :c AND ml.race_id IS NULL GROUP BY 1""", c=competition)
    got = {r["exchange"]: r for r in df.to_dict("records") if r["exchange"] not in HIDDEN_VENUES.get()}
    limit = stale_hours()
    now = now or pd.Timestamp.now(tz="UTC")
    codes = [v.code for v in V.shown(V.EXCHANGES, HIDDEN_VENUES.get()) if v.status == "live"]
    codes += [c for c in sorted(got) if c not in codes]
    out = []
    for code in codes:
        r = got.get(code)
        links = int(r["links"]) if r else 0
        at = pd.Timestamp(r["synced_at"]) if r and r["synced_at"] is not None and not pd.isna(r["synced_at"]) else None
        age = (now - at).total_seconds() / 3600 if at is not None else None
        why = None
        if upcoming:
            why = ("no linked markets" if not links else "never synced" if at is None else
                   f"newest sync {age:.1f} h old (limit {limit:g} h)" if age > limit else None)
        out.append(dict(venue=code, links=links, synced_at=P.plain(at), age_hours=P.plain(age), stale=why is not None,
                        reason=why))
    return dict(stale_hours=limit, any_stale=any(r["stale"] for r in out), venues=out)


def _tokens_for(conn, race_id=None, kind=None, athlete_id=None, subject=None, exchange=None, competition=None):
    where, params = ["prediction <> 'unmodeled'"], {}
    if race_id is not None:
        where.append("ml.race_id = :r"); params["r"] = int(race_id)
    if competition:
        where.append("co.code = :c AND ml.race_id IS NULL"); params["c"] = competition
    if kind:
        where.append("ml.prediction = :k"); params["k"] = kind
    if athlete_id is not None:
        where.append("ml.athlete_id = :a"); params["a"] = int(athlete_id)
    if subject:
        where.append("(a.display_name ILIKE '%' || :s || '%' OR ml.params->>'team' ILIKE '%' || :s || '%' OR ml.question ILIKE '%' || :s || '%')")
        params["s"] = subject
    if exchange:
        where.append("ml.exchange = :x"); params["x"] = exchange
    where.append("ml.exchange <> ALL(:hx)"); params["hx"] = sorted(HIDDEN_VENUES.get())     # an admin-only exchange
    df = data.q(conn, f"""
        SELECT ml.token_id, ml.exchange, ml.prediction AS kind, coalesce(a.display_name, ml.params->>'team', ml.outcome) AS subject,
               ml.outcome, ml.question, ml.invert, ml.closed, ml.resolved_yes, ml.race_id, co.code AS competition
        FROM market_links ml LEFT JOIN athletes a ON a.id = ml.athlete_id JOIN competitions co ON co.id = ml.competition_id
        WHERE {' AND '.join(where)} ORDER BY ml.id""", **params)
    return _drop_hidden(conn, df, competition="competitions").drop(columns="competition")


def get_market_history(conn, tokens=None, race_id=None, kind=None, athlete_id=None, subject=None, exchange=None,
                       series="prices", start=None, end=None, resample="1h", limit=None, offset=0):
    """Exchange time series from the archive and the database: prices (resampled: last price per bucket), trades
    (per bucket: count, volume, VWAP; resample=None lists trades) or books (best bid/ask per bucket). Pick the
    markets by tokens (comma-separated ids) or by race_id/kind/athlete_id/subject; a race defaults to the week
    before its start. Every token is capped at 200 points."""
    from racinglines.markets import store as MS
    if series not in ("prices", "trades", "books"):
        raise ValueError("series is prices, trades or books")
    if race_id is not None:
        _guard(conn, "race_ids", int(race_id), f"race {race_id}")
    if tokens:
        toks = [t.strip() for t in str(tokens).split(",") if t.strip()]
        links = _tokens_for(conn, exchange=exchange)
        links = links[links["token_id"].isin(toks)] if len(links) else links
        names = {r["token_id"]: r for r in links.to_dict("records")}
        toks = list(dict.fromkeys(toks))
        if HIDDEN_VENUES.get():              # an admin-only exchange's token reads as no market too
            gone = {r[0] for r in conn.execute(text("SELECT token_id FROM market_links WHERE exchange = ANY(:x) AND token_id = ANY(:t)"),
                                               dict(x=sorted(HIDDEN_VENUES.get()), t=toks)).fetchall()}
            toks = [t for t in toks if t not in gone]
        h = _hidden(conn)
        if h:                                # a hidden sport's token reads as no market (_tokens_for left it out)
            gone = {r[0] for r in conn.execute(text(
                "SELECT ml.token_id FROM market_links ml JOIN competitions co ON co.id = ml.competition_id "
                "WHERE co.code = ANY(:c) AND ml.token_id = ANY(:t)"), dict(c=sorted(h["competitions"]), t=toks)).fetchall()}
            toks = [t for t in toks if t not in gone]
    else:
        links = _tokens_for(conn, race_id=race_id, kind=kind, athlete_id=athlete_id, subject=subject, exchange=exchange)
        if len(links) > 40:
            raise ValueError(f"{len(links)} markets match; narrow with kind/athlete_id/subject/exchange or pass tokens")
        toks = links["token_id"].tolist()
        names = {r["token_id"]: r for r in links.to_dict("records")}
    if not toks:
        return dict(tokens=[], series=series, rows=P.page([], limit=limit), note="no markets match")
    if start is None and end is None and race_id is not None:
        from racinglines.markets import venues as V
        info = V.race_info(conn, int(race_id))
        t1 = pd.Timestamp(info["race_start"] if info and info.get("race_start") is not None else info["start_date"]).tz_localize("UTC")
        start, end = t1 - pd.Timedelta(days=7), t1 + pd.Timedelta(hours=6)
    start = pd.Timestamp(start).tz_localize("UTC") if start is not None and pd.Timestamp(start).tzinfo is None else \
        (pd.Timestamp(start) if start is not None else None)
    end = pd.Timestamp(end).tz_localize("UTC") if end is not None and pd.Timestamp(end).tzinfo is None else \
        (pd.Timestamp(end) if end is not None else None)
    df = MS.read(conn, series, tokens=toks, start=start, end=end)
    if not len(df):
        return dict(tokens=[P.record(names.get(t, dict(token_id=t))) for t in toks], series=series,
                    rows=P.page([], limit=limit), note="no stored rows in that window")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    span = df["ts"].max() - df["ts"].min()
    step = None
    if resample:
        step = pd.Timedelta(resample)
        if span / step > 200:
            step = span / 200
    out = []
    for tok, g in df.groupby("token_id"):
        g = g.sort_values("ts")
        if step is None:
            cols = [c for c in ("ts", "price", "size", "side", "best_bid", "best_ask") if c in g.columns]
            s = g[cols].tail(200)
            recs = s.to_dict("records")
        elif series == "prices":
            s = g.set_index("ts")["price"].resample(step).last().dropna()
            recs = [dict(ts=t, price=v) for t, v in s.items()]
        elif series == "trades":
            gi = g.set_index("ts")
            agg = gi.resample(step).agg(trades=("price", "size"), volume=("size", "sum"))
            vw = (gi["price"] * gi["size"]).resample(step).sum() / gi["size"].resample(step).sum()
            agg["vwap"] = vw
            agg = agg[agg["trades"] > 0]
            recs = [dict(ts=t, **r) for t, r in agg.to_dict("index").items()]
        else:
            s = g.set_index("ts")[["best_bid", "best_ask"]].resample(step).last().dropna(how="all")
            recs = [dict(ts=t, **r) for t, r in s.to_dict("index").items()]
        for r in recs:
            r["token_id"] = tok
        out.extend(recs)
    out.sort(key=lambda r: (r["token_id"], r["ts"]))
    return dict(tokens=[P.record(names.get(t, dict(token_id=t))) for t in toks], series=series,
                window=dict(start=P.plain(df["ts"].min()), end=P.plain(df["ts"].max()), rows_read=int(len(df)),
                            bucket=str(step) if step is not None else None),
                rows=P.page(out, limit=limit or 200, offset=offset, max_bytes=P.MAX_BYTES * 2),
                sources=_token_sources(conn, toks))


# ---------------------------------------------------------------------------------------------------
# model runs, predictions, forecasts
# ---------------------------------------------------------------------------------------------------

def list_model_runs(conn, kind=None, competition=None, sport=None, season=None, limit=None, offset=0):
    comp = _sport_filter(conn, sport, competition)
    df = data.q(conn, """
        SELECT mr.id, mr.kind, co.code AS competition, s.year AS season, c.code AS category, mr.model, mr.created_at,
               mr.data_through, mr.params->>'variant' AS variant, mr.params->>'cutoff' AS cutoff, mr.params->>'event_key' AS event_key,
               mr.params->>'label' AS label, mr.params->>'name' AS name, mr.params->>'year' AS year,
               (SELECT count(*) FROM race_predictions rp WHERE rp.model_run_id = mr.id) AS predictions,
               pg_column_size(mr.metrics) AS metrics_bytes
        FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
        LEFT JOIN seasons s ON s.id = mr.season_id LEFT JOIN categories c ON c.id = mr.category_id
        WHERE (CAST(:k AS text) IS NULL OR mr.kind = CAST(:k AS text)) AND (CAST(:c AS text) IS NULL OR co.code = CAST(:c AS text))
          AND (CAST(:y AS int) IS NULL OR s.year = CAST(:y AS int) OR (mr.params->>'year')::int = CAST(:y AS int))
        ORDER BY mr.id DESC""", k=kind, c=comp, y=season)
    df = _drop_hidden(conn, df, competition="competitions")
    return P.page(df, limit=limit, offset=offset, note=f"kinds: {', '.join(RUN_KINDS)}")


def compare_runs(conn, run_a, run_b, kinds=None, stage=None, diagnostics=False):
    """Two saved runs paired by event on their prediction records (pipelines/eval_compare.py, `racinglines eval compare`):
    per kind x season and pooled, log loss and Brier of each, B - A with its standard error over events and whether 0 is
    outside ±2 SE. diagnostics=True adds the pass test's tables (buckets, favourites, pole: reliability buckets, favourite wins
    and the pole-sitter win rate, as in `eval compare --diagnostics`). Read-only: nothing is written. Both runs need records
    (RACINGLINES_PREDICTION_RECORDS=1 with --save)."""
    from racinglines.pipelines import eval_compare as EC
    for rid in (run_a, run_b):
        run = data.model_run(conn, int(rid))
        if not run:
            raise ValueError(f"no model run {rid}")
        _guard(conn, "competitions", run["competition"], f"model run {rid}")
    kinds = [k.strip() for k in kinds.split(",") if k.strip()] if isinstance(kinds, str) else kinds
    res = EC.run(int(run_a), int(run_b), kinds=kinds or None, stage=stage, echo=lambda s: None, write=False,
                 diagnostics=bool(diagnostics), engine_url=conn.engine.url.render_as_string(hide_password=False))
    c = res["counts"]
    out = dict(run_a=int(run_a), run_b=int(run_b), kinds=kinds, stage=stage, diff="B - A (lower is better)",
               paired=c["paired"], only_a=c["only_a"], only_b=c["only_b"], unsettled=res["unsettled"],
               summary=P.page(res["summary"], limit=P.MAX_LIMIT))
    for name in ("buckets", "favourites", "pole"):
        if name in res:
            out[name] = P.page(res[name], limit=P.MAX_LIMIT)
    return out


def _dig(obj, path):
    cur = obj
    for part in [p for p in re.split(r"[./]", path) if p]:
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                raise ValueError(f"no key {part!r}; keys: {list(cur)[:40]}")
            cur = cur[part]
        else:
            raise ValueError(f"{path}: {part!r} is not inside a dict or list")
    return cur


def get_model_run(conn, run_id, path=None):
    """One run with its params and metrics. Large metrics come back as a key map with sizes; `path` (e.g.
    'metrics.weekends' or 'metrics.weekends.0') returns that part whole (paged when it is a list)."""
    run = data.model_run(conn, int(run_id))
    if not run:
        raise ValueError(f"no model run {run_id}")
    _guard(conn, "competitions", run["competition"], f"model run {run_id}")
    targets = data.run_race_targets(conn, run_id)
    if path:
        part = _dig(dict(params=run.get("params") or {}, metrics=run.get("metrics") or {}), path)
        if isinstance(part, list):
            rows = part if part and isinstance(part[0], dict) else [dict(value=v) for v in part]
            return dict(run_id=int(run_id), path=path, value=P.page(rows, limit=P.DEFAULT_LIMIT))
        val = P.plain(part, preview=False)
        s = json.dumps(val, default=str)
        if len(s) > P.MAX_BYTES:
            return dict(run_id=int(run_id), path=path, size=len(s),
                        keys={k: len(json.dumps(v, default=str)) for k, v in part.items()} if isinstance(part, dict) else None,
                        note="too large; ask for a sub-path")
        return dict(run_id=int(run_id), path=path, value=val)
    params, metrics = run.get("params") or {}, run.get("metrics") or {}
    head = {k: v for k, v in run.items() if k not in ("params", "metrics")}
    out = dict(run=P.record(head), params=P.plain(params, preview=False),
               targets=[P.record(r) for r in targets.to_dict("records")][:40])
    ms = json.dumps(metrics, default=str)
    if len(ms) <= P.MAX_BYTES // 2:
        out["metrics"] = P.plain(metrics, preview=False)
    else:
        out["metrics"] = dict(_size=len(ms), keys={k: len(json.dumps(v, default=str)) for k, v in metrics.items()},
                              note="pass path='metrics.<key>' for one part")
    return out


def get_predictions(conn, run_id, target=None, top=20, standings=False):
    """A run's per-athlete probabilities for one target (race), or its projected standings."""
    _guard(conn, "run_ids", int(run_id), f"model run {run_id}")
    if standings:
        return P.page(data.standings_predictions(conn, int(run_id)), limit=top)
    targets = data.run_race_targets(conn, int(run_id))
    if not len(targets):
        st = data.standings_predictions(conn, int(run_id))
        if len(st):
            return dict(note="this run has standings only", standings=P.page(st, limit=top))
        raise ValueError(f"run {run_id} has no predictions")
    if target is None:
        if len(targets) > 1:
            return dict(note="pick a target", targets=P.page(targets, limit=60))
        target = targets["target"].iloc[0]
    df = data.race_predictions(conn, int(run_id), target, limit=P.clamp(top, 20))
    t = targets[targets["target"] == target]
    return dict(target=P.record(t.iloc[0].to_dict()) if len(t) else dict(target=target), predictions=P.page(df, limit=top),
                sources=_sources(_run_competition(conn, run_id), run=int(run_id)))


def get_forecast(conn, competition=None, sport=None, category=None, top=10):
    """The live forecast (the run the web app shows): its next races' top probabilities and the championship."""
    comp = _sport_filter(conn, sport, competition)
    fc = _drop_hidden(conn, data.latest_forecasts(conn), competition="competitions")
    if comp:
        fc = fc[fc["competition"] == comp]
    if category:
        fc = fc[fc["category"] == category]
    if not len(fc):
        return dict(forecasts=[], note="no forecast run stored" + (f" for {comp}" if comp else ""))
    out = []
    for r in fc.to_dict("records"):
        rid = int(r["id"])
        targets = data.run_race_targets(conn, rid)
        upcoming = targets[targets["status"].fillna("scheduled") != "completed"] if len(targets) else targets
        races = []
        for t in upcoming.head(3).to_dict("records"):
            preds = data.race_predictions(conn, rid, t["target"], limit=P.clamp(top, 10))
            races.append(dict(target=t["target"], event=t.get("event_name"), start_date=P.plain(t.get("start_date")),
                              top=[P.record(x) for x in preds.to_dict("records")]))
        st = data.standings_predictions(conn, rid, limit=P.clamp(top, 10))
        out.append(dict(run=P.record(r), next_races=races, standings=[P.record(x) for x in st.to_dict("records")],
                        targets=int(len(targets)), sources=_sources(r.get("competition"), run=rid, as_of=r.get("created_at"))))
    return dict(forecasts=out)


# ---------------------------------------------------------------------------------------------------
# strategy research: Edge Finder, diagnostics, maker replay
# ---------------------------------------------------------------------------------------------------

def edge_finder(conn, year=2026, strategy=None, venue=None, limit=None, offset=0):
    """Every configuration with a full-season sweep of `year`, with the full-season recap of each strategy
    (P&L, volume, weekends up, drawdown, consistency) and its closing-line value (`clv`: metrics.clv's by_strategy
    row, None for a sweep saved before CLV was): what the Lab's Edge Finder shows, from saved runs only."""
    from racinglines.pipelines import sweep_settings as SS
    from racinglines.web import edge as E
    if venue and venue not in SS.VENUES:
        raise ValueError(f"venue is one of {', '.join(SS.VENUES)} (the exchanges a sweep trades)")
    cfgs = {k: c for k, c in E.configs(conn, int(year)).items()
            if not venue or (c["settings"]["venue"] or SS.DEFAULT_VENUE) == venue}
    strategies = [strategy] if strategy else E.STRATEGY_KEYS
    unknown = [s for s in strategies if s not in E.STRATEGY_LABEL]
    if unknown:
        raise ValueError(f"unknown strategy {unknown}; one of {E.STRATEGY_KEYS}")
    rows = []
    for c in cfgs.values():
        run = E._run(conn, c["run_id"])
        weekends = (run or {}).get("metrics", {}).get("weekends") or []
        for s in strategies:
            rc = E.recap(weekends, s)
            rows.append(dict(config=c["label"], settings_key=c["key"], run_id=c["run_id"],
                             venue=c["settings"]["venue"] or SS.DEFAULT_VENUE, strategy=s,
                             strategy_label=E.STRATEGY_LABEL[s], **{k: v for k, v in rc.items()},
                             clv=E.clv_of((run or {}).get("metrics"), s)))
    rows.sort(key=lambda r: -(r["pnl"] or 0))
    return dict(year=int(year), venue=venue, configurations=len(cfgs), strategies={k: E.STRATEGY_LABEL[k] for k in strategies},
                changed_settings={c["key"]: c["settings"].changed() for c in cfgs.values()},
                rows=P.page(rows, limit=limit, offset=offset))


def list_candidates(conn, limit=None, offset=0):
    """Named configurations x strategy saved in the Lab (searches add theirs; strategy profiles A and C are candidates)."""
    from racinglines.web import edge as E
    rows = [{k: v for k, v in c.items() if k != "settings"} | dict(settings=c["settings"]) for c in E.candidates(conn)]
    return P.page(rows, limit=limit, offset=offset)


def list_diagnostics(conn, limit=None, offset=0):
    """Events with as-of diagnostic runs (the maker replay's inputs): run ids per event and cutoffs."""
    runs = data.q(conn, """
        SELECT mr.params->>'event_key' AS event_key, count(*) AS runs, min(mr.id) AS first_run, max(mr.id) AS last_run,
               min(mr.params->>'cutoff') AS first_cutoff, max(mr.params->>'cutoff') AS last_cutoff,
               bool_or(mr.params ? 'sweep_stage') AS from_sweep
        FROM model_runs mr WHERE mr.kind = 'diagnostic' GROUP BY 1 ORDER BY max(mr.params->>'cutoff') DESC""")
    runs = _drop_hidden(conn, runs, event_key="event_keys")
    return P.page(runs, limit=limit, offset=offset, note="get_diagnostic(run_id) for one; replay_maker(run_id, ...) to re-quote it")


def _stored_exp_points(conn, run_id):
    """{athlete_id: expected points} from race_predictions.exp_points of one run ({} when none is stored)."""
    df = data.q(conn, """SELECT athlete_id, exp_points FROM race_predictions
                         WHERE model_run_id = :i AND exp_points IS NOT NULL""", i=int(run_id))
    return {int(a): float(v) for a, v in zip(df["athlete_id"], df["exp_points"])} if len(df) else {}


def _with_exp_points(preds, stored=None):
    """A diagnostic's predictions with each driver's expected points: the stored column, else the row's `extra`
    (exp_points); the column is left out when neither has any."""
    stored = stored or {}
    if not len(preds):
        return preds
    extra = preds["extra"] if "extra" in preds.columns else pd.Series([None] * len(preds), index=preds.index)
    vals = [stored.get(int(a)) if stored.get(int(a)) is not None else
            ((x or {}).get("exp_points") if isinstance(x, dict) else None)
            for a, x in zip(preds["athlete_id"], extra)]
    if all(v is None for v in vals):
        return preds
    return preds.assign(exp_points=[None if v is None else float(v) for v in vals])


def get_diagnostic(conn, run_id, top=25):
    """One as-of diagnostic run: our prices vs the exchange at the cutoff, scored per kind, and the result."""
    from racinglines.web import diag as D
    _guard(conn, "run_ids", int(run_id), f"diagnostic run {run_id}")
    d = D.load(conn, int(run_id))
    if d is None:
        raise ValueError(f"no diagnostic run {run_id}")
    mk = d["markets"]
    cols = ["kind", "subject", "outcome_label", "fair", "pm_cut", "pm_24h_before", "pm_move_24h", "edge", "paper_side", "result", "paper_pnl"]
    mk = mk[[c for c in cols if c in mk.columns]] if len(mk) else mk
    if len(mk):
        # edge is None (an object column) where a market is unpriced
        mk = mk.assign(_abs=pd.to_numeric(mk["edge"], errors="coerce").abs().fillna(-1)).sort_values("_abs", ascending=False).drop(columns="_abs")
    return dict(run=P.record({k: d["run"][k] for k in ("id", "kind", "created_at", "data_through") if k in d["run"]}),
                cutoff=P.plain(d["cutoff"]), event=P.record(d["event"]), race_id=d["race_id"],
                params=P.plain(d["params"], preview=False),
                scores=[P.record(r) for r in d["scores"].to_dict("records")], coherence=P.plain(d["coherence"], preview=False),
                predictions=P.page(_with_exp_points(d["preds"], _stored_exp_points(conn, int(run_id))).drop(
                    columns=["extra"], errors="ignore"), limit=top, note="by win probability"),
                markets=P.page(mk, limit=top, note="largest |edge| first"),
                result=P.page(d["result"], limit=top),
                paper_pnl=P.plain(mk["paper_pnl"].sum()) if len(mk) and "paper_pnl" in mk else None,
                sources=_sources(_run_competition(conn, run_id), ["polymarket"] if "pm_cut" in mk.columns else None,
                                 int(run_id), d["cutoff"]))


REPLAY_KNOBS = ("fill", "half_spread", "size", "max_pos", "max_capital", "skew", "max_disagree", "min_volume_24h", "pull_min")


def replay_maker(conn, run_id, fill="through", half_spread=0.02, size=50, max_pos=250, max_capital=1000, skew=1.0,
                 max_disagree=0.15, min_volume_24h=100, pull_min=15, exchange="polymarket", with_sweep=False, top=20):
    """Replay a maker quoting the exchange (polymarket, or kalshi with its maker fee) through an event's diagnostic
    runs against the real trade tape, with the diagnostic page's knobs. Synchronous, a few seconds; nothing is stored."""
    from racinglines.web import diag as D
    if fill not in ("through", "touch", "queue"):
        raise ValueError("fill is through, touch or queue")
    if exchange not in ("polymarket", "kalshi"):
        raise ValueError("exchange is polymarket or kalshi")
    knobs = dict(fill=fill, half_spread=float(half_spread), size=float(size), max_pos=float(max_pos), max_capital=float(max_capital),
                 skew=float(skew), max_disagree=float(max_disagree), min_volume_24h=float(min_volume_24h), pull_min=float(pull_min))
    _guard(conn, "run_ids", int(run_id), f"diagnostic run {run_id}")
    rp = D.replay(conn, int(run_id), with_sweep=bool(with_sweep), exchange=exchange, **knobs)
    out = dict(run_ids=list(rp["runs"]), exchange=exchange, knobs=knobs,
               summary=[P.record(r) for r in rp["summary"].to_dict("records")],
               by_stage=P.page(rp["stages"], limit=20) if isinstance(rp["stages"], pd.DataFrame) else P.plain(rp["stages"]),
               skips=[P.record(r) for r in rp["skips"].to_dict("records")],
               positions=P.page(rp["positions"], limit=top, note="worst P&L first"),
               fills=dict(count=int(len(rp["fills"])),
                          first=P.page(rp["fills"], limit=min(P.clamp(top), 20), note="first fills; the count is the total")))
    if with_sweep and rp["sweep"] is not None:
        out["sweep"] = P.page(rp["sweep"], limit=40)
    return out


# ---------------------------------------------------------------------------------------------------
# paper trading: positions, track record, signals; live events; change log
# ---------------------------------------------------------------------------------------------------

TRACK_RECORD_VENUES = ("polymarket", "kalshi", "private")


def _basic_viewer(viewer):
    """A basic caller (dict(id, username, role), server.caller()): its own account only, never which strategy made
    a pick (web/roles.basic_*). None for everyone else (stdio, admin, pro): output as before."""
    from racinglines.web import roles as R
    return viewer if viewer and R.canonical(viewer.get("role")) == "basic" else None


def track_record(conn, user, venue="polymarket", sport=None, viewer=None):
    """Every weekend of a user's paper record: strategy, trades or fills, positions, P&L. venue: polymarket, kalshi,
    private or all. 'all' lists one row per weekend AND venue (a `venue` column; the maker's weekends have a
    Polymarket and a Kalshi row) with `totals` per venue next to the grand total; `weekends` counts distinct weekends."""
    from racinglines.pipelines import sport_paper as SP
    from racinglines.pipelines import story as S
    from racinglines.web import roles as R
    if sport in HIDDEN.get():
        raise not_available(sport)
    basic = _basic_viewer(viewer)
    uid, user = _account_id(conn, basic["username"] if basic else user, viewer)
    shown = lambda rows: _drop_hidden(conn, rows, event_key="event_keys")      # noqa: E731
    full = lambda *a, **k: shown(S.track_record(*a, sport=sport, sports=SP.enabled(), **k))      # noqa: E731
    tr = (lambda *a, **k: [R.basic_row(r) for r in full(*a, **k)]) if basic else full
    if venue != "all":
        rows = tr(conn, uid, venue=venue)
        pnl = sum(r["pnl"] for r in rows)
        return dict(user=user, venue=venue, sport=sport, weekends=len(rows), pnl=P.plain(pnl), up=sum(1 for r in rows if r["pnl"] > 0),
                    rows=P.page([{k: v for k, v in r.items() if k != "date"} for r in rows], limit=P.MAX_LIMIT),
                    sources=_sources(sport, [venue], model=True))
    rows, totals = [], []
    for v in TRACK_RECORD_VENUES:
        part = tr(conn, uid, venue=v)
        rows += [dict(event_key=r["event_key"], venue=v, **{k: x for k, x in r.items() if k not in ("event_key", "date")})
                 for r in part]
        if part:
            totals.append(dict(venue=v, weekends=len(part), pnl=P.plain(sum(r["pnl"] for r in part)),
                               up=sum(1 for r in part if r["pnl"] > 0)))
    rows.sort(key=lambda r: (r["event_key"], TRACK_RECORD_VENUES.index(r["venue"])))
    pnl = sum(r["pnl"] for r in rows)
    return dict(user=user, venue=venue, sport=sport, weekends=len({r["event_key"] for r in rows}), pnl=P.plain(pnl),
                up=sum(1 for r in rows if r["pnl"] > 0), totals=totals, rows=P.page(rows, limit=P.MAX_LIMIT),
                sources=_sources(sport, [x["venue"] for x in totals], model=True))


def list_positions(conn, user, venue=None, event_key=None, open_only=False, limit=None, offset=0, viewer=None):
    basic = _basic_viewer(viewer)
    uid, user = _account_id(conn, basic["username"] if basic else user, viewer)
    df = data.q(conn, """
        SELECT pp.event_key, pp.venue, pp.kind, pp.subject, pp.market_key, pp.yes_shares, pp.no_shares, pp.cash, pp.mark, pp.outcome,
               pp.bid, pp.ask, pp.quote_state, pp.updated_at,
               pp.cash + pp.yes_shares * coalesce(pp.outcome::int, pp.mark, 0) + pp.no_shares * (1 - coalesce(pp.outcome::int, pp.mark, 0)) AS pnl
        FROM paper_positions pp WHERE pp.user_id = :u AND (CAST(:v AS text) IS NULL OR pp.venue = CAST(:v AS text))
          AND (CAST(:e AS text) IS NULL OR pp.event_key = CAST(:e AS text))
          AND (NOT :o OR pp.outcome IS NULL) ORDER BY pp.event_key DESC, pp.venue, pp.kind, pp.subject""",
        u=uid, v=venue, e=event_key, o=bool(open_only))
    df = _drop_hidden(conn, df, event_key="event_keys")
    totals = df.groupby("venue")["pnl"].agg(["sum", "count"]).reset_index() if len(df) else pd.DataFrame()
    return dict(user=user, totals=[P.record(r) for r in totals.to_dict("records")], positions=P.page(df, limit=limit, offset=offset))


def list_signals(conn, user=None, event_key=None, status=None, action=None, limit=None, offset=0, viewer=None):
    """Paper signals, newest first. A basic viewer: its own only, as roles.basic_signal shows them (no profile,
    strategy, member, fair value or edge; a `stars` rating instead)."""
    basic = _basic_viewer(viewer)
    scoped = _account_viewer(viewer)
    uid = _account_id(conn, basic["username"] if basic else user, viewer)[0] if scoped or user else None
    df = data.q(conn, """
        SELECT ss.id, u.username AS "user", ss.profile, ss.strategy, ss.event_key, ss.kind, ss.subject, ss.stage, ss.action, ss.side,
               ss.shares, ss.limit_price, ss.fair, ss.price, ss.edge, ss.heat, ss.target_cost, ss.status, ss.signal_ts, ss.market_key,
               ss.detail->>'venue' AS venue, ss.detail->>'backfill' AS backfill, ss.detail->>'member' AS _member
        FROM strategy_signals ss JOIN users u ON u.id = ss.user_id
        WHERE (CAST(:u AS int) IS NULL OR ss.user_id = CAST(:u AS int)) AND (CAST(:e AS text) IS NULL OR ss.event_key = CAST(:e AS text))
          AND (CAST(:s AS text) IS NULL OR ss.status = CAST(:s AS text)) AND (CAST(:a AS text) IS NULL OR ss.action = CAST(:a AS text))
        ORDER BY ss.id DESC""", u=uid, e=event_key, s=status, a=action)
    df = _drop_hidden(conn, df, event_key="event_keys")
    if basic:
        from racinglines.pipelines import profiles as PF
        from racinglines.web import roles as R
        prof = PF.of_user(conn, uid)
        recs = [R.basic_signal(dict(r, detail={"member": r.pop("_member")}), prof) for r in df.to_dict("records")]
        for r in recs:
            r.pop("detail", None)
        cols = [c for c in df.columns if c not in R.BASIC_HIDDEN and c != "_member"] + ["stars"]
        df = pd.DataFrame(recs, columns=cols)
    else:
        df = df.drop(columns="_member")
    return P.page(df, limit=limit, offset=offset)


def list_live_events(conn, limit=None, offset=0):
    """Settled live private-book events (the live_events table) and the run folders present on this machine."""
    from racinglines.pipelines import live as L
    df = data.q(conn, """SELECT run, sport, event_key, title, opened_at, settled_at, maker_pnl, crowd_pnl, taker_pnl, fills, volume
                         FROM live_events ORDER BY opened_at DESC NULLS LAST""")
    df = df[~df["sport"].isin(HIDDEN.get())] if len(df) else df
    try:
        folders = [dict(run=e["run"], sport=e["sport"], event_key=e["event_key"], title=e["title"],
                        updated=datetime.fromtimestamp(e["mtime"], tz=timezone.utc)) for e in L.events()]
        state = L.state()
    except Exception as ex:  # noqa: BLE001
        folders, state = [], f"unreadable: {ex}"
    folders = [f for f in folders if f["sport"] not in HIDDEN.get()]
    seen = sorted({*(df["sport"].dropna() if len(df) else ()), *(f["sport"] for f in folders if f.get("sport"))})
    return dict(state=state, settled=P.page(df, limit=limit, offset=offset), run_folders=[P.record(f) for f in folders][:20],
                sources=_sources(seen, model=True))


def pull_manifest(conn, sport, since=None):
    """C51 `racinglines data pull`: the files one sport's pull holds (db/pull.py: the registry, the sport's race data,
    model runs and exchange prices, trades and books), built on request; pull_chunk fetches each. since: a UTC date
    for a top-up. Never accounts, positions, signals, orders or jobs."""
    from racinglines import sports
    from racinglines.db import pull as PL
    if sport not in sports.SPORT_CODES:
        raise ValueError(f"unknown sport {sport!r}; one of {', '.join(sports.SPORT_CODES)}")
    if sport in HIDDEN.get():
        raise not_available(sport)
    return PL.manifest(conn, sport, since=since)


def pull_chunk(conn, pull_id, name, offset=0):
    """One chunk (base64, at most db/pull.py CHUNK bytes) of a file of a pull_manifest, from byte `offset`."""
    from racinglines.db import pull as PL
    sport = str(pull_id).split("-", 1)[0]
    if sport in HIDDEN.get():
        raise not_available(sport)
    return PL.chunk(conn, pull_id, name, offset=offset)


def storage_status(conn, refresh=False):
    """Where the data lives (racinglines/storage_status.py): the last snapshot of the VM's disk, data folders and
    databases and the data bucket's prefixes. refresh=True starts a new snapshot in the background (the bucket listing
    takes minutes); this call returns the last one either way. Sizes and counts only."""
    from racinglines import storage_status as ST
    started = ST.refresh_in_background(force=bool(refresh))
    snap = ST.read()
    return dict(snapshot=snap, age_seconds=ST.age_seconds(snap), refreshing=ST.refreshing() or started)


def data_changes(conn, sport=None, limit=None, offset=0):
    df = data.q(conn, """SELECT id, at, sport, kind, summary, by, detail FROM data_changes
                         WHERE (CAST(:s AS text) IS NULL OR sport = CAST(:s AS text)) ORDER BY at DESC, id DESC""", s=sport)
    return P.page(df, limit=limit, offset=offset)


# ---------------------------------------------------------------------------------------------------
# sql: read-only, capped
# ---------------------------------------------------------------------------------------------------

_SQL_FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|copy|vacuum|analyze|"
                            r"reindex|cluster|lock|listen|notify|set|reset|call|do|refresh|comment|security|pg_sleep|"
                            r"pg_read_file|pg_ls_dir|lo_import|lo_export|dblink)\b", re.I)
# functions that run a query given as text (which hides a table name in a string), read server files or settings,
# or reach other roles' secrets; and the catalogs holding role password hashes
_SQL_FORBIDDEN_FN = re.compile(r"\b(query_to_xml\w*|cursor_to_xml\w*|table_to_xml\w*|schema_to_xml\w*|database_to_xml\w*|"
                               r"ts_stat|ts_rewrite|set_config|pg_read_\w+|pg_stat_file|pg_ls_\w+|pg_file_\w+|lo_\w+|"
                               r"dblink\w*|pg_terminate_backend|pg_cancel_backend|pg_reload_conf|pg_rotate_logfile|"
                               r"pg_authid|pg_shadow|pg_user_mappings|pg_hba_file_rules)\b", re.I)


def _strip_comments(sql):
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


# one left-to-right pass, so a quote inside one kind of token never starts another: dollar-quoted, E'' (backslash
# escapes), plain '' and "identifier"
_SQL_TOKENS = re.compile(r"\$(\w*)\$.*?\$\1\$|(?<!\w)[eE]'(?:[^'\\]|\\.|'')*'|'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", re.S)


def _strip_literals(sql):
    """The query with its string constants blanked ('..', E'..', $tag$..$tag$), so a word inside a value (a variant
    named 'gridq+pretrain+reset') is not read as a keyword. Identifiers in double quotes are kept."""
    return _SQL_TOKENS.sub(lambda m: m.group(0) if m.group(0).startswith('"') else "''", sql)


def check_sql(query):
    """The query text a client may run: one SELECT (or WITH ... SELECT), no writes, no hidden tables.

    A first filter only, with friendly errors: the real boundary is the database role the sql tool connects as
    (RACINGLINES_MCP_SQL_URL, docs/mcp.md), which can read only the allowed tables."""
    q = _strip_comments(query or "").strip().rstrip(";").strip()
    if not q:
        raise ValueError("empty query")
    if ";" in q:
        raise ValueError("one statement only")
    if not re.match(r"(?is)^(select|with|table|values|explain)\b", q):
        raise ValueError("only SELECT (or WITH ... SELECT / EXPLAIN) queries")
    if re.search(r"\bu&['\"]", q, re.I):
        raise ValueError("Unicode-escaped names and strings (U&) are not accepted")
    code = _strip_literals(q)
    if _SQL_FORBIDDEN.search(code) or _SQL_FORBIDDEN_FN.search(code):
        raise ValueError("only read-only queries: no writes, DDL, settings, or server-side functions")
    for t in SQL_HIDDEN:
        if re.search(rf"\b{t}\b", q, re.I):
            raise ValueError(f"the {t} table is not readable through this tool")
    return q


def sql(conn, query, limit=None, offset=0):
    """Run a read-only query. The transaction is READ ONLY with a statement timeout, and the result is paged
    (`limit` rows, at most 500; an outer LIMIT/OFFSET is applied for you). Hidden: users, orders."""
    q = check_sql(query)
    lim, off = P.clamp(limit), max(0, int(offset or 0))
    conn.execute(text(f"SET LOCAL statement_timeout = '{SQL_TIMEOUT}'"))
    if q.lower().startswith("explain"):
        res = conn.execute(text(q))
        return dict(plan=[r[0] for r in res.fetchall()][:60])
    res = conn.execute(text(f"SELECT * FROM ({q}) AS _q LIMIT {lim + 1} OFFSET {off}"))
    cols = list(res.keys())
    fetched = res.fetchall()
    more = len(fetched) > lim
    rows = [dict(zip(cols, r)) for r in fetched[:lim]]
    total = None
    if not more and off == 0:
        total = len(rows)
    out = P.page(rows, limit=lim, offset=0, columns=cols, total=len(rows))
    out.update(offset=off, truncated=bool(more), next_offset=(off + lim) if more else None, total=total)
    out["note"] = (f"{len(rows)} rows shown, more exist; pass offset={off + lim} for the next page" if more else
                   f"{len(rows)} rows" + (f" from offset {off}" if off else "")) + ("; " + out["note"] if "cut to" in out["note"] else "")
    return out


# ---------------------------------------------------------------------------------------------------
# jobs: the Lab's simulations (a CLI subprocess per job, saved as a model run)
# ---------------------------------------------------------------------------------------------------

def list_job_types():
    from racinglines.pipelines import sweep_settings as SS
    from racinglines.web import jobs as J
    hidden = HIDDEN.get()
    out = []
    for jt in J.CATALOG.values():
        if jt.sport in hidden:
            continue
        knobs = []
        for k in jt.knobs:
            if k.type in ("sweep_settings", "sport_sweep_settings"):
                knobs.append(dict(name="settings", type="object", help="any sweep setting (see sweep_settings below); "
                                  "unset ones keep their defaults"))
            else:
                choices = [c for c in k.choices if not (k.name == "sport" and c in hidden)] if k.choices else None
                knobs.append(dict(name=k.name, type=k.type, default=k.default, min=k.min, max=k.max,
                                  choices=choices or None, help=k.help))
        out.append(dict(job_type=jt.code, sport=jt.sport, label=jt.label, what=jt.what, takes=jt.minutes, knobs=knobs))
    settings = [dict(name=s.name, group=s.group, type=s.type, default=(list(s.default) if isinstance(s.default, tuple) else s.default),
                     min=s.min, max=s.max, choices=list(s.choices) or None, help=s.help) for s in SS.SETTINGS]
    from racinglines.pipelines import season_sweep as SW
    sport_settings = {
        code: [dict(name=s.name, group=s.group, type=s.type,
                    default=list(s.default) if isinstance(s.default, tuple) else s.default,
                    min=s.min, max=s.max, choices=list(s.choices) or None, help=s.help)
               for s in SW.settings_class(code).SPEC]
        for code in J.modeled_sports() if code != "f1" and code not in hidden and SW.supports(code)}
    return dict(job_types=out, sweep_settings=settings, sport_sweep_settings=sport_settings, model_variants=J.MODEL_CHOICES,
                note="run_job(job_type, params) queues one; get_job(job_id) follows it; forecasts save as scenarios "
                     "(never the live prices); event is 'YYYY-R' (e.g. 2026-15), cutoff 'YYYY-MM-DDTHH:MM' UTC")


def _job_params(job_type, params):
    from racinglines.web import jobs as J
    jt = J.CATALOG.get(job_type)
    if jt is None:
        raise ValueError(f"unknown job type {job_type!r}; one of {list(J.CATALOG)}")
    params = dict(params or {})
    form = {}
    for k in jt.knobs:
        if k.type in ("sweep_settings", "sport_sweep_settings"):
            continue
        v = params.pop(k.name, None)
        if v is not None:
            form[k.name] = json.dumps(v) if isinstance(v, (list, dict)) else str(v).strip()   # combo legs as JSON
    settings = params.pop("settings", None)
    if params:
        raise ValueError(f"unknown knobs for {job_type}: {sorted(params)}")
    out = J.parse(jt, form)
    if any(k.type in ("sweep_settings", "sport_sweep_settings") for k in jt.knobs):
        from racinglines.pipelines import sweep_settings as SS
        from racinglines.pipelines import season_sweep as SW
        cls = SW.settings_class(jt.sport) if any(k.type == "sport_sweep_settings" for k in jt.knobs) else SS.Settings
        st = cls.from_dict({k: v for k, v in (settings or {}).items()})
        out["settings"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in st.changed().items()}
    return jt, out


def run_job(engine, job_type, params=None, user_id=None):
    """Queue a Lab job (filed under `user_id`: the account behind the request in --http mode, none over stdio).
    Returns its id; `get_job` follows it. The job runs once a worker (this server's, or the web app's) picks it up."""
    from racinglines.web import jobs as J
    jt, p = _job_params(job_type, params)
    hidden = HIDDEN.get()
    if jt.sport in hidden or jt.sport_of(p) in hidden:          # as the Lab's Run form (web/views.py lab_run)
        raise not_available(jt.sport if jt.sport in hidden else jt.sport_of(p))
    job_id = J.submit(jt, p, user_id, engine=engine)
    return dict(job_id=int(job_id), job_type=jt.code, params=p, status="queued",
                note=f"{jt.label}: {jt.minutes or 'a while'}. get_job({job_id}) for progress; the run it saves is result_run_id")


JOB_LOG_LINES_MAX = 500


def _job_row(r, log_lines):
    """A job row with the last `log_lines` lines of its log (at most JOB_LOG_LINES_MAX, whole: the log is the point,
    so it is not cut to the 20-item list preview the other fields get)."""
    log = r.pop("log", None) or ""
    lines = log.splitlines()
    n = min(int(log_lines or 0), JOB_LOG_LINES_MAX)
    r["log_lines"] = len(lines)
    out = P.record(r)
    out["log_tail"] = lines[-n:] if n else []
    return out


def get_job(conn, job_id, log_lines=20, viewer=None):
    scoped = _account_viewer(viewer)
    df = data.q(conn, """SELECT * FROM jobs WHERE id = :i
                        AND (CAST(:u AS int) IS NULL OR user_id = :u)""",
                i=int(job_id), u=scoped["id"] if scoped else None)
    if not len(df):
        raise ValueError(f"no job {job_id}")
    row = df.iloc[0].to_dict()
    if row.get("sport") in HIDDEN.get():
        raise not_available(f"job {job_id}")
    return _job_row(row, log_lines)


def list_jobs(conn, status=None, limit=None, offset=0, viewer=None):
    scoped = _account_viewer(viewer)
    df = data.q(conn, """SELECT j.id, j.kind, j.sport, j.status, j.progress, j.created_at, j.started_at, j.finished_at, j.result_run_id,
                                j.params, u.username AS "user" FROM jobs j LEFT JOIN users u ON u.id = j.user_id
                         WHERE (CAST(:s AS text) IS NULL OR j.status = CAST(:s AS text))
                         AND (CAST(:u AS int) IS NULL OR j.user_id = :u) ORDER BY j.id DESC""",
                s=status, u=scoped["id"] if scoped else None)
    df = df[~df["sport"].isin(HIDDEN.get())] if len(df) else df
    return P.page(df, limit=limit, offset=offset)


def cancel_job(engine, job_id, viewer=None):
    """Cancel a job that is still queued (a running subprocess is left to finish)."""
    with engine.begin() as c:
        get_job(c, job_id, log_lines=0, viewer=viewer)
        scoped = _account_viewer(viewer)
        n = c.execute(text("""UPDATE jobs SET status = 'failed', finished_at = now(), progress = 'cancelled before it started'
                              WHERE id = :i AND status = 'queued'
                              AND (CAST(:u AS int) IS NULL OR user_id = :u)"""),
                      dict(i=int(job_id), u=scoped["id"] if scoped else None)).rowcount
    return dict(job_id=int(job_id), cancelled=bool(n), note="" if n else "not queued (already running, done or failed)")


# ---------------------------------------------------------------------------------------------------
# sportsbook slips: map, price and settle a pasted book (racinglines/books/slips.py; nothing is stored)
# ---------------------------------------------------------------------------------------------------

def list_kinds(sport=None):
    """Every market kind in the registry (markets/kinds.toml, markets/kinds.py) with the sports that model it: what a
    book's `kind = ...` may name, and where price_book can expect a model price."""
    from racinglines import sports as SPS
    from racinglines.markets import kinds as K
    from racinglines.pipelines import coverage as C
    if sport in HIDDEN.get():
        raise not_available(sport)
    codes = [c for c in SPS.SPORT_CODES if c not in HIDDEN.get()]
    if sport and sport not in codes:
        raise ValueError(f"no sport {sport!r}; one of {', '.join(codes)}")
    rows = []
    for code, k in K.KINDS.items():
        modeled = [s for s in codes if C.modeled(s, code)]
        if sport and sport not in modeled:
            continue
        rows.append(dict(kind=code, label=k.label, payoff=k.payoff, subject=k.subject, session=k.session,
                         default=k.default, modeled_for=modeled))
    return dict(sport=sport, kinds=len(rows), rows=P.page(rows, limit=P.MAX_LIMIT),
                note="modeled_for: sports whose schema names a pricing model and lists the kind (coverage.modeled, the "
                     "parity counter's rule); price_book reads a leg's model price from the stored run, so a kind outside "
                     "this list prices only where a run stored it")


def _book(text_):
    from racinglines.books import slips as B
    if not (text_ or "").strip():
        raise ValueError("book: the book as TOML text (docs/sportsbook/slips.md)")
    try:
        return B.load_text(text_)
    except Exception as ex:  # noqa: BLE001  (tomllib and the validator: a bad book is the caller's error)
        raise ValueError(f"book: {ex}") from ex


def _runs(conn, run_ids):
    out = {}
    for rid in run_ids or ():
        r = data.model_run(conn, int(rid))
        if r is None:
            raise ValueError(f"no model run {rid}")
        _guard(conn, "competitions", r["competition"], f"model run {rid}")
        out[int(rid)] = r["competition"]
    return out


def map_book(conn, book):
    """Each line and leg of a book matched to an exact race and athlete in the database, or unmapped with the reason."""
    from racinglines.books import slips as B
    return P.plain(_guard_tree(conn, B.map_book(conn, _guard_tree(conn, _book(book)))), preview=False)


def price_book(conn, book, run_ids=None):
    """map_book plus, per leg and line: the book's own probability, the model's (a stored run), the linked prediction
    market's, and EV against each. No blend of the two (an owner decision); both are shown."""
    from racinglines.books import slips as B
    return P.plain(_guard_tree(conn, B.price_book(conn, _guard_tree(conn, _book(book)), runs=_runs(conn, run_ids))),
                   preview=False)


def settle_book(conn, book):
    """map_book plus each leg's result and each line's result, payout and profit from the stored results."""
    from racinglines.books import slips as B
    return P.plain(_guard_tree(conn, B.settle_book(conn, _guard_tree(conn, _book(book)))), preview=False)


# ---------------------------------------------------------------------------------------------------
# fantasy trading (F$ only: fantasy/exchange.py, fantasy/ledger.py; nothing reaches an exchange)
# ---------------------------------------------------------------------------------------------------
#
# Guardrails: every read runs in the server's READ ONLY transaction and only on the caller's own account (a hosted
# non-admin caller can't name another user; an admin or stdio may). The one write, fantasy_order, goes through
# fantasy/exchange.place, the web Trade page's path: the account's row lock, the balance check against the shared
# F$1,000, the per-order cap, the depth-10 walk, the rate limit, the audit row (tagged via=mcp). It always trades
# as the caller over HTTP, admin included: nobody places an order on someone else's account. Demo accounts and an
# account from another database (the shared-store viewer, id NO_ACCOUNT) are refused before anything is written.
# Recorded books and resolutions come from fantasy/exchange.books_engine (read-only).

FANTASY_HIDDEN = ("polymarket-takers",)             # the system account, as web/fantasy_routes.HIDDEN


def _fantasy_on():
    from racinglines.fantasy import ledger as FL
    if not FL.on():
        raise ValueError("fantasy trading is off on this server (RACINGLINES_FANTASY)")


def _fantasy_ready(conn):
    from racinglines.fantasy import ledger as FL
    from racinglines.web import accounts as ACC
    _fantasy_on()
    if not (ACC.ready(conn) and FL.ready(conn)):
        raise ValueError("fantasy trading isn't set up on this database yet (racinglines users setup)")


def _fantasy_account(conn, user, viewer):
    """(id, username) of the account a fantasy tool reads: the caller's own for a hosted non-admin, else `user`
    (an admin or stdio may name anyone; omitted: the caller)."""
    if viewer and user is None:
        user = viewer.get("username")
    uid, name = _account_id(conn, user, viewer)
    if uid < 1:
        raise ValueError("this server's accounts live in another database: no fantasy account here")
    return uid, name


def fantasy_balance(conn, user=None, viewer=None, books=None):
    """One account's F$: total (the 1,000 plus both P&Ls), cash it can trade with now (shared with its strategy),
    open hand trades at their mark, hand-trade P&L and the followed strategy's P&L."""
    from racinglines.fantasy import exchange as FX
    from racinglines.fantasy import ledger as FL
    _fantasy_ready(conn)
    uid, name = _fantasy_account(conn, user, viewer)
    q = FX.quotes(books or conn.engine, FL.open_tokens(conn, uid))
    return dict(user=name, **P.record(FL.summary(conn, uid, quotes=q)))


def fantasy_positions(conn, user=None, open_only=True, limit=None, offset=0, viewer=None, books=None):
    """One account's hand-trade positions: market, side, shares, cost, fees, worth now (the best recorded bid), P&L;
    with open_only=False also the settled and closed ones (outcome, payout)."""
    from racinglines.fantasy import exchange as FX
    from racinglines.fantasy import ledger as FL
    _fantasy_ready(conn)
    uid, name = _fantasy_account(conn, user, viewer)
    q = FX.quotes(books or conn.engine, FL.open_tokens(conn, uid))
    keep = ("market_link_id", "exchange", "question", "group_title", "event_title", "kind", "side", "shares", "cost",
            "fees", "value", "pnl", "status", "outcome", "payout", "settled_at", "updated_at")
    rows = [{k: r.get(k) for k in keep} for r in FL.positions(conn, uid, open_only=open_only, quotes=q)]
    return dict(user=name, positions=P.page(rows, limit=limit, offset=offset))


def fantasy_orders(conn, user=None, limit=None, offset=0, viewer=None):
    """One account's orders by hand, newest first, refused ones with their reason."""
    from racinglines.fantasy import exchange as FX
    from racinglines.fantasy import ledger as FL
    _fantasy_ready(conn)
    uid, name = _fantasy_account(conn, user, viewer)
    rows = FL.orders(conn, uid, limit=P.MAX_LIMIT)
    for r in rows:
        r["reason_text"] = FX.TEXT.get(r["reject_reason"]) if r["reject_reason"] else None
    return dict(user=name, orders=P.page(rows, limit=limit, offset=offset))


def fantasy_leaderboard(conn, limit=None, offset=0, books=None):
    """The leaderboard: score = ROI x (trades / 10)^0.3 (ROI = P&L over F$ staked), at least ledger.MIN_TRADES trades to
    rank. P&L and ROI only, never a player's balance (owner, 2026-10-12)."""
    from racinglines.fantasy import exchange as FX
    from racinglines.fantasy import ledger as FL
    from racinglines.web import demo
    _fantasy_ready(conn)
    board = FL.ranked(conn, hide=(*FANTASY_HIDDEN, *demo.DEMO_USERS), quotes=FX.quotes(books or conn.engine, FL.open_tokens(conn)))
    keep = ("rank", "username", "score", "roi", "trades", "pnl", "manual_pnl", "strategy_pnl")
    return dict(min_trades=FL.MIN_TRADES, rows=P.page([{k: r.get(k) for k in keep} for r in board], limit=limit, offset=offset))


def fantasy_markets(conn, q=None, limit=None, offset=0, books=None):
    """The markets a player may trade now (Formula 1 only for now, on Kalshi and Polymarket) with the best recorded YES bid and ask,
    when that book was recorded and when the market closes (closes_at: the deciding session's start, null for a season
    future or when no start is stored); markets with no book in the last day are left out. q: a search on the
    question, driver or event."""
    from racinglines.fantasy import exchange as FX
    _fantasy_on()
    rows = FX.markets(conn, (q or "").strip()[:60])
    quotes = FX.quotes(books or conn.engine, [m["token_id"] for m in rows])
    out = []
    for m in rows:
        b, a, ts = quotes.get(m["token_id"], (None, None, None))
        if ts is not None:
            out.append(dict(market_link_id=m["id"], exchange=m["exchange"], kind=m["kind"], question=m["question"],
                            subject=m["group_title"], event=m["event_title"], start_date=m["start_date"],
                            closes_at=m["closes_at"], bid=b, ask=a, book_ts=ts))
    return dict(markets=P.page(out, limit=limit, offset=offset), max_stake=FX.MAX_STAKE, max_age_min=FX.MAX_AGE_MIN)


def fantasy_book(conn, market_link_id, books=None):
    """One tradable market's latest recorded book: YES bids and asks and the NO side derived from them, up to the
    depth an order may walk (exchange.DEPTH = 10), with its time and the market's close (closes_at: null for a season
    future or when no session start is stored)."""
    from racinglines.fantasy import exchange as FX
    _fantasy_on()
    link = FX._link(conn, int(market_link_id))
    if link is not None and not FX.sport_ok(link):
        raise ValueError(FX.TEXT["sport_not_tradable"])
    if link is None or link["exchange"] not in FX.EXCHANGES:
        raise ValueError(f"no tradable market {market_link_id}")
    closes_at = FX.closes_for(conn, [link])[link["id"]]
    book = FX.latest_book(books or conn.engine, link["token_id"])
    if book is None:
        return dict(market_link_id=int(market_link_id), closes_at=P.plain(closes_at), book=None, note=FX.TEXT["book_unavailable"])
    nb, na = FX.yes_to_no(book["bids"], book["asks"])
    d = FX.DEPTH
    return dict(market_link_id=int(market_link_id), exchange=link["exchange"], kind=link["prediction"], ts=P.plain(book["ts"]),
                closes_at=P.plain(closes_at), yes_bids=book["bids"][:d], yes_asks=book["asks"][:d], no_bids=nb[:d], no_asks=na[:d])


def _fantasy_trader(conn, user, viewer):
    """The account an order or a quote is for (dict id, username, role, via=mcp): over HTTP the caller's own, admin
    included; over stdio (the owner) `user` names it. Demo accounts and a shared store's accounts are refused."""
    from racinglines.web import demo
    from racinglines.web import roles as R
    _fantasy_on()
    if viewer is not None:
        if user is not None and user != viewer.get("username"):
            raise ValueError("MCP orders are placed on your own account only")
        if viewer["id"] < 1:
            raise ValueError("this server's accounts live in another database: no fantasy account here")
    elif user is None:
        raise ValueError("which account? pass user (stdio is the owner's: name the account to trade as)")
    _fantasy_ready(conn)
    row = conn.execute(text("SELECT id, username, role, active FROM users WHERE "
                            + ("id = :i" if viewer is not None else "username = :n")),
                       dict(i=(viewer or {}).get("id"), n=user)).mappings().first()
    if row is None or not row["active"]:
        raise ValueError("no active account to trade as")
    who = dict(id=row["id"], username=row["username"], role=R.canonical(row["role"]), via="mcp")
    if demo.is_demo(who):
        raise ValueError("demo accounts look only; they can't trade")
    return who


def fantasy_quote(conn, market_link_id, side, action, shares, limit_price, user=None, viewer=None, books=None):
    """What fantasy_order would do with this order right now, writing nothing (fantasy/exchange.quote: the same
    checks, the same walk of the latest recorded book, or the venue's live one when that is over 15 minutes old, the
    same fee): the fills, the average price, the fee, what
    it would cost or bring, or the reason it would be refused. Read without the account's lock, so a quote that
    fills can still be refused a moment later."""
    from racinglines.fantasy import exchange as FX
    who = _fantasy_trader(conn, user, viewer)
    r = FX.quote(conn, who, market_link_id, side, action, shares, limit_price, books=books)
    r["fee"] = P.plain(r["fee"])
    return dict(user=who["username"], **r)


def fantasy_order(engine, market_link_id, side, action, shares, limit_price, client_key=None, user=None, viewer=None):
    """Place one immediate-or-cancel F$ order, as the web Trade page does (fantasy/exchange.place). Over HTTP it is
    always the caller's own account (an admin too); over stdio (the owner) `user` names the account. Every refusal
    comes back in the one shape (ok=false, reason, message), the rate limit included (it writes an audit row and no
    order row, as the Trade page's refusal does); a tool error is for a caller who can't trade at all."""
    from racinglines.fantasy import exchange as FX
    from racinglines.fantasy import ledger as FL
    with engine.connect() as c:
        who = _fantasy_trader(c, user, viewer)
    if not FX.rate_ok(who["id"]):
        with engine.begin() as c:
            FX._audit(c, who, "fantasy_order_rejected", reason="rate_limited", link=market_link_id)
            cash = FL.available(c, who["id"])
        return dict(user=who["username"], ok=False, order_id=None, status="rejected", reason="rate_limited",
                    message=FX.TEXT["rate_limited"], fills=[], fee=0.0, position=None, cash=round(cash, 2))
    FX.settle(engine, FX.books_engine(engine))                          # pay out anything resolved first
    r = FX.place(engine, who, market_link_id, side, action, shares, limit_price, client_key=client_key)
    with engine.connect() as c:
        cash = FL.available(c, who["id"])
    return dict(user=who["username"], ok=r["ok"], order_id=r["order_id"], status=r["status"], reason=r["reason"],
                message=r["message"] or None, fills=r["fills"], fee=P.plain(r["fee"]), position=r.get("position"),
                book_source=r.get("book_source"), cash=round(cash, 2))


# --- following a strategy (fantasy/follow.py, the path /strategy/pick saves through) ---------------------------------
#
# fantasy_strategy reads what an account follows, the five style cards and the race-weekend lock; fantasy_follow
# follows a style (or stops) on the caller's own account only, admin included, exactly as the page saves it: the same
# gates and words (demo accounts, F1 on Polymarket, the weekend lock, sizing, the codes allowed, roles.allowed_profile),
# the exchange rate limit, and a `strategy_pick` audit row tagged via=mcp. A refusal comes back as ok=false with its
# reason and the page's message, nothing written. Codes: the five alpha styles and "none"; an admin also the classic
# page's codes (follow.allowed_codes with alpha=True).

def _follower(conn, uid):
    """dict(id, username, role) for follow, with the role a web request runs as (web/layout.role); None if inactive."""
    from racinglines.web import layout as L
    from racinglines.web import roles as R
    row = conn.execute(text("SELECT id, username, role, active FROM users WHERE id = :i"), dict(i=uid)).mappings().first()
    if row is None or not row["active"]:
        return None
    return dict(id=row["id"], username=row["username"], role=L.role(R.canonical(row["role"])))


def fantasy_strategy(conn, user=None, viewer=None):
    """What one account follows (profile: base code, name, sizing, kelly, adjusted, rolled; None when nothing), the
    race-weekend lock (weekend, reopens, message; None between weekends), the five style cards (options) and every
    code it may pass to fantasy_follow, with the sizing choices."""
    from racinglines.fantasy import follow as FW
    from racinglines.pipelines import profiles as PF
    _fantasy_ready(conn)
    uid, name = _fantasy_account(conn, user, viewer)
    who = _follower(conn, uid)
    if who is None:
        raise ValueError("no active account")
    codes = FW.allowed_codes(who, alpha=True)
    return dict(user=name, **FW.current(conn, who), options=FW.options(who),
                codes=[*PF.ALPHA_ORDER, *sorted(codes - set(PF.ALPHA_ORDER)), "none"],
                sizing=dict(choices=list(FW.SIZINGS), kelly_lo=FW.KELLY["lo"], kelly_hi=FW.KELLY["hi"],
                            kelly_step=FW.KELLY["step"], kelly_default=FW.KELLY["default"]))


def fantasy_follow(engine, code, sizing="standard", kelly=0.25, min_edge=None, min_volume_24h=None, rolled=False,
                   user=None, viewer=None):
    """Follow a strategy, or stop with code "none", as /strategy/pick saves it (fantasy/follow.follow). Over HTTP it is
    always the caller's own account (an admin too); over stdio (the owner) `user` names the account."""
    from racinglines.fantasy import exchange as FX
    from racinglines.fantasy import follow as FW
    from racinglines.pipelines import profiles as PF
    _fantasy_on()
    if viewer is not None:
        if user is not None and user != viewer.get("username"):
            raise ValueError("MCP strategy changes are made on your own account only")
        if viewer["id"] < 1:
            raise ValueError("this server's accounts live in another database: no fantasy account here")
    elif user is None:
        raise ValueError("which account? pass user (stdio is the owner's: name the account to follow for)")
    with engine.connect() as c:
        _fantasy_ready(c)
        uid = viewer["id"] if viewer is not None else _user_id(c, user)
        who = _follower(c, uid)
    if who is None:
        raise ValueError("no active account to follow for")
    who["via"] = "mcp"
    out = dict(user=who["username"], ok=False, reason=None, message=None, profile=None)
    if not FX.rate_ok(who["id"]):
        return dict(out, reason="rate_limited", message=FX.TEXT["rate_limited"])
    tweaks = {k: v for k, v in (("min_edge", min_edge), ("min_volume_24h", min_volume_24h)) if v is not None}
    try:
        with engine.begin() as c:
            old = PF.of_user(c, who["id"])
            new = FW.follow(c, who, code, sizing=sizing, kelly=kelly, tweaks=tweaks, rolled=bool(rolled), alpha=True)
            FX._audit(c, who, "strategy_pick", old=(old or {}).get("name"), new=(new or {}).get("name"),
                      adjusted=(new or {}).get("adjusted"), rolled=bool((new or {}).get("rolled")))
    except FW.FollowError as ex:
        return dict(out, reason=ex.reason, message=ex.message)
    return dict(out, ok=True, message=FW.saved_message(new), profile=FW.summary(new))
