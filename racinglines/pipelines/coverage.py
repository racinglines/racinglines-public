"""
The coverage counter (`racinglines backtest coverage`, docs/parity-rebuild.md "C0"): read-only. For every sport x
exchange x market kind x season, how many races are linked, settled and taped (prices, trades, books, counted in
Postgres and the Parquet archive through markets/store.py), the tier that gives (the parity bar), and whether the
kind is modeled; season futures in their own table; and per sport, the depth of race results in the database and the
results sources its schema declares. Everything it knows about a sport comes from sports/<code>.toml. It writes
nothing to the database.
"""

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from racinglines import sports
from racinglines.markets import kinds as K
from racinglines.markets import store

BAR = 8                    # races per season for a parity tier (docs/parity-rebuild.md)
STORES = ("prices", "trades", "books")

_LINKS = """
SELECT ml.token_id, ml.exchange, co.code AS competition, ml.race_id, ml.closed,
       ml.prediction = 'unmodeled' AS unmodeled, ml.resolved_yes IS NOT NULL AS settled,
       CASE WHEN ml.prediction = 'unmodeled' THEN coalesce(nullif(ml.params->>'kind', ''), 'unmodeled')
            ELSE ml.prediction END AS kind,
       CAST(coalesce(extract(year FROM e.start_date), extract(year FROM ml.end_date)) AS int) AS season
FROM market_links ml
JOIN competitions co ON co.id = ml.competition_id
LEFT JOIN races r ON r.id = ml.race_id
LEFT JOIN events e ON e.id = r.event_id
"""

_EVENTS = """
SELECT co.code AS competition, s.year AS season, e.id AS event_id,
       EXISTS (SELECT 1 FROM races ra JOIN rounds ro ON ro.race_id = ra.id JOIN results rs ON rs.round_id = ro.id
               WHERE ra.event_id = e.id) AS has_results
FROM events e JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
WHERE e.status <> 'cancelled'
"""


def _sport_of():
    """competition code -> sport code, from the schemas."""
    return {sports.load(s)["competition"]["code"]: s for s in sports.SPORT_CODES}


def _kind_lists(schema):
    """Every kind a schema names in a `kinds` or `*_kinds` list, at any depth ([markets], [replay], [live.markets])."""
    out = set()
    for k, v in schema.items():
        if isinstance(v, dict):
            out |= _kind_lists(v)
        elif (k == "kinds" or k.endswith("_kinds")) and isinstance(v, list):
            out |= {x for x in v if isinstance(x, str)}
    return out


def modeled(sport, kind):
    """A kind is modeled when markets/kinds.py knows it, the sport's schema names a pricing_model, and the schema lists
    the kind among those it prices or trades."""
    if sport not in sports.SPORT_CODES or kind not in K.KINDS:
        return False
    schema = sports.load(sport)
    return bool(schema["sport"].get("pricing_model")) and kind in _kind_lists(schema)


def tier(per_season):
    """The parity bar over a combo's usable races per season: parity-2 at BAR or more in two seasons, parity-1 in one,
    thin with any race, else missing."""
    full = sum(n >= BAR for n in per_season)
    return "parity-2" if full >= 2 else "parity-1" if full == 1 else "thin" if any(n > 0 for n in per_season) else "missing"


def _taped(conn, links, root, wait=True):
    """Per store, the set of linked tokens with at least one row in Postgres or the Parquet archive."""
    tokens = sorted(set(links["token_id"]))
    return {name: {t for t, (n, _) in store.counts(conn, name, tokens, root=root, wait=wait).items() if n > 0}
            for name in STORES}


def markets(conn, root=None, seasons=None):
    """(race combos, season futures) as DataFrames. root: one Parquet tree to read (default every exchange's)."""
    links = pd.read_sql(text(_LINKS), conn)
    sport_of = _sport_of()
    links["sport"] = links["competition"].map(sport_of).fillna(links["competition"])
    if seasons:
        links = links[links["season"].isin(list(seasons))]
    taped = _taped(conn, links, root)
    for name in STORES:
        links[name] = links["token_id"].isin(taped[name])
    keys = ["sport", "exchange", "kind", "season"]

    race = links[links["race_id"].notna()]
    rows = []
    for key, g in race.groupby(keys, sort=True):
        def races(mask, g=g):
            return int(g.loc[mask, "race_id"].nunique())
        rows.append(dict(zip(keys, key), links=len(g), races=int(g["race_id"].nunique()),
                         settled_races=races(g["settled"]), price_races=races(g["prices"]),
                         trade_races=races(g["trades"]), book_races=races(g["books"]),
                         unmodeled_links=int(g["unmodeled"].sum()), modeled="yes" if modeled(key[0], key[2]) else "no"))
    cols = keys + ["links", "races", "settled_races", "price_races", "trade_races", "book_races", "tier", "maker_tier",
                   "modeled", "unmodeled_links"]
    combos = pd.DataFrame(rows).reindex(columns=cols)
    if len(combos):
        combo = [combos[k] for k in keys[:3]]                  # one combo = sport x exchange x kind, over its seasons
        for col, tape in (("tier", "price_races"), ("maker_tier", "book_races")):
            usable = combos[["settled_races", tape]].min(axis=1)
            combos[col] = usable.groupby(combo).transform(lambda s: tier(list(s)))

    fut = links[links["race_id"].isna()]
    frows = []
    for key, g in fut.groupby(keys, sort=True, dropna=False):
        frows.append(dict(zip(keys, key), links=len(g), open=int((~g["closed"]).sum()), settled=int(g["settled"].sum()),
                          price_links=int(g["prices"].sum()), trade_links=int(g["trades"].sum()),
                          book_links=int(g["books"].sum()), modeled="yes" if modeled(key[0], key[2]) else "no"))
    futures = pd.DataFrame(frows, columns=keys + ["links", "open", "settled", "price_links", "trade_links", "book_links",
                                                  "modeled"])
    return combos, futures


def results(conn, seasons=None):
    """Per sport (every schema): the depth of race results in the database and the results sources it declares."""
    ev = pd.read_sql(text(_EVENTS), conn)
    if seasons:
        ev = ev[ev["season"].isin(list(seasons))]
    rows = []
    for code in sports.SPORT_CODES:
        schema = sports.load(code)
        g = ev[ev["competition"] == schema["competition"]["code"]]
        done = g[g["has_results"]]
        res = schema.get("results")
        source = next((schema[s]["source"] for s in ("results", "replay", "model") if schema.get(s, {}).get("source")), "")
        fallbacks = [f"{f['source']} ({f.get('quality', '?')})" for f in (res or {}).get("fallbacks", []) if f.get("allow")]
        rows.append(dict(sport=code, first_season=int(done["season"].min()) if len(done) else None,
                         last_season=int(done["season"].max()) if len(done) else None, events=len(g),
                         events_with_results=len(done), results_source=source, fallbacks=", ".join(fallbacks),
                         source_modules=", ".join(m.rsplit(".", 1)[-1] for m in schema["sport"].get("results_modules", [])),
                         # cleared: the schema declares where its results come from ([results], or the history source
                         # its model and replay read) or a pricing model already trains on them (downhill); a schema
                         # that does neither (IndyCar, on purpose) is not cleared to ingest
                         cleared="yes" if res is not None or source or schema["sport"].get("pricing_model") else "no"))
    return pd.DataFrame(rows).astype({"first_season": "Int64", "last_season": "Int64"})


def where(conn, root=None):
    """What this run counted: the database, the archive trees and their Parquet row counts, and Postgres rows per
    store. On staging this says whether staging's own database and archive hold any tape."""
    db = conn.execute(text("SELECT current_database()")).scalar()
    trees = [root] if root is not None else store._roots()
    out = dict(database=db, data=os.environ.get("RACINGLINES_DATA") or str(store.paths.DATA), trees=[str(t) for t in trees])
    for name, s in store.STORES.items():
        files = [f for t in trees for f in (Path(t) / name).rglob("*.parquet")]
        out[name] = dict(postgres=conn.execute(text(f"SELECT count(*) FROM {s['table']}")).scalar(),
                         parquet_files=len(files),
                         parquet=sum(store.pq.read_metadata(f).num_rows for f in files))
    return out


def run(conn, root=None, seasons=None):
    combos, futures = markets(conn, root, seasons)
    return dict(where=where(conn, root), combos=combos, futures=futures, results=results(conn, seasons))


def format_text(res):
    w = res["where"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 2000)
    lines = [f"database {w['database']} · data {w['data']} · archive trees: {', '.join(w['trees']) or 'none'}"]
    for name in store.STORES:
        s = w[name]
        lines.append(f"  {name:<6} postgres {s['postgres']:>12,} rows · parquet {s['parquet']:>12,} rows "
                     f"in {s['parquet_files']:,} files")
    if not any(w[n]["postgres"] or w[n]["parquet"] for n in store.STORES):
        lines.append("  no market tape here: every price/trade/book count below is 0")
    c = res["combos"]
    lines += ["", (f"=== Race markets (tier at {BAR}+ races a season with both a settlement and a price tape; "
                   "maker_tier needs a book tape) ===")]
    lines.append(c.to_string(index=False) if len(c) else "no race markets linked")
    if len(c):
        t = (c.drop_duplicates(["sport", "exchange", "kind"]).groupby("tier").size()
             .reindex(["parity-2", "parity-1", "thin", "missing"], fill_value=0))
        lines.append("combos per tier: " + ", ".join(f"{k} {v}" for k, v in t.items()))
    f = res["futures"]
    lines += ["", "=== Season futures (no race) ===", f.to_string(index=False) if len(f) else "none linked"]
    lines += ["", "=== Results depth and sources (per sport) ===", res["results"].to_string(index=False),
              "", ("More results deepen the models' training (walk-forward burn-in); P&L backtests also need market "
                   "tape, which starts when recording began.")]
    return "\n".join(lines)


def write(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("combos", "futures", "results"):
        res[name].to_csv(out / f"coverage_{name}.csv", index=False)
    return out
