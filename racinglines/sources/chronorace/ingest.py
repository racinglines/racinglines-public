"""
Load parsed result files into the database.

Each downloaded file (data/raw/mtb_dh/chronorace/*.md) is one event x category,
which becomes one Race. Ingest is idempotent:

- A file whose content hash hasn't changed since the last ingest is skipped
  (unless force=True).
- Otherwise the race's rounds, results and splits are deleted and re-inserted
  from the file, so re-downloading a file and re-ingesting it just works.

Athletes are matched through AthleteIdentifier(scheme="uci", value=<UCI ID>) first,
when the file has a UCI ID column (downloads since 2026-09-28), then
AthleteIdentifier(scheme="name", value=<name key>), where the key is
parser.normalize_rider_id without its "name:" prefix. A UCI ID and a name key that
point to different athletes are reported for `racinglines db merge-athletes`.
"""


import hashlib
import re
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert, select

from racinglines import paths
from racinglines.db import models as m
from racinglines.db import registry
from racinglines.core.stats import to_ms as _ms
from racinglines.db.ingest import _none, _upsert, resolve_athletes, resolve_venue, seed  # noqa: F401  (seed re-exported)

SOURCE = "chronorace"
PARSER = "chronorace_md"

def ingest_file(session, path, competition="uci_dhi_wc", force=False):
    """Ingest one downloaded event file. Returns a short status string."""
    from racinglines.sources.chronorace.parse import is_markdown_tables_file, parse_markdown_tables_file
    from racinglines.models.inputs.run_times import event_format

    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    key = paths.rel(path)                       # relative to data/: stable across machines and moves
    src = session.scalars(select(m.SourceFile).filter_by(path=key)).first()
    if src and src.sha256 == sha and not force:
        return "unchanged"
    if not is_markdown_tables_file(path):
        return "skipped (not a results-table file)"

    df = pd.DataFrame(parse_markdown_tables_file(path))
    if df.empty:
        _upsert(session, m.SourceFile, dict(path=key), sha256=sha, parser=PARSER, race_id=None)
        return "no timing data (PDF-only)"

    comp = session.scalars(select(m.Competition).filter_by(code=competition)).one()
    first = df.iloc[0]
    category = session.scalars(select(m.Category).filter_by(competition_id=comp.id, code=first["category"])).first()
    if category is None:
        category = _upsert(session, m.Category, dict(competition_id=comp.id, code=first["category"]),
                           name=first["category"])
    event_date = date.fromisoformat(first["event_date"])
    has_final = bool(((df["round"] == "final") & (df["status"] == "OK")).any())
    season = _upsert(session, m.Season, dict(competition_id=comp.id, year=event_date.year))
    title = first.get("event_name") or ""
    country = title.rsplit(",", 1)[-1].strip() if re.search(r",\s*[A-Z]{3}\s*$", title) else None
    venue = resolve_venue(session, first["venue"], country)
    event = _upsert(session, m.Event, dict(season_id=season.id, source=SOURCE, source_key=first["source_key"]),
                    name=title, start_date=event_date, venue_id=venue.id,
                    series_round=None if pd.isna(first["series_round"]) else int(first["series_round"]),
                    status="completed" if has_final else "in_progress")
    race = _upsert(session, m.Race, dict(event_id=event.id, category_id=category.id))
    race.format = event_format(df, df["event_id"].iloc[0]) if has_final else None

    # replace this race's rounds (results and splits cascade)
    session.execute(delete(m.Round).where(m.Round.race_id == race.id))
    round_ids = {}
    for kind in df["round"].unique():
        rnd = m.Round(race_id=race.id, kind=kind, ordinal=registry.ROUND_ORDER.get(kind, 9), name=kind)
        session.add(rnd)
        session.flush()
        round_ids[kind] = rnd.id

    conflicts = []
    athletes = resolve_athletes(session, df[["rider_id", "rider_name", "nation"] + (["uci_id"] if "uci_id" in df else [])],
                                conflicts)

    fin = df[df["sector_id"] == "FINISH"].drop_duplicates(["round", "rider_id"])
    result_rows = [dict(
        round_id=round_ids[r.round], athlete_id=athletes[r.rider_id],
        position=None if pd.isna(r.rank_at_split) else int(r.rank_at_split),
        status=r.status, time_ms=_ms(r.cum_time_s),
        bib=_none(str(r.bib)) if _none(r.bib) is not None else None,
        team=_none(r.team), nation=_none(r.nation),
    ) for r in fin.itertuples()]
    result_ids = session.scalars(
        insert(m.Result).returning(m.Result.id, sort_by_parameter_order=True), result_rows).all()
    rid_by_key = dict(zip(zip(fin["round"], fin["rider_id"]), result_ids))

    sp = df[(df["sector_id"] != "FINISH") & df["cum_time_s"].notna()]
    split_rows = [dict(
        result_id=rid_by_key[(r.round, r.rider_id)], idx=int(r.sector_id[1:]), cum_time_ms=_ms(r.cum_time_s),
        rank=None if pd.isna(r.rank_at_split) else int(r.rank_at_split),
    ) for r in sp.itertuples() if (r.round, r.rider_id) in rid_by_key]
    if split_rows:
        session.execute(insert(m.Split), split_rows)

    _upsert(session, m.SourceFile, dict(path=key), sha256=sha, parser=PARSER, race_id=race.id)
    merges = "".join(f"; same UCI ID, two athletes: merge {drop} into {keep}" for keep, drop in sorted(set(conflicts)))
    return f"{len(result_rows)} results, {len(split_rows)} splits{merges}"


def ingest_paths(session, paths, competition="uci_dhi_wc", force=False, echo=print):
    """Ingest files and/or directories (non-recursive, *.md). Commits per file."""
    seed(session)
    session.commit()
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.md")) if p.is_dir() else [p]
    counts = {}
    for f in files:
        try:
            status = ingest_file(session, f, competition=competition, force=force)
            session.commit()
        except Exception as e:  # keep going; report at the end
            session.rollback()
            status = f"ERROR: {e}"
        kind = "ingested" if status[0].isdigit() else status.split(" ")[0].rstrip(":")
        counts[kind] = counts.get(kind, 0) + 1
        echo(f"  {f.name}: {status}")
    return counts
