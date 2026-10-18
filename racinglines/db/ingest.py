"""
Shared database helpers for ingesters: idempotent upserts, seeding the sports /
leagues / competitions registry, and resolving venues and athletes.
"""

import re
import unicodedata

import pandas as pd
from sqlalchemy import insert, select

from racinglines.db import models as m
from racinglines.db import registry



# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def _upsert(session, model, lookup, **values):
    obj = session.scalars(select(model).filter_by(**lookup)).first()
    if obj is None:
        obj = model(**lookup, **values)
        session.add(obj)
        session.flush()
    else:
        for k, v in values.items():
            setattr(obj, k, v)
    return obj


def seed(session):
    """Upsert the sports, leagues, competitions, categories and venues from
    registry.py. Safe to run any number of times."""
    sports = {code: _upsert(session, m.Sport, dict(code=code), name=name, result_kind=kind)
              for code, (name, kind) in registry.SPORTS.items()}
    leagues = {code: _upsert(session, m.League, dict(code=code), name=name, organizer=org)
               for code, (name, org) in registry.LEAGUES.items()}
    for code, spec in registry.COMPETITIONS.items():
        comp = _upsert(session, m.Competition, dict(code=code), name=spec["name"],
                       league_id=leagues[spec["league"]].id, sport_id=sports[spec["sport"]].id)
        for cat_code, (name, gender, age) in spec["categories"].items():
            _upsert(session, m.Category, dict(competition_id=comp.id, code=cat_code),
                    name=name, gender=gender, age_group=age)
    venues = {slug: _upsert(session, m.Venue, dict(slug=slug), name=name, country=country)
              for slug, (name, country) in registry.VENUES.items()}
    for alias, slug in registry.VENUE_ALIASES.items():
        _upsert(session, m.VenueAlias, dict(alias=alias), venue_id=venues[slug].id)
    session.flush()


def seed_sport(session, sport):
    """Upsert one sport's rows (sport, league, competition, categories) from its schema (sports/<sport>.toml).
    The same upserts as `seed`, for the one sport a sync is about to file links under. Idempotent."""
    from racinglines import sports as S
    schema = S.load(sport)
    scode, lcode, ccode = schema["sport"]["code"], schema["league"]["code"], schema["competition"]["code"]
    name, kind = registry.SPORTS[scode]
    sp = _upsert(session, m.Sport, dict(code=scode), name=name, result_kind=kind)
    lname, org = registry.LEAGUES[lcode]
    lg = _upsert(session, m.League, dict(code=lcode), name=lname, organizer=org)
    spec = registry.COMPETITIONS[ccode]
    comp = _upsert(session, m.Competition, dict(code=ccode), name=spec["name"], league_id=lg.id, sport_id=sp.id)
    for cat_code, (cname, gender, age) in spec["categories"].items():
        _upsert(session, m.Category, dict(competition_id=comp.id, code=cat_code), name=cname, gender=gender, age_group=age)
    session.flush()
    return comp


def ensure_competition(session, sport):
    """(Competition, its first Category) of a sport, as a market sync files links under. When the database has
    not been seeded for the sport (a fresh deploy: migrations create tables, not rows) the rows are created from
    the sport's schema first, so a sync never stops on a missing competition."""
    from racinglines import sports as S
    schema = S.load(sport)
    cat_code = next(iter(schema["competition"]["categories"]))
    comp = session.scalars(select(m.Competition).filter_by(code=schema["competition"]["code"])).one_or_none()
    cat = session.scalars(select(m.Category).filter_by(competition_id=comp.id, code=cat_code)).one_or_none() if comp else None
    if comp is None or cat is None:
        comp = seed_sport(session, sport)
        cat = session.scalars(select(m.Category).filter_by(competition_id=comp.id, code=cat_code)).one()
    return comp, cat


def _slugify(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def resolve_venue(session, raw_slug, country=None):
    slug = _slugify(raw_slug)
    alias = session.get(m.VenueAlias, slug)
    if alias:
        return session.get(m.Venue, alias.venue_id)
    venue = session.scalars(select(m.Venue).filter_by(slug=slug)).first()
    if venue is None:  # unseen venue: create it; add it to registry.VENUES for a proper name
        venue = _upsert(session, m.Venue, dict(slug=slug), name=slug.replace("-", " ").title(), country=country)
    return venue


def _identifiers(session, scheme, values):
    return dict(session.execute(
        select(m.AthleteIdentifier.value, m.AthleteIdentifier.athlete_id)
        .where(m.AthleteIdentifier.scheme == scheme, m.AthleteIdentifier.value.in_(list(values)))
    ).all())


def resolve_athletes(session, riders, conflicts=None):
    """riders: DataFrame with rider_id ("name:<key>"), rider_name, nation, and optionally uci_id.
    Returns {rider_id: athlete_id}, creating athletes that don't exist yet.

    With a uci_id, the UCI identifier is matched first, then the name key; a rider matched by
    name gains its UCI identifier. When a UCI ID and a name key point to different athletes
    (a name change, e.g. WILLIAMS Robert Jordan / WILLIAMS Jordan), the UCI ID wins and the pair
    (uci athlete, name athlete) is appended to `conflicts` for a merge (db merge-athletes)."""
    riders = riders.drop_duplicates("rider_id")
    keys = {rid: rid.split(":", 1)[1] for rid in riders["rider_id"]}
    found = _identifiers(session, "name", keys.values())
    out = {rid: found[key] for rid, key in keys.items() if key in found}
    uci = {}
    if "uci_id" in riders:
        uci = {r.rider_id: str(r.uci_id).strip() for r in riders.itertuples()
               if _none(r.uci_id) is not None and str(r.uci_id).strip()}
        by_uci = _identifiers(session, "uci", uci.values())
        for rid, u in uci.items():
            if u in by_uci:
                if rid in out and out[rid] != by_uci[u] and conflicts is not None:
                    conflicts.append((by_uci[u], out[rid]))
                out[rid] = by_uci[u]
        known = set(by_uci)
        gain = [dict(scheme="uci", value=u, athlete_id=out[rid]) for rid, u in uci.items()
                if rid in out and u not in known]
        # a new spelling matched by UCI ID: its name key now points to the same athlete
        gain += [dict(scheme="name", value=keys[rid], athlete_id=out[rid]) for rid in uci
                 if rid in out and keys[rid] not in found]
        if gain:
            session.execute(insert(m.AthleteIdentifier), gain)
    new = riders[~riders["rider_id"].isin(out)]
    if len(new):
        ids = session.scalars(
            insert(m.Athlete).returning(m.Athlete.id, sort_by_parameter_order=True),
            [dict(display_name=r.rider_name, nation=_none(r.nation)) for r in new.itertuples()],
        ).all()
        session.execute(insert(m.AthleteIdentifier), [
            dict(scheme="name", value=keys[rid], athlete_id=aid) for rid, aid in zip(new["rider_id"], ids)]
            + [dict(scheme="uci", value=uci[rid], athlete_id=aid) for rid, aid in zip(new["rider_id"], ids)
               if rid in uci])
        out.update(zip(new["rider_id"], ids))
    return out


def merge_athletes(session, keep, drop, dry_run=False):
    """Move everything that refers to athlete `drop` to athlete `keep` (results, identifiers,
    predictions, standings, market links, ...), then delete `drop`. Every table with a foreign
    key to athletes.id is found from the models, so new ones are covered. Refuses, changing
    nothing, when a row would collide with one `keep` already has (e.g. both in the same round).
    Returns {table: rows moved}."""
    from sqlalchemy import text
    if keep == drop or session.get(m.Athlete, keep) is None or session.get(m.Athlete, drop) is None:
        raise ValueError(f"need two different existing athletes, got {keep} and {drop}")
    refs = [(t, c) for t in m.Base.metadata.sorted_tables for c in t.columns
            for fk in c.foreign_keys if fk.column.table.name == "athletes"]
    clashes, moved = [], {}
    for t, c in refs:
        n = session.execute(text(f'SELECT count(*) FROM "{t.name}" WHERE "{c.name}" = :d'), dict(d=drop)).scalar()
        moved[t.name] = n
        keys = [list(u.columns) for u in t.constraints if hasattr(u, "columns") and c in list(u.columns)
                and u.__class__.__name__ in ("UniqueConstraint", "PrimaryKeyConstraint") and len(u.columns) > 1]
        for cols in keys:
            others = [x.name for x in cols if x is not c]
            on = " AND ".join(f'a."{o}" IS NOT DISTINCT FROM b."{o}"' for o in others)
            k = session.execute(text(f'SELECT count(*) FROM "{t.name}" a JOIN "{t.name}" b ON {on} '
                                     f'WHERE a."{c.name}" = :d AND b."{c.name}" = :k'), dict(d=drop, k=keep)).scalar()
            if k:
                clashes.append(f"{t.name}: {k} row(s) on ({', '.join(others)})")
    if clashes:
        raise ValueError("can't merge, rows would collide: " + "; ".join(clashes))
    if not dry_run:
        for t, c in refs:
            session.execute(text(f'UPDATE "{t.name}" SET "{c.name}" = :k WHERE "{c.name}" = :d'), dict(k=keep, d=drop))
        session.delete(session.get(m.Athlete, drop))
        session.flush()
    return {k: v for k, v in moved.items() if v}


def _none(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else v


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

