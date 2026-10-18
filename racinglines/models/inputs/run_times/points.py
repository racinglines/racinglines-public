"""
Championship points tables for UCI downhill, by season (docs/todo.md, Points validation).

Two sources:
  schema   sports/mtb_dh.toml [points]: one PLACEHOLDER table for every season (today's default)
  db       the points_schemes table: one row per competition, era and round kind, entered by the
           owner from the official UCI scales (`racinglines data points mtb_dh import`); seasons with no
           row fall back to the schema

    schemes = load(conn)                  # {season: Scheme} from points_schemes
    with use(scheme_for(2026, schemes)):  # model.FINAL_POINTS / QUAL_POINTS / QUAL_POINTS_ROUND set
        pts = actual_event_points(target) # ... and restored afterwards

`reconcile` compares cumulative points with an official standings file, rider by rider.
"""

import tomllib
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import text

from racinglines.models.inputs.run_times import model as M

QUAL_ROUNDS = ("qual", "qual1", "qual2", "semi")


@dataclass(frozen=True)
class Scheme:
    final: tuple
    qual: tuple
    qual_round: dict = field(default_factory=dict)     # weekend format -> round the qualifying points are paid on
    official: bool = False
    source: str = "schema"


def schema_scheme():
    """Today's tables (the schema's placeholders, or whatever the model module holds now)."""
    return Scheme(tuple(M.FINAL_POINTS), tuple(M.QUAL_POINTS), dict(M.QUAL_POINTS_ROUND))


@contextmanager
def use(scheme):
    """The model's points tables set to `scheme` for the duration (None = unchanged)."""
    if scheme is None:
        yield schema_scheme()
        return
    old = (M.FINAL_POINTS, M.QUAL_POINTS, M.QUAL_POINTS_ROUND)
    M.FINAL_POINTS, M.QUAL_POINTS = list(scheme.final), list(scheme.qual)
    M.QUAL_POINTS_ROUND = {**M.QUAL_POINTS_ROUND, **scheme.qual_round}
    try:
        yield scheme
    finally:
        M.FINAL_POINTS, M.QUAL_POINTS, M.QUAL_POINTS_ROUND = old


def from_rows(rows, seasons):
    """{season: Scheme} from points_schemes rows (dicts with season_from, season_to, round_kind,
    points, is_official) for each season in `seasons`. A season needs a final row; its qualifying
    row (qual / qual1 / semi) gives the qualifying points. Which round pays them in each weekend
    format stays the schema's `qual_round` (juniors ride a single qualifier in every era). Seasons
    without a final row are left out."""
    out = {}
    for season in seasons:
        era = [r for r in rows if r["season_from"] <= season and (r["season_to"] is None or season <= r["season_to"])]
        final = [r for r in era if r["round_kind"] == "final"]
        if not final:
            continue
        quals = [r for r in era if r["round_kind"] in QUAL_ROUNDS]
        qual = quals[0] if quals else None
        out[season] = Scheme(final=tuple(final[0]["points"]), qual=tuple(qual["points"]) if qual else (),
                             official=all(r["is_official"] for r in [final[0]] + quals[:1]),
                             source="db")
    return out


def load(conn, competition="uci_dhi_wc", seasons=range(2010, 2031)):
    rows = conn.execute(text("""SELECT ps.season_from, ps.season_to, ps.round_kind, ps.points, ps.is_official
                                FROM points_schemes ps JOIN competitions c ON c.id = ps.competition_id
                                WHERE c.code = :c ORDER BY ps.season_from, ps.id"""), dict(c=competition))
    return from_rows([dict(r._mapping) for r in rows], seasons)


def scheme_for(season, schemes):
    """The season's scheme, or the schema's placeholders when there's none."""
    return schemes.get(int(season)) or schema_scheme()


def read_file(path):
    """Tables the owner enters (TOML): one [[scheme]] per era and round kind:

        [[scheme]]
        season_from = 2025
        season_to = 2026            # optional: omit while current
        round_kind = "final"        # final | qual | qual1 | semi
        points = [250, 210, ...]    # points[0] = 1st place
        official = true
        note = "UCI MTB regulations Part 4, 2026 edition, table ..."
    """
    with open(path, "rb") as f:
        rows = tomllib.load(f).get("scheme", [])
    for r in rows:
        if r["round_kind"] not in ("final",) + QUAL_ROUNDS:
            raise ValueError(f"unknown round_kind {r['round_kind']!r}")
        if any(p < 0 for p in r["points"]) or list(r["points"]) != sorted(r["points"], reverse=True):
            raise ValueError(f"{r['season_from']} {r['round_kind']}: points must be non-negative and non-increasing")
    return rows


def import_rows(session, rows, competition="uci_dhi_wc"):
    """Replace the competition's points_schemes rows for the same (season_from, round_kind)."""
    from racinglines.db.models import Competition, PointsScheme
    comp = session.query(Competition).filter_by(code=competition).one()
    n = 0
    for r in rows:
        session.query(PointsScheme).filter_by(competition_id=comp.id, season_from=r["season_from"],
                                              round_kind=r["round_kind"]).delete()
        session.add(PointsScheme(competition_id=comp.id, season_from=r["season_from"], season_to=r.get("season_to"),
                                 round_kind=r["round_kind"], points=list(r["points"]),
                                 is_official=bool(r.get("official", False)), note=r.get("note")))
        n += 1
    session.commit()
    return n


# --- reconciliation with official standings --------------------------------------------------------

def read_standings(path):
    """Official standings as DataFrame(rider, points), from a CSV (columns rider, points) or a TOML
    file entered by hand (committed ones are TOML; git ignores CSVs):

        source = "UCI DHI World Cup standings after round 7, 2026 (ChronoRace PDF)"
        [points]
        "Loïc Bruni" = 1234
    """
    path = str(path)
    if path.endswith(".toml"):
        with open(path, "rb") as f:
            pts = tomllib.load(f)["points"]
        return pd.DataFrame(dict(rider=list(pts), points=[float(v) for v in pts.values()]))
    return pd.read_csv(path)[["rider", "points"]]


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(s.replace("-", " ").split()))


def through_event(target, series_round):
    """The event_id that "after round N" means in one season (`target`), for reconcile's `through`. Round numbers
    come from ChronoRace's titles (`DHI #n`) and are unique only within a season, and not always there: 2022
    numbers Leogang and Lenzerheide both #3, 2025 Lake Placid and Mont-Sainte-Anne both #9. So this is the last
    event, by date, that carries the number: standings "after round 3" of 2022 include both #3 weekends. Every
    season reuses 1, 2, 3...: callers pass one season's rows, never several."""
    ev = target.drop_duplicates("event_id").set_index("event_id")["series_round"]
    hits = [e for e in M.event_order(target) if ev.get(e) == series_round]
    if not hits:
        raise ValueError(f"no event numbered round {series_round} in this season")
    return hits[-1]


def reconcile(target, official, through=None):
    """Cumulative points per rider from the results (`target`: one season and category) against an
    official standings table (columns `rider` and `points`, optionally `uci_id`), after the events
    up to `through` (an event_id; default every event in `target`). One row per rider in either,
    with `diff` = computed - official; every row should be 0."""
    order = M.event_order(target)
    if through is not None:
        order = order[:order.index(through) + 1]
    pts = M.actual_event_points(target[target["event_id"].isin(order)])
    names = target.drop_duplicates("rider_id").set_index("rider_id")["rider_name"]
    ours = pts.groupby("rider_id")["points"].sum().rename("computed").reset_index()
    ours["rider"] = ours["rider_id"].map(names)
    ours["key"] = ours["rider"].map(_norm)
    off = official.rename(columns={"points": "official"}).assign(key=lambda d: d["rider"].map(_norm))
    m = ours.merge(off[["key", "official"]], on="key", how="outer")
    m["rider"] = m["rider"].fillna(m["key"])
    m[["computed", "official"]] = m[["computed", "official"]].fillna(0.0)
    m["diff"] = m["computed"] - m["official"]
    return m[["rider", "computed", "official", "diff"]].sort_values(["official", "computed"], ascending=False,
                                                                     ignore_index=True)
