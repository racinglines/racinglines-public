"""
Load data/raw/f1/fastf1/<year>/<round>_<session>.* (from racinglines/sources/fastf1/fetch.py) into the database.

    event    one Grand Prix        source "f1timing", source_key "2024-08"
    race     event x category DRV  format = {"kind": "f1", "sprint": bool, "total_laps": n}
    rounds   qual | sprint | race  extra = weather, total laps, session date
    results  one per driver        position, status (OK/DNF/DSQ/DNS), time_ms, bib = car number,
                                   extra = grid, points, laps, status text, Q1-Q3, team id
    laps     every lap             lap/sector ms, speed traps, tyre, pit, track status
    track_profiles                 per-event features from qualifying laps + race results

Athletes are matched on AthleteIdentifier(scheme="f1", value=<Ergast driverId>),
e.g. "max_verstappen". Idempotent like the DH ingest: an event's rounds are
replaced when its files change (hash of all its files).
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import insert, select

from racinglines.db import models as m
from racinglines.db import registry
from racinglines.db.ingest import _upsert, resolve_venue, seed

ROOT = Path(__file__).resolve().parents[3]      # the repository
from racinglines import paths  # noqa: E402
from racinglines.core.stats import to_ms as _ms

DATA = paths.F1_RAW
COMPETITION = "f1_wdc"
CATEGORY = "DRV"
SOURCE = "f1timing"
ROUND_KIND = {"FP1": "fp1", "FP2": "fp2", "FP3": "fp3", "SQ": "sprint_qual", "Q": "qual", "S": "sprint", "R": "race"}
ROUND_ORDINAL = {"fp1": -3, "fp2": -2, "fp3": -1, "sprint_qual": 1, "qual": 2, "sprint": 4, "race": 5}
PRACTICE = ("fp1", "fp2", "fp3")
SPRINT_FORMATS = ("sprint", "sprint_shootout", "sprint_qualifying")


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _status(row, kind):
    """(position, status) from a FastF1 results row."""
    if kind in PRACTICE:
        return _int(row.get("Position")), "OK"
    if kind in ("qual", "sprint_qual"):
        best = [row.get(c) for c in ("Q1", "Q2", "Q3") if pd.notna(row.get(c))]
        return _int(row.get("Position")), ("OK" if best else "DNS")
    cls = str(row.get("ClassifiedPosition") or "")
    if cls.isdigit():
        return int(cls), "OK"
    status = {"R": "DNF", "N": "DNF", "D": "DSQ", "E": "DSQ", "W": "DNS", "F": "DNS"}.get(cls, "DNF")
    return _int(row.get("Position")), status


def event_files(year, rnd):
    return sorted((DATA / str(year)).glob(f"{rnd:02d}_*"))


Q3_CARS = 10        # qualifying knockout: the last segment's field; Q1 and Q2 each drop half of the rest


def knockout_from_laps(res, laps):
    """Qualifying classification from the laps when the results carry none (FastF1 right after the session, before
    the official classification is out: no Position, no Q1-Q3). Each driver's segment is read from when they last
    crossed the line (Q3_CARS latest are Q3, then the Q2 and Q1 eliminations by count); the segments' best valid laps
    become Q1-Q3 and order each group. Returns res with Position and Q1-Q3 filled, or res unchanged when it has
    positions or the laps have no session times."""
    if res["Position"].notna().any() or "Time" not in laps or laps["Time"].isna().all():
        return res
    num = laps["DriverNumber"].astype(str)
    last = laps.groupby(num)["Time"].max()
    order = list(last.sort_values(ascending=False).index)
    n = len(res)
    q2_out = max(0, (n - Q3_CARS) // 2)
    q3, q2 = order[:Q3_CARS], order[Q3_CARS:Q3_CARS + q2_out]
    q2_end = last[q2].max() if q2 else None
    q1_end = last[order[Q3_CARS + q2_out:]].max() if len(order) > Q3_CARS + q2_out else None
    ok = laps["LapTime"].notna() & ~laps.get("Deleted", pd.Series(False, index=laps.index)).fillna(False).astype(bool)
    seg = pd.Series(3, index=laps.index)
    if q2_end is not None:
        seg[laps["Time"] <= q2_end] = 2
    if q1_end is not None:
        seg[laps["Time"] <= q1_end] = 1
    best = {k: laps[ok & (seg == k)].groupby(num[ok & (seg == k)])["LapTime"].min() for k in (1, 2, 3)}
    res = res.copy()
    key = res["DriverNumber"].astype(str)
    for k in (1, 2, 3):
        res[f"Q{k}"] = key.map(best[k])
    group = key.map({d: 0 for d in q3} | {d: 1 for d in q2}).fillna(2)
    seg_best = [res["Q3"], res["Q2"], res["Q1"]]
    own = pd.Series([seg_best[int(g)].iloc[i] for i, g in enumerate(group)], index=res.index)
    rank = pd.DataFrame(dict(g=group, t=own.fillna(float("inf")), q2=res["Q2"].fillna(float("inf")),
                             q1=res["Q1"].fillna(float("inf")))).sort_values(["g", "t", "q2", "q1"])
    res.loc[rank.index, "Position"] = range(1, len(rank) + 1)
    return res


def track_profile(q_laps, race_results, meta_q, meta_r, venue_slug):
    """Track features from each driver's fastest clean qualifying lap and the race classification."""
    f = {"venue": venue_slug, "street": venue_slug in registry.STREET_CIRCUITS}
    laps = q_laps.copy()
    if len(laps):
        ok = laps["LapTime"].notna() & laps[["Sector1Time", "Sector2Time", "Sector3Time"]].notna().all(axis=1)
        if "Deleted" in laps:
            ok &= ~laps["Deleted"].astype("boolean").fillna(False).astype(bool)
        best = laps[ok].sort_values("LapTime").groupby("Driver").head(1)
        if len(best) >= 8:
            lap = best["LapTime"].median()
            secs = [best[f"Sector{k}Time"].median() for k in (1, 2, 3)]
            shares = np.array(secs) / sum(secs)
            speeds = [best[c].median() for c in ("SpeedI1", "SpeedI2", "SpeedFL")]
            f.update(lap_s=float(lap), s1_share=float(shares[0]), s2_share=float(shares[1]), s3_share=float(shares[2]),
                     v_i1=_num(speeds[0]), v_i2=_num(speeds[1]), v_fl=_num(speeds[2]),
                     v_st=_num(best["SpeedST"].median()),
                     # time-weighted trap speed: how much of the lap is spent fast
                     speed_index=float(np.nansum(shares * np.array(speeds, dtype=float))),
                     pole_gap_p10=float(best["LapTime"].nsmallest(10).iloc[-1] / best["LapTime"].min() - 1)
                     if len(best) >= 10 else None)
    if race_results is not None and len(race_results):
        rr = race_results[race_results["ClassifiedPosition"].astype(str).str.isdigit()
                          & (race_results["GridPosition"] > 0)]
        if len(rr) >= 8:
            fin = rr["ClassifiedPosition"].astype(int)
            f["mean_places_gained"] = float((rr["GridPosition"] - fin).abs().mean())
            f["grid_finish_rank_corr"] = float(rr["GridPosition"].rank().corr(fin.rank()))
    for label, meta in (("qual", meta_q), ("race", meta_r)):
        w = (meta or {}).get("weather") or {}
        if w:
            f[f"{label}_rain_share"] = w.get("rain_share")
            f[f"{label}_track_temp"] = w.get("track_temp")
    return f


def ingest_event(session, year, rnd, force=False):
    files = event_files(year, rnd)
    metas = {p.name.split("_")[1].split(".")[0]: json.loads(p.read_text()) for p in files if p.suffix == ".json"}
    if not metas:
        return "no sessions"
    # a qualifying stored with its order read from the laps (knockout_from_laps) is a different reading of the same
    # files: the marker makes a weekend stored before that reading existed ingest once more
    q_res = DATA / str(year) / f"{rnd:02d}_Q.results.parquet"
    from_laps = b"q-order-from-laps" if "Q" in metas and q_res.exists() and pd.read_parquet(q_res)["Position"].isna().all() else b""
    sha = hashlib.sha256(b"".join(p.read_bytes() for p in files) + from_laps).hexdigest()
    key = f"f1:{year}-{rnd:02d}"
    src = session.scalars(select(m.SourceFile).filter_by(path=key)).first()
    if src and src.sha256 == sha and not force:
        return "unchanged"

    # a weekend is stored as each session ends (FP1 first), not only once qualifying or the race is in
    meta = metas.get("R") or metas.get("Q") or next(metas[c] for c in ROUND_KIND if c in metas)
    comp = session.scalars(select(m.Competition).filter_by(code=COMPETITION)).one()
    cat = session.scalars(select(m.Category).filter_by(competition_id=comp.id, code=CATEGORY)).one()
    season = _upsert(session, m.Season, dict(competition_id=comp.id, year=year))
    venue = resolve_venue(session, meta.get("location") or meta["event_name"])
    has_race = "R" in metas
    event = _upsert(session, m.Event, dict(season_id=season.id, source=SOURCE, source_key=f"{year}-{rnd:02d}"),
                    name=meta.get("official_name") or meta["event_name"], start_date=date.fromisoformat(meta["event_date"]),
                    venue_id=venue.id, series_round=rnd, status="completed" if has_race else "in_progress")
    race = _upsert(session, m.Race, dict(event_id=event.id, category_id=cat.id))
    race.format = dict(kind="f1", sprint="S" in metas or "SQ" in metas or meta.get("event_format") in SPRINT_FORMATS, total_laps=(metas.get("R") or {}).get("total_laps"),
                       event_format=meta.get("event_format"), event_name=meta.get("event_name"))
    # the sessions still to come keep their scheduled rounds (db/schedule.py), so their markets keep a close
    from racinglines.db import schedule as SCH
    SCH.clear_for_ingest(session, race.id, {ROUND_KIND[c] for c in metas if c in ROUND_KIND}, final=has_race)
    session.flush()

    n_results = n_laps = 0
    frames = {}
    # some sessions (e.g. sprint qualifying) come without driver/team ids: fill them from the
    # weekend's qualifying or race classification by car number
    ref = {}
    # classification by car number; before this weekend's qualifying, the season's earlier rounds fill the gaps
    for path in [DATA / str(year) / f"{rnd:02d}_{code}.results.parquet" for code in ("R", "Q") if code in metas] + \
            [p for r_ in range(rnd - 1, 0, -1) for p in (DATA / str(year) / f"{r_:02d}_R.results.parquet",
                                                          DATA / str(year) / f"{r_:02d}_Q.results.parquet")]:
        if not path.exists():
            continue
        for r in pd.read_parquet(path).itertuples():
            if pd.notna(r.DriverId) and str(r.DriverId).strip():
                ref.setdefault(str(r.DriverNumber), dict(DriverId=r.DriverId, TeamId=r.TeamId, CountryCode=r.CountryCode))
    for code, kind in ROUND_KIND.items():
        if code not in metas:
            continue
        base = DATA / str(year) / f"{rnd:02d}_{code}"
        res = pd.read_parquet(base.with_suffix(".results.parquet"))
        for col in ("DriverId", "TeamId", "CountryCode"):
            blank = res[col].isna() | (res[col].astype(str).str.strip() == "")
            res.loc[blank, col] = res.loc[blank, "DriverNumber"].astype(str).map(lambda n: ref.get(n, {}).get(col))
        res = res[res["DriverId"].notna() & (res["DriverId"].astype(str).str.strip() != "")]
        laps = pd.read_parquet(base.with_suffix(".laps.parquet"))
        if code == "Q":
            res = knockout_from_laps(res, laps)
        frames[code] = (res, laps)
        mt = metas[code]
        rnd_row = m.Round(race_id=race.id, kind=kind, ordinal=ROUND_ORDINAL[kind], name=kind,
                          extra=dict(weather=mt.get("weather"), total_laps=mt.get("total_laps"),
                                     session_date=mt.get("session_date")))
        session.add(rnd_row)
        session.flush()

        # athletes by Ergast driverId
        ids = [str(v) for v in res["DriverId"].dropna()]
        found = dict(session.execute(select(m.AthleteIdentifier.value, m.AthleteIdentifier.athlete_id).where(
            m.AthleteIdentifier.scheme == "f1", m.AthleteIdentifier.value.in_(ids))).all())
        for r in res.itertuples():
            did = str(r.DriverId)
            if did not in found:
                a = m.Athlete(display_name=str(r.FullName), nation=str(r.CountryCode)[:3] if pd.notna(r.CountryCode) else None)
                session.add(a)
                session.flush()
                session.add(m.AthleteIdentifier(scheme="f1", value=did, athlete_id=a.id))
                found[did] = a.id

        winner_time = None
        if kind in ("race", "sprint"):
            wt = res.loc[res["ClassifiedPosition"].astype(str) == "1", "Time"]
            winner_time = float(wt.iloc[0]) if len(wt) and pd.notna(wt.iloc[0]) else None
        rows = []
        for _, r in res.iterrows():
            pos, status = _status(r, kind)
            if kind in PRACTICE:
                t = None
            elif kind in ("qual", "sprint_qual"):
                qs = [r.get(c) for c in ("Q1", "Q2", "Q3") if pd.notna(r.get(c))]
                t = min(qs) if qs else None
            elif str(r.get("ClassifiedPosition")) == "1":
                t = winner_time
            else:
                t = winner_time + float(r["Time"]) if winner_time is not None and pd.notna(r.get("Time")) else None
            rows.append(dict(
                round_id=rnd_row.id, athlete_id=found[str(r["DriverId"])], position=pos, status=status, time_ms=_ms(t),
                bib=str(r["DriverNumber"]), team=str(r["TeamName"]) if pd.notna(r["TeamName"]) else None,
                nation=str(r["CountryCode"])[:3] if pd.notna(r["CountryCode"]) else None,
                extra={k: v for k, v in dict(
                    grid=_int(r.get("GridPosition")), points=_num(r.get("Points")), laps=_num(r.get("Laps")),
                    status_text=str(r.get("Status")) if pd.notna(r.get("Status")) else None,
                    classified=str(r.get("ClassifiedPosition")) if pd.notna(r.get("ClassifiedPosition")) else None,
                    team_id=str(r.get("TeamId")) if pd.notna(r.get("TeamId")) and str(r.get("TeamId")).lower() != "nan" else None,
                    q1_ms=_ms(r.get("Q1")), q2_ms=_ms(r.get("Q2")), q3_ms=_ms(r.get("Q3")),
                    abbreviation=str(r.get("Abbreviation"))).items() if v is not None}))
        rows = list({row["athlete_id"]: row for row in rows}.values())
        result_ids = session.scalars(insert(m.Result).returning(m.Result.id, sort_by_parameter_order=True), rows).all()
        by_number = {row["bib"]: rid for row, rid in zip(rows, result_ids)}
        n_results += len(rows)

        lap_rows = []
        for r in laps.itertuples(index=False):
            rid = by_number.get(str(r.DriverNumber))
            if rid is None or pd.isna(r.LapNumber):
                continue
            lap_rows.append(dict(
                result_id=rid, lap=int(r.LapNumber), lap_time_ms=_ms(r.LapTime), s1_ms=_ms(r.Sector1Time),
                s2_ms=_ms(r.Sector2Time), s3_ms=_ms(r.Sector3Time), speed_i1=_num(r.SpeedI1), speed_i2=_num(r.SpeedI2),
                speed_fl=_num(r.SpeedFL), speed_st=_num(r.SpeedST),
                compound=str(r.Compound) if pd.notna(r.Compound) else None, tyre_life=_num(r.TyreLife),
                stint=_int(r.Stint), pit_in=bool(pd.notna(r.PitInTime)), pit_out=bool(pd.notna(r.PitOutTime)),
                track_status=str(r.TrackStatus) if pd.notna(r.TrackStatus) else None, position=_int(r.Position),
                is_accurate=bool(r.IsAccurate) if pd.notna(r.IsAccurate) else None,
                deleted=bool(r.Deleted) if pd.notna(r.Deleted) else None))
        lap_rows = list({(x["result_id"], x["lap"]): x for x in lap_rows}.values())
        if lap_rows:
            session.execute(insert(m.Lap), lap_rows)
        n_laps += len(lap_rows)

    q_laps = frames["Q"][1] if "Q" in frames else pd.DataFrame()
    r_res = frames["R"][0] if "R" in frames else None
    feats = track_profile(q_laps, r_res, metas.get("Q"), metas.get("R"), venue.slug)
    _upsert(session, m.TrackProfile, dict(event_id=event.id), venue_id=venue.id, features=feats)
    _upsert(session, m.SourceFile, dict(path=key), sha256=sha, parser="f1_fastf1", race_id=race.id)
    return f"{n_results} results, {n_laps} laps, profile speed_index={feats.get('speed_index', float('nan')):.0f}"


def ingest(session, years, force=False, echo=print):
    seed(session)
    session.commit()
    counts = {}
    for year in years:
        rounds = sorted({int(p.name[:2]) for p in (DATA / str(year)).glob("[0-9][0-9]_*.meta.json")})
        for rnd in rounds:
            try:
                status = ingest_event(session, year, rnd, force=force)
                session.commit()
            except Exception as e:
                session.rollback()
                status = f"ERROR {type(e).__name__}: {e}"
            k = "ingested" if status[0].isdigit() else status.split(" ")[0]
            counts[k] = counts.get(k, 0) + 1
            echo(f"  {year} R{rnd:02d}: {status}")
    return counts


# the event schedule's session names (fetch.SCHEDULE_NAME) -> session codes
SCHEDULE_CODE = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Sprint Qualifying": "SQ",
                 "Sprint Shootout": "SQ", "Qualifying": "Q", "Sprint": "S", "Race": "R"}


def schedule_sessions(event):
    """One schedule row's sessions as scheduled rounds (db/schedule.write_sessions): kind, ordinal, name and the UTC
    start in the ingest's session_date format ("YYYY-MM-DD HH:MM:SS"). A session with no UTC start is left out."""
    out = []
    for i in range(1, 6):
        code = SCHEDULE_CODE.get(str(event.get(f"Session{i}")))
        t = event.get(f"Session{i}DateUtc")
        if code is None or t is None or pd.isna(t):
            continue
        kind = ROUND_KIND[code]
        out.append(dict(kind=kind, ordinal=ROUND_ORDINAL[kind], name=kind,
                        start=str(pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t))))
    return out


def write_schedule(session, schedule, year, through, today, echo=print):
    """Scheduled rounds (db/schedule.py) for the weekends in FastF1's event schedule (fetch.event_schedule) dated up to
    `through` that are stored as events and not completed. An event whose stored date differs from the schedule's
    race day is skipped (a calendar change the forecast has not filed yet). Returns per-outcome counts."""
    from racinglines.db import schedule as SCH
    counts = {}
    for ev in schedule.to_dict("records"):
        rnd = int(ev["RoundNumber"])
        day = pd.Timestamp(ev["EventDate"]).date()
        if not SCH.in_window(day, today, through):
            continue
        key = f"{year}-{rnd:02d}"
        event, race = SCH.upcoming_race(session, SOURCE, key)
        if event is None:
            status = "no stored event"
        elif race is None:
            status = "completed, untouched" if event.status == "completed" else "no race row"
        elif event.start_date != day:
            status = f"skipped: stored date {event.start_date} is not the schedule's {day}"
        else:
            sess = schedule_sessions(ev)
            n = SCH.write_sessions(session, race.id, sess, "session_date", "fastf1 event schedule")
            status = ", ".join(f"{v} {k}" for k, v in n.items() if v) or "no sessions dated"
            for k, v in n.items():
                counts[k] = counts.get(k, 0) + v
            status += "  [" + "; ".join(f"{s['kind']} {s['start']}" for s in sess) + "]"
        echo(f"  {key} {ev.get('EventName')}: {status}")
    return counts
