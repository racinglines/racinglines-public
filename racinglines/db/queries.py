"""
Read the database into the tidy frame the timed-runs model uses, and write model
runs / predictions back.

The tidy frame has the same columns as the parse module's CSV output, plus race_id
and athlete_id, so the model code doesn't care where the data came from.
rider_id is "ath:<athlete id>", so riders merged in the database (e.g. via a
UCI ID) are one rider to the model too.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import insert, select, text

from racinglines.db import models as m

RESULTS_SQL = text("""
SELECT r.id AS result_id, ra.id AS race_id, e.source_key, e.start_date AS event_date,
       e.series_round, e.name AS event_name, v.slug AS venue, c.code AS category,
       sp.code AS discipline, ro.kind AS round, a.id AS athlete_id,
       a.display_name AS rider_name, r.team, r.bib, r.nation, r.position, r.status, r.time_ms
FROM results r
JOIN rounds ro       ON ro.id = r.round_id
JOIN races ra        ON ra.id = ro.race_id
JOIN events e        ON e.id = ra.event_id
JOIN seasons s       ON s.id = e.season_id
JOIN competitions co ON co.id = s.competition_id
JOIN sports sp       ON sp.id = co.sport_id
JOIN categories c    ON c.id = ra.category_id
JOIN athletes a      ON a.id = r.athlete_id
LEFT JOIN venues v   ON v.id = e.venue_id
WHERE co.code = :competition
""")

SPLITS_SQL = text("""
SELECT sp.result_id, sp.idx, sp.cum_time_ms, sp.rank
FROM splits sp
JOIN results r       ON r.id = sp.result_id
JOIN rounds ro       ON ro.id = r.round_id
JOIN races ra        ON ra.id = ro.race_id
JOIN events e        ON e.id = ra.event_id
JOIN seasons s       ON s.id = e.season_id
JOIN competitions co ON co.id = s.competition_id
WHERE co.code = :competition
""")


def load_tidy(engine, competition="uci_dhi_wc", with_splits=True):
    """All results of a competition as the tidy long frame (FINISH rows plus,
    optionally, one row per split)."""
    with engine.connect() as conn:
        res = pd.read_sql(RESULTS_SQL, conn, params=dict(competition=competition))
        spl = pd.read_sql(SPLITS_SQL, conn, params=dict(competition=competition)) if with_splits else None
    if res.empty:
        raise SystemExit(f"No results in the database for competition '{competition}'. "
                         f"Run: racinglines data ingest mtb_dh data/raw/mtb_dh/chronorace")

    res["event_id"] = res["source_key"] + "_" + res["category"]
    res["event_date"] = pd.to_datetime(res["event_date"]).dt.strftime("%Y-%m-%d")
    res["rider_id"] = "ath:" + res["athlete_id"].astype(str)
    res["cum_time_s"] = res["time_ms"] / 1000.0
    base_cols = ["result_id", "race_id", "athlete_id", "event_id", "event_date", "discipline", "category",
                 "round", "rider_id", "rider_name", "team", "bib", "nation", "venue", "series_round",
                 "event_name", "source_key"]
    finish = res[base_cols + ["cum_time_s", "position", "status"]].rename(columns={"position": "rank_at_split"})
    finish["sector_id"] = "FINISH"

    frames = [finish]
    if with_splits and spl is not None and not spl.empty:
        spl = spl.sort_values(["result_id", "idx"])
        spl["cum_time_s"] = spl["cum_time_ms"] / 1000.0
        spl["split_time_s"] = spl.groupby("result_id")["cum_time_s"].diff().fillna(spl["cum_time_s"])
        last = spl.groupby("result_id")["cum_time_s"].last()
        finish["split_time_s"] = finish["cum_time_s"] - finish["result_id"].map(last)
        s = spl.merge(res[base_cols], on="result_id")
        s["sector_id"] = "S" + s["idx"].astype(str)
        s["status"] = "OK"
        frames.append(s[base_cols + ["sector_id", "cum_time_s", "split_time_s", "rank"]]
                      .rename(columns={"rank": "rank_at_split"}))
    out = pd.concat(frames, ignore_index=True)
    out["start_order"] = np.nan
    out["track_condition"] = "unknown"
    return out


def code_version():
    """The checkout's git describe (commit, "-dirty" when the tree has changes), as model_runs.code_version records it;
    None when git is unavailable."""
    return _git_commit()


def _git_commit():
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[2], timeout=5).stdout.strip() or None
    except Exception:
        return None


def _athlete_id(rider_id):
    if not str(rider_id).startswith("ath:"):
        raise ValueError(f"Can only save predictions made from database data (rider_id {rider_id!r}).")
    return int(str(rider_id)[4:])


def _clean(v):
    """JSON-safe scalars for JSONB columns."""
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _deep_clean(v):
    """_clean through nested dicts/lists (NaN -> null anywhere in params/metrics)."""
    if isinstance(v, dict):
        return {k: _deep_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep_clean(x) for x in v]
    return _clean(v)


def _records(df):
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def save_model_run(session, *, competition, season=None, category=None, model="season_sim", kind,
                   data_through=None, params=None, metrics=None, race_predictions=None, standings=None):
    """Store one model run plus its per-race and standings predictions.

    race_predictions: DataFrame with rider_id, target, optional race_id, and any of
        win_prob / podium_prob / top10_prob / make_final_prob / exp_points (other
        columns go to `extra`).
    standings: DataFrame from simulate_standings (rider_id, current_points, ...).
    Returns the new model_runs.id."""
    comp = session.scalars(select(m.Competition).filter_by(code=competition)).one()
    season_row = session.scalars(select(m.Season).filter_by(competition_id=comp.id, year=int(season))).first() \
        if season else None
    cat_row = session.scalars(select(m.Category).filter_by(competition_id=comp.id, code=category)).first() \
        if category else None
    run = m.ModelRun(
        competition_id=comp.id, season_id=season_row.id if season_row else None,
        category_id=cat_row.id if cat_row else None, model=model, kind=kind,
        data_through=pd.to_datetime(data_through).date() if data_through else None,
        code_version=_git_commit(),
        params=_deep_clean(params or {}),
        metrics=_deep_clean(metrics),
    )
    session.add(run)
    session.flush()

    if race_predictions is not None and len(race_predictions):
        cols = ["win_prob", "podium_prob", "top10_prob", "make_final_prob", "exp_points"]
        skip = set(cols) | {"rider_id", "rider_name", "target", "race_id"}
        rows = []
        for rec in race_predictions.to_dict("records"):
            rows.append(dict(
                model_run_id=run.id, athlete_id=_athlete_id(rec["rider_id"]),
                race_id=_clean(rec.get("race_id")) if pd.notna(rec.get("race_id", np.nan)) else None,
                target=str(rec.get("target", "")),
                **{c: _clean(rec.get(c)) for c in cols},
                extra={k: _clean(v) for k, v in rec.items() if k not in skip and _clean(v) is not None} or None,
            ))
        session.execute(insert(m.RacePrediction), rows)

    if standings is not None and len(standings):
        cols = ["current_points", "exp_points", "points_p10", "points_p90", "champion_prob",
                "top3_prob", "top10_prob", "exp_rank"]
        session.execute(insert(m.StandingsPrediction), [
            dict(model_run_id=run.id, athlete_id=_athlete_id(rec["rider_id"]), **{c: _clean(rec.get(c)) for c in cols},
                 extra=rec.get("extra") if isinstance(rec.get("extra"), dict) else None)
            for rec in standings.to_dict("records")])
    session.commit()
    return run.id


def records(df):
    """DataFrame -> JSON-safe list of dicts (for ModelRun.metrics)."""
    return _records(df)
