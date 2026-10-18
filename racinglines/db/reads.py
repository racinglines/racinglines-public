"""Read-side queries shared by the web app, pipelines and strategies. Each returns a pandas DataFrame or dict."""

import pandas as pd
from sqlalchemy import text

from racinglines.markets import kinds as K


def q(conn, sql, **params):
    return pd.read_sql(text(sql), conn, params=params)


def latest_forecasts(conn):
    """Latest forecast run per competition x category."""
    return q(conn, """
        SELECT DISTINCT ON (mr.competition_id, mr.category_id)
               mr.id, mr.created_at, mr.data_through, mr.code_version, mr.params->>'cutoff' AS cutoff, co.code AS competition,
               co.name AS competition_name, c.code AS category, c.name AS category_name, s.year AS season
        FROM model_runs mr
        JOIN competitions co ON co.id = mr.competition_id
        LEFT JOIN categories c ON c.id = mr.category_id
        LEFT JOIN seasons s ON s.id = mr.season_id
        WHERE mr.kind = 'forecast'
        ORDER BY mr.competition_id, mr.category_id,
                 coalesce((mr.params->>'promoted_at')::timestamptz, mr.created_at) DESC, mr.id DESC""")


def run_race_targets(conn, run_id):
    """Distinct prediction targets in a run, with event info when linked to a race."""
    return q(conn, """
        SELECT rp.target, rp.race_id, e.id AS event_id, e.name AS event_name, e.start_date, e.status,
               v.name AS venue, e.source_key, count(*) AS n
        FROM race_predictions rp
        LEFT JOIN races ra ON ra.id = rp.race_id
        LEFT JOIN events e ON e.id = ra.event_id
        LEFT JOIN venues v ON v.id = e.venue_id
        WHERE rp.model_run_id = :run
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
        ORDER BY e.start_date NULLS LAST, rp.target""", run=run_id)


def race_predictions(conn, run_id, target, limit=None):
    df = q(conn, f"""
        SELECT a.id AS athlete_id, a.display_name AS athlete, a.nation, rp.win_prob, rp.podium_prob,
               rp.top10_prob, rp.make_final_prob, rp.exp_points, rp.extra
        FROM race_predictions rp JOIN athletes a ON a.id = rp.athlete_id
        WHERE rp.model_run_id = :run AND rp.target = :target
        ORDER BY rp.win_prob DESC NULLS LAST, rp.exp_points DESC NULLS LAST
        {'LIMIT :limit' if limit else ''}""", run=run_id, target=target, limit=limit)
    extra = pd.json_normalize(df.pop("extra").fillna({}).tolist()) if len(df) else pd.DataFrame()
    for col in ("attend_prob", "actual_final_rank", "actual_points"):
        if col in extra:
            df[col] = extra[col].to_numpy()
    return df


def standings_predictions(conn, run_id, limit=None):
    return q(conn, f"""
        SELECT a.id AS athlete_id, a.display_name AS athlete, sp.current_points, sp.exp_points,
               sp.points_p10, sp.points_p90, sp.champion_prob, sp.top3_prob, sp.top10_prob, sp.exp_rank
        FROM standings_predictions sp JOIN athletes a ON a.id = sp.athlete_id
        WHERE sp.model_run_id = :run
        ORDER BY sp.exp_points DESC {'LIMIT :limit' if limit else ''}""", run=run_id, limit=limit)


def model_run(conn, run_id):
    df = q(conn, """
        SELECT mr.*, co.code AS competition, s.year AS season, c.code AS category_code
        FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
        LEFT JOIN seasons s ON s.id = mr.season_id LEFT JOIN categories c ON c.id = mr.category_id
        WHERE mr.id = :run""", run=run_id)
    return df.iloc[0].to_dict() if len(df) else None


def events(conn, competition=None, season=None):
    return q(conn, """
        SELECT e.id, s.year AS season, e.start_date, e.name, v.name AS venue, v.country, e.series_round,
               e.status, e.source_key, co.code AS competition,
               (SELECT count(*) FROM races ra WHERE ra.event_id = e.id) AS races
        FROM events e JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
        LEFT JOIN venues v ON v.id = e.venue_id
        WHERE (CAST(:comp AS text) IS NULL OR co.code = CAST(:comp AS text)) AND (CAST(:season AS int) IS NULL OR s.year = CAST(:season AS int))
        ORDER BY e.start_date DESC""", comp=competition, season=season)


def event(conn, event_id):
    df = q(conn, """
        SELECT e.*, s.year AS season, v.name AS venue, v.country, co.code AS competition
        FROM events e JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
        LEFT JOIN venues v ON v.id = e.venue_id WHERE e.id = :id""", id=event_id)
    return df.iloc[0].to_dict() if len(df) else None


def event_results(conn, event_id):
    return q(conn, """
        SELECT ra.id AS race_id, c.code AS category, ro.kind AS round, ro.ordinal, r.position, a.id AS athlete_id,
               a.display_name AS athlete, r.nation, r.team, r.status, r.time_ms / 1000.0 AS time_s
        FROM results r JOIN rounds ro ON ro.id = r.round_id JOIN races ra ON ra.id = ro.race_id
        JOIN categories c ON c.id = ra.category_id JOIN athletes a ON a.id = r.athlete_id
        WHERE ra.event_id = :id
        ORDER BY c.code, ro.ordinal, r.position NULLS LAST, a.display_name""", id=event_id)


def event_predictions(conn, event_id):
    """Every stored prediction for this event's races, newest run first, with the actual Final result."""
    return q(conn, """
        SELECT rp.model_run_id AS run, mr.kind, mr.created_at, c.code AS category, a.id AS athlete_id,
               a.display_name AS athlete, rp.win_prob, rp.podium_prob, rp.top10_prob, rp.make_final_prob,
               rp.exp_points, fin.position AS actual_final_pos
        FROM race_predictions rp JOIN model_runs mr ON mr.id = rp.model_run_id
        JOIN races ra ON ra.id = rp.race_id JOIN categories c ON c.id = ra.category_id
        JOIN athletes a ON a.id = rp.athlete_id
        LEFT JOIN rounds fr ON fr.race_id = ra.id AND fr.kind = 'final'
        LEFT JOIN results fin ON fin.round_id = fr.id AND fin.athlete_id = rp.athlete_id
        WHERE ra.event_id = :id
        ORDER BY rp.model_run_id DESC, rp.win_prob DESC NULLS LAST""", id=event_id)


def athletes(conn, search=None, limit=200):
    return q(conn, """
        SELECT a.id, a.display_name AS athlete, a.nation, count(r.id) AS results,
               max(e.start_date) AS last_race,
               count(*) FILTER (WHERE ro.kind = 'final' AND r.position = 1) AS wins
        FROM athletes a
        LEFT JOIN results r ON r.athlete_id = a.id LEFT JOIN rounds ro ON ro.id = r.round_id
        LEFT JOIN races ra ON ra.id = ro.race_id LEFT JOIN events e ON e.id = ra.event_id
        WHERE (CAST(:s AS text) IS NULL OR a.display_name ILIKE '%' || CAST(:s AS text) || '%')
        GROUP BY a.id ORDER BY last_race DESC NULLS LAST, results DESC LIMIT :limit""", s=search, limit=limit)


def athlete(conn, athlete_id):
    df = q(conn, "SELECT * FROM athletes WHERE id = :id", id=athlete_id)
    if not len(df):
        return None
    out = df.iloc[0].to_dict()
    out["identifiers"] = q(conn, "SELECT scheme, value FROM athlete_identifiers WHERE athlete_id = :id",
                           id=athlete_id).to_dict("records")
    return out


def athlete_results(conn, athlete_id):
    return q(conn, """
        SELECT e.id AS event_id, e.start_date, v.name AS venue, c.code AS category, ro.kind AS round,
               r.position, r.status, r.time_ms / 1000.0 AS time_s, r.team
        FROM results r JOIN rounds ro ON ro.id = r.round_id JOIN races ra ON ra.id = ro.race_id
        JOIN events e ON e.id = ra.event_id JOIN categories c ON c.id = ra.category_id
        LEFT JOIN venues v ON v.id = e.venue_id
        WHERE r.athlete_id = :id AND ro.kind <> 'practice'
        ORDER BY e.start_date DESC, ro.ordinal DESC""", id=athlete_id)


def athlete_competitions(conn, athlete_id):
    """The competition codes an athlete has a result or a prediction in."""
    return set(conn.execute(text("""
        SELECT co.code FROM results r JOIN rounds ro ON ro.id = r.round_id JOIN races ra ON ra.id = ro.race_id
        JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
        JOIN competitions co ON co.id = s.competition_id WHERE r.athlete_id = :a
        UNION SELECT co.code FROM race_predictions rp JOIN model_runs mr ON mr.id = rp.model_run_id
        JOIN competitions co ON co.id = mr.competition_id WHERE rp.athlete_id = :a"""), dict(a=int(athlete_id))).scalars())


def athlete_predictions(conn, athlete_id):
    return q(conn, """
        SELECT rp.model_run_id AS run, mr.kind, mr.created_at, rp.target, e.id AS event_id,
               v.name AS venue, rp.win_prob, rp.podium_prob, rp.top10_prob, rp.make_final_prob, rp.exp_points
        FROM race_predictions rp JOIN model_runs mr ON mr.id = rp.model_run_id
        LEFT JOIN races ra ON ra.id = rp.race_id LEFT JOIN events e ON e.id = ra.event_id
        LEFT JOIN venues v ON v.id = e.venue_id
        WHERE rp.athlete_id = :id ORDER BY rp.model_run_id DESC, rp.target""", id=athlete_id)


def market_links(conn):
    return q(conn, """
        SELECT ml.*, a.display_name AS athlete, e.name AS event_name, v.name AS venue, co.code AS competition,
               c.code AS category
        FROM market_links ml LEFT JOIN athletes a ON a.id = ml.athlete_id
        JOIN competitions co ON co.id = ml.competition_id LEFT JOIN categories c ON c.id = ml.category_id
        LEFT JOIN races ra ON ra.id = ml.race_id LEFT JOIN events e ON e.id = ra.event_id
        LEFT JOIN venues v ON v.id = e.venue_id
        ORDER BY ml.active DESC, ml.id DESC""")


PREDICTION_COLUMNS = {
    "race_win": "win_prob", "race_podium": "podium_prob", "race_top10": "top10_prob",
    "race_make_final": "make_final_prob", "champion": "champion_prob", "standings_top3": "top3_prob",
}
# every prediction kind a market can be linked to (see racinglines/markets/polymarket/sync.py): the one registry
PREDICTION_KINDS = [c for c in K.KINDS]
# F1 sprint weekends (Kalshi's sprint markets, on unless RACINGLINES_KALSHI_SPRINTS=0): a run that simulates the sprint
# stores extra.sprint_pole_prob / sprint_win_prob; until then the sprint is priced as the model's race:
# sprint pole from the qualifying-pace pole probability (SQ3 is the same session type), the sprint winner from
# the race win probability (the season forecast already simulates sprints this way, with sprint points)
SPRINT_FALLBACK = {"race_sprint_pole": "pole_prob", "race_sprint_win": "win_prob"}
# per-race kinds read from race_predictions.extra: fl_prob only from runs priced with the position_sim `fastlap`
# variant; top5_prob / mover_prob from F1 runs saved since 2026-10-04 (older runs leave those links unpriced); top20_prob
# from the NASCAR / MotoGP replay and forecast runs (pipelines/position_replay.py, sport_forecast.py)
EXTRA_PROB = {"race_pole": "pole_prob", "race_fastest_lap": "fl_prob", "race_top5": "top5_prob",
              "race_top20": "top20_prob",
              "race_biggest_mover": "mover_prob"}


def _id(v):
    """Row values from pandas: NaN -> None, numpy ints -> int."""
    return None if v is None or (not isinstance(v, (dict, list)) and pd.isna(v)) else int(v)


def _params(link):
    p = link.get("params")
    if isinstance(p, str):
        import json
        p = json.loads(p)
    return p or {}


def latest_forecast_run(conn, competition_id, category_id=None):
    run = q(conn, """
        SELECT id, metrics FROM model_runs WHERE kind = 'forecast' AND competition_id = :comp
          AND (CAST(:cat AS int) IS NULL OR category_id = CAST(:cat AS int))
        ORDER BY coalesce((params->>'promoted_at')::timestamptz, created_at) DESC, id DESC LIMIT 1""",
            comp=competition_id, cat=category_id)
    return (int(run["id"].iloc[0]), run["metrics"].iloc[0] or {}) if len(run) else (None, {})


def model_prob(conn, link, _cache=None, run_id=None):
    """(probability, model_run_id) for a market link from the latest FORECAST run of
    its competition/category (or from `run_id`, e.g. a diagnostic run); inverted for
    "No"-type tokens. None if the model doesn't price it (unmodeled kind, driver not
    in the field, race already run)."""
    kind = link.get("prediction")
    if kind not in PREDICTION_KINDS:
        return None, None
    comp, cat = _id(link.get("competition_id")), _id(link.get("category_id"))
    athlete, race = _id(link.get("athlete_id")), _id(link.get("race_id"))
    params = _params(link)
    key = (comp, cat, run_id)
    if _cache is not None and key in _cache:
        run_id, metrics = _cache[key]
    else:
        if run_id is None:
            run_id_, metrics = latest_forecast_run(conn, comp, cat)
        else:
            run_id_, metrics = run_id, (q(conn, "SELECT metrics FROM model_runs WHERE id = :i", i=run_id)["metrics"].iloc[0] or {})
        if _cache is not None:
            _cache[key] = (run_id_, metrics)
        run_id = run_id_
    if run_id is None:
        return None, None

    p = None
    if kind in ("champion", "standings_top3", "season_wins_ge", "standings_h2h"):
        row = _standings_row(conn, run_id, athlete, _cache)
        if row is not None:
            p_, extra = row
            if kind in ("champion", "standings_top3"):
                p = p_[kind]
            elif kind == "season_wins_ge":
                p = (extra.get("wins_ge") or {}).get(str(params.get("n")))
            else:
                p = (extra.get("ahead_of") or {}).get(str(params.get("opponent_id")))
    elif kind == "constructors_champion":
        team = params.get("team")
        p = next((c["champion_prob"] for c in metrics.get("constructors", []) if c.get("team_key") == team), None)
    elif kind == "race_constructor_top":
        p = ((metrics.get("race_constructor_top") or {}).get(params.get("event_key")) or {}).get(params.get("team"))
    else:  # per-race, per-athlete
        row = _race_pred_row(conn, run_id, race, athlete, _cache)
        if row is not None:
            df, extra = row
            if kind == "race_h2h":
                p = (extra.get("h2h") or {}).get(str(params.get("opponent_id")))
            elif kind in EXTRA_PROB:
                p = extra.get(EXTRA_PROB[kind])
            elif kind in SPRINT_FALLBACK:
                p = extra.get(kind.replace("race_", "") + "_prob")
                if p is None:
                    p = extra.get("pole_prob") if kind == "race_sprint_pole" else df["win_prob"]
            else:     # a kind no stored column holds (the sprint's podium, top 8, ...): unpriced, not an error
                p = df.get(PREDICTION_COLUMNS.get(kind))
    if p is None or pd.isna(p):
        return None, run_id
    p = float(p)
    return (1 - p if link.get("invert") else p), run_id


def _standings_row(conn, run_id, athlete, _cache):
    """({champion,standings_top3}_prob, extra) for one athlete of a standings run: one query per run_id, not
    per link, when `_cache` is given (market_links can list hundreds of outcomes per run)."""
    k = ("standings", run_id)
    if _cache is not None and k in _cache:
        by_athlete = _cache[k]
    else:
        df = q(conn, "SELECT athlete_id, champion_prob, top3_prob, extra FROM standings_predictions WHERE model_run_id = :run",
               run=run_id)
        by_athlete = {int(r.athlete_id): (dict(champion=r.champion_prob, standings_top3=r.top3_prob), r.extra or {})
                     for r in df.itertuples()}
        if _cache is not None:
            _cache[k] = by_athlete
    return by_athlete.get(athlete)


def _race_pred_row(conn, run_id, race, athlete, _cache):
    """(row dict, extra) for one athlete of a race (or the remaining-round target when `race` is None): one
    query per (run_id, race), not per link."""
    k = ("race_preds", run_id, race)
    if _cache is not None and k in _cache:
        by_athlete = _cache[k]
    else:
        where = "race_id = :race" if race else "target = 'remaining_round'"
        df = q(conn, f"""SELECT athlete_id, win_prob, podium_prob, top10_prob, make_final_prob, extra
                         FROM race_predictions WHERE model_run_id = :run AND {where}""", run=run_id, race=race)
        by_athlete = {int(r.athlete_id): (dict(win_prob=r.win_prob, podium_prob=r.podium_prob, top10_prob=r.top10_prob,
                                              make_final_prob=r.make_final_prob), r.extra or {})
                     for r in df.itertuples()}
        if _cache is not None:
            _cache[k] = by_athlete
    return by_athlete.get(athlete)


def orders(conn, limit=200):
    return q(conn, """
        SELECT o.*, ml.question, ml.outcome, a.display_name AS athlete
        FROM orders o LEFT JOIN market_links ml ON ml.id = o.market_link_id
        LEFT JOIN athletes a ON a.id = ml.athlete_id
        ORDER BY o.id DESC LIMIT :limit""", limit=limit)


def upcoming_races(conn):
    """Races of events that aren't completed yet (for linking markets)."""
    return q(conn, """
        SELECT ra.id AS race_id, e.name AS event_name, v.name AS venue, e.start_date, c.code AS category,
               co.id AS competition_id, c.id AS category_id
        FROM races ra JOIN events e ON e.id = ra.event_id JOIN categories c ON c.id = ra.category_id
        JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
        LEFT JOIN venues v ON v.id = e.venue_id
        WHERE e.status <> 'completed' ORDER BY e.start_date, c.code""")
