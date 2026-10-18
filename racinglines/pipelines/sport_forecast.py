"""
A live forecast for the result-only sports' next races (NASCAR Cup, MotoGP): the sport's pricing model
([sport] pricing_model, the one the taker replay uses) run on every result stored before today, for each scheduled race
in the calendar, stored as a `forecast` model run so the board, the race pages and the exchange markets get a fair
price, as F1 has from its own forecast.

    racinglines model forecast nascar                       # print the next races' favourites, write nothing
    racinglines model forecast nascar --save --backup FILE  # store one forecast run (model_runs + race_predictions)
    racinglines model forecast nascar --undo RUN_ID         # delete a stored run (its predictions go with it)

The field for an upcoming race is its own entry list where one is stored (the entrants of a session of that race
already ingested from the weekend feed, such as practice or qualifying, less any DNS), else the latest completed
race's classified entrants (the calendar holds no entry list before the weekend). The model settings are the replay's (sports/<sport>.toml [replay], seed 7): nothing is tuned here. A sport with
no scheduled race stored (MotoGP until its calendar is ingested) has nothing to forecast and says so.

A schema may add a [forecast] table (every key optional; without it, the behaviour above):

    [forecast]
    field = "entry_list"     price only races with their own entry list (a sport whose fields change from race to race,
                             road cycling: the last race's field would be a different race's riders)
    h2h = true               also store each entrant's head-to-head chance against every other entrant
                             (race_predictions.extra.h2h {opponent id: P(ahead)}), which prices race_h2h links

The sport's events source is its [replay] source, else its [model] source (a sport with no taker replay).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.pipelines import position_replay as P

N_RACES = 3


def spec(sport):
    """position_replay.spec, or for a sport with no [replay] table the same keys from [model] source."""
    from racinglines import sports
    s = sports.load(sport)
    if s.get("replay"):
        return P.spec(sport)
    src = s.get("model", {}).get("source")
    if not src:
        raise ValueError(f"sports/{sport}.toml: the forecast needs [replay] source or [model] source")
    return dict(sport=sport, source=src, competition=s["competition"]["code"], model=s["sport"]["pricing_model"])


def options(sport):
    """The schema's [forecast] table with its defaults: field ("last_race" or "entry_list"), h2h (bool)."""
    from racinglines import sports
    f = dict(sports.load(sport).get("forecast", {}))
    out = dict(field=f.pop("field", "last_race"), h2h=bool(f.pop("h2h", False)))
    if f or out["field"] not in ("last_race", "entry_list"):
        raise ValueError(f"sports/{sport}.toml [forecast]: field is last_race | entry_list, h2h a bool; got {f or out}")
    return out


def upcoming(conn, sp, n=N_RACES, today=None):
    """The sport's next scheduled races (not completed, from today on): race_id, event_key, name, season, start."""
    df = pd.read_sql(text("""
        SELECT ra.id AS race_id, e.source_key AS event_key, e.name, s.year AS season, e.start_date AS start,
               ra.category_id
        FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
        JOIN competitions co ON co.id = s.competition_id
        WHERE co.code = :c AND e.source = :src AND e.status NOT IN ('completed', 'cancelled')
          AND e.start_date >= CAST(:t AS date)
        ORDER BY e.start_date, ra.id LIMIT :n"""), conn,
                     params=dict(c=sp["competition"], src=sp["source"], n=n,
                                 t=str(today or datetime.now(timezone.utc).date())))
    return df


def last_field(conn, sp):
    """(event_key, [athlete ids]) of the latest completed race: the field the next races are priced for."""
    rs = P.races(conn, sp)
    if not len(rs):
        return None, []
    r = rs.iloc[-1]
    res = P.race_results(conn, int(r["race_id"]))
    return r["event_key"], sorted(int(a) for a in res["athlete_id"])


def entry_list(conn, race_id):
    """[athlete ids] who have a result in any stored session of the race other than a DNS (practice, qualifying,
    a sprint, from the weekend feed already ingested): the race's start list; [] when no session is stored."""
    rows = conn.execute(text("""
        SELECT DISTINCT r.athlete_id FROM results r JOIN rounds ro ON ro.id = r.round_id
        WHERE ro.race_id = :r AND r.athlete_id IS NOT NULL AND upper(coalesce(r.status, 'OK')) <> 'DNS'
        ORDER BY r.athlete_id"""), dict(r=int(race_id))).all()
    return [int(a) for (a,) in rows]


def forecast(engine, sport, n=N_RACES, today=None, data=None):
    """dict(sport, model, settings, field_from, races=[dict(race row, sims)], data) for the next n races."""
    from racinglines.models.race_model import Event
    sp = spec(sport)
    opt = options(sport)
    model = P.model_for(sp)
    st = model.Settings.from_dict({"seed": P.SEED})
    with engine.connect() as conn:
        nxt = upcoming(conn, sp, n, today)
        src, field = last_field(conn, sp)
        own = {int(r.race_id): entry_list(conn, r.race_id) for r in nxt.itertuples()}
    if opt["field"] == "entry_list":                         # only a race's own entry list is its field
        src, field = "each race's own entry list", []
    out = dict(sport=sport, sp=sp, model=model, settings=st, field_from=src, field=field, races=[], h2h=opt["h2h"])
    if not len(nxt) or not (field or any(own.values())):
        return dict(out, data=data)
    if data is None:
        data = model.load(engine.url.render_as_string(hide_password=False))
    hist = model.history(data, st)
    rng = np.random.default_rng(st.rng_seed)
    for r in nxt.itertuples():
        entrants = own.get(int(r.race_id)) or field          # the race's own entry list, else the last race's field
        if not entrants:
            continue
        ev = Event(id=r.event_key, season=int(r.season), cutoff=r.start, name=str(r.name),
                   info={"field": entrants})
        sims = model.price(hist, ev, st, rng)
        if sims is not None:
            out["races"].append(dict(race=r, sims=sims, field_from=r.event_key if own.get(int(r.race_id)) else src))
    return dict(out, data=data)


def table(conn, fc, top=5):
    """The forecast as text: per race, the favourites' win / podium / top 10."""
    if not fc["races"]:
        return f"{fc['sport']}: nothing to forecast (no scheduled race stored, or no completed race to take a field from)"
    names = dict(conn.execute(text("SELECT id, display_name FROM athletes WHERE id = ANY(:i)"),
                              dict(i=sorted({int(a) for x in fc["races"] for a in x["sims"].entrants}))).all())
    lines = [f"{fc['sport']} forecast ({fc['model'].name}; field from {fc['field_from']}, {len(fc['field'])} entrants)"]
    for x in fc["races"]:
        r, sims = x["race"], x["sims"]
        win = {a: P.fair(sims, "race_win", a) for a in sims.entrants}
        lines.append(f"  {r.start} {r.name}")
        for a in sorted(win, key=win.get, reverse=True)[:top]:
            lines.append(f"    {names.get(a, a)!s:28s} win {win[a]:6.1%}  podium {P.fair(sims, 'race_podium', a):6.1%}"
                         f"  top 10 {P.fair(sims, 'race_top10', a):6.1%}")
    return "\n".join(lines)


def save(engine_url, fc):
    """Store the forecast as one model run (kind 'forecast', the races' category) with race_predictions per race and
    entrant (win / podium / top 10; top 5 and top 20 in extra). Returns the run id, or None when there is nothing."""
    from racinglines.db import models as m
    from racinglines.db.config import get_session
    if not fc["races"]:
        return None
    sp, st = fc["sp"], fc["settings"]
    first = fc["races"][0]["race"]
    with get_session(engine_url) as s:
        comp = s.execute(text("SELECT id FROM competitions WHERE code = :c"), dict(c=sp["competition"])).scalar()
        season = s.execute(text("SELECT id FROM seasons WHERE competition_id = :c AND year = :y"),
                           dict(c=comp, y=int(first.season))).scalar()
        done = s.execute(text("""SELECT max(e.start_date) FROM events e JOIN seasons se ON se.id = e.season_id
                                 WHERE se.competition_id = :c AND e.status = 'completed'"""), dict(c=comp)).scalar()
        run = m.ModelRun(competition_id=comp, season_id=season, category_id=int(first.category_id),
                         model=fc["model"].name, kind="forecast", data_through=done,
                         params=dict(mode="sport forecast", sport=fc["sport"], field_from=fc["field_from"],
                                     cutoff=str(pd.Timestamp.now(tz="UTC").floor("s")), model_settings=st.to_json(),
                                     model_key=st.model_key, sims=int(fc["races"][0]["sims"].n_sims),
                                     races=[x["race"].event_key for x in fc["races"]]))
        s.add(run)
        s.flush()
        for x in fc["races"]:
            r, sims = x["race"], x["sims"]
            p = {k: [P.fair(sims, k, a) for a in sims.entrants] for k in P.N_OF}
            h2h = head_to_heads(sims) if fc.get("h2h") else None
            for i, a in enumerate(sims.entrants):
                extra = dict(top5_prob=p["race_top5"][i], top20_prob=p["race_top20"][i])
                if h2h is not None:
                    extra["h2h"] = h2h[i]
                s.add(m.RacePrediction(model_run_id=run.id, race_id=int(r.race_id), target=f"race:{r.event_key}",
                                       athlete_id=int(a), win_prob=p["race_win"][i], podium_prob=p["race_podium"][i],
                                       top10_prob=p["race_top10"][i], extra=extra))
        s.commit()
        return run.id


def head_to_heads(sims, digits=4):
    """Per entrant (sims.entrants order): {opponent id (str): P(finishing ahead of them)}, every other entrant, as
    position_replay.fair's race_h2h (the better simulated rank; retirements are ranked last)."""
    rank = np.asarray(sims.rank)
    ids = [str(int(a)) for a in sims.entrants]
    out = []
    for i in range(rank.shape[1]):
        p = (rank[:, [i]] < rank).mean(axis=0)
        out.append({ids[j]: round(float(p[j]), digits) for j in range(rank.shape[1]) if j != i})
    return out


def undo(session, run_id):
    """Delete one forecast run this module stored (its race_predictions first). Returns runs deleted (0 or 1)."""
    ok = session.execute(text("""SELECT 1 FROM model_runs WHERE id = :i AND kind = 'forecast'
                                 AND params->>'mode' = 'sport forecast'"""), dict(i=run_id)).scalar()
    if not ok:
        return 0
    session.execute(text("DELETE FROM race_predictions WHERE model_run_id = :i"), dict(i=run_id))
    return session.execute(text("DELETE FROM model_runs WHERE id = :i"), dict(i=run_id)).rowcount
