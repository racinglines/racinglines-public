"""
A read-only JSON API next to the pages: the same data the events and athletes pages show every signed-in user,
behind the same login (session cookie or HTTP Basic), for scripts and collaborators. Off by default: every
route answers 404 unless RACINGLINES_JSON_API=1 is set where the web app runs.

    GET /api/v1/events[?season=2026&competition=f1_wdc]   events, newest first
    GET /api/v1/events/{id}                               one event with its classification (every round)
    GET /api/v1/athletes[?q=name&limit=200]               athletes with result counts
    GET /api/v1/athletes/{id}                             one athlete with results (practice left out)

No model prices, quotes, positions or market data: fair values are the makers' edge (the pages hide them from
takers too), and what goes public beyond the private demo is the owner's call (docs/webapp.md, JSON API).
"""

import json
import os

from fastapi import Depends, HTTPException, Request

from racinglines.db import reads as data
from racinglines.web import audience as A
from racinglines.web.app import app, conn

PREFIX = "/api/v1"


def enabled():
    return os.environ.get("RACINGLINES_JSON_API", "").lower() in ("1", "true", "yes")


def _on():
    if not enabled():
        raise HTTPException(404)


def _records(df):
    """DataFrame -> JSON-safe records (dates as ISO strings, NaN as null)."""
    if df is None or not len(df):
        return []
    return json.loads(df.convert_dtypes().to_json(orient="records", date_format="iso"))    # whole-number floats -> ints


def _plain(d):
    return json.loads(json.dumps(d, default=str))


@app.get(PREFIX + "/events", dependencies=[Depends(_on)])
def api_events(request: Request, season: int | None = None, competition: str | None = None, c=Depends(conn)):
    A.guard(getattr(request.state, "user", None), competition)
    df = data.events(c, competition=competition, season=season)
    hide = A.hidden_comps(getattr(request.state, "user", None))   # sports this user doesn't see ([sport] audience)
    if hide and len(df) and "competition" in df:
        df = df[~df["competition"].isin(hide)]
    return dict(events=_records(df))


@app.get(PREFIX + "/events/{event_id}", dependencies=[Depends(_on)])
def api_event(request: Request, event_id: int, c=Depends(conn)):
    ev = data.event(c, event_id)
    if not ev:
        raise HTTPException(404, f"no event {event_id}")
    A.guard(getattr(request.state, "user", None), ev["competition"])
    keep = ("id", "name", "start_date", "status", "source", "source_key", "series_round", "season", "venue",
            "country", "competition")
    return dict(event=_plain({k: ev.get(k) for k in keep}), results=_records(data.event_results(c, event_id)))


@app.get(PREFIX + "/athletes", dependencies=[Depends(_on)])
def api_athletes(q: str | None = None, limit: int = 200, c=Depends(conn)):
    return dict(athletes=_records(data.athletes(c, q, limit=max(1, min(limit, 1000)))))


@app.get(PREFIX + "/athletes/{athlete_id}", dependencies=[Depends(_on)])
def api_athlete(request: Request, athlete_id: int, c=Depends(conn)):
    a = data.athlete(c, athlete_id)
    if not a:
        raise HTTPException(404, f"no athlete {athlete_id}")
    A.guard_athlete(c, getattr(request.state, "user", None), athlete_id)
    keep = ("id", "display_name", "nation", "identifiers")
    return dict(athlete=_plain({k: a.get(k) for k in keep if k in a}), results=_records(data.athlete_results(c, athlete_id)))
