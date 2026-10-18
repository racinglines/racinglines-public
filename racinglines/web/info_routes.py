"""
Read-only pages: model runs, events, athletes, the built docs, the intro and the pitch deck.
Split out of web/app.py, which imports this module to register the routes.
"""

import os
import posixpath
from pathlib import Path

import pandas as pd
from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from racinglines.paths import CHECKOUT, HOME
from racinglines.web import audience as A
from racinglines.web import roles as R
from racinglines.web.app import ANY, PRO, allow, app, audit, check_csrf, conn, data, render, rows  # noqa: F401
from racinglines.web.app import ROOT, SITE

# The internal onboarding book (a built Jupyter Book), served at /docs/internal/ to admins only. It is data, not
# code: it lives on the VM under data/internal-docs/ (vm.sh internal-docs copies it there) and never in git.
INTERNAL_DOCS = Path(os.environ.get("RACINGLINES_INTERNAL_DOCS") or HOME / "data" / "internal-docs")


PRED_COLS = ["athlete", "nation", "win_prob", "podium_prob", "top10_prob", "make_final_prob", "exp_points"]
STAND_COLS = ["athlete", "current_points", "exp_points", "points_p10", "points_p90", "champion_prob",
              "top3_prob", "exp_rank"]


@app.get("/lab/runs/{run_id}", response_class=HTMLResponse, dependencies=[allow(*PRO)])
def run_detail(request: Request, run_id: int, c=Depends(conn)):
    run = data.model_run(c, run_id)
    if not run:
        raise HTTPException(404)
    A.guard(request.state.user, run["competition"])
    raw = dict(run.get("metrics") or {})
    if run["kind"] == "sweep" and raw.get("totals"):     # P&L and closing-line value per strategy, side by side
        from racinglines.web import edge as EDGE
        clv = raw.pop("clv", None)
        raw = {"strategies": EDGE.strategy_rows(dict(raw, clv=clv)), **raw}
        if isinstance(clv, dict):                          # metrics.clv's tables (pipelines/clv.py sweep_summary)
            raw.update({f"clv {k}": v for k, v in clv.items()})
    metrics = {k: pd.DataFrame(v) if isinstance(v, list) else pd.DataFrame([v])
               for k, v in raw.items()}
    targets = rows(data.run_race_targets(c, run_id))
    preds = [dict(t, table=rows(data.race_predictions(c, run_id, t["target"]))) for t in targets]
    return render(request, "run.html", run=run, metrics={k: (list(v.columns), rows(v)) for k, v in metrics.items()},
                  preds=preds, standings=rows(data.standings_predictions(c, run_id)),
                  pred_cols=PRED_COLS + ["attend_prob", "actual_final_rank", "actual_points"], stand_cols=STAND_COLS)


@app.get("/events/by-key/{source_key}")
def event_by_key(source_key: str, c=Depends(conn)):
    """Stable link to an event by its source key (e.g. ChronoRace 20260925_mtb)."""
    ev = data.q(c, "SELECT id FROM events WHERE source_key = :k ORDER BY id DESC LIMIT 1", k=source_key)
    if not len(ev):
        raise HTTPException(404, f"no event {source_key}")
    return RedirectResponse(f"/events/{int(ev['id'].iloc[0])}", status_code=303)


@app.get("/racinglines101", response_class=HTMLResponse)
def racinglines101(request: Request):
    """Plain-language intro in three parts with a section nav: how we trade, how the prediction engine works, using the app (public, linked from the login page)."""
    return render(request, "racinglines101.html")


@app.get("/racinglines102")
def racinglines102():
    """The engine half of Racinglines 101, which is one page in three parts (public)."""
    return RedirectResponse("/racinglines101#engine", status_code=307)


@app.get("/pitch", response_class=HTMLResponse)
def pitch():
    """The pitch deck (pitch.html at the repo root), public like /login and /signup."""
    from fastapi.responses import FileResponse
    if not (ROOT / "pitch.html").exists():          # a checkout's only; the public release has no deck
        raise HTTPException(404)
    return FileResponse(ROOT / "pitch.html", media_type="text/html")




@app.get("/llms.txt")
@app.get("/llms-full.txt")
def llms_txt(request: Request):
    """The AI-readable docs index and full text (llmstxt.org), public by exact path: mkdocs writes them into site/ from
    the nav's public pages only (never docs/internal or the project library)."""
    from fastapi.responses import FileResponse
    f = SITE / request.url.path.lstrip("/")
    if not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, media_type="text/plain; charset=utf-8")


@app.get("/docs")
def docs_root():
    return RedirectResponse("/docs/", status_code=307)


@app.get("/docs/{path:path}")
def docs(request: Request, path: str):
    """The built docs (site/), behind the same login. /docs/internal/... is the internal book (INTERNAL_DOCS), for
    admins only: the path is normalized first, so /docs/quickstart/term/../../internal/ is caught too."""
    from fastapi.responses import FileResponse
    rel = posixpath.normpath("/" + path).lstrip("/")
    base = SITE
    if rel == "internal" or rel.startswith("internal/"):
        u = getattr(request.state, "user", None)
        if not u or R.canonical(u["role"]) != "admin":
            raise HTTPException(404)                    # not 403: non-admins aren't told the book exists
        if rel == "internal" and not path.endswith("/"):
            return RedirectResponse("/docs/internal/", status_code=307)   # the book's relative links need the slash
        base, path = INTERNAL_DOCS, rel[len("internal"):].lstrip("/")
    f = (base / path).resolve()
    if not f.is_relative_to(base.resolve()):
        raise HTTPException(404)
    if f.is_dir():
        f = f / "index.html"
    if not f.is_file():
        if base is SITE and not SITE.is_dir():
            if not CHECKOUT:                            # pip install: the docs aren't in the package; the site has them
                return RedirectResponse("https://racinglines.bet/docs/" + path, status_code=307)
            raise HTTPException(404, "Docs not built: run python -m mkdocs build -d site")
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/events/{event_id}", response_class=HTMLResponse)
def event_detail(request: Request, event_id: int, c=Depends(conn)):
    ev = data.event(c, event_id)
    if not ev:
        raise HTTPException(404)
    A.guard(request.state.user, ev["competition"])
    res = data.event_results(c, event_id)
    groups = [dict(category=cat, round=rnd, table=rows(g))
              for (cat, _, rnd), g in res.groupby(["category", "ordinal", "round"], sort=True)] if len(res) else []
    preds = data.event_predictions(c, event_id)
    pred_groups = [dict(run=int(run), kind=g["kind"].iloc[0], category=cat, table=rows(g.head(30)))
                   for (run, cat), g in preds.groupby(["run", "category"], sort=False)] if len(preds) else []
    if R.is_basic(request.state.user):
        pred_groups = []  # model fair values are the makers' edge
    return render(request, "event.html", ev=ev, groups=groups, pred_groups=pred_groups)


@app.get("/athletes/{athlete_id}", response_class=HTMLResponse)
def athlete_detail(request: Request, athlete_id: int, c=Depends(conn)):
    a = data.athlete(c, athlete_id)
    if not a:
        raise HTTPException(404)
    A.guard_athlete(c, request.state.user, athlete_id)
    preds = [] if R.is_basic(request.state.user) else rows(data.athlete_predictions(c, athlete_id))
    return render(request, "athlete.html", a=a, results=rows(data.athlete_results(c, athlete_id)), preds=preds,
                  competitions=sorted(data.athlete_competitions(c, athlete_id)))   # for the source note
