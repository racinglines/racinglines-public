"""
Single-event diagnostics (pro and admin): as-of prices vs the exchange vs the result, and the maker replay.
Split out of web/app.py, which imports this module to register the routes.
"""

import pandas as pd
from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from racinglines.db.config import get_session
from racinglines.markets import venues as V
from racinglines.web import diag
from racinglines.web import users as U
from racinglines.web.app import ANY, PRO, allow, app, audit, check_csrf, conn, data, render, rows  # noqa: F401
from racinglines.web.app import _user_dict


def _guard(request, c, run_id):
    """404 for a run of a sport this user doesn't see ([sport] audience, web/audience.py)."""
    from racinglines.web import audience as A
    run = data.model_run(c, run_id)
    if run:
        A.guard(request.state.user, run["competition"])


@app.get("/lab/diagnostics/{run_id}", response_class=HTMLResponse, dependencies=[allow(*PRO)])
def diag_page(request: Request, run_id: int, msg: str = "", fill: str = "through", h: float = 0.02, size: float = 50,
              max_pos: float = 250, cap: float = 1000, skew: float = 1.0, disagree: float = 0.15, min_vol: float = 100,
              pull: int = 15, venue: str = "", c=Depends(conn)):
    _guard(request, c, run_id)
    d = diag.load(c, run_id)
    venue = "kalshi" if venue == "kalshi" and V.KALSHI_VENUE else ""      # the maker replay on Kalshi's tape
    if d is None:
        raise HTTPException(404)
    fill = fill if fill in ("touch", "through") else "through"
    mk = d["markets"]
    groups = [(k, rows(g.sort_values("fair", ascending=False, na_position="last")))
              for k, g in mk.groupby("kind", sort=False)] if len(mk) else []
    paper = mk[mk["paper_side"].notna()] if len(mk) and "paper_side" in mk else pd.DataFrame()
    knobs = dict(fill=fill, half_spread=min(max(h, 0.005), 0.2), size=min(max(size, 1), 5000),
                 max_pos=min(max(max_pos, 1), 50000), max_capital=min(max(cap, 10), 1e6), skew=min(max(skew, 0), 5),
                 max_disagree=None if disagree <= 0 else min(disagree, 1), min_volume_24h=max(min_vol, 0),
                 pull_min=min(max(pull, 0), 240))
    rp = diag.replay(c, run_id, **knobs, **(dict(exchange=venue) if venue else {}))
    f = rp["fills"]
    top = f.reindex(f["pnl"].abs().sort_values(ascending=False).index).head(15) if len(f) else f
    bets = d["bets"]
    summary = dict(
        paper_n=len(paper), paper_staked=len(paper) * diag.PAPER_STAKE,
        paper_pnl=float(paper["paper_pnl"].fillna(0).sum()) if len(paper) else 0.0,
        taker_n=len(bets), taker_staked=float(bets["stake"].sum()) if len(bets) else 0.0,
        taker_pnl=float(bets["taker_pnl"].sum()) if len(bets) else 0.0)
    return render(request, "diag.html", d=d, groups=groups, scores=rows(d["scores"]), result=rows(d["result"]),
                  bets=rows(bets), paper=rows(paper), summary=summary, msg=msg,
                  coherence=d.get("coherence", {}), rp=rp, rp_stages=rows(rp["stages"]), rp_summary=rows(rp["summary"]),
                  rp_skips=rows(rp["skips"]), rp_sweep=rows(rp["sweep"]), rp_top=rows(top),
                  rp_positions=rows(rp["positions"]), fill=fill, h=h, runs=rows(diag.event_runs(c, run_id)),
                  consts=dict(paper_edge=diag.PAPER_EDGE, paper_stake=diag.PAPER_STAKE), venue=venue, kalshi=V.KALSHI_VENUE)


@app.get("/lab/diagnostics/{run_id}/tape.json", dependencies=[allow(*PRO)])
def diag_tape(request: Request, run_id: int, venue: str = "", c=Depends(conn)):
    """The event's tape for the maker replay in the browser (static/replay.js): markets, fairs per stage, minute
    mids, trades and outcomes, with the page's default knobs. Gzipped (sent as such when the browser accepts it)."""
    if not len(diag.event_runs(c, run_id)):
        raise HTTPException(404)
    _guard(request, c, run_id)
    exchange = "kalshi" if venue == "kalshi" and V.KALSHI_VENUE else "polymarket"
    body = diag.tape(c, run_id, exchange)
    headers = {"Cache-Control": "private, max-age=3600", "Vary": "Accept-Encoding"}
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(body, media_type="application/json", headers=dict(headers, **{"Content-Encoding": "gzip"}))
    import gzip
    return Response(gzip.decompress(body), media_type="application/json", headers=headers)


@app.post("/lab/diagnostics/{run_id}/example", dependencies=[Depends(check_csrf)])
def diag_example(request: Request, run_id: int, fill: str = Form("through"), c=Depends(conn),
                 user=allow(*PRO)):
    """Replay fills become bets by the Polymarket-takers system account (not the demo taker) on the demo
    'maker' account's markets."""
    with get_session() as s:
        maker, taker = U.get_user(s, username="maker"), U.ensure_replay_taker(s)
        if maker is None:
            raise HTTPException(400, "needs the demo 'maker' account")
        out = diag.create_example(s, c, run_id, maker.id, _user_dict(taker), fill=fill)
    audit(request, "diag_example", run_id=run_id, **out)
    return RedirectResponse(f"/lab/diagnostics/{run_id}?msg=Recorded {out['bets']} fills as Polymarket-taker bets on {out['markets']} markets",
                            status_code=303)
