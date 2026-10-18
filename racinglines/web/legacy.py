"""
Old page URLs, redirected to their current routes (bookmarks, links in old messages and alerts).
Routes were renamed to match page names (docs/webapp.md, Pages):

    /            -> /markets              /bet -> /markets           /me -> /positions
    /signals     -> /strategy             /pm  -> /markets/polymarket
    /house       -> /book/quotes          /house/sheet -> /book/quotes/sheet    /house/{id} -> /book/markets/{id}
    /race/{id}   -> /races/{id}           /season/{code} -> /seasons/{code}
    /diag[/{id}] -> /lab/diagnostics[/{id}]                /runs[/{id}] -> /lab/runs[/{id}]
    /markets/{id}, /markets/lookup -> /markets/linked/{id}, /markets/linked/lookup   (admin: linked markets)

GET only: forms post to the current routes. Registered after every real route, so they never shadow one.
"""

from fastapi.responses import RedirectResponse
from starlette.requests import Request

from racinglines.web.app import app

MOVED = {"/bet": "/markets", "/me": "/positions", "/signals": "/strategy",
         "/pm": "/markets/polymarket", "/house": "/book/quotes", "/house/sheet": "/book/quotes/sheet",
         "/diag": "/lab/diagnostics", "/runs": "/lab/runs", "/markets/lookup": "/markets/linked/lookup"}
MOVED_PREFIX = {"/house/{market_id:int}": "/book/markets/{market_id}", "/race/{race_id:int}": "/races/{race_id}",
                "/season/{code}": "/seasons/{code}", "/diag/{run_id:int}": "/lab/diagnostics/{run_id}",
                "/runs/{run_id:int}": "/lab/runs/{run_id}", "/markets/{link_id:int}": "/markets/linked/{link_id}"}


def _redirect(target):
    def moved(request: Request):
        url = target.format(**request.path_params)
        return RedirectResponse(url + (f"?{request.url.query}" if request.url.query else ""), status_code=308)
    return moved


# "/" was here (to /markets) until the landing page moved to the index (owner, 2026-10-12); since 2026-10-13 a
# signed-in browser sees the landing page there too (app.landing_page).
for old, new in {**MOVED, **MOVED_PREFIX}.items():
    app.add_api_route(old, _redirect(new), methods=["GET"], include_in_schema=False)
