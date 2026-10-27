"""
racinglines web app (FastAPI, server-rendered).

Three roles (see roles.py): admin, pro, basic (`maker` / `taker` rows read as pro / basic). Every route needs
a signed-in user (session cookie from /login, or HTTP Basic) and declares which roles may use it (allow(*PRO)
for the maker tools and the Lab); every form POST carries a CSRF token; every meaningful action is
written to activity_log. Run with `racinglines web`.
"""

import contextvars
import hashlib
import importlib.util
import hmac
import os
import re
import secrets
import threading
import time
from urllib.parse import quote
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]      # the repository

from racinglines.cli import present  # noqa: E402  (whether a module the public release may leave out is installed)
from racinglines.db import models as m  # noqa: E402
from racinglines.db.config import get_engine, get_session  # noqa: E402

from racinglines.db import reads as data  # noqa: E402,F401  (route modules import it from here)
from racinglines.markets.polymarket import trade as polymarket

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Auth (users table, roles) + CSRF
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402

from racinglines.web import users as U  # noqa: E402
from racinglines.web import ranges as _ranges  # noqa: E402
from racinglines.web import roles as R  # noqa: E402
from racinglines.web import layout as _layout  # noqa: E402

_SECRET = os.environ.get("APP_SECRET") or secrets.token_hex(32)
CSRF_TOKEN = hmac.new(_SECRET.encode(), b"csrf", hashlib.sha256).hexdigest()

security = HTTPBasic(realm="racinglines", auto_error=False)
SESSION_COOKIE = "rl_session"
SESSION_HOURS = 12
PUBLIC_PATHS = ("/login", "/signin", "/static", "/racinglines101", "/racinglines102", "/signup", "/forgot", "/pitch", "/mybets", "/term")
PUBLIC_EXACT = {"/", "/docs/quickstart/term", "/llms.txt", "/llms-full.txt", "/robots.txt", "/sitemap.xml", "/docs"}  # llms*.txt: the AI-readable docs index (llmstxt.org), built from the public docs only  # public by exact path only: the landing page; a prefix would open /docs/quickstart/term/../..

# failed-login throttle: per client IP, MAX_FAILURES within FAILURE_WINDOW seconds -> 429
MAX_FAILURES = 8
FAILURE_WINDOW = 15 * 60
MAX_TRACKED = 10_000      # addresses kept at most: expired ones go first, then the longest-quiet
_failures: dict[str, list[float]] = {}


def client_ip(request):
    """Real client address; behind a Cloudflare tunnel every request comes from localhost, so the tunnel's header
    is trusted from loopback (or with RACINGLINES_TRUST_PROXY=1) only (users._ip)."""
    return U._ip(request) or "unknown"


def _throttled(ip):
    now = time.time()
    recent = [t for t in _failures.get(ip, []) if now - t < FAILURE_WINDOW]
    if recent:
        _failures[ip] = recent
    else:
        _failures.pop(ip, None)
    return len(recent) >= MAX_FAILURES


def _record_failure(ip):
    now = time.time()
    if ip not in _failures and len(_failures) >= MAX_TRACKED:
        for k in [k for k, ts in _failures.items() if not ts or now - ts[-1] >= FAILURE_WINDOW]:
            del _failures[k]
        while len(_failures) >= MAX_TRACKED:
            del _failures[min(_failures, key=lambda k: _failures[k][-1])]
    _failures.setdefault(ip, []).append(now)


def _session_value(user_id, expires, sid=""):
    """Signed cookie: user id, expiry, and a per-sign-in session id (a demo session's key, web/demo.py)."""
    body = f"{user_id}|{expires}|{sid}"
    sig = hmac.new(_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}|{sig}"


def _session(cookie):
    """(user id, session id) from a valid, unexpired cookie, else (None, None). Accepts the older
    three-part cookie (no session id)."""
    try:
        parts = cookie.split("|")
        body, sig = "|".join(parts[:-1]), parts[-1]
        good = hmac.new(_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        if secrets.compare_digest(sig, good) and int(parts[1]) > time.time():
            return int(parts[0]), (parts[2] if len(parts) == 4 else None)
    except (AttributeError, ValueError, IndexError):
        pass
    return None, None


def _session_user_id(cookie):
    return _session(cookie)[0]


def _user_dict(u):
    return dict(id=u.id, username=u.username, role=_layout.role(R.canonical(u.role)),   # alpha: basic runs as pro
                display_name=u.display_name or u.username)


DOCS_ASSETS = ("assets/", "css/", "img/")     # mkdocs theme files and the docs' own css and images
_public_docs = {"mtime": None, "pages": frozenset()}


def public_doc(path):
    """Whether /docs/<path> is readable signed out (owner, 2026-10-21): a page in the docs nav, the same pages
    /llms-full.txt already publishes (site/public-pages.txt, written at build by scripts/docs/llms_txt.py), or a theme
    asset. Never internal/ (the admin book) or a page outside the nav. The path is normalized first, so
    /docs/quickstart/../internal/ is judged as internal/."""
    if not path.startswith("/docs/"):
        return False
    rel = os.path.normpath("/" + path[len("/docs/"):]).lstrip("/")
    if rel == "internal" or rel.startswith("internal/"):
        return False
    if rel.startswith(DOCS_ASSETS):
        return True
    f = SITE / "public-pages.txt"
    try:
        mtime = f.stat().st_mtime
    except OSError:
        return False
    if mtime != _public_docs["mtime"]:
        _public_docs.update(mtime=mtime, pages=frozenset(ln.strip().strip("/") for ln in f.read_text().splitlines()))
    for suffix in ("index.html", "index.md"):
        if rel == suffix or rel.endswith("/" + suffix):
            rel = rel[:-len(suffix)].rstrip("/")
    return rel in _public_docs["pages"]


def authenticate(request: Request, creds: HTTPBasicCredentials | None = Depends(security)):
    """Signed session cookie (from /login) or HTTP Basic credentials (scripts / curl).
    The user and role are re-read from the database on every request, so
    deactivating a user or changing a role takes effect immediately."""
    if request.url.path in PUBLIC_EXACT or request.url.path.startswith(PUBLIC_PATHS) or public_doc(request.url.path):
        return None
    uid, sid = _session(request.cookies.get(SESSION_COOKIE))
    if uid is not None:
        with get_session() as s:
            u = U.get_user(s, user_id=uid)
            if u and u.active:
                request.state.user = dict(_user_dict(u), sid=sid)
                _layout.guard(request)                  # the alpha layout's archived pages (web/layout.py)
                return request.state.user
    ip = client_ip(request)
    if creds:
        if _throttled(ip):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed logins; try again later")
        with get_session() as s:
            u = U.authenticate(s, creds.username, creds.password)
            if u:
                request.state.user = _user_dict(u)
                _layout.guard(request)
                return request.state.user
        _record_failure(ip)
        U.log(get_engine(), None, "login_failed", request, username=creds.username, via="basic")
    if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
        raise HTTPException(status.HTTP_303_SEE_OTHER, headers={"Location": "/signin?next=" + quote(
                                request.url.path + (f"?{request.url.query}" if request.url.query else ""), safe="/")})
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                        headers={"WWW-Authenticate": 'Basic realm="racinglines"'})


def allow(*roles):
    """Route dependency: only these roles may use the route. Returns the user dict."""
    def dep(request: Request):
        u = getattr(request.state, "user", None)
        if not u or R.canonical(u["role"]) not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"This page is only for: {', '.join(roles)}.")
        return u
    return Depends(dep)


ANY, PRO = R.ANY, R.PRO


def need_mcp():
    """A 404 for the MCP routes when racinglines.mcp is not installed (the public release leaves it out), instead of
    an ImportError and a 500."""
    if not present("mcp"):
        raise HTTPException(404)


def check_csrf(csrf_token: str = Form(...)):
    if not secrets.compare_digest(csrf_token, CSRF_TOKEN):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Bad CSRF token")


def audit(request, action, **detail):
    U.log(get_engine(), getattr(request.state, "user", None), action, request, **detail)


@asynccontextmanager
async def lifespan(app):
    with get_session() as s:
        U.ensure_admin(s)
        if os.environ.get("RACINGLINES_DEMO"):          # `racinglines web --demo`: the demo store, ready to sign in to
            U.ensure_demo_users(s)
            admin = os.environ.get("ADMIN_USERNAME", "admin")
            print(f"Sign in: \"Try the demo\" (maker / {U.DEMO_PASSWORD}), or {admin} with "
                  + ("the ADMIN_PASSWORD you set" if os.environ.get("ADMIN_PASSWORD") else "the password printed when it was created"),
                  flush=True)
    if os.environ.get("RACINGLINES_DEMO"):
        from racinglines.web import accounts as ACC
        ACC.setup(get_engine())                        # the F$ ledger and trading tables, and every account's grant
    threading.Thread(target=_warm_diagnostics, daemon=True).start()
    from racinglines.web import jobs
    jobs.start_worker()
    if present("vm_history"):                      # the hosted site's; the public release leaves it out
        from racinglines import vm_history
        vm_history.start_sampler()                 # CPU, memory, load, disk every 5 min (admin Database page)
    yield


def _warm_diagnostics():
    """Compute the event replays (and their sweeps) once at startup so /diag pages open fast."""
    from racinglines.web import diag
    try:
        with get_engine().connect() as c:
            for e in diag.event_summaries(c):
                diag.replay(c, e["run_id"], **diag.DEFAULT_KNOBS)
    except Exception as ex:  # noqa: BLE001  (a cold cache is only slower)
        print(f"diagnostics warm-up skipped: {ex}")


from racinglines.web import demo as _demo  # noqa: E402
app = FastAPI(title="racinglines", dependencies=[Depends(authenticate), Depends(_demo.guard)], docs_url=None, redoc_url=None,
              lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/media", StaticFiles(directory=HERE / "media", check_dir=False), name="media")   # empty in a wheel (only .gitkeep)
templates = Jinja2Templates(directory=HERE / "templates")


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

from racinglines.web.formats import (LABELS, MONEY_COLS, SIGNED_MONEY_COLS, fbucks, fmt, kind, money,  # noqa: E402,F401
                                     when)

def fmt_epoch(t):
    """Epoch seconds as a short UTC time: 14 Oct 20:05Z (the VM history charts)."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, timezone.utc).strftime("%-d %b %H:%MZ") if t else ""


templates.env.filters.update(fmt=fmt, money=money, fbucks=fbucks, kind=kind, when=when, fmt_epoch=fmt_epoch)
templates.env.globals["LABELS"] = LABELS              # short column headers for the `table` macro
# Demo-only explanations (the demo accounts' story, the demo itself) render in `demo_context` bubbles
# (_macros.html) tagged data-tag="demo-context". RACINGLINES_DEMO_CONTEXT=0 hides them all; for real users,
# delete every `demo_context` call.
DEMO_CONTEXT = {"on": os.environ.get("RACINGLINES_DEMO_CONTEXT", "1") != "0"}
_RENDER_USER = contextvars.ContextVar("render_user", default=None)   # set by render(): imported macros don't see the page's `user`
templates.env.globals["demo_context_on"] = lambda: DEMO_CONTEXT["on"] and _demo.is_demo(_RENDER_USER.get())   # read at render time; demo accounts only
templates.env.globals["csrf_token"] = CSRF_TOKEN
# RACINGLINES_ENV=staging (set only in /etc/racinglines-staging.env on the VM): every response carries
# X-Racinglines-Env so a deploy's smoke check can tell the staging instance from production, and the top bar shows
# the label. Unset (production, local): no header, no label, nothing changes.
ENV_LABEL = os.environ.get("RACINGLINES_ENV", "").strip()
templates.env.globals["env_label"] = lambda: ENV_LABEL
if ENV_LABEL:
    @app.middleware("http")
    async def _env_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Racinglines-Env"] = ENV_LABEL
        return response


# X-Racinglines-Layout: alpha or classic (web/layout.py), so the smoke check knows whether live events and private books
# are archived (404 to non-admins) or open, without a setting of its own.
@app.middleware("http")
async def _layout_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Racinglines-Layout"] = "alpha" if _layout.alpha() else "classic"
    return response
# Maintenance popup on the sign-in page and every app page, dismissed with "ok" once per browser session
# (_site_notice.html). Set to "" to turn it off.
MAINTENANCE_NOTICE = ""      # off since the landing rebuild (owner, 2026-10-08: "drop the modal"); text here turns it back on
templates.env.globals["maintenance_notice"] = lambda: MAINTENANCE_NOTICE   # read at render time
templates.env.globals["signup_on"] = lambda: signup_on()                    # RACINGLINES_SIGNUP=1: beta sign-up
templates.env.globals["is_demo"] = lambda u: _demo.is_demo(u)
templates.env.globals["has_mcp"] = lambda: present("mcp")          # the public release has no MCP server
from racinglines.web import audience as _audience  # noqa: E402  [sport] audience: which sports a user sees
templates.env.globals["sport_shown"] = lambda code, user=None: _audience.shown(user, code)   # signed out: as a user
from racinglines import attribution as _attribution  # noqa: E402  the one-line source note (_macros.html source_note)
templates.env.globals["sources_note"] = _attribution.note


FOLLOW_PROMPT_SKIP = ("/strategy", "/settings", "/signup", "/login", "/logout")


def follow_prompt(user, path=""):
    """The follow-a-strategy popup (_site_notice.html; owner, 2026-10-21, replacing the add-email popup): a signed-in
    registered account that follows nothing, under the alpha layout, off the strategy and settings pages, and not
    between a weekend's last-chance cutoff and the end of its window (follow.open_weekend: a pick is refused then). -> dict(opts, venues) for the popup, or None.
    Read at render time; None if anything fails."""
    if not isinstance(user, dict) or not user.get("id") or _demo.is_demo(user) or not _layout.alpha():
        return None
    if any(path == p or path.startswith(p + "/") for p in FOLLOW_PROMPT_SKIP):
        return None
    try:
        from racinglines.fantasy import follow as FW
        from racinglines.pipelines import profiles as PF
        with get_engine().connect() as c:
            if PF.of_user(c, user["id"]) is not None:
                return None
            chosen = FW.venues_of(c, user["id"])
        if FW.open_weekend() is not None:          # past the last-chance cutoff: a pick would be refused
            return None
        return dict(opts=FW.options(user), venues=[dict(key=k, label=lab, live=live, note=note, on=k in chosen)
                                                   for k, lab, live, note in FW.VENUES])
    except Exception:                                   # noqa: BLE001  the popup never breaks a page
        return None


templates.env.globals["follow_prompt"] = follow_prompt
templates.env.globals["ui_range"] = _ranges.ui_range                             # capped numeric inputs (web/ranges.py)
from racinglines.fantasy import ledger as _FL  # noqa: E402
templates.env.globals["fantasy_on"] = _FL.on                                    # /trade and /history (staging only)
templates.env.globals["alpha_layout"] = _layout.alpha                           # the four-tab alpha bar (web/layout.py)
templates.env.globals["pitch_on"] = lambda: (ROOT / "pitch.html").exists()      # /pitch serves it; the public release has no deck
# the stylesheet's and scripts' URLs carry the newest static file's mtime, so a change is a new URL: no stale copy from the browser or
# Cloudflare's edge cache (read at render time)
templates.env.globals["css_v"] = lambda: int(max(p.stat().st_mtime for p in (HERE / "static").glob("*.*")))


def _seen_since(user):
    """A demo session's "signals seen up to" time (web/demo.py overlay), else None (the database's seen_at)."""
    from racinglines.web import demo
    return demo.prefs(user["sid"]).get("signals_seen_at") if demo.is_demo(user) and user.get("sid") else None


def _signals_nav(user):
    """The Signals link for the nav (users with a strategy profile, and admins): unread signals, and the
    account's paper summary (profile, bankroll, P&L) for the banner on the home and trading pages."""
    if not user:
        return None
    from racinglines.pipelines import sport_paper as SP        # RACINGLINES_SPORT_PAPER=1: NASCAR / MotoGP demo rows count
    try:
        with get_engine().connect() as c:
            r = c.execute(text("""SELECT prefs ? 'strategy_profile', prefs->'strategy_profile'->>'name',
                                         (prefs->'strategy_profile'->'bankroll'->>'start')::float,
                                         (SELECT count(*) FROM strategy_signals WHERE user_id = :u AND seen_at IS NULL
                                            AND status <> 'expired'
                                            AND NOT coalesce(detail->>'sport', '') = ANY(CAST(:hs AS text[]))
                                            AND (CAST(:since AS timestamptz) IS NULL OR created_at > CAST(:since AS timestamptz))),
                                         (SELECT sum(cash + yes_shares * coalesce(outcome::int, mark)
                                                 + no_shares * (1 - coalesce(outcome::int, mark)))
                                            FROM paper_positions WHERE user_id = :u AND (venue = 'polymarket'
                                                 OR (:sp AND event_key IN (""" + SP.SPORT_KEYS
                                                 + " AND NOT detail->>'sport' = ANY(CAST(:hs AS text[]))" + """)))),
                                         (SELECT sum(cash + yes_shares * coalesce(outcome::int, mark)
                                                 + no_shares * (1 - coalesce(outcome::int, mark)))
                                            FROM paper_positions WHERE user_id = :u AND venue = 'private')
                                  FROM users WHERE id = :u"""),
                          dict(u=user["id"], since=_seen_since(user), sp=SP.enabled(),
                               hs=list(_audience.hidden(user)))).first()    # [sport] audience: hidden sports don't count
    except Exception:                                   # noqa: BLE001  the nav never breaks a page
        return None
    if r is None or not (r[0] or user["role"] == "admin"):
        return None
    pnl = float(r[4] or 0.0)
    return dict(unread=int(r[3] or 0), profile=r[1], start=r[2], pnl=pnl, private=None if r[5] is None else float(r[5]),
                balance=(r[2] + pnl) if r[2] else None, ret=(pnl / r[2]) if r[2] else None)


def render(request, name, **ctx):
    ctx.setdefault("trading", polymarket.TradingConfig.from_env())
    ctx.setdefault("user", getattr(request.state, "user", None))
    ctx.setdefault("signals_nav", _signals_nav(ctx["user"]))
    ctx.setdefault("schema_exchanges", _audience.venues(ctx["user"], V.SCHEMA_EXCHANGES))   # schema venues whose switch is on, for this user
    ctx.setdefault("tapes", V.TAPES)                            # RACINGLINES_TAPES=1: the tape-only sports' market data
    try:                                               # a live event (pipelines/live.py): the Live tab, green while
        from racinglines.pipelines.live import state      # it runs, grey once it is over (a replay)
        ctx.setdefault("live_nav", state() if ctx["user"] else None)
    except Exception:                                  # noqa: BLE001
        ctx.setdefault("live_nav", False)
    from racinglines.web import demo
    u = ctx["user"]
    _RENDER_USER.set(u)
    ctx.setdefault("storage_ns", f"demo.{u.get('sid')}." if demo.is_demo(u) and u.get("sid") else "")
    return templates.TemplateResponse(request, name, ctx)


# Browser requests (Accept: text/html) get a readable error page for 403 / 404 / 422 / 429; JSON and API clients keep
# FastAPI's JSON error. A throttled sign-in goes back to the form with a sentence instead.
ERROR_TEXT = {403: "You don't have access to this page.", 404: "That page doesn't exist, or it has moved.",
              422: "Something in that link or form wasn't valid.", 429: "Too many attempts. Wait a few minutes and try again."}


def _wants_html(request):
    return "text/html" in request.headers.get("accept", "") and not request.url.path.startswith("/api/")


def _error_page(request, code):
    user = getattr(request.state, "user", None)
    try:
        resp = render(request, "error.html", code=code, message=ERROR_TEXT[code], home="/markets" if user else "/")
        resp.status_code = code
        return resp
    except Exception:                                  # noqa: BLE001  never let the error page itself fail
        return None


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 429 and request.method == "POST" and request.url.path == "/login" and _wants_html(request):
        return RedirectResponse("/signin?error=throttled", status_code=303)
    if exc.status_code in ERROR_TEXT and _wants_html(request):
        resp = _error_page(request, exc.status_code)
        if resp is not None:
            return resp
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    if _wants_html(request):
        resp = _error_page(request, 422)
        if resp is not None:
            return resp
    return await request_validation_exception_handler(request, exc)


def rows(df):
    """DataFrame -> list of dicts for templates. ID columns come back from pandas as
    floats when they have gaps (e.g. 45.0); turn them into ints / None so links work."""
    if df is None:
        return []
    df = df.copy()
    for col in df.columns:
        if (col == "id" or col.endswith("_id") or col == "run") and df[col].dtype.kind in "fiu":
            df[col] = pd.Series([None if pd.isna(v) else int(v) for v in df[col]], index=df.index, dtype=object)
    # missing values -> None, so templates' "is not none" checks work (NaN would print "nan")
    return [{k: (None if not isinstance(v, (dict, list, str)) and pd.isna(v) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def conn():
    with get_engine().connect() as c:
        yield c


SITE = ROOT / "site"     # the built docs (mkdocs build -d site; the pre-push hook builds it); served by web/info_routes.py


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------

from racinglines.paths import HOSTED as LANDING  # noqa: E402  (racinglines.bet's landing page; not in the public package)
templates.env.globals["site_url"] = "https://racinglines.bet" if LANDING else ""   # canonical/OG links, hosted site only


@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request, next: str = "/"):
    """The public landing page, at the site's index (owner, 2026-10-12). A signed-in browser sees it too, with an
    "Open the app" link in place of "Sign in" (owner, 2026-10-13: the Home link must reach the landing page)."""
    signed_in = _session(request.cookies.get(SESSION_COOKIE))[0] is not None
    if not LANDING:                 # the public package (oss/manifest.toml leaves the branded landing out): sign-in is the index
        if signed_in:
            return RedirectResponse("/markets", status_code=303)
        return signin_page(request, next=next)
    return templates.TemplateResponse(request, "login.html", dict(next=next, support=support_email(), signed_in=signed_in))


@app.get("/signin", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
def signin_page(request: Request, next: str = "/", error: str = ""):
    """Sign in (owner, 2026-10-12: off the landing, on its own page). /login shows the same page, so old links,
    smoke.sh and `vm.sh status` keep a 200 there; the form posts to /login."""
    return templates.TemplateResponse(request, "signin.html", dict(next=next, error=error, support=support_email()))


def _safe_next(nxt):
    """The post-login target: a same-site relative path only (one leading slash, no scheme, host, backslash or
    control characters), else /markets."""
    if (not nxt.startswith("/") or nxt.startswith("//") or "\\" in nxt or any(ord(ch) < 32 or ord(ch) == 127 for ch in nxt)):
        return "/markets"
    return nxt


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/")):
    ip = client_ip(request)
    if _throttled(ip):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many failed logins; try again in 15 minutes")
    with get_session() as s:
        u = U.authenticate(s, username, password)
        user = _user_dict(u) if u else None
    if user is None:
        _record_failure(ip)
        U.log(get_engine(), None, "login_failed", request, username=username)
        return RedirectResponse(f"/signin?next={quote(_safe_next(next), safe='/')}&error=1", status_code=303)
    target = _safe_next(next) if next != "/" else "/markets"
    return _start_session(request, user, target)


def _start_session(request, user, target):
    """Sign `user` in: a fresh session cookie, logged, then a redirect to `target`."""
    from racinglines.web import demo
    sid = demo.new_sid()                           # a fresh session: a demo account starts from its baseline
    U.log(get_engine(), user, "login", request, sid=sid, demo=demo.is_demo(user) or None)
    resp = RedirectResponse(target, status_code=303)
    expires = int(time.time()) + SESSION_HOURS * 3600
    https = request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"
    resp.set_cookie(SESSION_COOKIE, _session_value(user["id"], expires, sid), max_age=SESSION_HOURS * 3600,
                    httponly=True, samesite="lax", secure=https)
    return resp


@app.get("/logout")
def logout(request: Request):
    from racinglines.web import demo
    audit(request, "logout")
    sid = _session(request.cookies.get(SESSION_COOKIE))[1]
    if sid:
        demo.reset(sid)
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ---------------------------------------------------------------------------
# User settings: email, exchange preferences, sports subscriptions
# ---------------------------------------------------------------------------

def _prefs(user_id):
    with get_session() as s:
        return dict(s.query(m.User).filter_by(id=user_id).first().prefs or {})


def _mcp_context(user):
    if not present("mcp"):                         # the public release has no MCP server: Settings hides its card
        return dict(mcp_available=False)
    from racinglines.mcp import auth as mcp_auth, oauth
    prefs = _prefs(user["id"])
    return dict(mcp_url=oauth.MCP_URL, mcp_issued_at=(prefs.get("mcp") or {}).get("issued_at"),
                mcp_clients=(prefs.get("mcp_oauth") or {}).get("clients") or {},
                mcp_allowed=user["role"] in mcp_auth.ROLES, mcp_roles=", ".join(mcp_auth.ROLES), mcp_available=True)


@app.get("/settings", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def settings_page(request: Request):
    """User settings: email, exchange preferences, sports subscriptions, MCP token. Demo users cannot access."""
    user = request.state.user
    if _demo.is_demo(user):
        raise HTTPException(404)
    prefs = _prefs(user["id"])
    return templates.TemplateResponse(request, "settings.html", dict(
        email=prefs.get("email", ""),
        exchange=prefs.get("exchange", "polymarket"),
        sports=prefs.get("sports", ["f1"]),
        **_mcp_context(user), hidden_sports=_audience.hidden(user),
    ))


@app.post("/settings", dependencies=[Depends(check_csrf), allow(*ANY)])
def update_settings(request: Request, email: str = Form(""), exchange: str = Form("polymarket"),
                   sports: list[str] = Form(None)):
    """Update user settings: email, exchange preferences, sports subscriptions."""
    user = request.state.user
    if _demo.is_demo(user):
        raise HTTPException(404)
    sports = sports or []
    email = email.strip()
    if email and not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return templates.TemplateResponse(request, "settings.html", dict(
            email=email, exchange=exchange, sports=sports, error="Invalid email address", **_mcp_context(user),
            hidden_sports=_audience.hidden(user),
        ), status_code=400)
    with get_session() as s:     # merge: the session user carries no prefs, and prefs also hold the MCP token and seen-at marks
        u = s.query(m.User).filter_by(id=user["id"]).first()
        u.prefs = {**(u.prefs or {}), "email": email, "exchange": exchange, "sports": sports or ["f1"]}
        s.commit()
    audit(request, "update_settings", email=bool(email), exchange=exchange, sports=sports)
    return templates.TemplateResponse(request, "settings.html", dict(
        email=email, exchange=exchange, sports=sports, success="Settings saved", **_mcp_context(user),
        hidden_sports=_audience.hidden(user),
    ))


# ---------------------------------------------------------------------------
# MCP token from the Settings page: the same token `racinglines mcp token <account>` issues (racinglines/mcp/auth.py),
# one per account, shown once; the hosted MCP server accepts it at once.
# ---------------------------------------------------------------------------

@app.post("/api/settings/token/generate", dependencies=[Depends(need_mcp), Depends(check_csrf), allow(*ANY)])
def generate_mcp_token(request: Request):
    from racinglines.mcp import auth as mcp_auth
    user = request.state.user
    if _demo.is_demo(user):
        raise HTTPException(403, "Demo accounts cannot get an MCP token")
    try:
        token = mcp_auth.new_token(get_engine(), user["username"])
    except ValueError as ex:
        raise HTTPException(403, str(ex)) from ex
    audit(request, "generate_mcp_token")
    from racinglines.mcp import oauth
    return {"token": token, "url": oauth.MCP_URL}


@app.post("/api/settings/token/revoke", dependencies=[Depends(need_mcp), Depends(check_csrf), allow(*ANY)])
def revoke_mcp_token(request: Request):
    from racinglines.mcp import auth as mcp_auth
    user = request.state.user
    if _demo.is_demo(user):
        raise HTTPException(403, "Demo accounts have no MCP token")
    if not mcp_auth.revoke(get_engine(), user["username"]):
        raise HTTPException(404, "No MCP token to revoke")
    audit(request, "revoke_mcp_token")
    return {"status": "revoked"}


@app.post("/api/settings/mcp/disconnect", dependencies=[Depends(need_mcp), Depends(check_csrf), allow(*ANY)])
def disconnect_mcp(request: Request):
    from racinglines.mcp import oauth
    user = request.state.user
    if _demo.is_demo(user):
        raise HTTPException(403, "Demo accounts have no MCP sign-ins")
    oauth.disconnect(get_engine(), user["id"])
    audit(request, "mcp_disconnect")
    return {"status": "disconnected"}


# ---------------------------------------------------------------------------
# MCP sign-in (racinglines/mcp/oauth.py): the MCP server's /authorize sends the browser here with a signed request;
# a signed-in account allowed MCP access presses Allow and goes back to the client with a code.
# ---------------------------------------------------------------------------

@app.get("/mcp/authorize", response_class=HTMLResponse, dependencies=[Depends(need_mcp), allow(*ANY)])
def mcp_authorize_page(request: Request, req: str = ""):
    from racinglines.mcp import oauth
    user = request.state.user
    info = oauth.request_info(req)
    allowed = not _demo.is_demo(user) and oauth.account(get_engine(), user_id=user["id"]) is not None
    resp = render(request, "mcp_authorize.html", req=req, info=info, allowed=allowed, roles=", ".join(oauth.auth.ROLES))
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Frame-Options"] = "DENY"                       # an Allow button must not be clickable inside another site
    resp.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    return resp


@app.post("/mcp/authorize", dependencies=[Depends(need_mcp), Depends(check_csrf), allow(*ANY)])
def mcp_authorize(request: Request, req: str = Form(...), decision: str = Form(...)):
    from racinglines.mcp import oauth
    user = request.state.user
    if decision != "allow":
        url = oauth.deny_url(req)
    elif _demo.is_demo(user) or oauth.account(get_engine(), user_id=user["id"]) is None:
        raise HTTPException(403, "This account may not use the MCP server")
    else:
        url = oauth.approve(req, user)
        audit(request, "mcp_authorize", client=(oauth.request_info(req) or {}).get("client"))
    if url is None:
        raise HTTPException(400, "This sign-in request expired: start again from your MCP client")
    return RedirectResponse(url, status_code=303)


# ---------------------------------------------------------------------------
# Beta sign-up (/signup, RACINGLINES_SIGNUP=1, off by default; web/accounts.py): username and password, as pro. One
# transaction creates the account (scrypt hash) and its 1,000 fantasy-bucks grant, then signs the person in.
# Until `racinglines users setup` has made the ledger, the page says sign-up isn't open yet.
# ---------------------------------------------------------------------------

SIGNUP_ROLE = "pro"                         # every beta sign-up starts as pro (owner, 2026-10-01); tiers later
SIGNUP_MAX_PER_IP = 10                      # sign-up attempts per client IP per hour (successful or not)
_signup_hits: dict[str, list[float]] = {}


def signup_on() -> bool:
    return os.environ.get("RACINGLINES_SIGNUP", "0") == "1"


def _signup_ready() -> bool:
    from racinglines.web import accounts as ACC
    try:
        with get_engine().connect() as c:
            return bool(ACC.ready(c))
    except Exception:                               # noqa: BLE001  no database: fail soft, sign-up closed
        return False


def _signup_page(request, form=None, error="", status_code=200):
    from racinglines.web import accounts as ACC
    return templates.TemplateResponse(request, "signup.html", dict(form=form or {}, error=error,
                                      ready=_signup_ready(), grant=ACC.SIGNUP_GRANT,
                                      pw_min=ACC.PASSWORD_MIN), status_code=status_code)


SUPPORT_EMAIL = "hello@racinglines.bet"      # password resets and business enquiries (the landing page) come here


def support_email():
    """SUPPORT_EMAIL on the hosted site only; a self-hosted copy (no branded landing) names no address."""
    return SUPPORT_EMAIL if LANDING else ""


@app.get("/forgot", response_class=HTMLResponse)
def forgot_password(request: Request):
    """Forgot your password: no reset email is sent (older accounts have no email), so the page writes a reset request to SUPPORT_EMAIL for the
    person to send; an admin resets it at /admin/users and replies with a one-time password."""
    if not signup_on():
        raise HTTPException(404)
    return templates.TemplateResponse(request, "forgot.html", dict(support=support_email()))


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    if not signup_on():
        raise HTTPException(404)
    return _signup_page(request)


@app.post("/signup", response_class=HTMLResponse)
def signup(request: Request, username: str = Form(""), password: str = Form(""), confirm: str = Form(""),
           email: str = Form(""), adult: str = Form(""), website: str = Form("")):
    from racinglines.web import accounts as ACC
    if not signup_on():
        raise HTTPException(404)
    if website:                                     # honeypot: people never see this field, bots fill it
        raise HTTPException(400)
    ip = client_ip(request)
    now = time.time()
    hits = [t for t in _signup_hits.get(ip, []) if now - t < 3600]
    if len(hits) >= SIGNUP_MAX_PER_IP:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many sign-up attempts; try again in an hour")
    _signup_hits[ip] = hits + [now]
    if not _signup_ready():
        return _signup_page(request, status_code=503)
    username = username.strip().lower()
    tier = SIGNUP_ROLE
    email = email.strip().lower()
    form = dict(username=username[:30], email=email[:254])
    with get_engine().begin() as c:
        error = (ACC.check_username(c, username) or ACC.check_password(password, confirm, username)
                 or ACC.check_email(c, email) or ("Please confirm you're 18 or older." if not adult else ""))
        if error:
            return _signup_page(request, form, error, status_code=400)
        uid = c.execute(text("""INSERT INTO users (username, display_name, role, password_hash, active, prefs)
                                VALUES (:u, :u, :r, :h, true, jsonb_build_object('email', CAST(:e AS text))) RETURNING id"""),
                        dict(u=username, r=tier, h=U.hash_password(password), e=email)).scalar()
        ACC.grant(c, uid)
    # No starting strategy (owner, 2026-10-11: "no default strategy until they select one"; replaces the TAKER_TOP
    # round-robin of 2026-10-06): an account paper-trades nothing until its user picks a strategy.
    with get_session() as s:
        user = _user_dict(U.get_user(s, user_id=uid))
    U.log(get_engine(), user, "signup", request, grant=ACC.SIGNUP_GRANT)
    return _start_session(request, user, "/markets")


# Kalshi (RACINGLINES_KALSHI_VENUE=1, off by default) and the schema venues: their switches, read by render()
from racinglines.markets import venues as V  # noqa: E402


# Route groups split out of this file; each imports the shared helpers above and registers its routes.
from racinglines.web import info_routes  # noqa: E402,F401  (runs, events, athletes, docs, intro, pitch)
from racinglines.web import order_routes  # noqa: E402,F401  (linked markets and Polymarket orders, admin)
from racinglines.web import book_routes  # noqa: E402,F401  (the private book and the taker's calls)
from racinglines.web import exchange_routes  # noqa: E402,F401  (exchange boards and tapes)
from racinglines.web import diag_routes  # noqa: E402,F401  (single-event diagnostics)
# /mybets, the owner's bets calendar from the data folder: the hosted site only, the public package leaves it out
if importlib.util.find_spec("racinglines.web.mybets"):
    importlib.import_module("racinglines.web.mybets")
from racinglines.web import term_routes  # noqa: E402,F401  (/docs/quickstart/term, recorded CLI runs)
from racinglines.web import strategy_pick  # noqa: E402,F401  (/strategy/pick, the account's own strategy)
from racinglines.web import fantasy_routes  # noqa: E402,F401  (/trade, /history: fantasy trading; staging only)
from racinglines.web.book_routes import bet_markets, polymarket_calls  # noqa: E402,F401  (used by views.py)


async def _f1_live_ws(websocket):
    """Live F1 timing for one viewer (web/f1_live.py). A plain Starlette route: the app's dependencies (HTTP Basic,
    the demo guard) can't run on a websocket, so the session cookie is checked here."""
    from starlette.concurrency import run_in_threadpool

    from racinglines.web import f1_live

    def who():
        uid = _session_user_id(websocket.cookies.get(SESSION_COOKIE))
        if uid is None:
            return None
        with get_session() as s:
            u = U.get_user(s, user_id=uid)
            return _user_dict(u) if u and u.active else None

    await f1_live.serve(websocket, await run_in_threadpool(who))


from starlette.routing import WebSocketRoute  # noqa: E402

app.router.routes.append(WebSocketRoute("/ws/f1/live", _f1_live_ws))


from racinglines.web import admin  # noqa: E402,F401  (registers /admin routes)
from racinglines.web import views  # noqa: E402,F401  (registers the board, race, season, book and lab pages)
from racinglines.web import api  # noqa: E402,F401  (the read-only JSON API; off unless RACINGLINES_JSON_API=1)
from racinglines.web import legacy  # noqa: E402,F401  (old page URLs redirect to the current routes; register last)
