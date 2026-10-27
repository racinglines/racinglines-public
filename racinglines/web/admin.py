"""
Admin-only pages: overview, activity log, users (create / role / reset password /
deactivate), per-user trading view, database explorer (browse, edit, delete rows)
and a SQL console (read-only by default; writes need an explicit toggle).
Everything that changes data is written to activity_log.
"""

import json
from datetime import date, datetime
from urllib.parse import quote

from fastapi import Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, Integer, Text, cast, delete, func, select, text, update
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from racinglines.cli import present
from racinglines.db import models as m
from racinglines.db.config import get_engine, get_session

from racinglines.markets import private_book as house
from racinglines.pipelines import profiles as PF
from racinglines.web import roles as R
from racinglines.web import accounts as ACC
from racinglines.web import jobs as J
from racinglines.web import users as U
from racinglines.web.app import app, allow, audit, check_csrf, conn, data, need_mcp, render, rows

ADMIN = [allow("admin")]
HIDDEN_COLUMNS = {("users", "password_hash")}
PROTECTED_USERS = {"maker", "taker", U.REPLAY_TAKER}    # the demo logins and the replay counterparty
PAGE = 50
SQL_ROW_LIMIT = 500


# ---------------------------------------------------------------------------
# Overview, activity, users
# ---------------------------------------------------------------------------

USERS_SQL = """
SELECT u.id, u.username, u.role, u.display_name, u.active, u.created_at,
       (SELECT max(ts) FROM activity_log al WHERE al.user_id = u.id) AS last_seen,
       (SELECT count(*) FROM activity_log al WHERE al.user_id = u.id AND al.action = 'login') AS logins,
       (SELECT count(*) FROM house_markets hm WHERE hm.maker_id = u.id) AS markets_made,
       (SELECT count(*) FROM house_bets hb JOIN house_markets hm ON hm.id = hb.market_id
         WHERE hm.maker_id = u.id AND hb.status <> 'void') AS bets_against,
       (SELECT count(*) FROM house_bets hb WHERE hb.taker_id = u.id) AS bets_placed,
       (SELECT coalesce(sum(stake), 0) FROM house_bets hb WHERE hb.taker_id = u.id) AS staked
FROM users u ORDER BY u.role, u.username
"""


def bet_totals(bets):
    """Taker totals for the admin overview: people's bets only. The replay counterparty
    (`polymarket-takers`, Polymarket's real takers filling a replayed maker) is counted apart."""
    replay = bets["taker"] == U.REPLAY_TAKER
    people, rp = bets[~replay], bets[replay]
    return dict(bets=len(people), staked=float(people["stake"].sum()), taker_pnl=float(people["pnl"].sum()),
                replay_bets=len(rp), replay_pnl=float(rp["pnl"].sum()))


@app.get("/admin", response_class=HTMLResponse, dependencies=ADMIN)
def admin_home(request: Request, c=Depends(conn)):
    users = rows(data.q(c, USERS_SQL))
    recent = rows(data.q(c, "SELECT id, ts, username, role, action, detail, ip FROM activity_log ORDER BY id DESC LIMIT 25"))
    book = house.book(c)
    totals = dict(markets=len(book), open=int((book["status"] == "open").sum()) if len(book) else 0,
                  **bet_totals(house.taker_bets(c)), worst=float(book["worst"].sum()) if len(book) else 0.0)
    return render(request, "admin.html", users=users, recent=recent, totals=totals)


@app.get("/admin/activity", response_class=HTMLResponse, dependencies=ADMIN)
def admin_activity(request: Request, user: str = "", action: str = "", limit: int = 200, c=Depends(conn)):
    df = data.q(c, """
        SELECT id, ts, username, role, action, detail, ip, path FROM activity_log
        WHERE (CAST(:u AS text) IS NULL OR username = CAST(:u AS text))
          AND (CAST(:a AS text) IS NULL OR action = CAST(:a AS text))
        ORDER BY id DESC LIMIT :lim""", u=user or None, a=action or None, lim=min(limit, 2000))
    actions = [r["action"] for r in rows(data.q(c, "SELECT DISTINCT action FROM activity_log ORDER BY action"))]
    usernames = [r["username"] for r in rows(data.q(c, "SELECT DISTINCT username FROM activity_log WHERE username IS NOT NULL ORDER BY username"))]
    return render(request, "admin_activity.html", events=rows(df), actions=actions, usernames=usernames,
                  f_user=user, f_action=action, limit=limit)


@app.get("/admin/users", response_class=HTMLResponse, dependencies=ADMIN)
def admin_users(request: Request, msg: str = "", c=Depends(conn)):
    return _users_page(request, c, msg)


def _users_page(request, c, msg="", onetime=None):
    """The users page. onetime: (username, temporary password) after a reset, shown once and never stored."""
    resp = render(request, "admin_users.html", users=rows(data.q(c, USERS_SQL)), roles=U.ROLES, msg=msg,
                  balances=ACC.balances(c), ledger=ACC.ready(c), grant=ACC.SIGNUP_GRANT, onetime=onetime,
                  pw_min=ACC.PASSWORD_MIN)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/admin/users", dependencies=[Depends(check_csrf), allow("admin")])
def admin_user_create(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...),
                      display_name: str = Form("")):
    try:
        username = username.strip().lower()
        with get_engine().begin() as c:
            error = ACC.check_username(c, username) or ACC.check_password(password, password, username)   # one password field: no confirm to match
            if error:
                raise ValueError(error)
            user_id = c.execute(text("""INSERT INTO users (username, display_name, role, password_hash, active)
                                        VALUES (:u, :d, :r, :h, true) RETURNING id"""),
                                dict(u=username, d=display_name or username, r=role,
                                     h=U.hash_password(password))).scalar()
        audit(request, "user_create", target=username, role=role, user_id=user_id)
        msg = f"Created {role} {username}."
    except ValueError as e:
        msg = f"Error: {e}"
    except Exception as e:
        msg = f"Error: {e}"
    return RedirectResponse(f"/admin/users?msg={msg}", status_code=303)


@app.post("/admin/users/{user_id}", dependencies=[Depends(check_csrf), allow("admin")])
def admin_user_update(request: Request, user_id: int, role: str = Form(...), active: str = Form(""),
                      display_name: str = Form(""), new_password: str = Form("")):
    me = request.state.user
    with get_session() as s:
        u = s.get(m.User, user_id)
        if u is None:
            raise HTTPException(404)
        if u.id == me["id"] and (role != "admin" or not active):
            return RedirectResponse("/admin/users?msg=Error: you can't demote or deactivate yourself.", status_code=303)
        role = R.canonical(role)
        if role not in U.ROLES:
            raise HTTPException(400)
        if not R.allowed_profile(role, PF.of_user(s.connection(), user_id), user_id=user_id, legacy_ok=True):
            return RedirectResponse(f"/admin/users?msg=Error: a {role} account can't run this strategy profile; "
                                    "clear or change the profile first.", status_code=303)
        changes = {}
        if u.role != role:
            changes["role"] = [u.role, role]
            u.role = role
        if u.active != bool(active):
            changes["active"] = [u.active, bool(active)]
            u.active = bool(active)
        if display_name and display_name != u.display_name:
            changes["display_name"] = [u.display_name, display_name]
            u.display_name = display_name
        if new_password:
            if len(new_password) < 8:
                return RedirectResponse("/admin/users?msg=Error: password must be at least 8 characters.",
                                        status_code=303)
            u.password_hash = U.hash_password(new_password)
            changes["password"] = "reset"
        s.commit()
        username = u.username
    audit(request, "user_update", target=username, user_id=user_id, changes=changes)
    return RedirectResponse(f"/admin/users?msg=Updated {username}.", status_code=303)


@app.post("/admin/users/{user_id}/reset-password", response_class=HTMLResponse,
          dependencies=[Depends(check_csrf), allow("admin")])
def admin_user_reset_password(request: Request, user_id: int, c=Depends(conn)):
    """A new random password, shown to the admin once on this response (no-store) and stored only as a hash.
    Existing sessions of that account end at their expiry (cookies carry no password)."""
    temp = ACC.temp_password()
    with get_session() as s:
        u = s.get(m.User, user_id)
        if u is None:
            raise HTTPException(404)
        if u.id == request.state.user["id"]:
            return RedirectResponse("/admin/users?msg=Error: change your own password in the New password box.",
                                    status_code=303)
        u.password_hash = U.hash_password(temp)
        s.commit()
        username = u.username
    audit(request, "user_password_reset", target=username, user_id=user_id)       # never the password
    return _users_page(request, c, f"Password reset for {username}.", onetime=(username, temp))


@app.post("/admin/users/{user_id}/delete", dependencies=[Depends(check_csrf), allow("admin")])
def admin_user_delete(request: Request, user_id: int):
    """Remove an account that has no market or bet history (its ledger rows, signals and paper positions go with it).
    One with history is refused: deactivate it instead, so the book keeps who made and took each bet."""
    with get_session() as s:
        u = s.get(m.User, user_id)
        if u is None:
            raise HTTPException(404)
        username = u.username
        if u.id == request.state.user["id"] or R.canonical(u.role) == "admin" or username in PROTECTED_USERS:
            return RedirectResponse(f"/admin/users?msg=Error: {username} can't be removed.", status_code=303)
        history = s.execute(text("""SELECT (SELECT count(*) FROM house_markets WHERE maker_id = :u)
                                         + (SELECT count(*) FROM house_bets WHERE taker_id = :u)"""),
                            dict(u=user_id)).scalar()
        if history:
            return RedirectResponse(f"/admin/users?msg=Error: {username} has markets or bets; untick Active to "
                                    "deactivate it instead.", status_code=303)
        s.delete(u)
        s.commit()
    audit(request, "user_delete", target=username, user_id=user_id)
    return RedirectResponse(f"/admin/users?msg=Removed {username}.", status_code=303)


@app.get("/admin/users/{user_id}", response_class=HTMLResponse, dependencies=ADMIN)
def admin_user_detail(request: Request, user_id: int, c=Depends(conn)):
    u = data.q(c, "SELECT id, username, role, display_name, active, created_at FROM users WHERE id = :i", i=user_id)
    if not len(u):
        raise HTTPException(404)
    book = house.book(c, maker_id=user_id)
    bets = house.taker_bets(c, user_id)
    activity = data.q(c, """SELECT id, ts, action, detail, ip FROM activity_log WHERE user_id = :i
                            ORDER BY id DESC LIMIT 200""", i=user_id)
    maker_summary = dict(markets=len(book), bets=int(book["bets"].sum()), staked=float(book["staked"].sum()),
                         ev=float(book["ev"].sum()), worst=float(book["worst"].sum()),
                         settled=float(book["settled_pnl"].dropna().sum())) if len(book) else None
    taker_summary = dict(bets=len(bets), staked=float(bets["stake"].sum()), pnl=float(bets["pnl"].sum()),
                         open=int((bets["status"] == "open").sum())) if len(bets) else None
    cands = [dict(id=i, name=p.get("name"), strategy=p.get("strategy")) for i, p in PF._candidates(c)]
    combos = [dict(code=code, name=pr["name"], members=", ".join(f"{m} x{w:g}" for m, w in pr["members"]))
              for code, pr in PF.PROFILES.items() if PF.is_combo(pr)]
    slots = [dict(s, profile=PF.of_user(c, user_id, s["venue"], s["sport"])) for s in profile_slots()]
    return render(request, "admin_user.html", u=rows(u)[0], book=rows(book), bets=rows(bets), activity=rows(activity),
                  maker_summary=maker_summary, taker_summary=taker_summary, profile=slots[0]["profile"], slots=slots,
                  candidates=cands, combos=combos, basic_draw=", ".join(R.basic_members(user_id)))


def profile_slots():
    """Where a user can hold a strategy profile: one slot per sport with a pricing model and exchange that lists it
    (sports/<code>.toml [markets] venues, exchanges/<code>.toml [sports]), F1 on Polymarket first (the one live
    paper signals run today, pipelines/signals.py). -> [dict(sport, venue, key, label)]."""
    out = [dict(sport=code, venue=venue, key=f"{code}:{venue}", label=f"{J.sport_name(code)} · {J.venue_name(venue)}")
           for code in J.modeled_sports() for venue in J.sport_venues(code)]
    return sorted(out, key=lambda s: s["key"] != "f1:polymarket")


@app.post("/admin/users/{user_id}/profile", dependencies=[Depends(check_csrf), allow("admin")])
def admin_user_profile(request: Request, user_id: int, candidate_id: str = Form(""), slot: str = Form("f1:polymarket")):
    """Assign a Lab candidate as the user's strategy profile (live paper signals), or clear it, for one sport and
    exchange (`slot`, profile_slots). candidate_id may also be a blend's code (profiles.COMBOS) or "basic": the
    account's own draw (roles.basic_profile)."""
    where = next((s for s in profile_slots() if s["key"] == slot), None)
    if where is None:
        raise HTTPException(400, "unknown sport and exchange")
    venue, sport = where["venue"], where["sport"]
    with get_engine().begin() as c:
        old = PF.of_user(c, user_id, venue, sport)
        new = R.basic_profile(c, user_id) if candidate_id == "basic" else PF.load(c, candidate_id) if candidate_id else None
        role = c.execute(text("SELECT role FROM users WHERE id = :u"), dict(u=user_id)).scalar()
        if not R.allowed_profile(role, new, user_id=user_id):
            names = ", ".join(R.basic_profile_names()) or "none"
            return RedirectResponse(f"/admin/users/{user_id}?msg=Error: a basic account may only run: {names}.",
                                    status_code=303)
        if new is not None and slot != "f1:polymarket":     # the slot it was assigned to travels with it (as assign_demo)
            new = dict(new, venue=venue, sport=sport)
        PF.assign(c, user_id, new, venue, sport)
    audit(request, "user_profile", user_id=user_id, old=(old or {}).get("name"), new=(new or {}).get("name"),
          candidate_id=(new or {}).get("candidate_id"), **({} if slot == "f1:polymarket" else dict(venue=venue, sport=sport)))
    return RedirectResponse(f"/admin/users/{user_id}", status_code=303)


# ---------------------------------------------------------------------------
# Database explorer
# ---------------------------------------------------------------------------

def _table(name):
    t = m.Base.metadata.tables.get(name)
    if t is None:
        raise HTTPException(404, f"no table {name}")
    return t


def _visible_columns(t):
    return [col for col in t.columns if (t.name, col.name) not in HIDDEN_COLUMNS]


def _pk_filter(t, request):
    pk = {c.name: request.query_params.get(f"pk_{c.name}") for c in t.primary_key.columns}
    if any(v is None for v in pk.values()):
        raise HTTPException(400, "missing primary key")
    return [c == _convert(c, pk[c.name]) for c in t.primary_key.columns], pk


def _convert(col, raw):
    """Form string -> Python value for a column ('' -> NULL where allowed)."""
    if raw is None or (raw == "" and col.nullable):
        return None
    ty = col.type
    if isinstance(ty, (JSONB, ARRAY)):
        return json.loads(raw)
    if isinstance(ty, Boolean):
        return raw.strip().lower() in ("true", "1", "yes", "t", "on")
    if isinstance(ty, (Integer, BigInteger)):
        return int(raw)
    if isinstance(ty, Float):
        return float(raw)
    if isinstance(ty, DateTime):
        return datetime.fromisoformat(raw)
    if isinstance(ty, Date):
        return date.fromisoformat(raw)
    return raw


def _display(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return "" if v is None else str(v)


@app.get("/admin/monitors", response_class=HTMLResponse, dependencies=ADMIN)
def admin_monitors(request: Request, c=Depends(conn)):
    """Lab monitors (owner, 2026-10-17, takers only): one hidden lab-<candidate id> account per F1 taker candidate, each
    following its candidate on Polymarket from now to the end of the season. Their paper record, from story.track_record
    (the same as the MCP tool track_record(user="lab-<id>")), best P&L first."""
    import pandas as pd

    from racinglines.pipelines import story as S
    prefix = getattr(PF, "MONITOR_PREFIX", "lab-")
    users = rows(data.q(c, """SELECT id, username, display_name, created_at FROM users
                              WHERE starts_with(username, :p) ORDER BY username""", p=prefix))
    out = []
    for u in users:
        rec = sorted(S.track_record(c, u["id"], venue="polymarket"),          # by event_key there: by date here
                     key=lambda r: (r["date"] is None, r["date"] or pd.Timestamp.min, r["event_key"]))
        last = rec[-1] if rec else None
        out.append(dict(u, name=(u["display_name"] or u["username"]).removeprefix("Lab monitor: "), weekends=len(rec),
                        up=sum(1 for r in rec if r["pnl"] > 0), pnl=sum(r["pnl"] for r in rec),
                        taken=sum(int(r["taken"] or 0) for r in rec), open=sum(1 for r in rec if not r["settled"]),
                        last=last, record=rec))
    out.sort(key=lambda m: (m["weekends"] == 0, -m["pnl"], m["username"]))
    return render(request, "admin_monitors.html", monitors=out, prefix=prefix)


def _lock_rows(now=None):
    """The F1 weekends still to come (or under way) this season with their pick lock: the default times, the admin's
    override (fantasy/follow.lock_overrides) and whether picks are locked now. -> [dict]."""
    import pandas as pd

    from racinglines.fantasy import follow as FW
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    over, live = FW.lock_overrides(), FW.open_weekend(now)
    out = []
    for rnd, w in sorted(FW._schedule(now.year).items()):
        if FW.default_reopens(w) <= now:
            continue
        o = over.get(w["event_key"], {})
        out.append(dict(round=rnd, event_key=w["event_key"], name=w["name"], race_start=w["race_start"],
                        first=min((t for _, t in w.get("sessions") or ()), default=w["race_start"]),
                        closes=FW.cutoff(w), reopens=FW.reopens(w), default_reopens=FW.default_reopens(w),
                        override=bool(o), off=bool(o.get("off")),
                        locked=live is not None and live["event_key"] == w["event_key"]))
    return out


@app.get("/admin/locks", response_class=HTMLResponse, dependencies=ADMIN)
def admin_locks(request: Request, msg: str = ""):
    """Strategy-pick locks (owner, 2026-10-21: "UI for the admin to override/ edit upcoming locks"): each coming F1
    weekend's lock, by default from lights out to a day after the race (fantasy/follow.open_weekend), with a form to
    move either time or turn the lock off for that weekend. Saved in the data folder (follow.locks_file)."""
    try:
        weekends = _lock_rows()
    except Exception as ex:                             # noqa: BLE001  no schedule: say so instead of a 500
        weekends, msg = [], msg or f"Error: no F1 schedule ({ex})"
    return render(request, "admin_locks.html", weekends=weekends, msg=msg)


@app.post("/admin/locks", dependencies=[Depends(check_csrf), allow("admin")])
def admin_locks_save(request: Request, event_key: str = Form(...), closes: str = Form(""), reopens: str = Form(""),
                     off: str = Form(""), reset: str = Form("")):
    """Save one weekend's override (times are UTC, from datetime-local inputs), or clear it with `reset`."""
    import pandas as pd

    from racinglines.fantasy import follow as FW
    if event_key not in {w["event_key"] for w in _lock_rows()}:
        raise HTTPException(400, "not a coming weekend")
    if reset:
        closes = reopens = off = ""
    try:
        if closes and reopens and pd.Timestamp(closes) >= pd.Timestamp(reopens):
            return RedirectResponse("/admin/locks?msg=Error: the lock must close before it reopens.", status_code=303)
        FW.save_lock_override(event_key, closes or None, reopens or None, off == "1")
    except ValueError:
        return RedirectResponse("/admin/locks?msg=Error: not a date and time.", status_code=303)
    audit(request, "pick_lock", event_key=event_key, closes=closes or None, reopens=reopens or None, off=off == "1",
          reset=bool(reset))
    return RedirectResponse("/admin/locks?msg=" + quote(f"Saved {event_key}."), status_code=303)


@app.get("/admin/storage", dependencies=ADMIN)
def admin_storage():
    return RedirectResponse("/admin/db#storage", status_code=303)       # the storage status lives on /admin/db (owner)


@app.post("/admin/storage", dependencies=[Depends(check_csrf), allow("admin")])
def admin_storage_refresh(request: Request):
    if not present("storage_status"):
        raise HTTPException(404)
    from racinglines import storage_status as ST
    ST.refresh_in_background(force=True)
    audit(request, "storage_refresh")
    return RedirectResponse("/admin/db#storage", status_code=303)


@app.get("/admin/db", response_class=HTMLResponse, dependencies=ADMIN)
def admin_db(request: Request, c=Depends(conn)):
    tables = []
    for name in sorted(m.Base.metadata.tables):
        n = c.execute(text(f'SELECT count(*) FROM "{name}"')).scalar()
        tables.append(dict(name=name, rows=n))
    # the storage status and the VM's history (owner, 2026-10-14: "keep adding to admin/db", VM and bucket health):
    # racinglines/storage_status.py's snapshot (a stale or missing one refreshes in the background) and vm_history.py
    if not (present("storage_status") and present("vm_history")):     # the public release: the tables only
        return render(request, "admin_db.html", tables=tables, hosted=False)
    from racinglines import storage_status as ST
    from racinglines import vm_history as VH
    started = ST.refresh_in_background()
    snap = ST.read()
    return render(request, "admin_db.html", tables=tables, hosted=True, snap=snap, age=ST.age_seconds(snap), health=ST.health(snap),
                  refreshing=ST.refreshing() or started, human=ST.human, stale=ST.STALE_SECONDS, vm_charts=VH.charts())


@app.get("/admin/db/{name}", response_class=HTMLResponse, dependencies=ADMIN)
def admin_table(request: Request, name: str, col: str = "", val: str = "", page: int = 0, msg: str = "",
                c=Depends(conn)):
    t = _table(name)
    cols = _visible_columns(t)
    q = select(*cols)
    count_q = select(func.count()).select_from(t)
    if col and col in t.c and (t.name, col) not in HIDDEN_COLUMNS:
        cond = cast(t.c[col], Text) == val if val != "NULL" else t.c[col].is_(None)
        q, count_q = q.where(cond), count_q.where(cond)
    pk_cols = list(t.primary_key.columns)
    q = q.order_by(*[p.desc() for p in pk_cols]).offset(page * PAGE).limit(PAGE)
    result = c.execute(q)
    records = [dict(r._mapping) for r in result]
    total = c.execute(count_q).scalar()
    return render(request, "admin_table.html", table_name=name, columns=[x.name for x in cols],
                  pk=[p.name for p in pk_cols], records=[{k: _display(v) for k, v in r.items()} for r in records],
                  raw=records, col=col, val=val, page=page, total=total, page_size=PAGE, msg=msg)


@app.get("/admin/db/{name}/row", response_class=HTMLResponse, dependencies=ADMIN)
def admin_row(request: Request, name: str, c=Depends(conn)):
    t = _table(name)
    where, pk = _pk_filter(t, request)
    row = c.execute(select(*_visible_columns(t)).where(*where)).first()
    if row is None:
        raise HTTPException(404)
    fields = [dict(name=col.name, value=_display(row._mapping[col.name]), type=str(col.type),
                   nullable=col.nullable, pk=col.primary_key) for col in _visible_columns(t)]
    return render(request, "admin_row.html", table_name=name, fields=fields, pk=pk,
                  pk_query="&".join(f"pk_{k}={v}" for k, v in pk.items()))


@app.post("/admin/db/{name}/row", dependencies=[Depends(check_csrf), allow("admin")])
async def admin_row_save(request: Request, name: str):
    t = _table(name)
    where, pk = _pk_filter(t, request)
    form = await request.form()
    action = form.get("action")
    pk_query = "&".join(f"pk_{k}={v}" for k, v in pk.items())
    try:
        with get_engine().begin() as c:
            before = c.execute(select(*_visible_columns(t)).where(*where)).first()
            if before is None:
                raise HTTPException(404)
            before = {k: _display(v) for k, v in before._mapping.items()}
            if action == "delete":
                if form.get("confirm") != "yes":
                    raise ValueError("tick confirm to delete")
                c.execute(delete(t).where(*where))
                audit(request, "db_delete", table=name, pk=pk, row=before)
                return RedirectResponse(f"/admin/db/{name}?msg=Deleted row {pk}.", status_code=303)
            values = {}
            for col in _visible_columns(t):
                if col.primary_key or col.name not in form:
                    continue
                if form[col.name] != before[col.name]:
                    values[col.name] = _convert(col, form[col.name])
            if values:
                c.execute(update(t).where(*where).values(**values))
            audit(request, "db_update", table=name, pk=pk,
                  changes={k: [before[k], _display(v)] for k, v in values.items()})
        return RedirectResponse(f"/admin/db/{name}/row?{pk_query}&msg=Saved", status_code=303)
    except (ValueError, json.JSONDecodeError) as e:
        return RedirectResponse(f"/admin/db/{name}/row?{pk_query}&msg=Error: {e}", status_code=303)
    except Exception as e:  # database errors (constraints, types) shown to the admin, not raised
        return RedirectResponse(f"/admin/db/{name}/row?{pk_query}&msg=Error: {type(e).__name__}: "
                                f"{str(e).splitlines()[0][:200]}", status_code=303)


# ---------------------------------------------------------------------------
# SQL console
# ---------------------------------------------------------------------------

@app.get("/admin/sql", response_class=HTMLResponse, dependencies=ADMIN)
def admin_sql_page(request: Request):
    return render(request, "admin_sql.html", sql="SELECT * FROM activity_log ORDER BY id DESC LIMIT 20",
                  columns=None, records=None, info=None, error=None, write=False)


@app.post("/admin/sql", response_class=HTMLResponse, dependencies=[Depends(check_csrf), allow("admin")])
def admin_sql_run(request: Request, sql: str = Form(...), write: str = Form(""), confirm: str = Form("")):
    """Read mode runs inside a READ ONLY transaction (Postgres rejects any write,
    including data-modifying WITH queries). Write mode needs the toggle AND the
    confirm box, and the statement is logged in full either way."""
    write_mode = bool(write)
    columns = records = info = error = None
    if write_mode and confirm != "yes":
        error = "Write mode needs the confirm box ticked."
    else:
        try:
            with get_engine().begin() as c:
                c.execute(text("SET LOCAL statement_timeout = '15s'"))
                if not write_mode:
                    c.execute(text("SET TRANSACTION READ ONLY"))
                res = c.execute(text(sql))
                if res.returns_rows:
                    columns = list(res.keys())
                    fetched = res.fetchmany(SQL_ROW_LIMIT + 1)
                    records = [[_display(v) for v in r] for r in fetched[:SQL_ROW_LIMIT]]
                    info = f"{len(records)} row(s)" + (f" (first {SQL_ROW_LIMIT} shown)" if len(fetched) > SQL_ROW_LIMIT else "")
                else:
                    info = f"{res.rowcount} row(s) affected"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e).splitlines()[0][:500]}"
    audit(request, "sql_write" if write_mode else "sql_read", sql=sql[:4000], error=error, info=info)
    return render(request, "admin_sql.html", sql=sql, columns=columns, records=records, info=info, error=error,
                  write=write_mode)


# ---------------------------------------------------------------------------
# MCP access: who holds an rl_ token or has signed in from an MCP client; disconnect or revoke
# ---------------------------------------------------------------------------

@app.get("/admin/mcp", response_class=HTMLResponse, dependencies=ADMIN)
def admin_mcp(request: Request, msg: str = ""):
    if not present("mcp"):
        raise HTTPException(404)
    from racinglines.mcp import auth as mcp_auth
    from racinglines.mcp import oauth
    resp = render(request, "admin_mcp.html", conns=oauth.connections(get_engine()), roles=", ".join(mcp_auth.ROLES),
                  mcp_url=oauth.MCP_URL, signin=oauth.enabled(), msg=msg)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/admin/mcp", dependencies=[Depends(need_mcp), Depends(check_csrf), allow("admin")])
def admin_mcp_action(request: Request, action: str = Form(...), username: str = Form("")):
    from racinglines.mcp import auth as mcp_auth
    from racinglines.mcp import oauth
    eng = get_engine()
    names = [c["username"] for c in oauth.connections(eng)] if action == "disconnect_all" else [username]
    with eng.connect() as c:
        ids = dict(c.execute(text("SELECT username, id FROM users WHERE username = ANY(:u)"), dict(u=names)).fetchall())
    if action in ("disconnect", "disconnect_all"):
        for name in names:
            if name in ids:
                oauth.disconnect(eng, ids[name])
        msg = f"Disconnected {', '.join(names) or 'nobody'}: their MCP clients must sign in again."
    elif action == "revoke_token":
        msg = f"Revoked {username}'s token." if mcp_auth.revoke(eng, username) else f"{username} had no token."
    else:
        raise HTTPException(400, "unknown action")
    audit(request, "admin_mcp", what=action, users=names)
    return RedirectResponse("/admin/mcp?msg=" + quote(msg), status_code=303)
