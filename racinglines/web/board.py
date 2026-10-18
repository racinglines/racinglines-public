"""
The maker's home page: every sport, what's next, what's live, what just happened.

Per sport: the next races (fair favourites, venue status, my exposure), the season
markets, and recent results with how our pre-race price did. Plus a few headline
numbers across everything.
"""

import os
from datetime import datetime, timezone

import pandas as pd

from racinglines.db import reads as data
from racinglines.markets import private_book as house
from racinglines.markets.venues import (VENUES, SPORT_NAME, SPORT_ORDER, event_matrix, exchange_breakdown, race_title,
                                        season_matrix, venue_summary)


def _countdown(d):
    if d is None or pd.isna(d):
        return ""
    days = (pd.Timestamp(d).date() - datetime.now(timezone.utc).date()).days
    return "today" if days == 0 else ("tomorrow" if days == 1 else (f"in {days} days" if days > 0 else f"{-days} days ago"))


def _top(df, kind, n=3):
    g = df[df["kind"] == kind] if len(df) else df
    # a mixed column (an unpriced outcome among priced ones) holds NaN, which the templates would print as "nan%"
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in g.head(n).to_dict("records")]


def _mine(df):
    p = [x for x in (df["private"].dropna() if len(df) else []) if x["status"] == "open"]
    return dict(open=len(p), worst=sum(x["worst"] for x in p), staked=sum(x["staked"] for x in p))


def _card(conn, race_id, maker_id, show_kind="race_win"):
    info, pricing, df = event_matrix(conn, race_id, maker_id)
    return dict(info=info, pricing=pricing, top=_top(df, show_kind), venues=venue_summary(df), mine=_mine(df),
                countdown=_countdown(info["start_date"]), outcomes=len(df))


def recent_results(conn, competition_id, n=3):
    races = data.q(conn, """
        SELECT ra.id AS race_id FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
        JOIN categories c ON c.id = ra.category_id
        WHERE s.competition_id = :c AND e.status = 'completed' AND c.code IN ('DRV', 'ME', 'RDR')
          AND EXISTS (SELECT 1 FROM rounds ro JOIN results r ON r.round_id = ro.id
                      WHERE ro.race_id = ra.id AND ro.kind IN ('race', 'final'))
        ORDER BY e.start_date DESC LIMIT :n""", c=competition_id, n=n)
    bt = data.q(conn, """SELECT metrics->'events' AS ev FROM model_runs WHERE kind = 'backtest' AND competition_id = :c
                           AND coalesce(params->>'variant', 'baseline') = 'baseline'
                         ORDER BY id DESC LIMIT 1""", c=competition_id)
    bt_ev = {}
    for e in (bt["ev"].iloc[0] if len(bt) and bt["ev"].iloc[0] else []):
        if e.get("mode") == "pre_race" and e.get("track_features") in (True, None):
            bt_ev[e["event_id"]] = e
    out = []
    for rid in races["race_id"]:
        info, pricing, df = event_matrix(conn, int(rid))
        res = house.race_outcomes(conn, int(rid)).sort_values("position")
        win = res[(res["position"] == 1) & (res["status"] == "OK")]
        winner_id = int(win["athlete_id"].iloc[0]) if len(win) else None
        w = df[(df["kind"] == "race_win") & (df["athlete_id"] == winner_id)] if len(df) and winner_id else pd.DataFrame()
        our = float(w["fair"].iloc[0]) if len(w) and w["fair"].iloc[0] is not None else None
        pm = float(w["pm_mid"].iloc[0]) if len(w) and w["pm_mid"].iloc[0] is not None and pd.notna(w["pm_mid"].iloc[0]) else None
        kq = (w["venues"].iloc[0] or {}).get("kalshi") if len(w) else None
        kalshi = kq["mid"] if kq and kq.get("mid") is not None else None
        source = pricing["source"]
        if our is None and info["event_id"] in bt_ev:
            our, source = bt_ev[info["event_id"]].get("winner_prob"), "backtest (as of before the race)"
        name = data.q(conn, "SELECT display_name FROM athletes WHERE id = :a", a=winner_id)["display_name"].iloc[0] \
            if winner_id else None
        diag = data.q(conn, """SELECT max(id) AS id FROM model_runs WHERE kind = 'diagnostic'
                               AND params->>'event_key' = :k""", k=info["source_key"])["id"].iloc[0]
        out.append(dict(info=info, winner=name, our=our, pm=pm, kalshi=kalshi, source=source,
                        diag=int(diag) if diag is not None and pd.notna(diag) else None,
                        countdown=_countdown(info["start_date"])))
    return out


RESOLVED_DAYS = 5      # a resolved race stays among the board's race cards this many days after it ran


def recently_resolved(conn, competition_id, days=RESOLVED_DAYS):
    """The competition's races (elite category) completed with a race classification in the last `days` days,
    newest first: the board keeps them up as resolved before the next races."""
    return data.q(conn, """
        SELECT ra.id AS race_id FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
        JOIN categories c ON c.id = ra.category_id
        WHERE s.competition_id = :c AND e.status = 'completed' AND c.code IN ('DRV', 'ME', 'RDR')
          AND e.start_date >= current_date - CAST(:d AS int)
          AND EXISTS (SELECT 1 FROM rounds ro JOIN results r ON r.round_id = ro.id
                      WHERE ro.race_id = ra.id AND ro.kind IN ('race', 'final'))
        ORDER BY e.start_date DESC""", c=competition_id, d=days)


def _resolved_card(conn, race_id, maker_id):
    card = _card(conn, race_id, maker_id)
    res = house.race_outcomes(conn, race_id)
    win = res[(res["position"] == 1) & (res["status"] == "OK")]
    card["winner"] = (data.q(conn, "SELECT display_name FROM athletes WHERE id = :a",
                             a=int(win["athlete_id"].iloc[0]))["display_name"].iloc[0] if len(win) else None)
    return card


def _with_resolved(conn, comp_id, maker_id, upcoming):
    """Recently resolved race cards first, then the upcoming ones (a race never twice); `next` marks the first
    race still to run."""
    done = [dict(_resolved_card(conn, int(r), maker_id), new=0) for r in recently_resolved(conn, comp_id)["race_id"]]
    ids = {u["info"]["race_id"] for u in done}
    cards = done + [u for u in upcoming if u["info"]["race_id"] not in ids]
    nxt = next((u for u in cards if u["info"]["status"] not in ("completed", "in_progress")), None)
    for u in cards:
        u["next"] = u is nxt
    return cards


def next_races(conn, competition_id, n=8):
    """The competition's next scheduled races (elite category), soonest first."""
    return data.q(conn, """
        SELECT ra.id AS race_id, e.id AS event_id, e.name, e.start_date FROM races ra JOIN events e ON e.id = ra.event_id
        JOIN seasons s ON s.id = e.season_id JOIN categories c ON c.id = ra.category_id
        WHERE s.competition_id = :c AND e.status NOT IN ('completed', 'cancelled') AND e.start_date >= current_date
          AND c.code IN ('DRV', 'ME', 'RDR')
        ORDER BY e.start_date, ra.id LIMIT :n""", c=competition_id, n=n)


def board(conn, maker_id, only=None, detail=True, hide=frozenset()):
    """One dict per sport shown on the Markets board. detail=False: only what a sport's collapsed header shows (name,
    model run, status chips); the page loads each sport's body when it is opened (`only` = that competition code)."""
    from racinglines import exchanges as EX
    from racinglines import sports as SP
    from racinglines.markets import alerts
    fresh = list(alerts.new_links(conn).values()) if conn is not None and detail else []  # (race_id, competition_id) per new token

    def _new_for(race_id=None, comp_id=None):
        """How many new tokens are for this exact race, or (race_id=None) this exact sport's season markets.
        Both race_id and competition_id are market_links columns (no fuzzy text matching)."""
        return sum(1 for rid, cid in fresh if rid == race_id and (race_id is not None or cid == comp_id))
    forecasts = {}
    if conn is not None:
        forecasts = {r["competition"]: r for r in data.latest_forecasts(conn).to_dict("records")}
    from racinglines.web import sport_status as SS
    status_on = SS.enabled()                       # RACINGLINES_SPORT_STATUS=1: model sections for sports with as-of runs
    ss_by_comp = {r["competition"]: r for r in SS.status(conn, hide)} if conn is not None and not detail else {}  # header chips
    asof_by_comp = {}
    if conn is not None and status_on:
        for r in data.q(conn, """SELECT co.code AS competition, count(DISTINCT mr.params->>'event_key') AS races,
                                        max(mr.created_at) AS at
                                 FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
                                 WHERE mr.kind = 'diagnostic' AND mr.params ? 'replay_batch' GROUP BY 1""").to_dict("records"):
            asof_by_comp[r["competition"]] = r
    exch_by_comp = {}
    linked_comps = set()
    if detail:
        comps = {s["competition"]["code"]: s for s in map(SP.load, SP.SPORT_CODES)
                 if s["competition"]["code"] == only} if only is not None else None
        for b in exchange_breakdown(conn, comps):
            exch_by_comp.setdefault(b["competition"], []).append(b)
    elif conn is not None:
        links = data.q(conn, """SELECT DISTINCT co.code AS competition
                               FROM market_links ml JOIN competitions co ON co.id = ml.competition_id""")
        linked_comps = set(links["competition"])
    sports = []
    for schema in sorted(map(SP.load, SP.SPORT_CODES), key=lambda s: s["sport"]["display_order"]):
        code, sport_code = schema["competition"]["code"], schema["sport"]["code"]
        if (only is not None and code != only) or code in hide:
            continue                                     # hide: competitions this user doesn't see (web/audience.py)
        tape = schema["sport"].get("model_family", "none") == "none"
        run, exch = forecasts.get(code), exch_by_comp.get(code, [])
        if run is None and not exch and code not in linked_comps:
            continue                                     # nothing to show yet: no forecast, no linked market
        asof = asof_by_comp.get(code) if run is None and status_on else None
        for b in exch:                                    # where "N markets on Kalshi" etc. links to
            b["url"] = (f"/markets/tapes#tapes-{sport_code}-{b['exchange']}" if tape
                        else f"/markets/{b['exchange']}" if b["exchange"] in ("polymarket", "kalshi") or b["exchange"] in EX.CODES
                        else None)
        if not detail:
            sports.append(dict(code=code, name=SPORT_NAME.get(code, schema["sport"]["name"]), run=run, tape=tape, asof=asof,
                               calibration=schema["sport"].get("calibration"), status=ss_by_comp.get(code)))
            continue
        upcoming, later, season, recent = [], [], None, []
        if run:
            comp_id = int(data.q(conn, "SELECT id FROM competitions WHERE code = :c", c=code)["id"].iloc[0])
            targets = data.run_race_targets(conn, run["id"])
            targets = targets[targets["race_id"].notna() & ~targets["target"].astype(str).str.startswith("backtest:")]
            upcoming = _with_resolved(conn, comp_id, maker_id,
                                      [dict(_card(conn, int(r), maker_id), new=_new_for(int(r))) for r in targets["race_id"].head(3)])
            later = [dict(title=race_title(code, t["source_key"], t["venue"], t["event_name"]), event_id=t["event_id"],
                          race_id=int(t["race_id"]), date=t["start_date"], new=_new_for(int(t["race_id"]))) for t in targets.iloc[3:].to_dict("records")]
            s_info, s_pricing, s_df = season_matrix(conn, code, maker_id)
            from racinglines.web.views import latest_season_strategy
            season = dict(new=_new_for(None, comp_id), info=s_info, top=_top(s_df, "champion"), venues=venue_summary(s_df), mine=_mine(s_df),
                          outcomes=len(s_df), constructors=_top(s_df, "constructors_champion", 2),
                          strategy=latest_season_strategy(conn, code))
            recent = True                                # loaded when its section is opened (views.board_recent)
        elif not tape and asof is not None:
            # a modeled sport with no live forecast run (NASCAR, MotoGP): the next races with the exchanges' prices
            # (no fair price until a forecast is stored), and the recent races with the model's as-of price
            # (`<sport> replay --save`: made before each race) against the winner
            comp_id = int(data.q(conn, "SELECT id FROM competitions WHERE code = :c", c=code)["id"].iloc[0])
            nxt = next_races(conn, comp_id)
            upcoming = _with_resolved(conn, comp_id, maker_id,
                                      [dict(_card(conn, int(r), maker_id), new=_new_for(int(r))) for r in nxt["race_id"].head(3)])
            later = [dict(title=t["name"], event_id=t["event_id"], race_id=int(t["race_id"]), date=t["start_date"],
                          new=_new_for(int(t["race_id"]))) for t in nxt.iloc[3:].to_dict("records")]
            recent = True                                # loaded when its section is opened (views.board_recent)
        elif exch:
            tape_events = []
            for b in exch:
                for e in b.get("events", []):
                    tape_events.append(dict(title=e["title"], date=e.get("end_date"), status="open" if e.get("open") else "settled",
                                           exchanges=[b["exchange_name"]], url=b.get("url"), new=0))
            upcoming = tape_events[:3]
            later = tape_events[3:]
            season = None
            recent = []
        sched = [u for u in upcoming if u.get("info") and u["info"].get("status") not in ("completed", "cancelled")]
        if sched and conn is not None:                    # how fresh each venue's prices are, per upcoming event
            fresh_by = event_sync(conn, [int(u["info"]["event_id"]) for u in sched],
                                  [v for v in schema.get("markets", {}).get("venues", []) if v in EXCHANGE_CODES])
            for u in sched:
                u["sync"] = fresh_by.get(int(u["info"]["event_id"]), [])
        sports.append(dict(code=code, name=SPORT_NAME.get(code, schema["sport"]["name"]), run=run, tape=tape, asof=asof,
                           calibration=schema["sport"].get("calibration"),
                           upcoming=upcoming, later=later, season=season, recent=recent, exchanges=exch,
                           status=ss_by_comp.get(code)))
    sports.sort(key=lambda s: SPORT_ORDER.get(s["code"], 9))
    return sports


# An upcoming event's prices on a venue are flagged when the venue has no market linked to the event, or when the newest
# sync of its links (market_links.synced_at) is older than this many hours (owner's missing-price P0, 2026-10-07).
# RACINGLINES_SYNC_STALE_HOURS sets it; the syncs are moving to every 5 minutes, so it can come down then.
SYNC_STALE_HOURS = float(os.environ.get("RACINGLINES_SYNC_STALE_HOURS", "3"))
EXCHANGE_CODES = {v.code for v in VENUES if v.kind == "exchange" and v.status == "live"}


def event_sync(conn, event_ids, venues, now=None):
    """{event_id: [dict(code, name, links, last, hours, stale)]} for each of `venues` (exchange codes, in that order):
    the market links whose race belongs to the event (exact race_id keys), the newest synced_at among them, its age
    in hours, and stale = no links, or no sync time, or older than SYNC_STALE_HOURS."""
    if not event_ids or not venues:
        return {}
    df = data.q(conn, """SELECT ra.event_id, ml.exchange, count(*) AS links, max(ml.synced_at) AS last
                         FROM market_links ml JOIN races ra ON ra.id = ml.race_id
                         WHERE ra.event_id = ANY(:e) AND ml.exchange = ANY(:x) GROUP BY 1, 2""",
                e=[int(x) for x in event_ids], x=list(venues))
    got = {(int(r["event_id"]), r["exchange"]): r for r in df.to_dict("records")}
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    names = {v.code: v.name for v in VENUES}
    out = {}
    for e in event_ids:
        rows = []
        for code in venues:
            r = got.get((int(e), code)) or {}
            last = r.get("last")
            last = None if last is None or pd.isna(last) else pd.Timestamp(last)
            if last is not None and last.tzinfo is None:
                last = last.tz_localize("UTC")
            hours = None if last is None else max(0.0, (now - last).total_seconds() / 3600)
            rows.append(dict(code=code, name=names.get(code, code), links=int(r.get("links") or 0), last=last, hours=hours,
                             stale=last is None or hours > SYNC_STALE_HOURS, limit=SYNC_STALE_HOURS))
        out[int(e)] = rows
    return out


STALE_MIN = 15          # a venue with no book snapshot for this long is "stale" (its recorder passes every 5 minutes)


def recorder_status(conn, codes, now=None):
    """Per venue in `codes` (in that order): open markets linked, how many have a book snapshot in the last 10 minutes,
    the latest snapshot's time and minutes elapsed, and a state: "live", "stale" (nothing for STALE_MIN minutes) or
    "none" (no snapshot yet, or nothing linked). Fails soft: a query error gives every venue state "unknown" (and
    rolls the read back), so a page showing it still loads with whatever else it has."""
    now = now or datetime.now(timezone.utc)
    try:
        df = data.q(conn, """
            SELECT l.exchange, count(*) AS linked, max(t.ts) AS last,
                   count(t.ts) FILTER (WHERE t.ts > now() - interval '10 minutes') AS recent
            FROM (SELECT DISTINCT token_id, exchange FROM market_links WHERE NOT closed AND exchange = ANY(:x)) l
            LEFT JOIN LATERAL (SELECT ts FROM market_book_snapshots b WHERE b.token_id = l.token_id
                               ORDER BY ts DESC LIMIT 1) t ON true
            GROUP BY 1""", x=list(codes))
        rows = {r["exchange"]: r for r in df.to_dict("records")}
    except Exception:  # noqa: BLE001  (never take the page down for a status line)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return [dict(code=c, name=next((v.name for v in VENUES if v.code == c), c), linked=0, recent=0, last=None,
                     state="unknown") for c in codes]
    out = []
    for c in codes:
        r = rows.get(c) or {}
        last = r.get("last")
        last = None if last is None or pd.isna(last) else pd.Timestamp(last).to_pydatetime()
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        minutes = None if last is None else (now - last).total_seconds() / 60
        state = "none" if last is None else "live" if minutes <= STALE_MIN else "stale"
        out.append(dict(code=c, name=next((v.name for v in VENUES if v.code == c), c), linked=int(r.get("linked") or 0),
                        recent=int(r.get("recent") or 0), last=last, minutes=minutes, state=state))
    return out


def headline(conn, maker_id):
    ex = data.q(conn, """
        SELECT count(*) AS outcomes, count(DISTINCT market_slug) AS markets,
               coalesce(sum(DISTINCT volume), 0) AS volume, max(synced_at) AS synced
        FROM market_links WHERE NOT closed AND prediction <> 'unmodeled'""").iloc[0]
    # Bucketing to 10-minute slots keeps the page stable between repeated renders while still showing the most
    # recent recorder tick. The live order-book feed is updated every minute, so exact second values jitter the
    # rendered HTML even when the underlying state is otherwise unchanged.
    rec = data.q(conn, """
        SELECT date_trunc('hour', ts) + floor(extract(minute FROM ts) / 10.0) * interval '10 minutes' AS ts,
               count(DISTINCT token_id) AS n
        FROM market_book_snapshots WHERE ts > now() - interval '10 minutes'
        GROUP BY 1 ORDER BY 1 DESC LIMIT 1""")
    rec_ts = rec["ts"].iloc[0] if len(rec) and not pd.isna(rec["ts"].iloc[0]) else None
    rec_n = int(rec["n"].iloc[0] or 0) if len(rec) and rec["n"].iloc[0] is not None else 0
    bk = house.book(conn, maker_id=maker_id, status="open")
    jobs = data.q(conn, "SELECT count(*) FILTER (WHERE status IN ('queued', 'running')) AS active FROM jobs").iloc[0]
    brier_win, brier_grid = model_brier(conn)
    return dict(outcomes=int(ex["outcomes"]), markets=int(ex["markets"]), volume=float(ex["volume"]), synced=ex["synced"],
                recording=rec_n, recorded_at=rec_ts,
                my_open=len(bk), my_worst=float(bk["worst"].sum()) if len(bk) else 0.0,
                my_staked=float(bk["staked"].sum()) if len(bk) else 0.0,
                jobs_active=int(jobs["active"]), brier_win=brier_win, brier_grid=brier_grid)


def model_brier(conn):
    """The Lab's headline: (win Brier, grid-only win Brier) of the latest baseline backtest, each None without one."""
    try:
        bt = data.q(conn, """SELECT metrics->'summary'->'pre_race|track=True' AS s FROM model_runs
                             WHERE kind = 'backtest' AND coalesce(params->>'variant', 'baseline') = 'baseline' ORDER BY id DESC LIMIT 1""")
        s = bt["s"].iloc[0] if len(bt) else None
    except Exception:                                   # noqa: BLE001  the Lab opens without it
        s = None
    return (s.get("brier_win") if s else None, s.get("brier_win_grid") if s else None)
