"""
Live paper signals for any sport on an exchange, run by the sport's own stage mode (pipelines/season_sweep.py):

    racinglines strategy signals nascar                      # every user with a NASCAR profile, on every venue it names
    racinglines strategy signals motogp --venue og --profile 123 --event 2026-5630 --asof 2026-10-03T20:00   # replay, prints only

A profile names its sport (profile["sport"]; F1 when unset) and its venue (the `venue` setting; unset = the sport's
first exchange, season_sweep.settings_class). A user's profile for a sport and venue lives in users.prefs under
profiles.pref(venue, sport). Nothing here names a sport: what differs comes from sports/<code>.toml.

  "sessions" mode (F1): pipelines/signals.py, unchanged (its compute and run_all; this module hands F1 to them).
  "weekend" mode (NASCAR, MotoGP): per race, the schema's fixed [replay] stages (hours from 00:00 UTC on race day).
    1. Data (live only): the venue's tape for the race's markets over the last 26 h (position_replay.pull: the
       24 h volume floor reaches back a day). The recorder's timers keep Kalshi's and OG.com's books as before.
    2. Pricing: the sport's pricing model ([sport] pricing_model) from results strictly before the event, seeded per
       event exactly as the sweep (season_sweep.replay_sweep). Live, each stage whose time has passed gets one stored
       run (model_runs kind 'diagnostic', params mode 'live signals', sweep_stage = the stage label, live = true, the
       field it priced), priced once when the stage first comes due with the race's start list (its own entry list
       once a session is ingested, else the last race's field) and reused by every later pass; a new run is logged
       in data_changes as "live-price". A replay (live=False) prices the race in memory on its classified field, as
       the sweep does, and stores nothing.
    3. Markets: season_sweep.race_markets on the venue (markets/venue_replay.py), each stage read live when its run
       was priced (its trades couldn't be made earlier), in a replay at the stage's time (as the sweep).
    4. Takers: taker_weekend.run_market per market (signals.taker_signals) with the sweep's taker parameters
       (weekend_sweep.taker_params: the venue's fee). Makers: maker_replay on season_sweep.maker_event, up to now.
    5. Store and alert with signals.store / signals._run_one: rows carry detail.sport and detail.venue, positions the
       venue and the sport's own event key.

Paper only: nothing here places an order or reads a trading flag.
"""

import zlib
from dataclasses import replace

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.pipelines import season_sweep as SW
from racinglines.pipelines import signals as SG
from racinglines.pipelines import sweep_settings as SS

LIVE_MODE = "live signals"                 # model_runs.params.mode of a stage priced by this module
TAPE_LOOKBACK = pd.Timedelta(hours=26)     # live refresh window: the 24 h volume floor plus slack


def sport_of(profile):
    """The sport a profile trades: its `sport`, else F1 (every profile before this module)."""
    return (profile or {}).get("sport") or "f1"


def settings_of(sport, profile):
    return SW.settings_class(sport).from_dict(profile["settings"], strict=False)


# --- the event --------------------------------------------------------------------------------------------

def races(conn, sp):
    """The sport's races, scheduled or run (not cancelled), in date order: race_id, event_key, name, season, start."""
    return pd.read_sql(text("""
        SELECT ra.id AS race_id, e.source_key AS event_key, e.name, s.year AS season, e.start_date AS start
        FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
        JOIN competitions co ON co.id = s.competition_id
        WHERE co.code = :c AND e.source = :src AND e.status <> 'cancelled'
        ORDER BY e.start_date, ra.id"""), conn, params=dict(c=sp["competition"], src=sp["source"]))


def window(race, sp):
    """(first signal time, last) of a race: from WEEKEND_LEAD before race day to WEEKEND_TAIL after it ends."""
    day = pd.Timestamp(race.start) + pd.Timedelta(days=int(sp.get("race_day_offset", 0)))
    return day - SG.WEEKEND_LEAD, day + pd.Timedelta(days=1) + SG.WEEKEND_TAIL


def pick_race(rs, sp, now, event="next"):
    """The race row of `event` (its event key), or the first race whose window hasn't ended by `now`; None."""
    if event not in (None, "", "next"):
        hit = rs[rs["event_key"] == str(event)]
        return next(hit.itertuples(), None)
    return next((r for r in rs.itertuples() if window(r, sp)[1] > now), None)


def field_now(conn, sp, race_id):
    """The start list known now: the race's own entry list (sport_forecast.entry_list), else the last race's field."""
    from racinglines.pipelines import sport_forecast as SF
    return SF.entry_list(conn, race_id) or SF.last_field(conn, sp)[1]


# --- pricing ----------------------------------------------------------------------------------------------

def _price(model, hist, ms, race, field):
    from racinglines.models.race_model import Event
    ev = Event(id=race.event_key, season=int(race.season), cutoff=pd.Timestamp(race.start), name=str(race.name),
               info={"field": [int(a) for a in field]})
    return model.price(hist, ev, ms, np.random.default_rng([ms.rng_seed, zlib.crc32(str(race.event_key).encode())]))


def stored_runs(conn, sport, event_key, model_key):
    """{stage label: (run id, priced_at naive UTC, field)} of the live runs stored for this event and model."""
    rows = conn.execute(text("""SELECT id, coalesce(CAST(params->>'priced_at' AS timestamptz), created_at),
                                       params->>'sweep_stage', params->'field_ids' FROM model_runs
                                WHERE kind = 'diagnostic' AND params->>'mode' = :mode AND params->>'sport' = :s
                                  AND params->>'event_key' = :k AND params->>'model_key' = :m ORDER BY id"""),
                        dict(mode=LIVE_MODE, s=sport, k=event_key, m=model_key)).all()
    out = {}
    for i, t, lab, field in rows:
        out.setdefault(lab, (int(i), pd.Timestamp(t).tz_convert("UTC").tz_localize(None), list(field or [])))
    return out


def save_live_run(engine_url, sp, model, st, ms, race, sims, label, field, data_key, priced_at):
    """Store one stage's live pricing (position_replay.save_run with the live params) and log it in data_changes
    ("live-price": event key, run id, variant, sims, code_version, data_key). Returns the run id."""
    from racinglines.db import changes
    from racinglines.db.config import get_session
    from racinglines.db.queries import code_version
    from racinglines.pipelines import position_replay as P
    cv = code_version()
    variant = st.get("variant") if "variant" in st.BY else None
    rid = P.save_run(dict(engine_url=engine_url), sp, model, ms, race, sims,
                     params=dict(mode=LIVE_MODE, live=True, sweep_stage=label, model_key=st.model_key, data_key=data_key,
                                 field_ids=[int(a) for a in field], code_version=cv,
                                 priced_at=str(pd.Timestamp(priced_at).tz_localize("UTC")),
                                 **({"variant": variant} if variant else {})))
    with get_session(engine_url) as s:
        changes.record_run(s, "live-price", sp["sport"], [race.event_key], rid, variant=variant, sims=int(sims.n_sims),
                           code_version=cv, data_key=data_key, stage=label)
        s.commit()
    return rid


# --- one profile ------------------------------------------------------------------------------------------

def compute(engine, engine_url, profile, now=None, event="next", live=True, fetch=True, echo=print, cache=None,
            venue=None, scale=1.0):
    """signals.compute for a profile of any sport: F1 ("sessions") goes to signals.compute unchanged; a "weekend"
    sport is computed here. -> the same dict (event, stages, signals, positions, maker_state, venue, note), plus
    `sport`; the event dict carries event_key and name."""
    sport = sport_of(profile)
    st = settings_of(sport, profile)
    if SW.mode_of(sport, st) == "sessions":
        return SG.compute(engine, engine_url, profile, now=now, event=event, live=live, fetch=fetch, echo=echo,
                          cache=cache, venue=venue, scale=scale)
    from racinglines.pipelines import position_replay as P
    venue = SG.resolve_venue(st, venue, live)
    if venue not in SW.venues(sport):
        raise ValueError(f"{sport}: venue {venue!r} is not one of {SW.venues(sport)}")
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    sp = dict(P.spec(sport), kinds=[k for k in SW.kinds(sport) if k in st["market_kinds"]])
    base = dict(profile=profile, sport=sport, event=None, stages=[], signals=[], positions=[], maker_state={}, venue=venue)
    with engine.connect() as c:
        race = pick_race(races(c, sp), sp, now, event)
    if race is None:
        return dict(base, note=f"no {sport} race left")
    w = dict(event_key=race.event_key, name=str(race.name), race_id=int(race.race_id))
    base.update(event=w, race_id=int(race.race_id))
    first, last = window(race, sp)
    if live and not first <= now <= last:
        return dict(base, note=f"{race.name}: signals start {first:%a %d %b}")
    mode = SW.mode_of(sport, st)
    stages = [(lab, t) for lab, t in P.stage_times(race.start, dict(sp, stages=P.stage_list(sp, mode))) if t <= now]
    if not stages:
        return dict(base, note=f"{race.name}: no stage due yet")
    with engine.connect() as c:
        links = P.links(c, sp, venue)
        links = links[links["race_id"] == race.race_id] if len(links) else links
        res = P.race_results(c, race.race_id)
    if not len(links):
        return dict(base, note=f"{race.name}: no {venue} markets linked")
    if live and fetch:
        refresh(engine, engine_url, race, links, venue, now, echo)
    cache = {} if cache is None else cache
    model = P.model_for(sp)
    ms = model.Settings.from_dict({k: st[k] for k in model.Settings.BY})
    if ("data", sport) not in cache:
        cache[("data", sport)] = model.load(engine.url.render_as_string(hide_password=False))
    data = cache[("data", sport)]
    hk = ("hist", sport, st.model_key)
    if hk not in cache:
        cache[hk] = model.history(data, ms)
    hist = cache[hk]
    priced, sims_of = price_stages(engine, engine_url, sp, model, st, ms, hist, data, race, stages, res, live, now,
                                  echo)
    base["stages"] = priced
    if not priced:
        return dict(base, note=f"{race.name}: the model has no price for this race")
    read_at = [(lab, max(t, at) if live else t) for lab, t, _, at in priced]
    strategy = profile["strategy"]
    with engine.connect() as c:
        # a replay reads the tape as the sweep does (from an hour before the first stage); live, a stage is read when
        # its run was priced, so the tape is loaded from P.STALE before that, the oldest price the venue still uses
        since = min(t for _, t in read_at) - P.STALE + pd.Timedelta(hours=1)       # _venue loads from an hour before
        v = P._venue(c, venue, links, read_at if not live else [("from", since)] + read_at, sp)
        v.tol_by_kind = dict(SS.parse_map(st["coherence_tol_by_kind"]))
        thin = st["thin_edge_mult"] is not None
        if thin:
            v.load_books(c, min(t for _, t in read_at) - pd.Timedelta(hours=1), max(t for _, t in read_at))
        from racinglines.pipelines import weekend_sweep as WS
        if strategy in WS.TAKER_MODES:
            markets = stage_markets(v, links, sims_of, res, read_at, st["min_volume_24h"], thin)
            p, _ = WS.taker_params(st)
            p = replace(p, mode=strategy)
            if scale != 1.0:
                p = replace(p, scale=p.scale * scale)
            base["signals"], base["positions"] = SG.taker_signals(markets, p)
            base["_taker"] = SG.taker_inputs(markets, p)
        elif strategy in WS.MAKERS:
            base.update(_maker(c, links, venue, sims_of, res, priced, sp, st, strategy, now, race))
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
    return SG.tag(base, detail=dict(sport=sport, venue=venue), position=dict(venue=venue))


def price_stages(engine, engine_url, sp, model, st, ms, hist, data, race, stages, res, live, now, echo=print):
    """[(label, time, run id, priced_at)] and {label: sims} for the stages due. Live: each stage's stored run (its
    field; the sims re-made from it, deterministic), else priced now on the start list, stored and logged. A replay:
    one in-memory pricing on the classified field (the sweep's), run id None. now: the pass's time (priced_at of a
    stage priced in this pass)."""
    sport = sp["sport"]
    if not live:
        field = sorted(int(a) for a in res["athlete_id"]) if len(res) else None
        if not field:
            with engine.connect() as c:
                field = field_now(c, sp, race.race_id)
        sims = _price(model, hist, ms, race, field) if field else None
        if sims is None:
            return [], {}
        return [(lab, t, None, t) for lab, t in stages], {lab: sims for lab, _ in stages}
    with engine.connect() as c:
        have = stored_runs(c, sport, race.event_key, st.model_key)
        field = None
    out, sims_of, made = [], {}, {}
    for lab, t in stages:
        if lab in have:
            rid, at, f = have[lab]
        else:
            if field is None:
                with engine.connect() as c:
                    field = field_now(c, sp, race.race_id)
            f, rid, at = field, None, now
        key = tuple(sorted(int(a) for a in f))
        if key not in made:
            made[key] = _price(model, hist, ms, race, key) if key else None
        sims = made[key]
        if sims is None:
            echo(f"  {lab}: no price (no field, or no results before the race)")
            break
        if rid is None:
            rid = save_live_run(engine_url, sp, model, st, ms, race, sims, lab, key, SW.data_key(data), now)
            echo(f"  {lab}: priced (run #{rid})")
        out.append((lab, t, rid, at))
        sims_of[lab] = sims
    return out, sims_of


def stage_markets(v, links, sims_of, res, read_at, min_volume_24h, thin):
    """season_sweep.race_markets with each stage's own fairs (its run's sims): one market list, stage by stage."""
    per = {lab: SW.race_markets(v, links, sims_of[lab], res, [(lab, t)], min_volume_24h, thin) for lab, t in read_at}
    first = per[read_at[0][0]]
    return [dict(m, stages=[per[lab][i]["stages"][0] for lab, _ in read_at]) for i, m in enumerate(first)]


def _maker(c, links, venue, sims_of, res, priced, sp, st, strategy, now, race):
    """The maker replay up to now on season_sweep.maker_event (each stage's fairs from its run)."""
    from racinglines.markets.strategies import maker_replay as R
    from racinglines.pipelines import weekend_sweep as WS
    stages = [(lab, t) for lab, t, _, _ in priced]
    day = pd.Timestamp(race.start) + pd.Timedelta(days=int(sp.get("race_day_offset", 0)))
    rd = SW.mode_of(sp["sport"], st) == "race_day"
    until = day + pd.Timedelta(hours=float(SW.spec(sp["sport"])["race_day_quote_until_hours" if rd else "quote_until_hours"]))
    ev = SW.maker_event(c, links, venue, sims_of[stages[-1][0]], res, stages, max(until, stages[-1][1]),
                        books=st["fill"] == "queue")
    from racinglines.pipelines import position_replay as P
    for m in ev["markets"]:                       # each stage quotes around its own run's fair
        link = m.link or {}
        if link.get("athlete_id") is not None:
            b = (link.get("params") or {}).get("opponent_id")
            m.fairs.update({i: P.fair(sims_of[lab], link["prediction"], int(link["athlete_id"]), b)
                            for i, (lab, _) in enumerate(stages)})
    ev = dict(ev, markets=[m for m in ev["markets"] if m.kind in st["market_kinds"]])
    return SG.maker_run(ev, st, strategy, now, {i: lab for i, (lab, _) in enumerate(stages)}, venue)


def refresh(engine, engine_url, race, links, venue, now, echo=print):
    """The venue's tape for the race's markets over the last TAPE_LOOKBACK (position_replay.pull; idempotent)."""
    from racinglines.db.config import get_session
    from racinglines.pipelines import position_replay as P
    start = (now - TAPE_LOOKBACK).tz_localize("UTC").to_pydatetime()
    end = now.tz_localize("UTC").to_pydatetime()
    try:
        with engine.connect() as c, get_session(engine_url) as s:
            n = P.pull(s, c, [(race, links, start, end)], venue, echo=lambda *a: None)
            s.commit()
        echo(f"  {venue}: {n['prices']} price points, {n['trades']} trades")
    except Exception as ex:                       # noqa: BLE001  (a venue down: trade on what is stored)
        echo(f"  {venue}: refresh failed ({ex}); using the stored tape")


# --- every user -------------------------------------------------------------------------------------------

def run_all(engine, engine_url, sport, venues=None, users=None, profile_ref=None, now=None, event="next", fetch=True,
            alert=True, echo=print):
    """Every active user with a profile for `sport` on each venue (`venues`, default every exchange the sport's
    schema lists), one computation per distinct profile, stored and alerted by signals._run_one. F1 on Polymarket is
    signals.run_all (the timer's pass), unchanged."""
    from racinglines.pipelines import profiles as PF
    report = []
    for venue in venues or SW.venues(sport):
        if SW.mode_of(sport) == "sessions":      # F1: the session-schedule engine, signals.run_all
            report += SG.run_all(engine, engine_url, users=users, profile_ref=profile_ref, now=now, event=event,
                                 fetch=fetch, alert=alert, echo=echo, venue=venue)
            continue
        with engine.connect() as c:
            targets = [(uid, prof) for uid, name, _, prof in PF.assigned(c, venue=venue, sport=sport)
                       if not users or name in users]
            if profile_ref is not None:
                prof = dict(PF.load(c, profile_ref), sport=sport)
                targets = [(uid, prof) for uid, _ in targets]
        groups, rate, kelly = {}, {}, {}
        for uid, prof in targets:
            groups.setdefault(SG.combo_key(prof), (prof, []))[1].append(uid)
            rate[uid] = prof.get("follow_rate")
            kelly[uid] = PF.sizing_of(prof)
        cache = {}
        for prof, uids in groups.values():
            g = SG.guard(prof, venue, sport) if profile_ref is None else None
            prof = dict(prof, sport=sport)
            echo(f"{prof['name']} ({prof['strategy']}, {sport} on {venue}) for users {uids}")
            out = compute(engine, engine_url, prof, now=now, event=event, fetch=fetch, echo=echo, cache=cache)
            report.append(SG._run_one(engine, engine_url, prof, out, uids, rate, now, alert, echo, upcoming=None,
                                      kelly=kelly, guard=g))
    return report
