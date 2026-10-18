"""
The season sweep for any sport: every raced event of a season, traded through its stages by the taker modes
(update / hold / last / early) and the maker variants (maker, maker_flat, maker_skew, maker_widen, maker_all) of
pipelines/weekend_sweep.py, against one exchange's recorded tape (markets/venue_replay.py EXCHANGES: Polymarket,
Kalshi, OG.com), then settled on the result. Tuned on one season, held out on another (the search, sweeps/*.toml).

Nothing here names a sport. The sport's schema (sports/<code>.toml) says everything that differs:

    [sweep] stages      the default stage mode, i.e. where the stages and the fair values come from; a sweep picks
                        another one the schema supports with its `stages` setting (sweep_settings.py; a search job's
                        `stages = "weekend"`), so the earlier weekend-based results stay reproducible beside a
                        session-aware path:
        "sessions"      the session schedule ([sessions.schedule], [stages]) through the stage engine (core/stages.py
                        build): a stage before any running, then one after each session (its start + its minutes +
                        lag_minutes), until the race. Needs [sessions.schedule]. Where the sessions come from:
                        - F1 (no [sessions] time_key): FastF1's event schedule, one diagnostic run per stage priced as
                          of each cutoff (pipelines/weekend_sweep.py run_sweep; F1's default and only mode, its results
                          as they were)
                        - [sessions] time_key (NASCAR, MotoGP): the event's stored rounds whose kind is a schedule
                          name, each starting at rounds.extra[time_key] (session_stages). A round that stores no time
                          is not a stage; the race without a stored start ends the stages at [stages]
                          until_fallback_hours on race day; an event with no stored session time at all is traded
                          on its fixed [replay] stages instead (its weekends row says format "weekend"). The pricing
                          model reprices at every stage (stage_sims) with what is known by then: the start list and
                          the sessions run so far (Event.info "field", "sessions")
        "weekend"       fixed weekend stages, [replay] stages (hours from 00:00 UTC on race day), priced once per race
                        by [sport] pricing_model from results strictly before the event (a results-only model: one
                        pricing serves every stage), settled by markets/kinds.settle (pipelines/position_replay.py;
                        NASCAR's and MotoGP's default). Needs [replay] stages
    kinds               [sweep] kinds, else [markets] weekend_kinds ("sessions", where the schema has them) or
                        [replay] kinds
    venues              [markets] venues that are exchanges, then OG.com where exchanges/og.toml lists the sport
    [sweep] late_stages the stages the stage-aware taker ("early") skips (default none)
    [sweep] quote_until_hours
                        "weekend": the hour (UTC) of race day the maker stops quoting

    run(engine, engine_url, "nascar", 2026, settings=settings_class("nascar").from_dict({"venue": "kalshi"}))
        -> dict(weekends, by_stage, by_kind, totals, trades, scores, calibration, reliability, params)
           params carries sport, venue, the settings (settings_key, model_key) and the data fingerprint (data_key)
    settings_class(sport)       the sweep's settings for the sport: its pricing model's own settings plus the shared
                                entry timing, taker, maker and markets settings (pipelines/sweep_settings.py), with the
                                stage labels, kinds and venues of the schema as their choices (F1: sweep_settings.Settings)

Read-only: the "weekend" mode stores nothing (the CLI's --save stores the sweep run itself, kind 'sweep').
"""

import hashlib
import zlib
from dataclasses import replace
from datetime import timedelta
from functools import cache

import numpy as np
import pandas as pd

from racinglines import progress as PG
from racinglines import sports
from racinglines.pipelines import sweep_settings as SS

MODES = SS.STAGE_MODES + ("race_day",)   # ("sessions", "weekend"), plus "race_day": the weekend stages and [replay]
                                         # race_day_stages (opt-in; F1's settings keep their two modes)


# --- the schema ------------------------------------------------------------------------------------------

def spec(sport):
    """The sport's [sweep] table (its default stage mode, late stages, quoting window), with defaults."""
    s = sports.load(sport)
    sw = dict(s.get("sweep") or {})
    if not sw:
        raise ValueError(f"sport {sport!r} has no [sweep] table in sports/{sport}.toml (stages, ...)")
    if sw.get("stages") not in modes(sport):
        raise ValueError(f"sports/{sport}.toml [sweep] stages {sw.get('stages')!r}: one of the modes its schema "
                         f"supports, {modes(sport)} (sessions: [sessions.schedule]; weekend: [replay] stages)")
    sw = dict(dict(late_stages=[], quote_until_hours=0.0), **sw)
    return dict(dict(race_day_quote_until_hours=sw["quote_until_hours"]), **sw)


def modes(sport):
    """The stage modes the sport's schema supports: "sessions" with a session schedule ([sessions.schedule]),
    "weekend" with fixed weekend stages ([replay] stages), "race_day" with those and [replay] race_day_stages."""
    s = sports.load(sport)
    have = dict(sessions=bool(s.get("sessions", {}).get("schedule")), weekend=bool(s.get("replay", {}).get("stages")),
                race_day=bool(s.get("replay", {}).get("stages") and s["replay"].get("race_day_stages")))
    return tuple(m for m in MODES if have[m])


def mode_of(sport, settings=None):
    """The stage mode of a sweep: its `stages` setting, else the schema's default ([sweep] stages)."""
    m = (settings or {}).get("stages") or spec(sport)["stages"]
    if m not in modes(sport):
        raise ValueError(f"{sport}: stages {m!r} is not a mode its schema supports, {modes(sport)}")
    return m


def time_key(sport):
    """The rounds.extra key holding a session's start ([sessions] time_key): the "sessions" mode reads the event's
    stored rounds (NASCAR, MotoGP). None: its sessions come from FastF1's schedule (F1, pipelines/weekend_sweep.py)."""
    return sports.load(sport).get("sessions", {}).get("time_key")


def engine_of(sport):
    """The schema's default stage mode (mode_of without settings)."""
    return mode_of(sport)


def supports(sport):
    """True when the sport's schema has what a sweep needs: a [sweep] stage mode its schema supports and a pricing model."""
    try:
        spec(sport)
        from racinglines.models import race_model as RM
        RM.model_class(sport)
    except (ValueError, FileNotFoundError):
        return False
    return True


def kinds(sport, mode=None):
    """The market kinds the sweep trades: [sweep] kinds, else the stage mode's own list."""
    s, sw = sports.load(sport), spec(sport)
    if sw.get("kinds"):
        return tuple(sw["kinds"])
    if (mode or sw["stages"]) == "sessions" and s["markets"].get("weekend_kinds"):
        return tuple(s["markets"]["weekend_kinds"])
    return tuple(s["replay"]["kinds"])


def venues(sport):
    """The exchanges a sweep of the sport can trade: [markets] venues that are exchanges (markets/venue_replay.py),
    in the schema's order, then OG.com where its schema (exchanges/og.toml) lists the sport."""
    from racinglines import exchanges
    from racinglines.markets.venue_replay import EXCHANGES
    out = [v for v in sports.load(sport).get("markets", {}).get("venues", ()) if v in EXCHANGES]
    out += [v for v in EXCHANGES if v not in out and sport in _exchange_sports(exchanges, v)]
    return tuple(out)


def _exchange_sports(exchanges, code):
    """The sports an exchange's own schema (exchanges/<code>.toml) lists; () for one defined in code."""
    try:
        return exchanges.sports(code)
    except FileNotFoundError:                    # Polymarket, Kalshi: no schema file
        return ()


def stage_labels(sport, mode=None):
    """The labels of the sport's stages, in order ("weekend": [replay] stages; "sessions": [stages] pre_label, then
    "after <short label>" of each [sessions.schedule] entry, in the schedule's order: F1's are
    sweep_settings.STAGES)."""
    if (mode or engine_of(sport)) == "sessions":
        from racinglines.core import stages as STG
        after = [f"after {short}" for _, short in STG.schedule(sport).values()]
        return (STG.spec(sport)["pre_label"], *dict.fromkeys(after))
    from racinglines.pipelines import position_replay as P
    return tuple(str(label) for label, _ in P.stage_list(sports.load(sport)["replay"], mode or engine_of(sport)))


def all_labels(sport):
    """Every stage label a sweep of the sport can trade, over the modes its schema supports (the default mode's
    first): a "sessions" sweep also trades the [replay] stages of an event with no stored session time."""
    return tuple(dict.fromkeys(x for m in (engine_of(sport), *modes(sport)) for x in stage_labels(sport, m)))


def late_labels(sport, mode=None):
    """The stages the stage-aware taker ("early") skips by default in a mode: [sweep] late_stages among its labels
    (a "sessions" sweep: also the [replay] ones, for its fallback events)."""
    labels = stage_labels(sport, mode) + (stage_labels(sport, "weekend") if (mode or engine_of(sport)) == "sessions"
                                          and "weekend" in modes(sport) else ())
    return tuple(x for x in dict.fromkeys(labels) if x in spec(sport)["late_stages"])


@cache
def settings_class(sport):
    """The sweep's settings for the sport. A pricing model whose settings already hold the shared sweep settings
    (F1's: sweep_settings.Settings) is used as it is; any other gets them added, with the schema's stage labels,
    kinds and venues as their choices and defaults (the model's own settings keep their names, so its model_key is
    the same as its walk-forward's)."""
    from racinglines.models import race_model as RM
    base = RM.model_class(sport).Settings
    shared = [s.name for s in SS.SHARED_SETTINGS]
    if all(n in base.BY for n in shared):
        return base
    labels, every, ks, vs = stage_labels(sport), all_labels(sport), kinds(sport), venues(sport)
    if not vs:
        raise ValueError(f"sport {sport!r}: no exchange to sweep ([markets] venues, exchanges/*.toml)")
    over = dict(       # defaults: the schema's default mode; choices: the labels of every mode it supports
        taker_stages=dict(default=labels, choices=every),
        late_stages=dict(default=late_labels(sport), choices=every),
        min_edge_by_kind=dict(choices=ks),
        coherence_tol_by_kind=dict(choices=ks),
        market_kinds=dict(default=ks, choices=ks, help=f"Any of sports/{sport}.toml's sweep kinds."),
        venue=dict(choices=vs, unset=vs[0], help=f"Whose markets and recorded tape the strategies trade: one of "
                                                 f"{', '.join(vs)}. Unset = {vs[0]}."),
        stages=dict(choices=modes(sport), unset=engine_of(sport),
                    help=f"Stage mode, one of {', '.join(modes(sport))}. Unset = {engine_of(sport)} (sports/{sport}.toml "
                         "[sweep] stages)."))
    extra = [replace(s, **over.get(s.name, {})) for s in SS.SHARED_SETTINGS]
    spec_ = list(base.SPEC) + [s for s in extra if s.name not in base.BY]
    return type(f"{base.__name__}Sweep", (base,), dict(SPEC=spec_, BY={s.name: s for s in spec_}, MODEL=list(base.MODEL),
                                                       DEFAULT_VENUE=vs[0], SPORT=sport))


# --- the season ------------------------------------------------------------------------------------------

def run(engine, engine_url, sport, year, rounds=None, settings=None, fetch=False, reprice=False, echo=print):
    """The sport's season sweep in its stage mode (mode_of: the `stages` setting, else the schema's default).
    settings: settings_class(sport) (None = the defaults). rounds: event numbers within the season (1 = its first
    raced event), None = all. fetch / reprice: the "sessions" mode's (Polymarket download, re-pricing stored stages)."""
    if mode_of(sport, settings) == "sessions" and not time_key(sport):
        from racinglines.pipelines import weekend_sweep as SW
        return SW.run_sweep(engine, engine_url, year, rounds, fetch=fetch, reprice=reprice, settings=settings, echo=echo)
    return replay_sweep(engine, sport, year, rounds, settings, echo)


def _rounds(rs, rounds):
    rs = rs.assign(round=np.arange(1, len(rs) + 1))
    return rs[rs["round"].isin(list(rounds))] if rounds else rs


def data_key(data):
    """Fingerprint of the model's data (row count and numeric column sums, database ids left out), as
    sweep_settings.data_key does for F1's stages."""
    if not isinstance(data, pd.DataFrame):
        return None
    num = data.select_dtypes("number")
    num = num[[c for c in num.columns if not (c == "id" or str(c).endswith("_id"))]]
    blob = f"{len(data)}:{float(num.fillna(0).to_numpy().sum()) if len(num.columns) else 0:.8g}"
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def replay_sweep(engine, sport, year, rounds=None, settings=None, echo=print):
    """A sweep priced by the sport's pricing model (every mode but F1's "sessions"): per race, the field (its start
    list, start_list), its stages ("weekend": the [replay] stages; "sessions": session_stages, else the [replay]
    stages for an event with no stored session time), the model's pricing (one pricing as of the event's first day
    for the [replay] stages; stage_sims at every session stage), each stage read on the venue (price, bid / ask,
    24 h volume, open, coherent group), the takers and makers traded, then the race settled."""
    from racinglines.models.race_model import Event
    from racinglines.pipelines import position_replay as P
    from racinglines.pipelines import weekend_sweep as SW
    cls = settings_class(sport)
    st = settings if settings is not None else cls.from_dict()
    mode = mode_of(sport, st)
    venue = SS.venue_of(st)
    if venue not in venues(sport):
        raise ValueError(f"{sport}: venue {venue!r} is not one of {venues(sport)}")
    sw = spec(sport)
    sp = dict(P.spec(sport), kinds=[k for k in kinds(sport, mode) if k in st["market_kinds"]])
    model = P.model_for(sp)
    ms = model.Settings.from_dict({k: st[k] for k in model.Settings.BY})
    base, params_list = SW.taker_params(st)
    if mode != engine_of(sport) and st["late_stages"] == st.BY["late_stages"].default:
        base = replace(base, late_stages=late_labels(sport, mode))            # the mode's own default late stages
        params_list = [replace(q, late_stages=base.late_stages) for q in params_list]
    with engine.connect() as conn:
        rs = _rounds(P.races(conn, sp, [year]), rounds)
        lk = P.links(conn, sp, venue)
        times = session_times(conn, sport, rs["race_id"].tolist()) if mode == "sessions" else {}
    echo(f"progress {sport} {year} on {venue}: {len(rs)} races, {len(lk)} markets of kinds {', '.join(sp['kinds'])}")
    data = model.load(engine.url.render_as_string(hide_password=False))
    hist = model.history(data, ms)
    tol = SS.parse_map(st["coherence_tol_by_kind"])
    thin = st["thin_edge_mult"] is not None
    plans = {int(r.race_id): race_plan(sport, mode, r, times.get(int(r.race_id), []), sp, sw) for r in rs.itertuples()}
    if mode == "sessions":
        fell = [str(r.event_key) for r in rs.itertuples() if plans[int(r.race_id)]["format"] == "weekend"]
        echo(f"  {len(rs) - len(fell)} of {len(rs)} events on their stored sessions ([sessions] time_key "
             f"{time_key(sport)!r})" + (f"; no session time stored, on the [replay] stages: {', '.join(fell)}" if fell else ""))
    events = [dict(round=int(r.round), event_key=r.event_key, event=str(r.name), format=plans[int(r.race_id)]["format"],
                   stages=len(plans[int(r.race_id)]["stages"]), last_run_id=None, race=r) for r in rs.itertuples()]

    def trade_event(ev, plist, widen_kinds):
        r = ev["race"]
        race_links = lk[lk["race_id"] == r.race_id] if len(lk) else lk
        if not len(race_links):
            return None
        plan = plans[int(r.race_id)]
        stages, until = plan["stages"], plan["until"]
        with engine.connect() as conn:
            res = P.race_results(conn, r.race_id)
            if res.empty:
                return None
            field = start_list(res)
            seed = [ms.rng_seed, zlib.crc32(str(r.event_key).encode())]
            if plan["sessions"] is None:                 # the [replay] stages: one pricing serves every stage
                sims = model.price(hist, Event(id=r.event_key, season=int(r.season), cutoff=r.start, name=str(r.name),
                                               info={"field": field}), ms, np.random.default_rng(seed))
            else:
                sims = stage_sims(model, hist, ms, r, field, stages, plan["sessions"], sport, seed)
            v = P._venue(conn, venue, race_links, stages, sp, tape=st.get("tape_source"))
            v.tol_by_kind = dict(tol)
            if thin:
                v.load_books(conn, min(t for _, t in stages) - timedelta(hours=1), max(t for _, t in stages))
            markets = race_markets(v, race_links, sims, res, stages, st["min_volume_24h"], thin, plan["closes"])

            def tape():
                return maker_event(conn, race_links, venue, sims, res, stages, until, books=st["fill"] == "queue")
            return SW.trade(markets, [lab for lab, _ in stages], plist, st, tape, widen_kinds, tuple(sp["kinds"]), echo)

    order = stage_labels(sport, mode) + (stage_labels(sport, "weekend") if mode == "sessions" else ())
    out = SW.season(events, trade_event, st, params_list, tuple(dict.fromkeys(order)), echo=echo)
    SW.season_clv(engine, out, venue, sport, tuple(dict.fromkeys(order)), echo)
    params = dict({k: v for k, v in base.__dict__.items() if k != "stages" and not
                   (k in ("scale", "max_deployed", "min_edge_by_kind", "thin_edge_mult", "kelly", "bankroll", "root_unit")
                    and v == SW.RB.TakerParams.__dataclass_fields__[k].default)},
                  sport=sport, venue=venue, model=model.name, stages_mode=mode,
                  n_sims=ms.get("sims"), stages=[list(x) for x in sp["stages"]], kinds=list(sp["kinds"]),
                  min_volume_24h=st["min_volume_24h"], coherence_tol=P.coherence_tol(sp, venue),
                  group_target=dict(sp.get("group_target") or {}), quote_until_hours=sw["quote_until_hours"],
                  settings=st.to_json(), settings_key=st.key, model_key=st.model_key, data_key=data_key(data),
                  label=st.label(), weekends=len(out["weekends"]))
    if mode == "sessions":       # the schedule it ran on, and how many events had stored sessions or fell back
        from racinglines.core import stages as STG
        params["sessions"] = dict(time_key=time_key(sport), labels=list(stage_labels(sport, mode)),
                                  schedule={k: list(v) for k, v in STG.schedule(sport).items()},
                                  stages=dict(STG.spec(sport)),
                                  events=sum(p["format"] == "sessions" for p in plans.values()),
                                  fallback_events=sum(p["format"] == "weekend" for p in plans.values()))
    return dict(out, params=params)


def start_list(res):
    """The race's start list: its race round's entrants but the non-starters (status DNS), as the model's own
    events() builds the field (models/model_global.py), so the sweep never prices a car or rider that didn't start."""
    status = res["status"].fillna("OK").astype(str).str.upper()
    return sorted(int(a) for a in res.loc[status != "DNS", "athlete_id"])


def session_times(conn, sport, race_ids):
    """{race_id: [(round kind, start)]}: the races' stored rounds whose kind is a [sessions.schedule] name or the
    [stages] until session, started at rounds.extra[time_key] (naive UTC). A round that stores no time is left out."""
    from sqlalchemy import text

    from racinglines.core import stages as STG
    if not race_ids:
        return {}
    names = [*STG.schedule(sport), STG.spec(sport)["until"]]
    df = pd.read_sql(text("""SELECT race_id, kind, extra->>:k AS start FROM rounds
                             WHERE race_id = ANY(:r) AND kind = ANY(:n) AND extra->>:k IS NOT NULL
                             ORDER BY race_id, ordinal, id"""),
                     conn, params=dict(k=time_key(sport), r=[int(x) for x in race_ids], n=names))
    out = {}
    for rid, kind, start in df.itertuples(index=False):
        t = pd.Timestamp(start)
        out.setdefault(int(rid), []).append((str(kind), t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t))
    return out


def race_plan(sport, mode, race, sessions, sp, sw):
    """One race's stages in the sweep's mode: dict(stages=[(label, time)], until (the maker quotes until then),
    closes, sessions, format). "sessions": session_stages from its stored session times (format "sessions"), or
    the [replay] stages when none is stored (format "weekend"); "weekend": the [replay] stages (format None, as
    before); "race_day": those and [replay] race_day_stages, the maker quoting until [sweep]
    race_day_quote_until_hours (format None)."""
    from racinglines.pipelines import position_replay as P
    if mode == "sessions":
        got = session_stages(sport, race.start, sessions, sp)
        if got is not None:
            return dict(got, format="sessions")
    day = pd.Timestamp(race.start) + pd.Timedelta(days=int(sp.get("race_day_offset", 0)))
    rd = mode == "race_day"
    hours = sw["race_day_quote_until_hours" if rd else "quote_until_hours"]
    stages = P.stage_times(race.start, dict(sp, stages=P.stage_list(sp, "race_day" if rd else "weekend")))
    return dict(stages=stages, until=day + pd.Timedelta(hours=float(hours)),
                closes={}, sessions=None, format="weekend" if mode == "sessions" else None)


def session_stages(sport, race_start, sessions, sp):
    """The "sessions" mode's stages of one event, from its stored session times through the stage engine F1 uses
    (core/stages.py build): dict(stages, until, closes, sessions), or None when no session of the schedule stores a
    time. The race's start, when it isn't stored, is [stages] until_fallback_hours: hours from 00:00 UTC on race
    day (race_day_offset days after the event's first day). No stage is at or after the race's start."""
    from racinglines.core import stages as STG
    sched, until = STG.schedule(sport), STG.spec(sport)["until"]
    run = sorted(((n, t) for n, t in sessions if n in sched), key=lambda x: x[1])
    if not run:
        return None
    end = next((t for n, t in sessions if n == until), None)
    if end is None:
        day = pd.Timestamp(race_start) + pd.Timedelta(days=int(sp.get("race_day_offset", 0)))
        end = day + pd.Timedelta(hours=float(STG.spec(sport).get("until_fallback_hours", 0)))
    b = STG.build(run + [(until, end)], sport)
    if b is None or b["stages"][0][1] >= b["until"]:
        return None
    return dict(stages=[(lab, pd.Timestamp(t)) for lab, t in b["stages"]], until=pd.Timestamp(b["until"]),
                closes=b["closes"], sessions=run)


def stage_sims(model, hist, ms, race, field, stages, sessions, sport, seed):
    """{stage label: OutcomeSims}: the pricing model asked at every stage with what is known by then. Its as-of for
    results is the event's first day, as in the "weekend" mode (results strictly before the event); Event.info
    carries the start list ("field") and the sessions whose data is in by the stage ("sessions": [(name, start)];
    a session's data is in [sessions.schedule] minutes + [stages] lag_minutes after it starts). Every stage draws
    from the event's own seed, so a fair moves only when what the model reads moves: a model that reads race
    results only (GlobalModel) prices every stage the same. Priced once per distinct information set."""
    from racinglines.core import stages as STG
    from racinglines.models.race_model import Event
    sched, lag = STG.schedule(sport), STG.spec(sport)["lag_minutes"]
    out, memo = {}, {}
    for lab, t in stages:
        known = tuple((n, s) for n, s in sessions if s + timedelta(minutes=sched[n][0] + lag) <= t)
        if known not in memo:
            memo[known] = model.price(hist, Event(id=race.event_key, season=int(race.season), cutoff=race.start,
                                                  name=str(race.name), info={"field": field, "sessions": list(known)}),
                                      ms, np.random.default_rng(seed))
        out[lab] = memo[known]
    return out


def _sims_at(sims, label):
    """The simulations a stage is priced from: one pricing for every stage, or {label: sims} (stage_sims)."""
    return sims.get(label) if isinstance(sims, dict) else sims


def race_markets(venue, race_links, sims, res, stages, min_volume_24h, thin_depth=False, closes=None):
    """Per market: each stage's fair, exchange price (and bid / ask where the venue quotes one, so a taker buys at
    the ask and sells at the bid), tradeable flag and the outcome, in the shape taker_weekend.run_weekend reads.
    The F1 sweep's rules (weekend_sweep.weekend_markets): tradeable = priced inside (0, 1), over the volume floor,
    open (before its end date and, with `closes`, before its kind's closing session: core/stages.py is_open), a
    coherent group; with thin_depth, a stage that fails only the volume floor is marked `thin` with the size at the
    touch. sims: one pricing for every stage, or {stage label: sims}. Fairs and settlement: pipelines/position_replay.py."""
    from racinglines.core import stages as STG
    from racinglines.pipelines import position_replay as P
    coherent = {(k, lab): venue.coherent(k, t) for lab, t in stages for k in venue.group_target}
    out = []
    for link in race_links.to_dict("records"):
        kind, a = link["prediction"], int(link["athlete_id"])
        b = (link["params"] or {}).get("opponent_id")
        st = []
        for lab, t in stages:
            at = _sims_at(sims, lab)
            f = P.fair(at, kind, a, b) if at is not None else None
            price, _, liquid = venue.view(link, t, min_volume_24h)
            open_ = P._open(link, t) and STG.is_open(kind, t, closes)
            ok = liquid and f is not None and open_ and coherent.get((kind, lab), True)
            stage = dict(label=lab, t=t, fair=f, price=price, tradeable=ok, open=open_)
            q = venue.quote(link["token_id"], t)
            if q is not None:
                stage.update(bid=q[0], ask=q[1])
            if thin_depth and not liquid and price is not None and 0 < price < 1 and f is not None and open_ \
                    and coherent.get((kind, lab), True):
                d = venue.touch_depth(link["token_id"], t)
                if d is not None and (d[0] > 0 or d[1] > 0):
                    stage.update(thin=True, depth_yes=d[0], depth_no=d[1])
            st.append(stage)
        subject = link.get("athlete") or link.get("group_title")
        if kind == "race_h2h":
            subject = f"{subject} v {b}"
        out.append(dict(key=link["token_id"], kind=kind, subject=subject, stages=st, link=link,
                        outcome=P.settle(kind, a, b, res)))
    return out


def maker_event(conn, race_links, venue, sims, res, stages, until, books=False):
    """The maker replay's event (maker_replay.replay): the race's markets from the venue's tape, each stage quoting
    from its time to the next stage's (the last to `until`, then pulled), fairs keyed by the stage's index. sims:
    one pricing for every stage, or {stage label: sims}."""
    from racinglines.markets.strategies import maker_replay as R
    from racinglines.pipelines import position_replay as P
    times = [t for _, t in stages] + [pd.Timestamp(until)]
    st = [dict(run_id=i, start=R._ns(times[i]), end=R._ns(times[i + 1]), session_end=i == len(stages) - 1)
          for i in range(len(stages)) if times[i + 1] > times[i]]
    if not st:
        raise ValueError(f"no maker stage before {until} (quote_until_hours)")

    def fairs_of(link):
        b = (link["params"] or {}).get("opponent_id")
        out = {}
        for s in st:
            at = _sims_at(sims, stages[s["run_id"]][0])
            out[s["run_id"]] = P.fair(at, link["prediction"], int(link["athlete_id"]), b) if at is not None else None
        return out

    def outcome_of(link, mids):
        return P.settle(link["prediction"], int(link["athlete_id"]), (link["params"] or {}).get("opponent_id"), res)

    t0, t1 = st[0]["start"] - int(48 * 3600e9), st[-1]["end"] + int(6 * 3600e9)
    markets = R.tape_markets(conn, race_links, venue, t0, t1, fairs_of, outcome_of, books)
    return dict(markets=markets, stages=st, qual_start=None, runs=[dict(run_id=s["run_id"], cutoff=s["start"]) for s in st])


# --- saving, for the CLI and the search -------------------------------------------------------------------

def save(engine_url, sport, year, rounds, out, reliability=False):
    """Store one sweep (model_runs kind 'sweep', the sport's competition, params with sport and venue); its id."""
    from racinglines.db.config import get_session
    from racinglines.db.queries import records, save_model_run
    s_ = sports.load(sport)
    comp = s_["competition"]
    with get_session(engine_url) as s:
        return save_model_run(s, competition=comp["code"], season=year, category=next(iter(comp.get("categories") or {}), None),
                              model=out["params"].get("model") or "sweep", kind="sweep",
                              params=dict(out["params"], year=year, rounds=rounds),
                              metrics=dict(weekends=records(out["weekends"]), totals=out["totals"],
                                           by_stage=records(out["by_stage"]), by_kind=records(out["by_kind"]),
                                           scores=records(out["scores"]),
                                           **(dict(calibration=records(out["calibration"]),
                                                   reliability=records(out["reliability"])) if reliability else {}),
                                           **({"clv": out["clv"]} if out.get("clv") is not None else {})))
