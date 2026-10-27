"""
Stages from the sport's schema (docs/backtest-core.md, step 5): the moments an event is priced and traded,
as new information arrives. A sport's `[stages]` table says how its session schedule becomes stages:

    [stages]
    pre_label = "pre-weekend"   # the first stage, before any running ...
    pre_minutes = 60            # ... this long before the first session
    lag_minutes = 30            # a session's data is used this long after it ends
    until = "Race"              # stages stop before this session (it is never traded)
    closes = { race_pole = "Qualifying" }   # a market kind closes when this session starts (or a list of
                                            # names: the first one on the schedule, e.g. SQ's two names)

    build(sessions, "f1")   -> dict(stages=[(label, cutoff)], closes={kind: time or None}, until=time or None)

sessions: [(name, start)] in time order (naive UTC). A session counts when the schema's session schedule
(`[sessions.schedule]`: name -> [minutes, short label]) lists it; its stage is "after <short label>".
"""

from datetime import timedelta

from racinglines import sports

DEFAULTS = dict(pre_label="pre-weekend", pre_minutes=60, lag_minutes=30, until=None, closes={}, pre_race_minutes=None)
PRE_RACE = "pre-race"


def spec(sport):
    return dict(DEFAULTS, **sports.load(sport).get("stages", {}))


def schedule(sport):
    """{session name: (minutes, short label)} from the schema."""
    return {k: (m, s) for k, (m, s) in sports.load(sport).get("sessions", {}).get("schedule", {}).items()}


def build(sessions, sport):
    sp, sched = spec(sport), schedule(sport)
    until = next((t for n, t in sessions if n == sp["until"]), None)
    if not sessions or (sp["until"] and until is None):
        return None
    stages = [(sp["pre_label"], sessions[0][1] - timedelta(minutes=sp["pre_minutes"]))]
    for name, start in sessions:
        if name in sched:
            cut = start + timedelta(minutes=sched[name][0] + sp["lag_minutes"])
            if until is None or cut < until:
                stages.append((f"after {sched[name][1]}", cut))
    closes = {k: next((t for n, t in sessions if n in ([s] if isinstance(s, str) else list(s))), None)
              for k, s in sp["closes"].items()}
    return dict(stages=stages, closes=closes, until=until)


def pre_race(until, sport, last=None):
    """(PRE_RACE, time) of the live pre-race stage, `[stages] pre_race_minutes` before `until` (lights out), or None
    when the schema has none, there is no `until`, or it would not come after `last` (the last stage's cutoff). Only
    the live signal pass trades it (pipelines/signals.price_stages_now), priced by the last stage's run: no session
    runs in between, only the market moves (owner, 2026-10-26: "everyone gets a valid trading window before finals").
    build() leaves it out, so backtests, sweeps and the golden tests see the same stages as before."""
    m = spec(sport).get("pre_race_minutes")
    if not m or until is None:
        return None
    t = until - timedelta(minutes=m)
    return None if last is not None and t <= last else (PRE_RACE, t)


def is_open(kind, t, closes):
    """A market of `kind` can still be traded at t (its closing session hasn't started)."""
    c = (closes or {}).get(kind)
    return c is None or t < c


def known_at(sport, sessions, label):
    """The sessions whose data is in at stage `label` of an event, from its sessions [(schedule name, start)] in
    running order: none at the first stage ([stages] pre_label), else every session up to and including the first one
    whose short label makes `label` ("after <short>"); None when the event has no such session. build()'s rule (a
    session's data is in after it ends) for sessions that don't overlap, without needing their times: the walk-forward
    engine's stages (`backtest walk-forward --stage`)."""
    sp, sched = spec(sport), schedule(sport)
    if label == sp["pre_label"]:
        return []
    run = [(n, t) for n, t in sessions if n in sched]
    for i, (n, _) in enumerate(run):
        if f"after {sched[n][1]}" == label:
            return run[:i + 1]
    return None


def at_stage(sport, sessions, label):
    """(sessions known, exact stage cutoff) or None when the event has no such stage.

    A session's data is usable after its scheduled duration and the schema's lag, matching build()'s stage timing.
    """
    known = known_at(sport, sessions, label)
    if known is None:
        return None
    sp, sched = spec(sport), schedule(sport)
    run = [(n, t) for n, t in sessions if n in sched]
    if not run:
        if label == sp["pre_label"]:
            return known, None
        return None
    if label == sp["pre_label"]:
        return known, run[0][1] - timedelta(minutes=sp["pre_minutes"])
    for n, start in run:
        minutes, short = sched[n]
        if f"after {short}" == label:
            return known, start + timedelta(minutes=minutes + sp["lag_minutes"])
    return None
