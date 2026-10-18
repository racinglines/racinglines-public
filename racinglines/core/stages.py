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

DEFAULTS = dict(pre_label="pre-weekend", pre_minutes=60, lag_minutes=30, until=None, closes={})


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


def is_open(kind, t, closes):
    """A market of `kind` can still be traded at t (its closing session hasn't started)."""
    c = (closes or {}).get(kind)
    return c is None or t < c
