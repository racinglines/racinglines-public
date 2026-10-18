"""
Scheduled sessions: a race's sessions still to run, stored as rounds with a start time and no results, so a market
on an upcoming session has a close (pipelines/clv.closes reads each round's start from rounds.extra).

A scheduled round's extra is {<time_key>: "<UTC start>", "scheduled": true, "source": "<where the time came from>"};
time_key is the key the sport's own ingest writes (F1 session_date, NASCAR run_date_utc), so every reader of session
starts sees it unchanged. Rules:

  * a completed event is never touched;
  * a round the ingest wrote (no "scheduled" flag: the session has run) is kept as it is;
  * a scheduled round's start is updated when the source moves it, and nothing is ever deleted here;
  * the sport's ingest rebuilds a race's rounds when a session ends; it clears them with clear_for_ingest, which keeps
    the scheduled rounds of sessions still to come (F1 since 2026-10-13), so a mid-weekend ingest no longer drops them;
  * scripts/vm/schedule_sync.sh (racinglines-schedule-sync.timer) reruns the schedule every 6 hours and writes, after
    its own backup, only when a session was added or moved.

Times are the source's own; nothing here guesses one. A session with no start in the source is skipped and counted.
"""

from datetime import date

from sqlalchemy import select

from racinglines.db import models as m


def write_sessions(session, race_id, sessions, time_key, source):
    """Store `sessions` (dicts: kind, ordinal, name, start = the UTC start as text, in the sport's own format) as
    scheduled rounds of race `race_id`. Returns {"added": n, "moved": n, "unchanged": n, "ran": n}."""
    have = {r.kind: r for r in session.scalars(select(m.Round).filter_by(race_id=race_id))}
    out = dict(added=0, moved=0, unchanged=0, ran=0)
    for s in sessions:
        r = have.get(s["kind"])
        if r is None:
            session.add(m.Round(race_id=race_id, kind=s["kind"], ordinal=s["ordinal"], name=s.get("name"),
                                extra={time_key: s["start"], "scheduled": True, "source": source}))
            out["added"] += 1
        elif not (r.extra or {}).get("scheduled"):
            out["ran"] += 1
        elif r.extra.get(time_key) != s["start"] or r.ordinal != s["ordinal"]:
            r.extra = dict(r.extra, **{time_key: s["start"], "source": source})
            r.ordinal, r.name = s["ordinal"], s.get("name") or r.name
            out["moved"] += 1
        else:
            out["unchanged"] += 1
    session.flush()
    return out


def clear_for_ingest(session, race_id, kinds, final=False):
    """Delete race `race_id`'s rounds before an ingest rebuilds them from its files, except the scheduled rounds (no
    results) of sessions the ingest has no files for yet (kind not in `kinds`), which keep their start time until their
    own session is ingested. `final` (the race is in, the event completes) clears those too. Returns how many it kept."""
    from sqlalchemy import delete
    keep = [] if final else [r.id for r in session.scalars(select(m.Round).filter_by(race_id=race_id))
                             if (r.extra or {}).get("scheduled") and r.kind not in kinds
                             and not session.scalars(select(m.Result.id).filter_by(round_id=r.id).limit(1)).first()]
    session.execute(delete(m.Round).where(m.Round.race_id == race_id, m.Round.id.not_in(keep)))
    return len(keep)


def upcoming_race(session, source, source_key):
    """(event, race) of a stored event by its source key, or (event or None, None) when there is none or it is
    completed (never written to)."""
    ev = session.scalars(select(m.Event).filter_by(source=source, source_key=source_key)).first()
    if ev is None or ev.status == "completed":
        return ev, None
    race = session.scalars(select(m.Race).filter_by(event_id=ev.id)).first()
    return ev, race


def in_window(day, today, through):
    """Is an event dated `day` in [today - 3 days, through]? (A weekend that started a few days ago is still filled.)"""
    return (day - today).days >= -3 and day <= through


def parse_day(text):
    return date.fromisoformat(str(text)[:10])


def undo(session, sport_source):
    """Delete the scheduled rounds (flagged, no results) of `sport_source`'s events that are not completed. Returns n."""
    from sqlalchemy import text
    return session.execute(text("""
        DELETE FROM rounds ro USING races ra, events e
        WHERE ra.id = ro.race_id AND e.id = ra.event_id AND e.source = :src AND e.status <> 'completed'
          AND ro.extra->>'scheduled' = 'true'
          AND NOT EXISTS (SELECT 1 FROM results r WHERE r.round_id = ro.id)"""), dict(src=sport_source)).rowcount


def run(args, sport, source, plan, echo=print):
    """The `data schedule SPORT` command: plan(session, through, today) writes the rounds; a dry run (no --apply)
    rolls them back. --apply needs --backup, a database dump under 24 hours old (CLAUDE.md: back up before a write)."""
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from racinglines.db import changes
    from racinglines.db.config import get_session

    backup = Path(args.backup) if args.backup else None
    if (args.apply or args.undo) and not (backup and backup.is_file() and backup.stat().st_size
                                          and time.time() - backup.stat().st_mtime < 86400):
        sys.exit("--apply / --undo write rounds: give --backup FILE, the database dump taken for this step within the "
                 f"last 24 hours (data/backups/db/racinglines-before-{sport}-schedule-<UTC>.sql.gz; docs/data-changes.md)")
    today = datetime.now(timezone.utc).date()
    through = date.fromisoformat(args.through)
    with get_session(args.db) as s:
        if args.undo:
            n = undo(s, source)
            changes.record(s, "schedule-undo", f"{sport} scheduled session rounds removed: {n}; backup {backup.name}",
                           sport=sport, detail=dict(backup=backup.name, removed=n))
            s.commit()
            echo(f"Removed {n} scheduled rounds.")
            return 0
        echo(f"{sport} sessions still to run, events dated {today} (less 3 days) to {through}:")
        counts = plan(s, through, today)
        echo("Totals: " + (", ".join(f"{v} {k}" for k, v in counts.items()) or "nothing to write"))
        if not args.apply:
            s.rollback()
            echo("\nDry run: nothing written. Add --apply --backup FILE to write these rounds.")
            return 0
        written = counts.get("added", 0) + counts.get("moved", 0)
        if written:
            changes.record(s, "schedule", f"{sport} scheduled session rounds through {through}: {counts.get('added', 0)} "
                           f"added, {counts.get('moved', 0)} moved; backup {backup.name}", sport=sport,
                           detail=dict(backup=backup.name, through=str(through), **counts))
        s.commit()
        echo(f"\nWrote {written} rounds." if written else "\nNothing to change.")
    return 0


def add_parser(sub, sport):
    p = sub.add_parser("schedule", help="Store the start times of sessions still to run as scheduled rounds (db/schedule.py), "
                                        "so markets on upcoming sessions have a close. A dry run unless --apply.")
    p.add_argument("--through", default=f"{date.today().year}-11-30", help="Last event date to fill (default: 30 Nov).")
    p.add_argument("--apply", action="store_true", help="Write the rounds (default: a dry run that only reports).")
    p.add_argument("--undo", action="store_true", help="Remove the scheduled rounds of events not yet completed.")
    p.add_argument("--backup", default=None, help=f"With --apply/--undo: the database dump taken for this step "
                                                   f"(data/backups/db/racinglines-before-{sport}-schedule-<UTC>.sql.gz).")
    return p
