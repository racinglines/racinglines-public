"""
The data change log (table data_changes): what changed in the race history, when, by whom and why.

    record(session, "ingest", "12 files re-ingested ...", sport="mtb_dh", detail={...})
    racinglines db changes [--limit N] [--sport mtb_dh]      list, newest first
    racinglines db changes --add "why" [--sport mtb_dh]      a manual note (e.g. a re-download's reason)

Ingests that change nothing aren't logged. docs/data-changes.md keeps the story of the major updates.
"""

import getpass
import socket

from sqlalchemy import select

from racinglines.db import models as m


def _by():
    try:
        return f"{getpass.getuser()}@{socket.gethostname()}"[:80]
    except Exception:                                    # pragma: no cover
        return None


def record(session, kind, summary, sport=None, detail=None):
    """Add one row (the caller commits)."""
    row = m.DataChange(kind=kind, summary=summary, sport=sport, detail=detail, by=_by())
    session.add(row)
    return row


def record_run(session, kind, sport, event_keys, run_id, variant=None, sims=None, code_version=None, data_key=None,
               **more):
    """One row for a stored pricing run that moves the board: kind "forecast" (a `forecast --save`) or "live-price"
    (a live reprice by the signal engine). detail: the event keys, run id, model variant, simulations, code_version
    and data_key (plus `more`). The caller commits."""
    keys = list(event_keys or [])
    shown = ", ".join(keys[:3]) + (f" and {len(keys) - 3} more" if len(keys) > 3 else "")
    summary = (f"{sport} {kind} run {run_id}: {shown or 'no event'}"
               + (f", {variant}" if variant else "") + (f", {sims} sims" if sims else ""))
    detail = dict(run=run_id, events=keys, variant=variant, sims=sims, code_version=code_version, data_key=data_key,
                  **more)
    return record(session, kind, summary, sport=sport, detail=detail)


def recent(session, limit=20, sport=None):
    q = select(m.DataChange).order_by(m.DataChange.at.desc(), m.DataChange.id.desc()).limit(limit)
    if sport:
        q = q.where(m.DataChange.sport == sport)
    return session.scalars(q).all()
