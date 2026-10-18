"""
A walk-forward's fair values scored beside an exchange's prices (docs/backtest-core.md): for every event the
model priced, the venue's last price of each linked market at the event's cutoff (the model's own information:
results on earlier days), against the same result.

    racinglines backtest walk-forward f1 --model global --seasons 2025 2026 --venue polymarket kalshi

The cutoff is the event's day (00:00 UTC), which is after qualifying on a race weekend: the market already knows
the grid and the model does not. A sport whose sessions are scheduled (F1) is read instead at a stage of its schema's
`[stages]` (stage_times): by default the first, 1 h before any running, when market and model know the same things.

Prices are read through markets/venue_replay.py, so nothing after the cutoff is seen, a price older than six hours
is no price, and a multi-outcome group whose prices don't sum near its target (an empty or stale book) gives no
price, as in the sweep. Markets pair on exact keys only: market_links.race_id (the event's race), the kind and
athlete_id. A sport whose links carry no race_id has nothing to pair, and the command says so. The score is
scorecard.score's: n, Brier and log loss of the model on every linked market with a fair value and a result;
paired, and model and venue side by side on those the venue also priced.
"""

from datetime import timedelta

import pandas as pd
from sqlalchemy import text

from racinglines.markets import venue_replay as VR
from racinglines.pipelines import scorecard as SC
from racinglines.pipelines import weekend_sweep as WS

KINDS = ("race_win", "race_podium", "race_top10")
VENUES = SC.VENUES
LOOKBACK = timedelta(days=1)          # how far before the cutoff prices are loaded; a price counts for six hours
MID_COLS = ["event_id", "kind", "athlete_id", "token_id", "mid", "coherent", "at"]


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def market_mids(links, cutoff, venue, kinds=KINDS):
    """One row per linked market of `kinds` with an athlete: its price at the cutoff (None when there is none) and
    whether its group was coherent then. `venue`: a venue_replay venue built on `links`."""
    cutoff = _naive(cutoff)
    coherent = {k: venue.coherent(k, cutoff) for k in kinds}
    rows = []
    for link in venue.markets():
        if link["prediction"] not in kinds or pd.isna(link["athlete_id"]):
            continue
        mid = venue.price(link["token_id"], cutoff)
        rows.append(dict(kind=link["prediction"], athlete_id=int(link["athlete_id"]), token_id=link["token_id"],
                         mid=mid if mid is not None and 0 < mid < 1 else None,
                         coherent=coherent[link["prediction"]]))
    return pd.DataFrame(rows, columns=[c for c in MID_COLS if c not in ("event_id", "at")])


def read_mids(conn, cutoffs, venue, kinds=KINDS):
    """market_mids for every event of `cutoffs` ({events.id: cutoff}) that has a race with linked markets on the venue."""
    races = pd.read_sql(text("SELECT id AS race_id, event_id FROM races WHERE event_id = ANY(:e)"), conn,
                        params=dict(e=[int(e) for e in cutoffs]))
    cls = VR.Kalshi if venue == "kalshi" else VR.Polymarket
    cut = {int(e): _naive(c) for e, c in cutoffs.items()}
    out = []
    for r in races.itertuples():
        links = SC.venue_links(conn, int(r.race_id), venue)
        links = links[links["prediction"].isin(kinds) & links["athlete_id"].notna()]
        if not len(links):
            continue
        t = cut[int(r.event_id)]
        ven = cls(conn, links, t - LOOKBACK, t, WS.GROUP_TARGET, WS.COHERENCE_TOL, WS.STALE)
        out.append(market_mids(links, t, ven, kinds).assign(event_id=int(r.event_id), at=t))
    return pd.concat(out, ignore_index=True)[MID_COLS] if out else pd.DataFrame(columns=MID_COLS)


def pair(rows, mids, kinds=KINDS):
    """The walk-forward's scored rows (event_id, kind, athlete_id, fair, y, ...) joined to the venue's markets on
    (event, kind, athlete): every linked market the model priced and the result settled, with its price where there is one."""
    r = rows[rows["kind"].isin(kinds)].dropna(subset=["athlete_id", "fair", "y"]).copy()
    r["event_id"], r["athlete_id"] = r["event_id"].astype(int), r["athlete_id"].astype(int)
    m = mids.assign(event_id=mids["event_id"].astype(int), athlete_id=mids["athlete_id"].astype(int))
    return r.merge(m, on=["event_id", "kind", "athlete_id"], how="inner")


def scores(paired):
    """(per kind, per season and kind): scorecard.score's columns, the model on every paired-or-not linked market
    and model against venue on the markets both priced."""
    rows = paired.assign(open=True)
    return SC.score(rows, by=("kind",)), SC.score(rows, by=("season", "kind"))


def stage_times(sport, event_ids, stage=None, engine_url=None):
    """{events.id: the time of a stage} from the sport's session schedule (sports/<code>.toml [stages]); `stage` is a
    stage label, default the first (before any running). {} for a sport with no schedule (only F1's is read today,
    from weekend_sweep.schedule, keyed by events.source_key 'YYYY-RR'), and no entry for an event the schedule lacks."""
    if sport != "f1" or not len(event_ids):
        return {}
    from racinglines.db.config import get_engine
    with get_engine(engine_url).connect() as conn:
        keys = conn.execute(text("SELECT id, source_key FROM events WHERE id = ANY(:e)"),
                            dict(e=[int(e) for e in event_ids])).fetchall()
    sched, out = {}, {}
    for event_id, key in keys:
        year, rnd = (int(x) for x in str(key).split("-")[:2])
        if year not in sched:
            sched[year] = WS.schedule(year)
        stages = sched[year].get(rnd, {}).get("stages") or []
        hit = stage_time(stages, stage)
        if hit is not None:
            out[int(event_id)] = hit
    return out


def stage_time(stages, stage=None):
    """The time of `stage` in [(label, time)] (the first when None), or None when the weekend has no such stage."""
    hit = stages[0] if stage is None and stages else next((s for s in stages if s[0] == stage), None)
    return hit[1] if hit else None


def compare(out, venue, engine_url=None, kinds=KINDS, at=None):
    """(paired rows, scores per kind, scores per season and kind) for a walk-forward's output on one venue;
    the first is empty when the venue has no market linked to any of its events. at: {event id: time} to read the
    prices at instead of the event's cutoff (stage_times); events it lacks keep their cutoff."""
    from racinglines.db.config import get_engine
    with get_engine(engine_url).connect() as conn:
        mids = read_mids(conn, {**out["cutoffs"], **(at or {})}, venue, kinds)
    if not len(mids):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    paired = pair(out["rows"], mids, kinds)
    return (paired, *scores(paired)) if len(paired) else (paired, pd.DataFrame(), pd.DataFrame())
