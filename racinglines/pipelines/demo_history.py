"""
The demo accounts' track record: every past weekend with Polymarket race markets, replayed with the
profile each account ran then (profiles.HISTORY), stored as that account's paper signals and positions.

These are backtest replays, not live paper trades: real Polymarket prices and trade tapes, our as-of
pricing, the backtest's strategy code (signals.compute with live=False, the same path as the sweep).
Every row is flagged detail.backfill and shown as a backtest replay in the app. Live paper signals
(signals.run_all) take over from the next race weekend.

    racinglines strategy demo-history f1              # backfill (idempotent: existing rows are kept)
    racinglines strategy demo-history f1 --reset      # delete the backfill first, then rebuild it
    racinglines strategy demo-history f1 --venue kalshi [--reset]   # the maker's record on Kalshi instead (off by default)

--venue kalshi replays the maker's same profiles against Kalshi's recorded tape (markets/kalshi/ stores it in
the shared tables), with Kalshi's maker fee, as paper_positions.venue = 'kalshi' and signals flagged
detail.venue = 'kalshi'. The Polymarket record and its reset don't touch those rows, and vice versa. Maker
accounts only (a taker's Kalshi paper trading is a profile with the `venue = kalshi` setting, signals.py).
"""

import pandas as pd
from sqlalchemy import text

from racinglines import progress as PG
from racinglines.pipelines import profiles as PF
from racinglines.pipelines import signals as SG
from racinglines.pipelines import weekend_sweep as WS

SETTLED = pd.Timedelta(days=1)       # replay each weekend as of a day after its race: every stage known, settled


def reset(engine, usernames, venue="polymarket"):
    """Delete the accounts' backfilled signals for `venue` and their positions on that venue only (a Polymarket
    reset leaves the private book's, Kalshi's and OG.com's rows for the same events alone)."""
    sig = "coalesce(detail->>'venue', 'polymarket') = :v"
    pos = "venue = :v"
    with engine.begin() as c:
        ids = c.execute(text("SELECT id FROM users WHERE username = ANY(:u)"), dict(u=list(usernames))).scalars().all()
        keys = c.execute(text(f"""SELECT DISTINCT event_key FROM strategy_signals WHERE user_id = ANY(:i)
                                 AND detail->>'backfill' = 'true' AND {sig}"""), dict(i=ids, v=venue)).scalars().all()
        n = c.execute(text(f"DELETE FROM strategy_signals WHERE user_id = ANY(:i) AND detail->>'backfill' = 'true' AND {sig}"),
                      dict(i=ids, v=venue)).rowcount
        c.execute(text(f"DELETE FROM paper_positions WHERE user_id = ANY(:i) AND event_key = ANY(:k) AND {pos}"),
                  dict(i=ids, k=list(keys), v=venue))
    return n


def backfill(engine, engine_url=None, usernames=None, now=None, echo=print, venue="polymarket"):
    """Replay every past weekend of each demo account's history. Returns [(user, profile, event, signals, P&L)].
    venue="kalshi": the maker accounts' record on Kalshi (taker accounts are skipped)."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    plan = {u: h for u, h in PF.HISTORY.items() if not usernames or u in usernames}
    other = {} if venue == "polymarket" else dict(venue=venue)
    if other:
        plan = {u: h for u, h in plan.items() if all(PF.HISTORY_PROFILES.get(c, PF.PROFILES.get(c)).get("strategy") == "maker"
                                                     for c, *_ in h)}
    with engine.begin() as c:
        ids = PF.ensure_candidates(c, history=True)
        users = dict(c.execute(text("SELECT username, id FROM users WHERE username = ANY(:u)"),
                               dict(u=list(plan))).all())
        profs = {code: PF.load(c, i) for code, i in ids.items()}
        profs.update({code: PF.load(c, code) for code, pr in PF.PROFILES.items() if PF.is_combo(pr)})   # blends
    cache, report, scheds = {}, [], {}
    for username, phases in plan.items():
        uid = users.get(username)
        if uid is None:
            echo(f"no user {username!r}: skipped")
            continue
        for code, year, r0, r1 in phases:
            prof = profs[code]
            rate = PF.follow_rate(username, prof["strategy"])
            sched = scheds.setdefault(year, WS.schedule(year))
            for rnd in PG.track(sorted(r for r in sched if r0 <= r <= r1), "round", name=lambda r: f"{username} {code} {year}"):
                w = sched[rnd]
                if w["race_start"] + SETTLED > now:
                    break                                     # not raced yet: live signals cover it
                outs = SG.compute_all(engine, engine_url, prof, now=w["race_start"] + SETTLED, event=f"{year}-{rnd}",
                                      live=False, fetch=False, echo=lambda m: None, cache=cache, **other)
                outs = [o for o in outs if o["stages"] and (o["signals"] or o["positions"])]
                if not outs:
                    echo(f"{username} {w['event_key']} {w['name']}: no markets")
                    continue
                out = dict(outs[0], signals=[s for o in outs for s in o["signals"]])     # for the report line
                with engine.begin() as c:
                    for o in outs:                                # a blend: each member's rows (compute_all)
                        SG.store(c, uid, o, follow_rate=rate, history=True, **other)
                    pnl = c.execute(text("""SELECT coalesce(sum(cash + yes_shares * outcome::int + no_shares * (1 - outcome::int)), 0)
                                            FROM paper_positions WHERE user_id = :u AND event_key = :e AND outcome IS NOT NULL
                                            AND venue = :v"""),
                                    dict(u=uid, e=w["event_key"], v=venue)).scalar()
                report.append((username, prof["name"], w["event_key"], len(out["signals"]), float(pnl)))
                echo(f"{username} {w['event_key']} {w['name']:28s} {code:2s} {len(out['signals']):4d} signals  "
                     f"P&L {pnl:+8.2f}")
    return report
