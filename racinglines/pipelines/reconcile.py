"""
Per-weekend reconciliation: a profile's live paper trading of one race weekend against the backtest's
replay of the same weekend on the recorded tape (docs/paper-trading.md, validation rule 1).

    racinglines strategy reconcile f1 --event 2026-15 --profile C --venue polymarket
    racinglines strategy reconcile f1 --event 2026-15 --profile A --user taker --replicates 0 --markdown

Live side: the strategy_signals and paper_positions the signal engine (pipelines/signals.py) stored for
the account on that weekend and venue. Replay side: signals.compute(live=False) as of a day after the
race, the same path as the sweep and the demo backfill (markets read at each stage's cutoff, the maker's
conservative "through" fill rule, the profile's own model and settings). Both sides are reduced to the
same three numbers with the same code:

  fills        paper fills (a taker's buys and sells, a maker's fills)
  markout 60m  each fill's move of the mid over the next hour, signed for our side, summed (adverse
               selection when negative), read from the same tape for both sides
  P&L          held to resolution over the settled markets, as the app sums it (a maker's and a follower's
               from the positions; a taker's own calls from the fills and the outcomes)

The rules (pre-registered, docs/paper-trading.md#validation-plan): live fills and markouts within
+-25% of the replay's, live P&L inside the replay's noise band. The band is built from seed replicates
(docs/backtest-core.md, "noise floor"): the weekend replayed with the profile's model at REPLICATE_SEEDS
besides the default seed (each replicate reprices the stages, cached as diagnostic runs exactly as a sweep
caches them), and the band is the replicates' P&L range, at least +-NOISE_FLOOR wide around their mean
(the search's configured season floors, 150 taker / 350 maker, scaled to one weekend of a 24-round season).
A weekend whose live rows are a backfilled replay (detail.backfill) is a self-check: replay against replay
must match exactly.

Read-only on the accounts' records: nothing here writes a signal or a position. --no-price makes the
command read-only on the database altogether (the band then rests on the cached seeds only, and says so).
"""

import math
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.pipelines import demo_history as DH
from racinglines.pipelines import profiles as PF
from racinglines.pipelines import signals as SG
from racinglines.pipelines import sweep_settings as SS
from racinglines.pipelines import weekend_sweep as WS

PCT = 0.25                                  # fills and markouts: +-25% of the replay's
MARKOUT_ABS = 1.0                           # $: a markout difference under this is never a miss
EXACT_ABS = 0.005                           # $: "exactly" for the backfill self-check
NOISE_FLOOR = {"taker": 150.0 / math.sqrt(24), "maker": 350.0 / math.sqrt(24)}   # $ per weekend
REPLICATE_SEEDS = (43, 44)                  # besides the profile's own seed (sweep_settings.DEFAULT_SEED)
HORIZON = timedelta(minutes=60)
TAKER_FILLS = ("buy", "sell")


# ---------------------------------------------------------------------------
# Pure pieces (tested without a database)
# ---------------------------------------------------------------------------

def family(strategy):
    return "taker" if strategy in WS.TAKER_MODES else "maker"


def seeds(n):
    """The first n replicate seeds: REPLICATE_SEEDS, then the integers after them."""
    s = list(REPLICATE_SEEDS)
    while len(s) < n:
        s.append(s[-1] + 1)
    return tuple(s[:max(n, 0)])


def _ts(v):
    """Naive UTC Timestamp from a tz-aware or naive one (or NaT)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    t = pd.Timestamp(v)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _f(v):
    return float("nan") if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def fills_of(signals, strategy):
    """The paper fills among a list of signal rows (strategy_signals rows or signals.compute's), one
    shape for both: DataFrame(key, stage, kind, subject, ts, side_yes, entry, sign, qty, mid, price, followed).
    side_yes: the side held (a maker's YES buy or a taker's YES trade); entry: a buy (a maker's fills always
    are; a taker's sells are exits); sign: +1 for long YES exposure (a YES buy, a maker buying YES), -1 for
    short (a NO buy, a YES sell, a maker selling YES); mid: the market's mid at the fill; price: what the
    fill paid per share of its side."""
    rows = []
    for s in signals:
        act, side = s.get("action"), s.get("side")
        if strategy in WS.TAKER_MODES:
            if act not in TAKER_FILLS:
                continue
        elif act != "fill":
            continue
        side_yes, entry = side == "YES", act != "sell"
        sign = (1.0 if side_yes else -1.0) * (1.0 if entry else -1.0)
        det = s.get("detail") or {}
        rows.append(dict(key=s["market_key"], stage=s.get("stage"), kind=s.get("kind"), subject=s.get("subject"),
                         ts=_ts(s.get("signal_ts")), side_yes=side_yes, entry=entry, sign=sign,
                         qty=_f(s.get("shares")), mid=_f(s.get("price")), price=_f(s.get("limit_price")),
                         followed=bool(s.get("followed", det.get("followed", True)))))
    cols = ["key", "stage", "kind", "subject", "ts", "side_yes", "entry", "sign", "qty", "mid", "price", "followed"]
    return pd.DataFrame(rows, columns=cols)


def as_stored(signals):
    """A replay's signals as signals.store keeps them: one row per (market, dedupe, action, side), the first
    wins (its ON CONFLICT DO NOTHING). Maker fills sharing a market, side and timestamp (common on Kalshi's
    tape, whose trades are timed to the second) are stored as one. -> (signals, n_collapsed)."""
    seen, out = set(), []
    for s in signals:
        k = (s["market_key"], s.get("dedupe"), s.get("action"), s.get("side"))
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out, len(signals) - len(out)


def markouts(fills, mid_at, horizon=HORIZON):
    """fills with a markout_60m column: sign x (mid `horizon` after the fill - mid at the fill) x qty, NaN
    where either mid is unknown. mid_at(key, ts) -> the market's last mid at or before ts, or None."""
    out = []
    for r in fills.itertuples():
        later = None if pd.isna(r.ts) else mid_at(r.key, r.ts + horizon)
        mid = r.mid if not np.isnan(r.mid) else (mid_at(r.key, r.ts) if not pd.isna(r.ts) else None)
        out.append(float("nan") if later is None or mid is None or np.isnan(mid) else r.sign * (later - mid) * r.qty)
    return fills.assign(markout_60m=pd.Series(out, index=fills.index, dtype=float))


def positions_pnl(positions):
    """(pnl, n_unsettled) of position rows (paper_positions or signals.compute's): cash plus the shares'
    payout, over settled markets."""
    pnl, missing = 0.0, 0
    for p in positions:
        o = p.get("outcome")
        if o is None or (isinstance(o, float) and np.isnan(o)):
            missing += 1
            continue
        y = 1.0 if o else 0.0
        pnl += _f(p.get("cash")) + _f(p.get("yes_shares")) * y + _f(p.get("no_shares")) * (1 - y)
    return pnl, missing


def summarize(fills, pnl):
    """The three numbers (plus notional) of one side."""
    return dict(fills=int(len(fills)), notional=float((fills["qty"] * fills["price"]).sum()) if len(fills) else 0.0,
                markout_60m=float(fills["markout_60m"].sum()) if len(fills) else 0.0,
                markouts_known=int(fills["markout_60m"].notna().sum()) if len(fills) else 0,
                pnl=None if pnl is None else float(pnl))


def noise_band(pnls, fam):
    """[lo, hi] from the replicates' P&L: their range, at least +-NOISE_FLOOR[fam] around their mean."""
    ps = [float(p) for p in pnls if p is not None]
    if not ps:
        return None
    m, floor = float(np.mean(ps)), NOISE_FLOOR[fam]
    return [min(min(ps), m - floor), max(max(ps), m + floor)]


def within_pct(live, replay, pct=PCT, abs_floor=0.0):
    """|live - replay| <= max(pct x |replay|, abs_floor)."""
    return abs(live - replay) <= max(pct * abs(replay), abs_floor) + 1e-9


def compare(live, replay, band, fam, exact=False):
    """Judge one pair of summaries. -> dict(rows=[dict(measure, live, replay, diff, rule, ok)], ok, flags)
    exact: the backfill self-check (everything must match to the cent)."""
    rows = []

    def row(measure, lv, rv, rule, ok, fmt):
        rows.append(dict(measure=measure, live=lv, replay=rv, diff=None if lv is None or rv is None else lv - rv,
                         rule=rule, ok=ok, fmt=fmt))

    lf, rf = live["fills"], replay["fills"]
    ok_f = lf == rf if exact else within_pct(lf, rf)
    row("fills", lf, rf, "exact" if exact else "+-25%", ok_f, "d")
    row("notional $", live["notional"], replay["notional"], "shown", None, "f")
    lm, rm = live["markout_60m"], replay["markout_60m"]
    ok_m = abs(lm - rm) <= EXACT_ABS if exact else within_pct(lm, rm, abs_floor=MARKOUT_ABS)
    row("markout 60m $", lm, rm, "exact" if exact else "+-25% (or $1)", ok_m, "f")
    lp, rp = live["pnl"], replay["pnl"]
    if lp is None or rp is None:
        row("P&L $", lp, rp, "unsettled", None, "f")
    elif exact:
        row("P&L $", lp, rp, "exact", abs(lp - rp) <= EXACT_ABS, "f")
    elif band is None:
        row("P&L $", lp, rp, "no band (no replicates)", None, "f")
    else:
        row("P&L $", lp, rp, f"band [{band[0]:+.2f}, {band[1]:+.2f}]", band[0] - 1e-9 <= lp <= band[1] + 1e-9, "f")
    flags = [r["measure"].split(" ")[0] for r in rows if r["ok"] is False]
    return dict(rows=rows, ok=not flags, flags=flags, family=fam)


# ---------------------------------------------------------------------------
# The live rows and the replay
# ---------------------------------------------------------------------------

def live_rows(conn, user_id, candidate_id, event_key, venue):
    """The account's stored signals and positions for the weekend on the venue. -> dict(signals=[rows],
    positions=[rows], backfill: every signal flagged detail.backfill, stored: last created_at)."""
    sig = conn.execute(text("""SELECT * FROM strategy_signals WHERE user_id = :u AND candidate_id IS NOT DISTINCT FROM :c
                               AND event_key = :e AND coalesce(detail->>'venue', 'polymarket') = :v
                               ORDER BY signal_ts, id"""),
                       dict(u=user_id, c=candidate_id, e=event_key, v=venue)).mappings().all()
    pos = conn.execute(text("""SELECT * FROM paper_positions WHERE user_id = :u AND candidate_id IS NOT DISTINCT FROM :c
                               AND event_key = :e AND venue = :v ORDER BY id"""),
                       dict(u=user_id, c=candidate_id, e=event_key, v=venue)).mappings().all()
    sig, pos = [dict(r) for r in sig], [dict(r) for r in pos]
    flags = [(r.get("detail") or {}).get("backfill") is True for r in sig]
    return dict(signals=sig, positions=pos, backfill=bool(sig) and all(flags), mixed=any(flags) and not all(flags),
                stored=max((r["created_at"] for r in sig), default=None))


def _mid_resolver(conn, fills_keys, strategy, venue, run_ids, start, end):
    """mid_at(key, ts) over the recorded tape: a taker's markets by token (the Polymarket price store), a
    maker's by condition through maker_replay.load_event (the mids its replay used)."""
    from racinglines.markets import store as MS
    from racinglines.markets.strategies import maker_replay as R
    series = {}
    if strategy in WS.TAKER_MODES:
        keys = sorted(set(fills_keys))
        px = MS.read(conn, "prices", tokens=keys, start=pd.Timestamp(start).tz_localize("UTC"),
                     end=pd.Timestamp(end).tz_localize("UTC")) if keys else pd.DataFrame(columns=["token_id", "ts", "price"])
        for tok, g in px.groupby("token_id"):
            g = g.sort_values("ts")
            series[tok] = (pd.to_datetime(g["ts"], utc=True).dt.tz_localize(None).to_numpy(), g["price"].to_numpy(float))
    else:
        ev = R.load_event(conn, run_ids, **({} if venue == "polymarket" else dict(exchange=venue)))
        for mk in ev["markets"]:
            series[mk.cond] = (mk.mid_ts.astype("datetime64[ns]"), mk.mid_px)

    def mid_at(key, ts):
        s = series.get(key)
        if s is None:
            return None
        i = np.searchsorted(s[0], np.datetime64(pd.Timestamp(ts), "ns"), side="right") - 1
        return float(s[1][i]) if i >= 0 else None
    return mid_at


def _side(fills, strategy, outcomes=None, positions=None, followed=None):
    """One side's summary from its fills (with markouts) and, for the P&L, either the fills and outcomes
    (a taker's own calls) or the positions."""
    f = fills if followed is None else fills[fills["followed"] == followed]
    pnl, missing = positions_pnl(positions) if positions is not None else calls_pnl(f, outcomes)
    n = len(positions) if positions is not None else len(f)
    # over the settled markets, as demo_history and the app sum it; unsettled only when nothing has settled
    return dict(summarize(f, None if n and missing == n else pnl), unsettled=missing)


def calls_pnl(fills, outcomes):
    """A taker's own calls to resolution, taker_weekend's rule per fill: shares x (the side's payout - price),
    shares negative on an exit. outcomes: {market key: True / False / None}. -> (pnl, n_unsettled)."""
    pnl, missing = 0.0, 0
    for r in fills.itertuples():
        o = outcomes.get(r.key)
        if o is None:
            missing += 1
            continue
        y = 1.0 if o else 0.0
        v = y if r.side_yes else 1.0 - y
        pnl += (v - r.price) * r.qty * (1.0 if r.entry else -1.0)
    return pnl, missing


def _stage_table(live_f, rep_f, order):
    """Per stage (in the weekend's order): fills and markouts on both sides."""
    stages = [st for st in order if st in set(rep_f["stage"]) | set(live_f["stage"])]
    stages += [st for st in dict.fromkeys([*rep_f["stage"], *live_f["stage"]]) if st not in stages]
    rows = []
    for st in stages:
        a, b = live_f[live_f["stage"] == st], rep_f[rep_f["stage"] == st]
        rows.append(dict(stage=st, live_fills=len(a), replay_fills=len(b),
                         live_markout=float(a["markout_60m"].sum()), replay_markout=float(b["markout_60m"].sum())))
    return rows


def reconcile(engine, engine_url, event, profile, venue="polymarket", user=None, replicates=REPLICATE_SEEDS,
              price=True, now=None, echo=print):
    """Compare one account's live paper weekend with its replay. -> dict(event, profile, user, venue, live,
    replay, band, replicates=[(seed, pnl)], verdict=compare(), stages, notes, exact)."""
    now_ = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    with engine.connect() as c:
        prof = PF.load(c, profile)
        code = next((k for k, v in {**PF.PROFILES, **PF.HISTORY_PROFILES}.items() if v["name"] == prof["name"]), None)
        username = user or next((u for u, k in PF.DEMO.items() if k == code), None) or \
            next((u for u, hs in PF.HISTORY.items() if any(k == code for k, *_ in hs)), None)
        if username is None:
            raise ValueError(f"profile {prof['name']!r} is nobody's by default: pass --user")
        uid = c.execute(text("SELECT id FROM users WHERE username = :u"), dict(u=username)).scalar()
        if uid is None:
            raise ValueError(f"no user {username!r}")
    strategy, fam = prof["strategy"], family(prof["strategy"])
    if fam == "taker" and venue != "polymarket":
        raise ValueError("taker profiles trade Polymarket only")
    year = int(str(event).split("-")[0]) if "-" in str(event) else now_.year
    rnd = int(str(event).split("-")[-1])
    w = WS.schedule(year)[rnd]
    settled = w["race_start"] + DH.SETTLED
    asof = min(settled, now_)
    notes = []
    if asof < settled:
        notes.append(f"the weekend isn't settled yet: replayed as of {asof:%Y-%m-%d %H:%M} UTC")
    with engine.connect() as c:
        live = live_rows(c, uid, prof["candidate_id"], w["event_key"], venue)
    if not live["signals"] and not live["positions"]:
        notes.append("no live rows: the signal engine stored nothing for this account, weekend and venue")
    if live["mixed"]:
        notes.append("live rows mix backfilled and live signals")
    exact = live["backfill"]
    if exact:
        notes.append(f"the live rows are a backfilled replay (stored {live['stored']:%Y-%m-%d %H:%M} UTC): "
                     "self-check, replay against replay must match exactly")
    rate = PF.follow_rate(username, prof["strategy"])
    cache = {}
    other = {} if venue == "polymarket" else dict(venue=venue)

    def replay(seed):
        p = prof if seed is None else dict(prof, settings=dict(prof["settings"], seed=seed))
        if seed is not None:               # the trained history (pricing.history) doesn't depend on the seed
            k0 = ("hist", SS.Settings.from_dict(prof["settings"], strict=False).model_key)
            k1 = ("hist", SS.Settings.from_dict(p["settings"], strict=False).model_key)
            if k0 in cache and k1 not in cache:
                cache[k1] = cache[k0]
        return SG.compute(engine, engine_url, p, now=asof, event=w["event_key"], live=False, fetch=False,
                          echo=lambda m: echo(f"  {m}"), cache=cache, **other)

    echo(f"replaying {w['event_key']} {w['name']} for {prof['name']} on {venue} (seed {SS.DEFAULT_SEED})")
    base = replay(None)
    if not base["stages"]:
        raise ValueError(f"{w['name']}: {base.get('note') or 'no stage priced'}")
    run_ids = [rid for _, _, rid, _ in base["stages"]]
    rep_sigs, rep_pos = base["signals"], base["positions"]
    rep_sigs, collapsed = as_stored(rep_sigs)
    if collapsed:
        notes.append(f"{collapsed} of the replay's fills share a market, side and timestamp with another and are "
                     "stored as one by signals.store: compared as stored (the P&L, from the positions, is unaffected)")
    if fam == "taker":
        rep_sigs, rep_pos_followed = SG.apply_follow(rep_sigs, rep_pos, uid, rate)
    start, end = base["stages"][0][1] - timedelta(hours=1), asof + HORIZON
    live_f, rep_f = fills_of(live["signals"], strategy), fills_of(rep_sigs, strategy)
    with engine.connect() as c:
        mid_at = _mid_resolver(c, [*live_f["key"], *rep_f["key"]], strategy, venue, run_ids, start, end)
    live_f, rep_f = markouts(live_f, mid_at), markouts(rep_f, mid_at)
    outcomes = {p["market_key"]: p.get("outcome") for p in [*rep_pos, *live["positions"]]}
    sides = {}
    if fam == "taker":
        sides["own calls"] = (_side(live_f, strategy, outcomes=outcomes), _side(rep_f, strategy, outcomes=outcomes))
        sides["followed"] = (_side(live_f, strategy, positions=live["positions"], followed=True),
                             _side(rep_f, strategy, positions=rep_pos_followed, followed=True))
    else:
        sides["maker"] = (_side(live_f, strategy, positions=live["positions"]),
                          _side(rep_f, strategy, positions=rep_pos))
    # seed replicates -> the noise band of the judged P&L
    judged = "own calls" if fam == "taker" else "maker"
    reps = [(SS.DEFAULT_SEED, sides[judged][1]["pnl"])]
    for seed in replicates or ():
        if not price and not _cached(engine, prof, seed, w):
            notes.append(f"seed {seed}: stage pricings not cached (run without --no-price to price them)")
            continue
        echo(f"replicate seed {seed}")
        r = replay(seed)
        s, _ = as_stored(r["signals"])
        if fam == "taker":
            s, _ = SG.apply_follow(s, r["positions"], uid, rate)
            f = markouts(fills_of(s, strategy), mid_at)
            o = dict(outcomes, **{p["market_key"]: p.get("outcome") for p in r["positions"]})
            reps.append((seed, _side(f, strategy, outcomes=o)["pnl"]))
        else:
            reps.append((seed, _side(markouts(fills_of(s, strategy), mid_at), strategy, positions=r["positions"])["pnl"]))
    band = noise_band([p for _, p in reps], fam) if len(reps) >= 2 else None
    if band is None and not exact:
        notes.append("no noise band: fewer than two seed replicates (the P&L rule can't be judged)")
    verdicts = {name: compare(lv, rv, band if name == judged else None, fam, exact=exact) for name, (lv, rv) in sides.items()}
    verdict = verdicts[judged]
    if fam == "taker" and not exact:       # the followed third is shown, not judged (docs: judged on A's own calls)
        for r in verdicts["followed"]["rows"]:
            r["ok"], r["rule"] = None, "shown"
        verdicts["followed"]["ok"], verdicts["followed"]["flags"] = True, []
    elif fam == "taker":
        verdict = dict(verdict, ok=verdict["ok"] and verdicts["followed"]["ok"],
                       flags=sorted(set(verdict["flags"]) | {f"followed {x}" for x in verdicts["followed"]["flags"]}))
    return dict(event=w, profile=prof, user=username, venue=venue, live=live, sides=sides, verdicts=verdicts,
                judged=judged, verdict=verdict, band=band, replicates=reps, exact=exact, notes=notes,
                stages=_stage_table(live_f, rep_f, [lab for lab, *_ in base["stages"]]), asof=asof, stage_runs=base["stages"])


def _cached(engine, prof, seed, w):
    st = SS.Settings.from_dict(dict(prof["settings"], seed=seed), strict=False)
    with engine.connect() as c:
        n = c.execute(text("""SELECT count(DISTINCT params->>'cutoff') FROM model_runs WHERE kind = 'diagnostic'
                              AND params ? 'sweep_stage' AND params->>'model_key' = :m AND params->>'event_key' = :k"""),
                      dict(m=st.model_key, k=w["event_key"])).scalar()
    return n >= len(w["stages"])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(v, fmt):
    if v is None:
        return "-"
    return f"{v:d}" if fmt == "d" else f"{v:+.2f}"


def _head(res):
    w, p = res["event"], res["profile"]
    return f"{w['event_key']} {w['name']} · {p['name']} · {res['user']} on {res['venue']}"


def _seeds(res):
    return ", ".join(f"{s}: {_fmt(p, 'f')}" for s, p in res["replicates"])


def format_table(res):
    lines = [_head(res)]
    for n in res["notes"]:
        lines.append(f"  note: {n}")
    lines.append(f"  replay as of {res['asof']:%Y-%m-%d %H:%M} UTC, stages "
                 + ", ".join(f"{lab} #{rid}" for lab, _, rid, _ in res["stage_runs"]))
    lines.append(f"  P&L by seed: {_seeds(res)}" + (f" -> band [{res['band'][0]:+.2f}, {res['band'][1]:+.2f}]"
                                                    if res["band"] else ""))
    for name, v in res["verdicts"].items():
        lines.append(f"  {name}{' (judged)' if name == res['judged'] else ''}")
        lines.append(f"    {'measure':14s} {'live':>10s} {'replay':>10s} {'diff':>9s}  {'rule':28s} verdict")
        for r in v["rows"]:
            ok = "-" if r["ok"] is None else ("ok" if r["ok"] else "MISS")
            lines.append(f"    {r['measure']:14s} {_fmt(r['live'], r['fmt']):>10s} {_fmt(r['replay'], r['fmt']):>10s} "
                         f"{_fmt(r['diff'], r['fmt']):>9s}  {r['rule']:28s} {ok}")
    for name, (lv, rv) in res["sides"].items():
        if lv["unsettled"] or rv["unsettled"]:
            lines.append(f"  {name}: {lv['unsettled']} live / {rv['unsettled']} replay markets without an outcome, left out of the P&L")
    if res["stages"]:
        lines.append("  fills by stage (live / replay, markout 60m live / replay)")
        for s in res["stages"]:
            lines.append(f"    {s['stage']:13s} {s['live_fills']:4d} / {s['replay_fills']:<4d} "
                         f"{s['live_markout']:+8.2f} / {s['replay_markout']:+8.2f}")
    v = res["verdict"]
    lines.append("  verdict: " + ("within the rules" if v["ok"] else "FLAGGED: " + ", ".join(v["flags"]))
                 + (" (self-check)" if res["exact"] else ""))
    return "\n".join(lines)


def format_markdown(res):
    """A block for the weekend's report (docs/f1-live-roadmap.md, after the weekend)."""
    v = res["verdict"]
    out = [f"### Reconciliation · {_head(res)}", ""]
    out.append(("**Within the rules**" if v["ok"] else "**Flagged:** " + ", ".join(v["flags"]))
               + (" (backfilled weekend: self-check, exact match required)." if res["exact"] else
                  " (live fills and markouts within ±25% of the replay's; live P&L inside the replay's noise band)."))
    out.append("")
    for name, vv in res["verdicts"].items():
        out.append(f"**{name}**" + (" (judged)" if name == res["judged"] else ""))
        out.append("")
        out.append("| Measure | Live | Replay | Diff | Rule | Verdict |")
        out.append("|---|---:|---:|---:|---|---|")
        for r in vv["rows"]:
            ok = "–" if r["ok"] is None else ("ok" if r["ok"] else "**miss**")
            out.append(f"| {r['measure']} | {_fmt(r['live'], r['fmt'])} | {_fmt(r['replay'], r['fmt'])} | "
                       f"{_fmt(r['diff'], r['fmt'])} | {r['rule']} | {ok} |")
        out.append("")
    if res["stages"]:
        out.append("| Stage | Fills live / replay | Markout 60m live / replay |")
        out.append("|---|---:|---:|")
        for s in res["stages"]:
            out.append(f"| {s['stage']} | {s['live_fills']} / {s['replay_fills']} | "
                       f"{s['live_markout']:+.2f} / {s['replay_markout']:+.2f} |")
        out.append("")
    out.append(f"Replay as of {res['asof']:%Y-%m-%d %H:%M} UTC on the recorded tape (through fills). P&L by seed: "
               f"{_seeds(res)}" + (f"; noise band [{res['band'][0]:+.2f}, {res['band'][1]:+.2f}]." if res["band"] else "."))
    for name, (lv, rv) in res["sides"].items():
        if lv["unsettled"] or rv["unsettled"]:
            out.append(f"Note: {name}: {lv['unsettled']} live / {rv['unsettled']} replay markets without an outcome, "
                       "left out of the P&L.")
    for n in res["notes"]:
        out.append(f"Note: {n}.")
    return "\n".join(out)
