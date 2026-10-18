"""
Single-event diagnostic: our prices for one past race as of a cutoff (from a
kind='diagnostic' model run) vs Polymarket's prices at that same cutoff, the
24 hours before it, and the actual result.

Order of operations, to keep it honest:
  1. Our prices come from the stored diagnostic run (priced with data before the cutoff).
  2. Polymarket prices are read from stored price history at/before the cutoff.
  3. Any decisions (paper trades, the example taker's bets) use only 1 and 2.
  4. Only then is the official result read, to score and settle.
"""

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.db import reads as data
from racinglines.markets import private_book as house

RACE_KINDS = ("race_win", "race_podium", "race_h2h", "race_constructor_top")
PAPER_EDGE = 0.05       # paper strategy: trade Polymarket when our fair differs by >= 5 pts
PAPER_STAKE = 10.0
COHERENCE_TOL = 0.15    # multi-outcome Polymarket prices must sum to within 15% of the winner count


def _prices_at(conn, tokens, ts):
    from racinglines.markets import store as MS
    return MS.last_before(conn, list(tokens), ts)


def load(conn, run_id):
    run = data.q(conn, "SELECT * FROM model_runs WHERE id = :i AND kind = 'diagnostic'", i=run_id)
    if not len(run):
        return None
    run = run.iloc[0].to_dict()
    params, metrics = run["params"] or {}, run["metrics"] or {}
    cutoff = pd.Timestamp(params["cutoff"]).tz_localize("UTC")
    race_id = data.q(conn, "SELECT DISTINCT race_id FROM race_predictions WHERE model_run_id = :i", i=run_id)["race_id"].iloc[0]
    race_id = int(race_id)
    ev = data.q(conn, """SELECT e.id, e.name, e.source_key, v.name AS venue FROM races ra JOIN events e ON e.id = ra.event_id
                         LEFT JOIN venues v ON v.id = e.venue_id WHERE ra.id = :r""", r=race_id).iloc[0].to_dict()

    # 1. our prices (as of the cutoff)
    preds = data.q(conn, """
        SELECT rp.athlete_id, a.display_name AS driver, rp.win_prob, rp.podium_prob, rp.top10_prob, rp.extra
        FROM race_predictions rp JOIN athletes a ON a.id = rp.athlete_id WHERE rp.model_run_id = :i
        ORDER BY rp.win_prob DESC""", i=run_id)
    preds["team"] = preds["extra"].map(lambda x: (x or {}).get("team_key"))
    preds["grid"] = None

    # 2. Polymarket at the cutoff, 24 h before, and latest
    links = data.q(conn, """SELECT ml.*, a.display_name AS athlete FROM market_links ml
                            LEFT JOIN athletes a ON a.id = ml.athlete_id
                            WHERE ml.race_id = :r AND ml.prediction = ANY(:k) AND ml.exchange = 'polymarket'""",
                     r=race_id, k=list(RACE_KINDS))
    toks = links["token_id"].tolist()
    at_cut = _prices_at(conn, toks, cutoff.to_pydatetime())
    at_24 = _prices_at(conn, toks, (cutoff - pd.Timedelta(hours=24)).to_pydatetime())
    latest = _prices_at(conn, toks, pd.Timestamp.now(tz="UTC").to_pydatetime())
    cache = {}
    rows = []
    for link in links.to_dict("records"):
        fair, _ = data.model_prob(conn, link, cache, run_id=run_id)
        p_cut, p_24 = at_cut.get(link["token_id"]), at_24.get(link["token_id"])
        rows.append(dict(link_id=link["id"], kind=link["prediction"], subject=link["athlete"] or (link["params"] or {}).get("team"),
                         opponent_id=(link["params"] or {}).get("opponent_id"), outcome_label=link["outcome"],
                         question=link["question"], token_id=link["token_id"], athlete_id=link["athlete_id"],
                         params=link["params"], fair=fair, pm_cut=p_cut, pm_24h_before=p_24,
                         pm_move_24h=(p_cut - p_24) if p_cut is not None and p_24 is not None else None,
                         pm_latest=latest.get(link["token_id"]),
                         edge=(fair - p_cut) if fair is not None and p_cut is not None else None))
    mk = pd.DataFrame(rows)

    # Polymarket coherence at the cutoff: a multi-outcome group whose prices don't
    # sum to roughly its number of winners (1 for win / constructor, 3 for podium)
    # is an empty or stale book; its prices are shown but not used as a benchmark.
    coherence = {}
    for kind, target in (("race_win", 1), ("race_podium", 3), ("race_constructor_top", 1)):
        g = mk[mk["kind"] == kind] if len(mk) else mk
        if len(g) and g["pm_cut"].notna().any():
            tot = float(g["pm_cut"].fillna(0).sum())
            ok = abs(tot - target) <= COHERENCE_TOL * target
            coherence[kind] = dict(sum=tot, target=target, ok=ok)
            if not ok:
                mk.loc[g.index, "edge"] = None
    if len(mk):
        mk["pm_usable"] = ~mk["kind"].map(lambda k: not coherence.get(k, {}).get("ok", True))

    # 3. paper strategy on Polymarket (decided with fair + price at cutoff only)
    def paper(r):
        if r["edge"] is None or pd.isna(r["edge"]) or not r["pm_usable"] or r["pm_cut"] in (None, 0, 1):
            return None, None
        if r["edge"] >= PAPER_EDGE:
            return "YES", r["pm_cut"]
        if r["edge"] <= -PAPER_EDGE:
            return "NO", 1 - r["pm_cut"]
        return None, None
    if len(mk):
        mk[["paper_side", "paper_price"]] = mk.apply(lambda r: pd.Series(paper(r)), axis=1)

    # 4. outcome (read last) and scoring
    res = house.race_outcomes(conn, race_id)
    if len(mk):
        ath = mk["athlete_id"].map(lambda v: None if pd.isna(v) else int(v))
        mk["result"] = [house.outcome_for(k, a, p, res) for k, a, p in zip(mk["kind"], ath, mk["params"])]
        pnl = []
        for r in mk.itertuples():
            if r.paper_side is None or r.result is None:
                pnl.append(None)
                continue
            won = (r.paper_side == "YES") == bool(r.result)
            pnl.append(PAPER_STAKE / r.paper_price - PAPER_STAKE if won else -PAPER_STAKE)
        mk["paper_pnl"] = pnl
    scores = []
    for kind, g in (mk.groupby("kind") if len(mk) else []):
        g = g[g["pm_usable"]].dropna(subset=["fair", "pm_cut", "result"])
        if not len(g):
            continue
        y = g["result"].astype(float)
        f, p = g["fair"].clip(1e-3, 1 - 1e-3), g["pm_cut"].astype(float).clip(1e-3, 1 - 1e-3)
        scores.append(dict(kind=kind, markets=len(g),
                           brier_model=float(((f - y) ** 2).mean()), brier_polymarket=float(((p - y) ** 2).mean()),
                           logloss_model=float(-(y * np.log(f) + (1 - y) * np.log(1 - f)).mean()),
                           logloss_polymarket=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())))
    result = res.merge(data.q(conn, "SELECT id AS athlete_id, display_name AS driver FROM athletes"), on="athlete_id")
    result = result.sort_values("position")

    # 5. in-app example: maker book + taker bets tagged with this diagnostic run
    book = house.book(conn, race_id)
    book = book[book["params"].map(lambda p: isinstance(p, dict) and p.get("diagnostic_run") == run_id)]
    bets = data.q(conn, """
        SELECT hb.*, hm.title, hm.kind, coalesce(u.username, hb.counterparty) AS taker
        FROM house_bets hb JOIN house_markets hm ON hm.id = hb.market_id LEFT JOIN users u ON u.id = hb.taker_id
        WHERE hm.params->>'diagnostic_run' = :r ORDER BY hb.id""", r=str(run_id))
    if len(bets):
        bets["taker_pnl"] = [(b.payout - b.stake) if b.status == "won" else (-b.stake if b.status == "lost" else 0.0)
                             for b in bets.itertuples()]
    return dict(run=run, params=params, metrics=metrics, cutoff=cutoff, event=ev, race_id=race_id, preds=preds,
                markets=mk, coherence=coherence, scores=pd.DataFrame(scores), result=result, book=book, bets=bets)


# ---------------------------------------------------------------------------
# Maker replay against the real Polymarket tape (racinglines/markets/strategies/maker_replay.py)
# ---------------------------------------------------------------------------

_REPLAY_CACHE = {}


def event_runs(conn, run_id):
    """All diagnostic runs of the same event as `run_id`, in cutoff order."""
    return data.q(conn, """
        SELECT id, params->>'cutoff' AS cutoff FROM model_runs WHERE kind = 'diagnostic'
          AND params->>'event_key' = (SELECT params->>'event_key' FROM model_runs WHERE id = :i)
        ORDER BY (params->>'cutoff')::timestamp""", i=run_id)


_EVENT_CACHE = {}
# the diagnostic page's default strategy knobs (also used to warm the cache at startup)
DEFAULT_KNOBS = dict(fill="through", half_spread=0.02, size=50, max_pos=250, max_capital=1000, skew=1.0,
                     max_disagree=0.15, min_volume_24h=100, pull_min=15)


def _event(conn, runs, exchange="polymarket"):
    """The event's tape (maker_replay.load_event), loaded once per (runs, exchange)."""
    from racinglines.markets.strategies import maker_replay as R
    ek = runs if exchange == "polymarket" else (runs, exchange)
    if ek not in _EVENT_CACHE:
        _EVENT_CACHE[ek] = R.load_event(conn, runs, exchange=exchange)
    return _EVENT_CACHE[ek]


_TAPE_CACHE = {}


def tape(conn, run_id, exchange="polymarket"):
    """The event's tape and the page's default knobs as gzipped JSON, for the replay that runs in the browser
    (wasm/, static/replay.js): the same data and settings diag.replay uses, so the browser's numbers match the
    server's for the same knobs. Cached per (runs, exchange)."""
    import gzip
    import json

    from racinglines.markets.strategies import maker_replay as R
    from racinglines.markets.strategies.replay_json import event_json, params_json
    runs = tuple(int(i) for i in event_runs(conn, run_id)["id"])
    key = (runs, exchange)
    if key not in _TAPE_CACHE:
        knobs = dict(DEFAULT_KNOBS, **(dict(maker_fee=R.KALSHI_MAKER_FEE) if exchange != "polymarket" else {}))
        body = dict(event=event_json(_event(conn, runs, exchange)), params=params_json(R.Params(**knobs)),
                    exchange=exchange, runs=list(runs))
        _TAPE_CACHE[key] = gzip.compress(json.dumps(body, separators=(",", ":")).encode(), compresslevel=6)
    return _TAPE_CACHE[key]


def replay(conn, run_id, with_sweep=True, exchange="polymarket", **knobs):
    """Replay a maker quoting Polymarket (or `exchange="kalshi"`: Kalshi's tape, with its maker fee) through the
    event, repricing at each diagnostic run's cutoff. `knobs` are markets.strategies.maker_replay.Params fields
    (fill, half_spread, size, max_pos, max_capital, skew, max_disagree, min_volume_24h, pull_min, ...).
    Cached per (runs, knobs); the event's tape is loaded once."""
    from racinglines.markets.strategies import maker_replay as R
    runs = tuple(int(i) for i in event_runs(conn, run_id)["id"])
    if exchange != "polymarket":
        knobs = dict(knobs, maker_fee=knobs.get("maker_fee", R.KALSHI_MAKER_FEE))
    key = (runs, tuple(sorted(knobs.items())), with_sweep) + ((exchange,) if exchange != "polymarket" else ())
    if key in _REPLAY_CACHE:
        return _REPLAY_CACHE[key]
    ev = _event(conn, runs, exchange)
    p = R.Params(**knobs)
    res = R.replay(ev, p)
    f = res["fills"]
    out = dict(event=ev, params=p, result=res, runs=runs,
               stages=R.by_stage(res, ev), summary=R.summary(res).reset_index(),
               skips=R.skip_reasons(res).rename_axis("reason").reset_index(name="market_steps"),
               fills=f.assign(ts=pd.to_datetime(f["ts"], utc=True)) if len(f) else f,
               positions=res["positions"][res["positions"]["inventory"].abs() > 1e-9].sort_values("pnl"),
               sweep=R.sweep(ev, disagree=(p.max_disagree,), base=p) if with_sweep else None)
    _REPLAY_CACHE[key] = out
    return out


def create_example(session, conn, run_id, maker_id, taker, fill="through"):
    """Every fill the replayed maker got (from real Polymarket taker trades) becomes a bet by `taker`, the
    `polymarket-takers` system account (users.REPLAY_TAKER), on the maker's in-app market for that
    outcome, at the maker's price; its P&L is the maker's reversed. That account is Polymarket's crowd,
    not the demo `taker` login, whose paper trades come from its own strategy profile (pipelines/signals.py).
    Replaces any earlier example."""
    from racinglines.db import models as m
    old = [i for (i,) in conn.execute(text("SELECT id FROM house_markets WHERE params->>'diagnostic_run' = :r"),
                                      dict(r=str(run_id)))]
    if old:
        session.execute(text("DELETE FROM house_bets WHERE market_id = ANY(:i)"), dict(i=old))
        session.execute(text("DELETE FROM house_markets WHERE id = ANY(:i)"), dict(i=old))
        session.commit()
    rp = replay(conn, run_id, fill=fill, with_sweep=False)
    fills, by = rp["fills"], {mk.cond: mk for mk in rp["event"]["markets"]}
    markets = {}
    for cond in fills["cond"].unique() if len(fills) else []:
        mk = by[cond]
        link, g = mk.link, fills[fills["cond"] == cond]
        last = g.iloc[-1]
        hm = m.HouseMarket(race_id=int(link["race_id"]), athlete_id=None if pd.isna(link["athlete_id"]) else int(link["athlete_id"]),
                           kind=link["prediction"], title=house.mirror_title(link), fair_prob=float(last["fair"]),
                           spread=2 * rp["params"].half_spread, yes_price=None, no_price=None, maker_id=maker_id,
                           market_link_id=int(link["id"]),
                           params=dict(link["params"] or {}, diagnostic_run=run_id, replay_runs=list(rp["runs"])))
        session.add(hm)
        session.flush()
        markets[cond] = hm.id
    session.commit()
    for f in fills.itertuples():
        if f.qty * min(f.price, 1 - f.price) < 0.01:
            continue
        # maker bought YES at b  <=> taker bought NO at 1 - b; maker sold YES at a <=> taker bought YES at a
        side, price = ("NO", 1 - f.price) if f.side == "buy" else ("YES", f.price)
        house.record_bet(session, markets[f.cond], taker["username"], side, round(f.qty * price, 2), round(price, 4),
                         note=f"Polymarket taker trade {f.ts:%d %b %H:%M} UTC filled the replayed maker",
                         taker_id=taker["id"])
    settled = house.settle_from_results(session, conn, rp["event"]["race_id"], market_ids=list(markets.values()))
    return dict(markets=len(markets), bets=len(fills), settled=len(settled))


# ---------------------------------------------------------------------------
# Summary for the /diag hub
# ---------------------------------------------------------------------------

def event_summaries(conn, hide=()):
    """One row per event with diagnostics: runs, model vs Polymarket at the latest
    cutoff, and the maker replay's P&L (conservative fills, +-2c). hide: competition codes left out (web/audience.py)."""
    runs = data.q(conn, """
        SELECT mr.id, mr.params->>'event_key' AS event_key, mr.params->>'cutoff' AS cutoff
        FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
        WHERE mr.kind = 'diagnostic' AND NOT mr.params ? 'sweep_stage' AND NOT co.code = ANY(CAST(:h AS text[]))
        ORDER BY (mr.params->>'cutoff')::timestamp""", h=list(hide))
    out = []
    for key, g in runs.groupby("event_key", sort=False):
        last = int(g["id"].iloc[-1])
        d = load(conn, last)
        sc = d["scores"].set_index("kind") if len(d["scores"]) else None
        row = dict(event_key=key, venue=d["event"]["venue"], event_id=d["event"]["id"], run_id=last, runs=len(g),
                   cutoffs=", ".join(str(c)[5:16] for c in g["cutoff"]))
        for k, lab in (("race_win", "win"), ("race_podium", "podium")):
            row[f"brier_{lab}_model"] = float(sc.loc[k, "brier_model"]) if sc is not None and k in sc.index else None
            row[f"brier_{lab}_pm"] = float(sc.loc[k, "brier_polymarket"]) if sc is not None and k in sc.index else None
        try:
            rp = replay(conn, last, **DEFAULT_KNOBS)
            t = rp["summary"].set_index("kind").loc["total"]
            row.update(replay_fills=int(t["fills"]), replay_pnl=float(t["pnl"]), replay_markout=float(t["markout_60m"]))
        except Exception:  # noqa: BLE001  (no tape stored for this event)
            row.update(replay_fills=None, replay_pnl=None, replay_markout=None)
        from racinglines.markets import venues as V
        if V.KALSHI_VENUE:                          # the same replay on Kalshi's tape (with Kalshi's maker fee)
            try:
                t = replay(conn, last, with_sweep=False, exchange="kalshi", **DEFAULT_KNOBS)["summary"].set_index("kind").loc["total"]
                row.update(kalshi_fills=int(t["fills"]), kalshi_pnl=float(t["pnl"]))
            except Exception:  # noqa: BLE001  (no Kalshi tape for this event)
                row.update(kalshi_fills=None, kalshi_pnl=None)
        out.append(row)
    return out


def recent_runs(conn, limit=20, hide=()):
    """Every kind of model run, newest first; forecasts that a newer forecast of the
    same competition replaced are marked superseded. hide: competition codes left out (web/audience.py)."""
    df = data.q(conn, """
        SELECT mr.id, mr.created_at, mr.kind, co.name AS competition, s.year AS season,
               coalesce(mr.params->>'event_key', '') AS event_key, coalesce(mr.params->>'cutoff', '') AS cutoff,
               mr.id <> max(mr.id) OVER (PARTITION BY mr.competition_id, mr.category_id, mr.kind) AS superseded
        FROM model_runs mr JOIN competitions co ON co.id = mr.competition_id
        LEFT JOIN seasons s ON s.id = mr.season_id WHERE NOT co.code = ANY(CAST(:h AS text[]))
        ORDER BY mr.id DESC LIMIT :n""", n=limit, h=list(hide))
    df["superseded"] = df["superseded"] & (df["kind"] != "diagnostic")
    df["season"] = df["season"].astype("Int64").astype(object).where(df["season"].notna(), None)
    df["cutoff"] = df["cutoff"].str[:16]
    df["link"] = [f"/lab/diagnostics/{i}" if k == "diagnostic" else f"/lab/runs/{i}" for i, k in zip(df["id"], df["kind"])]
    return df
