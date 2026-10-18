"""
Default season-long strategy for the championship markets (drivers' and constructors'
champion), replayed through the season on Polymarket's recorded prices.

Decisions: pre-season (PRESEASON_DAYS before the first practice), then one hour
after every race ends. At each, an as-of season forecast (inputs.laps_pricing.forecast with the
cutoff; only data from sessions that had ended) gives our fair values; the strategy
(racinglines/markets/strategies/season.py) rebalances and holds. Execution is at Polymarket's hourly price one
hour after the decision plus a per-market cost (half the recorded spread, 0.5c if
none recorded, plus 0.5c slippage). Markets with under MIN_VOLUME traded are skipped.

Demo liberty (documented): the pre-season forecast simulates the published 2026 entry
list (identities only) instead of the 2025 field.

The championship sleeve (paper only, off by default; docs/paper-trading.md):

    racinglines strategy season-strategy f1 --paper --venue polymarket --after-round 15
    racinglines strategy season-strategy f1 --paper --venue kalshi --after-round 15

The replay itself (no --paper) takes the same --venue, and also OG.com (`--venue og`, OG.com's 20 drivers' and
constructors' champion contracts, exchanges/og.toml): Polymarket stays the default and its output is unchanged. OG.com
prices are read by the replay venue's rule (markets/venue_replay.OG: every stored price, trade or book quote except an
empty book's 0.50), cost its flat fee per contract plus slippage, and are not held to the $ volume floor (its books
are thin and its API keeps about a month, so only the decisions from late August on can trade).

replays the same strategy from pre-season through the decision after round N on one exchange
(Polymarket's championship markets, or Kalshi's KXF1 / KXF1CONSTRUCTORS champion markets, keyed by
ticker) and stores the state after that rebalance as the demo maker's paper positions, marked as
paper_positions.venue = 'season:<exchange>' with event_key '<year>-season' and no candidate. The accounts'
track records (pipelines/story.py track_record) and the signal engine's rebuilds (pipelines/signals.py store)
select their own venues and weekends, and the Positions page shows only markets with signals, so A, C and K's
records don't see the sleeve; `racinglines mcp`'s positions tool lists it under its own venue.
"""

from datetime import timedelta

import pandas as pd
from sqlalchemy import text

from racinglines.markets.strategies import season as SS
from racinglines.core.stats import last_at
from racinglines.markets.strategies.sizing import target_shares

KINDS = ("champion", "constructors_champion")
MIN_VOLUME = 100_000
PRESEASON_DAYS = 5
DECISION_LAG = timedelta(hours=1)          # after the race has ended
DEFAULT_HALF_SPREAD, SLIPPAGE = 0.005, 0.005
HISTORY_START = pd.Timestamp("2025-12-09", tz="UTC")


SLEEVE_USER = "maker"                      # the demo maker: the account the championship sleeve belongs to
SLEEVE_EVENT = "{year}-season"             # paper_positions.event_key of the sleeve
KALSHI_TAKER_FEE = 0.07                    # Kalshi's taker fee rate: fee = rate x price x (1 - price) per contract
VENUES = ("polymarket", "kalshi", "og")    # the exchanges run_season replays (`f1 season-strategy --venue`)
MIN_VOLUME_BY = {"og": 0}                  # the exchanges held to another floor than MIN_VOLUME (OG.com: none, thin books)


def sleeve_venue(exchange):
    """paper_positions.venue of the championship sleeve on one exchange ('season:polymarket', 'season:kalshi')."""
    return f"season:{exchange}"


def season_links(conn, year, exchange="polymarket"):
    """The season's championship markets on one exchange with at least MIN_VOLUME traded (Kalshi: contracts).
    One row per token: Polymarket's YES token, or Kalshi's ticker (KXF1-26-XX), whose condition_id is the
    event ticker shared by every driver."""
    return pd.read_sql(text("""
        SELECT ml.*, a.display_name AS athlete FROM market_links ml LEFT JOIN athletes a ON a.id = ml.athlete_id
        WHERE ml.race_id IS NULL AND ml.prediction = ANY(:k) AND coalesce(ml.volume, 0) >= :v AND ml.exchange = :x
          AND ml.competition_id = (SELECT id FROM competitions WHERE code = 'f1_wdc')
          AND extract(year FROM ml.end_date) IN (:y, :y + 1)
        ORDER BY ml.prediction, ml.volume DESC"""), conn,
        params=dict(k=list(KINDS), v=MIN_VOLUME_BY.get(exchange, MIN_VOLUME), y=year, x=exchange))


def fetch_history(session, conn, tokens, start=HISTORY_START, end=None, echo=print):
    """Hourly price history in 7-day windows (the API's limit); weeks already stored are skipped."""
    from racinglines.markets import store as MS

    from racinglines.markets.polymarket.sync import fetch_history as fh
    end = end or pd.Timestamp.now(tz="UTC")
    n, a = 0, start
    while a < end:
        b = min(a + timedelta(days=7), end)
        have = MS.read(conn, "prices", tokens=tokens, start=a, end=b)
        if have["token_id"].nunique() < max(1, int(0.8 * len(tokens))) or b > end - timedelta(days=8):
            n += fh(session, conn, None, a.to_pydatetime(), b.to_pydatetime(), fidelity=60, tokens=tokens)
        a = b
    echo(f"progress season history: {n} new price points")
    return n


def decision_times(meas, year):
    """[(label, t)] UTC: pre-season, then after each race that has run."""
    res = meas.res[meas.res["year"] == year]
    first = res["session_ts"].min()
    out = [("pre-season", (first - timedelta(days=PRESEASON_DAYS)).tz_localize("UTC"))]
    races = res[res["round"] == "race"].groupby("series_round")[["session_ts", "venue"]].first().sort_index()
    for rnd, r in races.iterrows():
        t = (r["session_ts"] + timedelta(minutes=150) + DECISION_LAG).tz_localize("UTC")
        out.append((f"after R{int(rnd):02d} {r['venue']}", t))
    return out


def asof_forecasts(engine, engine_url, meas, hist, year, times, n_sims=5000, reforecast=False, echo=print,
                   variant="baseline"):
    """Championship forecast as of each decision (kind='season_asof'); stored ones with the
    same model variant are reused (runs stored without a variant are the baseline)."""
    from racinglines.db.queries import records

    from racinglines.models.inputs import laps_pricing as run
    with engine.connect() as c:
        have = pd.read_sql(text("""SELECT id, params->>'cutoff' AS cutoff FROM model_runs WHERE kind = 'season_asof'
                                   AND (params->>'year')::int = :y
                                   AND coalesce(params->>'variant', 'baseline') = :v ORDER BY id"""), c,
                           params=dict(y=year, v=variant))
    have = {str(pd.Timestamp(cu)): int(i) for i, cu in zip(have["id"], have["cutoff"])}
    first_event = meas.res[meas.res["year"] == year].sort_values("session_ts")["event_id"].iloc[0]
    schedule = run.upcoming_schedule(year)
    out = []
    for i, (label, t) in enumerate(times):
        cutoff = t.tz_convert("UTC").tz_localize(None)
        if str(cutoff) in have and not reforecast:
            out.append((label, t, have[str(cutoff)]))
            continue
        field = run.entry_list(meas, first_event) if label == "pre-season" else None   # demo liberty: published entry list
        _, standings, ex = run.forecast(meas, hist, year, cutoff=cutoff, n_sims=n_sims, schedule=schedule, field=field,
                                        race_prices=False)
        rid = run.save_forecast(engine_url, year, [], standings,
                                dict(cutoff=str(cutoff), year=year, label=label, sims=n_sims, drift=ex["drift"],
                                     **({} if variant == "baseline" else dict(variant=variant))),
                                dict(constructors=records(ex["constructors"]), latest_data=ex["latest_data"]),
                                kind="season_asof")
        out.append((label, t, rid))
        echo(f"progress {i + 1}/{len(times)} season forecast {label}: run #{rid}")
    return out


def market_costs(conn, tokens):
    """$ per share per trade: half the median recorded spread (0.5c default) + slippage."""
    from racinglines.markets import store as MS
    bk = MS.read(conn, "books", tokens=tokens)
    half = {}
    if len(bk):
        bk = bk.dropna(subset=["best_bid", "best_ask"])
        half = ((bk["best_ask"] - bk["best_bid"]) / 2).groupby(bk["token_id"]).median().to_dict()
    return {t: max(half.get(t, DEFAULT_HALF_SPREAD), 0.0005) + SLIPPAGE for t in tokens}


def kalshi_costs(by, toks, asof=None):
    """$ per contract per trade on Kalshi: the taker fee at the market's price as of `asof` (the latest
    recorded price by default; the worst case, 0.5, without one) + slippage."""
    out = {}
    for t in toks:
        g = by.get(t)
        p = None
        if g is not None and len(g):
            h = g if asof is None else g[g["ts"] <= asof]
            p = float(h["price"].iloc[-1]) if len(h) else None
        p = 0.5 if p is None else p
        out[t] = KALSHI_TAKER_FEE * p * (1 - p) + SLIPPAGE
    return out


def og_prices(conn, toks):
    """OG.com's prices for the tokens, as its replay venue reads them (markets/venue_replay.OG.observed: minute prices
    other than an empty book's 0.50, every trade, each book's mid or its one quoting side), ts UTC-aware."""
    from racinglines.markets import store as MS
    from racinglines.markets.venue_replay import OG
    root = MS.root_for("og")
    px = OG.observed(*(MS.read(conn, name, tokens=toks, root=root) for name in ("prices", "trades", "books")))
    return px.assign(ts=px["ts"].dt.tz_localize("UTC"))


def og_costs(toks):
    """$ per contract per trade on OG.com: its flat taker fee (exchanges/og.toml, unverified) + slippage."""
    from racinglines.markets.venue_replay import OG
    fee = OG.fee_per_contract()
    return {t: fee + SLIPPAGE for t in toks}


def build_markets(conn, links, exchange="polymarket", asof=None):
    """{token_id: SeasonMarket} from the exchange's recorded prices (the market store reads every exchange's
    tree, and token ids don't collide). Kalshi markets are grouped by ticker (their token_id; the
    condition_id is the event) and cost the taker fee instead of a spread. OG.com's prices are its replay
    venue's (og_prices) and cost its flat fee per contract."""
    from racinglines.markets import store as MS
    toks = links["token_id"].tolist()
    px = og_prices(conn, toks) if exchange == "og" else MS.read(conn, "prices", tokens=toks)
    by = dict(tuple(px.groupby("token_id"))) if len(px) else {}
    costs = kalshi_costs(by, toks, asof) if exchange == "kalshi" else og_costs(toks) if exchange == "og" \
        else market_costs(conn, toks)
    out = {}
    for link in links.to_dict("records"):
        g = by.get(link["token_id"])
        if g is None or not len(g):
            continue
        closed = bool(link["closed"]) and link["resolved_yes"] is not None and not pd.isna(link["resolved_yes"])
        subject = link["athlete"] or (link["params"] or {}).get("team") or link["group_title"]
        out[link["token_id"]] = SS.SeasonMarket(
            key=link["token_id"], kind=link["prediction"], subject=subject,
            ts=g["ts"].to_numpy(), px=g["price"].to_numpy(float), cost=costs[link["token_id"]],
            outcome=bool(link["resolved_yes"]) if closed else None, closed_at=g["ts"].iloc[-1] if closed else None)
    return out


def fairs(conn, links, run_id):
    from racinglines.db import reads as D
    cache = {}
    return {lk["token_id"]: D.model_prob(conn, lk, cache, run_id=run_id)[0] for lk in links.to_dict("records")}


def run_season(engine, engine_url, year=2026, params=SS.SeasonParams(), fetch=True, reforecast=False, n_sims=5000,
               echo=print, variant="baseline", exchange="polymarket"):
    """The replay on one exchange (Polymarket, the default; Kalshi or OG.com read their stored tape: fetch is
    Polymarket's history API, so it runs only for Polymarket)."""
    from racinglines.db.config import get_session
    from racinglines.db import reads as D

    from racinglines.models.inputs import laps_pricing as run
    with engine.connect() as c:
        links = season_links(c, year, exchange)
    if exchange == "polymarket":
        echo(f"progress {len(links)} championship markets with >= ${MIN_VOLUME:,} traded")
    else:
        echo(f"progress {len(links)} {exchange} championship markets with >= {MIN_VOLUME_BY.get(exchange, MIN_VOLUME):,} traded")
    if fetch and exchange == "polymarket":
        with engine.connect() as c, get_session(engine_url) as s:
            fetch_history(s, c, links["token_id"].tolist(), echo=echo)
    echo("progress 0/1 building history")
    meas = run.Measurements.load(engine)
    hist = run.history(meas)
    times = decision_times(meas, year)
    runs = asof_forecasts(engine, engine_url, meas, hist, year, times, n_sims=n_sims, reforecast=reforecast, echo=echo,
                          variant=variant)
    with engine.connect() as c:
        markets = build_markets(c, links, exchange=exchange)
        decisions = [dict(t=t, label=label, run_id=rid, fairs=fairs(c, links, rid)) for label, t, rid in runs]
        live_run, _ = D.latest_forecast_run(c, int(links["competition_id"].iloc[0])) if len(links) else (None, None)
        live_fair = fairs(c, links, live_run) if live_run else {}
    now = pd.Timestamp.now(tz="UTC")
    marks = pd.date_range(times[0][1].normalize(), now, freq="D")
    res = SS.replay(markets, decisions, params, now=now, marks=marks)
    hold = SS.replay(markets, decisions, SS.SeasonParams(**{**params.__dict__, "mode": "hold"}), now=now, marks=marks)
    # what the strategy would do right now (live forecast vs the latest price)
    pos = res["positions"].set_index("key") if len(res["positions"]) else pd.DataFrame()
    now_rows = []
    for k, mk in markets.items():
        if mk.outcome is not None:
            continue
        price = last_at(mk.ts, mk.px, now, timedelta(days=3))
        fair = live_fair.get(k)
        if price is None or fair is None:
            continue
        held_side = pos.loc[k, "side"] if k in pos.index and pos.loc[k, "status"] == "open" else None
        held = float(pos.loc[k, "shares"]) if held_side else 0.0
        ty, tn = target_shares(fair, price, mk.cost, params.min_edge, params.stake_per_edge, params.max_stake)
        if not params.price_band[0] <= price <= params.price_band[1]:      # outside the band: only reduce
            ty = min(ty, held if held_side == "YES" else 0.0)
            tn = min(tn, held if held_side == "NO" else 0.0)
        tgt_side, tgt = ("YES", ty) if ty else (("NO", tn) if tn else (None, 0.0))
        if tgt_side == held_side and abs(tgt - held) * (price if tgt_side == "YES" else 1 - price) < params.min_trade:
            action = "hold" if held else "no position"
        elif tgt_side is None:
            action = f"sell {held_side}" if held else "no position"
        elif tgt_side == held_side:
            action = f"{'add' if tgt > held else 'trim'} {tgt_side}"
        else:
            action = f"buy {tgt_side}" + (f" (close {held_side})" if held else "")
        now_rows.append(dict(key=k, kind=mk.kind, subject=mk.subject, price=price, fair=fair, edge=fair - price,
                             holding=f"{held:.0f} {held_side}" if held_side else "", target=f"{tgt:.0f} {tgt_side}" if tgt_side else "",
                             action=action, cost=mk.cost))
    now_df = pd.DataFrame(now_rows)
    if len(now_df):
        now_df = now_df.assign(a=now_df["edge"].abs()).sort_values(["kind", "a"], ascending=[False, False]).drop(columns="a")
    return dict(result=res, hold=hold, now=now_df, decisions=[dict(label=d["label"], t=d["t"], run_id=d["run_id"])
                                                               for d in decisions],
                params=params, live_run=live_run, markets=len(markets), exchange=exchange)


# ---------------------------------------------------------------------------
# The championship sleeve: the strategy's state after one round, stored as paper positions
# ---------------------------------------------------------------------------

def sleeve_positions(markets, trades, now):
    """The sleeve's paper_positions rows from a replay's trades: one per market traded (market_key, kind,
    subject, yes_shares, no_shares, cash, mark, outcome). Shares and cash are the trades' sums (a settled
    market keeps its shares: the outcome values them, as every reader of paper_positions does); the mark
    is the latest recorded price as of `now` (within 30 days), the outcome only for markets closed by then."""
    rows = []
    for k, mk in markets.items():
        g = trades[trades["key"] == k] if len(trades) else trades
        if not len(g):
            continue
        yes, no = (float(g.loc[g["side"] == sd, "shares"].sum()) for sd in ("YES", "NO"))
        yes, no = (0.0 if abs(x) < 1e-9 else x for x in (yes, no))      # a closed position, not -0.000
        cash = -float((g["shares"] * g["price"]).sum())
        settled = mk.outcome is not None and mk.closed_at is not None and mk.closed_at <= now
        mark = last_at(mk.ts, mk.px, now, timedelta(days=30))
        rows.append(dict(market_key=k, kind=mk.kind, subject=mk.subject, yes_shares=yes, no_shares=no, cash=cash,
                         mark=None if mark is None else float(mark), outcome=bool(mk.outcome) if settled else None))
    return rows


def sleeve_pnl(rows):
    """P&L of the rows as the track record values them: settled at the outcome, else marked; unmarked rows count 0."""
    tot = 0.0
    for r in rows:
        y = r["outcome"] if r["outcome"] is not None else r["mark"]
        if y is not None:
            tot += r["cash"] + r["yes_shares"] * y + r["no_shares"] * (1 - y)
    return float(tot)


def sleeve_rebalance(engine, engine_url, year=2026, after_round=None, exchange="polymarket", params=SS.SeasonParams(),
                     n_sims=5000, variant="baseline", now=None, echo=print):
    """Replay the default strategy on one exchange from pre-season through the decision after round
    `after_round` (the last raced round when None) and return the sleeve's state after that rebalance:
    dict(label, t, exec_at, run_id, exchange, venue, event_key, trades (this decision's), positions
    (paper_positions rows: market_key, kind, subject, yes_shares, no_shares, cash, mark, outcome), summary).
    Marks are the latest recorded price as of `now` (default: now); markets that had closed by then carry
    their outcome. Nothing is written."""
    from racinglines.models.inputs import laps_pricing as run
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    with engine.connect() as c:
        links = season_links(c, year, exchange)
    echo(f"progress {len(links)} {exchange} championship markets with >= {MIN_VOLUME:,} traded")
    meas = run.Measurements.load(engine)
    hist = run.history(meas)
    times = decision_times(meas, year)
    raced = [int(label.split()[1][1:]) for label, _ in times[1:]]
    if after_round is None:
        after_round = max(raced)
    if after_round not in raced:
        raise ValueError(f"round {after_round} has not been raced in {year} (raced: {raced})")
    times = times[:raced.index(after_round) + 2]        # pre-season, then after R01 .. after R<after_round>
    label, t = times[-1]
    exec_at = t + params.exec_delay
    runs = asof_forecasts(engine, engine_url, meas, hist, year, times, n_sims=n_sims, echo=echo, variant=variant)
    with engine.connect() as c:
        markets = build_markets(c, links, exchange=exchange, asof=t)
        decisions = [dict(t=tt, label=lab, run_id=rid, fairs=fairs(c, links, rid)) for lab, tt, rid in runs]
    res = SS.replay(markets, decisions, params, now=exec_at, marks=[exec_at])
    tr = res["trades"]
    rows = sleeve_positions(markets, tr, now)
    this = tr[tr["decision"] == label] if len(tr) else tr
    summary = dict(pnl=sleeve_pnl(rows), trades=int(len(this)),
                   positions=sum(1 for r in rows if r["yes_shares"] or r["no_shares"]),
                   settled=sum(1 for r in rows if r["outcome"] is not None), replay=res["summary"])
    return dict(label=label, t=t, exec_at=exec_at, run_id=runs[-1][2], exchange=exchange, venue=sleeve_venue(exchange),
                event_key=SLEEVE_EVENT.format(year=year), year=year, after_round=after_round, trades=this,
                positions=rows, summary=summary, markets=len(markets), decisions=[lab for lab, _, _ in runs])


def store_sleeve(conn, user_id, out):
    """Replace the user's championship-sleeve paper positions on the exchange (venue 'season:<exchange>',
    event_key '<year>-season', no candidate) with the rebalance's. Returns the number of rows stored.
    Only that venue's rows are touched: the weekend records (venue polymarket / kalshi / private) are not."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from racinglines.db import models as m
    conn.execute(text("DELETE FROM paper_positions WHERE user_id = :u AND venue = :v AND event_key = :e"),
                 dict(u=user_id, v=out["venue"], e=out["event_key"]))
    for p in out["positions"]:
        conn.execute(pg_insert(m.PaperPosition).values(dict(user_id=user_id, candidate_id=None, race_id=None,
                                                            event_key=out["event_key"], venue=out["venue"], **p)))
    return len(out["positions"])
