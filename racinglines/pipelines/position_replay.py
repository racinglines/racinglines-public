"""
Taker replay for the result-only sports (NASCAR Cup, MotoGP): every raced event of a season, priced by the sport's
model and traded at a few fixed times before the race against an exchange's recorded prices.

    out = run(engine, "nascar", [2025, 2026], venue="kalshi")
    out["races"]     one row per race: markets, tradeable, P&L per taker mode (update / hold / last)
    out["trades"]    every trade of the `update` mode, with its P&L to resolution
    out["totals"]    per mode: P&L, Kalshi's taker fees, P&L after fees, races up
    out["by_kind"]   P&L of the `update` trades by market kind
    out["calibration"], out["reliability"]   model vs exchange, scored on the results (core/calibration.py)

The F1 weekend sweep (pipelines/weekend_sweep.py) is not used or changed: its stages come from F1's session
schedule and its fairs from position_sim. Here, per race:

  1. The field is the race's start list (the drivers or riders with a result in its race round: known before the
     start; their positions are read only in step 4).
  2. The model (sports/<code>.toml [sport] pricing_model) prices the race from results strictly before the event
     (race_model.Event with the field in `info`). It uses race results only, so one pricing serves every stage.
  3. At each stage of [replay] stages (hours from 00:00 UTC on race day), the venue (markets/venue_replay.py:
     Kalshi's or Polymarket's recorded prices and tape) says the price and whether the market was tradeable (priced,
     liquid, open, a coherent group), and the taker strategies (markets/strategies/taker_weekend.py) trade.
  4. Only then the result settles each market, through markets/kinds.settle (the walk-forward's settlement): a top-n
     market is YES when the driver is classified, by the sport's [results] classified rule, in the top n (NASCAR
     "placed": every car is ranked, retirements by laps run; MotoGP "status_ok": an unclassified rider is NO); a
     head-to-head goes by position.

Markets: the sport's links on the venue whose kind is in [replay] kinds. The kind, driver and race come from the
stored link (NASCAR's `nascar link --apply` writes params.kind, athlete_id and race_id), else from the sport's
matcher run in memory ([replay] linker): nothing is written to market_links.

Run with `racinglines strategy replay nascar` / `racinglines strategy replay motogp` (every exchange the sport lists on, by default).
`save` also stores one as-of model run per race (model_runs kind 'diagnostic' + race_predictions, and prediction
records with the sims, on for these sports unless RACINGLINES_PREDICTION_RECORDS is set off); the CLI asks for a
fresh backup first and every saved run carries the batch
id that `undo` deletes.

`buy_all` (off by default; `--buy-all` or RACINGLINES_BUY_ALL=1) adds the debug mode of
markets/strategies/buy_everything.py: one YES and one NO share of every market open and priced at a stage, with no
edge, volume or coherence filter. It is reported beside the taker's modes, which it does not change.
"""

import importlib
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines import progress as PG
from racinglines import sports
from racinglines.core import calibration as CAL
from racinglines.markets.strategies import taker_weekend as RB

N_OF = {"race_win": 1, "race_podium": 3, "race_top5": 5, "race_top10": 10, "race_top20": 20}
TAKER_MODES = ("update", "hold", "last")
UNMARKED_RULE = "placed"             # who is classified in a results frame without a `classified` column (see settle)
STALE = timedelta(hours=6)
COHERENCE_TOL = 0.25                 # as the F1 sweep's (a priori, not tuned for these sports)
MIN_VOLUME_24H = 50.0                # $ traded in the market over the previous 24 h (the F1 taker's floor)
SEED = 7                             # the model's Monte Carlo seed when the settings leave it unset


def spec(sport):
    """The sport's [replay] section, with its competition, events source and pricing model."""
    s = sports.load(sport)
    rp = s.get("replay")
    if not rp:
        raise ValueError(f"sport {sport!r} has no [replay] section in sports/{sport}.toml")
    return dict(rp, sport=sport, competition=s["competition"]["code"], model=s["sport"]["pricing_model"])


def venues(sport):
    """The exchanges the sport's schema lists ([markets] venues)."""
    return tuple(sports.load(sport)["markets"]["venues"])


def replay_venues(sport):
    """The exchanges `--venue all` replays: Kalshi then Polymarket where the sport lists them, then OG.com where its
    schema (exchanges/og.toml) lists the sport."""
    from racinglines import exchanges
    return tuple(v for v in ("kalshi", "polymarket") if v in venues(sport)) + \
        (("og",) if sport in exchanges.sports("og") else ())


def records_on():
    """Prediction records for a replay save: on unless RACINGLINES_PREDICTION_RECORDS is set to an off value
    (the replay of these sports writes them by default; every other writer keeps the switch's own default, off)."""
    import os

    from racinglines.db import records as REC
    v = os.environ.get(REC.SWITCH)
    return True if v is None or not v.strip() else REC.enabled()


def model_for(sp):
    from racinglines.models import race_model as RM
    mod, _, cls = sp["model"].partition(":")
    return RM.bind(getattr(importlib.import_module(mod), cls), sp["sport"])()


# --- what was knowable, per race ------------------------------------------------------------------------

def races(conn, sp, seasons=None):
    """The sport's completed races, in date order: race_id, event_key, name, season, start (the event's first day)."""
    df = pd.read_sql(text("""
        SELECT ra.id AS race_id, e.source_key AS event_key, e.name, s.year AS season, e.start_date AS start
        FROM races ra JOIN events e ON e.id = ra.event_id JOIN seasons s ON s.id = e.season_id
        JOIN competitions co ON co.id = s.competition_id
        WHERE co.code = :c AND e.source = :src AND e.status = 'completed'
          AND EXISTS (SELECT 1 FROM rounds ro WHERE ro.race_id = ra.id AND ro.kind = 'race')
        ORDER BY e.start_date, ra.id"""), conn, params=dict(c=sp["competition"], src=sp["source"]))
    if seasons:
        df = df[df["season"].isin(list(seasons))]
    return df.reset_index(drop=True)


def select(rs, events=None):
    """Only the races of `events` (event keys; "latest" = the last race of `rs`); all of `rs` when None."""
    if not events:
        return rs
    keys = set(events) - {"latest"} | ({rs["event_key"].iloc[-1]} if "latest" in events and len(rs) else set())
    return rs[rs["event_key"].isin(keys)].reset_index(drop=True)


def race_results(conn, race_id):
    """The race round's classification: athlete_id, position (NaN = unclassified), status, grid (the starting slot
    stored in results.extra.grid, NaN where none is), and `classified` by the sport's [results] rule."""
    from racinglines.markets import payoffs as PO
    return PO.mark_sport(_race_results(conn, race_id), sports.race_sport(conn, race_id))


def _race_results(conn, race_id):
    return pd.read_sql(text("""
        SELECT r.athlete_id, r.position, r.status, nullif(r.extra->>'grid', '')::float AS grid
        FROM results r JOIN rounds ro ON ro.id = r.round_id
        WHERE ro.race_id = :r AND ro.kind = 'race' ORDER BY r.position NULLS LAST, r.athlete_id"""),
                       conn, params=dict(r=int(race_id)))


def stage_list(sp, mode="weekend"):
    """The [(label, hours)] of a stage mode: "weekend" (or None) the [replay] stages; "race_day" those plus the
    [replay] race_day_stages (opt-in, so the weekend results stay reproducible: decision log 2026-10-08)."""
    if mode in (None, "weekend"):
        return list(sp["stages"])
    if mode != "race_day" or not sp.get("race_day_stages"):
        raise ValueError(f"stages {mode!r}: one of weekend, race_day (race_day needs [replay] race_day_stages)")
    return list(sp["stages"]) + list(sp["race_day_stages"])


def stage_times(race_start, sp):
    """[(label, time)] of the race's stages: [replay] stages are hours from 00:00 UTC on race day, which is
    race_day_offset days after the event's first day. Naive UTC."""
    day = pd.Timestamp(race_start) + pd.Timedelta(days=int(sp.get("race_day_offset", 0)))
    return [(str(label), day + pd.Timedelta(hours=float(h))) for label, h in sp["stages"]]


def _identifier(sp, conn):
    name = sp.get("linker")
    if not name:
        return None
    L = importlib.import_module(f"racinglines.sources.{name}.links").Linker(conn)
    if name == "nascar":                                   # Found -> the dict the MotoGP linker returns
        def ident(link):
            f = L.identify(link)
            return dict(kind=f.kind, athlete_id=f.athlete_id, race_id=f.race_id, opponent_id=f.opponent_id)
        return ident
    return L.identify


def links(conn, sp, venue):
    """The sport's links on `venue` with a kind of [replay] kinds, one per market (Polymarket: the first token of
    each condition, as the F1 sweep; Kalshi: every ticker). `prediction` is set to the kind (in this frame only)."""
    df = pd.read_sql(text("""SELECT ml.*, a.display_name AS athlete FROM market_links ml
                             JOIN competitions co ON co.id = ml.competition_id
                             LEFT JOIN athletes a ON a.id = ml.athlete_id
                             WHERE co.code = :c AND ml.exchange = :x ORDER BY ml.id"""),
                     conn, params=dict(c=sp["competition"], x=venue))
    if not len(df):
        return df
    ident = _identifier(sp, conn)
    rows = []
    for link in df.to_dict("records"):
        p = dict(link.get("params") or {})
        kind, ath, rid, opp = p.get("kind"), link.get("athlete_id"), link.get("race_id"), p.get("opponent_id")
        ath = None if ath is None or pd.isna(ath) else int(ath)
        rid = None if rid is None or pd.isna(rid) else int(rid)
        if ident is not None and (kind is None or ath is None or rid is None or (kind == "race_h2h" and opp is None)):
            got = ident(link)
            kind, ath, rid, opp = kind or got["kind"], ath or got["athlete_id"], rid or got["race_id"], opp or got["opponent_id"]
        if kind not in sp["kinds"] or ath is None or rid is None or (kind == "race_h2h" and opp is None):
            continue
        if opp is not None:
            p["opponent_id"] = int(opp)
        rows.append(dict(link, prediction=kind, athlete_id=ath, race_id=rid, params=p))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    return out.drop_duplicates("condition_id" if venue == "polymarket" else "token_id", keep="first").reset_index(drop=True)


# --- fair values and settlement -------------------------------------------------------------------------

def fair(sims, kind, athlete_id, opponent_id=None):
    """P(YES) from the simulations, or None when the model has no column for the athlete (or opponent)."""
    try:
        i = sims.index(athlete_id)
        j = sims.index(opponent_id) if kind == "race_h2h" else None
    except ValueError:
        return None
    if kind == "race_h2h":
        return float((sims.rank[:, i] < sims.rank[:, j]).mean())
    return float(((sims.rank[:, i] <= N_OF[kind]) & sims.finished[:, i]).mean())


def settle(kind, athlete_id, opponent_id, res):
    """YES / NO / None through the one settlement of every kind (markets/kinds.settle; top-n: payoffs.top_n, classified
    in the top n, so the replay, the season sweep and the walk-forward agree on a retirement or a disqualification).
    race_results marks who is classified by the sport's [results] classified rule (NASCAR: placed; MotoGP: status OK).
    A market whose driver has no row in the classification stays unsettled here (None: the link may name a driver
    the result doesn't list), where kinds.settle says NO for a top-n market."""
    from racinglines.markets import kinds as K
    from racinglines.markets import payoffs as PO
    if not (res["athlete_id"] == athlete_id).any():
        return None
    if PO.CLASSIFIED_COL not in res:         # a frame not loaded by race_results: the replay's rule before C5, any
        res = PO.mark_classified(res, UNMARKED_RULE)       # placed car (race_results marks it by the sport's rule)
    return K.settle(kind, athlete_id, None if opponent_id is None else {"opponent_id": opponent_id}, res)


def _open(link, t):
    end = link.get("end_date")
    if end is None or pd.isna(end):
        return True
    end = pd.Timestamp(end)
    end = end.tz_convert("UTC").tz_localize(None) if end.tzinfo is not None else end
    return t < end


def race_markets(venue, race_links, sims, res, stages, min_volume_24h=MIN_VOLUME_24H):
    """Per market: each stage's fair, exchange price and tradeable flag, and the outcome, in the shape
    taker_weekend.run_weekend reads. venue: a venue_replay venue over these links (view, coherent)."""
    coherent = {(k, lab): venue.coherent(k, t) for lab, t in stages for k in venue.group_target}
    out = []
    for link in race_links.to_dict("records"):
        kind, a = link["prediction"], int(link["athlete_id"])
        b = (link["params"] or {}).get("opponent_id")
        f = fair(sims, kind, a, b) if sims is not None else None
        st = []
        for lab, t in stages:
            price, _, liquid = venue.view(link, t, min_volume_24h)
            ok = liquid and f is not None and _open(link, t) and coherent.get((kind, lab), True)
            st.append(dict(label=lab, t=t, fair=f, price=price, tradeable=ok, open=_open(link, t)))
        subject = link.get("athlete") or link.get("group_title")
        if kind == "race_h2h":
            subject = f"{subject} v {b}"
        out.append(dict(key=link["token_id"], kind=kind, subject=subject, stages=st, link=link,
                        outcome=settle(kind, a, b, res)))
    return out


def venue_fees(venue, trades):
    """The venue's taker fees on the trades, in dollars: Kalshi's schedule, OG.com's flat fee per contract, else 0."""
    from racinglines.markets.venue_replay import OG
    return kalshi_fees(trades) if venue == "kalshi" else OG.fees(trades) if venue == "og" else 0.0


def kalshi_fees(trades):
    """Kalshi's taker fee on each trade (venue_replay.Kalshi.taker_fee at the mid, per order), in dollars."""
    from racinglines.markets.venue_replay import Kalshi
    if not len(trades):
        return 0.0
    return float(sum(Kalshi.taker_fee(m, abs(s)) for m, s in zip(trades["mid"], trades["shares"])))


# --- the season -----------------------------------------------------------------------------------------

def coherence_tol(sp, venue):
    """The group coherence tolerance on `venue`: the sport's [replay] coherence_tol for that venue, else COHERENCE_TOL.
    A per-kind sweep setting (coherence_tol_by_kind) still overrides it."""
    return float((sp.get("coherence_tol") or {}).get(venue, COHERENCE_TOL))


def _venue(conn, venue, race_links, stages, sp, tape=None):
    """The race's backtest venue: our recorded tape, or Probalytics' (the sport's [markets.probalytics] tag) when the
    tape_source setting (`tape`) or RACINGLINES_TAPE_SOURCE is probalytics (markets/probalytics.py; off by default)."""
    from racinglines.markets import probalytics as PB
    from racinglines.markets.venue_replay import EXCHANGES
    start = min(t for _, t in stages) - timedelta(hours=1)
    end = max(t for _, t in stages) + timedelta(hours=1)
    gt = {k: int(v) for k, v in (sp.get("group_target") or {}).items()}
    if PB.enabled(tape) and venue in PB.EXCHANGES:
        return PB.EXCHANGES[venue](conn, race_links, start, end, gt, coherence_tol(sp, venue),
                                   tag=PB.tag_of(sp["sport"]), book_stale=PB.book_stale_of(sp["sport"]))
    return EXCHANGES[venue](conn, race_links, start, end, gt, coherence_tol(sp, venue), STALE)


def run(engine, sport, seasons=None, venue="kalshi", taker=None, min_volume_24h=MIN_VOLUME_24H, model_settings=None,
        kinds=None, data=None, save=None, events=None, echo=print, buy_all=None, on_race=None, stage_mode="weekend"):
    """The replay over `seasons` (None = every season with races). taker: TakerParams (None = the defaults).
    model_settings: dict for the model's Settings (seed defaults to SEED). kinds: a subset of [replay] kinds.
    data: the model's frame, already loaded. save: None, or dict(engine_url, batch) to store one model run per race.
    events: only these event keys ("latest" = the last race of the selection), e.g. for a spot check; the model still
    learns from every earlier race. buy_all: add the buy-one-of-everything mode (None = RACINGLINES_BUY_ALL).
    on_race: None, or a callable(race, markets) given each race's markets (race_markets) before they are traded
    (pipelines/sport_paper.py stores the taker's trades from them as demo paper positions). stage_mode: stage_list's
    mode ("race_day" adds the [replay] race_day_stages)."""
    from racinglines.markets.strategies import buy_everything as BA
    buy_all = BA.enabled(buy_all)
    sp = spec(sport)
    sp = dict(sp, stages=stage_list(sp, stage_mode))
    if kinds:
        bad = set(kinds) - set(sp["kinds"])
        if bad:
            raise ValueError(f"kinds {sorted(bad)} are not in sports/{sport}.toml [replay] kinds {sp['kinds']}")
        sp = dict(sp, kinds=list(kinds))
    model = model_for(sp)
    st = model.Settings.from_dict({"seed": SEED, **(model_settings or {})})
    t = taker or RB.TakerParams()
    plist = [RB.TakerParams(**{**t.__dict__, "mode": m}) for m in TAKER_MODES]
    from racinglines.models.race_model import Event
    with engine.connect() as conn:
        rs = races(conn, sp, seasons)
        lk = links(conn, sp, venue)
    rs = select(rs, events)
    echo(f"progress {sport} {venue}: {len(rs)} races, {len(lk)} markets of kinds {', '.join(sp['kinds'])}")
    if data is None:
        data = model.load(engine.url.render_as_string(hide_password=False))
    hist = model.history(data, st)
    rng = np.random.default_rng(st.rng_seed)
    rows, trades, calib, ba_trades = [], [], [], []
    for r in PG.track(list(rs.itertuples()), "race", name=lambda r: f"{venue} {r.event_key}"):
        race_links = lk[lk["race_id"] == r.race_id] if len(lk) else lk
        with engine.connect() as conn:
            res = race_results(conn, r.race_id)
            if res.empty:
                continue
            field = sorted(int(a) for a in res["athlete_id"])
            ev = Event(id=r.event_key, season=int(r.season), cutoff=r.start, name=str(r.name), info={"field": field})
            sims = model.price(hist, ev, st, rng)            # every race is priced, traded or not: one rng stream
            if save is not None and sims is not None:
                save_run(save, sp, model, st, r, sims)
            if not len(race_links):
                continue
            stages = stage_times(r.start, sp)
            v = _venue(conn, venue, race_links, stages, sp)
            markets = race_markets(v, race_links, sims, res, stages, min_volume_24h)
        if on_race is not None:
            on_race(r, markets)
        row = dict(season=int(r.season), event_key=r.event_key, race=r.name, race_id=int(r.race_id), markets=len(markets),
                   priced=sum(any(s["price"] is not None for s in m["stages"]) for m in markets),
                   tradeable=sum(any(s["tradeable"] for s in m["stages"]) for m in markets))
        for p in plist:
            tr, per = RB.run_weekend(markets, p)
            s = RB.summarize(tr, per)
            fees = venue_fees(venue, tr)
            row.update({f"{p.mode}_{k}": s[k] for k in ("traded", "trades", "bought", "pnl")}, **{f"{p.mode}_fees": fees})
            if p.mode == "update" and len(tr):
                trades.append(tr.assign(event_key=r.event_key, race=r.name))
        if buy_all:
            tr, per = BA.run_weekend(markets, cost=t.cost)
            s = RB.summarize(tr, per)
            row.update({f"{BA.MODE}_{k}": s[k] for k in ("traded", "trades", "bought", "pnl")},
                       **{f"{BA.MODE}_fees": venue_fees(venue, tr)})
            if len(tr):
                ba_trades.append(tr.assign(event_key=r.event_key, race=r.name))
        calib += [dict(event_key=r.event_key, stage=s["label"], kind=m["kind"], model=s["fair"], market=s["price"],
                       y=float(m["outcome"])) for m in markets if m["outcome"] is not None
                  for s in m["stages"] if s["tradeable"]]
        rows.append(row)
        echo(f"progress {len(rows)} {r.event_key} {r.name}: {row['markets']} markets, {row['priced']} priced, "
             f"{row['tradeable']} tradeable, update {row['update_pnl']:+.2f} (fees {row['update_fees']:.2f})"
             + (f", buy_all {row['buy_all_traded']} bought {row['buy_all_pnl']:+.2f}" if buy_all else "")
             + ("  NO TAPE" if not row["priced"] else "  NOTHING TRADEABLE" if not row["tradeable"] else ""))
    out = summarize(pd.DataFrame(rows), trades, calib, dict(sport=sport, venue=venue, seasons=seasons,
                                                              taker={k: v for k, v in t.__dict__.items() if k != "mode"},
                                                              model=model.name, model_settings=st.to_json(),
                                                              min_volume_24h=min_volume_24h, kinds=sp["kinds"],
                                                              stages=sp["stages"], coherence_tol=coherence_tol(sp, venue),
                                                              group_target=sp.get("group_target") or {}))
    if buy_all:
        out = dict(out, **_buy_all(out, ba_trades))
    return dict(out, data=data)                  # data: the model's frame, for a second venue's pass


def summarize(races_df, trades, calib, params):
    trades = RB.core(pd.concat(trades, ignore_index=True)) if trades else pd.DataFrame()
    totals = {}
    for mode in TAKER_MODES:
        col = f"{mode}_pnl"
        if col in races_df:
            pnl, fees = float(races_df[col].fillna(0).sum()), float(races_df[f"{mode}_fees"].sum())
            totals[mode] = dict(pnl=pnl, fees=fees, net=pnl - fees, bought=float(races_df[f"{mode}_bought"].sum()),
                                races_up=int((races_df[col] > 0).sum()), races=int(races_df[col].notna().sum()))
    cal = pd.DataFrame(calib, columns=["event_key", "stage", "kind", "model", "market", "y"])
    cal_all, rel = CAL.table(cal, ("model", "market"), by=("kind",))
    return dict(races=races_df, trades=trades, totals=totals, by_kind=RB.by(trades, "kind"), calibration=cal_all,
                reliability=rel, params=params, tape=tape(races_df))


def _buy_all(out, ba_trades):
    """The buy-all mode's part of the run: its totals (beside the taker's), its trades, P&L by side and kind, and
    the tape's count of markets it bought."""
    from racinglines.markets.strategies import buy_everything as BA
    r, m = out["races"], BA.MODE
    trades = RB.core(pd.concat(ba_trades, ignore_index=True)) if ba_trades else pd.DataFrame()
    totals = dict(out["totals"])
    if f"{m}_pnl" in r:
        pnl, fees = float(r[f"{m}_pnl"].fillna(0).sum()), float(r[f"{m}_fees"].sum())
        totals[m] = dict(pnl=pnl, fees=fees, net=pnl - fees, bought=float(r[f"{m}_bought"].sum()),
                         races_up=int((r[f"{m}_pnl"] > 0).sum()), races=int(r[f"{m}_pnl"].notna().sum()))
    bought = int(r[f"{m}_traded"].sum()) if f"{m}_traded" in r else 0
    return dict(totals=totals, buy_all_trades=trades, buy_all_by_side=RB.by(trades, "side"),
                buy_all_by_kind=RB.by(trades, "kind"), tape=dict(out["tape"], bought=bought),
                params=dict(out["params"], buy_all=True))


def tape(races_df):
    """What the venue's tape allowed: races with markets, markets priced and tradeable, and the races with markets
    but no stored price at any stage (`no_tape`: the prices and trades were never pulled) or priced but never
    tradeable (`untradeable`: under the volume floor, closed, or an incoherent group)."""
    if not len(races_df):
        return dict(races=0, markets=0, priced=0, tradeable=0, no_tape=[], untradeable=[])
    r = races_df
    return dict(races=len(r), markets=int(r["markets"].sum()), priced=int(r["priced"].sum()),
                tradeable=int(r["tradeable"].sum()), no_tape=r.loc[r["priced"] == 0, "event_key"].tolist(),
                untradeable=r.loc[(r["priced"] > 0) & (r["tradeable"] == 0), "event_key"].tolist())


def traded(out):
    """True when at least one market of the run was tradeable at a stage (the spot check's pass)."""
    return out["tape"]["tradeable"] > 0


# --- the tape: probe and pull ---------------------------------------------------------------------------

TAPE_BEFORE = timedelta(days=2)      # the pull window: from 2 days before a race's first stage (the 24 h volume floor
TAPE_AFTER = timedelta(days=2)       # and the 6 h staleness reach back from it) to 2 days after its last


def tape_plan(engine, sport, venue, seasons=None, events=None):
    """[(race row, its links, window start, window end)] for the races `run` would trade on `venue` (the same
    selection and the same links), with the UTC window whose prices and trades it reads."""
    sp = spec(sport)
    with engine.connect() as conn:
        rs = select(races(conn, sp, seasons), events)
        lk = links(conn, sp, venue)
    out = []
    for r in rs.itertuples():
        g = lk[lk["race_id"] == r.race_id] if len(lk) else lk
        if len(g):
            t = [x for _, x in stage_times(r.start, sp)]
            out.append((r, g, (min(t) - TAPE_BEFORE).tz_localize("UTC").to_pydatetime(),
                        (max(t) + TAPE_AFTER).tz_localize("UTC").to_pydatetime()))
    return out


def probe(plan, venue, n=3, kc=None, echo=print):
    """Read-only: ask the exchange for the tape of `n` markets of the first race in `plan` (trades, and prices over
    the race's window) and print what came back. Writes nothing. Returns [(token, trades, prices)]."""
    if not plan:
        echo(f"probe {venue}: no markets to ask about (no links for these races)")
        return []
    r, g, start, end = plan[0]
    got = []
    if venue == "og":
        from racinglines.markets import exchange_driver as D
        kc = kc or D.Client("og")
        for link in g.head(n).to_dict("records"):
            got.append((link["token_id"], len(kc.window("trades", link["token_id"], start)),
                        len(kc.window("history", link["token_id"], start, end))))
    elif venue == "kalshi":
        from racinglines.markets.kalshi import client as K
        kc = kc or K.Client()
        for link in g.head(n).to_dict("records"):
            series = (link.get("params") or {}).get("series") or str(link["condition_id"]).split("-", 1)[0]
            tr = kc.trades(link["token_id"])
            px = kc.candlesticks(series, link["token_id"], start.timestamp(), end.timestamp(), 60)
            got.append((link["token_id"], len(tr), len(px)))
    else:
        import httpx

        from racinglines.markets.polymarket.sync import CLOB, DATA_API
        from racinglines.sources import http
        with httpx.Client(base_url=DATA_API, timeout=30) as d, httpx.Client(base_url=CLOB, timeout=20) as c:
            for link in g.head(n).to_dict("records"):
                t = http.get(d, "/trades", params={"market": link["condition_id"], "limit": 500, "takerOnly": "true"})
                h = http.get(c, "/prices-history", params={"market": link["token_id"], "startTs": int(start.timestamp()),
                                                          "endTs": int(end.timestamp()), "fidelity": 60})
                got.append((link["token_id"], len(t.json()) if t.status_code == 200 else f"HTTP {t.status_code}",
                            len(h.json().get("history", [])) if h.status_code == 200 else f"HTTP {h.status_code}"))
    echo(f"probe {venue}, {r.event_key} {r.name} ({len(g)} markets; window {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC):")
    for tok, tr, px in got:
        echo(f"  {tok}: {tr} trades, {px} hourly prices")
    return got


def pull(session, conn, plan, venue, kc=None, echo=print):
    """Store the tape the replay reads for every race in `plan`: the trades of its markets' events from the start of
    the race's window on (older ones never count toward a stage's 24 h volume) and hourly prices over the window (markets/kalshi/sync.py, markets/polymarket/sync.py; idempotent upserts). A database
    write: the CLI asks for a fresh backup. Returns dict(races, trades, prices)."""
    n = dict(races=0, trades=0, prices=0)
    for r, g, start, end in PG.track(list(plan), "tape race", name=lambda p: f"{venue} {getattr(p[0], 'event_key', '')}"):
        if venue == "og":
            from racinglines.markets import exchange_driver as D
            evs = sorted(set(g["condition_id"].dropna()))
            tr = D.fetch_trades(session, conn, "og", events=evs, since=start, client=kc)
            px = D.fetch_history(session, conn, "og", start, end, events=evs, client=kc)
        elif venue == "kalshi":
            from racinglines.markets.kalshi import sync as KS
            evs = sorted(set(g["condition_id"]))
            tr = KS.fetch_trades(session, conn, evs, since=start, kc=kc)
            px = KS.fetch_history(session, conn, evs, start, end, 60, kc=kc)
        else:
            from racinglines.markets.polymarket import sync as PS
            evs = sorted(set(g["event_slug"].dropna()))
            tr = PS.fetch_trades(session, conn, evs, since=start)
            px = PS.fetch_history(session, conn, evs, start, end, 60)
        n["races"] += 1
        n["trades"] += tr
        n["prices"] += px
        echo(f"pull {venue} {n['races']}/{len(plan)} {r.event_key} {r.name}: {len(g)} markets, {tr} trades, {px} prices")
    return n


# --- saved runs (the backfill) --------------------------------------------------------------------------

def batch_id():
    return "replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_run(save, sp, model, st, race, sims, params=None):
    """One as-of run of one race: model_runs (kind 'diagnostic', params.cutoff = the event's first day, so the race
    page reads it as the pre-race price) and race_predictions (win / podium / top 10; top 5 and top 20 in extra);
    prediction records beside it when RACINGLINES_PREDICTION_RECORDS is on. Returns the run id.
    params: the run's mode params instead of the replay's (mode "replay as-of" and save["batch"]), e.g. the live
    signal engine's (pipelines/sport_signals.py)."""
    from racinglines.db import models as m
    from racinglines.db import records as REC
    from racinglines.db.config import get_session
    from racinglines.models import outcomes as O
    p = {k: [fair(sims, k, a) for a in sims.entrants] for k in N_OF}
    cutoff = pd.Timestamp(race.start)
    with get_session(save["engine_url"]) as s:
        comp = s.execute(text("SELECT id FROM competitions WHERE code = :c"), dict(c=sp["competition"])).scalar()
        season = s.execute(text("SELECT id FROM seasons WHERE competition_id = :c AND year = :y"),
                           dict(c=comp, y=int(race.season))).scalar()
        run = m.ModelRun(competition_id=comp, season_id=season, model=model.name, kind="diagnostic",
                         data_through=(cutoff - pd.Timedelta(days=1)).date(),
                         params={**(dict(mode="replay as-of", replay_batch=save["batch"]) if params is None else {}),
                                 **dict(sport=sp["sport"], event_key=race.event_key, cutoff=str(cutoff),
                                        sims=int(sims.n_sims), model_settings=st.to_json(), model_key=st.model_key,
                                        field=len(sims.entrants)), **(params or {})})
        s.add(run)
        s.flush()
        for i, a in enumerate(sims.entrants):
            s.add(m.RacePrediction(model_run_id=run.id, race_id=int(race.race_id), target=f"asof:{race.event_key}",
                                   athlete_id=int(a), win_prob=p["race_win"][i], podium_prob=p["race_podium"][i],
                                   top10_prob=p["race_top10"][i],
                                   extra=dict(top5_prob=p["race_top5"][i], top20_prob=p["race_top20"][i])))
        s.commit()
        rid = run.id
    if records_on():
        df = O.to_records(sims, rid, sp["sport"], model.name, int(race.season), race.event_key, race.name, "asof",
                          cutoff, kinds=["race_win", "race_podium", "race_top10", "race_h2h"])
        REC.write(rid, df, sims)
    return rid


def undo(session, batch):
    """Delete every run a save pass stored (race_predictions go with them). Returns runs deleted."""
    ids = [r for (r,) in session.execute(text("SELECT id FROM model_runs WHERE params->>'replay_batch' = :b"),
                                         dict(b=batch)).all()]
    if ids:
        session.execute(text("DELETE FROM race_predictions WHERE model_run_id = ANY(:i)"), dict(i=ids))
        session.execute(text("DELETE FROM model_runs WHERE id = ANY(:i)"), dict(i=ids))
    return len(ids)


def format_report(out):
    """The run's text: what the tape allowed, then totals per mode, P&L by kind, calibration. A run with nothing
    tradeable says so and prints no P&L (a zero there would read as a result)."""
    p, tp = out["params"], out["tape"]
    lines = [f"{p['sport']} on {p['venue']}, seasons {p['seasons'] or 'all'}: {tp['races']} races with markets, "
             f"{tp['markets']} markets, {tp['priced']} priced, {tp['tradeable']} tradeable "
             f"(model {p['model']}; stages {', '.join(l for l, _ in p['stages'])})"]
    if not tp["races"]:
        lines.append(f"  NO MARKETS: no {p['venue']} links of kinds {', '.join(p['kinds'])} matched these races "
                     f"(the sport's {p['venue']} sync has not run on this database, or its markets are not identified).")
        return "\n".join(lines)
    ba = p.get("buy_all")
    if not tp["tradeable"] and not (ba and tp.get("bought")):
        why = ("NO TAPE: no stored price for any market at any stage. Pull the prices and trades first "
               f"(racinglines {p['sport']} replay ... --tape pull --backup FILE)." if not tp["priced"] else
               f"NOTHING TRADEABLE: markets are priced but none passed the ${p['min_volume_24h']:g} 24 h volume floor, "
               "was open, and sat in a coherent group at a stage.")
        lines.append(f"  NOT TRADED. {why} No P&L to report.")
        return "\n".join(lines)
    if ba:
        lines.append(f"  DEBUG buy_all: one YES and one NO share of {tp['bought']} markets open and priced at a stage, "
                     "no filters; a pair loses its costs and fees by construction (plumbing check, not an edge).")
    if not tp["tradeable"]:
        lines.append("  NOT TRADED by the taker (nothing passed its filters): buy_all only.")
    for key, what in (("no_tape", "NO TAPE (no stored price at any stage)"), ("untradeable", "NOTHING TRADEABLE")):
        if tp[key]:
            shown = ", ".join(tp[key][:8]) + (f" and {len(tp[key]) - 8} more" if len(tp[key]) > 8 else "")
            lines.append(f"  WARNING {len(tp[key])} of {tp['races']} races {what}: {shown}")
    for mode, v in out["totals"].items():
        if not tp["tradeable"] and mode != "buy_all":
            continue
        lines.append(f"  {mode:7} P&L {v['pnl']:+9.2f}  fees {v['fees']:7.2f}  net {v['net']:+9.2f}  bought {v['bought']:9.2f}  "
                     f"races up {v['races_up']}/{v['races']}")
    if len(out["by_kind"]):
        lines.append("  update trades by kind:")
        lines += ["    " + x for x in out["by_kind"].to_string(index=False).splitlines()]
    for key, what in (("buy_all_by_side", "side"), ("buy_all_by_kind", "kind")):
        if len(out.get(key, ())):
            lines.append(f"  buy_all trades by {what}:")
            lines += ["    " + x for x in out[key].to_string(index=False).splitlines()]
    if len(out["calibration"]):
        lines.append("  model vs exchange at tradeable stages (lower is better):")
        lines += ["    " + x for x in out["calibration"].to_string(index=False).splitlines()]
    return "\n".join(lines)


def write(out, folder):
    """races.csv, trades.csv, calibration.csv and summary.json into `folder` (and buy_all_trades.csv with buy_all)."""
    import json
    folder.mkdir(parents=True, exist_ok=True)
    out["races"].to_csv(folder / "races.csv", index=False)
    out["trades"].to_csv(folder / "trades.csv", index=False)
    out["calibration"].to_csv(folder / "calibration.csv", index=False)
    if "buy_all_trades" in out:
        out["buy_all_trades"].to_csv(folder / "buy_all_trades.csv", index=False)
    (folder / "summary.json").write_text(json.dumps(dict(params=out["params"], totals=out["totals"], tape=out["tape"]), indent=1,
                                                    default=lambda v: v.isoformat() if isinstance(v, date) else str(v)))
    return folder
