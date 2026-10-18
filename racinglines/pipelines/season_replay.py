"""
Championship replay for the result-only sports (NASCAR Cup, MotoGP): the default season strategy of the F1
championship sleeve (markets/strategies/season.py) traded against an exchange's recorded champion prices, with our
fairs from an as-of season forecast after every race.

    out = run(engine, "motogp", 2026, venue="kalshi")
    out["result"]     the strategy's trades, positions, equity and summary (mode update: rebalance after every race)
    out["hold"]       the same, entering at the first decision only and holding
    out["decisions"]  one row per decision: its time, races run and left, and the favourite's fair

Per decision (the day after each points race of the season, 12:00 UTC):

  1. The standings as of the decision, from the results of the rounds run before it (NASCAR: models/nascar_season
     standings_now, summed from race results; the official points feed is today's only, so it is not used here.
     MotoGP: models/motogp_season standings_asof, Grand Prix and sprint points; with no sprint result stored for
     the season, Grand Prix points only, on both sides, flagged SPRINT POINTS MISSING in the report).
  2. The sport's race model prices every remaining race with only the results before the decision (race_model.Event
     with the decision date as its cutoff and the last race's field), and the season is simulated through the
     format (NASCAR's 2026 Chase; MotoGP's points). The champion fairs are markets.kinds.season_fair("champion").
  3. The strategy rebalances at the exchange's recorded price an hour later, plus the venue's cost: Kalshi's taker
     fee at the price, OG.com's flat fee per contract, Polymarket's half-spread; each plus slippage
     (pipelines/season_strategy.py). A market that resolves settles; the rest are marked at their last price.

Markets: the sport's champion contracts for the season on the venue (NASCAR: pipelines/nascar_season.champion_links,
Cup only; MotoGP: Kalshi's KXMOTOGP series, or a listing that names the MotoGP championship), with the rider from the
stored link, else the sport's name resolver in memory. Nothing is written to the database.

NOT CALIBRATED: the season forecasts simulate races independently with the form as of the decision (see both
models' notes); NASCAR's was more favourite-heavy than every exchange on 2026-09-30. A P&L here is a first backtest
cell, not a validated strategy. Off by default: the commands run only with RACINGLINES_SEASON_REPLAY=1.
"""

import os
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines.markets.strategies import season as SS

SWITCH = "RACINGLINES_SEASON_REPLAY"
SPORTS = ("nascar", "motogp")
N_SIMS = 2000
SEED = 7
DECISION_HOUR = 12                          # UTC, the day after race day
DEFAULT_PARAMS = SS.SeasonParams()          # the F1 sleeve's defaults (3-pt edge, $500 per edge, $150 cap, $1,500)


def enabled():
    return os.environ.get(SWITCH, "").strip().lower() in ("1", "true", "yes", "on")


# --- the season's schedule and decisions -------------------------------------------------------------------------

def _spec(sport):
    from racinglines.pipelines import position_replay as PR
    return PR.spec(sport)


def schedule(conn, sport, year):
    """The season's points races in date order: event_id, name, race_day (a date), done (has race results), plus the
    format columns the sport's season model needs (NASCAR: chase, stages)."""
    sp = _spec(sport)
    if sport == "nascar":
        from racinglines.models import nascar_season as NS
        _, sch = NS.load_state(conn, year, competition=sp["competition"], source=sp["source"])
        return sch.assign(race_day=pd.to_datetime(sch["date"]).dt.date + timedelta(days=int(sp.get("race_day_offset", 0))))
    sch = pd.read_sql(text("""
        SELECT e.id AS event_id, e.name, e.source_key AS event_key, e.start_date AS date,
               EXISTS (SELECT 1 FROM races ra JOIN rounds ro ON ro.race_id = ra.id AND ro.kind = 'race'
                       JOIN results r ON r.round_id = ro.id WHERE ra.event_id = e.id) AS done
        FROM events e JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
        WHERE co.code = :c AND e.source = :src AND s.year = :y ORDER BY e.start_date, e.id"""), conn,
                      params=dict(c=sp["competition"], src=sp["source"], y=year))
    sch = pd.concat([sch, _motogp_unlisted(year, sch)], ignore_index=True).sort_values("date").reset_index(drop=True)
    return sch.assign(race_day=pd.to_datetime(sch["date"]).dt.date + timedelta(days=int(sp.get("race_day_offset", 0))))


def _motogp_unlisted(year, sch):
    """The season's Grands Prix not yet in the database (the ingest stores finished events only), from the fetched
    calendar (data/raw/motogp/pulselive/<year>/events.json; test events skipped). Empty without the file."""
    from racinglines.sources.motogp import fetch as F
    path = F.season_path(year) / "events.json"
    evs = F._read(path) or []
    have = set(sch["event_key"])
    rows = [dict(event_id=None, name=e.get("name"), event_key=f"{year}-{e['short_name']}",
                 date=date.fromisoformat(e["date_start"][:10]), done=False)
            for e in evs if not e.get("test") and e.get("short_name") and e.get("date_start")
            and f"{year}-{e['short_name']}" not in have]
    return pd.DataFrame(rows, columns=["event_id", "name", "event_key", "date", "done"])


def decision_times(sch):
    """[(label, t)] UTC-aware: 12:00 UTC the day after each raced points race."""
    out = []
    for i, r in enumerate(sch[sch["done"]].itertuples(), 1):
        t = pd.Timestamp(datetime.combine(r.race_day + timedelta(days=1), datetime.min.time())) + \
            pd.Timedelta(hours=DECISION_HOUR)
        out.append((f"after R{i:02d} {r.name}", t.tz_localize("UTC")))
    return out


# --- the as-of season forecast ---------------------------------------------------------------------------------

def _field(results, sch, cutoff):
    """The riders or drivers of the last race run before the cutoff."""
    done = sch[sch["done"] & (sch["race_day"] < cutoff)]
    last = done["event_id"].iloc[-1]
    return sorted(int(a) for a in results.loc[results["event_id"] == last, "athlete_id"].unique())


def forecaster(engine, sport, year, n_sims=N_SIMS, seed=SEED, model_settings=None, stage_noise=5.0):
    """A function cutoff (date) -> ({athlete_id: P(champion)}, audit) for `year`, as of that date, read-only. The
    frames are read once; each call prices the remaining races and simulates the season."""
    from racinglines.markets import kinds as K
    from racinglines.models.race_model import Event
    from racinglines.pipelines import position_replay as PR
    sp = PR.spec(sport)
    model = PR.model_for(sp)
    data = model.load(engine.url.render_as_string(hide_password=False))
    with engine.connect() as conn:
        sch = schedule(conn, sport, year)
        if sport == "nascar":
            from racinglines.models import nascar_season as NS
            results, _ = NS.load_state(conn, year, competition=sp["competition"], source=sp["source"])
            fmt = NS.FORMATS.get(year)
            if fmt is None:
                raise ValueError(f"no NASCAR championship format for {year}; known: {sorted(NS.FORMATS)}")
        else:
            results = motogp_points(conn, sp, year)
    sprints = sport == "motogp" and bool((results["kind"] == "sprint").any())
    settings = {"seed": seed, **(model_settings or {}), "sims": n_sims}
    if sport == "nascar" and getattr(model, "noise_unit", None) == "places":     # a noise in places, not strength
        noise = NS.estimate_noise(data[(data["season"] >= year - 1) & (pd.to_datetime(data["date"]) <
                                                                     pd.Timestamp(sch["race_day"].iloc[0]))])
        if noise is not None:
            settings["noise"] = round(float(noise), 2)
    st = model.Settings.from_dict(settings)
    hist = model.history(data, st)

    def at(cutoff):
        rng = np.random.default_rng(seed)
        rem = sch[sch["race_day"] >= cutoff].reset_index(drop=True)
        field = _field(results, sch, cutoff)

        def price(r):
            ev = Event(id=f"{year}::{r.event_id}", season=year, cutoff=cutoff, name=str(r.name), info={"field": field})
            return model.price(hist, ev, st, rng)

        if sport == "nascar":
            ran = set(sch.loc[sch["done"] & (sch["race_day"] < cutoff), "event_id"])
            state = NS.standings_now(fmt, results[results["event_id"].isin(ran)],
                                     sch.assign(done=sch["event_id"].isin(ran)))
            race_sims = [price(r) for r in rem.itertuples()]
            remaining = [dict(chase=bool(r.chase), stages=None if pd.isna(r.stages) else int(r.stages))
                         for r in rem.itertuples()]
            ss = NS.simulate_season(fmt, state, remaining, race_sims, rng, stage_noise=stage_noise) if len(rem) else None
        else:
            from racinglines.models import motogp_season as MS
            state = MS.standings_asof(results, cutoff)
            race_sims = [price(r) for r in rem.itertuples()]
            # sprints only when the season's sprint results are stored: otherwise a Grand Prix-only championship on
            # both sides (standings and the rest of the season), and the report says so
            sprint_sims = [price(r) if sprints else None for r in rem.itertuples()]
            ss = MS.simulate_season(state, race_sims, sprint_sims, rng) if len(rem) else None
        if ss is None:                          # the season is over: the standings leader is the champion
            lead = state.sort_values(["points", "wins"], ascending=False)["athlete_id"].iloc[0]
            fair = {int(a): float(a == lead) for a in state["athlete_id"]}
        else:
            fair = dict(zip((int(a) for a in ss.entrants), K.season_fair("champion", ss)))
        fav = max(fair, key=fair.get) if fair else None
        return fair, dict(races_left=len(rem), field=len(field), favourite=fav, favourite_fair=fair.get(fav),
                          leader=int(state["athlete_id"].iloc[0]) if len(state) else None)

    return at, dict(schedule=sch, settings=st.changed(), model=sp["model"],
                    sprints=None if sport != "motogp" else sprints)


def motogp_points(conn, sp, year):
    """Every scored MotoGP round of the season: event_id, date (the round's day: the event's first day + 2 for the
    Grand Prix, + 1 for the sprint), athlete_id, kind ('race' | 'sprint'), position, points (as the source gave them)."""
    df = pd.read_sql(text("""
        SELECT e.id AS event_id, e.start_date AS start, ro.kind, r.athlete_id, r.position,
               COALESCE((r.extra->>'points')::float, 0.0) AS points
        FROM events e JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
        JOIN races ra ON ra.event_id = e.id JOIN rounds ro ON ro.race_id = ra.id AND ro.kind IN ('race', 'sprint')
        JOIN results r ON r.round_id = ro.id
        WHERE co.code = :c AND e.source = :src AND s.year = :y"""), conn,
                     params=dict(c=sp["competition"], src=sp["source"], y=year))
    off = df["kind"].map({"race": int(sp.get("race_day_offset", 2)), "sprint": int(sp.get("race_day_offset", 2)) - 1})
    return df.assign(date=pd.to_datetime(df["start"]) + pd.to_timedelta(off, unit="D")).drop(columns=["start"])


# --- the exchange's champion markets -----------------------------------------------------------------------------

MOTOGP_CHAMPION = re.compile(r"\b(champion|championship|title)\b")
NOT_MOTOGP_CHAMPION = re.compile(r"\b(team|teams|constructor|constructors|manufacturer|moto2|moto3|rookie|"
                                 r"grand prix|gp|sprint|race)\b")


def _year_of(link):
    m = re.search(r"\b(20\d\d)\b", " ".join(str(link.get(k) or "") for k in ("event_title", "question", "event_slug")))
    if m:
        return int(m.group(1))
    m = re.search(r"-(\d\d)\b", str(link.get("event_slug") or ""))             # Kalshi: KXMOTOGP-26
    if m:
        return 2000 + int(m.group(1))
    end = link.get("end_date")
    return None if end is None or pd.isna(end) else pd.Timestamp(end).year


def motogp_champion(link):
    """True for a MotoGP riders' champion contract: Kalshi's KXMOTOGP series, else a listing naming the championship
    (not a team, constructor, Moto2 / Moto3, or a single Grand Prix)."""
    from racinglines.sources.motogp.links import _series
    from racinglines.sources.nascar.identity import norm
    if link.get("exchange") == "kalshi":
        return _series(link) == "KXMOTOGP"
    words = norm(" ".join(str(link.get(k) or "") for k in ("event_title", "question")))
    return bool(MOTOGP_CHAMPION.search(words)) and not NOT_MOTOGP_CHAMPION.search(words)


def champion_links(conn, sport, year, venue):
    """The season's champion contracts on `venue` with the athlete each names (stored, else resolved in memory)."""
    sp = _spec(sport)
    if sport == "nascar":
        from racinglines.pipelines import nascar_season as NSP
        df = NSP.champion_links(conn, year, competition=sp["competition"])
        return df[df["exchange"] == venue].reset_index(drop=True) if len(df) else df
    from racinglines.sources.motogp.links import NOT_A_RIDER, Linker
    from racinglines.sources.nascar.identity import norm
    df = pd.read_sql(text("""SELECT ml.* FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
                             WHERE co.code = :c AND ml.exchange = :x ORDER BY ml.id"""), conn,
                     params=dict(c=sp["competition"], x=venue))
    L, rows = Linker(conn), []
    for link in df.to_dict("records"):
        if not motogp_champion(link) or _year_of(link) != year:
            continue
        ath = link.get("athlete_id")
        if ath is None or pd.isna(ath):
            subject = link.get("group_title") if norm(link.get("group_title")) not in NOT_A_RIDER else link.get("outcome")
            ath = L.resolver(year).driver(subject) if norm(subject) not in NOT_A_RIDER else None
        rows.append(dict(link, athlete_id=None if ath is None or pd.isna(ath) else int(ath)))
    return pd.DataFrame(rows)


def build_markets(conn, links, venue, asof=None):
    """{token_id: SeasonMarket} from the venue's recorded prices, costed as pipelines/season_strategy does (Kalshi's
    taker fee, OG.com's flat fee, Polymarket's half-spread; each plus slippage). OG.com's prices are read by its
    replay venue's rule (an empty book's 0.50 dropped). A market with no stored price is left out."""
    from racinglines.markets import store as MS
    from racinglines.markets.venue_replay import OG
    from racinglines.pipelines import season_strategy as SE
    toks = links["token_id"].tolist() if len(links) else []
    root = MS.root_for(venue)
    if venue == "og":
        px = OG.observed(*(MS.read(conn, name, tokens=toks, root=root) for name in ("prices", "trades", "books")))
        px = px.assign(ts=px["ts"].dt.tz_localize("UTC"))
    else:
        px = MS.read(conn, "prices", tokens=toks, root=root)
    by = dict(tuple(px.groupby("token_id"))) if len(px) else {}
    if venue == "kalshi":
        costs = SE.kalshi_costs(by, toks, asof)
    elif venue == "og":
        costs = {t: OG.fee_per_contract() + SE.SLIPPAGE for t in toks}
    else:
        costs = SE.market_costs(conn, toks)
    out = {}
    for link in links.to_dict("records"):
        g = by.get(link["token_id"])
        if g is None or not len(g):
            continue
        res = link.get("resolved_yes")
        closed = bool(link.get("closed")) and res is not None and not pd.isna(res)
        out[link["token_id"]] = SS.SeasonMarket(
            key=link["token_id"], kind="champion", subject=str(link.get("group_title") or link.get("outcome") or ""),
            ts=g["ts"].to_numpy(), px=g["price"].to_numpy(float), cost=costs[link["token_id"]],
            outcome=bool(res) if closed else None, closed_at=g["ts"].iloc[-1] if closed else None)
    return out


def link_fairs(links, fair):
    """{token_id: fair} for the links whose athlete the forecast has (a "No"-type token is 1 - P)."""
    out = {}
    for link in links.to_dict("records"):
        p = fair.get(link.get("athlete_id"))
        if p is not None:
            out[link["token_id"]] = 1 - p if link.get("invert") else p
    return out


# --- the replay ----------------------------------------------------------------------------------------------------

def run(engine, sport, year, venue="kalshi", params=DEFAULT_PARAMS, n_sims=N_SIMS, seed=SEED, model_settings=None,
        now=None, echo=print):
    """The replay on one venue for one season (read-only). Returns dict(result, hold, decisions, markets, links,
    matched, params)."""
    if sport not in SPORTS:
        raise ValueError(f"season replay covers {SPORTS}, not {sport!r}")
    with engine.connect() as conn:
        links = champion_links(conn, sport, year, venue)
    matched = int(links["athlete_id"].notna().sum()) if len(links) else 0
    echo(f"progress {sport} {venue} {year}: {len(links)} champion contracts, {matched} with a matched athlete")
    at, meta = forecaster(engine, sport, year, n_sims=n_sims, seed=seed, model_settings=model_settings)
    times = decision_times(meta["schedule"])
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    times = [(lab, t) for lab, t in times if t <= now]
    decisions, rows = [], []
    for i, (label, t) in enumerate(times):
        fair, audit = at(t.date())
        decisions.append(dict(t=t, label=label, fairs=link_fairs(links, fair) if len(links) else {}))
        rows.append(dict(label=label, t=t, **audit))
        echo(f"progress {i + 1}/{len(times)} {label}: {audit['races_left']} races left, favourite fair "
             f"{(audit['favourite_fair'] or 0):.3f}")
    with engine.connect() as conn:
        markets = build_markets(conn, links, venue) if len(links) else {}
    marks = pd.date_range(times[0][1].normalize(), now, freq="D") if times else [now]
    res = SS.replay(markets, decisions, params, now=now, marks=marks)
    hold = SS.replay(markets, decisions, SS.SeasonParams(**{**params.__dict__, "mode": "hold"}), now=now, marks=marks)
    return dict(result=res, hold=hold, decisions=pd.DataFrame(rows), markets=len(markets), links=len(links),
                matched=matched, params=params, sport=sport, venue=venue, year=year, model=meta["model"],
                settings=meta["settings"], sprints=meta["sprints"])


def format_report(out):
    r, h = out["result"], out["hold"]
    L = [f"=== {out['sport']} champion replay on {out['venue']}, {out['year']}: {out['links']} contracts "
         f"({out['matched']} matched), {out['markets']} with prices, {len(out['decisions'])} decisions ===",
         f"model {out['model']} {out['settings'] or '(defaults)'}; NOT CALIBRATED (see the module's note)"]
    if out.get("sprints") is False:
        L.append("SPRINT POINTS MISSING: no sprint results stored for the season, so the championship is Grand Prix "
                 "points only (standings and the rest of the season alike); the exchanges price the real one")
    if not out["markets"]:
        L.append("NO TAPE: no champion contract has a stored price on this venue")
        return "\n".join(L)
    for k, v in r["summary"].items():
        L.append(f"  {k:15s} {v:,.2f}" if isinstance(v, float) else f"  {k:15s} {v}")
    L.append(f"  enter at the first decision & hold: {h['summary']['pnl']:+,.2f}")
    if len(r["positions"]):
        L += ["", r["positions"].drop(columns=["key"]).to_string(index=False, float_format="{:.3f}".format)]
    return "\n".join(L)


def write(out, folder):
    """races/decisions.csv, trades.csv, positions.csv, equity.csv, summary.json under `folder`."""
    import json
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out["decisions"].to_csv(folder / "decisions.csv", index=False)
    for name in ("trades", "positions", "equity"):
        df = out["result"].get(name)
        if df is not None and len(df):
            df.to_csv(folder / f"{name}.csv", index=False)
    summary = dict(sport=out["sport"], venue=out["venue"], year=out["year"], links=out["links"], sprints=out.get("sprints"),
                   matched=out["matched"], markets=out["markets"], model=out["model"], settings=out["settings"],
                   params={k: str(v) for k, v in out["params"].__dict__.items()},
                   update=out["result"]["summary"], hold=out["hold"]["summary"])
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return folder
