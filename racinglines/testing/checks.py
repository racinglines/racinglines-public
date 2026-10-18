"""
Quick checks: is the code working, and are the data sources reachable?

    code       the real pipelines on small synthetic data (racinglines.testing.synthetic):
               parse, model, price, backtest, forecast, strategies; sanity assertions only
    endpoints  one tiny request per data source (FastF1 / F1 live timing, ChronoRace,
               Polymarket's Gamma, CLOB and Data APIs)
    database   Postgres reachable and migrated (optional)

Each check returns Check(name, ok, detail, seconds). Used by `racinglines check` and by
tests/test_quick.py. For the full regression suite on real data see docs/testing.md.
"""

import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Check:
    group: str
    name: str
    ok: bool
    detail: str
    seconds: float


def _run(group, name, fn):
    t = time.time()
    try:
        detail = fn() or ""
        return Check(group, name, True, str(detail), time.time() - t)
    except Exception as ex:  # noqa: BLE001  report, don't raise
        last = traceback.extract_tb(ex.__traceback__)[-1]
        return Check(group, name, False, f"{type(ex).__name__}: {ex} ({Path(last.filename).name}:{last.lineno})",
                     time.time() - t)


def _close(x, target, tol, what):
    if abs(x - target) > tol:
        raise AssertionError(f"{what} = {x:.3f}, expected {target} ± {tol}")


# --- code on synthetic data -------------------------------------------------------

def f1_code_checks():
    from racinglines.models.inputs import laps_pricing as run
    from racinglines.testing import synthetic as SY
    state = {}

    def load():
        res, laps, prof = SY.f1_frames()
        state["meas"] = m = run.Measurements.from_frames(res, laps, prof)
        return f"{m.drivers['event_id'].nunique()} events, {len(laps):,} laps, {len(m.practice)} practice rows"

    def leakage():
        m = state["meas"]
        eid = int(m.drivers["event_id"].max())
        s = m.sessions(eid)
        v = m.view(s["qual"] + pd.Timedelta(minutes=30))           # qualifying under way: nothing from it yet
        if ((v.res["event_id"] == eid) & (v.res["round"] == "qual")).any():
            raise AssertionError("a session in progress leaked into the as-of view")
        return "sessions only usable after they end"

    def history():
        state["hist"] = h = run.history(state["meas"])
        return f"{len(h)} training rows"

    def price():
        m = state["meas"]
        eid = int(m.drivers["event_id"].max())
        s = m.sessions(eid)
        summ, ex = run.price_race(m, state["hist"], s["qual"] + pd.Timedelta(hours=2), eid, n_sims=1000,
                                  rng=np.random.default_rng(1))
        _close(summ["win_prob"].sum(), 1.0, 0.02, "sum of win probabilities")
        _close(summ["podium_prob"].sum(), 3.0, 0.1, "sum of podium probabilities")
        if not ex["audit"]["practice_prior"] and ex["audit"]["grid"] != "qualifying order":
            raise AssertionError(f"unexpected audit {ex['audit']}")
        return f"favourite {summ.iloc[0]['driver']} {summ.iloc[0]['win_prob']:.0%}"

    def backtest():
        bt = run.backtest(state["meas"], state["hist"], 2026, 400, True, last_n=2)
        if bt.empty or bt["brier_win"].isna().all():
            raise AssertionError("no backtest rows")
        return f"{len(bt)} race x mode rows, mean win Brier {bt['brier_win'].mean():.3f}"

    def forecast():
        m = state["meas"]
        last = m.sessions(int(m.drivers["event_id"].max()))["race"]
        _, st, ex = run.forecast(m, state["hist"], 2026, cutoff=last + pd.Timedelta(hours=4), n_sims=800,
                                 schedule=SY.f1_schedule(), race_prices=False)
        _close(st["champion_prob"].sum(), 1.0, 0.02, "sum of champion probabilities")
        _close(ex["constructors"]["champion_prob"].sum(), 1.0, 0.02, "sum of constructors' probabilities")
        return f"leader {st.iloc[0]['driver']} {st.iloc[0]['champion_prob']:.0%}"

    return [("load + measurements", load), ("as-of leakage guard", leakage), ("training history", history),
            ("race pricing", price), ("backtest", backtest), ("season forecast", forecast)]


def market_code_checks():
    from racinglines.markets.strategies import maker_replay as R
    from racinglines.markets.strategies import season as SS
    from racinglines.markets.strategies import taker_weekend as RB
    from racinglines.testing import synthetic as SY

    def replay():
        res = R.replay(SY.replay_markets(), R.Params(min_volume_24h=0))
        s = R.summary(res).loc["total"]
        _close(res["fills"]["pnl"].sum(), s["pnl"], 1e-6, "fills P&L vs positions P&L")
        return f"{int(s['fills'])} fills, P&L {s['pnl']:+.2f}"

    def taker():
        tr, per = RB.run_weekend(SY.weekend_markets(), RB.TakerParams())
        s = RB.summarize(tr, per)
        return f"{s['trades']} trades, P&L {s['pnl']:+.2f}"

    def season():
        mk, dec = SY.season_markets()
        p = SS.SeasonParams(capital=300)
        r = SS.replay(mk, dec, p, now=pd.Timestamp("2026-07-20", tz="UTC"))
        if r["summary"]["capital_used"] > p.capital + 1e-6:
            raise AssertionError("capital cap exceeded")
        return f"{r['summary']['trades']} trades, P&L {r['summary']['pnl']:+.2f}"

    return [("maker replay", replay), ("weekend taker", taker), ("season strategy", season)]


def mtb_code_checks():
    from racinglines.models.inputs import run_times as P
    from racinglines.sources.chronorace.parse import parse_markdown_tables_file
    from racinglines.testing import synthetic as SY
    state = {}

    def parse():
        d = Path(tempfile.mkdtemp(prefix="racinglines-check-"))
        for name, text in SY.mtb_results_md().items():
            (d / name).write_text(text)
        raw = pd.concat([pd.DataFrame(parse_markdown_tables_file(f)) for f in sorted(d.glob("*.md"))], ignore_index=True)
        state["raw"] = raw[raw["round"].isin(P.RUN_WEIGHTS)]
        return f"{len(raw)} rows, {raw['rider_id'].nunique()} riders, rounds {sorted(raw['round'].unique())}"

    def model():
        m = P.fit_season_model(state["raw"], category="ME")
        return f"sigma {m['sigma']:.4f}, {len(m['mu'])} riders"

    def forecast():
        tgt = P.select_target(state["raw"], 2026, "ME")
        _, _, _, st = P.forecast_season(state["raw"], tgt, n_remaining=2, n_sims=400, rng=np.random.default_rng(2))
        _close(st["champion_prob"].sum(), 1.0, 0.02, "sum of champion probabilities")
        return f"leader {st.sort_values('champion_prob').iloc[-1]['champion_prob']:.0%}"

    def backtest():
        tgt = P.select_target(state["raw"], 2026, "ME")
        done = tgt[tgt["event_id"].isin(P.completed_events(tgt))]
        _, reports, _, _ = P.backtest_season(state["raw"], done, n_holdout=1, n_sims=300, rng=np.random.default_rng(3))
        return f"{len(reports)} held-out round(s) scored"

    return [("parse results tables", parse), ("season model", model), ("season forecast", forecast),
            ("backtest", backtest)]


def run_code(sports=("f1", "mtb_dh")):
    groups = []
    if "f1" in sports:
        groups += [("f1 code", f1_code_checks()), ("markets code", market_code_checks())]
    if "mtb_dh" in sports:
        groups += [("mtb_dh code", mtb_code_checks())]
    return [_run(g, name, fn) for g, checks in groups for name, fn in checks]


# --- data endpoints ---------------------------------------------------------------

def _get(url, **params):
    import httpx
    from racinglines.sources import http
    with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": "racinglines-check"}) as c:
        r = http.get(c, url, params=params, tries=2)
    r.raise_for_status()
    return r


def f1_endpoint_checks(year=None):
    year = year or pd.Timestamp.now(tz="UTC").year

    def schedule():
        import logging

        import fastf1

        from racinglines.sources.fastf1.fetch import CACHE
        logging.getLogger("fastf1").setLevel(logging.ERROR)
        CACHE.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(CACHE))
        s = fastf1.get_event_schedule(year, include_testing=False)
        return f"{len(s)} rounds in {year}"

    def live_timing():
        r = _get(f"https://livetiming.formula1.com/static/{year}/Index.json")
        return f"{len(r.json().get('Meetings', []))} meetings indexed"

    return [("FastF1 schedule", schedule), ("F1 live-timing archive", live_timing)]


def polymarket_endpoint_checks():
    state = {}

    def gamma():
        ev = _get("https://gamma-api.polymarket.com/events", tag_slug="f1", closed="false", limit=5).json()
        mk = next(m for e in ev for m in e.get("markets", []) if m.get("clobTokenIds"))
        import json
        state["token"], state["cond"] = json.loads(mk["clobTokenIds"])[0], mk["conditionId"]
        return f"{len(ev)} open F1 events"

    def clob():
        h = _get("https://clob.polymarket.com/prices-history", market=state["token"], interval="1d", fidelity=60).json()
        b = _get("https://clob.polymarket.com/book", token_id=state["token"]).json()
        return f"{len(h.get('history', []))} price points, book with {len(b.get('bids', []))} bids"

    def data_api():
        t = _get("https://data-api.polymarket.com/trades", market=state["cond"], limit=5).json()
        return f"{len(t)} recent trades"

    return [("Polymarket events (Gamma)", gamma), ("Polymarket prices + book (CLOB)", clob),
            ("Polymarket trades (Data API)", data_api)]


def mtb_endpoint_checks(slug="20260925_mtb"):
    def chronorace():
        from racinglines.sources.chronorace.download import CMS_URL
        r = _get(CMS_URL.format(slug=slug))
        return f"event {slug}: {len(r.content):,} bytes"

    def wikipedia():
        year = pd.Timestamp.now(tz="UTC").year
        _get(f"https://en.wikipedia.org/wiki/{year}_UCI_Mountain_Bike_World_Cup")
        return f"{year} calendar page"

    return [("ChronoRace results API", chronorace), ("Wikipedia calendar (event discovery)", wikipedia)]


def run_endpoints(sports=("f1", "mtb_dh")):
    groups = []
    if "f1" in sports:
        groups += [("f1 endpoints", f1_endpoint_checks()), ("polymarket endpoints", polymarket_endpoint_checks())]
    if "mtb_dh" in sports:
        groups += [("mtb_dh endpoints", mtb_endpoint_checks())]
    return [_run(g, name, fn) for g, checks in groups for name, fn in checks]


def run_database():
    def db():
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        from sqlalchemy import text

        from racinglines.db.config import database_url, get_engine
        from racinglines.paths import ROOT
        with get_engine().connect() as c:
            rev = c.execute(text("SELECT version_num FROM alembic_version")).scalar()
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "migrations"))
        head = ScriptDirectory.from_config(cfg).get_current_head()
        url = database_url().split("@")[-1]
        if rev != head:
            raise AssertionError(f"{url}: schema at {rev}, latest is {head} (run: racinglines db init)")
        return f"{url}, schema up to date"

    return [_run("database", "Postgres", db)]
