"""
The kart pack's data generator (racinglines/demo/packs/kart/pack.toml names it): a synthetic Mario Kart "F1" world
(the pack's world.toml) written as DATA in the shapes the real code reads, never as code paths of its own. It
invents; the real ingest, model and commands do the rest.

    write_pack(out, input)  everything below, into a demo data folder (the store build calls it)
    write_fastf1(root)      raw/f1/fastf1/<year>/<RR>_<session>.{meta.json,results.parquet,laps.parquet} for every
                            raced session (FP1-FP3 or FP1/SQ/Sprint, Q, R), and <year>/schedule.parquet (the FastF1
                            event schedule's columns) for every season, as sources/fastf1/fetch.py writes them
    write_links(out)        archive/markets/<exchange>/links/market_links.parquet: Polymarket and Kalshi quotes on the
                            live rehearsal's and the book's races, in the file `racinglines data pm-links-export`
                            writes (markets/polymarket/links.py), for the real `data pm-links-import` to load

The world: each team's car has a pace (a fraction of a lap) and a slope (fast sectors suit it or not) that drift
from season to season; each driver an offset to the car. A lap's sectors follow the track's shares and trap speeds;
races add a start, a pit stop, safety cars, retirements and the cost of passing (world.toml hold_s). Seeded: the same
world.toml gives the same files byte for byte (no clock is read).
"""

import json
import re
import tomllib
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]
COUNTRY = "Mushroom Kingdom"


def world(path=None):
    """The world's settings (the kart pack's world.toml, or `path`)."""
    p = Path(path) if path else Path(str(files("racinglines.demo"))) / "packs" / "kart" / "world.toml"
    return tomllib.loads(p.read_text())


def slug(name, sep="_"):
    return re.sub(r"[^a-z0-9]+", sep, name.lower()).strip(sep)


def abbr(name):
    letters = re.sub(r"[^A-Za-z]", "", name).upper()
    return letters[:3]


def calendar(w=None):
    """[(year, round, track dict, race day (date), sprint)] for every season, raced or not."""
    w = w or world()
    out = []
    last = pd.Timestamp(w["last_raced"])
    for year in w["seasons"]:
        if str(year) in w.get("season_start", {}):
            day0 = pd.Timestamp(w["season_start"][str(year)])
        else:
            day0 = last - pd.Timedelta(days=14 * (w["raced_2026"] - 1))
        for rnd in range(1, w["rounds"] + 1):
            out.append((year, rnd, w["tracks"][(rnd - 1) % len(w["tracks"])], (day0 + pd.Timedelta(days=14 * (rnd - 1))).date(),
                        rnd in w["sprint_rounds"]))
    return out


def raced(w, year, rnd):
    return year < max(w["seasons"]) or rnd <= w["raced_2026"]


def sessions(day, sprint):
    """[(FastF1 session name, code, UTC start)] of a weekend whose race is on `day` (Friday to Sunday)."""
    d = pd.Timestamp(day)
    fri, sat, sun = d - pd.Timedelta(days=2), d - pd.Timedelta(days=1), d
    if sprint:
        return [("Practice 1", "FP1", fri + pd.Timedelta(hours=11, minutes=30)),
                ("Sprint Qualifying", "SQ", fri + pd.Timedelta(hours=15, minutes=30)),
                ("Sprint", "S", sat + pd.Timedelta(hours=11)),
                ("Qualifying", "Q", sat + pd.Timedelta(hours=15)),
                ("Race", "R", sun + pd.Timedelta(hours=14))]
    return [("Practice 1", "FP1", fri + pd.Timedelta(hours=11, minutes=30)),
            ("Practice 2", "FP2", fri + pd.Timedelta(hours=15)),
            ("Practice 3", "FP3", sat + pd.Timedelta(hours=10, minutes=30)),
            ("Qualifying", "Q", sat + pd.Timedelta(hours=14)),
            ("Race", "R", sun + pd.Timedelta(hours=14))]


class World:
    """The generator's hidden truth for one season: car pace and slope per team (drifting by season)."""

    def __init__(self, w, seed=None):
        self.w = w
        self.seed = w["seed"] if seed is None else seed
        self.teams = {t["id"]: t for t in w["teams"]}
        self.drivers = w["drivers"]
        rng = np.random.default_rng(self.seed)
        self.car = {}
        pace = {t["id"]: t["pace"] for t in w["teams"]}
        slope = {t["id"]: t["slope"] for t in w["teams"]}
        for year in sorted(w["seasons"], reverse=True):            # the newest season is world.toml's; older ones drift
            self.car[year] = {k: (pace[k], slope[k]) for k in pace}
            pace = {k: max(0.0, v + rng.normal(0, 0.0025)) for k, v in pace.items()}
            slope = {k: v + rng.normal(0, 0.0015) for k, v in slope.items()}
        self.dnf = {d["name"]: 0.04 + 0.04 * rng.random() for d in self.drivers}

    def deficit(self, year, d, x):
        """A driver's pace deficit (fraction of a lap) where the sector speed index is x."""
        a, b = self.car[year][d["team"]]
        return a + b * x + d["offset"]


def _stable(v):
    """A stable integer of a key part (Python's hash() is salted per process)."""
    if isinstance(v, (int, np.integer)):
        return int(v)
    return int.from_bytes(str(v).encode()[:8].ljust(8, b"\0"), "little") ^ len(str(v))


def _x(speed):
    return (speed - 250.0) / 50.0


def _lap_sectors(tr, defs, noise):
    """Sector times (s) per driver: shares x lap x (1 + deficit at that sector's speed + noise)."""
    out = []
    for k in range(3):
        out.append(tr["lap_s"] * tr["shares"][k] * (1 + defs[:, k] + noise[:, k]))
    return np.stack(out, axis=1)


def _trap(tr, rng, n):
    return np.array([rng.normal(s, 3.0, n) for s in tr["speeds"]]).T


def _sector_defs(W, year, tr):
    return np.array([[W.deficit(year, d, _x(tr["speeds"][k])) for k in range(3)] for d in W.drivers])


def _driver_cols(W, d, i):
    t = W.teams[d["team"]]
    return dict(DriverNumber=str(d["number"]), Abbreviation=abbr(d["name"]), DriverId=slug(d["name"]),
                FullName=d["name"], CountryCode="MKD", TeamName=t["name"], TeamId=t["id"])


LAP_NAN = dict(PitInTime=np.nan, PitOutTime=np.nan)


def _laps_frame(rows):
    df = pd.DataFrame(rows)
    for c in ("Driver", "DriverNumber", "Team", "Compound", "TrackStatus"):
        df[c] = df[c].astype("string")
    return df


def practice(W, year, tr, n_laps, rng, t0=0.0):
    """(results, laps) of a practice session or Sprint Qualifying run as practice laps: one-lap runs and a long run."""
    defs = _sector_defs(W, year, tr)
    rows, best = [], {}
    for i, d in enumerate(W.drivers):
        t, stint = t0 + 600 + 20 * i, 1
        for lap in range(1, n_laps + 1):
            long_run = lap > n_laps // 3
            stint = 1 if not long_run else 2
            noise = rng.normal(0.004 if long_run else 0.0015, 0.002, (1, 3))
            fuel = 0.012 if long_run else 0.0
            sec = _lap_sectors(tr, defs[i:i + 1] + fuel, noise)[0]
            lt = float(sec.sum())
            sp = _trap(tr, rng, 1)[0]
            rows.append(dict(Driver=abbr(d["name"]), DriverNumber=str(d["number"]), Team=W.teams[d["team"]]["name"],
                             LapNumber=float(lap), Stint=float(stint), LapTime=lt, Sector1Time=sec[0], Sector2Time=sec[1],
                             Sector3Time=sec[2], SpeedI1=sp[0], SpeedI2=sp[1], SpeedFL=sp[2], SpeedST=sp[3],
                             Compound="SOFT" if not long_run else "MEDIUM", TyreLife=float(lap), FreshTyre=lap == 1,
                             PitInTime=np.nan, PitOutTime=np.nan, TrackStatus="1", Position=np.nan, Deleted=False,
                             IsAccurate=True, LapStartTime=t, Time=t + lt))
            t += lt
            best[d["name"]] = min(best.get(d["name"], 1e9), lt)
    order = sorted(W.drivers, key=lambda d: best[d["name"]])
    res = []
    for pos, d in enumerate(order, 1):
        res.append(dict(_driver_cols(W, d, pos), Position=float(pos), ClassifiedPosition="", GridPosition=np.nan,
                        Status="", Points=np.nan, Laps=float(n_laps), Q1=np.nan, Q2=np.nan, Q3=np.nan, Time=np.nan))
    return pd.DataFrame(res), _laps_frame(rows)


def qualifying(W, year, tr, rng):
    """(results, laps, order) of a knockout qualifying: Q1 all, Q2 the best 15, Q3 the best 10, two flying laps each."""
    defs = _sector_defs(W, year, tr)
    form = rng.normal(0, 0.0015, len(W.drivers))                # the weekend's form, shared by a driver's laps
    rows, seg_best = [], {}
    alive = list(range(len(W.drivers)))
    t = 300.0
    for seg, keep in ((1, 15), (2, 10), (3, 0)):
        times = {}
        for i in alive:
            d = W.drivers[i]
            for k in range(2):
                noise = rng.normal(form[i], 0.0012, (1, 3))
                sec = _lap_sectors(tr, defs[i:i + 1] - 0.002 * seg, noise)[0]
                lt = float(sec.sum())
                sp = _trap(tr, rng, 1)[0]
                deleted = bool(rng.random() < 0.03)
                rows.append(dict(Driver=abbr(d["name"]), DriverNumber=str(d["number"]), Team=W.teams[d["team"]]["name"],
                                 LapNumber=float(2 * seg + k - 1), Stint=float(seg), LapTime=lt, Sector1Time=sec[0],
                                 Sector2Time=sec[1], Sector3Time=sec[2], SpeedI1=sp[0], SpeedI2=sp[1], SpeedFL=sp[2],
                                 SpeedST=sp[3], Compound="SOFT", TyreLife=float(k + 1), FreshTyre=k == 0,
                                 PitInTime=np.nan, PitOutTime=np.nan, TrackStatus="1", Position=np.nan, Deleted=deleted,
                                 IsAccurate=True, LapStartTime=t, Time=t + lt))
                if not deleted:
                    times[i] = min(times.get(i, 1e9), lt)
                t += 3.0
        for i in alive:
            seg_best[(seg, i)] = times.get(i, np.nan)
        t += 900.0
        ranked = sorted(alive, key=lambda i: times.get(i, 1e9))
        alive = ranked[:keep] if keep else ranked
        if seg == 3:
            q3 = ranked
    q2 = sorted([i for i in range(len(W.drivers)) if (2, i) in seg_best and i not in q3], key=lambda i: seg_best[(2, i)])
    q1 = sorted([i for i in range(len(W.drivers)) if i not in q3 and i not in q2], key=lambda i: seg_best[(1, i)])
    order = q3 + q2 + q1
    res = []
    for pos, i in enumerate(order, 1):
        d = W.drivers[i]
        res.append(dict(_driver_cols(W, d, pos), Position=float(pos), ClassifiedPosition="", GridPosition=np.nan,
                        Status="", Points=np.nan, Laps=np.nan, Q1=seg_best.get((1, i), np.nan),
                        Q2=seg_best.get((2, i), np.nan), Q3=seg_best.get((3, i), np.nan), Time=np.nan))
    return pd.DataFrame(res), _laps_frame(rows), order


def race(W, year, tr, grid, n_laps, rng, points, dnf_scale=1.0):
    """(results, laps) of a race from a grid (driver indexes in grid order): pace, start, pit stop, safety car,
    retirements, and the cost of passing."""
    n = len(W.drivers)
    x = sum(s * _x(v) for s, v in zip(tr["shares"], tr["speeds"][:3]))
    pace = np.array([W.deficit(year, d, x) for d in W.drivers]) + rng.normal(0, 0.0025, n)
    gpos = np.empty(n, int)
    gpos[grid] = np.arange(n)
    dnf_at = np.full(n, n_laps + 1)
    for i, d in enumerate(W.drivers):
        if rng.random() < W.dnf[d["name"]] * dnf_scale * (1.4 if tr["sc"] > 0.4 else 1.0):
            dnf_at[i] = rng.integers(1, n_laps + 1)
    sc_laps = set()
    if rng.random() < tr["sc"]:
        a = int(rng.integers(3, max(4, n_laps - 4)))
        sc_laps = set(range(a, a + int(rng.integers(2, 5))))
    pit = rng.integers(n_laps // 3, 2 * n_laps // 3 + 1, n) if n_laps >= 15 else np.full(n, -1)
    hold = tr["hold_s"] / max(n_laps, 1)
    cum = gpos * 0.25                                            # the start: a quarter second per grid slot
    rows, t_session = [], 0.0
    alive = np.ones(n, bool)
    laps_done = np.zeros(n, int)
    for lap in range(1, n_laps + 1):
        status = "4" if lap in sc_laps else "1"
        order_before = np.argsort(cum + np.where(alive, 0, 1e6))
        rank_before = np.empty(n, int)
        rank_before[order_before] = np.arange(n)
        noise = rng.normal(0, 0.0035, (n, 3))
        # race pace: the car and driver at this track plus the day's form, on fuel-heavy laps (1.5 % slower)
        sec = _lap_sectors(tr, np.repeat(pace[:, None], 3, axis=1) + 0.015, noise)
        lt = sec.sum(1) + hold * rank_before                     # dirty air: passing costs time, more the further back
        if status == "4":
            leader = lt.min()
            lt = np.full(n, leader * 1.4) + rng.normal(0, 0.3, n)
            gaps = cum - cum[alive].min() if alive.any() else cum
            cum = cum - gaps * 0.7                                # the field bunches up behind the safety car
        is_pit = pit == lap
        lt = lt + is_pit * 21.0
        for i in range(n):
            if not alive[i]:
                continue
            if lap >= dnf_at[i]:
                alive[i] = False
                continue
            cum[i] += lt[i]
            laps_done[i] = lap
        order = np.argsort(np.where(alive | (laps_done == lap), -laps_done * 1e6 + cum, 1e12))
        posn = np.empty(n, int)
        posn[order] = np.arange(1, n + 1)
        for i, d in enumerate(W.drivers):
            if laps_done[i] != lap:
                continue
            sp = _trap(tr, rng, 1)[0]
            rows.append(dict(Driver=abbr(d["name"]), DriverNumber=str(d["number"]), Team=W.teams[d["team"]]["name"],
                             LapNumber=float(lap), Stint=float(1 + (pit[i] < lap)), LapTime=float(lt[i]),
                             Sector1Time=float(sec[i, 0] if status == "1" else lt[i] / 3),
                             Sector2Time=float(sec[i, 1] if status == "1" else lt[i] / 3),
                             Sector3Time=float(lt[i] - (sec[i, 0] + sec[i, 1] if status == "1" else 2 * lt[i] / 3)),
                             SpeedI1=sp[0], SpeedI2=sp[1], SpeedFL=sp[2], SpeedST=sp[3],
                             Compound="MEDIUM" if pit[i] >= lap else "HARD", TyreLife=float(lap if pit[i] >= lap else lap - pit[i]),
                             FreshTyre=False, PitInTime=float(cum[i]) if is_pit[i] else np.nan,
                             PitOutTime=float(cum[i] - lt[i]) if pit[i] == lap - 1 else np.nan, TrackStatus=status,
                             Position=float(posn[i]), Deleted=False, IsAccurate=status == "1" and not is_pit[i] and pit[i] != lap - 1,
                             LapStartTime=float(cum[i] - lt[i]), Time=float(cum[i])))
    fin = [i for i in range(n) if laps_done[i] == n_laps]
    fin.sort(key=lambda i: cum[i])
    out = [i for i in range(n) if laps_done[i] < n_laps]
    out.sort(key=lambda i: (-laps_done[i], cum[i]))
    res = []
    win_t = cum[fin[0]] if fin else np.nan
    for pos, i in enumerate(fin + out, 1):
        d = W.drivers[i]
        ok = i in fin
        res.append(dict(_driver_cols(W, d, pos), Position=float(pos), ClassifiedPosition=str(pos) if ok else "R",
                        GridPosition=float(gpos[i] + 1), Status="Finished" if ok else "Retired",
                        Points=float(points[pos - 1]) if ok and pos <= len(points) else 0.0, Laps=float(laps_done[i]),
                        Q1=np.nan, Q2=np.nan, Q3=np.nan,
                        Time=(float(win_t) if pos == 1 else float(cum[i] - win_t)) if ok else np.nan))
    return pd.DataFrame(res), _laps_frame(rows), [i for i in fin + out]


def _strings(df):
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype("string")
    return df


def _weather(tr, rng):
    rain = float(rng.random() < tr["rain"]) * float(rng.uniform(0.3, 1.0))
    return dict(air_temp=float(rng.normal(24, 4)), track_temp=float(rng.normal(36, 6)), rain_share=rain,
                humidity=float(rng.uniform(35, 80)), wind_speed=float(rng.uniform(0.5, 4)))


def weekend(W, year, rnd, tr, day, sprint):
    """{code: (results, laps, meta)} of one raced weekend."""
    rng = np.random.default_rng([W.seed, year, rnd])
    ses = sessions(day, sprint)
    fmt = "sprint_qualifying" if sprint else "conventional"
    out = {}
    for name, code, start in ses:
        if code.startswith("FP"):
            res, laps = practice(W, year, tr, 12, rng)
            total = None
        elif code == "SQ":
            res, laps, sq_order = qualifying(W, year, tr, rng)
            total = None
        elif code == "Q":
            res, laps, q_order = qualifying(W, year, tr, rng)
            total = None
        elif code == "S":
            total = max(8, W.w["race_laps"] // 3)
            res, laps, _ = race(W, year, tr, sq_order, total, rng, SPRINT_POINTS, dnf_scale=0.5)
        else:
            total = W.w["race_laps"]
            res, laps, _ = race(W, year, tr, q_order, total, rng, RACE_POINTS)
        meta = dict(year=year, round=rnd, session=code, event_name=f"{tr['name']} Grand Prix",
                    official_name=f"{tr['name'].upper()} GRAND PRIX {year} (SYNTHETIC)", country=COUNTRY,
                    location=tr["name"], event_date=str(day), session_date=str(start), event_format=fmt,
                    total_laps=total, weather=_weather(tr, rng), synthetic=True, source="racinglines demo generator")
        out[code] = (_strings(res), _strings(laps), meta)
    return out


SCHEDULE_COLS = ["RoundNumber", "Country", "Location", "OfficialEventName", "EventDate", "EventName", "EventFormat"] + \
    [f"Session{i}{x}" for i in range(1, 6) for x in ("", "DateUtc")]


def schedule(w, year):
    """FastF1's event schedule columns for one season (sources/fastf1/fetch.event_schedule reads it offline)."""
    rows = []
    for y, rnd, tr, day, sprint in calendar(w):
        if y != year:
            continue
        r = dict(RoundNumber=rnd, Country=COUNTRY, Location=tr["name"],
                 OfficialEventName=f"{tr['name'].upper()} GRAND PRIX {year} (SYNTHETIC)",
                 EventDate=pd.Timestamp(day), EventName=f"{tr['name']} Grand Prix",
                 EventFormat="sprint_qualifying" if sprint else "conventional")
        for i, (name, _, start) in enumerate(sessions(day, sprint), 1):
            r[f"Session{i}"], r[f"Session{i}DateUtc"] = name, start
        rows.append(r)
    return pd.DataFrame(rows)[SCHEDULE_COLS]


def write_fastf1(root, w=None, echo=None):
    """Write every raced session and every season's schedule under root (= data/raw/f1/fastf1). Returns the count of
    sessions written. echo(done, total, what) reports progress."""
    w = w or world()
    W = World(w)
    root = Path(root)
    cal = [c for c in calendar(w) if raced(w, c[0], c[1])]
    n = 0
    for k, (year, rnd, tr, day, sprint) in enumerate(cal):
        if echo:
            echo(k, len(cal), f"{year} R{rnd:02d} {tr['name']}")
        d = root / str(year)
        d.mkdir(parents=True, exist_ok=True)
        for code, (res, laps, meta) in weekend(W, year, rnd, tr, day, sprint).items():
            base = d / f"{rnd:02d}_{code}"
            res.to_parquet(base.with_suffix(".results.parquet"), index=False)
            laps.to_parquet(base.with_suffix(".laps.parquet"), index=False)
            base.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
            n += 1
    for year in w["seasons"]:
        (root / str(year)).mkdir(parents=True, exist_ok=True)
        schedule(w, year).to_parquet(root / str(year) / "schedule.parquet", index=False)
    if echo:
        echo(len(cal), len(cal), "files written")
    return n


# ---------------------------------------------------------------------------
# Exchange quotes: what Polymarket and Kalshi would show, from the world's own odds plus an exchange's noise
# ---------------------------------------------------------------------------

def true_odds(W, year, rnd, tr, sims=600):
    """The world's odds for one race (no grid known): win, podium, pole, 2nd place and head-to-heads, by simulating
    the generator's own qualifying and race `sims` times. Data generation, not pricing: the real model never sees it."""
    n = len(W.drivers)
    win, pod, pole, p2 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    ahead = np.zeros((n, n))
    for s in range(sims):
        rng = np.random.default_rng([W.seed, year, rnd, 7, s])
        _, _, order = qualifying(W, year, tr, rng)
        pole[order[0]] += 1
        _, _, fin = race(W, year, tr, order, max(8, W.w["race_laps"] // 3), rng, RACE_POINTS)
        win[fin[0]] += 1
        p2[fin[1]] += 1
        for i in fin[:3]:
            pod[i] += 1
        rank = np.empty(n, int)
        rank[fin] = np.arange(n)
        ahead += rank[:, None] < rank[None, :]
    return win / sims, pod / sims, pole / sims, p2 / sims, ahead / sims


def exchange_quotes(w=None, rounds=None, sims=400, echo=None):
    """[dict(event_key, round, exchange, prediction, driver, opponent, bid, ask, synced_at, token_id, slug, title)]: the
    synthetic exchanges' quotes on the given rounds (default: the live rehearsal's and the book's)."""
    w = w or world()
    W = World(w)
    year = max(w["seasons"])
    rounds = rounds or sorted({w["live_round"], *w["book_rounds"]})
    cal = {c[1]: c for c in calendar(w) if c[0] == year}
    out = []
    names = [d["name"] for d in W.drivers]
    for k, rnd in enumerate(rounds):
        if echo:
            echo(k, len(rounds), f"exchange odds {year}-{rnd:02d}")
        _, _, tr, day, _ = cal[rnd]
        win, pod, pole, p2, ahead = true_odds(W, year, rnd, tr, sims)
        synced = datetime.combine(day, datetime.min.time()) - timedelta(days=3) + timedelta(hours=9)
        key = f"{year}-{rnd:02d}"
        ex_slug = slug(tr["name"], "-")
        for exchange, spread, noise in (("polymarket", 0.01, 0.12), ("kalshi", 0.02, 0.18)):
            rng = np.random.default_rng([W.seed, year, rnd, _stable(exchange)])

            def q(p, pred, i, opp=None, group=None):
                mid = float(np.clip(p * np.exp(rng.normal(0, noise)), 0.005, 0.97))
                bid, ask = max(0.001, round(mid - spread, 3)), min(0.999, round(mid + spread, 3))
                tok = f"demo-{exchange[:2]}-{key}-{pred}-{slug(names[i], '-')}" + (f"-{slug(names[opp], '-')}" if opp is not None else "")
                out.append(dict(event_key=key, round=rnd, exchange=exchange, prediction=pred, driver=names[i],
                                opponent=None if opp is None else names[opp], bid=bid, ask=ask, synced_at=synced,
                                token_id=tok, slug=f"demo-{ex_slug}-gp-{year}-{pred.replace('race_', '')}",
                                title=f"{tr['name']} Grand Prix {year}: {group or pred} (synthetic)"))

            for i in range(len(names)):
                if win[i] > 0.004 or exchange == "polymarket":
                    q(win[i], "race_win", i, group="winner")
                if pod[i] > 0.01:
                    q(pod[i], "race_podium", i, group="podium")
                if exchange == "polymarket" and pole[i] > 0.004:
                    q(pole[i], "race_pole", i, group="pole")
                if exchange == "kalshi" and p2[i] > 0.01:
                    q(p2[i], "race_p2", i, group="2nd place")
            if exchange == "polymarket":
                for t in range(0, len(names), 2):            # teammates' head-to-heads (drivers are listed in pairs)
                    q(ahead[t, t + 1], "race_h2h", t, t + 1, group="head to head")
    return out


LINK_COLS = ["exchange", "market_slug", "question", "condition_id", "token_id", "outcome", "event_slug", "event_title",
             "group_title", "prediction", "params", "last_bid", "last_ask", "last_price", "synced_at", "first_seen_at",
             "note", "race_key", "race_category", "athlete_key", "competition_code", "category_code"]


def links_frame(quotes):
    """The quotes as the rows of market_links.parquet (markets/polymarket/links.py: stable keys, not database ids)."""
    rows = []
    for q in quotes:
        at = pd.Timestamp(q["synced_at"]).tz_localize("UTC")
        rows.append(dict(exchange=q["exchange"], market_slug=q["token_id"],
                         question=f"{q['title']}: {q['driver']}" + (f" vs {q['opponent']}" if q["opponent"] else ""),
                         condition_id=q["token_id"], token_id=q["token_id"], outcome="Yes", event_slug=q["slug"],
                         event_title=q["title"], group_title=q["driver"], prediction=q["prediction"],
                         params=json.dumps({"opponent_key": slug(q["opponent"])}) if q["opponent"] else None,
                         last_bid=q["bid"], last_ask=q["ask"], last_price=round((q["bid"] + q["ask"]) / 2, 4),
                         synced_at=at, first_seen_at=at, note="synthetic: racinglines demo pack kart",
                         race_key=q["event_key"], race_category="DRV", athlete_key=slug(q["driver"]),
                         competition_code="f1_wdc", category_code="DRV"))
    df = pd.DataFrame(rows, columns=LINK_COLS)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    return df


def write_links(out, w=None, sims=400, echo=None):
    """archive/markets/<exchange>/links/market_links.parquet under `out`, one file per exchange. Returns the row count."""
    df = links_frame(exchange_quotes(w, sims=sims, echo=echo))
    for exchange, g in df.groupby("exchange"):
        p = Path(out) / "archive" / "markets" / exchange / "links" / "market_links.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(p, index=False)
    return len(df)


def write_pack(out, input=None, echo=None, seasons=None, quote_sims=400):
    """The kart pack's data under `out` (a demo data folder): the FastF1 files and the exchange links. seasons: only the
    newest n (a smaller store, e.g. the tests'). echo(done, total, what) reports progress. Returns a summary."""
    w = world(input)
    if seasons:
        w = dict(w, seasons=sorted(w["seasons"])[-seasons:])
    n = write_fastf1(Path(out) / "raw" / "f1" / "fastf1", w, echo=echo)
    links = write_links(out, w, sims=quote_sims, echo=echo)
    return dict(sessions=n, links=links, seasons=w["seasons"], weekends=sum(1 for c in calendar(w) if raced(w, c[0], c[1])))
