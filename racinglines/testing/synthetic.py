"""
Synthetic, seeded data for the quick check (`racinglines check`) and quick tests. No
network, no database, no third-party data: shapes match the real inputs so the real
code paths run, values are invented.

    f1_frames()        a fake F1 season in the model's raw-table shape (results, laps,
                       track profiles), as f1 model load_frames returns from SQL
    f1_sprint(...)     the same frames with a Sprint Qualifying and a Sprint added to some events
    f1_schedule()      the season calendar (a few rounds still to race)
    mtb_results_md()   downhill results files in ChronoRace's markdown-table format
    replay_markets()   exchange tapes (prices + trades) for the maker replay
    weekend_markets()  per-stage fair / price / tradeable rows for the weekend taker
    season_markets()   hourly price paths and decisions for the season strategy
"""

from datetime import timedelta

import numpy as np
import pandas as pd

SEED = 7
TEAMS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
F1_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def _venues(n, rng):
    return [dict(slug=f"track-{chr(97 + i)}", base=rng.uniform(78, 98), shares=rng.dirichlet([6, 6, 6]),
                 speeds=rng.uniform(190, 320, 4)) for i in range(n)]


def f1_frames(n_events=6, n_upcoming=2, seed=SEED):
    """(res, laps, prof): the three frames of inputs.laps.load_frames, for a fake season."""
    rng = np.random.default_rng(seed)
    car_a = np.sort(rng.uniform(0.0, 0.010, len(TEAMS)))         # team base deficit (fraction of lap time)
    car_b = rng.normal(0, 0.004, len(TEAMS))                       # sensitivity to fast sectors
    drivers = [dict(athlete_id=100 + 2 * i + k, driver=f"Driver {2 * i + k + 1:02d}", team=TEAMS[i].title(),
                    team_key=TEAMS[i], off=rng.normal(0, 0.002), abbr=f"D{2 * i + k + 1:02d}")
               for i in range(len(TEAMS)) for k in range(2)]
    venues = _venues(n_events + n_upcoming, rng)
    res, laps, prof = [], [], []
    start0 = pd.Timestamp("2026-03-06 10:00")
    for e in range(n_events):
        v, ev_id, day0 = venues[e], 1000 + e, start0 + timedelta(days=14 * e)
        x = float(np.sum(v["shares"] * (v["speeds"][:3] - 250) / 50))
        sessions = {"fp1": day0, "fp2": day0 + timedelta(hours=4), "fp3": day0 + timedelta(days=1),
                    "qual": day0 + timedelta(days=1, hours=4), "race": day0 + timedelta(days=2, hours=3)}
        common = dict(event_id=ev_id, start_date=day0.normalize() + timedelta(days=2), year=2026, series_round=e + 1,
                      event_status="completed", venue=v["slug"], race_id=2000 + e)

        def lap_time(d, factor, sd):
            t = drivers.index(d) // 2
            deficit = car_a[t] + car_b[t] * x + d["off"]
            return v["base"] * factor * (1 + deficit + rng.normal(0, sd))

        def add_laps(rnd, d, times, pit=()):
            for i, lt in enumerate(times, start=1):
                s = lt * v["shares"] * (1 + rng.normal(0, 0.001, 3))
                laps.append(dict(event_id=ev_id, round=rnd, athlete_id=d["athlete_id"], lap=i, lap_time_ms=round(lt * 1000),
                                 s1_ms=round(s[0] * 1000), s2_ms=round(s[1] * 1000), s3_ms=round(s[2] * 1000),
                                 pit_in=i in pit, pit_out=(i - 1) in pit, track_status="1", is_accurate=True, deleted=False,
                                 stint=1 + sum(i > p for p in pit), session_ts=str(sessions[rnd])))

        for rnd in ("fp1", "fp2", "fp3"):
            for d in drivers:
                add_laps(rnd, d, [lap_time(d, 1.02 if i > 3 else 1.0, 0.004) for i in range(10)], pit=(5,))
                res.append(dict(common, round=rnd, athlete_id=d["athlete_id"], driver=d["driver"], position=None,
                                status="OK", time_ms=None, team=d["team"],
                                extra=dict(team_id=d["team_key"], abbreviation=d["abbr"]), session_ts=str(sessions[rnd])))
        best = {}
        for d in drivers:
            ts = [lap_time(d, 1.0, 0.004) for _ in range(3)]
            add_laps("qual", d, ts)
            best[d["athlete_id"]] = min(ts)
        order = sorted(best, key=best.get)
        grid = {a: i + 1 for i, a in enumerate(order)}
        for d in drivers:
            res.append(dict(common, round="qual", athlete_id=d["athlete_id"], driver=d["driver"], position=grid[d["athlete_id"]],
                            status="OK", time_ms=round(best[d["athlete_id"]] * 1000), team=d["team"],
                            extra=dict(team_id=d["team_key"], abbreviation=d["abbr"], q1_ms=round(best[d["athlete_id"]] * 1000)),
                            session_ts=str(sessions["qual"])))
        total, done = {}, {}
        for d in drivers:
            n = 30 if rng.random() > 0.06 else int(rng.integers(5, 25))      # a few DNFs
            ts = [lap_time(d, 1.05 if i == 0 else 1.03, 0.006) + (grid[d["athlete_id"]] * 0.05 if i == 0 else 0)
                  for i in range(n)]
            ts[14:16] = [t + 20 for t in ts[14:16]] if n > 16 else ts[14:16]
            add_laps("race", d, ts, pit=(15,) if n > 16 else ())
            total[d["athlete_id"]], done[d["athlete_id"]] = sum(ts), n
        finish = sorted(total, key=lambda a: (-done[a], total[a]))
        for pos, a in enumerate(finish, start=1):
            d = next(x for x in drivers if x["athlete_id"] == a)
            ok = done[a] == 30
            res.append(dict(common, round="race", athlete_id=a, driver=d["driver"], position=pos if ok else None,
                            status="OK" if ok else "DNF", time_ms=round(total[a] * 1000) if ok else None, team=d["team"],
                            extra=dict(team_id=d["team_key"], abbreviation=d["abbr"], grid=grid[a],
                                       points=F1_POINTS[pos - 1] if ok and pos <= 10 else 0, laps=done[a]),
                            session_ts=str(sessions["race"])))
        gained = np.mean([abs(grid[a] - p) for p, a in enumerate(finish, start=1) if done[a] == 30])
        prof.append(dict(event_id=ev_id, start_date=common["start_date"], venue=v["slug"], ready_ts=str(sessions["race"]),
                         features=dict(lap_s=v["base"], s1_share=v["shares"][0], s2_share=v["shares"][1],
                                       s3_share=v["shares"][2], v_i1=v["speeds"][0], v_i2=v["speeds"][1],
                                       v_fl=v["speeds"][2], v_st=v["speeds"][3], speed_index=float(v["speeds"][:3] @ v["shares"]),
                                       mean_places_gained=float(gained), grid_finish_rank_corr=0.7, street=False,
                                       qual_rain_share=0.0, race_rain_share=0.0)))
    return pd.DataFrame(res), pd.DataFrame(laps), pd.DataFrame(prof)


def f1_sprint(res, laps, prof, event_ids, seed=SEED, sprint_points=(8, 7, 6, 5, 4, 3, 2, 1)):
    """f1_frames' (res, laps, prof) with a sprint weekend's Sprint Qualifying and Sprint added to `event_ids`:
    SQ the evening of FP1's day (laps, and classification rows as FastF1 stores them: no position, DNS), the
    Sprint the next day between FP3 and qualifying (classification, points, grid = the SQ order). New frames;
    the other events' rows are untouched."""
    rng = np.random.default_rng(seed)
    res, laps = res.copy(), laps.copy()
    add_res, add_laps = [], []
    for ev in event_ids:
        r = res[res["event_id"] == ev]
        day0 = pd.Timestamp(r.loc[r["round"] == "fp1", "session_ts"].iloc[0])
        sq_ts, sp_ts = day0 + timedelta(hours=8), day0 + timedelta(days=1, hours=1, minutes=30)
        q = laps[(laps["event_id"] == ev) & (laps["round"] == "qual")]
        rows = r[r["round"] == "race"].drop_duplicates("athlete_id")
        best = {}
        for a, g in q.groupby("athlete_id"):
            ts = g["lap_time_ms"].to_numpy(float) * (1 + rng.normal(0, 0.002, len(g)))
            best[a] = ts.min()
            for i, t in enumerate(ts, start=1):
                add_laps.append(dict(g.iloc[0].to_dict(), round="sprint_qual", lap=i, lap_time_ms=round(t),
                                     session_ts=str(sq_ts)))
        sq_order = {a: i + 1 for i, a in enumerate(sorted(best, key=best.get))}
        score = {a: sq_order.get(a, 21) + rng.normal(0, 2.0) for a in rows["athlete_id"]}
        dnf = {a: rng.random() < 0.05 for a in score}
        finish = sorted(score, key=lambda a: (dnf[a], score[a]))
        for pos, a in enumerate(finish, start=1):
            base = rows[rows["athlete_id"] == a].iloc[0].to_dict()
            ok = not dnf[a]
            ex = dict(team_id=base["extra"]["team_id"], abbreviation=base["extra"]["abbreviation"])
            add_res.append(dict(base, round="sprint_qual", position=None, status="DNS", time_ms=None, extra=dict(ex),
                                session_ts=str(sq_ts)))
            add_res.append(dict(base, round="sprint", position=pos, status="OK" if ok else "DNF", time_ms=None,
                                extra=dict(ex, grid=sq_order.get(a, 0),
                                           points=sprint_points[pos - 1] if ok and pos <= len(sprint_points) else 0),
                                session_ts=str(sp_ts)))
    return (pd.concat([res, pd.DataFrame(add_res).dropna(axis=1, how="all")], ignore_index=True),
            pd.concat([laps, pd.DataFrame(add_laps)], ignore_index=True), prof)


def f1_schedule(n_events=6, n_upcoming=2):
    rng = np.random.default_rng(SEED)
    venues = _venues(n_events + n_upcoming, rng)
    start0 = pd.Timestamp("2026-03-08")
    return pd.DataFrame([dict(round=i + 1, name=f"Grand Prix {i + 1}", official=f"Synthetic Grand Prix {i + 1}",
                              location=v["slug"], date=start0 + timedelta(days=14 * i), sprint=False)
                         for i, v in enumerate(venues)])


def _name(i):
    """A unique, letters-only rider name (the parser normalizes digits away)."""
    return f"{chr(65 + i // 26)}{chr(65 + i % 26)}SON Rider"


def _fmt(t):
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:06.3f}" if m else f"{s:.3f}"


def mtb_results_md(n_events=4, n_riders=30, seed=SEED):
    """{filename: markdown} for n_events x (Men Elite, Men Junior), in ChronoRace's table format."""
    rng = np.random.default_rng(seed)
    out = {}
    skill = {cat: rng.normal(0, 0.008, n_riders) for cat in ("Elite", "Junior")}
    for e in range(n_events):
        slug = f"2026{3 + e:02d}01_mtb"
        base = rng.uniform(170, 240)
        for cat, bib0 in (("Elite", 1000), ("Junior", 3000)):
            lines = [f"# UCI MOUNTAIN BIKE WORLD SERIES - DHI #{e + 1} - Synthetic Venue {e + 1}, 2026, XXX", "",
                     f"Category: Men {cat}", "", f"Source: ChronoRace (prod.chronorace.be), event slug `{slug}`", ""]
            riders = list(range(n_riders))
            for heading, field in (("Timed Training", riders), ("Qualification 1", riders), ("Final", riders[:20])):
                times = {r: base * (1 + skill[cat][r] + rng.normal(0, 0.01)) * (1.02 if cat == "Junior" else 1)
                         for r in field}
                order = sorted(times, key=times.get)
                lines += [f"## {heading}", "",
                          "| Pos | Bib | Rider | Team | Nation | Split 1 | Split 2 | Time | Gap | Status |",
                          "|---|---|---|---|---|---|---|---|---|---|"]
                best = times[order[0]]
                for pos, r in enumerate(order, start=1):
                    t = times[r]
                    gap = "" if pos == 1 else f"+{t - best:06.3f}"
                    name = _name(r + (0 if cat == "Elite" else 100))
                    lines.append(f"| {pos} | {bib0 + r} | {name} |  | XXX | {_fmt(t * 0.3)} | {_fmt(t * 0.65)} | "
                                 f"{_fmt(t)} | {_fmt(t)} | {gap} |  |".replace(f"| {_fmt(t)} | {_fmt(t)} |", f"| {_fmt(t)} |"))
                lines.append("")
            out[f"{slug}_dhi_{cat.lower()}-men.md"] = "\n".join(lines) + "\n"
    return out


def replay_markets(n=6, hours=48, seed=SEED, books=False, depth=(20, 300)):
    """dict(markets, stages) for markets.strategies.maker_replay.replay: one stage, n binary markets.
    books: also a recorded book per market (fill="queue"), one snapshot a minute, 10 levels a
    side from a tick either side of the mid, each level's size uniform in `depth`; trade
    prices are then on the 1c tick, as on Polymarket, so trades can land on a quote's level."""
    from racinglines.markets.strategies import maker_replay as R
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-09-24", tz="UTC").value
    step = int(60e9)
    markets = []
    for i in range(n):
        p0 = rng.uniform(0.1, 0.8)
        mids = np.clip(p0 + np.cumsum(rng.normal(0, 0.004, hours * 60)), 0.02, 0.98)
        ts = t0 + np.arange(hours * 60) * step
        k = int(rng.integers(200, 400))
        tr_ts = np.sort(rng.choice(ts[1:], k, replace=False)) + int(30e9)
        idx = ((tr_ts - t0) // step).astype(int)
        buy = rng.random(k) < 0.5
        px = np.clip(mids[idx] + np.where(buy, 1, -1) * rng.uniform(0.005, 0.03, k), 0.01, 0.99)
        markets.append(R.Market(cond=f"c{i}", kind="race_win", subject=f"outcome {i}", question=f"q{i}",
                                fairs={1: float(np.clip(p0 + rng.normal(0, 0.03), 0.02, 0.98))}, outcome=bool(rng.random() < p0),
                                mid_ts=ts, mid_px=mids, tr_ts=tr_ts, tr_px=px, tr_sz=rng.uniform(5, 80, k), tr_buy=buy))
    if books:                      # its own random stream: the tapes above keep their draws
        brng = np.random.default_rng(seed + 1)
        for mk in markets:
            mk.tr_px = np.round(mk.tr_px, 2)
            mk.bk_ts, mk.bk_bids, mk.bk_asks = mk.mid_ts.copy(), [], []
            for m in mk.mid_px:
                best_bid, best_ask = np.floor(m * 100 - 1e-9) / 100, np.ceil(m * 100 + 1e-9) / 100
                mk.bk_bids.append({round(best_bid - 0.01 * j, 4): float(brng.uniform(*depth))
                                   for j in range(10) if best_bid - 0.01 * j >= 0.01})
                mk.bk_asks.append({round(best_ask + 0.01 * j, 4): float(brng.uniform(*depth))
                                   for j in range(10) if best_ask + 0.01 * j <= 0.99})
    return dict(markets=markets, stages=[dict(run_id=1, start=t0 + 60 * step, end=t0 + (hours - 2) * 60 * step,
                                              session_end=True)])


def weekend_markets(n=20, stages=("pre-weekend", "after FP1", "after FP2", "after Quali"), seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p = rng.uniform(0.05, 0.6)
        rows, fair = [], float(np.clip(p + rng.normal(0, 0.08), 0.01, 0.99))
        for j, lab in enumerate(stages):
            p = float(np.clip(p + rng.normal(0, 0.05), 0.02, 0.98))
            rows.append(dict(label=lab, t=pd.Timestamp("2026-09-24") + timedelta(hours=8 * j), fair=fair, price=p,
                             tradeable=bool(rng.random() > 0.1)))
        out.append(dict(key=f"m{i}", kind="race_win" if i % 2 else "race_podium", subject=f"outcome {i}", stages=rows,
                        outcome=bool(rng.random() < p)))
    return out


def season_markets(n=5, weeks=20, seed=SEED):
    """(markets, decisions) for markets.strategies.season.replay."""
    from racinglines.markets.strategies import season as SS
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-03-01", tz="UTC")
    ts = pd.date_range(t0, periods=weeks * 7 * 24, freq="h")
    probs = rng.dirichlet(np.ones(n) * 2)
    markets = {}
    for i in range(n):
        path = np.clip(probs[i] + np.cumsum(rng.normal(0, 0.003, len(ts))), 0.005, 0.995)
        markets[f"k{i}"] = SS.SeasonMarket(key=f"k{i}", kind="champion", subject=f"Driver {i + 1:02d}",
                                           ts=ts.to_numpy(dtype=object), px=path, cost=0.01)
    decisions = [dict(t=t0 + timedelta(days=14 * w), label=f"after R{w:02d}",
                      fairs={k: float(np.clip(m.px[w * 14 * 24] + rng.normal(0, 0.05), 0.001, 0.999))
                             for k, m in markets.items()})
                 for w in range(weeks // 2)]
    return markets, decisions
