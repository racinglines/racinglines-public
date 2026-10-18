"""
Backtest analytics for a search (reads the finished jobs' saved sweep runs from the database):

    .venv/bin/python data/runs/search/params-4h/analyze.py [search-name] [--write-candidates]

Writes, next to this file:
  pnl_curves.json   every run x strategy: per-weekend P&L and cumulative P&L (2025 = historical
                    "live from its first race", 2026 = the season so far)
  all_sims.json     every simulated config (job, run, full settings, re-run command)
  stats.csv         one row per run x strategy: total P&L, weekends up, max drawdown, Sharpe, the
                    difference from the same season's baseline, and the full settings key
  ranking.csv       settings x strategy combos: 2026, 2025, label, score (= 2026 + 2025 P&L)
  candidates/NN-*.json  the top 25 combos: complete settings (the JSON sim definition: every
                    setting of racinglines/pipelines/sweep_settings.py), strategy, both seasons'
                    stats and the command that re-runs each season
  candidates.toml   the same 25 as [[candidate]] entries (paste into a queue file)

Sharpe is per weekend (mean / std of weekend P&L) x sqrt(24), i.e. a one-season figure.
Max drawdown is the largest peak-to-trough fall of cumulative P&L, starting from 0 before round 1.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from sqlalchemy import text

from racinglines import paths
from racinglines.db.config import get_engine
from racinglines.models import evaluate as EV
from racinglines.pipelines import sweep_settings as SS
from racinglines.reporting.metrics import curve_stats

OUT = None
TOP = 25
PER_STRATEGY = 6                  # keep the candidate list diverse across strategies
# noise floors ($ over a season) from re-running the same combo at 8k / 16k simulations: differences
# smaller than these are not evidence either way
NOISE = {"taker": 150.0, "maker": 350.0}
DEFAULT_SIMS = SS.BY_NAME["sims"].default
CONFIRM_SIMS = 16000
DEFAULT_KEY = SS.Settings.from_dict().key


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze saved F1 sweep-search runs.")
    parser.add_argument("search", help="Search name under data/runs/search/.")
    parser.add_argument("--write-candidates", action="store_true")
    args = parser.parse_args(argv)
    global OUT
    OUT = paths.runs("search", args.search)
    state = json.loads((OUT / "state.json").read_text())
    done = {k: v for k, v in state.items() if v.get("run_id") and v["kind"] == "sweep" and not v.get("rounds")}
    with get_engine().connect() as c:
        rows = c.execute(text("SELECT id, metrics FROM model_runs WHERE id = ANY(:ids)"),
                         dict(ids=[v["run_id"] for v in done.values()])).fetchall()
    metrics = {r[0]: r[1] for r in rows}
    curves, stats = [], []
    for jid, v in done.items():
        m = metrics.get(v["run_id"])
        if not m:
            continue
        st = SS.Settings.from_dict(v["settings"])
        wk = sorted(m.get("weekends") or [], key=lambda w: w["round"])
        for key, label in EV.STRATEGIES:
            pn = [float(w.get(f"{key}_pnl") or 0.0) for w in wk]
            s, cum = curve_stats(pn)
            rec = dict(job=jid, run_id=v["run_id"], year=v["year"], settings_key=st.key, label=st.label(),
                       strategy=key, strategy_label=label, note=v.get("note", ""), **s)
            stats.append(rec)
            curves.append(dict(rec, rounds=[w["round"] for w in wk], events=[w["event"] for w in wk],
                               weekend_pnl=pn, cumulative=cum, settings=st.to_json()))
    # each run is compared with the baseline at the same fidelity (simulations per stage)
    sims_of = {jid: v["settings"]["sims"] for jid, v in done.items()}
    base_keys = {SS.Settings.from_dict({"sims": n}).key: n for n in set(sims_of.values())}
    base = {(r["year"], base_keys[r["settings_key"]], r["strategy"]): r for r in stats if r["settings_key"] in base_keys}
    for r in stats + curves:
        b = base.get((r["year"], sims_of[r["job"]], r["strategy"]))
        r["vs_baseline"] = r["pnl"] - b["pnl"] if b else None
    (OUT / "pnl_curves.json").write_text(json.dumps(curves, default=str))
    # every simulated config, reproducible: full settings + the command that re-runs it (seeded simulations)
    sims = []
    for jid, v in sorted(done.items(), key=lambda kv: kv[1]["run_id"]):
        st = SS.Settings.from_dict(v["settings"])
        cmd = ["racinglines", "strategy", "sweep", "f1", "--variant", st["variant"], "--year", str(v["year"]), "--no-fetch", "--save"]
        sims.append(dict(job=jid, run_id=v["run_id"], year=v["year"], note=v.get("note", ""), label=st.label(),
                         settings_key=st.key, model_key=st.model_key, changed=st.changed(), settings=st.to_json(),
                         command=" ".join(cmd + [f"'{a}'" if " " in a else a for a in st.argv()])))
    (OUT / "all_sims.json").write_text(json.dumps(sims, indent=1, default=str) + "\n")
    cols = ["year", "strategy", "label", "pnl", "vs_baseline", "weekends_up", "weekends", "max_drawdown", "sharpe",
            "mean_weekend", "sd_weekend", "run_id", "job", "settings_key", "note"]
    with open(OUT / "stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(stats, key=lambda r: (r["year"], r["strategy"], -r["pnl"])):
            w.writerow({k: (round(r[k], 2) if isinstance(r.get(k), float) else r.get(k)) for k in cols})

    # combos: the same settings with the season-specific switch (reset) mapped out for 2025, where it's inert
    def model_twin(st):
        return SS.Settings.from_dict(dict(st, variant="+".join(p for p in st["variant"].split("+") if p != "reset")
                                          or "baseline")).key
    by = {}
    for r in stats:
        by.setdefault((r["year"], r["settings_key"], r["strategy"]), r)
    sets = {r["settings_key"]: SS.Settings.from_dict(next(c["settings"] for c in curves if c["settings_key"] == r["settings_key"]))
            for r in stats}
    rank = []
    for (year, key, strat), r in by.items():
        if year != 2026:
            continue
        st = sets[key]
        if st["sims"] != DEFAULT_SIMS:                      # higher-fidelity runs are confirmations, not combos
            continue
        r25 = by.get((2025, key, strat)) or by.get((2025, model_twin(st), strat))
        # label (held-out rule): better than the baseline in 2026 and, where run, in 2025
        d26, d25 = r["vs_baseline"], (r25 or {}).get("vs_baseline")
        if st["min_volume_24h"] < 50:
            label = "suspect (volume filter loosened: stale thin-market fills)"
        elif key == DEFAULT_KEY:
            label = "baseline"
        elif r25 is None:
            label = "2026 only (no 2025 check)"
        else:
            n = NOISE["maker" if strat.startswith("maker") else "taker"]
            up26, up25, dn25 = d26 > n, d25 > n, d25 < -n
            if up26 and up25:
                label = "robust"
            elif up26:
                label = "2026-specific" + (" (2025 neutral)" if not dn25 else "")
            elif abs(d26) <= n and up25:
                label = "2025-led (2026 within noise)"
            else:
                label = "not better"
        # confirmation: the same combo at 16k simulations, against the 16k baseline
        k16 = SS.Settings.from_dict(dict(st, sims=CONFIRM_SIMS)).key
        c26 = by.get((2026, k16, strat))
        c25 = None
        if r25:
            st25 = sets[r25["settings_key"]]
            c25 = by.get((2025, SS.Settings.from_dict(dict(st25, sims=CONFIRM_SIMS)).key, strat))
        e26, e25 = (c26 or {}).get("vs_baseline"), (c25 or {}).get("vs_baseline")
        if e26 is None or (label == "robust" and e25 is None):
            confirmed = "not run"
        elif label == "robust":
            confirmed = "yes" if e26 > 0 and e25 > 0 else "no"
        elif label.startswith("2026-specific"):
            confirmed = "yes" if e26 > 0 else "no"
        elif label.startswith("2025-led"):
            confirmed = "yes" if e25 is not None and e25 > 0 and e26 > -NOISE["maker"] else "no"
        else:
            confirmed = ""
        # score: P&L over both windows (38 weekends); a combo without a 2025 check sorts last
        score = r["pnl"] + r25["pnl"] if r25 else r["pnl"] - 1e6
        rank.append(dict(settings_key=key, label_settings=st.label(), strategy=strat, verdict=label, score=score,
                         pnl_2026=r["pnl"], vs_base_2026=d26, sharpe_2026=r["sharpe"], dd_2026=r["max_drawdown"],
                         up_2026=f"{r['weekends_up']}/{r['weekends']}",
                         pnl_2025=(r25 or {}).get("pnl"), vs_base_2025=d25, sharpe_2025=(r25 or {}).get("sharpe"),
                         dd_2025=(r25 or {}).get("max_drawdown"),
                         up_2025=f"{r25['weekends_up']}/{r25['weekends']}" if r25 else "",
                         run_2026=r["run_id"], run_2025=(r25 or {}).get("run_id"),
                         settings_2025=(sets.get(r25["settings_key"]).to_json() if r25 else None),
                         confirmed_16k=confirmed, pnl16_2026=(c26 or {}).get("pnl"), vs16_2026=e26,
                         pnl16_2025=(c25 or {}).get("pnl"), vs16_2025=e25))
    rank.sort(key=lambda x: -x["score"])
    with open(OUT / "ranking.csv", "w", newline="") as f:
        cols = ["score", "verdict", "strategy", "label_settings", "pnl_2026", "vs_base_2026", "sharpe_2026", "dd_2026",
                "up_2026", "pnl_2025", "vs_base_2025", "sharpe_2025", "dd_2025", "up_2025", "confirmed_16k", "pnl16_2026",
                "vs16_2026", "pnl16_2025", "vs16_2025", "run_2026", "run_2025",
                "settings_key"]
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rank:
            w.writerow({k: (round(r[k], 2) if isinstance(r.get(k), float) else r.get(k)) for k in cols})

    if not args.write_candidates:
        # the candidate list is frozen at the report's version (REPORT.md section 3.3): later runs would
        # otherwise reshuffle ranks that the report, the charts and the queue's [[candidate]] entries refer to
        print(f"{len(stats)} run x strategy rows, {len(rank)} 2026 combos; candidates/ left as is "
              f"(pass --write-candidates to regenerate) -> {OUT}")
        return
    cdir = OUT / "candidates"
    cdir.mkdir(exist_ok=True)
    for f in cdir.glob("*.json"):
        f.unlink()
    toml = ["# Top combos of this search as [[candidate]] entries (generated by analyze.py).", ""]
    # the volume filter is a realism guard (thin markets fill at stale prices): combos that loosen it are
    # reported in ranking.csv but never become candidates
    ok, seen = [], set()
    # identical results (a setting the strategy ignores) -> one candidate, the one with the fewest changed settings
    for x in sorted(rank, key=lambda x: (-round(x["score"], 2), len(sets[x["settings_key"]].changed()))):
        sig = (x["strategy"], round(x["pnl_2026"], 2), round(x["pnl_2025"] or 0, 2))
        if x["pnl_2025"] is None or sets[x["settings_key"]]["min_volume_24h"] < 50 or sig in seen:
            continue
        if sum(y["strategy"] == x["strategy"] for y in ok) >= PER_STRATEGY:
            continue
        seen.add(sig)
        ok.append(x)
    for i, r in enumerate(ok[:TOP], 1):
        st = sets[r["settings_key"]]
        name = f"{r['strategy']} · {st.label()}"
        fn = f"{i:02d}-{r['strategy']}-{r['settings_key']}.json"
        cmd = lambda y, s: " ".join(["racinglines", "strategy", "sweep", "f1", "--variant", s["variant"], "--year", str(y),
                                     "--no-fetch", "--save"] + [f"'{a}'" if " " in a else a for a in s.argv()])
        doc = dict(rank=i, name=name, strategy=r["strategy"], verdict=r["verdict"], settings=st.to_json(),
                   settings_key=st.key, changed={k: v for k, v in st.to_json().items() if st[k] != SS.BY_NAME[k].default},
                   stats={k: r[k] for k in r if k.endswith(("_2026", "_2025")) and not k.startswith("settings")},
                   settings_2025=r["settings_2025"],
                   rerun={"2026": cmd(2026, st),
                          "2025": cmd(2025, SS.Settings.from_dict(r["settings_2025"])) if r["settings_2025"] else None},
                   queue_job={k: v for k, v in st.to_json().items() if st[k] != SS.BY_NAME[k].default} | {"year": 2026})
        (cdir / fn).write_text(json.dumps(doc, indent=1, default=str) + "\n")
        why = (f"{r['verdict']}: 2026 {r['pnl_2026']:+,.0f} ({r['vs_base_2026']:+,.0f} vs baseline, Sharpe "
               f"{r['sharpe_2026']:.2f}, max DD {r['dd_2026']:,.0f}); 2025 {r['pnl_2025']:+,.0f} "
               f"({r['vs_base_2025']:+,.0f} vs baseline, Sharpe {r['sharpe_2025']:.2f}); 16k-sim check: "
               + (f"{r['confirmed_16k']} (2026 {r['vs16_2026']:+,.0f}" + (f", 2025 {r['vs16_2025']:+,.0f}" if r['vs16_2025'] is not None else "")
                  + " vs the 16k baseline)" if r['vs16_2026'] is not None else r['confirmed_16k']))
        toml += ["[[candidate]]", f'name = "#{i:02d} {name}"', "year = 2026", f'strategy = "{r["strategy"]}"',
                 f'why = "{why}"']
        for k, v in st.to_json().items():
            if st[k] != SS.BY_NAME[k].default or k == "variant":
                toml.append(f"{k} = " + (f'"{",".join(v)}"' if isinstance(v, list) else json.dumps(v)))
        toml.append("")
    (OUT / "candidates.toml").write_text("\n".join(toml))
    print(f"{len(stats)} run x strategy rows, {len(rank)} 2026 combos, "
          f"{min(TOP, sum(x['pnl_2025'] is not None for x in rank))} candidates -> {OUT}")


if __name__ == "__main__":
    main()
