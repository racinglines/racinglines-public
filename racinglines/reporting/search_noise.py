"""Error bars for the recommended set: A (#09 update), C (#24 maker) and the default baseline, re-run at several
simulation counts (the seed is fixed at 0, so the count is the only settings-level way to redraw the Monte Carlo;
replicates share their first draws, so this spread is a lower bound on the true simulation noise).
Reads pnl_curves.json (analyze.py); writes noise.csv and prints a summary.   .venv/bin/python .../noise.py"""

import csv
import json
import statistics as stat
from pathlib import Path

NOPRE = ["after FP1", "after FP2", "after FP3", "after SQ", "after Sprint", "after Quali"]
SETS = {  # name -> (strategy, match on the run's settings for each season)
    "A: #09 update": ("update", lambda s, y: s["variant"] == ("gridq+pretrain+reset" if y == 2026 else "gridq+pretrain")
                      and s["min_edge"] == 0.10 and list(s["taker_stages"]) == NOPRE and s["market_kinds"] == DEFAULT_KINDS
                      and s["half_life_days"] == 120.0 and s["stake_per_edge"] == 250.0 and s["max_stake"] == 50.0),
    "C: #24 maker": ("maker", lambda s, y: s["variant"] == "gbm" and s["max_disagree"] == 0.05 and s["size"] == 25.0
                     and s["half_spread"] == 0.02 and s["ridge_team"] == 2.0 and s["ridge_slope"] == 6.0
                     and s["half_life_days"] == 120.0 and s["practice_prior"]),
    "baseline update": ("update", lambda s, y: {k: v for k, v in s.items() if k != "sims"} == BASE),
    "baseline maker": ("maker", lambda s, y: {k: v for k, v in s.items() if k != "sims"} == BASE),
}


def main(out_dir):
    global DEFAULT_KINDS, BASE
    from racinglines.pipelines import sweep_settings as SS
    out_dir = Path(out_dir)
    BASE = {k: v for k, v in SS.Settings.from_dict().to_json().items() if k != "sims"}
    DEFAULT_KINDS = BASE["market_kinds"]
    curves = json.loads((out_dir / "pnl_curves.json").read_text())
    rows = []
    for name, (strat, match) in SETS.items():
        for year in (2026, 2025):
            got = {}
            for c in curves:
                if c["year"] == year and c["strategy"] == strat and match(c["settings"], year):
                    # A carries C's maker settings in some runs (they don't affect the taker): keep one per count
                    got.setdefault(c["settings"]["sims"], c["pnl"])
            if not got:
                continue
            v = list(got.values())
            rows.append(dict(set=name, year=year, n=len(v), sims=",".join(str(k) for k in sorted(got)),
                             mean=stat.mean(v), sd=stat.stdev(v) if len(v) > 1 else 0.0, min=min(v), max=max(v),
                             at_4000=got.get(4000)))
    with open(out_dir / "noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()})
    for r in rows:
        print(f"{r['set']:<16} {r['year']}  n={r['n']:<2} mean {r['mean']:+7.0f}  sd {r['sd']:5.0f}  "
              f"range {r['min']:+6.0f}…{r['max']:+6.0f}  (4k: {r['at_4000'] if r['at_4000'] is None else round(r['at_4000'])})  sims {r['sims']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    main(parser.parse_args().out_dir)
