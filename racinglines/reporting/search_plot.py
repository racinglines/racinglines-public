"""Cumulative P&L over the season, top 5 and bottom 5 simulated combos (strategy x settings) per season.
Reads pnl_curves.json (analyze.py), writes pnl_top_bottom.png.   .venv/bin/python .../plot.py"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # reference categorical slots 1-5
LOW = "#8f8d87"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SHORT = {"update": "update", "hold": "hold", "last": "after-quali", "early": "stage-aware", "maker": "maker",
         "maker_flat": "maker-flat", "maker_skew": "maker-skew", "maker_widen": "maker-widen", "maker_all": "maker-all"}


def short(label):
    """Compact settings label for legends."""
    for a, b in (("taker_stages=after FP1,after FP2,after FP3,after SQ,after Sprint,after Quali", "skip-pre"),
                 ("taker_stages=after FP1,after FP2,after SQ,after Sprint", "no-pre/FP3"),
                 ("taker_stages=pre-weekend,after FP1,after FP2,after SQ,after Sprint", "no-FP3"),
                 ("half_life_days=", "hl "), ("max_disagree=", "md "), ("min_edge=", "edge "), (".0,", ","),
                 ("stake_per_edge=", "stake/edge "), ("max_stake=", "max stake ")):
        label = label.replace(a, b)
    return label if len(label) <= 70 else label[:69] + "…"


def main(out_dir):
    out_dir = Path(out_dir)
    curves = json.loads((out_dir / "pnl_curves.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), facecolor=SURF)
    for ax, year in zip(axes, (2025, 2026)):
        cs = [c for c in curves if c["year"] == year]
        uniq = {}                     # identical curves (a setting the strategy ignores) -> one line, simplest label
        for c in sorted(cs, key=lambda c: (c["label"] != "baseline", len(c["label"]))):
            uniq.setdefault((c["strategy"], tuple(round(x, 2) for x in c["weekend_pnl"])), c)
        # default fidelity only (the 16k confirmations and noise replicates are re-draws of the same combos)
        cs = sorted((c for c in uniq.values() if c["settings"]["min_volume_24h"] >= 50 and c["settings"]["sims"] == 4000),
                    key=lambda c: -c["pnl"])
        ax.set_facecolor(SURF)
        n_runs = len({c["run_id"] for c in curves if c["year"] == year and c["settings"]["sims"] == 4000})
        if not cs:
            ax.set_title(f"{year}: no runs yet", color=INK, loc="left")
            continue
        base = next((c for c in cs if c["label"] == "baseline" and c["strategy"] == "maker"), None)
        show = [(c, TOP[i], "-", 2.2, f"{SHORT[c['strategy']]} · {short(c['label'])}") for i, c in enumerate(cs[:5])]
        show += [(c, LOW, (0, (4, 3)), 1.4, f"{SHORT[c['strategy']]} · {short(c['label'])}") for c in cs[-5:] if c not in cs[:5]]
        ax.axhline(0, color=INK2, lw=0.8)
        for c, col, ls, lw, lab in reversed(show):
            x = list(range(0, len(c["cumulative"]) + 1))
            y = [0.0] + c["cumulative"]
            ax.plot(x, y, color=col, ls=ls, lw=lw, solid_capstyle="round",
                    label=f"{lab}   {c['pnl']:+,.0f} · Sharpe {c['sharpe']:.2f} · maxDD {c['max_drawdown']:,.0f}")
        span = max(c["pnl"] for c in cs) - min(c["pnl"] for c in cs) or 1
        placed = []
        for c, col, *_ in show[:5]:                                   # direct labels: top 5 end values
            if any(abs(c["pnl"] - p) < 0.035 * span for p in placed):
                continue
            placed.append(c["pnl"])
            ax.annotate(f"{c['pnl']:+,.0f}", (len(c["cumulative"]), c["cumulative"][-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, len(cs[0]["cumulative"]) + 1.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xlabel("race weekend (round)", color=INK2, fontsize=9)
        ax.set_ylabel("cumulative P&L ($)", color=INK2, fontsize=9)
        what = "historical: live from its first race" if year == 2025 else "season so far"
        ax.set_title(f"{year} ({what}) · {n_runs} runs, {len(cs)} distinct strategy curves", color=INK,
                     fontsize=11, loc="left")
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7.2, frameon=False,
                  labelcolor=INK)
    fig.suptitle("Top 5 (colour) and bottom 5 (grey, dashed) strategy × settings combos, cumulative P&L\n"
                 "default fidelity (4k sims); volume filter kept at >= $50 (looser filters are a thin-market artifact)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.99, 0.965, datetime.now(timezone.utc).strftime("%H:%M UTC"), ha="right", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    dest = out_dir / "pnl_top_bottom.png"
    fig.savefig(dest, dpi=130, facecolor=SURF)
    print(dest)



def recommended(out_dir, picks=(("A", 9), ("C", 24), ("B", 6)), out="pnl_recommended.png"):
    """The report's recommended set (candidate ranks), both seasons, plus the A + C portfolio and the default
    baseline's update taker for reference."""
    from racinglines.pipelines import sweep_settings as SS
    out_dir = Path(out_dir)
    curves = json.loads((out_dir / "pnl_curves.json").read_text())
    cands = {json.loads(f.read_text())["rank"]: json.loads(f.read_text()) for f in (out_dir / "candidates").glob("*.json")}
    base_key = SS.Settings.from_dict().key

    def curve(year, settings, strat):
        k = SS.Settings.from_dict(settings).key
        return next(c for c in curves if c["year"] == year and c["settings_key"] == k and c["strategy"] == strat)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor=SURF)
    for ax, year in zip(axes, (2025, 2026)):
        ax.set_facecolor(SURF)
        ax.axhline(0, color=INK2, lw=0.8)
        series = {}
        for tag, rank in picks:
            d = cands[rank]
            series[tag] = (curve(year, d["settings"] if year == 2026 else d["settings_2025"], d["strategy"]),
                           f"{tag}: #{rank:02d} {SHORT[d['strategy']]} · {SS.Settings.from_dict(d['settings'])['variant'] if year == 2026 else SS.Settings.from_dict(d['settings_2025'])['variant']}")
        a, c = series["A"][0], series["C"][0]
        port = [x + y for x, y in zip(a["weekend_pnl"], c["weekend_pnl"])]
        rows = [(series["A"], TOP[0], "-", 2.2), (series["C"], TOP[1], "-", 2.2), (series["B"], TOP[3], "-", 1.6)]
        base = next(x for x in curves if x["year"] == year and x["settings_key"] == base_key and x["strategy"] == "update")
        cum = [0.0]
        for x in port:
            cum.append(cum[-1] + x)
        ax.plot(range(len(cum)), cum, color=TOP[2], lw=2.8, label=f"A + C portfolio   {cum[-1]:+,.0f}")
        ax.annotate(f"{cum[-1]:+,.0f}", (len(cum) - 1, cum[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
        for (cv, lab), col, ls, lw in rows:
            y = [0.0] + cv["cumulative"]
            ax.plot(range(len(y)), y, color=col, ls=ls, lw=lw,
                    label=f"{lab}   {cv['pnl']:+,.0f} · Sharpe {cv['sharpe']:.2f} · maxDD {cv['max_drawdown']:,.0f}")
        y = [0.0] + base["cumulative"]
        ax.plot(range(len(y)), y, color=LOW, ls=(0, (4, 3)), lw=1.4, label=f"reference: update · default settings   {base['pnl']:+,.0f}")
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xlim(0, len(cum) + 1)
        ax.set_xlabel("race weekend (round)", color=INK2, fontsize=9)
        ax.set_ylabel("cumulative P&L ($)", color=INK2, fontsize=9)
        ax.set_title(f"{year} ({'historical: live from its first race' if year == 2025 else 'season so far'})",
                     color=INK, fontsize=11, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7.6, frameon=False, labelcolor=INK)
    fig.suptitle("Recommended per-event set: A (core taker) + C (maker sleeve); B = stage-aware alternative to A",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dest = out_dir / out
    fig.savefig(dest, dpi=130, facecolor=SURF)
    print(dest)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--recommended", action="store_true")
    args = parser.parse_args()
    main(args.out_dir)
    if args.recommended:
        recommended(args.out_dir)
