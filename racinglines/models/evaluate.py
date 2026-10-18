"""
Comparing F1 model runs (docs/f1-roadmap.md, F1-1).

    paired(a, b)          challenger b minus baseline a, per metric and pricing stage:
                          mean difference ± 2 standard errors over the races both scored
                          (negative is better for Brier / log loss)
    reliability(rows)     predicted vs observed frequency in probability bins, per market
    calibration_error(..) the bins' weighted mean |predicted - observed| (ECE)

Inputs are backtest rows (pricing.backtest), e.g. the `events` stored with a saved
backtest run (`racinglines model backtest f1 --save`).
"""

import numpy as np
import pandas as pd

METRICS = ["brier_win", "brier_podium", "brier_top10", "brier_teammate_h2h", "brier_constructor_top",
           "logloss_win", "logloss_podium", "logloss_top10"]
MODES = ["pre_weekend", "pre_quali", "pre_race"]
MODE_LABEL = {"pre_weekend": "Before practice", "pre_quali": "Before qualifying", "pre_race": "After qualifying"}


def load_run(conn, run_id):
    """(params, rows) of a saved backtest run; rows with track features on only."""
    from sqlalchemy import text
    r = conn.execute(text("SELECT params, metrics->'events' FROM model_runs WHERE id = :i AND kind = 'backtest'"),
                     dict(i=int(run_id))).fetchone()
    if r is None:
        raise SystemExit(f"no saved backtest run {run_id}")
    rows = pd.DataFrame(r[1] or [])
    if "track_features" in rows:
        rows = rows[rows["track_features"].astype(bool)]
    return r[0] or {}, rows.reset_index(drop=True)


def paired(a, b, metrics=METRICS, modes=MODES):
    """b - a per metric and mode over races both scored: mean, 2 SE, n, and a verdict
    ('better' / 'worse' when the difference is beyond 2 SE)."""
    key = ["event_id", "mode"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    out = []
    for metric in metrics:
        for mode in modes:
            if f"{metric}_a" not in m:
                continue
            g = m[m["mode"] == mode]
            d = (g[f"{metric}_b"] - g[f"{metric}_a"]).dropna()
            if len(d) < 2:
                continue
            mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
            out.append(dict(metric=metric, mode=mode, n=len(d), a=float(g[f"{metric}_a"].mean()),
                            b=float(g[f"{metric}_b"].mean()), diff=mean, se2=2 * se,
                            verdict="better" if mean + 2 * se < 0 else "worse" if mean - 2 * se > 0 else ""))
    return pd.DataFrame(out)


def reliability(rows, market="win", bins=(0, .02, .05, .1, .2, .35, .5, .7, 1.0001)):
    """Pooled per-driver predictions vs outcomes in probability bins, per mode."""
    out = []
    for mode, g in rows[rows["pred"].notna()].groupby("mode") if "pred" in rows else []:
        p = np.concatenate([np.asarray(x[market], float) for x in g["pred"]])
        y = np.concatenate([np.asarray(x[f"y_{market}"], float) for x in g["pred"]])
        idx = np.digitize(p, bins) - 1
        for i in range(len(bins) - 1):
            sel = idx == i
            if sel.any():
                out.append(dict(mode=mode, market=market, lo=bins[i], hi=min(bins[i + 1], 1.0), n=int(sel.sum()),
                                predicted=float(p[sel].mean()), observed=float(y[sel].mean())))
    return pd.DataFrame(out)


def calibration_error(rel):
    """Weighted mean |predicted - observed| over the bins (per mode and market)."""
    if not len(rel):
        return pd.DataFrame()
    return (rel.assign(w=rel["n"] * (rel["predicted"] - rel["observed"]).abs())
            .groupby(["mode", "market"]).apply(lambda g: g["w"].sum() / g["n"].sum(), include_groups=False)
            .rename("ece").reset_index())


def format_paired(t, digits=4):
    """Markdown table in the docs' style: rows = metric, columns = stage, 'diff ± 2SE'."""
    if not len(t):
        return "(no overlapping races)"
    cell = lambda r: (f"**{r['diff']:+.{digits}f} ± {r['se2']:.{digits}f}**" if r["verdict"] == "better"
                      else f"{r['diff']:+.{digits}f} ± {r['se2']:.{digits}f}" + (" (worse)" if r["verdict"] == "worse" else ""))
    modes = [m for m in MODES if m in set(t["mode"])]
    lines = ["| Metric | " + " | ".join(MODE_LABEL[m] for m in modes) + " |", "|---" * (len(modes) + 1) + "|"]
    for metric, g in t.groupby("metric", sort=False):
        c = {r["mode"]: cell(r) for r in g.to_dict("records")}
        lines.append(f"| {metric} | " + " | ".join(c.get(m, "") for m in modes) + " |")
    n = t.groupby("mode")["n"].max()
    lines.append("")
    lines.append("Races per stage: " + ", ".join(f"{MODE_LABEL[m]} {int(n[m])}" for m in modes)
                 + ". Differences are challenger − baseline; negative is better; bold = beyond 2 SE.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The model x strategy matrix (latest saved run of each kind per variant)
# ---------------------------------------------------------------------------

ACCURACY_COLS = [("pre_weekend", "brier_podium", "Podium, before practice"),
                 ("pre_quali", "brier_podium", "Podium, before quali"),
                 ("pre_race", "brier_win", "Win, after quali"),
                 ("pre_race", "brier_podium", "Podium, after quali"),
                 ("pre_race", "brier_top10", "Top 10, after quali"),
                 ("pre_race", "brier_teammate_h2h", "Teammate h2h, after quali")]
STRATEGIES = [("update", "Taker: update every session"), ("hold", "Taker: enter & hold"),
              ("last", "Taker: after quali only"), ("early", "Taker: stage-aware (stop after FP2)"),
              ("maker", "Maker: conservative (±2¢, the default)"), ("maker_flat", "Maker: flatten before quali"),
              ("maker_skew", "Maker: info-timed skew"), ("maker_widen", "Maker: widen on bad markouts"),
              ("maker_all", "Maker: all three")]


# championship forecasts only simulate future races (no grid yet), so these variants'
# season strategy is identical to another's
def season_same(v):
    """The variant whose championship forecast this one shares: gridq only acts on a known grid."""
    return "+".join(p for p in v.split("+") if p != "gridq") or "baseline"


def plain_sweep(variant):
    """SQL filter: sweeps at default settings apart from the model variant (sweeps saved before the
    settings schema had only the variant and the four taker knobs, which weren't varied)."""
    assert all(ch.isalnum() or ch in "+_" for ch in variant), variant
    return f"AND (params->'settings' IS NULL OR params->>'label' = '{variant}')"


def _latest(conn, kind, variant, extra=""):
    from sqlalchemy import text
    return conn.execute(text(f"""SELECT id, params, metrics FROM model_runs WHERE kind = :k
                                 AND coalesce(params->>'variant', 'baseline') = :v {extra}
                                 ORDER BY id DESC LIMIT 1"""), dict(k=kind, v=variant)).fetchone()


def matrix(conn, variants, year=2026):
    """dict(accuracy, pnl, market, runs) DataFrames/dicts, one row per variant."""
    full = "AND params->>'races' IS NULL AND coalesce(params->>'track', 'on') = 'on'"
    base = _latest(conn, "backtest", "baseline", full)
    base_rows = load_run(conn, base[0])[1] if base else None
    acc, pnl, mkt, runs = [], [], [], {}
    for v in variants:
        bt = _latest(conn, "backtest", v, full)
        sw = _latest(conn, "sweep", v, f"AND (params->>'year')::int = {int(year)} {plain_sweep(v)}")
        ss = _latest(conn, "season_strategy", season_same(v), f"AND (params->>'year')::int = {int(year)}")
        runs[v] = dict(backtest=bt[0] if bt else None, sweep=sw[0] if sw else None, season=ss[0] if ss else None)
        row = dict(variant=v)
        if bt:
            rows = load_run(conn, bt[0])[1]
            diff = paired(base_rows, rows) if base_rows is not None and v != "baseline" else pd.DataFrame()
            for mode, metric, label in ACCURACY_COLS:
                g = rows[rows["mode"] == mode]
                row[label] = float(g[metric].mean()) if metric in g else np.nan
                d = diff[(diff["metric"] == metric) & (diff["mode"] == mode)] if len(diff) else diff
                row[f"{label} vs base"] = d["verdict"].iloc[0] if len(d) else ""
            if v == "baseline":
                g = rows[rows["mode"] == "pre_race"]
                acc.append(dict(variant="grid-only guess", **{"Win, after quali": g["brier_win_grid"].mean(),
                                                               "Podium, after quali": g["brier_podium_grid"].mean(),
                                                               "Top 10, after quali": g["brier_top10_grid"].mean()}))
        acc.append(row)
        prow = dict(variant=v)
        if sw:
            for key, _ in STRATEGIES:
                t = (sw[2].get("totals") or {}).get(key)
                prow[key] = t["pnl"] if t else np.nan
                prow[f"{key}_up"] = f"{t['weekends_up']}/{t['weekends']}" if t else ""
            sc = pd.DataFrame(sw[2].get("scores") or [])
            if len(sc):
                for stage in ("after FP2", "after Quali"):
                    s = sc[(sc["kind"] == "race_win") & (sc["stage"] == stage)]
                    if len(s):
                        mkt.append(dict(variant=v, stage=stage, model=float(s["brier_model"].iloc[0]),
                                        market=float(s["brier_market"].iloc[0])))
        if ss:
            prow["season"] = (ss[2].get("summary") or {}).get("pnl", np.nan)
            prow["season_up"] = "†" if season_same(v) != v else ""
            prow["season_hold"] = (ss[2].get("hold") or {}).get("pnl", np.nan)
        pnl.append(prow)
    return dict(accuracy=pd.DataFrame(acc), pnl=pd.DataFrame(pnl), market=pd.DataFrame(mkt), runs=runs)


def format_matrix(m):
    """Markdown: accuracy table, then the P&L matrix (variants x strategies)."""
    out = []
    a = m["accuracy"]
    labels = [lab for _, _, lab in ACCURACY_COLS]
    mark = {"better": " ▲", "worse": " ▼", "": ""}
    out.append("| Model | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for r in a.to_dict("records"):
        cells = []
        for lab in labels:
            x = r.get(lab)
            cells.append("" if x is None or pd.isna(x) else f"{x:.4f}{mark.get(r.get(f'{lab} vs base', ''), '')}")
        out.append(f"| {r['variant']} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Brier, lower is better. ▲ / ▼: better / worse than baseline beyond 2 standard errors (paired by race).")
    out.append("")
    p = m["pnl"]
    cols = [(k, lab) for k, lab in STRATEGIES if k in p] + ([("season", "Season strategy (titles)")] if "season" in p else [])
    out.append("| Model | " + " | ".join(lab for _, lab in cols) + " |")
    out.append("|---" * (len(cols) + 1) + "|")
    for r in p.to_dict("records"):
        cells = []
        for k, _ in cols:
            x = r.get(k)
            up = r.get(f"{k}_up", "")
            cells.append("" if x is None or pd.isna(x) else f"{x:+,.0f}" + (" †" if up == "†" else f" ({up})" if up else ""))
        out.append(f"| {r['variant']} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("P&L in $, 2026 rounds 1–15 on Polymarket's recorded prices and trades (weekends up in brackets); "
               "season strategy: championship markets through Baku. † same championship forecasts as "
               + " / ".join(f"{v} = {season_same(v)}" for v in p["variant"] if season_same(v) != v)
               + " (future races have no grid yet).")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Strategy P&L by model variant (the Lab's lead table): P&L only, no backtest loads
# ---------------------------------------------------------------------------

SEASON_ROWS = [("season", "Titles: update after every race"), ("ladder", "Titles: checkpoints, all three to date")]


def strategy_pnl(conn, year=2026):
    """dict(variants, rows, runs): one row per strategy, one column per model variant that has a
    saved sweep for `year` (baseline first). Cells: dict(pnl, up) or None."""
    from sqlalchemy import text
    from racinglines.models import variants as V
    found = [r[0] for r in conn.execute(text("""SELECT DISTINCT coalesce(params->>'variant', 'baseline') FROM model_runs
                                                WHERE kind = 'sweep' AND (params->>'year')::int = :y
                                                  AND coalesce(params->>'sport', 'f1') = 'f1'"""), dict(y=year))]
    order = list(V.SWITCHES)
    variants = sorted(found, key=lambda v: (v != "baseline", len(v.split("+")), [order.index(p) if p in order else 99
                                                                                 for p in v.split("+")]))
    ladder = {}
    for (m,) in conn.execute(text("""SELECT metrics FROM model_runs WHERE kind = 'season_checkpoints'
                                     AND (params->>'year')::int = :y ORDER BY id"""), dict(y=year)):
        rows = pd.DataFrame((m or {}).get("rows") or [])
        if len(rows):
            for v, g in rows[rows["window"] == "to date"].groupby("variant"):
                ladder[v] = float(g["pnl"].sum())                       # later runs overwrite earlier ones
    cells, runs = {}, {}
    for v in variants:
        sw = _latest(conn, "sweep", v, f"AND (params->>'year')::int = {int(year)} {plain_sweep(v)}")
        ss = _latest(conn, "season_strategy", season_same(v), f"AND (params->>'year')::int = {int(year)}")
        runs[v] = dict(sweep=sw[0] if sw else None, season=ss[0] if ss else None)
        totals = (sw[2].get("totals") or {}) if sw else {}
        for key, _ in STRATEGIES:
            t = totals.get(key)
            cells[key, v] = dict(pnl=float(t["pnl"]), up=f"{t['weekends_up']}/{t['weekends']}") if t else None
        cells["season", v] = dict(pnl=float(ss[2]["summary"]["pnl"]), up="") if ss and ss[2].get("summary") else None
        lv = ladder.get(season_same(v)) if season_same(v) in ladder else ladder.get(season_same(v).replace("gridq", "grid"))
        cells["ladder", v] = dict(pnl=lv, up="") if lv is not None else None
    out = []
    for key, label in STRATEGIES + SEASON_ROWS:
        vals = [c["pnl"] for v in variants if (c := cells.get((key, v)))]
        if not vals:
            continue
        best = max(vals)
        out.append(dict(key=key, label=label.replace("Taker: ", "").replace("Maker: ", "Maker, "),
                        group="Titles" if key in ("season", "ladder") else ("Maker" if key.startswith("maker") else "Taker"),
                        cells=[dict(c, best=c["pnl"] == best and len(vals) > 1) if (c := cells.get((key, v))) else None
                               for v in variants]))
    return dict(variants=variants, rows=out, runs=runs, year=year)
