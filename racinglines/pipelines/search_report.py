"""
The analysis of a finished search (racinglines/pipelines/search.py), with the checks the params-4h
search showed a backtest needs (docs/backtest-core.md, Part 1):

  - held-out rule: a combo is labelled against the baseline in the target season AND the held-out ones,
    beyond a noise floor: robust / target only / held-out-led / not better;
  - noise floor: the spread of seed replicates (jobs that differ only in `seed`) where the search ran
    them, else the configured floor per strategy family;
  - baselines at the same fidelity: each run is compared with the default settings at its own
    simulation count, over the same rounds;
  - confirmation: the same combo at `confirm_sims` simulations, when the search ran it;
  - concentration: P&L without the best event (and which event that was) and without the best two;
  - shape: steady (still up without its best two events in every season), concentrated (up in every season,
    but not without its best two in one of them), mixed (up in one season, down in another) or losing (down in
    every season);
  - stable candidate ids: `<strategy>-<settings_key>`, never a position in a list;
  - finished jobs only: a job still running (or failed) is never read.

    racinglines strategy search-report f1 sweeps/params-4h.toml

Configuration: an optional [report] table in the queue file (defaults below), and [report.<sport>] for
another sport's jobs. Outputs, next to the search's state.json: stats.csv, ranking.csv, pnl_curves.json,
candidates/<id>.json, candidates.toml, report.md; another sport's in a folder named after it.

Any sport: sweeps (F1's, and any other sport's in a folder <sport>-sweep, its settings under [report.<sport>-sweep])
are scored by each strategy's P&L per weekend; model-only walk-forwards (downhill)
by each market kind's score per event (−1000 × log loss, higher is better). The labels, noise floor,
baselines, confirmation and ids work the same on either.
"""

import csv
import json
import math

import pandas as pd

from racinglines.pipelines import sweep_settings as SS

DEFAULTS = dict(target=2026, holdout=[2025], confirm_sims=16000, rank_sims=None, top=25, season_events=24,
                noise=dict(taker=150.0, maker=350.0, model=25.0), value="pnl", unit="P&L")
# another sport's defaults, under [report.<sport>] in the queue; "global" is the results model's (any sport)
SPORT_DEFAULTS = {"mtb_dh": dict(confirm_sims=20000, season_events=10, value="score", unit="score"),
                  "global": dict(confirm_sims=8000, season_events=20, value="score", unit="score")}


def family(strategy):
    if strategy.startswith("race_"):              # a market kind scored model-only
        return "model"
    return "maker" if strategy.startswith("maker") else "taker"


def curve_stats(pnls, season_events):
    """Total, weekends up, max drawdown (peak to trough from 0 before the first event), Sharpe (mean / sd of
    event P&L x sqrt(events per season)), and the total without the best event and without the best two."""
    cum, peak, dd, c = [], 0.0, 0.0, 0.0
    for x in pnls:
        c += x
        cum.append(c)
        peak = max(peak, c)
        dd = max(dd, peak - c)
    n = len(pnls)
    mean = c / n if n else 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnls) / (n - 1)) if n > 1 else 0.0
    best = max(range(n), key=lambda i: pnls[i]) if n else None
    top2 = sum(sorted(pnls, reverse=True)[:2])
    return dict(pnl=c, weekends=n, weekends_up=sum(x > 0 for x in pnls), max_drawdown=dd,
                sharpe=(mean / sd * math.sqrt(season_events)) if sd > 0 else 0.0, mean_weekend=mean,
                sd_weekend=sd, pnl_without_best=c - pnls[best] if n else 0.0, pnl_without_best2=c - top2,
                best_index=best), cum


def shape(pnls, without_best2):
    """steady / concentrated / mixed / losing from each season's P&L and P&L without its best two events
    ("" if a season is missing). These describe the record, not the risk."""
    if any(v is None for v in list(pnls) + list(without_best2)):
        return ""
    if all(v <= 0 for v in pnls):
        return "losing"
    if any(v <= 0 for v in pnls):
        return "mixed"
    return "steady" if all(v > 0 for v in without_best2) else "concentrated"


def _config(cfg):
    out = dict(DEFAULTS, **(cfg or {}))
    out["noise"] = dict(DEFAULTS["noise"], **(cfg or {}).get("noise", {}))
    return out


def _without(st, *names):
    return {k: v for k, v in st.items() if k not in names}


def stats(jobs, metrics, strategies, cfg=None, cls=SS.Settings):
    """jobs: finished sweep jobs (state.json values: year, rounds, settings, run_id, id, note).
    metrics: {run_id: the saved sweep's metrics}. strategies: [(key, label)]. cls: the sport's settings.
    Returns (stats rows, curves): one row per job x strategy."""
    cfg = _config(cfg)
    rows, curves = [], []
    for j in jobs:
        m = metrics.get(j["run_id"])
        if not m:
            continue
        st = cls.from_dict(j["settings"])
        wk = sorted(m.get("weekends") or [], key=lambda w: w["round"])
        for key, label in strategies:
            pn = [float(w.get(f"{key}_{cfg['value']}") or 0.0) for w in wk]
            s, cum = curve_stats(pn, cfg["season_events"])
            bi = s.pop("best_index")
            rec = dict(job=j["id"], run_id=j["run_id"], year=j["year"], rounds=j.get("rounds"), settings_key=st.key,
                       venue=j.get("venue", "polymarket"),
                       combo=cls.from_dict(_without(st, "seed")).key, sims=st["sims"], seed=st["seed"], label=st.label(), strategy=key, strategy_label=label,
                       note=j.get("note", ""), best_event=wk[bi]["event"] if bi is not None else None, **s)
            rows.append(rec)
            curves.append(dict(rec, rounds_list=[w["round"] for w in wk], events=[w["event"] for w in wk],
                               weekend_pnl=pn, cumulative=cum, settings=st.to_json()))
    # each run against the default settings at its own fidelity, seed and rounds
    base = {}
    for r, c in zip(rows, curves):
        if cls.from_dict(_without(c["settings"], "sims", "seed")).key == cls.from_dict().key:
            base[(r["year"], r["rounds"], r["sims"], r["seed"], r["strategy"], r["venue"])] = r   # same venue's baseline
    by_row = {id(r): c for r, c in zip(rows, curves)}
    for r in rows:
        b = base.get((r["year"], r["rounds"], r["sims"], r["seed"], r["strategy"], r["venue"]))
        r["vs_baseline"] = r["pnl"] - b["pnl"] if b and b is not r else (0.0 if b is r else None)
        # concentration of the gain: without the event where it gained most over the baseline
        r["gain_without_best"], r["best_gain_event"] = None, None
        if b and b is not r:
            mine, theirs = by_row[id(r)], by_row[id(b)]
            bp = dict(zip(theirs["events"], theirs["weekend_pnl"]))
            diffs = [(x - bp[e], e) for e, x in zip(mine["events"], mine["weekend_pnl"]) if e in bp]
            if diffs:
                top = max(diffs)
                r["gain_without_best"], r["best_gain_event"] = r["vs_baseline"] - top[0], top[1]
    for c, r in zip(curves, rows):
        c["vs_baseline"] = r["vs_baseline"]
    return rows, curves


def noise_floor(rows, cfg=None):
    """{family: floor}: the largest sd of P&L across seed replicates (>= 3 seeds of the same combo, same
    season and fidelity) where the search ran them; the configured floor otherwise."""
    cfg = _config(cfg)
    reps = {}
    for r in rows:
        if r["rounds"]:
            continue
        st_key = (r["year"], r["sims"], r["strategy"], r["combo"])
        reps.setdefault(st_key, {})[r["seed"]] = r["pnl"]
    floor = dict(cfg["noise"])
    measured = {}
    for (_, _, strat, _), by_seed in reps.items():
        v = list(by_seed.values())
        if len(v) >= 3:
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
            measured[family(strat)] = max(measured.get(family(strat), 0.0), sd)
    floor.update(measured)
    return floor, measured


def rank(rows, curves, cfg=None, twin=None, cls=SS.Settings):
    """One row per target-season combo x strategy at the default fidelity, with its held-out twin(s),
    label, confirmation and score. twin(settings, year) -> settings the held-out season runs the same
    model with (e.g. a switch that's inert that season dropped); None = the same settings."""
    cfg = _config(cfg)
    settings = {c["settings_key"]: cls.from_dict(c["settings"]) for c in curves}
    floor, measured = noise_floor(rows, cfg)
    full = [r for r in rows if not r["rounds"]]
    by = {}
    for r in full:
        by.setdefault((r["year"], r["combo"], r["strategy"], r["venue"]), []).append(r)     # seed replicates group together

    def mean_of(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return sum(v) / len(v) if v else None

    def lookup(year, st, strat, venue="polymarket"):
        key = cls.from_dict(_without(st, "seed")).key
        return by.get((year, key, strat, venue))

    default_sims = cfg["rank_sims"] or cls.BY["sims"].default      # the fidelity combos are ranked at
    out = []
    for (year, combo, strat, venue), rs in by.items():
        st = settings[rs[0]["settings_key"]]
        if year != cfg["target"] or st["sims"] != default_sims:
            continue
        nf = floor[family(strat)]
        d_t = mean_of(rs, "vs_baseline")
        held = {}
        for hy in cfg["holdout"]:
            st_h = twin(st, hy) if twin else st
            held[hy] = lookup(hy, st_h, strat, venue) or lookup(hy, st, strat, venue)
        d_h = {hy: mean_of(h, "vs_baseline") if h else None for hy, h in held.items()}
        if cls.from_dict(_without(st, "seed")).key == cls.from_dict().key:
            label = "baseline"
        elif any(v is None for v in d_h.values()):
            label = "no held-out run"
        elif d_t > nf and all(v > nf for v in d_h.values()):
            label = "robust"
        elif d_t > nf:
            label = "target only"
        elif all(v > nf for v in d_h.values()) and d_t > -nf:
            label = "held-out-led"
        else:
            label = "not better"
        # confirmation at higher fidelity, in every season the label rests on
        conf = {}
        for yy, s_ in [(year, st)] + [(hy, twin(st, hy) if twin else st) for hy in cfg["holdout"]]:
            c = lookup(yy, dict(s_, sims=cfg["confirm_sims"]), strat, venue) \
                or lookup(yy, dict(st, sims=cfg["confirm_sims"]), strat, venue)
            conf[yy] = mean_of(c, "vs_baseline") if c else None
        if cfg["confirm_sims"] == default_sims:                    # ranked at the confirmation fidelity already
            confirmed = ""
        elif label in ("robust", "target only", "held-out-led"):
            need = {"robust": list(conf), "target only": [year], "held-out-led": cfg["holdout"]}[label]
            confirmed = "not run" if any(conf[y] is None for y in need) else \
                ("yes" if all(conf[y] > 0 for y in need) else "no")
        else:
            confirmed = ""
        pnl_t = mean_of(rs, "pnl")
        pnl_h = {hy: mean_of(h, "pnl") if h else None for hy, h in held.items()}
        pnl = cfg["value"] == "pnl"
        # ranked by money made (P&L), or by the gain over the baseline (scores: a log loss is never positive)
        v_t, v_h = (pnl_t, pnl_h) if pnl else (d_t, d_h)
        score = v_t + sum(min(0.0, v) for v in v_h.values() if v is not None) \
            if all(v is not None for v in v_h.values()) else v_t - 1e6
        # better than the baseline isn't the same as profitable (params-4h: hold strategies beat a losing
        # baseline in 2025 and still lost money)
        loses = [str(y) for y, v in [(year, pnl_t), *pnl_h.items()] if pnl and v is not None and v < 0]
        wb2_t = mean_of(rs, "pnl_without_best2")
        wb2_h = {hy: mean_of(h, "pnl_without_best2") if h else None for hy, h in held.items()}
        shp = shape([pnl_t, *pnl_h.values()], [wb2_t, *wb2_h.values()]) if pnl else ""
        rec = dict(id=f"{strat}-{combo}" + ("" if venue == "polymarket" else f"-{venue}"), strategy=strat,
                   settings_key=combo, label_settings=st.label(), venue=venue,
                   verdict=label, shape=shp, confirmed=confirmed, loses_money_in=",".join(loses), score=score, noise_floor=nf, replicates=len(rs),
                   pnl_target=pnl_t, vs_base_target=d_t, sharpe_target=mean_of(rs, "sharpe"),
                   dd_target=mean_of(rs, "max_drawdown"), without_best_target=mean_of(rs, "pnl_without_best"),
                   without_best2_target=wb2_t, weekends_up_target=mean_of(rs, "weekends_up"),
                   weekends_target=mean_of(rs, "weekends"),
                   best_event_target=rs[0]["best_event"], gain_without_best_target=mean_of(rs, "gain_without_best"),
                   best_gain_event_target=rs[0]["best_gain_event"], settings=st.to_json(),
                   run_target=rs[0]["run_id"])
        for hy in cfg["holdout"]:
            h = held[hy] or []
            rec.update({f"pnl_{hy}": pnl_h[hy], f"vs_base_{hy}": d_h[hy], f"sharpe_{hy}": mean_of(h, "sharpe"),
                        f"dd_{hy}": mean_of(h, "max_drawdown"), f"without_best_{hy}": mean_of(h, "pnl_without_best"),
                        f"without_best2_{hy}": wb2_h[hy], f"weekends_up_{hy}": mean_of(h, "weekends_up"),
                        f"weekends_{hy}": mean_of(h, "weekends"),
                        f"gain_without_best_{hy}": mean_of(h, "gain_without_best"),
                        f"run_{hy}": h[0]["run_id"] if h else None,
                        f"settings_{hy}": settings[h[0]["settings_key"]].to_json() if h else None})
        rec.update({f"confirm_{y}": v for y, v in conf.items()})
        out.append(rec)
    out.sort(key=lambda x: -x["score"])
    return out, floor, measured


def candidates(ranking, top):
    """The top combos that have a held-out run and aren't the baseline (or a loosened guard)."""
    return [r for r in ranking if r["verdict"] not in ("baseline", "no held-out run", "not better")][:top]


def write(out, jobs, metrics, strategies, cfg=None, twin=None, rerun=None, echo=print, cls=SS.Settings, sport="f1",
          model=None, kind=None):
    """Every output file, from finished jobs only. rerun(job-like dict) -> argv list, for re-run commands.
    kind: the jobs' kind (None: F1's sweep, any other sport's or model's walk-forward)."""
    cfg = _config(cfg)
    rows, curves = stats(jobs, metrics, strategies, cfg, cls)
    ranking, floor, measured = rank(rows, curves, cfg, twin, cls)
    kind = kind or ("sweep" if sport == "f1" and not model else "walk_forward")
    job = dict(kind=kind, **({"sport": sport} if sport != "f1" or model else {}), **({"model": model} if model else {}))
    (out / "pnl_curves.json").write_text(json.dumps(curves, default=str))
    cols = ["year", "rounds", "venue", "strategy", "label", "sims", "seed", "pnl", "vs_baseline", "weekends_up", "weekends",
            "max_drawdown", "sharpe", "pnl_without_best", "pnl_without_best2", "best_event", "gain_without_best", "best_gain_event", "run_id",
            "job", "settings_key", "note"]
    _csv(out / "stats.csv", sorted(rows, key=lambda r: (r["year"], r["strategy"], -r["pnl"])), cols)
    rcols = [c for c in ranking[0] if not c.startswith("settings")] if ranking else []
    _csv(out / "ranking.csv", ranking, rcols)
    cdir = out / "candidates"
    cdir.mkdir(exist_ok=True)
    for f in cdir.glob("*.json"):
        f.unlink()
    toml = ["# Top combos of this search as [[candidate]] entries (racinglines strategy search-report f1).", ""]
    top = candidates(ranking, cfg["top"])
    for i, r in enumerate(top, 1):
        st = cls.from_dict(r["settings"])
        doc = dict(r, rank=i, name=f"{r['strategy']} · {st.label()}", changed=st.changed())
        if rerun:
            jv = dict(job, **({} if r.get("venue", "polymarket") == "polymarket" else {"venue": r["venue"]}))
            doc["rerun"] = {str(cfg["target"]): " ".join(rerun(dict(jv, year=cfg["target"], settings=r["settings"])))}
            for hy in cfg["holdout"]:
                if r.get(f"settings_{hy}"):
                    doc["rerun"][str(hy)] = " ".join(rerun(dict(jv, year=hy, settings=r[f"settings_{hy}"])))
        (cdir / f"{r['id']}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        why = (f"{r['verdict']}: {cfg['target']} {r['pnl_target']:+,.0f} ({r['vs_base_target']:+,.0f} vs baseline); "
               + "; ".join(f"{hy} {r[f'pnl_{hy}']:+,.0f} ({r[f'vs_base_{hy}']:+,.0f})" for hy in cfg["holdout"]))
        toml += ["[[candidate]]", f'id = "{r["id"]}"', f'name = "{r["id"]}: {r["strategy"]} · {st.label()}"',
                 f"year = {cfg['target']}", f'strategy = "{r["strategy"]}"', f'why = "{why}"']
        if sport != "f1" or model:
            toml.append(f'sport = "{sport}"')
        if model:
            toml.append(f'model = "{model}"')
        if r.get("venue", "polymarket") != "polymarket":
            toml.append(f'venue = "{r["venue"]}"')
        for k, v in st.to_json().items():
            if k == "variant" or st[k] != cls.BY[k].default:
                toml.append(f"{k} = " + (f'"{",".join(v)}"' if isinstance(v, list) else json.dumps(v)))
        toml.append("")
    (out / "candidates.toml").write_text("\n".join(toml))
    (out / "report.md").write_text(summary_md(ranking, top, floor, measured, cfg))
    echo(f"search-report: {len(rows)} run x strategy rows, {len(ranking)} combos, {len(top)} candidates -> {out}")
    return rows, ranking


def summary_md(ranking, top, floor, measured, cfg):
    hy = cfg["holdout"]
    u = cfg["unit"]
    lines = ["# Search report", "",
             f"Target season {cfg['target']}, held out: {', '.join(map(str, hy))}. Noise floor per strategy family: "
             + ", ".join(f"{k} ±{v:,.0f}" + (" (seed replicates)" if k in measured else " (configured)")
                         for k, v in floor.items()) + ".", "",
             "Labels: **robust** beats the baseline by more than the noise floor in every season; **target only** "
             "in the target season only; **held-out-led** in the held-out seasons, with the target within noise. "
             "*Confirmed* re-checks the label's seasons at " f"{cfg['confirm_sims']:,} simulations. "
             f"*Without best* is the target-season {u} without its best event; *gain without best*, its gain over the "
             "baseline without the event where it gained most. *Loses money in*: better than the "
             "baseline there, but still a loss (P&L only). *Shape*: **steady** is still up without its best two events "
             "in every season, **concentrated** is up in every season but not without its best two in one of them, "
             "**mixed** is up in one season and down in another, **losing** is down in every season." + (" Scores are −1000 × log loss per event, summed: "
                                                                "higher is better." if u == "score" else ""), "",
             "| Id | Strategy | Settings | Label | Shape | Confirmed | " + f"{cfg['target']} | vs baseline | Without best | "
             "Gain without best | " + " | ".join(f"{y} | vs baseline" for y in hy) + " | Loses money in |",
             "|---" * (11 + 2 * len(hy)) + "|"]
    f = lambda v: "" if v is None else f"{v:+,.0f}"                                         # noqa: E731
    for r in top:
        lines.append(f"| `{r['id']}` | {r['strategy']} | {r['label_settings']} | {r['verdict']} | {r['shape']} | "
                     f"{r['confirmed']} | "
                     f"{f(r['pnl_target'])} | {f(r['vs_base_target'])} | {f(r['without_best_target'])} "
                     f"({r['best_event_target']}) | {f(r['gain_without_best_target'])} ({r['best_gain_event_target']}) | "
                     + " | ".join(f"{f(r[f'pnl_{y}'])} | {f(r[f'vs_base_{y}'])}" for y in hy) + f" | {r['loses_money_in']} |")
    counts = pd.Series([r["verdict"] for r in ranking]).value_counts().to_dict() if ranking else {}
    shapes = pd.Series([r["shape"] for r in ranking if r["shape"]]).value_counts().to_dict() if ranking else {}
    lines += ["", "Combos by label: " + ", ".join(f"{k} {v}" for k, v in counts.items()), ""]
    if shapes:
        lines += ["Combos by shape: " + ", ".join(f"{k} {v}" for k, v in shapes.items()), ""]
    return "\n".join(lines)


def _csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 2) if isinstance(r.get(k), float) else r.get(k)) for k in cols})


def run(queue_path, echo=print):
    """Load the search's finished sweep jobs and their saved runs, and write every output."""
    import tomllib

    from sqlalchemy import text

    from racinglines import paths
    from racinglines.db.config import get_engine
    from racinglines.models import evaluate as EV
    from racinglines.models import variants as V
    from racinglines.pipelines import search as S
    from racinglines.markets import kinds as K
    cfg, _, _ = S.load(queue_path)
    rep = tomllib.loads(open(queue_path).read()).get("report", {})
    out = paths.runs("search", cfg["name"])
    state = json.loads((out / "state.json").read_text())
    done = [v for v in state.values() if v.get("status") == "done" and v.get("run_id")]
    with get_engine().connect() as c:
        rows = c.execute(text("SELECT id, metrics FROM model_runs WHERE id = ANY(:ids)"),
                         dict(ids=[j["run_id"] for j in done])).fetchall()
    metrics = {r[0]: r[1] for r in rows}

    def twin(st, year):
        return SS.Settings.from_dict(dict(st, variant=V.for_season(st["variant"], year)))
    result = None
    f1 = [v for v in done if v["kind"] == "sweep" and v.get("sport", S.DEFAULT_SPORT) == S.DEFAULT_SPORT]
    if f1:
        result = write(out, f1, metrics, EV.STRATEGIES, {k: v for k, v in rep.items() if not isinstance(v, dict)
                                                          or k == "noise"}, twin=twin, rerun=S.argv, echo=echo)
    for sport in sorted({v["sport"] for v in done if v["kind"] == "sweep" and v.get("sport", S.DEFAULT_SPORT) != S.DEFAULT_SPORT}):
        jobs = [v for v in done if v["kind"] == "sweep" and v.get("sport") == sport]     # another sport's sweeps
        d = out / f"{sport}-sweep"
        d.mkdir(exist_ok=True)
        result = write(d, jobs, metrics, EV.STRATEGIES, dict({k: v for k, v in rep.items() if not isinstance(v, dict)
                                                               or k == "noise"}, **rep.get(f"{sport}-sweep", {})),
                       rerun=S.argv, echo=echo, cls=S.settings_class(sport, None, "sweep"), sport=sport, kind="sweep")
    for sport, model in sorted({(v.get("sport", "f1"), v.get("model") or "") for v in done if v["kind"] == "walk_forward"}):
        jobs = [v for v in done if v["kind"] == "walk_forward" and (v.get("sport", "f1"), v.get("model") or "") == (sport, model)]
        kinds = sorted({k[:-len("_score")] for j in jobs for w in (metrics.get(j["run_id"]) or {}).get("weekends") or []
                        for k in w if k.endswith("_score")})
        strategies = [(k, K.KINDS[k].label if k in K.KINDS else k) for k in kinds]
        d = out / (f"{sport}-{model}" if model else sport)
        d.mkdir(exist_ok=True)
        result = write(d, jobs, metrics, strategies, dict(SPORT_DEFAULTS.get(model or sport, {}), **rep.get(sport, {})),
                       rerun=S.argv, echo=echo, cls=S.settings_class(sport, model or None), sport=sport,
                       model=model or None)
    return result
