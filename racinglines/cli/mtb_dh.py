"""
racinglines mtb_dh <command>: UCI downhill.

    download   Download event result files from Chronorace (data/raw/mtb_dh/chronorace)
    parse      Parse downloaded / pasted result files into a tidy CSV
    ingest     Load downloaded event files into the database
    forecast   Simulate the rest of the season and the championship (was: racinglines model forecast mtb_dh)
    backtest   Walk-forward backtest over past seasons (was: racinglines model backtest mtb_dh)
    points     Championship points tables: import the official scales, show them, check against standings
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.models.inputs.run_times import (
    CATEGORY_WEIGHTS, HALF_LIFE_DAYS, RUN_WEIGHTS, _spearman, backtest_season, completed_events,
    forecast_season, select_target, walk_forward_season,
)
from racinglines.models.inputs.run_times.model import FINAL_POINTS, QUAL_POINTS, QUAL_POINTS_ROUND  # noqa: F401

def load_splits(args):
    """Tidy frame from the database (--db) or a tidy CSV from `racinglines data parse mtb_dh` (--data)."""
    if args.db is not None:
        from racinglines.db.config import get_engine
        from racinglines.db.queries import load_tidy
        raw = load_tidy(get_engine(args.db or None), competition=args.competition, with_splits=False)
        return raw[raw["round"].isin(RUN_WEIGHTS)]
    if not args.data:
        raise SystemExit("Pass --db [URL] (database) or --data splits.csv (`racinglines data parse mtb_dh` output).")
    raw = pd.read_csv(args.data)
    return raw[raw["discipline"].eq("DHI") & raw["round"].isin(RUN_WEIGHTS)] if "discipline" in raw else raw


def _model_switches(args):
    """--prior-n and --eps-df (docs/todo.md, Model): fit_kw for the first, the model's EPS_DF for the
    second. Unset = today's model."""
    from racinglines.models.inputs.run_times import model as TM
    if getattr(args, "eps_df", None) is not None:
        if args.eps_df <= 2:
            raise SystemExit("--eps-df must be above 2 (the t distribution's variance is finite above 2).")
        TM.EPS_DF = args.eps_df
    return {} if getattr(args, "prior_n", None) is None else dict(prior_n=args.prior_n)


def _check_save(args):
    if args.save and args.db is None:
        raise SystemExit("--save writes to the database, so it needs --db.")


def _points_scheme(args, season):
    """The season's points tables with --points db (points_schemes), else None (the schema's placeholders)."""
    if getattr(args, "points", "schema") == "schema":
        return None
    from racinglines.db.config import get_engine
    from racinglines.models.inputs.run_times import points as P
    with get_engine(args.db or None).connect() as c:
        scheme = P.scheme_for(season, P.load(c, args.competition))
    args.points_official = scheme.official
    print(f"Points {season}: {scheme.source}" + (" (official)" if scheme.official else " (placeholder)"))
    return scheme


def _save_run(args, raw, target, *, kind, fit_kw, metrics, race_predictions=None, standings=None):
    from racinglines.db.config import get_session
    from racinglines.db.queries import save_model_run

    params = {k: v for k, v in fit_kw.items() if k != "category_weights"}
    from racinglines.models.inputs.run_times import model as TM
    if TM.EPS_DF is not None:
        params["eps_df"] = TM.EPS_DF
    params.update(junior_weight=args.junior_weight, sims=args.sims, seed=args.seed, final_points=TM.FINAL_POINTS,
                  qual_points=TM.QUAL_POINTS, qual_points_round=TM.QUAL_POINTS_ROUND,
                  points_official=getattr(args, "points_official", False), label=getattr(args, "scenario", None))
    if getattr(args, "points", "schema") != "schema":
        params["points_source"] = args.points
    with get_session(args.db or None) as session:
        run_id = save_model_run(
            session, competition=args.competition, season=target["event_date"].astype(str).str[:4].iloc[0]
            if len(target) else None, category=args.category, kind=kind,
            data_through=raw["event_date"].max(), params=params, metrics=metrics,
            race_predictions=race_predictions, standings=standings)
    print(f"Saved model run {run_id} to the database.")


def cmd_backtest(args):
    """Walk-forward + end-of-season standings backtest for several seasons."""
    _check_save(args)
    raw = load_splits(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fit_kw = dict(train_scope=args.train_scope, half_life_days=args.half_life_days,
                  category_weights={"MJ": args.junior_weight}, **_model_switches(args))
    years = raw.loc[raw["category"] == args.category, "event_date"].astype(str).str[:4]
    seasons = [str(y) for y in args.seasons] if args.seasons else sorted(years.unique())
    if getattr(args, "points", "schema") == "schema":
        print("NOTE: points use placeholder tables; standings metrics are approximate.")
    print(f"Category {args.category}, training scope {args.train_scope}, "
          f"half-life {args.half_life_days:.0f} days, junior weight {args.junior_weight}\n")

    per_event, per_season, riders = [], [], []
    from racinglines.models.inputs.run_times import points as P
    for season in seasons:
        with P.use(_points_scheme(args, int(season))):
            target = select_target(raw, season, args.category)
            target = target[target["event_id"].isin(completed_events(target))]
            n_ev = target["event_id"].nunique()
            if n_ev < 3:
                print(f"{season}: {n_ev} event(s) with data, skipped (need 3+).")
                continue
            wf = walk_forward_season(raw, target, n_sims=args.sims, rng=rng,
                                     rider_rows=riders if args.reliability else None, **fit_kw)
            wf.insert(0, "season", season)
            per_event.append(wf)
            _, _, st, (train_ev, test_ev) = backtest_season(raw, target, n_holdout=2, n_sims=args.sims,
                                                              rng=rng, **fit_kw)
            champ = st.loc[st["actual_rank"] == 1].iloc[0]
            fav = st.loc[st["champion_prob"].idxmax()]
            scored = st[st["actual_points"] > 0]
            per_season.append(dict(
                season=season, events=n_ev, predicted=len(wf),
                formats="/".join(sorted(wf["format"].unique())),
                spearman_points=wf["spearman_points"].mean(),
                brier_win=wf["brier_win"].mean(), brier_win_base=wf["brier_win_base"].mean(),
                brier_podium=wf["brier_podium"].mean(), brier_podium_base=wf["brier_podium_base"].mean(),
                brier_final=wf["brier_final"].mean(), brier_final_base=wf["brier_final_base"].mean(),
                top10_hits=wf["top10_hits"].mean(),
                winner_win_prob=wf["winner_pred_win_prob"].mean(),
                standings_spearman=_spearman(scored["exp_points"], scored["actual_points"]),
                champion=champ["rider_name"], champion_prob=champ["champion_prob"],
                favourite=fav["rider_name"], favourite_prob=fav["champion_prob"],
            ))
            print(f"{season}: {len(wf)} rounds predicted, standings holdout = last {len(test_ev)} rounds", flush=True)

    ev = pd.concat(per_event, ignore_index=True)
    ss = pd.DataFrame(per_season)
    ev.to_csv(out_dir / "backtest_events.csv", index=False)
    ss.to_csv(out_dir / "backtest_seasons.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Per event (walk-forward) ===")
    cols = ["season", "venue", "format", "n_riders", "n_final", "spearman_points", "brier_win", "brier_podium",
            "brier_final", "brier_final_base", "top10_hits", "winner", "winner_pred_win_prob"]
    print(ev[cols].to_string(index=False, float_format="{:.3f}".format))
    print("\n=== Per season ===")
    print(ss.to_string(index=False, float_format="{:.3f}".format))
    all_mean = ev[["spearman_points", "brier_win", "brier_win_base", "brier_podium", "brier_podium_base",
                   "brier_final", "brier_final_base", "top10_hits", "winner_pred_win_prob"]].mean()
    print("\nAll events: " + ", ".join(f"{k}={v:.4f}" for k, v in all_mean.items()))
    if args.reliability:
        from racinglines.models.inputs.run_times.season import calibration
        rel = calibration(pd.concat(riders, ignore_index=True))
        rel.to_csv(out_dir / "backtest_reliability.csv", index=False)
        print("\n=== Reliability (every walk-forward round; z = (observed - predicted) / binomial SE) ===")
        print(rel.to_string(index=False, float_format="{:.4f}".format, na_rep=""))
    print(f"\nCSVs -> {out_dir}/")
    if args.save:
        from racinglines.db.queries import records
        _save_run(args, raw, raw.iloc[0:0], kind="backtest", fit_kw=fit_kw,
                  metrics=dict(all_events=records(all_mean.to_frame().T)[0], seasons=records(ss),
                               events=records(ev)))


def cmd_walk_forward(args):
    """Every event priced by the pricing model through the shared engine (racinglines/core/walk_forward.py),
    model-only: calibration per market kind. `racinglines backtest walk-forward mtb_dh`, with --data too."""
    from racinglines.cli.backtest import run_walk_forward
    from racinglines.models import race_model as RM
    from racinglines.pipelines import sweep_settings as SS
    _check_save(args)
    model = RM.get("mtb_dh")
    run_walk_forward(model, model.load(data=load_splits(args)), SS.from_args(args, model.Settings),
                     seasons=args.seasons, kinds=args.kinds.split(",") if args.kinds else None, out_dir=args.out_dir,
                     save=args.save, engine_url=args.db or None)


def cmd_season(args):
    _check_save(args)
    raw = load_splits(args)
    target = select_target(raw, args.season, args.category)
    if target.empty:
        raise SystemExit(f"No rows for season={args.season} category={args.category}.")
    season = target["event_date"].astype(str).str[:4].iloc[0]
    from racinglines.models.inputs.run_times import points as P
    with P.use(_points_scheme(args, int(season))):
        return _season(args, raw, target, season)


def _season(args, raw, target, season):
    fit_kw = dict(train_scope=args.train_scope, half_life_days=args.half_life_days,
                  category_weights={"MJ": args.junior_weight}, **_model_switches(args))
    done_target = target[target["event_id"].isin(completed_events(target))]  # for backtests
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    top = args.top
    pd.set_option("display.width", 200)
    fmt = {c: "{:.1%}".format for c in ("win_prob", "podium_prob", "top10_prob", "make_final_prob",
                                         "champion_prob", "top3_prob", "attend_prob")}
    if not getattr(args, "points_official", False):
        print("NOTE: FINAL_POINTS / QUAL_POINTS are placeholders, not the official UCI tables.")
    n_train_ev = (target if args.train_scope == "season" else raw)["event_id"].nunique()
    print(f"Target: {season} {args.category} ({target['event_id'].nunique()} events). "
          f"Training scope: {args.train_scope} ({n_train_ev} event/category sets), "
          f"half-life {args.half_life_days:.0f} days, junior weight {args.junior_weight}.\n")

    saved_metrics, saved_preds, saved_standings = {}, [], None
    if args.walk_forward:
        wf = walk_forward_season(raw, done_target, n_sims=min(args.sims, 5000), rng=rng, **fit_kw)
        wf.to_csv(out_dir / "walk_forward.csv", index=False)
        print(f"=== WALK-FORWARD: each {season} round predicted from everything before it ===")
        cols = ["venue", "n_riders", "spearman_points", "brier_win", "brier_win_base", "brier_podium",
                "brier_podium_base", "brier_final", "brier_final_base", "top10_hits", "winner",
                "winner_pred_win_prob"]
        print(wf[cols].to_string(index=False, float_format="{:.4f}".format))
        print("mean: " + ", ".join(f"{c}={wf[c].mean():.4f}" for c in cols[2:10]) + "\n")
        saved_metrics["walk_forward"] = wf

    if args.backtest:
        model, reports, standings, (train_ev, test_ev) = backtest_season(
            raw, done_target, n_holdout=args.backtest, n_sims=args.sims, rng=rng, **fit_kw)
        print(f"=== BACKTEST: fit on {len(train_ev)} {season} events (+ history if scope=all), predict {', '.join(test_ev)} ===")
        print(f"model: sigma={model['sigma']:.4f} tau={model['tau']:.4f} (log-time), "
              f"incident rate={model['p0']:.1%}, incidents that are DNF/DSQ={model['dnf_share']:.0%}\n")
        for metrics, summ in reports:
            print(f"--- {metrics['venue']} ({metrics['event_id']}) ---")
            print(f"  Spearman(exp points, actual points) = {metrics['spearman_points']:.3f}")
            for k in ("win", "podium", "final"):
                print(f"  Brier {k:<7} model {metrics[f'brier_{k}']:.4f}  vs uniform {metrics[f'brier_{k}_base']:.4f}")
            print(f"  top-10 hits: {metrics['top10_hits']}/10   winner {metrics['winner']} "
                  f"had win prob {metrics['winner_pred_win_prob']:.1%}")
            cols = ["rider_name", "win_prob", "podium_prob", "top10_prob", "make_final_prob",
                    "exp_points", "actual_final_rank", "actual_points"]
            print(summ[cols].head(top).to_string(index=False, formatters=fmt, float_format="{:.1f}".format))
            summ.to_csv(out_dir / f"backtest_{metrics['venue']}.csv", index=False)
            print()
            if "race_id" in target:
                race_id = target.loc[target["event_id"] == metrics["event_id"], "race_id"].iloc[0]
                saved_preds.append(summ.assign(target=f"backtest:{metrics['event_id']}", race_id=race_id))
        saved_metrics["backtest"] = pd.DataFrame([m for m, _ in reports])
        standings.to_csv(out_dir / "backtest_standings.csv", index=False)
        print(f"--- Standings after {test_ev[-1]}: predicted vs actual ---")
        s = standings[standings["actual_points"] > 0]
        print(f"  Spearman(exp points, actual points) = {_spearman(s['exp_points'], s['actual_points']):.3f}")
        cols = ["rider_name", "current_points", "exp_points", "champion_prob", "top3_prob",
                "actual_points", "actual_rank"]
        print(standings[cols].head(top).to_string(index=False, formatters=fmt, float_format="{:.0f}".format))
        print()

    if args.remaining:
        model, upcoming, per_round, standings = forecast_season(
            raw, target, n_remaining=args.remaining, n_sims=args.sims, rng=rng,
            unraced=getattr(args, "unraced_format", "default"), **fit_kw)
        print(f"=== FORECAST: {args.remaining} remaining round(s) "
              f"({len(upcoming)} in progress/upcoming with start lists) ===")
        for event_id, summ in upcoming:
            rows = target[target["event_id"] == event_id]
            venue = rows["venue"].iloc[0] if "venue" in rows else event_id
            print(f"--- {venue} ({event_id}): {len(summ)} starters; model fit on data before this weekend, "
                  f"weekend pace adjusted from timed training ---")
            cols = ["rider_name", "win_prob", "podium_prob", "top10_prob", "make_final_prob", "exp_points",
                    "tt_pace_adj_pct"]
            print(summ[cols].head(max(top, 15)).to_string(index=False, formatters=fmt,
                                                          float_format="{:.2f}".format))
            summ.to_csv(out_dir / f"forecast_{venue}.csv", index=False)
            race_id = rows["race_id"].iloc[0] if "race_id" in rows else None
            saved_preds.append(summ.assign(target=f"event:{event_id}", race_id=race_id))
            print()
        if per_round is not None:
            print("--- Per remaining round not yet in the data (venue unknown) ---")
            cols = ["rider_name", "attend_prob", "win_prob", "podium_prob", "top10_prob",
                    "make_final_prob", "exp_points"]
            print(per_round[cols].head(top).to_string(index=False, formatters=fmt, float_format="{:.1f}".format))
            per_round.to_csv(out_dir / "forecast_per_round.csv", index=False)
            saved_preds.append(per_round.assign(target="remaining_round"))
        print("\n--- Projected final championship standings ---")
        cols = ["rider_name", "current_rank", "current_points", "exp_points", "points_p10",
                "points_p90", "champion_prob", "top3_prob", "exp_rank"]
        print(standings[cols].head(top).to_string(index=False, formatters=fmt, float_format="{:.0f}".format))
        standings.to_csv(out_dir / "forecast_standings.csv", index=False)
        saved_standings = standings
    print(f"\nCSVs -> {out_dir}/")
    if args.save:
        from racinglines.db.queries import records
        kind = ("scenario" if getattr(args, "scenario", None) else "forecast") if args.remaining else "backtest"
        _save_run(args, raw, target, kind=kind, fit_kw=fit_kw,
                  metrics={k: records(v) for k, v in saved_metrics.items()},
                  race_predictions=pd.concat(saved_preds, ignore_index=True) if saved_preds else None,
                  standings=saved_standings)


def _add_source_args(p):
    p.add_argument("--db", nargs="?", const="", default=None, metavar="URL",
                   help="Read from the database ($DATABASE_URL or the docker-compose default if no URL).")
    p.add_argument("--data", help="Or read a tidy CSV from `racinglines data parse mtb_dh` instead of the database.")
    p.add_argument("--competition", default="uci_dhi_wc", help="Competition code in the database.")
    p.add_argument("--save", action="store_true", help="Store the model run and predictions in the database.")
    p.add_argument("--points", choices=["schema", "db"], default="schema",
                   help="Points tables: the schema's placeholders (default) or points_schemes by season (needs --db).")


def cmd_ingest(args):
    from racinglines.db.config import get_session
    from racinglines.paths import DH_RAW
    from racinglines.sources.chronorace.ingest import ingest_paths
    import re

    from racinglines.db import changes
    lines = []

    def echo(line):
        print(line)
        lines.append(line.strip())
    with get_session(args.db or None) as s:
        counts = ingest_paths(s, args.paths or [DH_RAW], competition=args.competition, force=args.force, echo=echo)
        if counts.get("ingested") or counts.get("ERROR"):     # the change log: only runs that changed something
            files = {n: st for n, st in (ln.split(": ", 1) for ln in lines if ": " in ln) if st != "unchanged"}
            merges = sorted({tuple(map(int, p)) for p in re.findall(r"merge (\d+) into (\d+)", "\n".join(lines))})
            summary = (f"{counts.get('ingested', 0)} downhill files ingested"
                       + (" (forced)" if args.force else "")
                       + (f", {len(merges)} same-UCI-ID pairs reported" if merges else "")
                       + (f": {args.note}" if args.note else ""))
            changes.record(s, "ingest", summary, sport="mtb_dh",
                           detail=dict(counts=counts, paths=[str(x) for x in args.paths or [DH_RAW]], files=files,
                                       merge_pairs=[dict(drop=d, keep=k) for d, k in merges], note=args.note))
            s.commit()
    print("Done: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("download", "parse"):          # standalone tools with their own options
        from racinglines.sources.chronorace import download, parse
        mod = download if argv[0] == "download" else parse
        sys.argv = [f"racinglines data {argv[0]} mtb_dh"] + argv[1:]
        return mod.main()
    from racinglines import paths
    ap = argparse.ArgumentParser(prog="racinglines <group> <command> mtb_dh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="Download Chronorace result files (see: racinglines data download mtb_dh -h).")
    sub.add_parser("parse", help="Parse result files into a tidy CSV (see: racinglines data parse mtb_dh -h).")

    ing = sub.add_parser("ingest", help="Load downloaded event files into the database.")
    ing.add_argument("paths", nargs="*", help="Files or directories (default: data/raw/mtb_dh/chronorace).")
    ing.add_argument("--db", default=None, help="Database URL (default: $DATABASE_URL / docker-compose).")
    ing.add_argument("--competition", default="uci_dhi_wc")
    ing.add_argument("--force", action="store_true", help="Re-ingest even if a file hasn't changed.")
    ing.add_argument("--note", help="Why (kept in the data change log, `racinglines db changes`).")
    ing.set_defaults(func=cmd_ingest)

    season_p = sub.add_parser("forecast", help="Backtest held-out rounds and forecast remaining "
                                               "rounds + championship standings.")
    _add_source_args(season_p)
    season_p.add_argument("--out-dir", default=str(paths.runs("mtb_dh", "forecast", mkdir=False)))
    season_p.add_argument("--backtest", type=int, default=2,
                          help="Hold out and predict the last N raced events (0 = skip).")
    season_p.add_argument("--remaining", type=int, default=2,
                          help="Number of unraced rounds left in the season (0 = skip).")
    season_p.add_argument("--season", help="Season (year) to predict; default = latest in the data.")
    season_p.add_argument("--category", default="ME", help="Category code to predict (ME, MJ, ...).")
    season_p.add_argument("--train-scope", choices=["all", "season"], default="all",
                          help="Train on every season/category in --data, or only the target season.")
    season_p.add_argument("--half-life-days", type=float, default=HALF_LIFE_DAYS,
                          help="Recency half-life for training runs.")
    season_p.add_argument("--junior-weight", type=float, default=CATEGORY_WEIGHTS["MJ"],
                          help="Training weight of junior runs relative to elite.")
    season_p.add_argument("--walk-forward", action="store_true",
                          help="Also predict every target round from everything before it and score it.")
    season_p.add_argument("--prior-n", type=float, default=None,
                          help="Shrinkage of rider pace toward the field (default 0.5; 1.5 before the "
                               "2026-09-28 tuning, see docs/model.md).")
    season_p.add_argument("--eps-df", type=float, default=None,
                          help="Student-t run noise with these degrees of freedom (default: normal).")
    season_p.add_argument("--unraced-format", choices=["default", "last"], default="default",
                          help="Format of unraced rounds: the 2026 elite men's (default, every category), or the "
                               "category's latest completed event's own (women, juniors).")
    season_p.add_argument("--sims", type=int, default=10000)
    season_p.add_argument("--scenario", default=None, metavar="LABEL",
                          help="With --save: store as kind='scenario' (not used for live prices until promoted).")
    season_p.add_argument("--seed", type=int, default=42)
    season_p.add_argument("--top", type=int, default=15, help="Rows to print per table.")
    season_p.set_defaults(func=cmd_season)

    bt_p = sub.add_parser("backtest", help="Walk-forward + standings backtest over several seasons.")
    _add_source_args(bt_p)
    bt_p.add_argument("--out-dir", default=str(paths.runs("mtb_dh", "backtests", mkdir=False)))
    bt_p.add_argument("--seasons", nargs="+", type=int, help="Seasons to backtest (default: all with data).")
    bt_p.add_argument("--category", default="ME")
    bt_p.add_argument("--train-scope", choices=["all", "season"], default="all")
    bt_p.add_argument("--half-life-days", type=float, default=HALF_LIFE_DAYS)
    bt_p.add_argument("--junior-weight", type=float, default=CATEGORY_WEIGHTS["MJ"])
    bt_p.add_argument("--prior-n", type=float, default=None,
                          help="Shrinkage of rider pace toward the field (default 0.5; 1.5 before the "
                               "2026-09-28 tuning, see docs/model.md).")
    bt_p.add_argument("--eps-df", type=float, default=None,
                          help="Student-t run noise with these degrees of freedom (default: normal).")
    bt_p.add_argument("--sims", type=int, default=4000)
    bt_p.add_argument("--seed", type=int, default=42)
    bt_p.add_argument("--reliability", action="store_true",
                      help="Also print and save reliability curves (win, podium, top 10, make the Final).")
    bt_p.set_defaults(func=cmd_backtest)

    wf_p = sub.add_parser("walk-forward", help="Price every event through the shared engine (model-only): "
                                                "calibration per market kind. What a downhill search job runs.")
    _add_source_args(wf_p)
    wf_p.add_argument("--out-dir", default=str(paths.runs("mtb_dh", "walk_forward", mkdir=False)))
    wf_p.add_argument("--seasons", nargs="+", type=int, help="Seasons (default: all with 3+ completed events).")
    wf_p.add_argument("--kinds", help="Market kinds, comma-separated (default: every per-rider kind the model "
                                      "prices; race_h2h adds head-to-heads).")
    from racinglines.models.inputs.run_times.settings import DHSettings
    from racinglines.pipelines import sweep_settings as SS
    SS.add_arguments(wf_p, DHSettings)
    wf_p.set_defaults(func=cmd_walk_forward)

    pts_p = sub.add_parser("points", help="Championship points tables: import, show, check against official standings.")
    pts_sub = pts_p.add_subparsers(dest="action", required=True)
    imp = pts_sub.add_parser("import", help="Load a points file (e.g. sports/points/uci_dhi_wc.toml) into points_schemes.")
    imp.add_argument("file")
    show = pts_sub.add_parser("show", help="The tables each season uses.")
    chk = pts_sub.add_parser("check", help="Cumulative points vs the official standings, rider by rider.")
    chk.add_argument("--standings", required=True, help="The official standings: a CSV (rider, points) or a TOML file (see sports/points/standings/).")
    chk.add_argument("--season", type=int, required=True)
    chk.add_argument("--category", default="ME")
    chk.add_argument("--through-round", type=int, help="Series round the standings are after (default: every round).")
    chk.add_argument("--data", help="Or a tidy CSV instead of the database.")
    for q in (imp, show, chk):
        q.add_argument("--db", nargs="?", const="", default=None, metavar="URL",
                       help="Database ($DATABASE_URL or the docker-compose default if no URL).")
        q.add_argument("--competition", default="uci_dhi_wc")
    chk.add_argument("--points", choices=["schema", "db"], default="db",
                     help="Tables to check: points_schemes by season (default) or the schema's placeholders.")
    pts_p.set_defaults(func=cmd_points)

    live_p = sub.add_parser("live", help="Follow a final live from UCI timing: rank probabilities, the maker's quotes, "
                                         "the private book's crowd (pipelines/live_dh.py). A demo experiment.")
    live_p.add_argument("--slug", required=True, help="ChronoRace event slug, e.g. 20260925_mtb")
    live_p.add_argument("--final", required=True, help="Live-timing key of the final (e.g. 3 = men's final)")
    live_p.add_argument("--quali", default="2,91", help="Live-timing keys of the qualifying sessions, comma-separated")
    live_p.add_argument("--conditions", default="", help='Free text, e.g. "clear, rutted" (rutted / wet widen the spread)')
    live_p.add_argument("--interval", type=int, default=20, help="Seconds between polls")
    live_p.add_argument("--minutes", type=int, default=0, help="Stop after this many minutes (0 = when the final is over)")
    live_p.add_argument("--once", action="store_true", help="One update, then exit")
    live_p.set_defaults(func=cmd_live)

    from racinglines.cli.verbs import retitle
    retitle(ap, "mtb_dh")
    args = ap.parse_args(argv)
    return args.func(args)


def cmd_points(args):
    from racinglines.models.inputs.run_times import points as P
    if args.action == "import":
        from racinglines.db.config import get_session
        rows = P.read_file(args.file)
        with get_session(args.db or None) as s:
            n = P.import_rows(s, rows, args.competition)
        placeholders = sum(not r.get("official", False) for r in rows)
        print(f"{n} table(s) imported" + (f"; {placeholders} still marked placeholder" if placeholders else ""))
        return 0
    if args.action == "show":
        from racinglines.db.config import get_engine
        with get_engine(args.db or None).connect() as c:
            schemes = P.load(c, args.competition)
        for season in range(2021, 2027):
            sc = P.scheme_for(season, schemes)
            print(f"{season}: {sc.source:6s} {'official' if sc.official else 'PLACEHOLDER':11s} "
                  f"final {len(sc.final)} places (1st {sc.final[0]}), qualifying {len(sc.qual)} places")
        return 0
    if args.db is None and not args.data:
        args.db = ""
    raw = load_splits(args)
    target = select_target(raw, args.season, args.category)
    if target.empty:
        raise SystemExit(f"No results for {args.season} {args.category}.")
    through = None
    if args.through_round:
        through = P.through_event(target, args.through_round)       # the season's last event with that number
    official = P.read_standings(args.standings)
    with P.use(_points_scheme(args, args.season)):
        rec = P.reconcile(target, official, through)
    bad = rec[rec["diff"].abs() > 1e-9]
    pd.set_option("display.width", 200)
    print(rec.head(40).to_string(index=False, float_format="{:.0f}".format))
    print(f"\n{len(rec) - len(bad)}/{len(rec)} riders match" + ("" if bad.empty else f"; {len(bad)} differ"))
    return 0 if bad.empty else 1


def cmd_live(args):
    from racinglines.pipelines import live_dh as L
    quali = [k for k in args.quali.split(",") if k]
    if args.once:
        s = L.update(args.slug, args.final, quali, args.conditions)
        print(s["ts"], s["counts"], "maker P&L", {k: round(v, 2) for k, v in (s.get("maker_pnl") or {}).items()})
        return 0
    L.run(args.slug, args.final, quali, args.conditions, interval=args.interval, minutes=args.minutes,
          echo=lambda m: print(m, flush=True))
    return 0
