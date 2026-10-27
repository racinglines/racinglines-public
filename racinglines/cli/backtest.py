"""
racinglines backtest <command>: the backtest core for any sport with a pricing model (docs/backtest-core.md).

    walk-forward SPORT   price every event from data before it (racinglines/core/walk_forward.py), model-only:
                         calibration per market kind; --save stores it as a walk_forward model run (what a
                         search job runs). The sport's pricing model is named in sports/<code>.toml
                         ([sport] pricing_model), and its settings are its flags.

    racinglines backtest walk-forward mtb_dh --seasons 2025 --half-life-days 120 [--save]
    racinglines backtest walk-forward motogp --model global --seasons 2026     # the one engine at the results level
    racinglines backtest walk-forward nascar --model nascar_recent_form        # a challenger ([sport] challengers)
    racinglines backtest walk-forward f1 --model global --seasons 2025 2026 --venue polymarket kalshi
                         # also scores each venue's price beside the model's (model_vs_market): for F1 at the first
                         # stage, 1 h before any running (--market-stage 'after Quali' for another)
    racinglines backtest walk-forward f1 --stage 'after Quali'
                         # every event priced at a stage of the schema's [stages], with the sessions run by then

    coverage             read-only: per sport x exchange x market kind x season, the races linked, settled and taped
                         (prices, trades, books in Postgres and the Parquet archive) and the parity tier that gives;
                         season futures; per sport, the race results in the database and the declared results
                         sources (racinglines/pipelines/coverage.py, docs/parity-rebuild.md "C0")

    racinglines backtest coverage [--seasons 2025 2026] [--out DIR]

    clv                  read-only: closing-line value of every paper bet (strategy_signals: the taker's followed
                         trades, the maker's fills) on any sport and venue, per bet and per sport x venue x kind x
                         strategy x profile: bets, mean CLV, share positive (racinglines/pipelines/clv.py, "C4")

    racinglines backtest clv [--user NAME ...] [--sport nascar] [--venue kalshi] [--event KEY] [--asof UTC] [--summary] [--json]
                             [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def pricing_model(sport, which=None):
    """The sport's pricing model ([sport] pricing_model: the one engine at the level its [model] inputs declare), or
    with which="global" the engine at the results level (finishing positions only: models/inputs/results.py, with the
    schema's [model] / [replay] data hooks), or with which = a name
    in [sport] challengers the challenger model it names (an earlier pricing model kept runnable)."""
    from racinglines.models import race_model as RM
    if which == "global":
        from racinglines.models.model_global import GlobalModel
        return GlobalModel.for_sport(sport, core="results")()
    if which:
        return RM.challenger(sport, which)()
    return RM.get(sport)


WF_OWN_FLAGS = ("venue",)   # walk-forward flags a model setting of the same name must not shadow (F1's sweep venue)


def model_choices(sport):
    """The --model values for `sport`: "global" and the names in its [sport] challengers."""
    from racinglines.models import race_model as RM
    return ("global",) + tuple(RM.challengers(sport)) if sport else ("global",)


def run_walk_forward(model, data, st, seasons=None, kinds=None, out_dir=None, save=False, engine_url=None,
                     echo=print, venues=(), market_stage=None, stage=None, tags=None):
    """The engine, its CSVs and (save) the model run. Returns the engine's output. venues: exchanges whose prices
    are scored beside the model's (pipelines/model_vs_market.py), read at the schema's first stage (market_stage: or
    another stage's label) for a sport with a session schedule, else at the event's cutoff. stage: price every event
    at that stage of the schema's [stages] (a model with at_stage; the records' stage column); None = once, pre-race.
    tags: optional provenance stored in the saved run's params; omitted tags preserve the existing params shape."""
    from racinglines.core import walk_forward as WF
    seasons = seasons or model.seasons(data, st)
    echo(f"Walk-forward {model.sport} {', '.join(map(str, seasons))} · {st.label()} (settings {st.key})"
         + (f" · stage {stage}" if stage is not None else ""))
    from racinglines.db import records as REC
    keep = bool(save and REC.enabled())
    out = WF.run(model, data, st, seasons=seasons, kinds=kinds, echo=echo, keep_sims=keep, stage=stage)
    kept = out.pop("sims", None) if keep else None            # the returned output is the same with records on or off
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("events", "calibration", "reliability"):
            out[name].to_csv(out_dir / f"walk_forward_{name}.csv", index=False)
        if len(out.get("teammates", ())):          # same-team pairs: the driver-vs-car diagnostic's input
            out["teammates"].to_csv(out_dir / "walk_forward_teammates.csv", index=False)
    pd.set_option("display.width", 250)
    cal = out["calibration"]
    echo("\n=== Calibration of the model's fair values (all seasons) ===")
    echo(cal[cal["season"] == "all"].drop(columns=["season", "source"]).to_string(index=False, float_format="{:.4f}".format))
    if out_dir:
        echo(f"\nCSVs -> {out_dir}/")
    at, when = None, "each event's cutoff"
    for venue in venues:
        from racinglines.pipelines import model_vs_market as MVM
        from racinglines.pipelines import scorecard as SC
        if at is None:
            at = MVM.stage_times(model.sport, list(out["cutoffs"]), market_stage, engine_url)
            when = (f"the {market_stage or 'first'} stage" if at else "each event's cutoff")
        paired, by_kind, by_season = MVM.compare(out, venue, engine_url, at=at)
        if not len(paired):
            echo(f"\n=== Model against {venue} ===\nno {venue} market of {', '.join(MVM.KINDS)} is linked to these events "
                 f"(market_links.race_id and athlete_id): nothing to pair")
            continue
        echo(f"\n=== Model against {venue}, prices at {when} ===")
        echo(SC.format_text(by_kind, first=("kind",)))
        echo(SC.format_text(by_season, first=("season", "kind")))
        if out_dir:
            paired.to_csv(out_dir / f"walk_forward_vs_{venue}.csv", index=False)
            by_season.to_csv(out_dir / f"walk_forward_vs_{venue}_scores.csv", index=False)
    if save:
        from racinglines import sports
        from racinglines.db.config import get_session
        from racinglines.db.queries import save_model_run
        params = dict(settings=st.to_json(), settings_key=st.key, model_key=st.model_key, label=st.label(),
                      seasons=list(map(int, seasons)), year=int(seasons[-1]) if len(seasons) == 1 else None,
                      sims=st["sims"], seed=st.rng_seed, sport=model.sport)
        if tags is not None:
            params["tags"] = tags
        with get_session(engine_url) as session:
            run_id = save_model_run(session, competition=sports.load(model.sport)["competition"]["code"],
                                    season=seasons[-1] if len(seasons) == 1 else None, category=st.get("category"),
                                    model=model.name, kind="walk_forward", data_through=model.data_through(data),
                                    params=params, metrics=WF.saved_metrics(out))
        echo(f"Saved walk-forward run {run_id}.")
        if keep and kept:
            frames = [sims.to_records(run_id, model.sport, model.name, ev.season, ev.id, ev.name,
                                      "pre_race" if stage is None else stage, ev.cutoff, kinds=kinds)
                      for ev, sims in kept]
            REC.write(run_id, REC.concat(frames), sims={ev.id: sims for ev, sims in kept})
            echo(f"Prediction records -> {REC.records_dir() / str(run_id)}/")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from racinglines import paths
    from racinglines.pipelines import sweep_settings as SS
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("cmd", nargs="?")
    pre.add_argument("sport", nargs="?")
    pre.add_argument("--model", default=None)
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(prog="racinglines backtest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    wf = sub.add_parser("walk-forward", help="Price every event through the shared engine (model-only).")
    wf.add_argument("sport", help="A sport code with a pricing model (sports/<code>.toml).")
    wf.add_argument("--db", default=None, metavar="URL", help="Database URL (default: $DATABASE_URL).")
    wf.add_argument("--seasons", nargs="+", type=int, help="Seasons (default: every season the model can score).")
    wf.add_argument("--kinds", help="Market kinds, comma-separated (default: every per-entrant kind the model prices).")
    wf.add_argument("--out-dir", help="Write the CSVs here (default: data/runs/<sport>/walk_forward).")
    wf.add_argument("--save", action="store_true", help="Store the run in the database (model_runs, kind walk_forward).")
    wf.add_argument("--model", choices=model_choices(known.sport if known.cmd == "walk-forward" else None), default=None,
                    help="global: the one engine at the results level (models/inputs/results.py); or a challenger named in "
                         "the schema's [sport] challengers (an earlier pricing model). Default: [sport] pricing_model.")
    wf.add_argument("--venue", nargs="+", choices=("polymarket", "kalshi"), default=[],
                    help="Also score each exchange's price beside the model's (exact market links only), read at the "
                         "schema's first stage before any running where the sport has a session schedule (F1), else at the "
                         "event's cutoff.")
    wf.add_argument("--market-stage", default=None, metavar="LABEL",
                    help="Read the --venue prices at this stage of the schema's [stages] instead (F1: 'after Quali').")
    wf.add_argument("--stage", default=None, metavar="LABEL",
                    help="Price every event at this stage of the schema's [stages] (its pre_label, or 'after <session>', "
                         "e.g. F1 'after Quali'): the model reads the sessions run by then (a stage-priced challenger's inputs). "
                         "Default: once before the event, as before.")
    cov = sub.add_parser("coverage", help="Read-only: linked, settled and taped races per sport, exchange, kind and season.")
    cov.add_argument("--db", default=None, metavar="URL", help="Database URL (default: $DATABASE_URL).")
    cov.add_argument("--seasons", nargs="+", type=int, help="Only these seasons (default: all).")
    cov.add_argument("--out", default=None, metavar="DIR", help="Also write coverage_{combos,futures,results}.csv here.")
    clv = sub.add_parser("clv", help="Read-only: closing-line value per paper bet, and per sport, venue, kind and profile.")
    clv.add_argument("--db", default=None, metavar="URL", help="Database URL (default: $DATABASE_URL).")
    clv.add_argument("--user", nargs="+", default=None, metavar="NAME", help="Only these users' bets.")
    clv.add_argument("--sport", default=None, help="Only this sport (a sports/<code>.toml code).")
    clv.add_argument("--venue", default=None, help="Only this venue (polymarket, kalshi, og).")
    clv.add_argument("--event", default=None, metavar="KEY", help="Only this event key.")
    clv.add_argument("--asof", default=None, metavar="UTC", help="Judge closes as of this time (default now).")
    clv.add_argument("--summary", action="store_true", help="Print the aggregate only, not every bet.")
    clv.add_argument("--json", action="store_true", help="Print JSON (summary and bets) instead of tables.")
    clv.add_argument("--out", default=None, metavar="DIR", help="Also write clv_bets.csv and clv_summary.csv here.")
    if known.cmd == "walk-forward" and known.sport:
        SS.add_arguments(wf, pricing_model(known.sport, known.model).Settings, skip=WF_OWN_FLAGS)
    args = ap.parse_args(argv)
    if args.cmd == "coverage":
        from racinglines.db.config import get_engine
        from racinglines.pipelines import coverage as COV
        with get_engine(args.db).connect() as c:
            res = COV.run(c, seasons=args.seasons)
        print(COV.format_text(res))
        if args.out:
            print(f"\nCSVs -> {COV.write(res, args.out)}/")
        return 0
    if args.cmd == "clv":
        import json

        from racinglines.db.config import get_engine
        from racinglines.pipelines import clv as CLV
        with get_engine(args.db).connect() as c:
            res = CLV.run(c, users=args.user, sport=args.sport, venue=args.venue, event=args.event,
                          now=pd.Timestamp(args.asof, tz="UTC") if args.asof else None)
        print(json.dumps(CLV.to_json(res), indent=1) if args.json else CLV.format_text(res, per_bet=not args.summary))
        if args.out:
            out = CLV.write(res, args.out)
            if not args.json:
                print(f"\nCSVs -> {out}/")
        return 0
    model = pricing_model(args.sport, args.model)
    st = SS.from_args(args, model.Settings, skip=WF_OWN_FLAGS)
    if args.stage is not None:
        from racinglines.pipelines import season_sweep as SW
        labels = SW.stage_labels(args.sport, "sessions")
        if args.stage not in labels:
            ap.error(f"--stage {args.stage!r}: {args.sport}'s stages are {list(labels)}")
        if not hasattr(model, "at_stage"):
            ap.error(f"--stage: the {model.name} model prices each event once (no at_stage); the engine and a "
                     "stage-priced challenger take a stage")
    data = model.load(args.db)
    run_walk_forward(model, data, st, seasons=args.seasons, kinds=args.kinds.split(",") if args.kinds else None,
                     out_dir=args.out_dir or paths.runs(args.sport, "walk_forward", mkdir=False), save=args.save,
                     engine_url=args.db, venues=args.venue, market_stage=args.market_stage, stage=args.stage)
    return 0
