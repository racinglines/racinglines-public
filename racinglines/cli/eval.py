"""
racinglines eval <command>: score saved runs on their prediction records (docs/backtest-core.md, "Prediction records").

    compare RUN_A RUN_B   two saved runs paired by event on (event, stage, kind, subject, params): log loss and Brier per
                          kind and season, B - A with its standard error over events and whether 0 is outside ±2 SE
                          (racinglines/pipelines/eval_compare.py). Both runs need records (RACINGLINES_PREDICTION_RECORDS=1
                          with --save). Read-only; writes compare.csv and compare.md under data/runs/eval/.

                          --diagnostics adds the pass test's three tables (reliability buckets, favourite wins, pole-sitter
                          win rate) and writes buckets.csv, favourites.csv and pole.csv beside compare.csv.

    racinglines eval compare 41 52 [--kinds race_win,race_podium] [--stage pre_race] [--out DIR] [--diagnostics]
"""

import argparse


def main(argv):
    ap = argparse.ArgumentParser(prog="racinglines eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="Two saved runs, paired by event on their prediction records, with ±2 SE.")
    c.add_argument("run_a", type=int, help="The baseline run (model_runs.id).")
    c.add_argument("run_b", type=int, help="The challenger run; diff = B - A.")
    c.add_argument("--db", default=None, metavar="URL", help="Database URL for the results (default: $DATABASE_URL).")
    c.add_argument("--kinds", help="Only these market kinds, comma-separated (default: every kind both runs hold).")
    c.add_argument("--stage", default=None, metavar="LABEL", help="Only this stage of the records (default: every stage).")
    c.add_argument("--out", default=None, metavar="DIR", help="Write compare.csv and compare.md here "
                                                              "(default: data/runs/eval/<A>-vs-<B>/).")
    c.add_argument("--diagnostics", action="store_true",
                   help="Also print the pass test's tables (reliability buckets, favourite wins, pole-sitter win rate) and "
                        "write buckets.csv, favourites.csv and pole.csv.")
    args = ap.parse_args(argv)
    from racinglines.pipelines import eval_compare as EC
    try:
        EC.run(args.run_a, args.run_b, kinds=args.kinds.split(",") if args.kinds else None, stage=args.stage,
               out_dir=args.out, engine_url=args.db, diagnostics=args.diagnostics)
    except ValueError as ex:
        print(f"error: {ex}")
        return 2
    return 0
