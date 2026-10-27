"""
`racinglines strategy replay nascar` / `racinglines strategy replay motogp`: the taker replay of a result-only sport against an
exchange's recorded prices (racinglines/pipelines/position_replay.py). Read-only unless --save.

    --venue all            the default: Kalshi then Polymarket (the sport's [markets] venues), then OG.com where
                           exchanges/og.toml lists the sport, one report each. OG.com's stored prices count whatever
                           their spread or depth, except an empty book's 0.50 (markets/venue_replay.py OG)
    --save --backup FILE   also store one as-of model run per race (model_runs + race_predictions + prediction records
                           and sims; records are on for this command unless RACINGLINES_PREDICTION_RECORDS=0); FILE is
                           the database dump taken for this step within the last 24 hours. Logged in data_changes with
                           its batch id. The runs are stored once, with the first venue's pass.
    --undo BATCH           delete the runs a --save pass stored (the batch id it printed).
    --tape probe           read-only: ask each exchange for the tape of 3 markets of the first selected race, print the
                           counts, and stop (no replay, nothing written).
    --tape pull --backup FILE
                           first store the tape the replay reads for the selected races (their markets' trades and
                           hourly prices over each race's window, from 2 days before its first stage), then replay. Logged in data_changes.
    --require-tradeable    exit 1 unless some market was tradeable on some venue (the spot check's pass).
    --buy-all              debug: also buy one YES and one NO share of every market open and priced at a stage, with
                           no edge, volume or coherence filter (markets/strategies/buy_everything.py; off by default,
                           or RACINGLINES_BUY_ALL=1). Reported as the mode buy_all; the taker's modes are unchanged.

A race with markets but no stored price says NO TAPE, and a venue with nothing tradeable prints NOT TRADED instead of
P&L lines.
"""

import sys
import time
from pathlib import Path


def add_parser(sub, sport):
    p = sub.add_parser("replay", help=f"Taker replay of {sport} races against an exchange's recorded prices "
                                      "(read-only unless --save).")
    p.add_argument("--years", default="2025-2026", help="e.g. 2026, 2025-2026 or 2024,2026.")
    p.add_argument("--venue", default="all", choices=["all", "kalshi", "polymarket", "og"],
                   help="all (the default): every exchange the sport lists on, one report each.")
    p.add_argument("--events", default=None, help="Only these event keys, comma list, or 'latest' (the last race of "
                                                  "--years): a spot check. The model still learns from earlier races.")
    p.add_argument("--kinds", default=None, help=f"Comma list, a subset of sports/{sport}.toml [replay] kinds.")
    p.add_argument("--min-edge", type=float, default=None, help="Taker threshold (default: TakerParams, 0.05).")
    p.add_argument("--stake-per-edge", type=float, default=None)
    p.add_argument("--max-stake", type=float, default=None)
    p.add_argument("--cost", type=float, default=None, help="$ per share on every trade (default 0.01).")
    p.add_argument("--min-volume", type=float, default=None, help="24 h volume floor in $ (default 50).")
    p.add_argument("--sims", type=int, default=None, help="Model simulations per race (default: the model's).")
    p.add_argument("--out", default=None, help="Write races.csv, trades.csv, calibration.csv, summary.json here "
                                               "(default data/runs/replay/<sport>-<venue>-<years>/).")
    p.add_argument("--save", action="store_true", help="Also store one as-of model run per race (needs --backup).")
    p.add_argument("--backup", default=None, help="With --save: the database dump taken for this step (under 24 hours old).")
    p.add_argument("--undo", default=None, help="Delete the runs of a previous --save pass (its batch id).")
    p.add_argument("--tape", default=None, choices=["probe", "pull"],
                   help="probe: read-only look at 3 markets' tape on each exchange, then stop. pull: store the selected "
                        "races' trades and prices first (needs --backup), then replay.")
    p.add_argument("--require-tradeable", action="store_true",
                   help="Exit 1 unless some market was tradeable on some venue.")
    p.add_argument("--stages", default="weekend", choices=["weekend", "race_day"],
                   help=f"weekend: sports/{sport}.toml [replay] stages (the default, reproducible); race_day: those "
                        "plus [replay] race_day_stages (opt-in).")
    p.add_argument("--buy-all", action="store_true",
                   help="Debug: also buy one YES and one NO of every priced, open market, no filters (mode buy_all).")
    return p


def _backup_ok(args, sport, why):
    backup = Path(args.backup) if args.backup else None
    if not (backup and backup.is_file() and backup.stat().st_size and time.time() - backup.stat().st_mtime < 86400):
        sys.exit(f"{why}: give --backup FILE, the database dump taken for this step within the last 24 hours "
                 f"(data/backups/db/racinglines-before-{sport}-replay-<UTC>.sql.gz)")


def run(args, sport, years):
    from racinglines import paths
    from racinglines.db import changes
    from racinglines.db.config import get_engine, get_session
    from racinglines.markets.strategies import taker_weekend as RB
    from racinglines.pipelines import position_replay as P

    if args.undo:
        with get_session(args.db) as s:
            n = P.undo(s, args.undo)
            changes.record(s, "replay-undo", f"{sport} replay runs deleted: {n} of batch {args.undo}", sport=sport,
                           detail=dict(batch=args.undo, runs=n))
            s.commit()
        print(f"Deleted {n} runs of batch {args.undo}")
        return 0
    save = None
    if args.save:
        _backup_ok(args, sport, "--save writes model_runs and race_predictions")
        save = dict(engine_url=args.db, batch=P.batch_id())
    if args.tape == "pull":
        _backup_ok(args, sport, "--tape pull writes market_trades and market_price_history")
    over = {k: v for k, v in dict(min_edge=args.min_edge, stake_per_edge=args.stake_per_edge, max_stake=args.max_stake,
                                  cost=args.cost).items() if v is not None}
    taker = RB.TakerParams(**over)
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    engine = get_engine(args.db)
    venues = list(P.replay_venues(sport)) if args.venue == "all" else [args.venue]
    events = args.events.split(",") if args.events else None
    if args.tape:
        for venue in venues:
            plan = P.tape_plan(engine, sport, venue, years, events)
            if args.tape == "probe":
                P.probe(plan, venue)
                continue
            with engine.connect() as c, get_session(args.db) as s:
                n = P.pull(s, c, plan, venue)
                changes.record(s, "tape-pull", f"{sport} {venue} tape pulled for the replay: {n['races']} races, "
                               f"{n['trades']} trades, {n['prices']} prices; backup {Path(args.backup).name}", sport=sport,
                               detail=dict(n, venue=venue, years=years, events=events, backup=Path(args.backup).name))
                s.commit()
            print(f"Pulled {venue}: {n['races']} races, {n['trades']} trades, {n['prices']} prices\n")
        if args.tape == "probe":
            return 0
    data, traded = None, False
    for i, venue in enumerate(venues):
        out = P.run(engine, sport, years, venue=venue, taker=taker, kinds=kinds, data=data,
                    events=events,
                    min_volume_24h=P.MIN_VOLUME_24H if args.min_volume is None else args.min_volume,
                    model_settings={"sims": args.sims} if args.sims else None, save=save if i == 0 else None,
                    buy_all=True if args.buy_all else None, stage_mode=args.stages)
        data = out.pop("data")
        traded = traded or P.traded(out)
        print(P.format_report(out))
        tag = f"{sport}-{venue}-{years[0]}-{years[-1]}" if years else f"{sport}-{venue}"
        folder = P.write(out, Path(args.out) / venue if args.out else paths.runs("replay") / tag)
        print(f"\nWrote {folder}\n")
    if save:
        with get_session(args.db) as s:
            n = s.execute(__import__("sqlalchemy").text(
                "SELECT count(*) FROM model_runs WHERE params->>'replay_batch' = :b"), dict(b=save["batch"])).scalar()
            changes.record(s, "replay-save", f"{sport} replay: {n} as-of model runs stored (batch {save['batch']}); "
                           f"backup {Path(args.backup).name}", sport=sport,
                           detail=dict(batch=save["batch"], runs=n, backup=Path(args.backup).name, venue=args.venue,
                                       years=years))
            s.commit()
        print(f"Stored {n} runs, batch {save['batch']}. Undo: racinglines {sport} replay --undo {save['batch']}")
    if args.require_tradeable and not traded:
        print(f"FAIL: no market was tradeable on {', '.join(venues)} for these races (see NO TAPE / NOT TRADED above)")
        return 1
    return 0


# --- demo-history: the replay's trades as a demo account's paper portfolio (pipelines/sport_paper.py) ------------

def add_demo_parser(sub, sport):
    p = sub.add_parser("demo-history", help=f"Store the {sport} taker replay's trades as a demo account's paper "
                                            "positions (pipelines/sport_paper.py; on unless RACINGLINES_SPORT_PAPER is set to something other than 1/true/yes/on).")
    p.add_argument("--grid", default=None, help=f"The settings grid to pick from (data/runs/replay-grid/{sport}).")
    p.add_argument("--pick", default=None, help="Print the selection from this grid folder and write nothing.")
    p.add_argument("--book", default="kinds", choices=["kinds", "blend", "best"],
                   help="kinds (default): each market kind at its best setting; blend: every kind at the one best setting; best: best effort (best total per kind; if none is positive, every kind at the best total, win or lose).")
    p.add_argument("--venue", default="kalshi", choices=["kalshi", "polymarket"])
    p.add_argument("--users", default="taker", help="Comma list of demo accounts (default: taker).")
    p.add_argument("--events", default=None, help="Only these event keys, comma list, or 'latest': a spot check.")
    p.add_argument("--reset", action="store_true", help="Delete this sport's demo rows on --venue for --users first "
                                                       "(alone: delete only).")
    p.add_argument("--backup", default=None, help="The database dump taken for this step (under 24 hours old).")
    p.add_argument("--stages", default="weekend", choices=["weekend", "race_day"],
                   help=f"weekend: sports/{sport}.toml [replay] stages (the default, reproducible); race_day: those "
                        "plus [replay] race_day_stages (opt-in).")
    p.add_argument("--grid-venue", default=None, choices=["kalshi", "polymarket"],
                   help="Pick the settings from this venue's grid runs (default: --venue). Polymarket has no grid: "
                        "--venue polymarket --grid-venue kalshi trades Polymarket's tape with Kalshi's selection.")
    return p


def run_demo(args, sport):
    from racinglines.db import changes
    from racinglines.db.config import get_engine, get_session
    from racinglines.pipelines import sport_paper as SP

    if args.pick:
        print(SP.selection_md(sport, SP.pick(args.pick, getattr(args, "grid_venue", None) or args.venue)))
        return 0
    if not SP.enabled():
        sys.exit(f"{SP.SWITCH} is off: set {SP.SWITCH}=1 to write (and show) the demo paper portfolio")
    if not (args.grid or args.reset):
        sys.exit("give --grid FOLDER (the settings grid to pick from), --reset, or --pick FOLDER")
    _backup_ok(args, sport, "demo-history writes strategy_signals and paper_positions")
    engine = get_engine(args.db)
    users = [u.strip() for u in args.users.split(",") if u.strip()]
    backup = Path(args.backup).name
    if args.reset:
        n, m = SP.reset(engine, users, sport, args.venue)
        with get_session(args.db) as s:
            changes.record(s, "demo-history-reset", f"{sport} demo paper rows deleted on {args.venue} for {', '.join(users)}: "
                           f"{n} signals, {m} positions; backup {backup}", sport=sport,
                           detail=dict(signals=n, positions=m, users=users, venue=args.venue, backup=backup))
            s.commit()
        print(f"Deleted {n} signals and {m} positions ({sport}, {args.venue}, {', '.join(users)})")
        if not args.grid:
            return 0
    sel = SP.pick(args.grid, getattr(args, "grid_venue", None) or args.venue)
    md = SP.selection_md(sport, sel)
    print(md)
    settings = SP.settings_for(sel, args.book)
    if not settings:
        print(f"Nothing to trade: no {args.book} setting made money in its worse season. Nothing written.")
        return 0
    (Path(args.grid) / f"demo-selection-{args.venue}.md").write_text(md)
    events = args.events.split(",") if args.events else None
    rep = SP.backfill(engine, sport, settings, usernames=users, seasons=sel["seasons"], venue=args.venue,
                      events=events, book=args.book, echo=lambda m: print(m, flush=True),
                      stage_mode=getattr(args, "stages", "weekend"))
    for u in sorted({r[0] for r in rep}):
        for y in sel["seasons"]:
            rs = [r for r in rep if r[0] == u and r[1].startswith(str(y))]
            if rs:
                print(f"{u} {y}: {len(rs)} races, paper P&L after fees {sum(r[4] for r in rs):+.2f}")
    with get_session(args.db) as s:
        changes.record(s, "demo-history", f"{sport} demo paper portfolio ({args.book}, {args.venue}, in-sample) for "
                       f"{', '.join(users)}: {len(rep)} account-races; backup {backup}", sport=sport,
                       detail=dict(book=args.book, venue=args.venue, users=users, events=events, rows=len(rep),
                                   settings={f"{e:g}/{v:g}": k for (e, v), k in settings.items()}, backup=backup))
        s.commit()
    print(f"Undo: racinglines {sport} demo-history --reset --venue {args.venue} --users {','.join(users)} --backup FILE")
    return 0


# --- forecast: the sport's next races, priced by its model (pipelines/sport_forecast.py) ---------------------------

def add_forecast_parser(sub, sport):
    p = sub.add_parser("forecast", help=f"Price the next {sport} races with the sport's model (read-only unless --save).")
    p.add_argument("--races", type=int, default=3, help="How many scheduled races (default 3).")
    p.add_argument("--save", action="store_true", help="Store the forecast as a model run (needs --backup).")
    p.add_argument("--backup", default=None, help="With --save: the database dump taken for this step (under 24 hours old).")
    p.add_argument("--undo", type=int, default=None, help="Delete a forecast run this command stored (its run id).")
    return p


def run_forecast(args, sport):
    from racinglines.db import changes
    from racinglines.db.config import get_engine, get_session
    from racinglines.pipelines import sport_forecast as SF
    if args.undo:
        with get_session(args.db) as s:
            n = SF.undo(s, args.undo)
            if n:
                changes.record(s, "forecast-undo", f"{sport} forecast run {args.undo} deleted", sport=sport,
                               detail=dict(run=args.undo))
            s.commit()
        print(f"Deleted {n} run(s)")
        return 0 if n else 1
    if args.save:
        _backup_ok(args, sport, "forecast --save writes model_runs and race_predictions")
    engine = get_engine(args.db)
    fc = SF.forecast(engine, sport, n=args.races)
    with engine.connect() as c:
        print(SF.table(c, fc))
    if not args.save or not fc["races"]:
        return 0
    rid = SF.save(engine.url.render_as_string(hide_password=False), fc)
    with get_session(args.db) as s:
        changes.record(s, "forecast", f"{sport} forecast run {rid}: {len(fc['races'])} races, field from "
                       f"{fc['field_from']}; backup {Path(args.backup).name}", sport=sport,
                       detail=dict(run=rid, races=[x['race'].event_key for x in fc['races']], backup=Path(args.backup).name))
        s.commit()
    print(f"Stored forecast run {rid}. Undo: racinglines {sport} forecast --undo {rid}")


# --- `season-replay`: the champion markets (pipelines/season_replay.py) ---------------------------------------------

def add_season_parser(sub, sport):
    p = sub.add_parser("season-replay", help=f"Champion-market replay of the {sport} season against an exchange's "
                                             "recorded prices (read-only; off unless RACINGLINES_SEASON_REPLAY=1).")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--venue", default="all", choices=["all", "kalshi", "polymarket", "og"],
                   help="all (the default): every exchange the sport lists on, one report each.")
    p.add_argument("--sims", type=int, default=None, help="Season simulations per decision (default 2000).")
    p.add_argument("--min-edge", type=float, default=None, help="Default 0.03, the F1 championship sleeve's.")
    p.add_argument("--stake-per-edge", type=float, default=None)
    p.add_argument("--max-stake", type=float, default=None)
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--out", default=None, help="Write decisions.csv, trades.csv, positions.csv, equity.csv, "
                                               "summary.json here (default data/runs/season-replay/<sport>-<venue>-<year>/).")
    return p


def run_season(args, sport):
    from racinglines import paths
    from racinglines.db.config import get_engine
    from racinglines.pipelines import position_replay as P
    from racinglines.pipelines import season_replay as SR
    if not SR.enabled():
        sys.exit(f"racinglines strategy season-replay {sport} is off by default: set {SR.SWITCH}=1 to run it (read-only)")
    over = {k: v for k, v in dict(min_edge=args.min_edge, stake_per_edge=args.stake_per_edge, max_stake=args.max_stake,
                                  capital=args.capital).items() if v is not None}
    params = SR.SS.SeasonParams(**{**SR.DEFAULT_PARAMS.__dict__, **over})
    engine = get_engine(args.db)
    venues = list(P.replay_venues(sport)) if args.venue == "all" else [args.venue]
    for venue in venues:
        out = SR.run(engine, sport, args.year, venue=venue, params=params, n_sims=args.sims or SR.N_SIMS,
                     echo=lambda m: print(m, flush=True))
        print(SR.format_report(out))
        folder = SR.write(out, Path(args.out) / venue if args.out else
                          paths.runs("season-replay") / f"{sport}-{venue}-{args.year}")
        print(f"\nWrote {folder}\n")
    return 0


# --- sweep: the season sweep, taker modes and maker variants (pipelines/season_sweep.py) -------------------------

def add_sweep_parser(sub, sport):
    """`racinglines strategy sweep <sport>`: the season sweep on the sport's [sweep] engine, as `racinglines strategy sweep f1` (the
    search runs both the same way: sweeps/*.toml jobs of kind "sweep")."""
    from racinglines.pipelines import season_sweep as S
    from racinglines.pipelines import sweep_settings as SS
    p = sub.add_parser("sweep", help=f"Season sweep of {sport}: the taker modes and maker variants through each race's "
                                     f"stages on one exchange's recorded tape (read-only unless --save).")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--rounds", default=None, help="Events within the season, 1 = the first raced, e.g. 1-8 "
                                                  "(default: every raced event).")
    p.add_argument("--no-fetch", action="store_true", help="Accepted for the search; nothing is downloaded (the tape is "
                                                           "what the recorder and archive hold).")
    p.add_argument("--save", action="store_true", help="Store the sweep (model_runs kind 'sweep', params with sport "
                                                       "and venue).")
    p.add_argument("--reliability", action="store_true", help="Also print and save the calibration tables.")
    p.add_argument("--trades-out", default=None, metavar="DIR",
                   help="Also write every trade (touch bid / ask / sizes, outcome) to DIR/trades.parquet.")
    SS.add_arguments(p.add_argument_group(f"settings (sports/{sport}.toml [sweep]; default = baseline)"),
                     S.settings_class(sport))
    return p


def run_sweep(args, sport):
    from racinglines.db.config import get_engine
    from racinglines.pipelines import season_sweep as S
    from racinglines.pipelines import sweep_settings as SS
    st = SS.from_args(args, S.settings_class(sport))
    rounds = _numbers(args.rounds) if args.rounds else None
    out = S.run(get_engine(args.db), args.db, sport, args.year, rounds, settings=st,
                echo=lambda m: print(m, flush=True))
    if getattr(args, "trades_out", None):
        from racinglines.pipelines import weekend_sweep as SW
        print(f"Trades -> {SW.write_trades(out, args.trades_out)}", flush=True)
    w = out["weekends"]
    cols = [c for c in ["event_key", "event", "stages", "markets", "tradeable_pre", "tradeable_quali", "update_trades",
                        "update_bought", "update_pnl", "hold_pnl", "last_pnl", "early_pnl", "maker_fills", "maker_pnl"]
            if c in w]
    p = out["params"]
    print(f"\n=== {sport} {args.year} on {p['venue']} ({p['label']}) ===")
    if len(w):
        print(w[cols].to_string(index=False, float_format="{:.2f}".format))
    print("\n=== Totals ===")
    from racinglines.pipelines import clv as CLV
    for k, v in out["totals"].items():
        print(f"{k:11s} P&L {v['pnl']:+9.2f} on ${v['bought']:,.0f} · {v['weekends_up']}/{v['weekends']} races up"
              f" · {CLV.sweep_line(out.get('clv'), k)}")
    if len(out["by_kind"]):
        print("\n=== Update strategy by market kind ===")
        print(out["by_kind"].to_string(index=False, float_format="{:+.2f}".format))
    if args.reliability and len(out["calibration"]):
        print("\n=== Calibration: model vs exchange, every tradeable stage (lower is better) ===")
        print(out["calibration"].to_string(index=False, float_format="{:.4f}".format))
    if args.save:
        print(f"Saved sweep run {S.save(args.db, sport, args.year, args.rounds, out, args.reliability)}.", flush=True)
    return 0


def _numbers(spec):
    """"3", "1-8" or "1,4,6" -> a list of ints."""
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


# --- signals: live paper signals of the sport's profiles (pipelines/sport_signals.py) ----------------------------

def add_signals_parser(sub, sport):
    from racinglines.pipelines import season_sweep as S
    p = sub.add_parser("signals", help=f"Live paper signals of users' {sport} strategy profiles on every exchange the "
                                       "sport lists (pipelines/sport_signals.py). Recommendations only: never places an order.")
    p.add_argument("--venue", default="all", choices=["all", *S.venues(sport)],
                   help="all (the default): every exchange of the sport's schema, each with its own users' profiles.")
    p.add_argument("--profile", default=None, help="Candidate id or name (default: each user's own).")
    p.add_argument("--user", nargs="*", default=None, help="Only these usernames.")
    p.add_argument("--event", default="next", help="next (default) or an event key.")
    p.add_argument("--asof", default=None, help="Replay at this UTC time (prints only; markets read at each stage's "
                                               "time, the race priced on its classified field, like the sweep).")
    p.add_argument("--no-fetch", action="store_true", help="Don't refresh the exchange's tape first.")
    p.add_argument("--no-alert", action="store_true")
    return p


def run_signals(args, sport):
    from racinglines.db.config import get_engine
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import season_sweep as S
    from racinglines.pipelines import signals as SG
    from racinglines.pipelines import sport_signals as SPS
    engine = get_engine(args.db)
    venues = list(S.venues(sport)) if args.venue == "all" else [args.venue]
    echo = lambda m: print(m, flush=True)        # noqa: E731
    if args.asof:
        for venue in venues:
            with engine.connect() as c:
                profs = [dict(PF.load(c, args.profile), sport=sport)] if args.profile else [
                    p for _, n, _, p in PF.assigned(c, venue=venue, sport=sport) if not args.user or n in args.user]
            for prof in profs:
                out = SPS.compute(engine, args.db, prof, now=args.asof, event=args.event, live=False, venue=venue,
                                  fetch=False, echo=echo)
                print(SG.format_replay(out))
        return 0
    rep = SPS.run_all(engine, args.db, sport, venues=venues, users=args.user, profile_ref=args.profile,
                      event=args.event, fetch=not args.no_fetch, alert=not args.no_alert, echo=echo)
    if not rep:
        print(f"no user has a {sport} strategy profile on {', '.join(venues)}")
    return 0
