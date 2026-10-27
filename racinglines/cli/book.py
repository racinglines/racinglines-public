"""
racinglines book <command> FILE [options]: generic sportsbook lines and slips against our model and the prediction
markets (racinglines/books/slips.py, docs/sportsbook/slips.md). CLI only, outside the web app; read-only on the
database.

    map      resolve every line's legs to a race, athletes and a kind by exact keys; list what stays unmapped
    price    + the model's fair, the linked exchange price, the book's implied probability, EV against each
    settle   + each leg's and line's result from the stored classification, and the payout per unit staked
    render   one HTML page (and --png) of the priced board: hud (on the book's screenshot), card, or agnostic (model
             and exchanges only), with stakes from markets/books.toml (docs/sportsbook/render.md)
    new      a book file draft from loose text ("A over B 1.57", "A vs B 1.57 2.25", "Name 4.35"): --from-text
    export   the book's staked lines (a line's `stake`, `placed_utc`) priced and settled, as ledger rows:
             `--ledger OUT.csv` (racinglines/books/ledger.py, docs/sportsbook/slips.md#ledger-export); refuses an
             existing OUT unless --force, never appends
    view     render a saved --json result of map, price or settle with Rich (no database)
    logout   forget the browser sign-in that --remote keeps
    config   show or set the CLI's defaults (racinglines/cli/config.py): `config`, `config KEY VALUE`, `config KEY --unset`

Where the data comes from (map, price, settle, export), three modes; the default is the defaults file's `source`, else local:
    --local                a local database (--db, else DATABASE_URL, else the defaults file's `db`): you rebuild the
                           data or run the app and its MCP server yourself
    --demo [PACK]          a demo store (racinglines/cli/source.py): a separate local database of synthetic data from a
                           data pack (default kart), built on first use, priced by the same code as --local. FILE
                           defaults to the pack's book (racinglines/demo/packs/kart/book.toml); `racinglines model
                           forecast kart --demo --save` first gives it model prices. Rich by default.
                           `--demo [PACK] --remote` (or the defaults file's source = remote) prices it on the pack's
                           hosted demo store instead (its [remote] mcp_url; synthetic data only)
    --remote               the VM's hosted MCP server (racinglines/cli/remote.py): live model runs and exchange prices.
                           Signs in through the browser the first time (kept in ~/.config/racinglines/mcp-login.json),
                           or uses RACINGLINES_MCP_TOKEN (an rl_ token) when set. --mcp-url URL, RACINGLINES_MCP_URL or
                           the defaults file's `mcp_url` point it at your own server (default
                           https://mcp.racinglines.bet/mcp)

Options (map, price, settle, export):
    --json                 the whole result as JSON (stable keys: docs/sportsbook/slips.md#json-output)
    --rich                 a coloured table (Rich; racinglines/cli/rich_view.py): the default; --plain (any command) or
                           RACINGLINES_PLAIN=1 prints the plain text
    --db URL               the database (default DATABASE_URL)
price and export:
    --run ID               price legs of that run's competition from it (repeatable; default: the app's own choice
                           per race, markets/venues.pricing_run)
    --sims RACE_ID=FILE    an OutcomeSims archive (.npz, models/outcomes.save_sims) for a race: legs on that race are
                           priced jointly from it (repeatable)
price:
    --asof UTC             price as it stood then (follow a book through a weekend): the newest model run cut off by
                           then and each exchange's last tape price by then, no bid/ask (local only)
render (FILE = a book file, priced like `price`; or a cycling price output folder or CSV; or leave it out with
--json-in):
    --style hud|card|agnostic   --image PNG   --out PATH.html   --png   --chrome PATH   --json-in PRICED.json
    --bank --kelly --cap --min-ev --scale --drop-under    override markets/books.toml [stake]
    --require model                                       size on the model's EV alone (no exchange quotes)
new:
    --from-text TXT --venue CODE --event KEY --odds FORMAT --out FILE.toml [--sport CODE] [--currency USD]
"""

import argparse
import importlib.util
import json
import os
import sys

import pandas as pd

BOOK_COMMANDS = ("map", "price", "settle", "export")


def _parser():
    p = argparse.ArgumentParser(prog="racinglines book", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    top = p.add_mutually_exclusive_group()                 # also before the command: `book --demo price`
    for flag in ("demo", "remote", "local"):
        top.add_argument(f"--{flag}", dest=f"top_{flag}", action="store_true", help=f"as `--{flag}` after the command")
    p.add_argument("--mcp-url", dest="top_mcp_url", default=None, help="as `--mcp-url` after the command")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="An HTML page (and PNG) of the priced board: hud, card or agnostic.")
    r.add_argument("file", nargs="?", help="a book file, or a cycling price output folder / futures.csv / matchups.csv")
    r.add_argument("--style", choices=("hud", "card", "agnostic"), default="card")
    r.add_argument("--image", help="the book's screenshot (hud; default: [book] image in the book file)")
    r.add_argument("--out", help="the HTML file (default: <file stem>-<style>.html next to the input)")
    r.add_argument("--png", action="store_true", help="also write a PNG beside the HTML (needs a Chromium)")
    r.add_argument("--chrome", help="the Chromium to use (default: $RACINGLINES_CHROME, then PATH, then Playwright's)")
    r.add_argument("--json-in", help="a saved `book price --json` result: render without a database")
    r.add_argument("--db", default=None, help="database URL (default DATABASE_URL)")
    r.add_argument("--run", type=int, action="append", default=[], help="a model run id (repeatable)")
    for opt, hlp in (("bank", "bank in $"), ("kelly", "Kelly fraction"), ("cap", "per venue per event cap in $"),
                     ("min-ev", "minimum EV against each required price"), ("scale", "multiply every stake"),
                     ("drop-under", "drop stakes under this, after --scale")):
        r.add_argument(f"--{opt}", type=float, default=None, help=hlp + " (default: markets/books.toml [stake])")
    r.add_argument("--require", nargs="+", choices=["model", "market"], default=None,
                   help="the prices the EV must clear (default: markets/books.toml [stake] require = model and market); "
                        "--require model sizes on the model alone, for a sport with no exchange quotes")
    n = sub.add_parser("new", help="A book file draft from loose text lines.")
    n.add_argument("--from-text", required=True, help="a text file, one bet per line ('-' for stdin)")
    n.add_argument("--venue", required=True, help="the venue code, e.g. book_a")
    n.add_argument("--event", required=True, help="the event key, e.g. 2026-17")
    n.add_argument("--odds", default="decimal", choices=("decimal", "american", "fractional", "cents", "dollars", "prob"))
    n.add_argument("--sport", help="[book] sport (sports/<code>.toml)")
    n.add_argument("--currency", default="USD")
    n.add_argument("--out", required=True, help="the book file to write (refuses to overwrite)")
    for name, hlp in (("map", "Resolve each line's legs by exact keys."),
                      ("price", "Model fair, market price, book implied probability and EV per leg and per slip."),
                      ("settle", "Results and payouts from the stored classification."),
                      ("export", "The staked lines, priced and settled, as ledger rows (--ledger OUT.csv).")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("file", nargs="?", help="a book file (books/<venue>/<date>-<event>.toml, "
                                               "docs/sportsbook/index.md); with --demo, default the pack's book")
        s.add_argument("--json", action="store_true", help="print the result as JSON")
        s.add_argument("--rich", action="store_true", help="print a coloured terminal table (Rich)")
        s.add_argument("--db", default=None, help="database URL (default DATABASE_URL)")
        mode = s.add_mutually_exclusive_group()
        mode.add_argument("--demo", action="store_true",
                          help="a demo store: a local database of synthetic data from a data pack (`--demo PACK` for another pack; default kart)")
        mode.add_argument("--remote", action="store_true",
                          help="the VM's hosted MCP server: browser sign-in, or RACINGLINES_MCP_TOKEN")
        mode.add_argument("--local", action="store_true", help="a local database (the default unless the defaults "
                                                               "file says otherwise)")
        s.add_argument("--mcp-url", default=None, help="the hosted server for --remote (default RACINGLINES_MCP_URL, "
                                                       "else https://mcp.racinglines.bet/mcp)")
        if name == "export":
            s.add_argument("--ledger", required=True, metavar="OUT.csv",
                           help="write the ledger rows here (a new file: refuses an existing one unless --force)")
            s.add_argument("--force", action="store_true", help="replace an existing OUT (never appends)")
        if name in ("price", "export"):
            s.add_argument("--run", type=int, action="append", default=[], help="a model run id (repeatable)")
            s.add_argument("--sims", action="append", default=[], metavar="RACE_ID=FILE",
                           help="simulations for one race, for joint same-race pricing (repeatable)")
        if name == "price":
            s.add_argument("--asof", default=None, metavar="UTC",
                           help="price as it stood then: the newest model run cut off by then, each exchange's last "
                                "tape price by then (local only; e.g. 2026-10-09T10:05Z)")
    v = sub.add_parser("view", help="Render a saved --json result with Rich (no database).")
    v.add_argument("file", help="the JSON a map, price or settle --json printed ('-' for stdin)")
    lo = sub.add_parser("logout", help="Forget the browser sign-in --remote keeps.")
    lo.add_argument("--mcp-url", default=None, help="only this server's sign-in (default: every server's)")
    cf = sub.add_parser("config", help="Show or set the CLI's defaults (source, mcp_url, db).")
    cf.add_argument("key", nargs="?", help="source (local / remote / demo), mcp_url or db")
    cf.add_argument("value", nargs="?")
    cf.add_argument("--unset", action="store_true", help="remove KEY")
    return p


def _config(C, args):
    if args.key is None:
        data = C.load()
        print(f"# {C.path()}" + ("" if data else " (none: every default applies)"))
        for k in C.KEYS:
            if k in data:
                print(f"{k} = {'<set>' if k == 'db' else data[k]}")       # a db URL can carry a password
        return 0
    if not args.unset and args.value is None:
        print(f"book config {args.key}: give a VALUE, or --unset", file=sys.stderr)
        return 2
    C.set_key(args.key, None if args.unset else args.value)
    print(f"{'removed' if args.unset else 'set'} {args.key} in {C.path()}")
    return 0


def _kind(out):
    """Which command made a result dict: price adds assumptions, settle adds a result to every leg."""
    if "assumptions" in out:
        return "price"
    return "settle" if any("result" in lg for r in out.get("lines", []) for lg in r.get("legs", [])) else "map"


def _sims(specs):
    from racinglines.models.outcomes import load_sims
    out = {}
    for spec in specs:
        rid, _, path = spec.partition("=")
        if not rid.isdigit() or not path:
            raise SystemExit(f"--sims {spec!r}: expected RACE_ID=FILE")
        out[int(rid)] = load_sims(path)
    return out


def _runs(conn, ids):
    from racinglines.db.reads import model_run
    out = {}
    for rid in ids:
        r = model_run(conn, rid)
        if r is None:
            raise SystemExit(f"--run {rid}: no such model run")
        out[rid] = r["competition"]
    return out


def run(cmd, book, conn, runs=(), sims=(), asof=None):
    """The command's result dict (what --json prints)."""
    from racinglines.books import slips as B
    B.readonly(conn)
    if cmd == "map":
        return B.map_book(conn, book)
    if cmd == "price":
        return B.price_book(conn, book, runs=_runs(conn, runs), sims=_sims(sims), asof=asof)
    return B.settle_book(conn, book)


TEXT_COLUMNS = {
    "map": (["leg", "status", "sport", "event_key", "race_id", "kind", "side", "athlete", "opponent", "team",
             "threshold", "reason"], ["id", "title", "selection", "kind", "status", "n_legs", "odds", "reason"]),
    "price": (["leg", "sport", "event_key", "kind", "side", "athlete", "opponent", "team", "odds", "book_prob",
               "model_prob", "run_id", "market_prob", "market_exchange", "ev_model", "ev_market", "model_note"],
              ["id", "selection", "n_legs", "decimal_odds", "book_prob", "model_prob", "model_method", "market_prob",
               "ev_model", "ev_market", "flags"]),
    "settle": (["leg", "sport", "event_key", "kind", "side", "athlete", "opponent", "team", "result"],
               ["id", "selection", "n_legs", "decimal_odds", "result", "payout", "profit"]),
}


def _print(cmd, out):
    from racinglines.books import slips as B
    b = out["book"]
    print(f"{b['venue']} · event {b['event']} · captured {b['captured_utc']} · odds {b['odds']}")
    legs, lines = B.leg_table(out), B.line_table(out)
    lc, rc = TEXT_COLUMNS[cmd]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 80):
        if len(legs):
            print("\nLegs\n" + legs[[c for c in lc if c in legs]].to_string(index=False))
        print("\nLines\n" + lines[[c for c in rc if c in lines]].to_string(index=False))
    if out["unmapped"]:
        print(f"\nUnmapped lines: {', '.join(out['unmapped'])} (exact keys only: fix the book file or its [aliases])")
    if cmd == "price":
        print("\nAssumptions:\n" + "\n".join(f"  - {a}" for a in out["assumptions"]))


def _render(args):
    from pathlib import Path

    from racinglines.books import render as R
    from racinglines.books import slips as B
    overrides = {k.replace("-", "_"): getattr(args, k.replace("-", "_"))
                 for k in ("bank", "kelly", "cap", "min-ev", "scale", "drop-under")}
    if getattr(args, "require", None):
        overrides["require"] = list(args.require)
    st = R.load_settings(overrides)
    src = Path(args.file) if args.file else None
    book = None
    if src is not None and (src.is_dir() or src.suffix == ".csv"):
        meta, rows = R.rows_from_cycling(src)
    else:
        if src is not None and src.suffix == ".json" and not args.json_in:
            args.json_in, src = str(src), None
        if src is not None:
            book = B.load_file(src)
        if args.json_in:
            priced = json.loads(Path(args.json_in).read_text())
        elif book is not None:
            from racinglines.db.config import get_engine
            with get_engine(args.db).connect() as conn:
                priced = run("price", book, conn, args.run)
                conn.rollback()
        else:
            raise R.RenderError("give a book file, a cycling price output, or --json-in PRICED.json")
        meta, rows = R.rows_from_priced(priced, st, book)
    image = args.image
    if image is None and meta.get("image") and src is not None:
        image = str((src.parent / meta["image"]))
    stem = (src or Path(args.json_in)).resolve()
    out = Path(args.out) if args.out else stem.parent / f"{stem.stem}-{args.style}.html"
    page, sized, summary = R.render(meta, rows, args.style, st, image)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out}")
    if args.style != "agnostic":
        for r in sized:
            if r["verdict"] == "BET":
                print(f"  BET ${r['stake']:7.2f}  {r['selection']}" + (f" over {r['opponent']}" if r["opponent"] else "")
                      + f" @ {r['decimal_odds']:.2f}")
        print(f"  {summary['n_bets']} bets, ${summary['total']:.2f}" + (" (scaled to the cap)" if summary["capped"] else ""))
    if args.png:
        png = R.to_png(out, out.with_suffix(".png"), args.chrome, st["render"]["dpr"][args.style])
        print(f"wrote {png}")
    return 0


def _new(args):
    import sys as _sys
    from pathlib import Path

    from racinglines.books import render as R
    text = _sys.stdin.read() if args.from_text == "-" else Path(args.from_text).read_text()
    lines, bad = R.parse_text(text, args.odds)
    out = Path(args.out)
    if out.exists():
        raise R.RenderError(f"{out} exists: not overwritten")
    if not lines:
        raise R.RenderError("no line parsed: " + "; ".join(f"line {n}: {s!r} ({why})" for n, s, why in bad))
    out.write_text(R.book_toml(lines, bad, args.venue, args.event, args.odds, args.sport, args.currency,
                               source_ref=f"from {args.from_text}"))
    print(f"wrote {out}: {len(lines)} lines, every market \"unmapped\" (write the market tables, then `book map`)")
    for n, s, why in bad:
        print(f"  not parsed, line {n}: {s!r} ({why})")
    return 0


def _export(args, book):
    """`book export`: price and settle the book (locally or on the hosted server), then write the ledger rows."""
    from racinglines.books import ledger as L
    if os.path.exists(args.ledger) and not args.force:
        print(f"book export: {args.ledger} exists: give --force to replace it (the exporter never appends)",
              file=sys.stderr)
        return 2
    try:
        L.check_units(book)
        if args.remote:
            from racinglines.cli import remote
            with open(args.file) as f:
                raw = f.read()
            try:
                priced = remote.run("price", raw, args.mcp_url, args.run)
                settled = remote.run("settle", raw, args.mcp_url)
            except remote.RemoteError as e:
                print(f"book --remote: {e}", file=sys.stderr)
                return 2
            rows, skipped = L.rows(book, priced, settled)
        else:
            from racinglines.books import slips as B
            from racinglines.db.config import get_engine
            with get_engine(args.db).connect() as conn:
                B.readonly(conn)
                rows, skipped = L.ledger_rows(conn, book, runs=_runs(conn, args.run), sims=_sims(args.sims))
                conn.rollback()
        L.write_csv(rows, args.ledger, force=args.force)
    except L.LedgerError as e:
        print(f"book export: {e}", file=sys.stderr)
        return 2
    settled_n = sum(r["result"] != "open" for r in rows)
    print(f"wrote {len(rows)} bet(s) to {args.ledger} ({settled_n} settled by our own results, "
          f"{len(rows) - settled_n} open)")
    if skipped:
        print(f"book export: {len(skipped)} line(s) without a stake skipped: {', '.join(skipped)}", file=sys.stderr)
    return 0


def main(argv=None):
    from racinglines.cli import tty as TTY
    parser = _parser()
    args = parser.parse_args(argv)
    tops = [f for f in ("demo", "remote", "local") if getattr(args, f"top_{f}")]
    if tops and args.cmd not in BOOK_COMMANDS:
        parser.error(f"--{tops[0]} applies to map, price, settle and export")
    if tops:
        if any(getattr(args, f) for f in ("demo", "remote", "local") if f not in tops):
            parser.error("give one of --demo, --remote, --local")
        setattr(args, tops[0], True)
    if args.cmd in ("render", "new"):
        from racinglines.books import render as R
        from racinglines.books import schema as S
        try:
            return _render(args) if args.cmd == "render" else _new(args)
        except (R.RenderError, S.BookError, OSError) as e:
            print(f"book {args.cmd}: {e}", file=sys.stderr)
            return 2
    if args.top_mcp_url and args.cmd in BOOK_COMMANDS + ("logout",):
        args.mcp_url = args.mcp_url or args.top_mcp_url
    if args.cmd == "view":
        from racinglines.cli import rich_view
        try:
            out = json.load(sys.stdin if args.file == "-" else open(args.file))
        except (OSError, ValueError) as e:
            print(f"book view: {e}", file=sys.stderr)
            return 2
        rich_view.render(_kind(out), out)
        return 0
    from racinglines.cli import config as C
    if args.cmd == "logout":
        from racinglines.cli import remote
        print("signed out" if remote.logout(args.mcp_url) else "no stored sign-in")
        return 0
    try:
        if args.cmd == "config":
            return _config(C, args)
        defaults = C.load()
    except ValueError as e:
        print(f"book: {e}", file=sys.stderr)
        return 2
    from racinglines.cli import source as SRC
    if SRC.demo_active():                                  # `--demo [PACK]`, taken out by the CLI's source resolver
        if args.remote:                                    # the pack's hosted store (source.use_remote_demo)
            args.file = args.file or (str(SRC.pack().book) if SRC.pack().book else None)
            from racinglines.cli import remote as RM
            RM.warn()                                      # the testing warning first, as on every remote run
            print(f"demo: the {SRC.demo_active()} pack's hosted demo store (synthetic data); nothing is traded",
                  file=sys.stderr)
        else:
            args.demo, args.local = True, False
    if not (args.demo or args.remote or args.local):
        source = defaults.get("source", "local")
        args.demo, args.remote = source == "demo", source == "remote"
    args.mcp_url = args.mcp_url or os.environ.get("RACINGLINES_MCP_URL") or defaults.get("mcp_url")
    if not args.db and not os.environ.get("DATABASE_URL"):
        args.db = defaults.get("db")
    if args.demo:
        if args.db and args.db != defaults.get("db"):
            print("book --demo: prices on the demo store; leave out --db (--local --db URL for your own)", file=sys.stderr)
            return 2
        args.db = SRC.use_demo()
        args.file = args.file or (str(SRC.pack().book) if SRC.pack().book else None)
        print("demo: the demo store's synthetic data (made-up names, results and prices); nothing is traded",
              file=sys.stderr)
        if not args.json and not args.rich and not importlib.util.find_spec("rich"):
            print("(plain text: pip install rich for the coloured tables)", file=sys.stderr)
    if not args.file:
        print(f"book {args.cmd}: a book FILE is needed (or --demo for a demo pack's book)", file=sys.stderr)
        return 2
    from racinglines.books import schema as S
    from racinglines.books import slips as B
    from racinglines.db.config import get_engine
    try:
        book = B.load_file(args.file)
    except (S.BookError, OSError) as e:
        print(f"book: {e}", file=sys.stderr)
        return 2
    if args.cmd == "export":
        return _export(args, book)
    if args.remote:
        from racinglines.cli import remote
        if (args.db and args.db != defaults.get("db")) or getattr(args, "sims", None) or getattr(args, "asof", None):
            print("book --remote: runs on the hosted server; --db, --sims and --asof are local only", file=sys.stderr)
            return 2
        try:
            out = remote.run(args.cmd, open(args.file).read(), args.mcp_url, getattr(args, "run", ()))
        except remote.RemoteError as e:
            print(f"book --remote: {e}", file=sys.stderr)
            return 2
    else:
        with get_engine(args.db).connect() as conn:
            out = run(args.cmd, book, conn, getattr(args, "run", ()), getattr(args, "sims", ()),
                      getattr(args, "asof", None))
            conn.rollback()
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    elif args.rich or not TTY.plain():            # Rich by default; --plain / RACINGLINES_PLAIN=1 for the plain text
        from racinglines.cli import rich_view
        rich_view.render(args.cmd, out)
    else:
        _print(args.cmd, out)
    return 0
