"""
Where a command's data comes from: the one resolver every command goes through (racinglines/cli/__init__.py calls
apply() before anything else is imported, so the settings below reach every module).

    --demo [PACK]   a demo store: a separate database (racinglines_demo_<pack>, on the same server as the configured
                    one) and data folder (data/demo/<pack>) filled from a data pack by the real pipeline
                    (racinglines/demo/store.py), built on first use. PACK defaults to kart; a pack is a folder with a
                    pack.toml (racinglines/demo/packs/, or RACINGLINES_DEMO_PACKS). It only switches the data source:
                    the commands themselves run unchanged. Sets DATABASE_URL, RACINGLINES_DATA and
                    RACINGLINES_OFFLINE=1 (FastF1's servers are never called; the store's files are the record).

    --demo [PACK] --remote   (or `source = "remote"` in the CLI defaults file) the pack's hosted store instead: the
                    commands the hosted server runs (`book map|price|settle`, `model backtest|forecast f1|mtb_dh`) are
                    sent to the pack's demo MCP server (pack.toml [remote] mcp_url) with their usual --remote path.

    First run: when no database is configured (no DATABASE_URL, no `db` or `source` in the CLI defaults file) the
    first command that needs one asks once, local or remote, and saves the answer as `source` (choose_first_run).

    RACINGLINES_DEMO_DATABASE_URL   the demo database (default: the configured URL, named racinglines_demo_<pack>)
    RACINGLINES_DEMO_DATA           the demo data folder (default data/demo/<pack>, gitignored)
    RACINGLINES_DEMO_MCP_URL        the hosted demo server (default: the pack's [remote] mcp_url)
"""

import os
import re
import sys
from pathlib import Path

# racinglines.paths.HOME, worked out here: importing paths would fix its DATA before use_demo points it at the demo's
_PKG = Path(__file__).resolve().parents[1]
HOME = _PKG.parent if (_PKG.parent / "pyproject.toml").is_file() else Path.home() / ".racinglines"


def _config():
    from racinglines.cli import config as C
    try:
        return C.load()
    except ValueError:
        return {}


def base_url():
    """The configured database: DATABASE_URL, else the CLI defaults file's `db`, else docker-compose's (asking the
    first-run question first when nothing is configured)."""
    return os.environ.get("DATABASE_URL") or default_url()


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def demo_url(pack):
    from sqlalchemy.engine import make_url
    if os.environ.get("RACINGLINES_DEMO_DATABASE_URL"):
        return os.environ["RACINGLINES_DEMO_DATABASE_URL"]
    return make_url(base_url()).set(database=f"racinglines_demo_{_slug(pack.name)}").render_as_string(hide_password=False)


def demo_data(pack):
    return Path(os.environ.get("RACINGLINES_DEMO_DATA") or HOME / "data" / "demo" / _slug(pack.name))


def demo_active():
    """The active demo pack's name, or None."""
    return os.environ.get("RACINGLINES_DEMO") or None


def pack(ref=None):
    """The demo pack `ref` (default: the active one, else kart). SystemExit naming the known packs if it isn't one."""
    from racinglines.demo import store
    p = store.find(ref or demo_active())
    if p is None:
        raise SystemExit(f"--demo: no pack {ref!r} (known: {', '.join(store.packs())}; RACINGLINES_DEMO_PACKS adds folders)")
    return p


def use_demo(ref=None, build=True, console=None):
    """Point this process at a demo pack's store (building it first when it is missing or outdated). Returns its URL."""
    pk = pack(ref)
    if demo_active() == pk.name:
        url, data = os.environ["DATABASE_URL"], Path(os.environ["RACINGLINES_DATA"])
    else:
        url, data = demo_url(pk), demo_data(pk)
        os.environ.update(DATABASE_URL=url, RACINGLINES_DATA=str(data), RACINGLINES_OFFLINE="1", RACINGLINES_DEMO=pk.name)
        paths = sys.modules.get("racinglines.paths")
        if paths is not None and Path(paths.DATA) != data:      # imported before: only the database follows
            print(f"demo: data folder stays {paths.DATA} in this process (the database is the demo's)", file=sys.stderr)
    if build:
        ensure(pk, url, data, console)
    return url


def ensure(pk, url, data, console=None):
    from racinglines.demo import store
    try:
        why = store.status(pk, url, data)
    except Exception as ex:                                   # noqa: BLE001  no server: say how to start one
        raise SystemExit(f"demo: no Postgres server for {_safe(url)} ({type(ex).__name__}). Start one with "
                         f"`docker compose up -d` (port 5433, docs/quickstart.md), or set DATABASE_URL to your own "
                         f"server: the demo database is created next to it.") from None
    if why == "ready":
        return
    err = _console(console)
    say = (lambda m: err.print(m)) if err else (lambda m: print(m, file=sys.stderr, flush=True))
    say(f"demo: building the {pk.name} pack's store ({why}): database {store._db_name(url)}, folder {data}. "
        f"Synthetic data, made once (a few minutes).")
    store.build(pk, url, data, console=err)
    say(f"demo: the {pk.name} store is ready ({pk.sport} data; `racinglines db stats --demo {pk.name}`)"
        + (f" {pk.manifest['tagline']}" if pk.manifest.get("tagline") else ""))


def _console(console):
    if console is not None:
        return console
    from racinglines.cli import tty
    return tty.console(stderr=True)


def _safe(url):
    from sqlalchemy.engine import make_url
    return make_url(url).render_as_string(hide_password=True)


def take_demo(argv):
    """(argv without `--demo [PACK]`, the pack ref or None, whether --demo was given). PACK is the next word when it
    names a pack (`--demo=PACK` too)."""
    from racinglines.demo import store
    out, ref, given, i = [], None, False, 0
    while i < len(argv):
        a = argv[i]
        if a == "--demo" or a.startswith("--demo="):
            given = True
            if "=" in a:
                ref = a.split("=", 1)[1]
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-") and store.find(argv[i + 1]) is not None:
                ref = argv[i + 1]
                i += 1
        else:
            out.append(a)
        i += 1
    return out, ref, given


# ---------------------------------------------------------------------------
# First run: no database configured (no DATABASE_URL, no `db` or `source` in the CLI defaults file)
# ---------------------------------------------------------------------------

FIRST_RUN = """racinglines needs a database, and none is configured (no DATABASE_URL, no `db` or `source` in {path}).
  1) local:  the Postgres of docker-compose.yml on localhost:5433; start it with one command, `docker compose up -d`
             (then every command works; add --demo for the synthetic demo store, docs/quickstart.md)
  2) remote: sign in to racinglines.bet (the browser opens on first use) and run `racinglines book map|price|settle`
             on the hosted server, like `racinglines remote ...` and `--remote` on a Lab-job command (in testing);
             other commands still need a local database"""


def configured():
    return bool(os.environ.get("DATABASE_URL")) or bool({"db", "source"} & set(_config()))


def default_url():
    """The database a command uses when it names none and DATABASE_URL is unset (db.config.fallback while a CLI command
    runs): the defaults file's `db`; else, once `source` is set, docker-compose's; else the first-run choice."""
    from racinglines.cli import config as C
    from racinglines.db.config import DEFAULT_URL
    cfg = _config()
    if cfg.get("db"):
        return cfg["db"]
    src = cfg.get("source") or choose_first_run()
    if src == "remote":
        raise SystemExit("this command needs a database, and the defaults file's source is remote (racinglines.bet runs "
                         "`racinglines book map|price|settle`, `racinglines remote` and `--remote` Lab jobs). Start a local one with `docker compose up -d` and "
                         f"set `racinglines book config source local`, or set DATABASE_URL ({C.path()}).")
    return DEFAULT_URL


def choose_first_run(ask=None):
    """Ask once which source to use (1 local, 2 remote), save it as `source` in the defaults file and return it. With no
    terminal, print the options and exit 2."""
    from racinglines.cli import config as C
    text = FIRST_RUN.format(path=C.path())
    tty = ask is not None or (sys.stdin.isatty() and sys.stderr.isatty())
    if not tty:
        print(text + "\n\nNo terminal to ask on: choose with `racinglines book config source local` (or remote), or set "
              "DATABASE_URL.", file=sys.stderr)
        raise SystemExit(2)
    ask = ask or (lambda prompt: input(prompt))
    print(text, file=sys.stderr)
    while True:
        a = ask("Choose 1 or 2 [1]: ").strip().lower() or "1"
        if a in ("1", "local"):
            src = "local"
            break
        if a in ("2", "remote"):
            src = "remote"
            break
    C.set_key("source", src)
    print(f"saved source = {src} in {C.path()} (`racinglines book config source local|remote` changes it)", file=sys.stderr)
    if src == "local":
        print("start the database with `docker compose up -d` if it isn't running (port 5433)", file=sys.stderr)
    return src


def source_choice():
    """local or remote: the defaults file's `source` (its `demo` counts as local), else local once a database is
    configured, else the first-run answer."""
    src = _config().get("source")
    if src:
        return "remote" if src == "remote" else "local"
    return "local" if configured() else choose_first_run()


# the commands the hosted server runs (their own --remote path): book map|price|settle|export, model backtest|forecast
REMOTE_COMMANDS = (("book", ("map", "price", "settle", "export"), None), ("model", ("backtest", "forecast"), ("f1", "mtb_dh")))


def remote_capable(argv):
    return any(argv[:1] == [g] and argv[1:2] and argv[1] in cmds and (sports is None or argv[2:3] and argv[2] in sports)
               for g, cmds, sports in REMOTE_COMMANDS)


def demo_mcp_url(pk):
    url = os.environ.get("RACINGLINES_DEMO_MCP_URL") or pk.remote.get("mcp_url")
    if not url:
        raise SystemExit(f"--demo {pk.name} --remote: the pack names no hosted demo server ([remote] mcp_url); "
                         f"run it locally with --local")
    return url


def use_remote_demo(ref, argv):
    """`--demo [PACK]` on the hosted store: the command's own --remote path, aimed at the pack's demo MCP server."""
    pk = pack(ref)
    if not remote_capable(argv):
        names = "; ".join(f"{g} {'|'.join(c)}" + (f" {'|'.join(s)}" if s else "") for g, c, s in REMOTE_COMMANDS)
        raise SystemExit(f"--demo with the remote source: the hosted demo store runs only {names}. This command needs "
                         f"a local store: add --local (book) or set `racinglines book config source local`.")
    os.environ.update(RACINGLINES_DEMO=pk.name, RACINGLINES_MCP_URL=demo_mcp_url(pk))
    return argv if "--remote" in argv else argv + ["--remote"]


def apply(argv):
    """The argv a command should parse, after the data source is set: `--demo [PACK]` is taken out (the commands don't
    know it) and the pack's store selected, local or hosted."""
    if argv[:2] == ["strategy", "search-plan"] or (argv[:1] == ["f1"] and "search-plan" in argv):
        return argv                 # no source to select; --demo must not build a database for a pure plan
    from racinglines.db import config as DBC
    DBC.fallback = default_url                       # the first-run choice, only when a command needs a database
    if "--plain" in argv:                            # the plain text instead of Rich output (racinglines/cli/tty.py)
        argv = [a for a in argv if a != "--plain"]
        os.environ["RACINGLINES_PLAIN"] = "1"
    rest, ref, given = take_demo(argv)
    if not given:
        return argv
    rest = sport_names(pack(ref), rest)
    helping = bool({"-h", "--help"} & set(rest))
    remote = "--remote" in rest or ("--local" not in rest and not helping and source_choice() == "remote")
    if remote:
        return use_remote_demo(ref, rest)
    use_demo(ref, build=not helping)
    return rest


def sport_names(pk, argv):
    """The pack's `sport_name` on the command line (and as a live spec's folder, `kart/demo-2026-09`) read as the
    sport it is stored under, so `model backtest kart --demo` runs the f1 commands on the kart store."""
    alias, sport = pk.sport_name, pk.sport
    if alias == sport:
        return argv
    return [sport if a == alias else sport + a[len(alias):] if a.startswith(alias + "/") else a for a in argv]


def reset():
    """After a command: library callers outside the CLI get DEFAULT_URL again."""
    from racinglines.db import config as DBC
    DBC.fallback = None
