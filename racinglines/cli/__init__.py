"""
racinglines <group> <command> [SPORT] [options]

    data      fetch | ingest | link | download | parse | record | pull | ...   SPORT   (results, sessions, feeds)
    model     forecast | backtest | compare | diagnostic | props | ...   SPORT   (pricing)
    strategy  replay | sweep | season-replay | signals | scorecard | ...  SPORT   (trading against exchange tapes)
    backtest  walk-forward SPORT | coverage | clv   (the backtest core, any sport with a pricing model)
    weather   probe | fetch | show   (weather forecasts per event: Weather Underground, docs/weather.md)
    markets   sync | history | trades | record | archive
    book      map | price | settle | view | render | new   (generic sportsbook lines and slips: model and market EV, docs/sportsbook/slips.md)
    db        init | seed | stats | export
    live      new | step | run | agent | status | report   (a live private-book event, any sport)
    web       serve the web app
    users     setup | list | add | reset-password | remove | deactivate | activate   (accounts, fantasy bucks)
    mcp       the MCP server: the data and simulations for a chat client (docs/mcp.md)
    tui       the web app's screens in the terminal, over MCP: hosted, or --local (docs/tui.md)
    remote    tools | call | job | logout   (the hosted server's tools from the command line, docs/mcp.md)
    check     quick validation: code on synthetic data, data endpoints, database

SPORT is f1, mtb_dh, nascar, motogp or cycling, e.g. `racinglines model forecast f1 --save`.
--demo on any command: the same command on the demo store, a separate database and data folder of synthetic data
(built on first use; racinglines/cli/source.py, docs/quickstart.md), e.g. `racinglines model forecast kart --demo --save`.
`racinglines <group> -h` lists a group's commands and the sports that offer each.
`racinglines --version` prints the installed version.
"""

import re
import sys

GROUPS = ("data", "model", "strategy", "weather", "backtest", "markets", "book", "db", "live", "web", "users", "mcp", "remote",
          "tui", "check")
# the sport-first form (`racinglines f1 forecast`): deprecated, an alias for one version with a notice on stderr, not listed
VERB_GROUPS = ("data", "model", "strategy")
SPORT_FIRST = ("f1", "mtb_dh", "nascar", "motogp", "cycling")


def present(module):
    """Whether racinglines.<module> is installed. The public release leaves some modules out (oss/manifest.toml), and
    the groups and sports it lacks are then neither listed nor offered."""
    import importlib.util
    try:
        return importlib.util.find_spec(f"racinglines.{module}") is not None
    except ModuleNotFoundError:
        return False


def _usage():
    """The docstring above without the groups this install lacks, and with the sports it has."""
    from racinglines.cli import verbs
    gone = [g for g in GROUPS if g not in VERB_GROUPS and not present(f"cli.{g}")]
    lines = [ln for ln in __doc__.splitlines() if not any(ln.startswith(f"    {g} ") for g in gone)]
    sports = list(verbs.MODULES)
    said = sports[0] if len(sports) == 1 else ", ".join(sports[:-1]) + " or " + sports[-1]
    return "\n".join(re.sub(r"^SPORT is .* or \w+, e\.g\.", f"SPORT is {said}, e.g.", ln) for ln in lines)


def _notice(argv):
    """One line on stderr naming the new form of a sport-first command (`racinglines f1 forecast` is now
    `racinglines model forecast f1`). Help and unknown commands print nothing."""
    from racinglines.cli import verbs
    try:
        cmd = next((a for a in argv[1:] if a in verbs.commands(argv[0])), None)
    except Exception:                              # never let the notice break the command
        return
    group = cmd and verbs.group_of(cmd, argv[0])
    if group and not {"-h", "--help"} & set(argv):
        print(f"deprecated: `racinglines {argv[0]} {cmd}` is going away, use `racinglines {group} {cmd} {argv[0]}`",
              file=sys.stderr)


# a module the core install leaves out (pyproject.toml dependencies) -> what `pip install` calls it
PIP_NAMES = {"sklearn": "scikit-learn", "bs4": "beautifulsoup4", "psycopg": "psycopg[binary]", "jinja2": "Jinja2",
             "multipart": "python-multipart", "socketio": "python-socketio", "py_clob_client_v2": "py-clob-client-v2"}


def version():
    from importlib.metadata import PackageNotFoundError, version as _version
    try:
        return _version("racinglines")
    except PackageNotFoundError:                   # a checkout on PYTHONPATH, not installed
        return "unknown"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--version"]:
        print(f"racinglines {version()}")
        return 0
    known = [g for g in GROUPS if g in VERB_GROUPS or present(f"cli.{g}")] + [s for s in SPORT_FIRST if present(f"cli.{s}")]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in known:
        print(_usage())
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    try:
        return _run(argv)
    except ModuleNotFoundError as ex:
        top = (ex.name or "").split(".")[0]
        if not top or top == "racinglines":
            raise
        print(f"racinglines {' '.join(argv[:2])}: needs {top}, which the core install leaves out: "
              f"pip install {PIP_NAMES.get(top, top)}", file=sys.stderr)
        return 2


def _run(argv):
    import importlib
    if argv[0] not in ("web", "mcp", "tui"):       # servers and the full-screen TUI, not jobs: no heartbeat
        from racinglines import progress
        progress.start(" ".join(a for a in argv[:3 if argv[0] in VERB_GROUPS else 2] if not a.startswith("-")))   # stderr, every 5 min (progress.py)
    from racinglines.cli import source
    argv = source.apply(argv)                      # the data source (--demo, first run) before any module reads it
    try:
        if argv[0] in VERB_GROUPS:
            from racinglines.cli import verbs
            return verbs.main(argv[0], argv[1:])
        if argv[0] in SPORT_FIRST:
            _notice(argv)
        mod = importlib.import_module(f"racinglines.cli.{argv[0]}")
        return mod.main(argv[1:])
    finally:
        source.reset()
