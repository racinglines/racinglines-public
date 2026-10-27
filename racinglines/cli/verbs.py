"""
Task-first commands: `racinglines <group> <command> SPORT [options]`, the sport an argument rather than a top-level
command. The groups and which command belongs to which are below; which sports offer a command, its help and its
options come from the sport's own parser (racinglines/cli/<sport>.py), read at run time, so nothing is listed twice.
Every caller in the repo uses this form (tests/test_no_old_cli_forms.py). The old sport-first form is deprecated: an
alias with a notice (cli/__init__.py) for one version, then removed.
"""

import argparse
import importlib
import sys

from racinglines.cli import present

# the CLI module of each sport (sports/<code>.toml codes, plus the short name the commands have always used)
# (those installed: the public release has f1 only, oss/manifest.toml)
MODULES = tuple(m for m in ("f1", "mtb_dh", "nascar", "motogp", "cycling", "fis_alpine") if present(f"cli.{m}"))
SPORTS = {k: m for k, m in {"f1": "f1", "mtb_dh": "mtb_dh", "nascar": "nascar", "motogp": "motogp",
                            "road_cycling": "cycling", "cycling": "cycling",
                            "fis_alpine": "fis_alpine"}.items() if m in MODULES}

GROUPS = {
    "data": ("results, sessions, feeds and market links into the database", (
        "fetch", "download", "parse", "ingest", "schedule", "link", "events", "startlist", "probe", "points", "record",
        "inrace-backfill", "pm-sync", "pm-links-export", "pm-links-import", "pm-history", "pm-trades", "pm-record",
        "pm-archive")),
    "model": ("pricing: forecasts, backtests, model comparisons, one-event diagnostics", (
        "forecast", "backtest", "walk-forward", "compare", "matrix", "diagnostic", "combo", "props", "sprint-check",
        "season", "price", "gc-price", "gc-backtest", "reliability", "search")),
    "strategy": ("trading: replays against exchange tapes, season sweeps, paper signals, strategy searches", (
        "replay", "sweep", "season-replay", "season-strategy", "season-checkpoints", "scorecard", "signals",
        "demo-history", "reconcile", "profiles", "search", "search-plan", "search-report", "search-import", "search-analyze", "live")),
}


# one line for a command several sports offer (each sport's own help says what it does there: `... SPORT -h`)
SHARED = {
    "fetch": "Download the sport's source data (results, sessions or feeds) to data/raw/.",
    "ingest": "Load the downloaded data into the database.",
    "forecast": "Price the upcoming races with the sport's model.",
    "backtest": "Price past races as of before each one and score them against the results.",
    "compare": "A challenger model against the baseline, race by race.",
    "replay": "Replay a strategy against an exchange's recorded prices.",
    "sweep": "The season sweep: the taker modes and maker variants through each race's weekend.",
    "season-replay": "Champion markets replayed through the season.",
    "signals": "Live paper signals of users' strategy profiles (recommendations only).",
    "demo-history": "Backfill the demo accounts' track record from backtest replays.",
}


class _Parser(Exception):
    pass


def parser_of(module):
    """The argparse parser a sport's CLI module builds, without running anything: its main() is stopped at the parse."""
    mod = importlib.import_module(f"racinglines.cli.{module}")
    grab = lambda self, *a, **k: (_ for _ in ()).throw(_Parser(self))      # noqa: E731
    saved = argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args
    argparse.ArgumentParser.parse_args = argparse.ArgumentParser.parse_known_args = grab
    try:
        mod.main(["-h"])
    except _Parser as p:
        return p.args[0]
    finally:
        argparse.ArgumentParser.parse_args, argparse.ArgumentParser.parse_known_args = saved
    raise RuntimeError(f"racinglines.cli.{module} has no argparse parser")


def retitle(parser, module):
    """Name a sport parser's commands by their new form, so `usage:` and error lines read `racinglines model forecast
    f1` (nested commands follow: `racinglines data points mtb_dh import`). Each sport module calls it before parsing."""
    parser.prog = f"racinglines <group> <command> {module}"

    def walk(p, prefix):
        for a in p._actions:
            if isinstance(a, argparse._SubParsersAction):
                for name, sub in a.choices.items():
                    sub.prog = prefix(name)
                    walk(sub, lambda n, base=sub.prog: f"{base} {n}")
    walk(parser, lambda name: f"racinglines {group_of(name, module) or '<group>'} {name} {module}")
    return parser


def _subcommands(parser):
    """{command: help} of a parser's subcommands."""
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            helps = {c.dest: c.help or "" for c in a._choices_actions}
            return {name: helps.get(name, "") for name in a.choices}
    return {}


def _doc_lines(module):
    """{command: description} from the module docstring's command list ("    fetch        Download ...", continued on
    deeper-indented lines)."""
    import re
    doc = importlib.import_module(f"racinglines.cli.{module}").__doc__ or ""
    out, cur = {}, None
    for line in doc.splitlines():
        m = re.match(r"^ {4}([a-z][a-z-]*) {2,}(\S.*)$", line)
        if m:
            cur = m.group(1)
            out[cur] = m.group(2)
        elif cur and re.match(r"^ {6,}\S", line):
            out[cur] += " " + line.strip()
        else:
            cur = None
    return out


def _first_sentence(text):
    import re
    text = " ".join(text.split())
    m = re.search(r"[.;:] (?=[A-Z(])", text)
    return text if m is None else text[:m.start() + 1]


def commands(module):
    """{command: one-line help} of a sport's CLI module (the parser's help, else the docstring's line)."""
    doc = _doc_lines(module)
    return {c: _first_sentence(h or doc.get(c, "")) for c, h in _subcommands(parser_of(module)).items()}


def group_of(command, module):
    """The group a sport's command is listed under (`search` is a model search for MotoGP, a strategy search for F1)."""
    if command == "search":
        return "model" if module == "motogp" else "strategy"
    return next((g for g, (_, cmds) in GROUPS.items() if command in cmds), None)


def _hoist(parser, rest):
    """Move the sport parser's own options (e.g. f1's --db, --variant), which argparse wants before the command, to
    the front, so they can be written anywhere after the sport."""
    own = {s: a for a in parser._actions if not isinstance(a, argparse._SubParsersAction)
           for s in a.option_strings if s not in ("-h", "--help")}
    front, back, i = [], [], 0
    while i < len(rest):
        tok = rest[i]
        name = tok.split("=", 1)[0]
        if name in own:
            a = own[name]
            if "=" in tok or a.nargs == 0:
                n = 0
            elif a.nargs in ("+", "*"):
                n = 0
                while i + 1 + n < len(rest) and not rest[i + 1 + n].startswith("-"):
                    n += 1
            else:
                n = a.nargs if isinstance(a.nargs, int) else 1
            front += rest[i:i + 1 + n]
            i += 1 + n
        else:
            back.append(tok)
            i += 1
    return front, back


def _help(group):
    what, cmds = GROUPS[group]
    rows = {}
    for m in MODULES:
        for c, h in commands(m).items():
            if group_of(c, m) == group:
                r = rows.setdefault(c, [[], h])
                r[0].append(m)
    lines = [f"racinglines {group} <command> SPORT [options]   ({what})", ""]
    for c in cmds:
        if c in rows:
            sports, h = rows[c]
            h = SHARED.get(c, h) if len(sports) > 1 else h
            lines.append(f"    {c:<18} {' '.join(sports):<28} {h}")
    if group == "data":
        lines.append(f"    {'pull':<18} {'any sport':<28} One sport's shared data from the hosted server into the local "
                     "database (racinglines data pull -h).")
    if group == "model":
        lines += ["", "No data yet? Add --demo to any command, e.g. `racinglines model backtest kart --demo`: the same"
                  " command on a synthetic demo store (docs/quickstart.md)."]
    lines += ["", f"`racinglines {group} <command> SPORT -h` for a command's options."]
    return "\n".join(lines)


def main(group, argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(_help(group))
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    if group == "data" and cmd == "pull":            # every sport the same way, from the hosted server (cli/pull.py)
        from racinglines.cli import pull
        return pull.main(rest)
    if not rest or rest[0] not in SPORTS:
        offered = [m for m in MODULES if cmd in commands(m) and group_of(cmd, m) == group]
        elsewhere = sorted({group_of(cmd, m) for m in MODULES if cmd in commands(m)} - {group})
        if not offered and elsewhere:
            print(f"racinglines {group} {cmd}: it is under {' and '.join(elsewhere)}: "
                  f"`racinglines {elsewhere[0]} {cmd} SPORT`", file=sys.stderr)
        elif not offered:
            print(f"racinglines {group}: no command {cmd!r}\n\n{_help(group)}", file=sys.stderr)
        else:
            print(f"racinglines {group} {cmd} SPORT: the sport is one of {', '.join(offered)}", file=sys.stderr)
        return 2
    module, rest = SPORTS[rest[0]], rest[1:]
    if cmd in commands(module) and group_of(cmd, module) != group:
        print(f"racinglines {group} {cmd}: it is `racinglines {group_of(cmd, module)} {cmd} {module}`", file=sys.stderr)
        return 2
    if cmd not in commands(module):
        print(f"racinglines {group} {cmd}: not offered for {module}; `racinglines {group} -h` lists what is",
              file=sys.stderr)
        return 2
    if group in ("model", "strategy") and "--remote" in rest and "-h" not in rest and "--help" not in rest:
        from racinglines.cli import remote          # the command's Lab job on the hosted server (cli/remote.py)
        try:
            return remote.run_model(module, cmd, rest)
        except remote.RemoteError as e:
            print(f"{group} {cmd} {module} --remote: {e}", file=sys.stderr)
            return 2
    front, back = _hoist(parser_of(module), rest)
    return importlib.import_module(f"racinglines.cli.{module}").main(front + [cmd] + back)
