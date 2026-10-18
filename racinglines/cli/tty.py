"""
Rich output for the commands (db stats, model backtest/forecast f1, live run, live report, book): colour, tables, bars
and a few emojis for steps and events. On by default, also when piped or recorded (a script, the web app's /term replay),
with one off switch: `--plain` on any command (racinglines/cli/source.py takes it out) or RACINGLINES_PLAIN=1, which
prints each command's plain text exactly as before. `--json` is always pure JSON.

Colour: NO_COLOR drops every escape code and keeps the layout and emojis; FORCE_COLOR keeps colour. Under systemd
(JOURNAL_STREAM set: the unit's output goes to the journal) the default is no escape codes, because journalctl shows a
line holding control characters as "[N blob data]" (journalctl --all shows it); FORCE_COLOR overrides. Redrawing
progress bars need a terminal: off one (or with TTY_INTERACTIVE=0) the same lines are printed once each. The
machine-read lines stay plain text in every mode: `progress i/n ...`, `Saved ... run N.`, `Per-race rows -> FILE`.
"""

import os
import sys


def plain():
    """True when the plain text is wanted: --plain, RACINGLINES_PLAIN=1, or no rich package."""
    if os.environ.get("RACINGLINES_PLAIN", "") not in ("", "0"):
        return True
    import importlib.util
    return importlib.util.find_spec("rich") is None


def colour():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return not os.environ.get("JOURNAL_STREAM")


def interactive(stream=None):
    """Redrawing (live bars, spinners) only on a terminal, and not with TTY_INTERACTIVE=0."""
    stream = stream or sys.stdout
    return stream.isatty() and os.environ.get("TTY_INTERACTIVE", "") != "0"


def console(stderr=False):
    """The Rich Console the commands print with (stdout, or stderr), or None in plain mode."""
    if plain():
        return None
    from rich.console import Console
    stream = sys.stderr if stderr else sys.stdout
    width = None if stream.isatty() else int(os.environ.get("COLUMNS") or 120)
    if colour():
        return Console(file=stream, force_terminal=True, width=width, highlight=False, emoji=False,
                       force_interactive=interactive(stream))
    return Console(file=stream, force_terminal=False, no_color=True, width=width, highlight=False, emoji=False)


def bar(p, width=18):
    """A probability as a block bar, eighths of a cell: `bar(0.5, 8)` -> '████    '."""
    p = 0.0 if p is None or p != p else max(0.0, min(1.0, float(p)))
    cells = p * width
    full = int(cells)
    part = " ▏▎▍▌▋▊▉"[int((cells - full) * 8)] if full < width else ""
    return ("█" * full + part).ljust(width)


def pct(x, digits=1):
    return "–" if x is None or x != x else f"{100 * x:.{digits}f}%"


def signed(x, digits=2):
    """Rich markup for a signed amount: green above zero, red below."""
    if x is None or x != x:
        return "[dim]–[/]"
    style = "bold green" if x > 0 else ("red" if x < 0 else "")
    return f"[{style}]{x:+,.{digits}f}[/]" if style else f"{x:+,.{digits}f}"


class Progress:
    """The backtest's `progress i/n label` lines as a Rich progress bar on a terminal (one task per pass, other lines
    printed above it); off a terminal, the plain `progress` lines, every `every`-th and the last (the Lab's job log
    reads them). Feeds the 5-minute heartbeat (racinglines/progress.py) either way."""

    def __init__(self, con, every=10):
        self.con, self.every, self.task, self.p = con, every, None, None
        if interactive(con.file):
            from rich.progress import BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn
            from rich.progress import Progress as P
            self.p = P(TextColumn("[bold]{task.description}"), BarColumn(bar_width=32), MofNCompleteColumn(),
                       TextColumn("[cyan]{task.fields[item]}"), TimeElapsedColumn(), console=con)

    def __enter__(self):
        if self.p:
            self.p.start()
        return self

    def __exit__(self, *exc):
        if self.p:
            self.p.stop()

    def new(self, title):
        if self.p:
            self.task = self.p.add_task(title, total=None, item="")
        else:
            self.con.print(f"[bold]{title}[/]")

    def echo(self, m):
        from racinglines import progress as PG
        if not (m.startswith("progress ") and "/" in m.split()[1]):
            (self.p.console if self.p else self.con).print(m, markup=False)
            return
        i, n = m.split()[1].split("/", 1)
        item = m.split(" ", 2)[2] if len(m.split(" ", 2)) > 2 else ""
        PG.update(int(i), int(n), item)
        if self.p:
            if self.task is None:
                self.new("")
            self.p.update(self.task, completed=int(i), total=int(n), item=item)
        elif not int(i) % self.every or i == n:
            print(m, file=self.con.file, flush=True)
