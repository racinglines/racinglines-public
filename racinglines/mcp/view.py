"""
A table view of a tool result, for whoever reads it in a terminal (Claude Code shows the tool's text). Every list of
records in the result (a list of dicts, at any depth) becomes one box-drawn table titled by its key path; the result's
scalar fields (total, truncated, next_offset, note, ...) go on one line above. The JSON the tool returns is unchanged
and still comes back as its own block, so the CLI's --remote path and any client that parses it read the same data.

Colour: off unless RACINGLINES_MCP_COLOR=1, since an ANSI escape is noise to a client that shows text as is. When on,
headers are bold, signed money and edge columns are green above zero and red below, and None is dim. Text is never
coloured to carry meaning the numbers don't already show.
"""

import json
import os

MAX_ROWS = 30                     # rows drawn per table; the JSON block carries every row the tool returned
MAX_COLS = 12                     # columns drawn per table, in the rows' own key order
MAX_CELL = 40                     # characters per cell before it is cut with an ellipsis
WIDTH = 160                       # the drawing width; a narrower terminal wraps the lines
SIGNED = ("ev", "edge", "pnl", "profit", "return", "clv", "change", "delta", "diff")


def color_on():
    return os.environ.get("RACINGLINES_MCP_COLOR", "").strip().lower() in ("1", "true", "yes", "on")


def _records(x):
    return isinstance(x, list) and x and all(isinstance(r, dict) for r in x)


def _tables(obj, path=""):
    """(title, rows) for every list of records in obj, outermost first; a record list's own nested lists are left
    to the JSON (one level of tables reads; a table per row would not)."""
    if _records(obj):
        yield path or "rows", obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tables(v, f"{path}.{k}" if path else str(k))


def _scalars(obj):
    if not isinstance(obj, dict):
        return []
    return [(k, v) for k, v in obj.items() if not isinstance(v, (dict, list))]


def _cell(v):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e6 else f"{v:,.0f}"
    if isinstance(v, (dict, list)):
        v = json.dumps(v, default=str, separators=(",", ":"))
    s = str(v)
    return s if len(s) <= MAX_CELL else s[:MAX_CELL - 1] + "…"


def _signed(col):
    c = col.lower()
    return any(c == s or c.startswith(s + "_") or c.endswith("_" + s) for s in SIGNED)


def render(obj, color=None):
    """The tables for obj as text (ANSI colour only when color, default color_on()), or None when obj holds no list of
    records: then the JSON alone is the result."""
    tables = list(_tables(obj))
    if not tables:
        return None
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    color = color_on() if color is None else color
    con = Console(width=WIDTH, force_terminal=color, no_color=not color, color_system="standard" if color else None,
                  highlight=False, emoji=False, soft_wrap=False)
    with con.capture() as cap:
        head = "  ".join(f"{k} {_cell(v)}" for k, v in _scalars(obj))
        if head:
            con.print(Text(head, style="dim"))
        for title, rows in tables:
            cols = list(dict.fromkeys(k for r in rows for k in r))
            shown = cols[:MAX_COLS]
            t = Table(title=title, title_justify="left", box=box.SQUARE, header_style="bold", show_lines=False)
            for c in shown:
                num = all(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
                          for r in rows if r.get(c) is not None)
                t.add_column(c, justify="right" if num else "left", overflow="fold")
            for r in rows[:MAX_ROWS]:
                cells = []
                for c in shown:
                    v = r.get(c)
                    style = ("dim" if v is None else
                             ("green" if v > 0 else "red" if v < 0 else "")
                             if _signed(c) and isinstance(v, (int, float)) and not isinstance(v, bool) else "")
                    cells.append(Text(_cell(v), style=style))
                t.add_row(*cells)
            more = []
            if len(rows) > MAX_ROWS:
                more.append(f"first {MAX_ROWS} of {len(rows)} rows")
            if len(cols) > MAX_COLS:
                more.append(f"{len(cols) - MAX_COLS} more columns: {', '.join(cols[MAX_COLS:])}")
            if more:
                t.caption = "; ".join(more) + " (all in the JSON)"
                t.caption_justify = "left"
            con.print(t)
    return "\n".join(line.rstrip() for line in cap.get().splitlines())
