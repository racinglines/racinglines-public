"""
Terminal view of a `racinglines book map | price | settle` result with Rich (https://github.com/Textualize/rich):
one table of lines with each slip's legs nested under it, EV and results coloured, and the flags, unmapped lines and
assumptions in panels below. It renders the same dict `--json` prints (racinglines/books/slips.py), so a saved result
renders without a database: `racinglines book view RESULT.json`.

Colours carry no extra meaning beyond the sign: green = EV above zero or a won line, red = below zero or lost, yellow =
a flag, void, pending or manual. Every number is the one in the JSON, formatted.
"""

EV_COLS = ("ev_model", "ev_market")
RESULT_STYLE = {"won": "bold green", "lost": "red", "void": "yellow", "pending": "yellow", "manual": "yellow"}
STATUS_STYLE = {"mapped": "green", "unmapped": "bold red"}


def _rich():
    try:
        from rich import box
        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError as e:                                      # pragma: no cover - rich is in requirements.txt
        raise SystemExit(f"book --rich needs the rich package (pip install -r requirements.txt): {e}")
    return box, Console, Group, Panel, Table, Text


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _ev(x, Text):
    if x is None:
        return Text("–", style="dim")
    return Text(f"{100 * x:+.1f}%", style="bold green" if x > 0 else ("red" if x < 0 else ""))


def _signed(x, Text):
    if x is None:
        return Text("–", style="dim")
    return Text(f"{x:+.2f}", style="bold green" if x > 0 else ("red" if x < 0 else ""))


def _odds(x):
    return "–" if x is None else f"{x:.2f}"


def _who(leg):
    parts = [leg.get("kind") or ""]
    if leg.get("athlete"):
        parts.append(leg["athlete"] + (f" v {leg['opponent']}" if leg.get("opponent") else ""))
    elif leg.get("team"):
        parts.append(leg["team"])
    if leg.get("threshold") is not None:
        parts.append(str(leg["threshold"]))
    if leg.get("side") and leg["side"] != "yes":
        parts.append(leg["side"])
    where = leg.get("event_key") or leg.get("event") or ""
    if leg.get("sport"):
        where = f"{leg['sport']} {where}".strip()
    return " ".join(p for p in parts if p) + (f"  ({where})" if where else "")


def _columns(cmd):
    """(header, justify, line value, leg value) per column; a value is fn(row, Text)."""
    base = [("id", "left", lambda r, T: r["id"], lambda g, T: T(f"└ {g['leg']}", style="dim")),
            ("selection", "left", lambda r, T: r.get("selection") or "", lambda g, T: T(_who(g), style="dim"))]
    if cmd == "map":
        return base + [
            ("kind", "left", lambda r, T: r.get("kind") or "", lambda g, T: ""),
            ("status", "left", lambda r, T: T(r["status"], style=STATUS_STYLE.get(r["status"], "")),
             lambda g, T: T(g["status"], style=STATUS_STYLE.get(g["status"], ""))),
            ("race", "right", lambda r, T: ", ".join(map(str, r.get("races") or [])),
             lambda g, T: "" if g.get("race_id") is None else str(g["race_id"])),
            ("reason", "left", lambda r, T: r.get("reason") or "", lambda g, T: T(g.get("reason") or "", style="dim")),
        ]
    if cmd == "settle":
        return base + [
            ("odds", "right", lambda r, T: _odds(r.get("decimal_odds")), lambda g, T: ""),
            ("result", "left", lambda r, T: T(r.get("result") or "–", style=RESULT_STYLE.get(r.get("result"), "")),
             lambda g, T: T(g.get("result") or "–", style=RESULT_STYLE.get(g.get("result"), "dim"))),
            ("payout", "right", lambda r, T: _odds(r.get("payout")), lambda g, T: ""),
            ("profit", "right", lambda r, T: _signed(r.get("profit"), T), lambda g, T: ""),
        ]
    return base + [
        ("odds", "right", lambda r, T: _odds(r.get("decimal_odds")),
         lambda g, T: T("" if g.get("odds") is None else str(g["odds"]), style="dim")),
        ("book", "right", lambda r, T: _pct(r.get("book_prob")), lambda g, T: T(_pct(g.get("book_prob")), style="dim")),
        ("model", "right", lambda r, T: _pct(r.get("model_prob")),
         lambda g, T: T(_pct(g.get("model_prob")), style="dim")),
        ("market", "right", lambda r, T: _pct(r.get("market_prob")),
         lambda g, T: T(_pct(g.get("market_prob")) + (f" {g['market_exchange']}" if g.get("market_exchange") else ""),
                        style="dim")),
        ("EV model", "right", lambda r, T: _ev(r.get("ev_model"), T), lambda g, T: ""),
        ("EV market", "right", lambda r, T: _ev(r.get("ev_market"), T), lambda g, T: ""),
        ("model method", "left", lambda r, T: T(r.get("model_method") or "", style="dim"), lambda g, T: ""),
    ]


def render(cmd, out, console=None):
    """Print a map / price / settle result dict to `console` (a rich Console; default: stdout)."""
    box, Console, Group, Panel, Table, Text = _rich()
    if console is None:
        from racinglines.cli import tty
        console = tty.console() or Console()      # forced colour off a terminal too (cli/tty.py), NO_COLOR honoured
    b = out["book"]
    head = " · ".join(str(x) for x in (b.get("venue"), f"event {b.get('event')}", b.get("sport"),
                                       f"captured {b.get('captured_utc')}", f"odds {b.get('odds')}",
                                       b.get("currency")) if x)
    cols = _columns(cmd)
    t = Table(title=f"book {cmd}", caption=head, box=box.SIMPLE_HEAVY, header_style="bold", expand=False)
    for name, justify, _, _ in cols:
        t.add_column(name, justify=justify, no_wrap=name in ("id", "odds"))
    for r in out["lines"]:
        t.add_row(*[f(r, Text) for _, _, f, _ in cols], style="dim" if r.get("status") == "unmapped" else None)
        if len(r.get("legs") or []) > 1 or (cmd == "map" and r.get("status") == "unmapped"):
            for g in r["legs"]:
                t.add_row(*[f(g, Text) for _, _, _, f in cols])
    console.print(t)

    flags = [Text.assemble((f"{r['id']}: ", "bold"), f) for r in out["lines"] for f in (r.get("flags") or [])]
    if flags:
        console.print(Panel(Group(*flags), title="flags", border_style="yellow", expand=False))
    if out.get("unmapped"):
        console.print(Panel(Text(", ".join(map(str, out["unmapped"])) + "\nexact keys only: fix the book file or its "
                                 "[aliases]"), title="unmapped lines", border_style="red", expand=False))
    if cmd == "price" and out.get("assumptions"):
        console.print(Panel(Group(*[Text(f"• {a}", style="dim") for a in out["assumptions"]]
                                  + [Text(f"blend: {out.get('blend')}", style="dim")]),
                            title="assumptions", border_style="dim", expand=False))
