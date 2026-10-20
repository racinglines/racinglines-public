"""
racinglines live: launch and run a live private-book event, for any sport (pipelines/live.py).

    racinglines live new f1 2026-16                        # write live/f1/2026-16.toml from the schedule
    racinglines live new mtb_dh 20260925_mtb --final 3 --quali 2,91
    racinglines live step live/f1/2026-16.toml             # one idempotent update (F1: acts only when an update is due)
    racinglines live run live/f1/2026-16.toml              # step at the sport's cadence until the event is settled
    racinglines live run live/mtb_dh/20260925_mtb.toml     # downhill: the poll loop, locked (one loop per final)
    racinglines live run live/f1/2026-15.toml --simulate --no-fetch     # a rehearsal on a simulated clock
    racinglines live run kart/demo-2026-09 --demo --simulate --no-fetch   # the kart demo pack's rehearsal (docs/quickstart.md)
    racinglines live agent live/f1/2026-16.toml --install  # a macOS LaunchAgent: one step every 5 minutes, locked
    racinglines live auto                                  # F1: step every event whose window is open (a 5-min timer)
    racinglines live status                                # every event: live / replay / settled, last update, lateness
    racinglines live report live/f1/2026-16.toml           # the event report (Markdown + charts + PDF)
    racinglines live settle live/mtb_dh/20260925_mtb.toml  # record a settled event in the database (live_events)
    racinglines live reprice live/mtb_dh/20260925_mtb.toml --every 10   # re-price from the logged raw feed

A spec is a path or sport/event (live/<sport>/<event>.toml). A demo experiment: play money, nothing is
traded anywhere.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

from racinglines import paths

LABEL = "bet.racinglines.live.{run}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="racinglines live", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="Write a launch spec from the schedule, with the sport's defaults.")
    p.add_argument("sport")
    p.add_argument("event", help="F1: an event key (2026-16); downhill: the ChronoRace slug (20260925_mtb)")
    p.add_argument("--final", help="Downhill: the final's live-timing key (e.g. 3 = men's final)")
    p.add_argument("--quali", default="2,91", help="Downhill: qualifying sessions' keys, comma-separated")
    p.add_argument("--conditions", default="", help='Downhill: e.g. "clear, rutted"')
    p.add_argument("--title", default=None)
    p.add_argument("--force", action="store_true", help="Overwrite an existing spec")
    for name, hlp in (("step", "One idempotent update."), ("run", "Step at the sport's cadence until the event is settled (downhill: one locked poll loop per final).")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("spec")
        p.add_argument("--now", default=None, help="F1: the (simulated) UTC time of this step, e.g. 2026-09-24T07:35")
        p.add_argument("--no-fetch", action="store_true", help="F1: don't fetch FastF1 data (offline, a rehearsal)")
        p.add_argument("--unfreeze", action="store_true", help="Run with changed settings (they are frozen at the opening)")
        p.add_argument("--no-sync", action="store_true", help="Don't write positions to the database")
        p.add_argument("--no-alert", action="store_true", help="Don't send the lateness alert")
        if name == "run":
            p.add_argument("--simulate", action="store_true",
                           help="F1: walk a simulated clock from before the first update to the results, in --tick steps")
            p.add_argument("--from", dest="start", default=None, help="Simulated clock start (UTC)")
            p.add_argument("--tick", type=int, default=None, help="Minutes between simulated steps (default: the sport's step_min)")
            p.add_argument("--minutes", type=int, default=0, help="Stop after this many (real) minutes")
    p = sub.add_parser("auto", help="F1: one step of every committed spec whose window is open (from its opening "
                                    "to a day after its results), so no event has to be started by hand.")
    p.add_argument("--sport", default="f1", help="Only F1 steps on a timer (downhill runs its own poll loop)")
    p.add_argument("--now", default=None, help="The (simulated) UTC time, e.g. 2026-10-10T09:30")
    p.add_argument("--list", action="store_true", help="Only print the open events' keys, one per line (f1_push_auto.sh)")
    for flag in ("--no-fetch", "--unfreeze", "--no-sync", "--no-alert"):
        p.add_argument(flag, action="store_true")
    p = sub.add_parser("agent", help="Write (and load) a macOS LaunchAgent for a spec: a step every 5 min, locked, logged.")
    p.add_argument("spec")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--install", action="store_true", help="Copy it to ~/Library/LaunchAgents and load it")
    g.add_argument("--remove", action="store_true", help="Unload and delete it")
    sub.add_parser("status", help="Every event: live / replay / settled, last update, next update, lateness.")
    p = sub.add_parser("report", help="The event report: Markdown, charts and a PDF, in the run folder.")
    p.add_argument("spec")
    p.add_argument("--pdf", action="store_true", help="Also render the PDF (needs weasyprint or a browser)")
    p = sub.add_parser("reprice", help="Re-price an event from its logged raw feed with this code (downhill); "
                                       "scores it against what was quoted live.")
    p.add_argument("spec")
    p.add_argument("--label", default="current", help="Name of the re-pricing (reprice/<label>.jsonl in the run folder)")
    p.add_argument("--every", type=int, default=1, help="Every n-th logged feed response only")
    p.add_argument("--sims", type=int, default=None, help="Simulations per update (default: the adapter's)")
    p = sub.add_parser("settle", help="Record a settled event in the database (live_events: dates, the book's P&L).")
    p.add_argument("spec")
    args = ap.parse_args(argv)
    return globals()[f"cmd_{args.cmd}"](args)


def _spec(ref):
    from racinglines.pipelines import live as LV
    return LV.load_spec(ref)


@contextmanager
def lock(run):
    """One step at a time per event: a non-blocking lock on the run folder (the LaunchAgent's runs can't overlap)."""
    from racinglines.pipelines import live as LV
    f = (LV.folder(run) / ".lock").open("w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def _echo(m):
    print(m, flush=True)


KIND_ICON = {"open": "🚦", "stage": "🔄", "close": "🔒", "results": "🏁"}


def _snap_panel(snap, top=8):
    """One update of a live F1 book as a Rich panel: the win markets (our fair, its move, our quotes, the exchanges'
    mids, the maker's inventory), the crowd and the maker's P&L. Printed once per update (static lines, so it reads the
    same in a terminal, a log or a recording)."""
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table

    from racinglines.cli import tty
    u = snap["update"]
    done = sum(1 for x in snap["updates"] if x["state"] == "done")
    wins = sorted((m for m in snap["markets"] if m["kind"] == "race_win" and m.get("fair") is not None),
                  key=lambda m: -m["fair"])[:top]
    venues = []
    for m in wins:
        for x in m.get("exchange_prices") or []:
            if x["name"] not in venues:
                venues.append(x["name"])
    t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
    for c, j in (("driver", "left"), ("fair", "right"), ("", "left"), ("move", "right"), ("bid", "right"),
                 ("ask", "right"), *((v, "right") for v in venues), ("inventory", "right")):
        t.add_column(c, justify=j)
    for m in wins:
        prev = m.get("prev_fair")
        move = "" if prev is None else (f"[green]▲ {100 * (m['fair'] - prev):.1f}[/]" if m["fair"] > prev + 0.0005 else
                                        f"[red]▼ {100 * (prev - m['fair']):.1f}[/]" if m["fair"] < prev - 0.0005 else "[dim]=[/]")
        who = m["subject"] + (" ✅" if m.get("outcome") is True else " ❌" if m.get("outcome") is False else "")
        ex = {x["name"]: x.get("mid") for x in m.get("exchange_prices") or []}
        inv = m.get("inv")
        t.add_row(f"[bold]{who}[/]" if m.get("outcome") else who, tty.pct(m["fair"]),
                  f"[green]{tty.bar(m['fair'], 12)}[/]", move, tty.pct(m.get("bid")), tty.pct(m.get("ask")),
                  *(f"[magenta]{tty.pct(ex.get(v))}[/]" for v in venues),
                  "[dim]–[/]" if not inv else f"{inv:+,.0f}")
    c, p = snap.get("crowd") or {}, snap.get("maker_pnl") or {}
    lines = [t,
             f"👥 crowd: [bold]{c.get('last_fills', 0)}[/] fills this update · {c.get('fills', 0):,} in all · "
             f"${c.get('volume', 0):,.0f} traded · {c.get('active', 0)} of {c.get('takers', 0)} takers still betting",
             f"💰 maker P&L {tty.signed(p.get('total'))}  [dim](crowd {p.get('crowd', 0):+,.2f}, "
             f"picks {p.get('taker', 0):+,.2f})[/]"]
    cls = (snap.get("classification") or {}).get("rows") or []
    if cls:
        rnd = (snap.get("classification") or {}).get("round") or ""
        lines.append(f"🏁 {rnd}: " + " · ".join(f"[bold]P{r['pos']}[/] {r['driver']}" for r in cls[:3]))
    for n in snap.get("note") or []:
        lines.append(f"[yellow]{n}[/]")
    if snap.get("next_label"):
        lines.append(f"[dim]next: {snap['next_label']} at {str(snap['next_at'])[:16]} UTC[/]")
    title = (f"{KIND_ICON.get(u['kind'], '•')} [bold]{snap.get('title') or snap['event_key']}[/] · {u['label']} "
             f"[dim]({done} of {len(snap['updates'])} · {str(snap['ts'])[:16]} UTC)[/]")
    style = "green" if snap.get("done") else ("yellow" if snap.get("closed") else "blue")
    return Panel(Group(*lines), title=title, title_align="left", border_style=style, box=box.ROUNDED, expand=False)


def _summary_panel(run):
    """A settled run's record (pipelines/live.py summary, read-only) as a small Rich panel."""
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    from racinglines.cli import tty
    from racinglines.pipelines import live as LV
    s = LV.summary(run)
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("opened", str(s["opened_at"])[:16] + " UTC")
    t.add_row("settled", str(s["settled_at"])[:16] + " UTC" if s["settled_at"] else "[yellow]not yet[/]")
    t.add_row("💰 maker P&L", tty.signed(s["maker_pnl"]))
    t.add_row("   from the crowd", tty.signed(s["crowd_pnl"]))
    t.add_row("   from the picks", tty.signed(s["taker_pnl"]))
    t.add_row("👥 fills", f"{s['fills'] or 0:,}  [dim](${s['volume'] or 0:,.0f} traded)[/]")
    cr = (s.get("detail") or {}).get("crowd_results") or {}
    if cr:
        t.add_row("   bettors", f"{cr.get('bettors', 0)}: [green]{cr.get('up', 0)} up[/], [red]{cr.get('down', 0)} down[/],"
                                f" best {tty.signed(cr.get('best'))}, worst {tty.signed(cr.get('worst'))}")
    return Panel.fit(t, title=f"🏁 [bold]{s['title']}[/]", title_align="left", border_style="green", box=box.ROUNDED)


def _step(spec, args, now=None, cache=None):
    from racinglines.pipelines import live as LV
    ad = LV.adapter(spec["sport"])
    kw = dict(fetch=not args.no_fetch, unfreeze=args.unfreeze, sync=not args.no_sync, alert=not args.no_alert,
              cache=cache) if spec["sport"] == "f1" else {}
    with lock(spec["run"]) as ok:
        if not ok:
            _echo(f"{spec['run']}: another step is running; skipped")
            return None
        return ad.step(spec, now=now or args.now, echo=_echo, **kw)


SETTLED_HINT = ("settled: nothing left to step (a per-event timer for it can go off: "
                "bash scripts/deploy/vm.sh live {event} off; live auto drops it a day after its results)")


def cmd_step(args, auto=False):
    spec = _spec(args.spec)
    s = _step(spec, args)
    if s is None and spec["sport"] == "f1":
        from racinglines.pipelines import live_f1 as F
        st = F.status(spec, args.now)
        if st["next"]:
            _echo(f"{spec['event']}: nothing new · next: {st['next']} (due {st['due'][:16]} UTC"
                  + (f", {st['late_h']} h late" if st["late_h"] else "") + ")")
        else:                                    # a step on a settled event does nothing: say what turns it off
            _echo(f"{spec['event']}: nothing new · " + ("settled" if auto else SETTLED_HINT.format(event=spec["event"])))
    return 0


AUTO_LEAD_H, AUTO_TAIL_H = 1, 24      # live auto: from an hour before the window opens to a day after its results


def due_specs(sport, now):
    """The committed specs (live/<sport>/*.toml) whose window is open at `now` (naive UTC): from AUTO_LEAD_H before
    the opening to AUTO_TAIL_H after the results, so each session's update, the close and the settlement all run."""
    import pandas as pd

    from racinglines.pipelines import live as LV
    out = []
    for p in sorted((LV.SPECS / sport).glob("*.toml")) if (LV.SPECS / sport).exists() else []:
        try:
            spec = LV.load_spec(p)
            w = spec.get("window") or {}
            if not w.get("open"):                        # a rehearsal (no window): only run by hand, --simulate
                continue
            a, b = pd.Timestamp(w["open"]), pd.Timestamp(w.get("results") or w["close"])
        except Exception as ex:                          # noqa: BLE001
            print(f"{p.name}: unreadable spec, skipped ({type(ex).__name__}: {ex})", file=sys.stderr, flush=True)
            continue
        if a - pd.Timedelta(hours=AUTO_LEAD_H) <= now <= b + pd.Timedelta(hours=AUTO_TAIL_H):
            out.append(spec)
    return out


def cmd_auto(args):
    import pandas as pd
    if args.sport != "f1":
        sys.exit("live auto: only F1 steps on a timer (downhill: `racinglines live run` per final)")
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC").tz_localize(None)
    specs = due_specs(args.sport, now)
    if args.list:
        for spec in specs:
            print(spec["event"])
        return 0
    if not specs:
        _echo(f"live auto: no {args.sport} event window open at {now:%Y-%m-%d %H:%M} UTC")
        return 0
    rc = 0
    for spec in specs:                           # one event's failure doesn't stop the others' steps
        args.spec = spec["path"]
        try:
            cmd_step(args, auto=True)
        except Exception as ex:                  # noqa: BLE001
            rc = 1
            _echo(f"{spec['event']}: step FAILED: {type(ex).__name__}: {ex}")
    return rc


def cmd_run(args):
    import pandas as pd
    spec = _spec(args.spec)
    if spec["sport"] != "f1":
        from racinglines.pipelines import live as LV
        LV.adapter(spec["sport"]).run_spec(spec, minutes=args.minutes, echo=_echo)     # downhill: the loop takes the run folder's lock
        return 0
    from racinglines.pipelines import live_f1 as F
    tick = pd.Timedelta(minutes=args.tick or spec["live"]["poll"].get("step_min", 5))
    cache = {}
    from racinglines.cli import tty
    con = tty.console()
    if args.simulate:
        ups, _ = F.plan(spec["event"])
        t = pd.Timestamp(args.start) if args.start else ups[0]["at"] - tick
        end = ups[-1]["at"] + pd.Timedelta(hours=6)
        n = 0
        while t <= end:
            snap = _step(spec, args, now=t, cache=cache)
            if snap is not None:
                n += 1
                if con is not None and isinstance(snap, dict) and snap.get("update"):
                    con.print(_snap_panel(snap))
            if F.status(spec, t)["settled"]:
                break
            t += tick
        _echo(f"simulated {spec['event']} to {t}: {n} updates")
        if con is not None and F.status(spec, t)["settled"]:
            if not n:
                con.print(f"[yellow]{spec['run']} was already settled: nothing new to simulate[/] [dim](its run folder "
                          f"{paths.rel(F.LV.folder(spec['run']))})[/]")
            con.print(_summary_panel(spec["run"]))
        return 0
    t0 = time.time()
    while True:
        try:
            _step(spec, args, cache=cache)
        except Exception as ex:                          # noqa: BLE001  keep going through data hiccups
            _echo(f"error: {ex}")
        if F.status(spec)["settled"] or (args.minutes and time.time() - t0 > args.minutes * 60):
            return 0
        time.sleep(tick.total_seconds())


# ---------------------------------------------------------------------------
# new: a launch spec from the schedule
# ---------------------------------------------------------------------------

F1_TEMPLATE = '''# Launch spec: {title} (racinglines live, docs/live-events.md). Written by `racinglines live new f1 {event}`;
# edit freely before the book opens (the settings freeze into meta.json at the first update).
sport = "f1"
event = "{event}"
title = "{title}"

# The window (UTC, from the FastF1 schedule): the book opens {open_at}, closes at lights out {close_at},
# and settles from the results ({results_at}).
[window]
open = "{open_at}"
close = "{close_at}"
results = "{results_at}"

# Overrides of sports/f1.toml [live], e.g. [live.quoting] max_pos = 1000.0
[live.markets]
h2h_from = "last_listed"        # head-to-head pairs: the last race Polymarket listed ({h2h_from})

# The demo taker's hype picks, placed at the opening at the maker's ask ($25 each): names and stories, not value.
# name: a driver's (or team's) name; market: race_win, race_podium, race_pole, race_h2h (with vs), race_constructor_top
{picks}'''

DEFAULT_PICKS = [
    ("Hamilton", "race_win", 5, "seven-time champion in red", None),
    ("Verstappen", "race_win", 5, "the name everyone knows", None),
    ("Leclerc", "race_pole", 4, "Saturday specialist", None),
    ("Norris", "race_podium", 4, "crowd favourite", None),
    ("Alonso", "race_podium", 5, "one more podium for the legend", None),
    ("Antonelli", "race_win", 4, "the young star", None),
    ("Hamilton", "race_h2h", 4, "the old master beats the new wave", "Piastri"),
    ("Ferrari", "race_constructor_top", 5, "the tifosi never doubt", None),
    ("Russell", "race_pole", 3, "one-lap pace", None),
    ("Bearman", "race_podium", 3, "rising star", None),
    ("Colapinto", "race_podium", 3, "the fans' pick", None),
    ("Piastri", "race_podium", 3, "quietly consistent", None),
]


def _picks_toml(picks):
    out = []
    for name, market, hype, why, vs in picks:
        out.append(f'[[picks]]\nname = "{name}"\nmarket = "{market}"\nhype = {hype}\nwhy = "{why}"'
                   + (f'\nvs = "{vs}"' if vs else ""))
    return "\n".join(out) + "\n"


def cmd_new(args):
    from racinglines.pipelines import live as LV
    path = LV.spec_path(args.sport, args.event)
    if path.exists() and not args.force:
        print(f"{paths.rel(path)} exists (--force to overwrite)")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.sport == "f1":
        from racinglines.db.config import get_engine
        from racinglines.pipelines import live_f1 as F
        ups, w = F.plan(args.event)
        try:
            with get_engine().connect() as c:
                _, src = F.h2h_pairs(c, args.event)
        except Exception:                                # noqa: BLE001  (no database: resolved at the opening)
            src = "resolved at the opening"
        body = F1_TEMPLATE.format(event=args.event, title=args.title or f"{w['name']} (private book)",
                                  open_at=ups[0]["at"].isoformat(), close_at=ups[-2]["at"].isoformat(),
                                  results_at=ups[-1]["at"].isoformat(), h2h_from=src, picks=_picks_toml(DEFAULT_PICKS))
    else:
        if not args.final:
            print("downhill needs --final (the final's live-timing key)")
            return 2
        quali = ", ".join(f'"{k}"' for k in args.quali.split(",") if k)
        body = (f'# Launch spec: a downhill final (racinglines live, docs/live-events.md).\n'
                f'sport = "{args.sport}"\nevent = "{args.event}"\nrun = "{args.event}_{args.final}"\n'
                f'title = "{args.title or args.event}"\n\n[feed]\nslug = "{args.event}"\nfinal = "{args.final}"\n'
                f'quali = [{quali}]\nconditions = "{args.conditions}"\n')
    path.write_text(body)
    LV.load_spec(path)                                   # it parses
    print(f"wrote {paths.rel(path)}")
    return 0


# ---------------------------------------------------------------------------
# agent: a macOS LaunchAgent per spec
# ---------------------------------------------------------------------------

def plist(spec, root=None, exe=None):
    """The LaunchAgent for a spec: F1 steps every step_min minutes (StartInterval; the lock stops overlaps);
    downhill runs the poll loop, restarted if it exits early (KeepAlive until the final is over).
    From a checkout (or with `root`) it runs the checkout's .venv in the checkout; from an installed package, the installed
    `racinglines` (else this Python's `-m racinglines`) in the data home, with the spec's absolute path."""
    run = spec["run"]
    f1 = spec["sport"] == "f1"
    if paths.CHECKOUT or root is not None:
        root = str(root or paths.ROOT)
        cmd = [exe or os.path.join(root, ".venv", "bin", "racinglines")]
        log = os.path.join(root, "data", "runs", "live", run, "agent.log")
        rel = os.path.relpath(spec["path"], paths.ROOT) if os.path.isabs(spec["path"]) else spec["path"]   # as in the repo
    else:
        import shutil
        root = str(paths.HOME)
        cmd = [exe] if exe else [shutil.which("racinglines")] if shutil.which("racinglines") else [sys.executable, "-m", "racinglines"]
        log = str(paths.DATA / "runs" / "live" / run / "agent.log")
        rel = os.path.abspath(spec["path"])
    args = [*cmd, "live", "step" if f1 else "run", rel]
    sched = (f"  <key>StartInterval</key><integer>{int(spec['live']['poll'].get('step_min', 5)) * 60}</integer>\n" if f1 else
             "  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>\n")
    argx = "".join(f"<string>{a}</string>" for a in args)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- {spec.get("title") or run}: racinglines live {"step" if f1 else "run"} {rel} (pipelines/live.py). Written by
     `racinglines live agent {rel}`. A lock in the run folder stops overlapping steps. Paper only. -->
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL.format(run=run)}</string>
  <key>ProgramArguments</key>
  <array>{argx}</array>
  <key>WorkingDirectory</key><string>{root}</string>
{sched}  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
'''


def cmd_agent(args):
    from racinglines.pipelines import live as LV
    spec = _spec(args.spec)
    label = LABEL.format(run=spec["run"])
    local = LV.folder(spec["run"]) / f"{label}.plist"
    target = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
    uid = os.getuid()
    if args.remove:
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/{label}"], check=False)
        if os.path.exists(target):
            os.remove(target)
        print(f"removed {label}")
        return 0
    if not paths.CHECKOUT:
        paths.HOME.mkdir(parents=True, exist_ok=True)       # the agent's working folder must exist
    local.write_text(plist(spec))
    print(f"wrote {paths.rel(local)}")
    if args.install:
        if sys.platform != "darwin":
            print("not macOS: copy the plist to ~/Library/LaunchAgents on the Mac and bootstrap it there")
            return 1
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w") as f:
            f.write(local.read_text())
        subprocess.run(["launchctl", "bootout", f"gui/{uid}/{label}"], check=False, capture_output=True)
        subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", target], check=True)
        print(f"loaded {label}: launchctl print gui/{uid}/{label} | grep -E 'state|pid'")
    else:
        print(f"install: racinglines live agent {args.spec} --install")
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args):
    from datetime import datetime, timezone

    from racinglines.pipelines import live as LV
    evs = LV.events()
    if not evs:
        print("no live events recorded")
        return 0
    specs = {}
    for p in sorted(LV.SPECS.glob("*/*.toml")) if LV.SPECS.exists() else []:
        try:
            s = LV.load_spec(p)
            specs[s["run"]] = s
        except Exception:                                # noqa: BLE001
            pass
    for e in evs:
        snap, _, _ = LV.load(e["run"])
        st = LV.state(e["run"])
        age = (datetime.now(timezone.utc).timestamp() - e["mtime"]) / 3600
        line = f"{e['run']:24s} {e['sport'] or '?':7s} {('settled' if snap and snap.get('done') else st):8s} " \
               f"last {snap['ts'][:16] if snap else '-'} ({age:.1f} h ago)"
        if e["sport"] == "f1" and e["run"] in specs:
            from racinglines.pipelines import live_f1 as F
            fs = F.status(specs[e["run"]])
            if fs["next"]:
                line += f" · next {fs['next']} {fs['due'][:16]}" + (f" · LATE {fs['late_h']} h" if fs["late_h"] else "")
        print(line + f"  {e['title']}")
    return 0


def cmd_report(args):
    from racinglines.pipelines import live_report as R
    spec = _spec(args.spec)
    out = R.write(spec, pdf=args.pdf)
    files = {k: paths.rel(v) if hasattr(v, "parts") else v for k, v in out.items()}
    from racinglines.cli import tty
    con = tty.console()
    if con is None:
        print(json.dumps(files, indent=1))
        return 0
    from rich import box
    from rich.table import Table
    if spec["sport"] == "f1":
        try:
            con.print(_summary_panel(spec["run"]))
        except (ValueError, OSError, KeyError) as ex:        # a run with no snapshot yet: the files are still listed
            con.print(f"[yellow]no summary: {ex}[/]")
    t = Table(box=box.SIMPLE_HEAD, header_style="bold", title="📄 [bold]report files[/]", title_justify="left")
    t.add_column("what")
    t.add_column("file", style="cyan")
    for k, v in files.items():
        t.add_row(k, str(v))
    con.print(t)
    return 0


def cmd_settle(args):
    from racinglines.pipelines import live as LV
    spec = _spec(args.spec)
    s = LV.settle(spec["run"])
    if not s["settled_at"]:
        print(f"{spec['run']}: recorded, but not settled yet (run it again after the results)")
    print(f"{s['run']}: {s['title']} · opened {str(s['opened_at'])[:16]} · maker P&L {s['maker_pnl'] or 0:+,.2f} "
          f"(crowd {s['crowd_pnl'] or 0:+,.2f}, demo taker {s['taker_pnl'] or 0:+,.2f}) · {s['fills'] or 0:,} fills")
    return 0


def cmd_reprice(args):
    from racinglines.pipelines import live as LV
    spec = _spec(args.spec)
    ad = LV.adapter(spec["sport"])
    if not hasattr(ad, "reprice"):
        print(f"{spec['sport']}: no raw feed to re-price (F1 re-prices stages with racinglines model diagnostic f1)")
        return 1
    f = spec["feed"]
    kw = dict(n_sims=args.sims) if args.sims else {}
    res = ad.reprice(f["slug"], str(f["final"]), label=args.label, every=args.every, echo=_echo, **kw)
    print(f"{len(res['rows'])} updates re-priced -> {paths.rel(LV.folder(spec['run']) / 'reprice' / (args.label + '.jsonl'))}")
    for who in ("live", "repriced"):
        for kind, v in sorted(res["scorecard"][who].items()):
            print(f"  {who:9s} {kind:7s} mean over {v['updates']} updates: Brier {v['brier']:.4f}, log loss {v['logloss']:.4f}")
    return 0
