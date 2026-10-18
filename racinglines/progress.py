"""
Progress lines for long-running jobs (owner rule, 2026-09-30): every long job, in any context (cloud, the Mac, the
VM; sweeps, backfills, replays, downloads), shows it is alive at least every 5 minutes, flushed as it happens.

`racinglines` starts one heartbeat per command (cli/__init__.py): a daemon thread that, every
RACINGLINES_PROGRESS_SEC seconds (default 300; 0 turns it off), prints one line to stderr:

    progress f1 sweep: 15 min elapsed · race 12 of 36 (2026-12) · about 30 min left

A command that finishes sooner prints nothing, and the lines go to stderr only, so stdout, reports and every saved
output stay byte-identical. A loop reports where it is with `update(done, total, item)` or by wrapping its iterable
in `track(...)`; without either the line still carries the elapsed time. stdout and stderr are also made
line-buffered, so a log written through a pipe or `tee` gets each line when it is printed, not in 8 KB blocks.
"""

import os
import sys
import threading
import time

_lock = threading.Lock()
_state = dict(label="", t0=time.monotonic(), done=None, total=None, item=None, t_done=None)
_thread = None


def interval():
    try:
        return float(os.environ.get("RACINGLINES_PROGRESS_SEC", "300"))
    except ValueError:
        return 300.0


def update(done=None, total=None, item=None):
    """Where the job is now: `done` of `total` units, and the unit being worked on. Cheap; call it every unit."""
    with _lock:
        if total is not None:
            _state["total"] = total
        if done is not None:
            if _state["done"] is None or done < _state["done"]:
                _state["t_done"] = time.monotonic()        # a new counted phase: time the rest from here
            _state["done"] = done
        if item is not None:
            _state["item"] = str(item)


def track(iterable, unit="item", total=None, name=None):
    """Yield from `iterable`, reporting "<unit> i of n (<name(x)>)" to the heartbeat as it goes."""
    try:
        n = len(iterable) if total is None else total
    except TypeError:
        n = None
    update(done=0, total=n)
    i = 0
    for x in iterable:
        label = name(x) if name else None
        update(done=i, item=f"{unit} {i + 1}" + (f" of {n}" if n else "") + (f" ({label})" if label else ""))
        yield x
        i += 1
    update(done=i)


def line(now=None):
    now = time.monotonic() if now is None else now
    with _lock:
        s = dict(_state)
    out = f"progress {s['label']}: {(now - s['t0']) / 60:.0f} min elapsed"
    if s["item"]:
        out += f" · {s['item']}"
    elif s["done"] is not None and s["total"]:
        out += f" · {s['done']} of {s['total']} done"
    done, total, t = s["done"], s["total"], s["t_done"]
    if done and total and done < total and t is not None and now > t:
        out += f" · about {(now - t) / done * (total - done) / 60:.0f} min left"
    return out


def start(label, every=None, stream=None):
    """Start the heartbeat for this process (once; a second call only relabels it). Returns the thread, or None
    when RACINGLINES_PROGRESS_SEC is 0."""
    global _thread
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    with _lock:
        _state.update(label=label, t0=time.monotonic(), done=None, total=None, item=None, t_done=None)
    every = interval() if every is None else every
    if every <= 0 or _thread is not None:
        return _thread
    stop = threading.Event()

    def beat():
        while not stop.wait(every):
            try:
                print(line(), file=stream or sys.stderr, flush=True)
            except Exception:
                pass                                        # a closed stream never fails the job

    _thread = threading.Thread(target=beat, name="racinglines-progress", daemon=True)
    _thread.stop = stop
    _thread.start()
    return _thread


def stop():
    global _thread
    if _thread is not None:
        _thread.stop.set()
        _thread = None
