"""
Live F1 timing on the Live page, opt-in per viewer: a "Live timing" button on an F1 event opens a websocket, and the
server polls FastF1 for the weekend's latest started session every RACINGLINES_F1_STREAM_SEC (default 120 s:
FastF1 allows about 500 API calls an hour) and sends the classification plus a debug line per step.

On only where RACINGLINES_F1_STREAM=1, which is the default on staging (RACINGLINES_ENV=staging) and off elsewhere.

FastF1's load blocks for seconds and is global state (its cache), so each poll runs in a subprocess:
    python -m racinglines.web.f1_live 2026 16        # prints one snapshot as JSON
FastF1 publishes a session's timing on its static archive during or after the session, so mid-session a poll can
come back empty; the debug log says so.

F1's archive answers 403 to the VM (3 Oct 2026), so on the VM the snapshot comes from the owner's Mac instead:
scripts/deploy/f1_relay.sh runs the same command there and writes the JSON to <data>/runs/f1_relay/<year>-<round>.json
on staging. A poll reads that file while it is fresh (RELAY_MAX_AGE) and runs FastF1 itself otherwise.
"""

import asyncio
import json
import os
import sys
import time

SESSION_ORDER = ("FP1", "FP2", "FP3", "SQ", "S", "Q", "R")
POLL_TIMEOUT = 180
RELAY_MAX_AGE = 600                                # seconds; the relay writes every 2 minutes by default


def enabled():
    return os.environ.get("RACINGLINES_F1_STREAM", "1" if os.environ.get("RACINGLINES_ENV", "").strip() == "staging"
                          else "0") == "1"


def interval():
    return max(30, int(os.environ.get("RACINGLINES_F1_STREAM_SEC", "120")))


def latest_session(event, now):
    """(schedule name, start) of the weekend's latest session that has started by `now` (naive UTC), else None."""
    import pandas as pd
    out = None
    for i in range(1, 6):
        name, t = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
        if name and str(name) != "None" and t is not None and not pd.isna(t) and pd.Timestamp(t) <= now:
            if out is None or pd.Timestamp(t) > out[1]:
                out = (str(name), pd.Timestamp(t))
    return out


def _secs(x):
    import pandas as pd
    if x is None or pd.isna(x):
        return None
    return round(x.total_seconds(), 3) if hasattr(x, "total_seconds") else float(x)


def snapshot(year, rnd, now=None):
    """The weekend's latest started session as plain data: dict(session, start, drivers=[...], laps, note)."""
    import logging

    import fastf1
    import pandas as pd
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    fastf1.Cache.set_disabled()                    # every poll re-reads FastF1, nothing stale from a cache
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    event = fastf1.get_event(year, rnd)
    pick = latest_session(event, now)
    if pick is None:
        return dict(event=str(event["EventName"]), session=None, note="no session of this weekend has started yet",
                    drivers=[], laps=0)
    name, start = pick
    s = fastf1.get_session(year, rnd, name)
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    try:                                           # a load that found nothing raises on first access
        res = s.results if s.results is not None else pd.DataFrame()
        laps = pd.DataFrame(s.laps) if s.laps is not None else pd.DataFrame()
    except Exception as ex:                        # noqa: BLE001
        return dict(event=str(event["EventName"]), session=name, start=start.isoformat(), drivers=[], laps=0,
                    note=f"FastF1 has no timing for {name} yet ({type(ex).__name__})")
    best = laps.groupby("Driver")["LapTime"].min() if len(laps) and "LapTime" in laps else pd.Series(dtype=object)
    last = laps.sort_values("LapNumber").groupby("Driver").tail(1).set_index("Driver") if len(laps) else pd.DataFrame()
    drivers = []
    for _, r in res.iterrows():
        code = r.get("Abbreviation")
        lr = last.loc[code] if code in last.index else None
        drivers.append(dict(pos=None if pd.isna(r.get("Position")) else int(r["Position"]), code=code,
                            name=r.get("FullName"), team=r.get("TeamName"), status=r.get("Status"),
                            best=_secs(best.get(code)), last=_secs(lr["LapTime"]) if lr is not None else None,
                            laps=int(lr["LapNumber"]) if lr is not None and not pd.isna(lr["LapNumber"]) else 0))
    drivers.sort(key=lambda d: (d["pos"] is None, d["pos"] or 0, d["best"] is None, d["best"] or 0))
    note = None if len(laps) else "FastF1 has no laps for this session yet (it publishes during or after the session)"
    return dict(event=str(event["EventName"]), session=name, start=start.isoformat(), drivers=drivers,
                laps=int(len(laps)), note=note)


def relay_file(year, rnd):
    from racinglines import paths
    return paths.DATA / "runs" / "f1_relay" / f"{year}-{rnd:02d}.json"


def relayed(year, rnd, now=None):
    """The Mac relay's snapshot (scripts/deploy/f1_relay.sh) with its age in seconds as relay_age, or None when there
    is none or it is older than RELAY_MAX_AGE."""
    p = relay_file(year, rnd)
    try:
        age = (now if now is not None else time.time()) - p.stat().st_mtime
        data = json.loads(p.read_text()) if age <= RELAY_MAX_AGE else None
    except (OSError, ValueError):
        return None
    if data is not None:
        data["relay_age"] = round(age)
    return data


async def poll(year, rnd):
    """One snapshot, from the Mac relay when it is fresh, else from a FastF1 subprocess: (data or None, error text
    or None, seconds)."""
    t = time.monotonic()
    data = relayed(year, rnd)
    if data is not None:
        return data, None, time.monotonic() - t
    p = await asyncio.create_subprocess_exec(sys.executable, "-m", "racinglines.web.f1_live", str(year), str(rnd),
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(), POLL_TIMEOUT)
    except asyncio.TimeoutError:
        p.kill()
        return None, f"FastF1 didn't answer within {POLL_TIMEOUT} s", time.monotonic() - t
    secs = time.monotonic() - t
    if p.returncode != 0:
        tail = err.decode(errors="replace").strip().splitlines()[-1:] or [f"exit {p.returncode}"]
        return None, tail[0][:300], secs
    return json.loads(out), None, secs


async def serve(ws, user):
    """One viewer's stream: wait for {"type": "start", "event": "2026-16"}, then poll until "stop" or disconnect."""
    from starlette.websockets import WebSocketDisconnect

    async def log(msg, level="info"):
        await ws.send_json(dict(type="log", level=level, msg=msg, ts=time.strftime("%H:%M:%S", time.gmtime())))

    await ws.accept()
    if not enabled():
        await log("live timing is off on this site (RACINGLINES_F1_STREAM)", "error")
        await ws.close(code=1008)
        return
    if user is None:
        await log("not signed in", "error")
        await ws.close(code=1008)
        return
    await log(f"connected as {user['username']}; send start to begin (poll every {interval()} s)")
    try:
        msg = json.loads(await ws.receive_text())
        if msg.get("type") != "start":
            await log(f"expected start, got {msg.get('type')}", "error")
            return
        try:
            year, rnd = (int(x) for x in str(msg.get("event", "")).split("-"))
        except ValueError:
            await log(f"bad event key {msg.get('event')!r} (want YEAR-ROUND)", "error")
            return
        n = 0
        while True:
            n += 1
            await log(f"poll {n}: FastF1 {year} round {rnd}, latest started session")
            data, err, secs = await poll(year, rnd)
            if err:
                await log(f"poll {n} failed after {secs:.1f} s: {err}", "error")
            else:
                await ws.send_json(dict(type="timing", data=data))
                what = data.get("session") or "no session"
                src = (f"from the Mac relay, {data['relay_age']} s old" if data.get("relay_age") is not None
                       else f"{secs:.1f} s")
                await log(f"poll {n}: {what}, {len(data['drivers'])} drivers, {data['laps']} laps ({src})"
                          + (f"; {data['note']}" if data.get("note") else ""))
            await log(f"next poll in {interval()} s")
            try:
                msg = json.loads(await asyncio.wait_for(ws.receive_text(), interval()))
                if msg.get("type") == "stop":
                    await log("stopped")
                    await ws.close()
                    return
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    print(json.dumps(snapshot(int(sys.argv[1]), int(sys.argv[2])), default=str))
