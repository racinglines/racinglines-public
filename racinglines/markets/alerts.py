"""
New-market alerts. Every Polymarket sync (the recorder re-syncs every --sync-every minutes) stamps the
tokens it sees for the first time with market_links.first_seen_at; those are grouped by event and
announced, upcoming F1 races first.

- **macOS notification** (osascript, local only), with a sound when a new market is for an upcoming race.
- **Phone push** via ntfy.sh, only when RACINGLINES_NTFY_TOPIC is set (off by default: it sends the
  market titles to ntfy's public server, so pick an unguessable topic name).
- **Webhook**: a JSON POST to ALERT_WEBHOOK_URL when set ({"text", "content", "title", ...}: Slack and
  Discord incoming webhooks read text / content).
- **Log**: one JSON line per event in data/runs/alerts/new_markets.jsonl.

Strategy-profile signals (pipelines/signals.py) go out through the same channels: signals_alert.
- **Web**: markets first seen within NEW_HOURS carry a "new" badge where they're listed (the Markets
  page's race cards, the Polymarket page).

    racinglines markets record                     # alerts on by default (--no-alerts to turn off)
    racinglines data pm-sync f1 --alert                 # a one-off sync that alerts too
"""

import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import text

from racinglines import paths

LOG = paths.DATA / "runs" / "alerts" / "new_markets.jsonl"
NEW_HOURS = 48              # "new" badge lifetime in the web app
UPCOMING_DAYS = 3           # a race that started up to this many days ago still counts as upcoming (in progress)
NTFY = "https://ntfy.sh"


def race_for(resolver, link):
    """(race_id, key) of the race a link is about: its own race_id, else a "<Name> Grand Prix" in its
    event title or question (safety-car, rain, red-flag props), tried on the last 1-3 words of the name."""
    if link.get("race_id"):
        return link["race_id"], None
    for s in (link.get("event_title") or "", link.get("question") or ""):
        mm = re.search(r"([\w .'-]+?)\s+Grand Prix", s)
        if not mm:
            continue
        words = mm.group(1).split()
        for k in (1, 2, 3):
            if len(words) >= k:
                rid, key = resolver.race(" ".join(words[-k:]), link.get("end_date"))
                if rid:
                    return rid, key
    return None, None


def _race_dates(conn, ids):
    if not ids:
        return {}
    rows = conn.execute(text("""SELECT r.id, r.format->>'event_name', e.name, e.start_date
                                FROM races r JOIN events e ON e.id = r.event_id WHERE r.id = ANY(:ids)"""),
                        dict(ids=list(ids))).all()
    return {i: (_race_name(short, official), d) for i, short, official, d in rows}


def _race_name(short, official):
    """The short name ("Singapore Grand Prix") when it is one, else the official event name."""
    return short if short and "grand prix" in short.lower() else (official or short or "")


def summarize(conn, new, resolver, today=None):
    """Group new links (dicts from sync(..., new=[])) by event. Each group: event_slug, event_title,
    new_event (every token of the event is new), kinds, n, race, race_date, upcoming. Open markets only;
    upcoming races first, soonest first."""
    today = today or date.today()
    new = [n for n in new if not n.get("closed")]
    if not new:
        return []
    for n in new:
        n["race_id"] = race_for(resolver, n)[0]
    races = _race_dates(conn, {n["race_id"] for n in new if n["race_id"]})
    # an event is known if it has a token this sync didn't add (first seen earlier, or before alerts existed)
    known = set(conn.execute(text("""SELECT DISTINCT event_slug FROM market_links WHERE event_slug = ANY(:s)
                                     AND first_seen_at IS DISTINCT FROM :t"""),
                             dict(s=list({n["event_slug"] for n in new}), t=new[0]["synced_at"])).scalars())
    groups = {}
    for n in new:
        g = groups.setdefault(n["event_slug"], dict(event_slug=n["event_slug"], event_title=n.get("event_title") or "",
                                                    new_event=n["event_slug"] not in known, kinds=set(), n=0,
                                                    race=None, race_date=None, upcoming=False))
        g["kinds"].add(n["prediction"])
        g["n"] += 1
        if n["race_id"] in races:
            g["race"], d = races[n["race_id"]]
            g["race_date"] = d.isoformat()
            g["upcoming"] |= d >= today - timedelta(days=UPCOMING_DAYS)
    out = [dict(g, kinds=sorted(g["kinds"])) for g in groups.values()]
    return sorted(out, key=lambda g: (not g["upcoming"], g["race_date"] or "9999", not g["new_event"], g["event_title"]))


def _line(g):
    what = "new event" if g["new_event"] else f"+{g['n']} outcomes"
    return f"{g['event_title'] or g['event_slug']} ({what})"


def message(groups):
    """(title, body, urgent) for one notification."""
    up = [g for g in groups if g["upcoming"]]
    title = (f"New markets for the {up[0]['race'] or up[0]['event_title']}" if up
             else f"{len(groups)} new F1 market{'s' if len(groups) != 1 else ''} on Polymarket")
    body = "; ".join(_line(g) for g in groups[:4]) + (f"; +{len(groups) - 4} more" if len(groups) > 4 else "")
    return title, body, bool(up)


def _mac(title, body, urgent):
    if sys.platform != "darwin":
        return False
    q = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')     # noqa: E731
    script = f'display notification "{q(body)}" with title "racinglines" subtitle "{q(title)}"'
    script += ' sound name "Glass"' if urgent else ""
    return subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10).returncode == 0


def _ntfy(title, body, urgent, topic):
    import httpx

    from racinglines.sources import http
    with httpx.Client(base_url=NTFY, timeout=15) as c:
        r = http.post(c, f"/{topic}", content=body.encode(),
                      headers={"Title": title, "Priority": "high" if urgent else "default", "Tags": "checkered_flag",
                               "Click": os.environ.get("RACINGLINES_URL", "https://racinglines.bet") + "/markets/polymarket"})
    return r.status_code < 300


def _webhook(url, title, body, payload):
    import httpx

    from racinglines.sources import http
    with httpx.Client(timeout=15) as c:
        r = http.post(c, url, json=dict(payload or {}, title=title, text=f"*{title}*\n{body}", content=f"**{title}**\n{body}"))
    return r.status_code < 300


def deliver(title, body, urgent=False, payload=None, mac=True):
    """Send one alert through every configured channel. Returns the channels that took it; a failing
    channel is skipped (an alert must never stop the recorder or the signal run)."""
    used = []
    channels = [("mac", lambda: mac and _mac(title, body, urgent))]
    if os.environ.get("RACINGLINES_NTFY_TOPIC"):
        channels.append(("ntfy", lambda: _ntfy(title, body, urgent, os.environ["RACINGLINES_NTFY_TOPIC"])))
    if os.environ.get("ALERT_WEBHOOK_URL"):
        channels.append(("webhook", lambda: _webhook(os.environ["ALERT_WEBHOOK_URL"], title, body, payload)))
    for name, send in channels:
        try:
            if send():
                used.append(name)
        except Exception:                               # noqa: BLE001
            pass
    return used


def notify(groups, log=LOG, mac=True):
    """Announce `groups` (from summarize) and append them to the log. Returns the channels used."""
    if not groups:
        return []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        for g in groups:
            f.write(json.dumps(dict(g, at=now)) + "\n")
    title, body, urgent = message(groups)
    return ["log"] + deliver(title, body, urgent, payload=dict(kind="new_markets", events=groups), mac=mac)


def signal_line(s, code=""):
    """One signal as text, without our fair value or edge (takers never see them): the heat instead."""
    from racinglines.pipelines.signals import HEAT_LABEL
    pre = f"{code} · {s['stage']} · " if code else f"{s['stage']} · "
    subj = f'"{s["subject"]}"'
    if s["action"] in ("buy", "sell"):
        h = f" · {HEAT_LABEL[s['heat']]}" if s.get("heat") else (" · exit" if s["action"] == "sell" else "")
        return f"{pre}{s['action'].upper()} {s['side']} {s['shares']:.0f} sh {subj} @ {s['limit_price']:.2f}{h}"
    if s["action"] == "fill":
        side = "bought YES" if s["side"] == "YES" else "sold YES"
        return f"{pre}paper fill: {side} {s['shares']:.0f} sh {subj} @ {s['limit_price']:.2f}"
    d = s.get("detail") or {}
    return f"{pre}{'quoting' if s['action'] == 'quote' else 'stopped quoting'} {subj}" + (
        f" ({d['reason']})" if s["action"] == "pull" and d.get("reason") else "")


def signals_alert(profile, event, signals, mac=True):
    """One batched alert for a signal run's new signals (one user's copy). Returns the channels used."""
    if not signals:
        return []
    code = profile["name"].split(" ")[0]
    entries = [s for s in signals if s["action"] == "buy"]
    title = f"{profile['name']}: {len(signals)} new signal{'s' if len(signals) != 1 else ''} · {event['name']}"
    lines = [signal_line(s, code) for s in signals]
    body = "\n".join(lines[:12]) + (f"\n+{len(lines) - 12} more" if len(lines) > 12 else "")
    return deliver(title, body, urgent=bool(entries), mac=mac,
                   payload=dict(kind="signals", profile=profile["name"], event=event["event_key"], lines=lines))


def sync_and_alert(session, conn, year, mac=True, **kw):
    """sync(), then alert on what it found for the first time. Returns (stats, groups, channels)."""
    from racinglines.markets.polymarket.sync import Resolver, sync
    new = []
    stats = sync(session, conn, year, new=new, **kw)
    groups = summarize(conn, new, Resolver(conn, year))
    return stats, groups, notify(groups, mac=mac)


def new_links(conn, hours=NEW_HOURS):
    """Open links first seen within `hours`: token_id -> (race_id, competition_id), both exact market_links
    columns (no text-matching inference — race_for's fuzzy "Grand Prix" guess is for the CLI/notification
    summary only, never for the web app's per-sport "N new" badges and links)."""
    rows = conn.execute(text("""SELECT token_id, race_id, competition_id FROM market_links
                                WHERE NOT closed AND first_seen_at > now() - make_interval(hours => :h)"""),
                        dict(h=hours)).mappings().all()
    return {r["token_id"]: (r["race_id"], r["competition_id"]) for r in rows}
