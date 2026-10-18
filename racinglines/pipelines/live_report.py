"""
The event report for any live private-book event (the Whistler report, made repeatable):

    racinglines live report live/f1/2026-16.toml [--pdf]

Written to the run folder's report/: report.md, report.html (the same, with the charts inline), pnl.svg,
scorecard.svg and, with --pdf, report.pdf (headless Chrome / Chromium if one is installed).

Sections: the event and its settings; the book's P&L by update (vs the crowd, vs the demo taker); the crowd
(fills, volume, takers up and down); the demo taker's picks; the fair-price scorecard: every market's fair
value at each update scored against the result (Brier score and log loss per market kind), and the markets
whose price moved most. Everything comes from the run folder; nothing is re-priced.
"""

import html as H
import json
import math
import shutil
import subprocess

import numpy as np
import pandas as pd

from racinglines.pipelines import live as LV

MAX_ROWS = 12                   # updates shown in the P&L and scorecard tables (evenly spaced when there are more)
EPS = 1e-6


def _kind(key):
    """A market key's kind: F1 'race_win:<id>' -> race_win; downhill '<bib>:win' -> win."""
    a, _, b = key.partition(":")
    return a if not a.isdigit() else b


def _pick(rows, n=MAX_ROWS):
    if len(rows) <= n:
        return rows
    idx = sorted(set(np.linspace(0, len(rows) - 1, n).round().astype(int)))
    return [rows[i] for i in idx]


def load(run):
    """Everything the report reads from the run folder."""
    import gzip
    out = LV.folder(run, mkdir=False)
    snaps = []
    for t in LV.snap_times(run):
        with gzip.open(out / "snaps" / f"{t}.json.gz", "rt") as f:
            snaps.append(json.load(f))
    latest, picks, hist = LV.load(run)
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    book = json.loads((out / "book.json").read_text()) if (out / "book.json").exists() else None
    opened = LV.book_opened(run)
    return dict(snaps=[s for s in snaps if opened is None or s["ts"] >= opened or not s.get("maker_pnl")],
                latest=latest, picks=picks, hist=hist, meta=meta, book=book, opened=opened)


def outcomes_of(snap):
    out = {}
    for o in (snap or {}).get("outcomes") or []:
        out[o.get("key") or f"{o['bib']}:{o['market']}"] = bool(o["yes"])
    return out


def names_of(snap):
    """{key: display name} from a snapshot (F1 markets carry subjects; downhill riders by bib)."""
    if snap.get("markets"):
        return {m["key"]: m["subject"] for m in snap["markets"]}
    riders = {r["bib"]: r["name"] for r in snap.get("riders") or []}
    out = {}
    for q in snap.get("quotes") or []:
        if "bib" in q:
            out[f"{q['bib']}:{q['market']}"] = riders.get(q["bib"], str(q["bib"]))
        else:                                            # any other sport: the key's subject part
            out[q["key"]] = q.get("subject") or q["key"].split(":", 1)[-1]
    return out


def pnl_by_update(snaps):
    rows = []
    for s in snaps:
        p = s.get("maker_pnl")
        if not p:
            continue
        lab = (s.get("update") or {}).get("label") or s["ts"][11:19]
        rows.append(dict(ts=s["ts"], label=lab, total=p["total"], crowd=p["crowd"], taker=p["taker"],
                         fills=(s.get("crowd") or {}).get("fills")))
    return rows


def scorecard(hist, outcomes):
    """Per update and market kind: markets scored, Brier score and log loss of the fair values against the
    results (only markets with a result and a fair value)."""
    rows = []
    for h in hist:
        by = {}
        for k, f in (h.get("fair") or {}).items():
            if f is None or k not in outcomes:
                continue
            y = 1.0 if outcomes[k] else 0.0
            p = min(max(float(f), EPS), 1 - EPS)
            d = by.setdefault(_kind(k), [0, 0.0, 0.0])
            d[0] += 1
            d[1] += (p - y) ** 2
            d[2] += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        for kind, (n, b, ll) in sorted(by.items()):
            rows.append(dict(ts=h["ts"], label=h.get("label") or h["ts"][11:19], kind=kind, n=n, brier=b / n, logloss=ll / n))
    return rows


def movers(hist, outcomes, names, n=10):
    """The markets whose fair value moved most from the first update to the last: key, name, first, last, result."""
    if len(hist) < 2:
        return []
    first, last = hist[0].get("fair") or {}, hist[-1].get("fair") or {}
    rows = [dict(key=k, name=names.get(k, k), kind=_kind(k), first=first[k], last=last.get(k), result=outcomes.get(k))
            for k in first if first[k] is not None and last.get(k) is not None]
    return sorted(rows, key=lambda r: -abs(r["last"] - r["first"]))[:n]


def svg(chart, title):
    """A standalone SVG from a web.viz chart dict (colours inline, no page CSS)."""
    if not chart:
        return None
    c = chart
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {c["w"]} {c["h"] + 18}" font-family="sans-serif" font-size="11">',
         f'<text x="{c["pad"]}" y="12" font-weight="bold">{H.escape(title)}</text><g transform="translate(0,18)">']
    for y in c["yticks"]:
        p.append(f'<line x1="{c["pad"]}" x2="{c["w"]}" y1="{y["y"]}" y2="{y["y"]}" stroke="#ddd"/>'
                 f'<text x="2" y="{y["y"] + 3}" fill="#666">{H.escape(y["label"])}</text>')
    for x in c["xticks"]:
        p.append(f'<text x="{x["x"]}" y="{c["h"] - 2}" text-anchor="middle" fill="#666">{H.escape(x["label"])}</text>')
    for i, ln in enumerate(c["lines"]):
        p.append(f'<polyline fill="none" stroke="{ln["color"]}" stroke-width="1.6" points="{ln["points"]}"/>'
                 f'<text x="{c["w"] - 4}" y="{14 * (i + 1)}" text-anchor="end" fill="{ln["color"]}">{H.escape(ln["label"])}</text>')
    p.append("</g></svg>")
    return "\n".join(p)


def _money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def build(run):
    """(markdown, {file name: svg}) for a run."""
    from racinglines.web.viz import line_chart
    d = load(run)
    snaps, latest, picks, hist = d["snaps"], d["latest"], d["picks"], d["hist"]
    if not latest:
        raise ValueError(f"{run}: nothing recorded")
    oc = outcomes_of(latest)
    names = names_of(latest)
    ev = LV.find(run) or {}
    title = d["meta"].get("title") or ev.get("title") or run
    pnl = pnl_by_update(snaps)
    sc = scorecard(hist, oc)
    mv = movers(hist, oc, names)
    crowd = latest.get("crowd") or {}
    res = crowd.get("results") or {}
    mp = latest.get("maker_pnl") or {}
    L = [f"# {title}: event report", ""]
    maker = d["meta"].get("maker") or "the demo maker"
    who = f"{crowd.get('takers') or 1000:,} simulated anonymous takers" + (f" ({d['meta']['crowd']})" if d["meta"].get("crowd") else "")
    L += [f"A private-book event (`{run}`, {ev.get('sport') or d['meta'].get('sport') or 'mtb_dh'}): {maker}'s own book, "
          f"{who}{' and the demo taker' if picks else ''}, play money. Nothing was traded anywhere. "
          f"{'Settled' if latest.get('done') else 'Not settled yet: open markets are marked to fair'}. "
          f"Book opened {(d['opened'] or '?')[:16].replace('T', ' ')} UTC; last update {latest['ts'][:16].replace('T', ' ')} UTC.", ""]
    L += ["## Summary", "", "| | |", "|---|---|",
          f"| The maker's event P&L | {_money(mp.get('total', 0))} (vs the crowd {_money(mp.get('crowd', 0))}"
          + (f", vs the demo taker {_money(mp.get('taker', 0))}" if picks else "") + ") |",
          f"| Crowd | {crowd.get('fills', 0):,} fills, ${crowd.get('volume', 0):,.0f} volume, {crowd.get('active', 0)} of {crowd.get('takers', 0)} takers bet |",
          f"| The crowd's results | {res.get('up', 0)} up, {res.get('down', 0)} down; median {_money(res.get('median', 0))}, best {_money(res.get('best', 0))}, worst {_money(res.get('worst', 0))} |" if res else "| The crowd's results | – |",
          f"| Updates | {len(snaps)} snapshots; {len(hist)} priced updates |",
          f"| Markets | {len(names)}; {len(oc)} settled |", ""]
    if pnl:
        L += ["## The book's P&L by update", "",
              "Marked to fair at each update (settled where the result was known).", "",
              "| Update | UTC | Crowd fills | vs the crowd |" + (" vs the demo taker |" if picks else "") + " Total |",
              "|---|---|---|---|" + ("---|" if picks else "") + "---|"]
        for r in _pick(pnl):
            L.append(f"| {r['label']} | {r['ts'][5:16].replace('T', ' ')} | {r['fills'] or 0:,} | {_money(r['crowd'])} |"
                     + (f" {_money(r['taker'])} |" if picks else "") + f" **{_money(r['total'])}** |")
        L += ["", "![The book's P&L](pnl.svg)", ""]
    if picks:
        L += ["## The demo taker's picks", "", "| Bet | Paid | Shares | Result | Taker P&L |", "|---|---|---|---|---|"]
        tot = 0.0
        for p in picks:
            k = p.get("key") or f"{p['bib']}:{p['market']}"
            y = oc.get(k)
            v = p["shares"] * float(y) - p["stake"] if y is not None else None
            tot += v or 0.0
            what = p.get("subject") or p.get("rider") or k
            L.append(f"| YES · {what} ({_kind(k)}) | {p['price']:.2f} | {p['shares']:.0f} | {'YES' if y else ('NO' if y is not None else 'open')} | {_money(v) if v is not None else '–'} |")
        L += ["", f"Together: {_money(tot)} on ${sum(p['stake'] for p in picks):,.0f} staked.", ""]
    if sc:
        L += ["## Fair-price scorecard", "",
              "Every market's fair value at each update, scored against the result: Brier score (0 is perfect; "
              "0.25 is a coin flip on a 50/50 market) and log loss, per market kind. Lower is better.", "",
              "| Update | Kind | Markets | Brier | Log loss |", "|---|---|---|---|---|"]
        labels = [h.get("label") or h["ts"][11:19] for h in hist]
        keep = set(_pick(labels))
        for r in sc:
            if r["label"] in keep:
                L.append(f"| {r['label']} | {r['kind']} | {r['n']} | {r['brier']:.4f} | {r['logloss']:.4f} |")
        L += ["", "![Brier score by update](scorecard.svg)", ""]
    if mv:
        L += ["### The biggest moves", "", "| Market | First | Last | Result |", "|---|---|---|---|"]
        for r in mv:
            L.append(f"| {r['name']} ({r['kind']}) | {r['first']:.1%} | {r['last']:.1%} | "
                     f"{'YES' if r['result'] else ('NO' if r['result'] is not None else 'open')} |")
        L.append("")
    s = d["meta"].get("settings")
    if s:
        L += ["## Settings (frozen at the opening)", "", "```json", json.dumps(s, indent=1, default=str)[:4000], "```", ""]
    from racinglines import attribution                  # the compact source note (results feed, model as of the last update)
    note = attribution.note(ev.get("sport") or d["meta"].get("sport") or "mtb_dh", None, None, latest["ts"], model=True)["text"]
    if note:
        L += [note + ".", ""]
    svgs = {}
    if pnl:
        ts = [pd.Timestamp(r["ts"]) for r in pnl]
        series = {"total": list(zip(ts, [r["total"] for r in pnl])), "crowd": list(zip(ts, [r["crowd"] for r in pnl]))}
        if picks:
            series["taker"] = list(zip(ts, [r["taker"] for r in pnl]))
        svgs["pnl.svg"] = svg(line_chart(series, {"total": "total", "crowd": "vs the crowd", "taker": "vs the demo taker"}),
                              "The maker's P&L ($), marked to fair")
    if sc:
        series = {}
        for r in sc:
            series.setdefault(r["kind"], []).append((pd.Timestamp(r["ts"]), r["brier"]))
        svgs["scorecard.svg"] = svg(line_chart(series, {k: k for k in series}, money=False), "Brier score by update")
    return "\n".join(L), {k: v for k, v in svgs.items() if v}


def to_html(md, svgs, title):
    try:
        import markdown
        body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:                                  # the docs requirements carry it; else plain text
        body = f"<pre>{H.escape(md)}</pre>"
    import re
    for name, s in svgs.items():                         # the charts inline, in place of their <img> tags
        body = re.sub(rf'<img[^>]*src="{re.escape(name)}"[^>]*>', lambda _: s, body)
    css = ("body{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#111}"
           "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:3px 7px}img{max-width:100%}"
           "pre{font-size:11px;background:#f6f6f6;padding:8px;overflow:auto}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{H.escape(title)}</title><style>{css}</style></head><body>{body}</body></html>"


def chrome():
    for exe in ("google-chrome", "chromium", "chromium-browser", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"):
        path = shutil.which(exe) or (exe if exe.startswith("/") and __import__("os").path.exists(exe) else None)
        if path:
            return path
    return None


def write(spec, pdf=False):
    """Write the report into the run folder's report/. Returns {name: path}."""
    run = spec["run"] if isinstance(spec, dict) else spec
    md, svgs = build(run)
    out = LV.folder(run) / "report"
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text(md)
    for name, s in svgs.items():
        (out / name).write_text(s)
    title = md.splitlines()[0].lstrip("# ")
    (out / "report.html").write_text(to_html(md, svgs, title))
    files = dict(md=out / "report.md", html=out / "report.html", **{n: out / n for n in svgs})
    if pdf:
        exe = chrome()
        if exe is None:
            files["pdf"] = "no Chrome / Chromium found: open report.html and print it"
        else:
            subprocess.run([exe, "--headless", "--disable-gpu", "--no-sandbox", f"--print-to-pdf={out / 'report.pdf'}",
                            (out / "report.html").as_uri()], capture_output=True, timeout=120)
            files["pdf"] = out / "report.pdf" if (out / "report.pdf").exists() else "Chrome failed to print"
    return files
