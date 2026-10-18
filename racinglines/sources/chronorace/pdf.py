"""
ChronoRace result PDFs -> the downloader's markdown tables, for rounds with no live timing
(2021 Leogang and Les Gets, some Timed Training rounds). Used by `mtb_dh download --pdf-results`
(off by default).

The text comes from poppler's `pdftotext -layout`, which keeps the columns lined up (a plain text
extraction repeats the bold letters, `BBBBRRRROOOO`). Two layouts:

- **Results** (Qualifying, Semi-Final, Final): one rider per two lines. Line 1: rank, bib, name, UCI ID,
  nation, year of birth, speed, intermediates I1 and I3 with their ranks, time, points. Line 2: team,
  I2 and I4, gap. Riders who didn't finish have no rank and a status (DNF/DNS/DSQ) for a time.
- **Timed Training**: the runs side by side under "RUN 1 / RUN 2 / ... / Best" (three in 2021, five in
  2025, where the speed column is gone and the nation follows the name in brackets). Each rider takes
  one line per split; the best run's splits are kept (the live-timing JSON also gives one run). There's
  no UCI ID in this layout.

The footer's weather, temperature and track length are written as a `Weather:` line under the
round's heading (the parser ignores it).
"""

import re
import shutil
import subprocess

TIME = r"\d+:\d{2}\.\d{3}"
TIME_RE = re.compile(rf"^{TIME}$")
STATUS = {"DNF", "DNS", "DSQ"}

# results layout: "  1. P     6 BROSNAN Troy     10007307417   AUS   1993   48.943 (38) ..."
RESULT_ROW = re.compile(
    r"^\s*(?:(?P<pos>\d+)\.)?\s*(?:P\s+)?(?P<bib>\d+)\s+(?P<name>\S.*?)\s{2,}"
    r"(?P<uci>\d{11})\s+(?P<nat>[A-Z]{3})\s+(?P<yob>\d{4})\s+(?P<rest>.*)$")
# timed training: "    1.    1 BROSNAN Troy      AUS     41.410   1:29.065 ..." (2021),
#                  "    1.    3 ALRAN Till (FRA)            0:45.530 ..." (2025: no speed column)
TT_ROW = re.compile(r"^\s*(?:(?P<pos>\d+)\.)?\s+(?P<bib>\d+)\s+(?P<name>\S.*?)"
                    r"(?:\s{2,}(?P<nat>[A-Z]{3})|\s+\((?P<nat2>[A-Z]{3})\))\s{2,}(?P<rest>.*)$")
PAIR = re.compile(r"(-|[\d:.]+)\s*\(\s*\d+\s*\)")
WEATHER = re.compile(r"^\s*\d+\s*/\s*\d+\s+(?:\d+\s+){4}(?P<weather>\S.*?)\s{2,}(?P<temp>-?\d+\s*°\s*[cC])\s+"
                     r"(?P<dist>[\d.]+\s*km)")


def pdf_text(data):
    """`pdftotext -layout` of a PDF's bytes (poppler, `brew install poppler`)."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError("pdftotext not found: install poppler (brew install poppler) to read result PDFs")
    return subprocess.run([exe, "-layout", "-", "-"], input=data, capture_output=True, check=True).stdout.decode()


def _row(pos, bib, name, team, nat, uci, splits, time, gap, status):
    return dict(pos=pos or "", bib=bib, name=name, team=team, nation=nat, uci_id=uci,
                splits=splits, time=time, gap=gap, status=status)


def _team(line):
    """Team text at the start of a continuation line (before the first time or dash column)."""
    m = re.match(r"^\s*(.*?)(?:\s{2,}|$)", line)
    t = m.group(1).strip() if m else ""
    return "" if (not t or t == "-" or re.match(rf"^[+-]?{TIME}|^[+-]?\d+\.\d{{3}}", t)) else t


def parse_results(text):
    """Rider rows from a Qualifying/Final results PDF's text."""
    lines = text.splitlines()
    rows = []
    i = 0
    while i < len(lines):
        m = RESULT_ROW.match(lines[i])
        if not m:
            i += 1
            continue
        rest = m["rest"]
        pairs = PAIR.findall(rest)
        tail = PAIR.sub(" ", rest).split()
        line1 = [p for p in pairs if TIME_RE.match(p)]  # the speed pair has no colon
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        cont = not RESULT_ROW.match(nxt) and bool(nxt.strip()) and not nxt.lstrip().startswith("Entries")
        line2 = PAIR.findall(nxt) if cont else []
        splits = []
        for k in range(max(len(line1), len(line2))):   # I1, I2, I3, I4 ...
            for col in (line1, line2):
                if k < len(col):
                    splits.append(col[k] if TIME_RE.match(col[k]) else "")
        time = next((t for t in tail if TIME_RE.match(t)), "")
        status = next((t for t in tail if t in STATUS), "")
        gap = ""
        if cont:
            g = re.search(rf"\+{TIME}|\+\d+\.\d{{3}}", PAIR.sub(" ", nxt))
            gap = g.group(0) if g else ""
        rows.append(_row(m["pos"], m["bib"], m["name"].strip(), _team(nxt) if cont else "", m["nat"], m["uci"],
                         splits, time, gap, "" if time else (status or "DNF")))
        i += 2 if cont else 1
    return rows


def _columns(header):
    """[(role, run, right edge)] from a timed-training column-header line
    ("Rank Nr Name / UCI MTB Team NAT Speed Splits Time Speed Splits Time ... Time"): the values are
    right-aligned under these words. Runs count from 0; the last "Time" is the best time (run None)."""
    cols = []
    run = -1
    for w in re.finditer(r"\S+", header):
        word = w.group(0)
        if word == "Splits":                             # one per run (Speed, where present, is ignored)
            run += 1
        role = word.lower() if word in ("Splits", "Time") and run >= 0 else "other"
        cols.append([role, run, w.end()])
    times = [c for c in cols if c[0] == "time"]
    if len(times) > run + 1:                             # one more Time than runs: the Best column
        times[-1][0], times[-1][1] = "best", None
    return cols


def _column(cols, tok_end):
    return min(cols, key=lambda c: abs(c[2] - tok_end))


def parse_timed_training(text):
    """Rider rows (best run) from a Timed Training PDF's text."""
    lines = text.splitlines()
    rows = []
    cols = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.search(r"\bSplits\b.*\bTime\b.*\bSplits\b", line):
            cols = _columns(line)
            i += 1
            continue
        m = TT_ROW.match(line) if cols else None
        if not m:
            i += 1
            continue
        n_runs = max(c[1] for c in cols if c[1] is not None) + 1
        runs = [dict(splits=[], time="") for _ in range(n_runs)]
        best = gap = ""
        for t in re.finditer(r"\S+", line[m.start("rest"):]):
            tok = t.group(0)
            role, run, _ = _column(cols, m.start("rest") + t.end())
            if not TIME_RE.match(tok):
                continue
            if role == "best":
                best = tok
            elif role == "splits":
                runs[run]["splits"].append(tok)
            elif role == "time":
                runs[run]["time"] = tok
        team = ""
        j = i + 1
        while j < len(lines) and lines[j].strip() and not TT_ROW.match(lines[j]) \
                and not lines[j].lstrip().startswith(("Entries", "Timing")):
            part = _team(lines[j]) if lines[j][:m.start("rest")].strip() else ""
            team = f"{team} {part}".strip() if part else team      # a long team name wraps onto the next line
            for t in re.finditer(r"\S+", lines[j]):
                tok = t.group(0)
                role, run, _ = _column(cols, t.end())
                if role == "best" and tok.startswith("+"):
                    gap = tok
                elif role == "splits" and (TIME_RE.match(tok) or tok == "-"):
                    runs[run]["splits"].append(tok if tok != "-" else "")
            j += 1
        run = next((r for r in runs if best and r["time"] == best), None)
        splits = run["splits"] if run else []
        rows.append(_row(m["pos"], m["bib"], m["name"].strip(), team, m["nat"] or m["nat2"], "", splits, best,
                         gap if gap != "+0.000" else "", "" if best else "DNF"))
        i = j
    return rows


def weather(text):
    """'Mostly Sunny, 23°C, 2.174km' from a results PDF's footer, or None."""
    for line in text.splitlines():
        m = WEATHER.match(line)
        if m:
            return f"{m['weather'].strip()}, {m['temp'].replace(' ', '').replace('c', 'C')}, {m['dist'].replace(' ', '')}"
    return None


def parse(text):
    """Rider rows from either layout."""
    return parse_timed_training(text) if re.search(r"RUN 1.*RUN 2", text) else parse_results(text)


def round_table(text, round_name):
    """A '## <round>' markdown section in the downloader's table format, or None if no riders."""
    rows = parse(text)
    if not rows:
        return None
    n = max(len(r["splits"]) for r in rows)
    header = "| Pos | Bib | Rider | Team | Nation | UCI ID |" + "".join(f" Split {k} |" for k in range(1, n + 1))
    out = [header + " Time | Gap | Status |", "|---|---|---|---|---|---|" + "---|" * n + "---|---|---|"]
    for r in rows:
        cells = r["splits"] + [""] * (n - len(r["splits"]))
        out.append(f"| {r['pos']} | {r['bib']} | {r['name']} | {r['team']} | {r['nation']} | {r['uci_id']} |"
                   + "".join(f" {c} |" for c in cells) + f" {r['time']} | {r['gap']} | {r['status']} |")
    w = weather(text)
    note = f"(From the result PDF: no live timing for this round.){f' Weather: {w}.' if w else ''}\n\n"
    return f"## {round_name}\n\n{note}" + "\n".join(out) + "\n"
