#!/usr/bin/env python3
"""
racinglines data parse mtb_dh: ingest UCI DH (ChronoRace-style) result exports into one tidy
long-format CSV the timed-runs model can consume.

WHY IT'S BUILT THIS WAY
------------------------
The ChronoRace results page (prod.chronorace.be/angular/...) is an Angular
single-page app: the HTML you get from a plain HTTP fetch is an empty shell,
and the actual rider/split data is loaded afterwards by JavaScript. That means
there's no single fixed table structure to hard-code against sight unseen.

So this script is deliberately a "point it at what you actually have" tool
with three input paths, auto-detected by file extension:

  1. .html / .htm  -> a SAVED, fully-rendered page (Ctrl+S / "Save As" after
                       the results table has loaded in your browser), or the
                       HTML of a copy-pasted table. Parsed with BeautifulSoup
                       + pandas.read_html.
  2. .csv           -> an export you already have (e.g. from the site's own
                       "export" button, if it has one, or a manual copy/paste
                       into a spreadsheet saved as CSV).
  3. .json          -> a raw JSON export, e.g. if you captured the XHR
                       response the Angular app makes internally (browser
                       devtools -> Network tab -> XHR -> Save Response).
  4. .md / .txt     -> a select-all + copy/paste of the rendered results
                       page saved as text. Each rider is a block of lines:
                       "1.\tn°100", a blank (flag) line, NAME, optional team,
                       then "cum_time (rank)" splits interleaved with +/- gaps.
                       Filenames like "uci:event:20260821_mtb:DHI:CG1:dh:91:res.md"
                       are normalized to event_id "20260821_mtb_DHI_CG1_dh_91".
  5. .md (tables)   -> one file per event as written by racinglines/sources/chronorace/download.py
                       (data/raw/mtb_dh/chronorace/*.md): "# <event title>", a "Source: ... <slug>"
                       line, then one "## <round>" section per round, each a
                       markdown table with Pos | Bib | Rider | Team | Nation |
                       Split 1..N | Time | Gap | Status. Auto-detected by
                       content, so .md files of either kind can share a dir.
                       Adds venue / series_round / nation columns.

Run with --inspect first on one file from each source you have. It won't
write output — it just prints the columns/keys it detected, so we can adjust
COLUMN_MAP below (or split the mapping logic) to match your actual headers
in one pass instead of guessing blind.

OUTPUT SCHEMA (long format, one row per rider per sector per run)
-------------------------------------------------------------------
event_id        str   e.g. "20260821_mtb_DHI_CG1_dh_91"
event_date      str   ISO date, YYYY-MM-DD
discipline      str   e.g. "DHI"
category        str   e.g. "CG1" (elite men / elite women / junior etc.)
round           str   "qual" | "semi" | "final" | "practice" | "seeding"
rider_id        str   stable rider identifier (UCI ID if available, else
                       normalized name — see normalize_rider_id())
rider_name      str
team            str   (optional, blank if unknown)
bib             str   (optional)
start_order     int   start/gate order within the round, if known
sector_id       str   "S1", "S2", ... or "FINISH" for the full run
split_time_s    float seconds for this sector (not cumulative)
cum_time_s      float cumulative time at this split, seconds
rank_at_split   int   rank among finishers at this split, this round
status          str   "OK" | "DNF" | "DNS" | "DSQ"
track_condition str   "dry" | "wet" | "mixed" | "unknown" (fill in manually
                       if the site doesn't expose it — see --conditions-file)

Usage
-----
    racinglines data parse mtb_dh --inspect raw/some_event.html
    racinglines data parse mtb_dh --input-dir raw/ --out splits.csv
    racinglines data parse mtb_dh --input-dir raw/ --out splits.csv --conditions-file conditions.csv
"""

import argparse
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# CALIBRATION ZONE — adjust these once you run --inspect on your real files.
# Keys are the header text you'll actually see; values are our schema names.
# Matching is case-insensitive and substring-based (see _map_columns).
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "rank": "rank_at_split",
    "clt": "rank_at_split",          # French "classement"
    "pos": "rank_at_split",
    "bib": "bib",
    "dossard": "bib",
    "no": "bib",
    "name": "rider_name",
    "nom": "rider_name",
    "athlete": "rider_name",
    "rider": "rider_name",
    "uci": "rider_id",
    "uci id": "rider_id",
    "licence": "rider_id",
    "team": "team",
    "equipe": "team",
    "nat": "nation",
    "nation": "nation",
    "status": "status",
    "result": "status",
}

# Sector/split columns are usually headed "Split 1", "Inter1", "I1", "S1",
# "Finish", "Total", "Net time", etc. This regex just needs to catch the
# common ChronoRace-family naming; tune it after --inspect.
SECTOR_COL_RE = re.compile(
    r"(split|inter|sect|s\d+|i\d+)\s*_?(\d+)?", re.IGNORECASE
)
FINISH_COL_RE = re.compile(r"(finish|total|net\s*time|result\s*time)", re.IGNORECASE)

# Columns already claimed by COLUMN_MAP targets are never sector/finish columns,
# even if their raw header text happens to contain "split"/"finish" etc.
# (e.g. a "Rank" header mapped to rank_at_split would otherwise false-match
# SECTOR_COL_RE's "split" pattern via "rank_at_split").
NON_SECTOR_SCHEMA_COLS = {
    "rank_at_split", "bib", "rider_name", "rider_id", "team", "nation", "status",
}

TIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*$")  # [h:]mm:ss[.ms]

STATUS_TOKENS = {"dnf": "DNF", "dns": "DNS", "dsq": "DSQ", "dq": "DSQ"}


def parse_time_to_seconds(raw):
    """Convert '1:23.456', '23.456', 'DNF', '+1:02.3' etc. to seconds (float)
    or a status string. Returns (seconds_or_None, status_or_None)."""
    if raw is None:
        return None, None
    s = str(raw).strip()
    if s == "" or s in {"-", "--"}:
        return None, None
    low = s.lower().replace(".", "")
    for tok, status in STATUS_TOKENS.items():
        if tok in low:
            return None, status
    s = s.lstrip("+")  # gap times are often shown as "+12.345"
    m = TIME_RE.match(s)
    if m:
        h, mm, ss = m.groups()
        total = float(mm) * 60 + float(ss)
        if h:
            total += float(h) * 3600
        return total, None
    try:
        return float(s), None
    except ValueError:
        return None, None  # unparseable — leave for manual review


def normalize_rider_id(name, uci_id=None):
    """Stable-ish rider key. Prefer a real UCI ID; fall back to a normalized
    name so the same rider merges across events even if IDs are missing."""
    if uci_id and str(uci_id).strip():
        return f"uci:{str(uci_id).strip()}"
    # ChronoRace spells the same rider differently across seasons: accents
    # present or dropped ("Léo"/"Leo"), a trailing " *" marker, double spaces,
    # apostrophe or space.
    n = unicodedata.normalize("NFKD", clean_rider_name(name)).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-z ]", "", re.sub(r"['’`]", " ", n.lower()))   # O'CALLAGHAN == O CALLAGHAN
    return f"name:{re.sub(r' +', ' ', n).strip()}"


def clean_rider_name(name):
    """Display name without ChronoRace's trailing '*' marker or doubled spaces."""
    return re.sub(r"\s+", " ", str(name or "").replace("*", " ")).strip()


def parse_event_meta_from_id(stem):
    """Pull (date, discipline, category) out of a filename/URL-slug that
    follows the pattern seen in the ChronoRace URL, e.g.:
      20260821_mtb_DHI_CG1_dh_91[_qual|_semi|_final]
    Falls back to Nones if the pattern doesn't match — fill in manually.
    """
    parts = re.split(r"[_/:]", stem)
    date = discipline = category = round_hint = None
    for p in parts:
        if re.fullmatch(r"\d{8}", p):
            date = f"{p[0:4]}-{p[4:6]}-{p[6:8]}"
        elif re.fullmatch(r"[A-Z]{2,4}\d?", p) and discipline is None and p.lower() != "mtb":
            discipline = p
        elif re.fullmatch(r"[A-Z]{2}\d", p):
            category = p
        elif p.lower() in {"qual", "quali", "qualifying", "semi", "final", "practice", "seeding"}:
            round_hint = p.lower()
    return date, discipline, category, round_hint


def _map_columns(columns):
    """Fuzzy-map raw header strings to our schema names using COLUMN_MAP,
    falling back to the raw (lowercased, underscored) header if unmatched."""
    mapped = {}
    for c in columns:
        key = str(c).strip().lower()
        target = None
        for pattern, schema_name in COLUMN_MAP.items():
            if pattern in key:
                target = schema_name
                break
        if target is None:
            target = re.sub(r"\s+", "_", key)
        mapped[c] = target
    return mapped


def inspect_file(path):
    path = Path(path)
    print(f"--- Inspecting {path} ---")
    if path.suffix.lower() in (".html", ".htm"):
        html = path.read_text(errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        print(f"Found {len(tables)} <table> element(s).")
        for i, t in enumerate(tables):
            try:
                df = pd.read_html(io.StringIO(str(t)))[0]
            except ValueError:
                continue
            print(f"\nTable #{i}: shape={df.shape}")
            print("Columns:", list(df.columns))
            print(df.head(3).to_string())
        if not tables:
            print("No <table> tags found. If this file is the raw page shell "
                  "(no data), you need to save the page AFTER it renders "
                  "(open in browser, wait for results to load, then Ctrl+S "
                  "'Webpage, Complete'), or export the underlying XHR JSON "
                  "from the Network tab instead.")
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        print(type(data))
        if isinstance(data, list) and data:
            print("First item keys:", list(data[0].keys()))
            print(json.dumps(data[0], indent=2)[:2000])
        elif isinstance(data, dict):
            print("Top-level keys:", list(data.keys()))
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=5)
        print("Columns:", list(df.columns))
        print(df.to_string())
    elif path.suffix.lower() in (".md", ".txt") and is_markdown_tables_file(path):
        df = pd.DataFrame(parse_markdown_tables_file(path))
        fin = df[df["sector_id"] == "FINISH"]
        print("Event:", fin[["event_id", "event_date", "venue", "series_round", "category"]].iloc[0].to_dict())
        print(fin.groupby("round")["status"].value_counts().unstack(fill_value=0).to_string())
        print(df.head(6).to_string())
    elif path.suffix.lower() in (".md", ".txt"):
        header, blocks = _split_text_page(path.read_text(errors="ignore"))
        print("Event id:", normalize_event_id(path.stem))
        print("Meta (date, discipline, category, round):", parse_event_meta_from_id(path.stem))
        print("Header lines:", [l for l in header if l.strip()])
        print(f"Found {len(blocks)} rider block(s).")
        for rank, bib, lines in blocks[:3]:
            print(f"  rank={rank} bib={bib} ->", _parse_text_block(lines))
    else:
        print("Unrecognized extension — expected .html/.htm, .json, .csv, .md or .txt")


def parse_html_file(path, default_round=None):
    html = Path(path).read_text(errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    rows_out = []
    stem = Path(path).stem
    date, discipline, category, round_hint = parse_event_meta_from_id(stem)
    round_ = default_round or round_hint or "unknown"
    event_id = stem

    for table in tables:
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            continue
        df.columns = [str(c) for c in df.columns]
        colmap = _map_columns(df.columns)
        df = df.rename(columns=colmap)

        sector_cols = [c for c in df.columns if c not in NON_SECTOR_SCHEMA_COLS and SECTOR_COL_RE.search(str(c))]
        finish_cols = [c for c in df.columns if c not in NON_SECTOR_SCHEMA_COLS and FINISH_COL_RE.search(str(c))]

        for _, row in df.iterrows():
            rider_name = row.get("rider_name")
            if not rider_name or str(rider_name).strip() == "":
                continue
            uci_id = row.get("rider_id")
            rider_id = normalize_rider_id(rider_name, uci_id)
            bib = row.get("bib")
            team = row.get("team")
            status_raw = row.get("status")
            base_status = STATUS_TOKENS.get(str(status_raw).strip().lower(), "OK") if status_raw else "OK"

            cum = 0.0
            for sc in sector_cols:
                secs, status = parse_time_to_seconds(row.get(sc))
                sector_status = status or base_status
                if secs is not None:
                    split_time = secs - cum if cum else secs
                    cum = secs
                else:
                    split_time = None
                rows_out.append(dict(
                    event_id=event_id, event_date=date, discipline=discipline,
                    category=category, round=round_, rider_id=rider_id,
                    rider_name=rider_name, team=team, bib=bib, start_order=None,
                    sector_id=str(sc), split_time_s=split_time, cum_time_s=secs,
                    rank_at_split=row.get("rank_at_split"), status=sector_status,
                    track_condition="unknown",
                ))
            for fc in finish_cols:
                secs, status = parse_time_to_seconds(row.get(fc))
                rows_out.append(dict(
                    event_id=event_id, event_date=date, discipline=discipline,
                    category=category, round=round_, rider_id=rider_id,
                    rider_name=rider_name, team=team, bib=bib, start_order=None,
                    sector_id="FINISH", split_time_s=None, cum_time_s=secs,
                    rank_at_split=row.get("rank_at_split"),
                    status=status or base_status, track_condition="unknown",
                ))
    return rows_out


def parse_csv_file(path, default_round=None):
    """Best-effort ingest of a CSV export. Assumes it's already close to
    tabular-per-rider (wide). Reuses the same column-mapping + time-parsing
    logic as the HTML path."""
    df = pd.read_csv(path)
    df.columns = [str(c) for c in df.columns]
    colmap = _map_columns(df.columns)
    df = df.rename(columns=colmap)
    stem = Path(path).stem
    date, discipline, category, round_hint = parse_event_meta_from_id(stem)
    round_ = default_round or round_hint or "unknown"
    event_id = stem

    sector_cols = [c for c in df.columns if c not in NON_SECTOR_SCHEMA_COLS and SECTOR_COL_RE.search(str(c))]
    finish_cols = [c for c in df.columns if c not in NON_SECTOR_SCHEMA_COLS and FINISH_COL_RE.search(str(c))]
    rows_out = []
    for _, row in df.iterrows():
        rider_name = row.get("rider_name")
        if not rider_name or str(rider_name).strip() == "":
            continue
        rider_id = normalize_rider_id(rider_name, row.get("rider_id"))
        base_status = STATUS_TOKENS.get(str(row.get("status")).strip().lower(), "OK") if row.get("status") else "OK"
        cum = 0.0
        for sc in sector_cols:
            secs, status = parse_time_to_seconds(row.get(sc))
            if secs is not None:
                split_time = secs - cum if cum else secs
                cum = secs
            else:
                split_time = None
            rows_out.append(dict(
                event_id=event_id, event_date=date, discipline=discipline,
                category=category, round=round_, rider_id=rider_id,
                rider_name=rider_name, team=row.get("team"), bib=row.get("bib"),
                start_order=row.get("start_order"), sector_id=str(sc),
                split_time_s=split_time, cum_time_s=secs,
                rank_at_split=row.get("rank_at_split"),
                status=status or base_status, track_condition="unknown",
            ))
        for fc in finish_cols:
            secs, status = parse_time_to_seconds(row.get(fc))
            rows_out.append(dict(
                event_id=event_id, event_date=date, discipline=discipline,
                category=category, round=round_, rider_id=rider_id,
                rider_name=rider_name, team=row.get("team"), bib=row.get("bib"),
                start_order=row.get("start_order"), sector_id="FINISH",
                split_time_s=None, cum_time_s=secs,
                rank_at_split=row.get("rank_at_split"),
                status=status or base_status, track_condition="unknown",
            ))
    return rows_out


def parse_json_file(path, default_round=None):
    """Best-effort ingest of a raw XHR JSON dump. Structure varies a lot
    between timing systems, so this handles the common shape (a list of
    rider dicts, each with a nested list of splits) and otherwise dumps
    a flat pass-through for manual inspection. Adjust the key names below
    to match --inspect output on your actual JSON."""
    data = json.loads(Path(path).read_text())
    stem = Path(path).stem
    date, discipline, category, round_hint = parse_event_meta_from_id(stem)
    round_ = default_round or round_hint or "unknown"
    event_id = stem

    riders = data if isinstance(data, list) else data.get("results") or data.get("riders") or []
    rows_out = []
    for r in riders:
        rider_name = r.get("name") or r.get("rider_name") or r.get("athlete")
        if not rider_name:
            continue
        rider_id = normalize_rider_id(rider_name, r.get("uci_id") or r.get("licence"))
        status = STATUS_TOKENS.get(str(r.get("status", "")).strip().lower(), "OK")
        splits = r.get("splits") or r.get("intermediates") or []
        cum = 0.0
        for i, sp in enumerate(splits, start=1):
            secs = sp.get("time_s") if isinstance(sp, dict) else sp
            if isinstance(secs, str):
                secs, st = parse_time_to_seconds(secs)
                status = st or status
            split_time = (secs - cum) if (secs is not None and cum) else secs
            if secs is not None:
                cum = secs
            rows_out.append(dict(
                event_id=event_id, event_date=date, discipline=discipline,
                category=category, round=round_, rider_id=rider_id,
                rider_name=rider_name, team=r.get("team"), bib=r.get("bib"),
                start_order=r.get("start_order"), sector_id=f"S{i}",
                split_time_s=split_time, cum_time_s=secs,
                rank_at_split=None, status=status, track_condition="unknown",
            ))
        finish_secs, fstatus = parse_time_to_seconds(r.get("finish_time") or r.get("total_time"))
        rows_out.append(dict(
            event_id=event_id, event_date=date, discipline=discipline,
            category=category, round=round_, rider_id=rider_id,
            rider_name=rider_name, team=r.get("team"), bib=r.get("bib"),
            start_order=r.get("start_order"), sector_id="FINISH",
            split_time_s=None, cum_time_s=finish_secs,
            rank_at_split=r.get("rank"), status=fstatus or status,
            track_condition="unknown",
        ))
    return rows_out


# --- Copy/pasted text page (.md / .txt) -------------------------------------
TEXT_BIB_RE = re.compile(r"^(?:(\d+)\.\s+)?n°\s*(\d+)\s*$")   # "1.\tn°100" or "n°71"
TEXT_SPLIT_RE = re.compile(r"^(\S+)\s*\((\d+)\)$")             # "1:49.498 (5)"
CLOCK_RE = re.compile(r"^(?:\d+:){0,2}\d+\.\d+$")               # "4:07.776", "35.954"
TEXT_ROUND_RE = [
    (re.compile(r"qualif", re.IGNORECASE), "qual"),
    (re.compile(r"semi", re.IGNORECASE), "semi"),
    (re.compile(r"seeding", re.IGNORECASE), "seeding"),
    (re.compile(r"practice|training", re.IGNORECASE), "practice"),
    (re.compile(r"final", re.IGNORECASE), "final"),
]


def normalize_event_id(stem):
    """'uci:event:20260821_mtb:DHI:CG1:dh:91:res' -> '20260821_mtb_DHI_CG1_dh_91'."""
    s = re.sub(r"^uci:event:", "", stem)
    s = re.sub(r":res$", "", s)
    return s.replace(":", "_")


def _split_text_page(text):
    """Split a pasted results page into (header_lines, rider_blocks), where each
    block is (overall_rank_or_None, bib, [lines after the bib line])."""
    lines = text.splitlines()
    header, blocks = [], []
    for line in lines:
        m = TEXT_BIB_RE.match(line.strip())
        if m:
            blocks.append((int(m.group(1)) if m.group(1) else None, m.group(2), []))
        elif blocks:
            blocks[-1][2].append(line)
        else:
            header.append(line)
    return header, blocks


def _parse_text_block(lines):
    """Return (rider_name, team, [(cum_s, rank), ...], finish_s, status) for one
    rider block. Tokens after the name are classified by shape, so a missing
    team line or missing split columns (DNF/DSQ) doesn't shift anything."""
    body = [l for l in lines if l.strip()]
    if not body:
        return None, None, [], None, None
    rider_name = body[0].strip()
    team, splits, finish_s, status = None, [], None, None
    tokens = [t.strip() for l in body[1:] for t in l.split("\t") if t.strip()]
    for tok in tokens:
        if tok[0] in "+-":
            continue  # gap to leader, recomputable from times
        if tok.lower() in STATUS_TOKENS:
            status = STATUS_TOKENS[tok.lower()]
            continue
        m = TEXT_SPLIT_RE.match(tok)
        if m and CLOCK_RE.match(m.group(1)):
            splits.append((parse_time_to_seconds(m.group(1))[0], int(m.group(2))))
        elif CLOCK_RE.match(tok):
            finish_s = parse_time_to_seconds(tok)[0]
        elif team is None and not splits:
            team = tok
    return rider_name, team, splits, finish_s, status


def parse_text_file(path, default_round=None):
    text = Path(path).read_text(errors="ignore")
    stem = Path(path).stem
    date, discipline, category, round_hint = parse_event_meta_from_id(stem)
    header, blocks = _split_text_page(text)

    header_round = None
    for line in header:
        if "live timing" in line.lower() or "results" in line.lower():
            for rx, label in TEXT_ROUND_RE:
                if rx.search(line):
                    header_round = label
                    break
        if header_round:
            break
    round_ = default_round or round_hint or header_round or "unknown"
    event_id = normalize_event_id(stem)

    n_sectors = max((len(re.findall(r"split\s*\d+", l, re.IGNORECASE)) for l in header), default=0)

    rows_out = []
    for overall_rank, bib, lines in blocks:
        rider_name, team, splits, finish_s, status = _parse_text_block(lines)
        if not rider_name:
            continue
        status = status or ("OK" if finish_s is not None else "DNF")
        common = dict(
            event_id=event_id, event_date=date, discipline=discipline,
            category=category, round=round_,
            rider_id=normalize_rider_id(rider_name), rider_name=rider_name,
            team=team, bib=bib, start_order=None, track_condition="unknown",
        )
        cum = 0.0
        for i in range(1, max(n_sectors, len(splits)) + 1):
            if i <= len(splits):
                secs, rank = splits[i - 1]
                rows_out.append(dict(
                    common, sector_id=f"S{i}", split_time_s=secs - cum,
                    cum_time_s=secs, rank_at_split=rank, status="OK",
                ))
                cum = secs
            else:
                rows_out.append(dict(
                    common, sector_id=f"S{i}", split_time_s=None,
                    cum_time_s=None, rank_at_split=None, status=status,
                ))
        rows_out.append(dict(
            common, sector_id="FINISH",
            split_time_s=(finish_s - cum) if finish_s is not None else None,
            cum_time_s=finish_s, rank_at_split=overall_rank, status=status,
        ))
    return rows_out


# --- Downloaded per-event markdown tables (data/*.md) ------------------------
TABLE_ROUND_MAP = [
    (re.compile(r"timed\s*training|practice", re.IGNORECASE), "practice"),
    (re.compile(r"qualif\w*\s*1", re.IGNORECASE), "qual1"),
    (re.compile(r"qualif\w*\s*2", re.IGNORECASE), "qual2"),
    (re.compile(r"semi", re.IGNORECASE), "semi"),
    (re.compile(r"qualif", re.IGNORECASE), "qual"),   # single-qualifier formats (2023-24 elite, juniors)
    (re.compile(r"final", re.IGNORECASE), "final"),
]
CATEGORY_CODES = {"men elite": "ME", "women elite": "WE", "men junior": "MJ", "women junior": "WJ"}
SLUG_RE = re.compile(r"\b(\d{8}_[a-z]+)\b")
SERIES_ROUND_RE = re.compile(r"DHI\s*#\s*(\d+)")


def is_markdown_tables_file(path):
    text = Path(path).read_text(errors="ignore")
    return bool(re.search(r"^\|\s*Pos\s*\|", text, re.MULTILINE))


def _table_round_label(heading):
    for rx, label in TABLE_ROUND_MAP:
        if rx.search(heading):
            return label
    return re.sub(r"\W+", "_", heading.strip().lower())


def parse_markdown_tables_file(path, default_round=None):
    """Parse a data/*.md event file into tidy rows. Split cells look like
    '1:32.423 (+00.109)' (cumulative time, gap to best in parens); a final
    split equal to Time is the finish line and isn't repeated as a sector.
    rank_at_split is recomputed per round/sector from the cumulative times."""
    text = Path(path).read_text(errors="ignore")
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "")
    slug_m = SLUG_RE.search(text)
    slug = slug_m.group(1) if slug_m else Path(path).stem
    date = f"{slug[0:4]}-{slug[4:6]}-{slug[6:8]}" if slug_m else None
    sr_m = SERIES_ROUND_RE.search(title)
    series_round = int(sr_m.group(1)) if sr_m else None
    cat_m = re.search(r"^Category:\s*(.+)$", text, re.MULTILINE)
    cat_name = cat_m.group(1).strip() if cat_m else ""
    category = CATEGORY_CODES.get(cat_name.lower(), cat_name or None)
    # venue from the title's last " - " part: "... - Les Gets, August 21-23, FRA" -> "les-gets"
    venue = title.split(" - ")[-1].split(",")[0].strip() if " - " in title else Path(path).stem
    venue = re.sub(r"[\s_-]+", "-", venue.lower())
    # category in the id: elite and junior race the same event on the same day
    event_id = f"{slug}_DHI_{category}" if category else f"{slug}_DHI"

    rows_out = []
    round_ = None
    header = None
    for line in lines:
        if line.startswith("## "):
            round_ = default_round or _table_round_label(line[3:])
            header = None
            continue
        if not line.startswith("|") or round_ is None:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].lower() == "pos":
            header = [c.lower() for c in cells]
            continue
        if header is None or set(line.replace("|", "").strip()) <= {"-"}:
            continue
        cells += [""] * (len(header) - len(cells))
        rec = dict(zip(header, cells))
        rider_name = clean_rider_name(rec.get("rider", ""))
        if not rider_name:
            continue
        split_cols = [h for h in header if h.startswith("split")]
        finish_s, _ = parse_time_to_seconds(rec.get("time"))
        if not finish_s:  # "00.000" / blank
            finish_s = None
        raw_status = rec.get("status", "").strip().lower()
        if raw_status == "na" and finish_s is None:
            status = "START"  # on the start list, not raced yet (round in progress / upcoming)
        else:
            status = STATUS_TOKENS.get(raw_status) or ("OK" if finish_s is not None else "DNF")
        pos = int(rec["pos"]) if rec.get("pos", "").isdigit() else None
        common = dict(
            event_id=event_id, event_date=date, discipline="DHI",
            category=category, round=round_,
            rider_id=normalize_rider_id(rider_name), rider_name=rider_name,
            team=rec.get("team") or None, bib=rec.get("bib") or None,
            nation=rec.get("nation") or None, start_order=None,
            venue=venue, series_round=series_round, track_condition="unknown",
            event_name=title, source_key=slug,
        )
        if "uci id" in header:          # files downloaded since the UCI ID column (older ones don't have it)
            common["uci_id"] = rec.get("uci id") or None
        cum = 0.0
        sector_i = 0
        for sc in split_cols:
            raw = re.sub(r"\s*\(.*\)\s*$", "", rec.get(sc, ""))
            secs, _ = parse_time_to_seconds(raw)
            if not secs:
                secs = None
            if secs is not None and finish_s is not None and abs(secs - finish_s) < 1e-6:
                break  # last split column is the finish line
            sector_i += 1
            rows_out.append(dict(
                common, sector_id=f"S{sector_i}",
                split_time_s=(secs - cum) if secs is not None and cum is not None else None,
                cum_time_s=secs, rank_at_split=None,
                status="OK" if secs is not None else status,
            ))
            cum = secs
        rows_out.append(dict(
            common, sector_id="FINISH",
            split_time_s=(finish_s - cum) if finish_s is not None and cum is not None else None,
            cum_time_s=finish_s, rank_at_split=pos, status=status,
        ))

    if rows_out:
        df = pd.DataFrame(rows_out)
        sec = df["sector_id"] != "FINISH"
        df.loc[sec, "rank_at_split"] = (
            df[sec].groupby(["event_id", "round", "sector_id"])["cum_time_s"]
            .rank(method="min")
        )
        rows_out = df.to_dict("records")
    return rows_out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", help="Print detected structure of a single file and exit.")
    ap.add_argument("--input-dir", help="Directory of raw .html/.csv/.json result files to parse.")
    ap.add_argument("--input-file", action="append", default=[], help="A single raw file (repeatable).")
    ap.add_argument("--out", default="splits.csv", help="Output tidy CSV path.")
    ap.add_argument("--round", help="Force a round label (qual/semi/final/...) for all input files.")
    ap.add_argument(
        "--conditions-file",
        help="Optional CSV with columns event_id,round,track_condition to backfill "
             "track_condition, since it's rarely in the timing data itself.",
    )
    ap.add_argument("--canonical-venues", action="store_true",
                    help="Write venues under their canonical slugs (sports/mtb_dh.toml [venue_aliases], as the "
                         "database stores them: vallnord -> pal-arinsal). Default: as spelled in the source.")
    args = ap.parse_args()

    if args.inspect:
        inspect_file(args.inspect)
        return

    files = list(args.input_file)
    if args.input_dir:
        files += [str(p) for p in Path(args.input_dir).glob("*") if p.suffix.lower() in (".html", ".htm", ".csv", ".json", ".md", ".txt")]
    if not files:
        print("No input files given. Use --inspect FILE first, or --input-dir / --input-file.", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for f in files:
        suffix = Path(f).suffix.lower()
        print(f"Parsing {f} ...")
        if suffix in (".html", ".htm"):
            all_rows += parse_html_file(f, default_round=args.round)
        elif suffix == ".csv":
            all_rows += parse_csv_file(f, default_round=args.round)
        elif suffix == ".json":
            all_rows += parse_json_file(f, default_round=args.round)
        elif suffix in (".md", ".txt") and is_markdown_tables_file(f):
            all_rows += parse_markdown_tables_file(f, default_round=args.round)
        elif suffix in (".md", ".txt"):
            all_rows += parse_text_file(f, default_round=args.round)

    out_df = pd.DataFrame(all_rows)

    if args.conditions_file:
        cond = pd.read_csv(args.conditions_file)
        out_df = out_df.drop(columns=["track_condition"]).merge(
            cond, on=[c for c in ("event_id", "round") if c in cond.columns], how="left"
        )
        out_df["track_condition"] = out_df["track_condition"].fillna("unknown")

    if args.canonical_venues and "venue" in out_df:
        from racinglines.db.registry import canonical_venue
        out_df["venue"] = out_df["venue"].map(lambda v: canonical_venue(v) if isinstance(v, str) else v)

    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")
    n_unparsed = out_df["cum_time_s"].isna().sum() if "cum_time_s" in out_df else 0
    if n_unparsed:
        print(f"Note: {n_unparsed} rows have no parseable time (check status/format for these).")


if __name__ == "__main__":
    main()
