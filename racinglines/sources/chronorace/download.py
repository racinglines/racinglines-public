#!/usr/bin/env python3
"""
racinglines/sources/chronorace/download.py — Download UCI Mountain Bike World Cup result data
(splits, times, gaps, status) from ChronoRace, for a given year, discipline,
and category, as one Markdown file per event.

HOW EVENT DISCOVERY WORKS (read this before trusting --year blindly)
----------------------------------------------------------------------
ChronoRace itself does NOT expose a "list all events for year X" endpoint —
every reasonable guess (/api/results/uci/dh/cms, /api/results/uci/dh/events,
/api/results/uci/events?year=..., etc.) 404s. There is no year-level index.

What this script does instead is exactly what a human would do: it reads the
Wikipedia page "{year} UCI Mountain Bike World Cup" (e.g.
https://en.wikipedia.org/wiki/2026_UCI_Mountain_Bike_World_Cup), which cites
a ChronoRace result PDF for every round of every discipline. Those citation
URLs look like:

    https://chronorace.blob.core.windows.net/webresources/20260821_mtb/gets_dhi_me_results_f.pdf

The first path segment (20260821_mtb) is ChronoRace's internal event slug,
and it's this slug (not the raw calendar date) that every other API call
needs. This script regexes those slugs out of the Wikipedia page's citation
list, deduplicates them, and uses that as its event list — filtered to the
discipline you asked for (DHI/XCO/XCC/EDR all use different slug suffixes
sometimes, e.g. "_edr" for standalone Enduro weekends vs "_mtb" for combined
XCO+XCC+DHI weekends).

If Wikipedia's page doesn't exist yet for a year (future year, or before the
page is created), is incomplete, or you'd rather skip the Wikipedia lookup entirely,
pass --events explicitly with known ChronoRace slugs (see --list-only to
preview what would be fetched without downloading anything).

WHAT'S CONFIRMED VS. UNVERIFIED
----------------------------------------------------------------------
- Downhill (DHI): fully verified end-to-end (CMS tree walk -> live-timing
  keys -> generic results JSON -> per-split markdown tables), because DHI is
  individual/time-trial format with real splits.
- XCO / XCC / EDR: the CMS tree walk works the same way (they're all under
  the same event JSON), but the generic per-split results endpoint
  (`/api/results/generic/uci/{slug}/dh?key=...`) was only confirmed against
  DHI. XCO/XCC are mass-start formats and may not expose the same
  split/gap structure, or may 404 on this endpoint entirely. If that
  happens for a round, this script automatically falls back to just listing
  that round's PDF links (Start List / Results / Standings) instead of a
  splits table, and prints a note — it will not silently give you an empty
  file.

NETWORK NOTE
----------------------------------------------------------------------
Run this from your own machine's normal Python/terminal — it needs to reach
prod.chronorace.be and en.wikipedia.org directly over HTTPS.

USAGE
----------------------------------------------------------------------
    pip install requests

    # Discover events for a year/discipline without downloading anything:
    racinglines data download mtb_dh --year 2026 --discipline DH --category "Elite Men" --list-only

    # Download everything found:
    racinglines data download mtb_dh --year 2026 --discipline DH --category "Elite Men" --out-dir results/

    # Skip the Wikipedia lookup and give exact ChronoRace slugs yourself:
    racinglines data download mtb_dh --events 20260821_mtb 20260611_mtb \\
        --discipline DH --category "Elite Men" --out-dir results/

Categories accepted (case/spacing-insensitive): "Elite Men", "Elite Women",
"Junior Men", "Junior Women", "U23 Men", "U23 Women" (U23 applies to
XCO/XCC, junior applies to DHI — ChronoRace uses whichever exists per
discipline and will just come up empty if you ask for a combo that doesn't
exist for that discipline/event).
"""

import argparse
import re
from pathlib import Path

import requests

from racinglines.sources import http

USER_AGENT = "racinglines (+https://github.com/racinglines/racinglines-public)"   # says who we are (owner, 26 Oct)
CMS_URL = "https://prod.chronorace.be/api/results/uci/dh/cms/{slug}"
GENERIC_RESULTS_URL = "https://prod.chronorace.be/api/results/generic/uci/{slug}/{disc_api}?key={key}"

# Friendly category name -> ChronoRace category code (the "Name" field on
# the CG node in the CMS tree, e.g. {"Id":"CG1","Name":"ME","DisplayName":"Men Elite"})
CATEGORY_CODES = {
    "elite men": "ME",
    "elite women": "WE",
    "junior men": "MJ",
    "junior women": "WJ",
    "u23 men": "MU",
    "u23 women": "WU",
    "men elite": "ME",
    "women elite": "WE",
    "men junior": "MJ",
    "women junior": "WJ",
    "men u23": "MU",
    "women u23": "WU",
}

# Friendly/short discipline name -> (CMS discipline Id, generic-results API segment)
# The generic-results segment is only confirmed for DH; others are best guesses
# based on the site's naming convention and will fail gracefully if wrong.
DISCIPLINE_CODES = {
    "dh": ("DHI", "dh"),
    "dhi": ("DHI", "dh"),
    "downhill": ("DHI", "dh"),
    "xco": ("XCO", "xco"),
    "xcc": ("XCC", "xcc"),
    "edr": ("EDR", "edr"),
    "enduro": ("EDR", "edr"),
}

ROUND_ORDER = [
    "Timed Training", "Practice", "Seeding",
    "Qualification", "Qualification 1", "Qualification 2",
    "Semi Final", "Final",
]


def normalize_key(s):
    return re.sub(r"[\s_-]+", " ", s.strip().lower())


def resolve_category(name):
    key = normalize_key(name)
    if key not in CATEGORY_CODES:
        raise SystemExit(
            f"Unrecognized category '{name}'. Known: {sorted(set(CATEGORY_CODES))}"
        )
    return CATEGORY_CODES[key]


def resolve_discipline(name):
    key = normalize_key(name)
    if key not in DISCIPLINE_CODES:
        raise SystemExit(
            f"Unrecognized discipline '{name}'. Known: {sorted(set(DISCIPLINE_CODES))}"
        )
    return DISCIPLINE_CODES[key]


def discover_event_slugs(year, discipline_id, session):
    """Read Wikipedia's '{year} UCI Mountain Bike World Cup' page for
    ChronoRace result-PDF citations and pull out the event slugs relevant to
    the requested discipline. This is the ONLY known way to get a year-level
    event list -- see the module docstring."""
    wiki_url = f"https://en.wikipedia.org/wiki/{year}_UCI_Mountain_Bike_World_Cup"
    resp = http.get(session, wiki_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    if resp.status_code != 200:
        raise SystemExit(
            f"Could not fetch {wiki_url} (status {resp.status_code}). "
            f"Pass --events with explicit ChronoRace slugs instead."
        )
    html = resp.text

    # Citation URLs look like:
    # https://chronorace.blob.core.windows.net/webresources/20260821_mtb/gets_dhi_me_results_f.pdf
    # Capture the slug and the discipline code embedded in the filename.
    pattern = re.compile(
        r"chronorace\.blob\.core\.windows\.net/webresources/([0-9]{8}_[a-z]+)/[a-z]+_([a-z]+)_",
        re.IGNORECASE,
    )
    slugs_by_discipline = {}
    for slug, disc_code in pattern.findall(html):
        slugs_by_discipline.setdefault(disc_code.lower(), set()).add(slug)

    wanted = discipline_id.lower()  # e.g. "dhi", "xco", "xcc", "edr"
    # Season pages also cite neighbouring seasons (e.g. the 2026 page links
    # the 2025 finals); keep only slugs dated in the requested year.
    found = {s for s in slugs_by_discipline.get(wanted, set()) if s.startswith(str(year))}
    if not found:
        available = {k: len(v) for k, v in slugs_by_discipline.items()}
        raise SystemExit(
            f"No '{discipline_id}' result citations found on {wiki_url}.\n"
            f"Disciplines that WERE found there: {available}\n"
            f"The page may not exist yet for {year}, may use different naming, "
            f"or that discipline may not have run yet. Pass --events explicitly instead."
        )
    return sorted(found)


PROBE_SUFFIXES = ("dh", "dhi", "mtb", "xco")


def probe_event_slugs(start, end, discipline_id, session, suffixes=PROBE_SUFFIXES, echo=print):
    """Find events missing from Wikipedia (e.g. 2021 Snowshoe) by trying `YYYYMMDD_<suffix>` for every
    date from `start` to `end` (datetime.date, inclusive) against the content-tree endpoint, which
    answers a JSON null for slugs that don't exist. Keeps slugs whose tree has `discipline_id`."""
    from datetime import timedelta
    found = []
    day = start
    while day <= end:
        for suffix in suffixes:
            slug = f"{day:%Y%m%d}_{suffix}"
            resp = http.get(session, CMS_URL.format(slug=slug), headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except ValueError:
                continue
            if isinstance(data, dict) and any(c.get("Id") == discipline_id for c in data.get("Childs") or []):
                echo(f"  probe: {slug} ({data.get('DisplayName', '')})")
                found.append(slug)
        day += timedelta(days=1)
    return found


def fetch_pdf_round(pdf_links, round_name, session):
    """A round's results table from its result PDF (no live timing), or None."""
    from racinglines.sources.chronorace import pdf
    link = next((u for label, u in pdf_links.items() if "result" in (label or "").lower()
                 and "team" not in (label or "").lower()), None)
    if link is None:
        return None
    try:
        resp = http.get(session, link, headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
            print(f"    [warn] result PDF for round '{round_name}' not available ({resp.status_code}): {link}")
            return None
        return pdf.round_table(pdf.pdf_text(resp.content), round_name)
    except Exception as e:
        print(f"    [warn] result PDF for round '{round_name}' failed: {e}")
        return None


def fetch_cms_tree(slug, session):
    url = CMS_URL.format(slug=slug)
    resp = http.get(session, url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    flat = []

    def flatten(node):
        flat.append(node)
        for child in node.get("Childs") or []:
            flatten(child)

    flatten(data)
    return data.get("DisplayName", slug), flat


def get_category_rounds(flat, discipline_id, category_code):
    """Walk the flattened CMS tree to find every round under
    {discipline_id}/{category_code} that has a 'Live Timing' child with a
    Route key, plus that round's static PDF links as a fallback/extra."""
    cat_node = next(
        (n for n in flat if n.get("ParentId") == discipline_id and n.get("Name") == category_code),
        None,
    )
    if cat_node is None:
        return None, []

    rounds = [n for n in flat if n.get("ParentId") == cat_node["Id"] and n.get("Type") == "Folder"]
    out = []
    for r in rounds:
        children = [n for n in flat if n.get("ParentId") == r["Id"]]
        live = next((c for c in children if c.get("Name") == "Live Timing" and c.get("Route")), None)
        key = None
        if live:
            try:
                import json as _json
                key = _json.loads(live["Route"])["Params"]["key"]
            except Exception:
                key = None
        pdf_links = {
            c.get("DisplayName") or c.get("Name"): c.get("Link")
            for c in children
            if c.get("Link")  # "File" nodes; 2019-20 events type them as "Folder"
        }
        out.append({
            "round": r.get("DisplayName") or r.get("Name"),
            "key": key,
            "pdf_links": pdf_links,
        })

    def sort_key(item):
        try:
            return ROUND_ORDER.index(item["round"])
        except ValueError:
            return len(ROUND_ORDER)

    out.sort(key=sort_key)
    return cat_node.get("DisplayName", category_code), out


def fmt_time(ms):
    if ms is None:
        return ""
    neg = ms < 0
    ms = abs(ms)
    total_sec = ms / 1000.0
    m = int(total_sec // 60)
    s = total_sec - m * 60
    s_str = f"{s:06.3f}"
    out = f"{m}:{s_str}" if m > 0 else s_str
    return f"-{out}" if neg else out


def fmt_gap(ms):
    if ms is None or ms == 0:
        return ""
    sign = "+" if ms > 0 else "-"
    return f"{sign}{fmt_time(abs(ms))}"


def build_round_table(slug, disc_api, key, round_name, session):
    """Fetch one round's split/result JSON and render a markdown table.
    Returns None if the endpoint doesn't return usable Results (so the
    caller can fall back to PDF links)."""
    url = GENERIC_RESULTS_URL.format(slug=slug, disc_api=disc_api, key=key)
    try:
        resp = http.get(session, url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [warn] generic results fetch failed for round '{round_name}' "
              f"(key={key}): {e}")
        return None

    if not isinstance(data, dict):  # endpoint can return a bare JSON null
        return None
    riders = data.get("Riders") or {}
    results = data.get("Results") or []
    if not results:
        return None

    results = sorted(results, key=lambda r: r.get("Position") or 9999)
    n_splits = max((len(r.get("Times") or []) for r in results), default=0)

    header = "| Pos | Bib | Rider | Team | Nation | UCI ID |"
    header += "".join(f" Split {i} |" for i in range(1, n_splits + 1))
    header += " Time | Gap | Status |"
    sep = "|---|---|---|---|---|---|" + "---|" * n_splits + "---|---|---|"
    rows = [header, sep]

    for r in results:
        rider = riders.get(str(r.get("RaceNr")), {})
        bib = rider.get("RaceNr", r.get("RaceNr"))
        name = rider.get("PrintName", "")
        team = rider.get("UciTeamName", "") or ""
        nation = rider.get("Nation", "") or ""
        uci_id = rider.get("UciRiderId", "") or ""
        pos = r.get("Position") or ""
        times = r.get("Times") or []
        split_cells = ""
        for i in range(n_splits):
            t = times[i] if i < len(times) else None
            if t:
                cell = fmt_time(t.get("RaceTime"))
                gap = t.get("TimeGap")
                if gap:
                    cell += f" ({fmt_gap(gap)})"
                split_cells += f" {cell} |"
            else:
                split_cells += " |"
        status = r.get("Status")
        finished = status == "Finished"
        time_str = fmt_time(r.get("RaceTime")) if finished else ""
        last_time = times[-1] if times else None
        gap_str = fmt_gap(last_time.get("TimeGap")) if (finished and last_time) else ""
        status_str = status if (status and not finished) else ""
        rows.append(
            f"| {pos} | {bib} | {name} | {team} | {nation} | {uci_id} |{split_cells} "
            f"{time_str} | {gap_str} | {status_str} |"
        )

    return f"## {round_name}\n\n" + "\n".join(rows) + "\n"


def build_event_doc(slug, discipline_id, disc_api, category_code, session, pdf_results=False):
    title, flat = fetch_cms_tree(slug, session)
    cat_display, rounds = get_category_rounds(flat, discipline_id, category_code)
    if cat_display is None:
        print(f"    [skip] no '{category_code}' category found under "
              f"'{discipline_id}' for event {slug}")
        return None

    doc = (
        f"# {title}\n\n"
        f"Category: {cat_display}\n\n"
        f"Source: ChronoRace (prod.chronorace.be), event slug `{slug}`\n\n"
    )
    any_data = False
    for r in rounds:
        table = None
        if r["key"]:
            table = build_round_table(slug, disc_api, r["key"], r["round"], session)
        if table is None and pdf_results and r["pdf_links"]:
            table = fetch_pdf_round(r["pdf_links"], r["round"], session)
            if table:
                table += "\n" + "".join(f"- [{label}]({link})\n" for label, link in r["pdf_links"].items())
        if table:
            doc += table + "\n"
            any_data = True
        elif r["pdf_links"]:
            doc += f"## {r['round']}\n\n(No live split data available for this round; static results below.)\n\n"
            for label, link in r["pdf_links"].items():
                doc += f"- [{label}]({link})\n"
            doc += "\n"
            any_data = True
        # else: round has neither a live-timing key nor pdf links (e.g. not yet run) -> skip silently

    if not any_data:
        print(f"    [skip] no data at all for {slug} / {category_code} "
              f"(event may not have happened yet)")
        return None
    return doc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--year", type=int, help="Season year, e.g. 2026 (used to read the Wikipedia calendar for event slugs).")
    ap.add_argument("--events", nargs="+", help="Explicit ChronoRace event slugs (e.g. 20260821_mtb), skips the Wikipedia lookup.")
    ap.add_argument("--discipline", required=True, help="DH, XCO, XCC, or EDR.")
    ap.add_argument("--category", required=True, help="'Elite Men', 'Elite Women', 'Junior Men', 'Junior Women', 'U23 Men', 'U23 Women'.")
    ap.add_argument("--out-dir", default=None, help="Directory to write one .md file per event into "
                    "(default data/raw/mtb_dh/chronorace).")
    ap.add_argument("--probe", nargs=2, metavar=("START", "END"),
                    help="Also find events by trying YYYYMMDD_{dh,dhi,mtb,xco} for every date from START to END "
                         "(YYYY-MM-DD) against ChronoRace, for rounds missing from Wikipedia (e.g. 2021 Snowshoe). "
                         "About 2 s per day at ChronoRace's polite pace.")
    ap.add_argument("--probe-suffixes", nargs="+", default=list(PROBE_SUFFIXES), metavar="SUFFIX",
                    help=f"Slug suffixes --probe tries (default: {' '.join(PROBE_SUFFIXES)}).")
    ap.add_argument("--pdf-results", action="store_true",
                    help="For rounds without live timing, read the result PDF into the round's table "
                         "(2021 Leogang and Les Gets, some Timed Training rounds). Needs poppler's pdftotext. "
                         "Default: list the PDF links only.")
    ap.add_argument("--list-only", action="store_true", help="Only print the discovered event slugs; download nothing.")
    args = ap.parse_args()

    if not args.year and not args.events and not args.probe:
        raise SystemExit("Pass --year (to auto-discover events), --events (explicit slugs) or --probe START END.")

    discipline_id, disc_api = resolve_discipline(args.discipline)
    category_code = resolve_category(args.category)

    session = requests.Session()

    if args.events:
        slugs = list(args.events)
    elif args.year:
        print(f"Discovering {discipline_id} events for {args.year} from Wikipedia...")
        slugs = discover_event_slugs(args.year, discipline_id, session)
    else:
        slugs = []
    if args.probe:
        from datetime import date
        start, end = (date.fromisoformat(d) for d in args.probe)
        print(f"Probing {discipline_id} slugs from {start} to {end}...")
        slugs = sorted(set(slugs) | set(probe_event_slugs(start, end, discipline_id, session,
                                                           suffixes=args.probe_suffixes)))

    print(f"Found {len(slugs)} event(s): {slugs}")
    if args.list_only:
        return

    from racinglines import paths
    out_dir = Path(args.out_dir) if args.out_dir else paths.DH_RAW
    out_dir.mkdir(parents=True, exist_ok=True)

    for slug in slugs:
        print(f"  Fetching {slug} ...")
        try:
            doc = build_event_doc(slug, discipline_id, disc_api, category_code, session,
                                  pdf_results=args.pdf_results)
        except requests.HTTPError as e:
            print(f"    [error] {e}")
            continue
        if doc is None:
            continue
        cat_slug = normalize_key(args.category).replace(" ", "-")
        disc_slug = discipline_id.lower()
        out_path = out_dir / f"{slug}_{disc_slug}_{cat_slug}.md"
        out_path.write_text(doc)
        print(f"    -> {out_path} ({len(doc)} chars)")

    print("Done.")


if __name__ == "__main__":
    main()
