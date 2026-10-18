"""
The compact source note: where the numbers on a page, an MCP answer, a book render or a report come from, in one line
(owner, 2026-10-13: "compact source notes EVERYWHERE it is relevant ... for all events and even 2ndary products").

    results_sources("f1")     the sport's results/timing feeds ([sources] results in sports/<code>.toml), as dicts
                              {name, url}; a sport code or a competition code; () when unknown
    venue_names(["kalshi"])   the display names of the price venues shown (the venues registry and the exchange
                              schemas); only exchanges are named, a sportsbook's lines are "generic sportsbook lines"
    note(...)                 {results, prices, fair, parts, text}: one note, for a template (the `source_note` macro in
                              web/templates/_macros.html), a JSON answer (MCP `sources`) or a text footer (`text`)

Everything here is read from the schemas, so a new sport or exchange needs no change in this module.
"""

from racinglines import exchanges, sports

SPORTSBOOK = "generic sportsbook lines"     # never a sportsbook's brand
MODEL = "racinglines model"


def _sport_code(code):
    """A sport code from a sport code or a competition code, None when no schema names it."""
    if not code:
        return None
    if code in sports.SPORT_CODES:
        return code
    for c in sports.SPORT_CODES:
        if sports.load(c)["competition"]["code"] == code:
            return c
    return None


def results_sources(code):
    """The results/timing feeds of a sport ([sources] results), as a tuple of {name, url} dicts; () when unknown."""
    sport = _sport_code(code)
    if sport is None:
        return ()
    got = sports.load(sport).get("sources", {}).get("results", ())
    if isinstance(got, str):                     # results = "unknown"
        return () if got == "unknown" else ({"name": got, "url": ""},)
    return tuple({"name": r["name"], "url": r.get("url", "")} for r in got)


def _exchange_names():
    """{code: display name} of every exchange the app can show: Polymarket and Kalshi (markets/venues.py VENUES) and
    every exchange schema (exchanges/<code>.toml [exchange] name), whether or not its switch is on."""
    from racinglines.markets import venues as V
    names = {v.code: v.name for v in V.VENUES if v.kind == "exchange"}
    for code in exchanges.CODES:
        names.setdefault(code, exchanges.load(code)["exchange"]["name"])
    return names


def venue_names(venues):
    """The display names of the price venues in `venues` (codes, names, Venue objects or dicts with a code or a
    venue), in order, without repeats. Our own private book is not a third-party source and is left out; an unknown
    venue is never named."""
    if not venues:
        return []
    if isinstance(venues, str):
        venues = [venues]
    names = _exchange_names()
    by_name = {n.lower(): n for n in names.values()}
    out = []
    for v in venues:
        if isinstance(v, dict):
            v = v.get("venue") or v.get("exchange") or v.get("code") or v.get("exchange_name") or v.get("name")
        v = getattr(v, "venue", v)
        v = getattr(v, "code", v)
        if not isinstance(v, str):
            continue
        name = SPORTSBOOK if v.lower() in ("sportsbook", "book", SPORTSBOOK) else names.get(v) or by_name.get(v.lower())
        if name and name not in out:
            out.append(name)
    return out


def _as_of(at):
    """A time as "YYYY-MM-DD HH:MM UTC" (naive times are taken as UTC); a string that isn't a time is kept as is."""
    if at is None or at == "":
        return None
    try:
        import pandas as pd
        if not isinstance(at, str) and pd.isna(at):
            return None
        ts = pd.Timestamp(at)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(at)


def fair_label(run=None, as_of=None):
    """"racinglines model (run #N, as of ...)": `run` is a run id or a run (dict or object with id and created_at)."""
    run_id = run
    if isinstance(run, dict):
        run_id, as_of = run.get("id"), as_of if as_of is not None else run.get("created_at")
    elif run is not None and not isinstance(run, (int, str)):
        run_id = getattr(run, "id", None)
        as_of = as_of if as_of is not None else getattr(run, "created_at", None)
    bits = []
    if run_id is not None and str(run_id) != "":
        bits.append(f"run #{run_id}")
    when = _as_of(as_of)
    if when:
        bits.append(f"as of {when}")
    return MODEL + (f" ({', '.join(bits)})" if bits else "")


def note(sport=None, venues=None, run=None, as_of=None, model=False, sportsbook=False):
    """One compact source note. `sport`: a sport or competition code, or several; `venues`: the price venues shown;
    `run` / `as_of`: the model run the fair values come from (or `model=True` when fair values are shown without a
    known run); `sportsbook`: generic sportsbook lines are shown too.

    Returns {results: [{name, url}], prices: [names], fair: str | None, parts: [{label, entries: [{name, url}]}],
    text: "Sources: results: ... · prices: ... · fair values: ..."} (text "" when there is nothing to name)."""
    codes = [sport] if isinstance(sport, str) or sport is None else list(sport)
    results = []
    for c in codes:
        for r in results_sources(c):
            if r not in results:
                results.append(r)
    prices = venue_names(venues)
    if sportsbook and SPORTSBOOK not in prices:
        prices.append(SPORTSBOOK)
    fair = fair_label(run, as_of) if (run is not None or model) else None
    parts = []
    if results:
        parts.append({"label": "results", "entries": results})
    if prices:
        parts.append({"label": "prices", "entries": [{"name": p, "url": ""} for p in prices]})
    if fair:
        parts.append({"label": "fair values", "entries": [{"name": fair, "url": ""}]})
    text = ("Sources: " + " · ".join(f"{p['label']}: " + ", ".join(i["name"] for i in p["entries"]) for p in parts)
            if parts else "")
    return {"results": results, "prices": prices, "fair": fair, "parts": parts, "text": text}
