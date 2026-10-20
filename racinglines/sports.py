"""
Sport schemas: what differs between sports, as data (sports/<code>.toml).

Existing modules keep their own constant names (registry.COMPETITIONS,
inputs.laps.SESSION_MINUTES, run_times.model.FINAL_POINTS, ...) and take
the values from here, so a new sport starts with a new schema file.

    load("f1")                  the parsed schema
    SPORT_CODES                 every sport with a schema, in display order
    by_competition("f1_wdc")    the schema whose competition has that code
    modeled(code)               False for a tape-only sport ([sport] model_family = "none": no model, no
                                pipeline; its exchange markets are only recorded, docs/todo.md U9)
    kalshi_series(code)         the Kalshi series prefixes a sport's [markets.kalshi] names, () if none
    kalshi_kinds(code)          {series ticker: kind} from [markets.kalshi] kinds, {} if none
    titles_classified(code)     [markets] titles: the syncs classify from titles with the F1 resolver
    link_kinds(code)            [markets] kinds: the kinds an identified link of the sport is filed as
    polymarket_tags(code)       the Gamma tag_slug values a sport's [markets.polymarket] names, () if none
    identity(code)              the package under racinglines/sources/ whose links module says which driver and race a
                                market is about ([identity] resolver in the schema), None if the sport has none
    frames(code)                the L1 input frames a sport's [data] frames names, () if the block is absent
                                (racinglines/frames, docs/frames.md)
    audience(code)              who the web app shows the sport to ([sport] audience): "all" (the default) or "admin"
    admin_only()                the sport codes whose audience is "admin"; admin_only_competitions() their competitions
    off_board_competitions()    the competitions left off the Markets page for everyone ([sport] markets_board = false)
    is_public(code, user)       the one gate (web pages and MCP tools): may this user see the sport or competition code?
"""

import tomllib
from functools import cache

from racinglines.paths import ROOT

SCHEMAS = ROOT / "sports"
SPORT_CODES = tuple(c for c in ("mtb_dh", "f1", "nascar", "motogp", "indycar", "road_cycling", "le_mans", "sailgp")
                    if (SCHEMAS / f"{c}.toml").exists())   # registry / seeding order (keeps database ids stable); the
                                                          # public release ships f1 and mtb_dh (oss/manifest.toml)


@cache
def load(code):
    return tomllib.loads((SCHEMAS / f"{code}.toml").read_text())


def by_competition(competition):
    return next(s for s in map(load, SPORT_CODES) if s["competition"]["code"] == competition)


def event_title(competition, event_key, default):
    """An event's one display name: its schema's [names] entry for the key, else `default`."""
    try:
        names = by_competition(competition).get("names", {})
    except StopIteration:
        return default
    return names.get(str(event_key), default)


_REQUIRED = object()


@cache
def declaring(model_input, default=_REQUIRED):
    """The one sport whose [model] inputs declares `model_input`. An input layer that reads its constants from a
    schema at import (inputs/laps.py, inputs/grid.py, inputs/run_times/model.py: points, sessions, regulation
    resets, formats) finds its schema here instead of naming a sport. ValueError when no sport or several do: a second
    sport with that input needs the layer's constants bound per sport first (decision log 2026-10-11 "One engine").
    `default` is returned instead when no sport does (an install without that sport's schema: the public release)."""
    codes = [c for c in SPORT_CODES if model_input in (load(c).get("model", {}).get("inputs") or ())]
    if not codes and default is not _REQUIRED:
        return default
    if len(codes) != 1:
        raise ValueError(f"[model] inputs: {len(codes)} sports declare {model_input!r} ({codes}); its layer reads "
                         "one sport's schema at import, so exactly one may until it is bound per sport")
    return codes[0]


AUDIENCES = ("all", "admin")


def audience(code):
    """Who the web app shows the sport to ([sport] audience): "all" (the default) or "admin" (owner, 2026-10-11, MotoGP
    for the Alpha: a venue's or data feed's terms keep its results and prices off user pages). Only what users see
    changes: the sport is still recorded, priced and settled, the CLI and admins see all of it, and nothing is deleted.
    Flipping the key back to "all" shows it again."""
    got = load(code)["sport"].get("audience", "all")
    if got not in AUDIENCES:
        raise ValueError(f"sports/{code}.toml [sport] audience: {got!r} not in {AUDIENCES}")
    return got


def is_public(code, user=None):
    """The one gate every user-facing reader calls (web pages, MCP tools): whether `user` may see a sport or competition
    code. True for an admin (user["role"] == "admin"), for None (an internal caller: the CLI, the stdio MCP server,
    pricing and recording) and for an empty or unknown code; else whether the sport's [sport] audience is "all". A
    signed-out web visitor is not None: the web app passes {} (web/audience.py)."""
    if user is None or not code or (isinstance(user, dict) and user.get("role") == "admin"):
        return True
    sport = code if code in SPORT_CODES else next((c for c in SPORT_CODES if load(c)["competition"]["code"] == code), None)
    return sport is None or audience(sport) == "all"


def admin_only():
    """The sport codes the web app shows to admins only ([sport] audience = "admin")."""
    return frozenset(c for c in SPORT_CODES if audience(c) == "admin")


def admin_only_competitions():
    """The competition codes of the admin-only sports."""
    return frozenset(load(c)["competition"]["code"] for c in admin_only())


def off_board_competitions():
    """The competition codes the Markets page leaves out for everyone, admins included ([sport] markets_board = false;
    owner, 2026-10-12, "Hide UCI DH for now"). Only the Markets page's sections and calendar: the sport's own pages,
    recording, pricing and settlement carry on. Deleting the key (default true) puts it back."""
    return frozenset(load(c)["competition"]["code"] for c in SPORT_CODES if not load(c)["sport"].get("markets_board", True))


def modeled(code):
    """Whether the sport has a model ([sport] model_family other than "none")."""
    return load(code)["sport"].get("model_family", "none") != "none"


CLASSIFIED_RULES = ("status_ok", "placed")


def classified_rule(code):
    """Who counts as classified when the sport's race markets settle ([results] classified, markets/payoffs.py):
    "status_ok" (the default: results status OK) or "placed" (any car with a position). None for an unknown sport."""
    try:
        rule = load(code).get("results", {}).get("classified", "status_ok")
    except FileNotFoundError:
        return None
    if rule not in CLASSIFIED_RULES:
        raise ValueError(f"sports/{code}.toml [results] classified: {rule!r} not in {CLASSIFIED_RULES}")
    return rule


def race_sport(conn, race_id):
    """The sport code of a race (its competition's schema), or None when no schema names the competition."""
    from sqlalchemy import text
    comp = conn.execute(text("""SELECT co.code FROM races ra JOIN events e ON e.id = ra.event_id
                                JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
                                WHERE ra.id = :r"""), dict(r=int(race_id))).scalar()
    for c in SPORT_CODES:
        try:
            if load(c)["competition"]["code"] == comp:
                return c
        except FileNotFoundError:
            continue
    return None


def kalshi_series(code):
    """The Kalshi series ticker prefixes a sport's schema names ([markets.kalshi] series), as a tuple."""
    return tuple(load(code).get("markets", {}).get("kalshi", {}).get("series", ()))


def probalytics(code):
    """A sport's Probalytics tape ([markets.probalytics]: `tag`, the sport's name in the tape's file names, and
    `book_stale_hours`), {} when it names none or there is no such sport."""
    try:
        return dict(load(code).get("markets", {}).get("probalytics", {}))
    except FileNotFoundError:
        return {}


@cache
def kalshi_kinds(code):
    """{Kalshi series ticker: prediction kind} a sport's schema names ([markets.kalshi] kinds), {} if none: the exact
    key a market's kind is read from before its titles."""
    return dict(load(code).get("markets", {}).get("kalshi", {}).get("kinds", {}))


@cache
def titles_classified(code):
    """Whether the syncs classify the sport's markets from the exchanges' titles and match them with the F1 resolver
    ([markets] titles = true); otherwise its [identity] resolver, when it has one, says what each market is about."""
    return bool(load(code).get("markets", {}).get("titles", False))


@cache
def link_kinds(code):
    """The kinds a sport's links are filed as (market_links.prediction) once identified ([markets] kinds), () if none:
    the kinds its pricing model prices. A link of another kind keeps params.kind and stays unmodeled."""
    return tuple(load(code).get("markets", {}).get("kinds", ()))


def polymarket_tags(code):
    """The Polymarket Gamma tag_slug values a sport's schema names ([markets.polymarket] tags), as a tuple."""
    return tuple(load(code).get("markets", {}).get("polymarket", {}).get("tags", ()))


def identity(code):
    """The name of the sport's market-identity resolver ([identity] resolver), or None."""
    return load(code).get("identity", {}).get("resolver")


def frames(code):
    """The L1 input frames a sport's schema says it provides ([data] frames), as a tuple; () if there is no block."""
    return tuple(load(code).get("data", {}).get("frames", ()))


def fallback_rule(code):
    """The source-quality rule a sport declares for missing or partial result data.

    Default is strict: official data is required; a fallback source is allowed only when the schema says so.
    """
    return load(code).get("results", {}).get("fallback_rule", "official_only")


def fallbacks(code):
    """Fallback sources declared in the sport schema, if any.

    Each entry is a generic dict: {source, quality, scope, notes, ...}. The schema may also define a
    `fallback_rule` describing the policy for when the fallback is acceptable.
    """
    entries = load(code).get("results", {}).get("fallbacks", ())
    if not entries:
        return ()
    if isinstance(entries, dict):
        entries = entries.get("sources", tuple(entries.values()))
    return tuple(entries)


def live_frames(code):
    """The frames whose source dates each row as it happens ([data] live), as a tuple; () if none. Every other
    session-keyed frame is published a whole session at a time, once the session has ended."""
    return tuple(load(code).get("data", {}).get("live", ()))


def int_keys(d):
    """TOML keys are strings; {"2021": x} -> {2021: x}."""
    return {int(k): v for k, v in d.items()}
