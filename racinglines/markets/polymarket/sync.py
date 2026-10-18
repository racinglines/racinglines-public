"""
Sync Polymarket's F1 markets into market_links and map each outcome to the
model prediction that prices it.

    "<GP>: Driver Winner"                 race_win          athlete, race
    "<GP>: Driver Podium Finish"          race_podium       athlete, race
    "<GP>: Head-to-Head"                  race_h2h          athlete, race, params.opponent_id (both tokens)
    "<GP>: Which Constructor Scores 1st?" race_constructor_top   race, params.team
    "<GP>: Driver Pole Position"          race_pole         athlete, race
    "<GP>: Driver Fastest Lap"            race_fastest_lap  athlete, race
    "<GP>: Sprint Winner"                 race_sprint_win   athlete, race (the sprint stage, models/inputs/laps_pricing.py)
        (2025: "F1 <GP>: Sprint Race Winner", "<GP> - Sprint Winner")
    "<GP>: Sprint Qualifying Pole Winner" race_sprint_pole  athlete, race
    "<GP>: 2nd Place" .. "5th Place"      race_p2 .. race_p5   athlete, race (an exact finishing place)
    "<GP>: Which Constructor scores 2nd?" .. "5th?", "... the second most points?"
                                          race_constructor_p2 .. p5   race, params.team (the team's points rank)
    "<GP>: Constructor Pole Position"     race_constructor_pole        race, params.team
    "<GP>: Constructor Fastest Lap"       race_constructor_fastest_lap race, params.team
    "<GP> – Constructor Matchups" / "McLaren vs. Red Bull"
                                          race_constructor_h2h  race, params.team and params.opponent (both tokens)
    "F1: Constructor to double podium at <GP>?"  race_team_double_podium  race, params.team
    "F1: Will A finish ahead of B in the <GP>?"  race_h2h  athlete, race, params.opponent_id (the Yes token)
    "Will there be a safety car / red flag during the ... <GP>?", "Rain during the <GP>?"
                                          race_safety_car / race_red_flag / race_rain   race (no athlete)
    "F1 Drivers' Champion"                champion          athlete
    "F1 Drivers Champion: 2nd Place" .. "5th Place"   standings_p2 .. p5   athlete
    "F1 Constructors' Champion"           constructors_champion  params.team
    "F1 Constructors Champion: 2nd Place" .. "5th Place"   constructors_p2 .. p5   params.team
    "Will X win N+ Grands Prix in YYYY?"  season_wins_ge    athlete, params.n
    "Will A finish ahead of B in the YYYY Drivers' Championship?"  standings_h2h  athlete, params.opponent_id
    anything else                         unmodeled (listed with prices, no model price); linked to the race
                                          when the title starts "<GP>: " (practice fastest lap, ...)

Who each token names follows the kind's subject in markets/kinds.toml (a driver: the group title; a team: the group
title as a team; a pair: the two outcomes, or the two names of a yes/no question), so a new kind with a known subject
needs a title pattern here and nothing else. A race is matched by Resolver.race, the same name-and-date lookup for
every family: a market on a race the calendar no longer has (the April 2026 Bahrain GP, cancelled; it ran at Sepang
in October) keeps no race_id and stays unmodeled.

Placeholders: Polymarket lists unnamed slots in some race events ("Driver A" to "Driver E", "Other" for "any
other driver / constructor") so it can add outcomes later. They stay unmodeled, linked to the race, with
params.placeholder = the kind they stand in for (race_win, race_pole, ...); the race page counts them.

Prices come from the Gamma API (bestBid / bestAsk / lastTradePrice /
outcomePrices, all for the market's first outcome; the second outcome's book
is the mirror image). Closed markets record which outcome resolved.
Idempotent: links are keyed by token id and updated in place.

The active listing (active=true, closed=false) drops an event once it closes, so a decided market (pole after
qualifying) would sit open in market_links with its last quote. Each pass therefore re-reads by slug the events
whose links are still open here but that the listing no longer returns (up to REREAD_MAX, most recently synced
first) and upserts them like the rest: closed, resolved and last prices come from Gamma's own market rows.

Other sports (docs/coverage.md, item 3): NASCAR Cup, MotoGP and IndyCar are tape-only sports (sports/<code>.toml,
[markets.polymarket] tags = the Gamma tag_slug values to page, e.g. "nascar"; unverified against the live API).
`sync(..., sport="nascar")` upserts their markets under their own competition and first category, every link
`unmodeled` (no Resolver, no classifier, no driver or race), so trades, history and books record them like F1's;
`sync --tags SLUG ...` overrides the schema's tags. fetch_trades / fetch_history / snapshot_books take `sport=` to
cover every Polymarket link of that sport's competition when no events are named. Nothing about them runs unless
the sport is named, and the sync is additive: it upserts by token id and never deletes or resets a row.
"""

import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select, text

from racinglines import sports
from racinglines.db import models as m
from racinglines.markets import identity
from racinglines.markets import kinds as K
from racinglines.markets.kalshi.sync import gp_name
from racinglines.sources import http

GAMMA = "https://gamma-api.polymarket.com"
HISTORY_WINDOW = timedelta(days=7)    # /prices-history span per request (the weekend windows that always worked)
MATCH_DAYS = 10     # a race market's end date must be within this many days of the race
TAGS = ("f1", "formula1")
REREAD_MAX = 40     # events re-read by slug per pass when the active listing stopped returning them

# Polymarket labels some drivers oddly in head-to-heads (Carlos Sainz Jr. -> "Jr.")
DRIVER_ALIASES = {"jr": "carlos sainz", "sainz jr": "carlos sainz", "kimi antonelli": "andrea kimi antonelli"}

TEAM_NAMES = {
    "mercedes": "mercedes", "ferrari": "ferrari", "mclaren": "mclaren", "red bull racing": "red_bull",
    "red bull": "red_bull", "racing bulls": "rb", "rb": "rb", "visa cash app rb": "rb", "aston martin": "aston_martin",
    "alpine": "alpine", "williams": "williams", "haas": "haas", "kick sauber": "sauber", "sauber": "sauber",
    "audi": "sauber", "cadillac": "cadillac",
}


# Polymarket's Grand Prix names -> FastF1's (2025 titles use country names and old event names)
GP_ALIASES = {"brazilian": "sao paulo", "brazil": "sao paulo", "china": "chinese", "italy": "italian",
              "japan": "japanese", "mexican": "mexico city", "mexico": "mexico city", "australia": "australian",
              "us": "united states", "usa": "united states", "spain": "spanish", "austria": "austrian",
              "belgium": "belgian", "hungary": "hungarian", "netherlands": "dutch", "canada": "canadian",
              "imola": "emilia romagna", "saudi": "saudi arabian",
              # 2026 round 16: the Bahrain GP held at Sepang ("... Bahrain Grand Prix in Malaysia 2026")
              "malaysian": "malaysia", "sepang": "malaysia", "kuala lumpur": "malaysia",
              "bahrain in malaysia": "malaysia"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


# the yes/no questions as listed (2026-10-04); a count or timing market ("How many safety cars ...") stays unmodeled
PROPS = [(r"^Will there be a safety car during", "race_safety_car"), (r"^Will there be a red[- ]flag during", "race_red_flag"),
         (r"^(?:Will it )?rain during", "race_rain")]
PROP_KINDS = tuple(k for _, k in PROPS)        # yes/no for the whole race: no athlete
PLACEHOLDER = re.compile(r"^(?:Driver [A-Z]|Other)$")      # unnamed slots in a race event (group title)


ORDINAL = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3, "4th": 4, "fourth": 4, "5th": 5,
           "fifth": 5}
_ORD = r"(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth)"
# a place n -> its kind, per family (place 1 is the existing win / top constructor / champion kind)
PLACE_KINDS = {
    "race": {1: "race_win", 2: "race_p2", 3: "race_p3", 4: "race_p4", 5: "race_p5"},
    "constructor": {1: "race_constructor_top", 2: "race_constructor_p2", 3: "race_constructor_p3",
                    4: "race_constructor_p4", 5: "race_constructor_p5"},
    "drivers": {1: "champion", 2: "standings_p2", 3: "standings_p3", 4: "standings_p4", 5: "standings_p5"},
    "constructors": {1: "constructors_champion", 2: "constructors_p2", 3: "constructors_p3", 4: "constructors_p4",
                     5: "constructors_p5"},
}
AHEAD = re.compile(r"Will (.+?) finish ahead of (.+?) (?:in|at) the ")       # a yes/no head-to-head: (A, B)


def _place(family, word):
    return PLACE_KINDS[family].get(ORDINAL[word.lower()], "unmodeled")


def classify(event_title, question):
    t, q = (event_title or "").strip(), (question or "").strip()
    gp, sep = r"^(?:F1:?\s+)?(.+?Grand Prix)", r"\s*(?::|–|-)?\s*"
    if m_ := re.search(r"Drivers'? Champion(?:ship)?:\s*" + _ORD + r" Place$", t, re.I):
        return _place("drivers", m_.group(1)), None
    if m_ := re.search(r"Constructors'? Champion(?:ship)?:\s*" + _ORD + r" Place$", t, re.I):
        return _place("constructors", m_.group(1)), None
    if m_ := re.search(gp + sep + r"Sprint (?:Race )?Winner$", t, re.I):
        return "race_sprint_win", m_.group(1)
    if m_ := re.search(gp + sep + r"Sprint Qualifying Pole(?: Winner| Position)?$", t, re.I):
        return "race_sprint_pole", m_.group(1)
    if m_ := re.search(gp + sep + _ORD + r" Place$", t, re.I):
        return _place("race", m_.group(2)), m_.group(1)
    if m_ := re.search(gp + sep + r"Which Constructor scores (?:the )?" + _ORD + r"(?: most points| highest)?\??$",
                       t, re.I):
        return _place("constructor", m_.group(2)), m_.group(1)
    if m_ := re.search(gp + sep + r"Constructor Pole(?: Position| Winner)?$", t, re.I):
        return "race_constructor_pole", m_.group(1)
    if m_ := re.search(gp + sep + r"Constructor Fastest Lap$", t, re.I):
        return "race_constructor_fastest_lap", m_.group(1)
    if m_ := re.search(gp + sep + r"Constructor Match-?ups?$", t, re.I):
        return "race_constructor_h2h", m_.group(1)
    if m_ := re.search(r"Constructor to double podium (?:at|in) (?:the )?(.+?Grand Prix)\??$", t, re.I):
        return "race_team_double_podium", m_.group(1)
    if (m_ := AHEAD.search(q)) and not re.search(r"Championship", q) and (g := gp_name(q)):
        return "race_h2h", g
    if m_ := re.search(r"^(.*Grand Prix): Sprint Winner$", t):
        return "race_sprint_win", m_.group(1)
    if m_ := re.search(r"^(.*Grand Prix): Sprint Qualifying Pole Winner$", t):
        return "race_sprint_pole", m_.group(1)
    if m_ := re.search(r"^(.*Grand Prix): Driver Winner", t):
        return "race_win", m_.group(1)
    if m_ := re.search(r"^(.*Grand Prix): Driver Podium", t):
        return "race_podium", m_.group(1)
    if m_ := re.search(r"^(.*Grand Prix): Head-to-Head", t):
        return "race_h2h", m_.group(1)
    if m_ := re.search(r"^(.*Grand Prix): (?:Which Constructor Scores 1st|Highest Scoring Constructor)", t):
        return "race_constructor_top", m_.group(1)
    if m_ := re.search(r"^(.*Grand Prix): Driver Pole Position", t):
        return "race_pole", m_.group(1)
    # 2025 titles ("F1 Austrian Grand Prix Winner", "China Grand Prix: Pole Winner", "F1 Dutch Grand Prix –
    # Head to Head Matchups", "...: Which Constructor scores the most points?"); checked after the 2026 forms
    gp, sep = r"^(?:F1:?\s+)?(.+?Grand Prix)", r"\s*(?::|–|-)?\s*"
    if m_ := re.search(gp + sep + r"(?:Driver\s+)?Winner$", t):
        return "race_win", m_.group(1)
    if m_ := re.search(gp + sep + r"(?:Driver\s+)?Pole\s+(?:Winner|Position)$", t):
        return "race_pole", m_.group(1)
    if m_ := re.search(gp + sep + r"Driver\s+Podium(?:\s+Finish)?$", t):
        return "race_podium", m_.group(1)
    if m_ := re.search(gp + sep + r"Head[- ]to[- ]Head(?:\s+Matchups)?$", t, re.I):
        return "race_h2h", m_.group(1)
    if m_ := re.search(gp + sep + r"(?:Which Constructor Scores (?:the most points|1st)\??|(?:Top|Highest) Scoring Constructor)$",
                       t, re.I):
        return "race_constructor_top", m_.group(1)
    if re.search(r"Drivers'? Champion$", t):
        return "champion", None
    if re.search(r"Constructors'? Champion$", t):
        return "constructors_champion", None
    if re.search(r"win \d+\+ Grands Prix in \d{4}", q):
        return "season_wins_ge", None
    if re.search(r"finish ahead of .+ in the \d{4} Drivers'? Championship", q):
        return "standings_h2h", None
    if m_ := re.search(r"^(.*Grand Prix): Driver Fastest Lap", t):
        return "race_fastest_lap", m_.group(1)
    gp = gp_name(t) or gp_name(q)
    for pat, kind in PROPS:
        if gp and re.search(pat, q or t, re.I):
            return kind, gp
    if m_ := re.search(r"^(.*Grand Prix): ", t):       # constructor fastest lap, ...: linked to the race, no price
        return "unmodeled", m_.group(1)
    return "unmodeled", None


class Resolver:
    """Driver / team / race lookups against the database."""

    def __init__(self, conn, year):
        self.year = year
        rows = conn.execute(text("""
            SELECT DISTINCT a.id, a.display_name FROM athletes a JOIN results r ON r.athlete_id = a.id
            JOIN rounds ro ON ro.id = r.round_id JOIN races ra ON ra.id = ro.race_id JOIN events e ON e.id = ra.event_id
            JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
            WHERE co.code = 'f1_wdc' AND s.year = :y"""), dict(y=year)).all()
        self.by_full = {_norm(n): i for i, n in rows}
        surname = {}
        for i, n in rows:
            surname.setdefault(_norm(n).split(" ")[-1], set()).add(i)
        self.by_surname = {k: next(iter(v)) for k, v in surname.items() if len(v) == 1}
        self.races = conn.execute(text("""
            SELECT ra.id, coalesce(ra.format->>'event_name', '') || ' ' || e.name, e.source_key, e.start_date
            FROM races ra JOIN events e ON e.id = ra.event_id
            JOIN seasons s ON s.id = e.season_id JOIN competitions co ON co.id = s.competition_id
            WHERE co.code = 'f1_wdc' AND s.year = :y"""), dict(y=year)).all()

    def driver(self, name):
        n = _norm(name)
        n = DRIVER_ALIASES.get(n, n) if n not in self.by_full else n
        if n in self.by_full:
            return self.by_full[n]
        parts = [p for p in n.split(" ") if p not in ("jr", "sr")]
        if parts and parts[-1] in self.by_surname:
            return self.by_surname[parts[-1]]
        # "Kimi Antonelli" vs "Andrea Kimi Antonelli": match on the last two tokens
        for full, i in self.by_full.items():
            if len(parts) >= 2 and full.endswith(" ".join(parts[-2:])):
                return i
        return None

    def team(self, name):
        """Exact name, else a known team name inside a sponsor label ("Mclaren Mastercard", "Tgr Haas")."""
        n = _norm(name)
        if n in TEAM_NAMES:
            return TEAM_NAMES[n]
        for label in sorted(TEAM_NAMES, key=len, reverse=True):
            if re.search(rf"\b{label}\b", n):
                return TEAM_NAMES[label]
        return None

    def race(self, gp_name, end_date=None, near=None):
        """Race by Grand Prix name, and (when the market's end date is known) held within
        MATCH_DAYS of it: a cancelled or rescheduled GP keeps its name but not its date.
        near (a date, for a venue whose expiry says nothing about the race, e.g. OG.com's): of the races with
        that name, the one held closest to it."""
        g = _norm(gp_name).replace(" grand prix", "")
        g = GP_ALIASES.get(g, g)
        found = []
        for rid, name, key, start in self.races:
            if not g or g not in _norm(name):
                continue
            if end_date is not None and start is not None and abs((end_date.date() - start).days) > MATCH_DAYS:
                continue
            if near is None:
                return rid, key
            found.append((abs((near - start).days) if start is not None else 10**6, rid, key))
        return min(found)[1:] if found else (None, None)


def targets_of(kind, question, group, outcomes, R):
    """[(token index, athlete_id, params, outcome label)]: which outcome tokens of a market to link and who each one
    names, from the kind's subject (markets/kinds.toml) and the listing. R: a Resolver (driver, team)."""
    subject = K.KINDS[kind].subject if kind in K.KINDS else None
    question, yes = question or "", outcomes[0] if outcomes else "Yes"
    if kind == "standings_h2h":
        mm = re.search(r"Will (.+?) finish ahead of (.+?) in the", question)
        a, b = (R.driver(mm.group(1)), R.driver(mm.group(2))) if mm else (None, None)
        return [(0, a, {"opponent_id": b}, yes)]
    if kind == "season_wins_ge":
        mm = re.search(r"Will (.+?) win (\d+)\+ Grands Prix", question)
        return [(0, R.driver(mm.group(1)) if mm else None, {"n": int(mm.group(2)) if mm else None}, yes)]
    if subject == "pair" and (mm := AHEAD.search(question)):           # one yes/no question: yes = A ahead of B
        a, b = R.driver(mm.group(1)), R.driver(mm.group(2))
        return [(0, a, {"opponent_id": b}, yes)]
    if subject == "pair" and len(outcomes) == 2:                       # two tokens, one per driver
        a, b = R.driver(outcomes[0]), R.driver(outcomes[1])
        return [(0, a, {"opponent_id": b}, outcomes[0]), (1, b, {"opponent_id": a}, outcomes[1])]
    if kind == "race_constructor_h2h":                                 # two tokens, one per team
        if len(outcomes) != 2:
            return [(0, None, None, yes)]
        a, b = R.team(outcomes[0]), R.team(outcomes[1])
        return [(0, None, {"team": a, "opponent": b}, outcomes[0]), (1, None, {"team": b, "opponent": a}, outcomes[1])]
    if subject == "team":                                              # the group title is the team
        return [(0, None, {"team": R.team(group)}, yes)]
    if subject == "driver":                                            # the group title is the driver
        return [(0, R.driver(group), None, yes)]
    return [(0, None, None, yes)]


def _events(closed_year=None, tags=TAGS):
    """Active events of the given Gamma tags (default F1's); with closed_year, also every closed event ending that year."""
    out = {}
    with httpx.Client(base_url=GAMMA, timeout=30) as c:
        for tag in tags:
            for e in http.get(c, "/events", params={"tag_slug": tag, "active": "true", "closed": "false",
                                                   "limit": 200}).json():
                out[e["slug"]] = e
            if closed_year:
                off = 0
                while off < 5000:
                    page = http.get(c, "/events", params={"tag_slug": tag, "closed": "true", "limit": 100,
                                                          "offset": off}).json()
                    if not page:
                        break
                    for e in page:
                        if str(e.get("endDate") or "").startswith(str(closed_year)):
                            out[e["slug"]] = e
                    off += 100
    return out


def _vanished(conn, competition_id, seen, limit=REREAD_MAX):
    """Slugs of events with links still open in market_links that the listing (`seen`) no longer returns,
    most recently synced first."""
    slugs = conn.execute(text("""SELECT event_slug FROM market_links
                                 WHERE exchange = 'polymarket' AND competition_id = :c AND closed IS NOT TRUE
                                   AND event_slug IS NOT NULL
                                 GROUP BY event_slug ORDER BY max(synced_at) DESC NULLS LAST"""),
                         dict(c=competition_id)).scalars()
    return [s for s in slugs if s not in seen][:limit]


def _by_slug(slugs):
    """Events by slug whatever their state (closed ones too), so their markets' closed flags and prices are read.
    A slug Gamma fails on (an error status, a network error, no event) is skipped for this pass."""
    out = {}
    if not slugs:
        return out
    with httpx.Client(base_url=GAMMA, timeout=30) as c:
        for slug in slugs:
            try:
                r = http.get(c, "/events", params={"slug": slug}, tries=2)
                body = r.json() if r.status_code == 200 else []
            except (httpx.HTTPError, ValueError):
                continue
            for e in body if isinstance(body, list) else []:
                if isinstance(e, dict) and e.get("slug"):
                    out[e["slug"]] = e
    return out


def _f(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def competition(session, sport):
    """(Competition, its first Category) of a sport, as the tape-only sync files links under."""
    schema = sports.load(sport)
    comp = session.scalars(select(m.Competition).filter_by(code=schema["competition"]["code"])).one()
    cat_code = next(iter(schema["competition"]["categories"]))
    return comp, session.scalars(select(m.Category).filter_by(competition_id=comp.id, code=cat_code)).one()


def sync(session, conn, year=2026, include_closed=False, new=None, sport="f1", tags=None):
    """Fetch every active event (and with include_closed, every closed one of `year`) and upsert one
    market_links row per outcome token. Tokens seen for the first time get first_seen_at and, if `new`
    is a list, are appended to it (markets/alerts.py). F1 (the default) is classified and matched to its
    drivers and races ([markets] titles in its schema); any other sport (nascar, motogp, indycar) pages its schema's
    Gamma tags (or `tags`) and files its links under its own competition: its [identity] resolver, when it has one,
    fills athlete_id, race_id and params.kind, and a link whose kind the schema's [markets] kinds list, fully
    identified, is filed as that kind (identity.promote); every other link stays `unmodeled`.
    Additive: never deletes a row."""
    tape = not sports.titles_classified(sport)    # [markets] titles: F1's title classifier and Resolver
    if tape:
        tags = tuple(tags or sports.polymarket_tags(sport))
        if not tags:
            raise ValueError(f"{sport}: no [markets.polymarket] tags in sports/{sport}.toml")
        comp, cat = competition(session, sport)
        R = None
        events = _events(year if include_closed else None, tags)
    else:
        comp = session.scalars(select(m.Competition).filter_by(code="f1_wdc")).one()
        cat = session.scalars(select(m.Category).filter_by(competition_id=comp.id, code="DRV")).one()
        R = Resolver(conn, year)
        events = _events(year if include_closed else None, tuple(tags) if tags else TAGS)
    reread = _by_slug(_vanished(conn, comp.id, events))     # closed since the last pass: off the active listing
    events.update(reread)
    now = datetime.now(timezone.utc)
    stats = dict(events=0, links=0, modeled=0, unmatched=0, new=0, reread=len(reread))
    who = identity.linker(sport, conn) if tape else None            # a tape-only sport with a resolver: driver, race, kind
    for slug, ev in events.items():
        stats["events"] += 1
        for mk in ev.get("markets", []):
            if not mk.get("clobTokenIds"):
                continue
            kind, gp = ("unmodeled", None) if tape else classify(ev.get("title"), mk.get("question"))
            outcomes = json.loads(mk.get("outcomes") or "[]")
            tokens = json.loads(mk.get("clobTokenIds") or "[]")
            prices = [_f(p) for p in json.loads(mk.get("outcomePrices") or "[]")]
            bid, ask = _f(mk.get("bestBid")), _f(mk.get("bestAsk"))
            end = datetime.fromisoformat(mk["endDate"].replace("Z", "+00:00")) if mk.get("endDate") else None
            race_id, race_key = R.race(gp, end) if gp and R else (None, None)
            group = mk.get("groupItemTitle") or ""
            closed = bool(mk.get("closed"))
            # which outcome tokens to link, and to whom; a tape-only sport: the first outcome token, as F1's
            # unmodeled markets (no athlete, no race)
            targets = ([(0, None, None, outcomes[0] if outcomes else "Yes")] if tape
                       else targets_of(kind, mk.get("question"), group, outcomes, R))

            for i, athlete_id, params, label in targets:
                if i >= len(tokens):
                    continue
                matched = kind == "unmodeled" or (
                    (athlete_id is not None or (params or {}).get("team") or kind in PROP_KINDS)
                    and (params is None or all(v is not None for v in params.values()))
                    and (not kind.startswith("race_") or race_id is not None))
                t_bid, t_ask = (bid, ask) if i == 0 else ((1 - ask) if ask is not None else None,
                                                         (1 - bid) if bid is not None else None)
                mid = (t_bid + t_ask) / 2 if t_bid is not None and t_ask is not None else (prices[i] if i < len(prices) else None)
                resolved = None
                if closed and i < len(prices) and prices[i] in (0.0, 1.0):
                    resolved = prices[i] == 1.0
                if params and race_key:
                    params = dict(params, event_key=race_key)
                elif race_key:
                    params = {"event_key": race_key}
                if race_id is not None and PLACEHOLDER.match(group) and kind.startswith("race_"):
                    params = dict(params or {}, placeholder=kind)
                values = dict(
                    exchange="polymarket", market_slug=mk.get("slug"), question=mk.get("question") or ev.get("title"),
                    condition_id=mk.get("conditionId"), outcome=str(label), neg_risk=bool(mk.get("negRisk") or ev.get("negRisk")),
                    tick_size=_f(mk.get("orderPriceMinTickSize")) or 0.01, min_size=_f(mk.get("orderMinSize")) or 5,
                    competition_id=comp.id, category_id=cat.id, athlete_id=athlete_id, race_id=race_id,
                    prediction=kind if matched else "unmodeled", params=params, invert=False,
                    event_slug=slug, event_title=ev.get("title"), group_title=group or str(label),
                    last_bid=t_bid, last_ask=t_ask, last_price=mid, volume=_f(mk.get("volume")),
                    end_date=datetime.fromisoformat(mk["endDate"].replace("Z", "+00:00")) if mk.get("endDate") else None,
                    closed=closed, resolved_yes=resolved, synced_at=now, active=not closed)
                if who:
                    who.fill([values])
                    identity.promote(sport, [values])     # prediction = params.kind where [markets] kinds files it
                link = session.scalars(select(m.MarketLink).filter_by(token_id=tokens[i])).first()
                if link is None:
                    session.add(m.MarketLink(token_id=tokens[i], first_seen_at=now, **values))
                    stats["new"] += 1
                    if new is not None:
                        new.append(dict(values, token_id=tokens[i]))
                else:
                    for k, v in values.items():
                        setattr(link, k, v)
                stats["links"] += 1
                if values["prediction"] != "unmodeled":
                    stats["modeled"] += 1
                elif not matched:
                    stats["unmatched"] += 1
    if who:
        stats["identity"] = dict(who.counts)
    session.commit()
    return stats


CLOB = "https://clob.polymarket.com"


def _sport_where(event_slugs, sport):
    """(SQL condition, params) selecting Polymarket links: of the given events, else of `sport`'s competition."""
    if event_slugs:
        return "ml.event_slug = ANY(:s)", dict(s=list(event_slugs))
    if not sport:
        raise ValueError("give event slugs or a sport")
    return "co.code = :c", dict(c=sports.load(sport)["competition"]["code"])


def fetch_history(session, conn, event_slugs, start, end, fidelity=60, tokens=None, sport=None):
    """Store price history for every outcome token of the given events (or the given
    `tokens`; or, with no events, every Polymarket link of `sport`'s competition), between `start`
    and `end` (datetimes, UTC). Returns points stored."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    if tokens is None and not event_slugs and sport:
        where, params = _sport_where(None, sport)
        tokens = conn.execute(text(f"""SELECT ml.token_id FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
                                       WHERE ml.exchange = 'polymarket' AND {where} ORDER BY ml.token_id"""), params).scalars().all()
    elif tokens is None:
        tokens = conn.execute(text("SELECT token_id FROM market_links WHERE event_slug = ANY(:s) AND exchange = 'polymarket'"),
                              dict(s=list(event_slugs))).scalars().all()
    n, refused = 0, {}
    with httpx.Client(base_url=CLOB, timeout=20) as c:
        for tok in tokens:
            pts, a = [], start
            while a < end:      # one request per HISTORY_WINDOW: a 37-day request stored nothing (2026-10-08 backfill)
                b = min(a + HISTORY_WINDOW, end)
                r = http.get(c, "/prices-history", params={"market": tok, "startTs": int(a.timestamp()),
                                                      "endTs": int(b.timestamp()), "fidelity": fidelity})
                if r.status_code == 200:
                    pts += r.json().get("history", [])
                else:
                    refused[r.status_code] = r.text[:160]
                a = b
            if not pts:
                continue
            # the API can repeat a timestamp, and one upsert can't touch a row twice: keep the last point per ts
            rows = list({p["t"]: dict(token_id=tok, ts=datetime.fromtimestamp(p["t"], tz=timezone.utc), price=float(p["p"]))
                         for p in pts}.values())
            session.execute(pg_insert(m.MarketPriceHistory).values(rows).on_conflict_do_update(
                index_elements=["token_id", "ts"], set_={"price": pg_insert(m.MarketPriceHistory).excluded.price}))
            n += len(rows)
    session.commit()
    for code, body in refused.items():      # a refusal is said, not stored as a silent 0
        print(f"polymarket history: HTTP {code} on some requests: {body}", file=sys.stderr)
    return n


DATA_API = "https://data-api.polymarket.com"


def fetch_trades(session, conn, event_slugs, page=500, max_offset=100_000, modeled_only=False, since=None, sport=None,
                 open_only=False):
    """Store every taker trade for the markets of the given events (Data API; with no events, of every
    Polymarket link of `sport`'s competition).
    `side` is the taker's side for `token_id`. Idempotent. Returns trades stored.
    since (datetime, UTC): stop paging a market once a page reaches trades this old (newest come first).
    open_only: markets with an open outcome (the recorder's pass)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    if not event_slugs and sport:
        where, params = _sport_where(None, sport)
        conds = conn.execute(text(f"""SELECT DISTINCT ml.condition_id FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
                                      WHERE ml.exchange = 'polymarket' AND {where} AND ml.condition_id IS NOT NULL
                                      AND (NOT CAST(:m AS boolean) OR ml.prediction <> 'unmodeled')
                                      AND (NOT CAST(:o AS boolean) OR NOT ml.closed)"""),
                             dict(params, m=modeled_only, o=open_only)).scalars().all()
    else:
        conds = conn.execute(text("""SELECT DISTINCT condition_id FROM market_links WHERE event_slug = ANY(:s)
                                     AND exchange = 'polymarket' AND condition_id IS NOT NULL AND (NOT CAST(:m AS boolean) OR prediction <> 'unmodeled')
                                     AND (NOT CAST(:o AS boolean) OR NOT closed)"""),
                             dict(s=list(event_slugs or []), m=modeled_only, o=open_only)).scalars().all()
    n = 0
    with httpx.Client(base_url=DATA_API, timeout=30) as c:
        for cond in conds:
            off, rows = 0, []
            while off <= max_offset:
                r = http.get(c, "/trades", params={"market": cond, "limit": page, "offset": off, "takerOnly": "true"})
                batch = r.json() if r.status_code == 200 else []
                if not isinstance(batch, list) or not batch:
                    break
                rows += [dict(token_id=t["asset"], condition_id=cond, outcome_index=int(t.get("outcomeIndex") or 0),
                              ts=datetime.fromtimestamp(int(t["timestamp"]), tz=timezone.utc), side=t["side"],
                              price=float(t["price"]), size=float(t["size"]), tx_hash=t["transactionHash"],
                              wallet=t.get("proxyWallet")) for t in batch]
                if len(batch) < page:
                    break
                if since is not None and min(int(t["timestamp"]) for t in batch) < since.timestamp():
                    break
                off += page
            for i in range(0, len(rows), 1000):
                session.execute(pg_insert(m.MarketTrade).values(rows[i:i + 1000]).on_conflict_do_nothing(
                    constraint="uq_market_trade"))
            n += len(rows)
    session.commit()
    return n


def _levels(side, best_first_desc):
    lv = sorted(((float(x["price"]), float(x["size"])) for x in side or []), reverse=best_first_desc)
    return [[p, s] for p, s in lv]


def snapshot_books(session, conn, event_slugs=None, depth=10, sport=None):
    """One order-book snapshot for every open outcome token of the given events
    (default: every open modeled market, season-long or for a race that hasn't been run, and every open F1 market
    linked to such a race even when its kind isn't modeled yet, so a new kind's tape starts when the market is found,
    not when the kind is promoted: practice fastest lap, extra drivers;
    with `sport` and no events: every open Polymarket link of that sport's competition).
    Returns snapshots stored."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    if event_slugs:
        toks = conn.execute(text("SELECT token_id FROM market_links WHERE event_slug = ANY(:s) AND NOT closed AND exchange = 'polymarket'"),
                            dict(s=list(event_slugs))).scalars().all()
    elif sport:
        where, params = _sport_where(None, sport)
        toks = conn.execute(text(f"""SELECT ml.token_id FROM market_links ml JOIN competitions co ON co.id = ml.competition_id
                                     WHERE NOT ml.closed AND ml.exchange = 'polymarket' AND {where} ORDER BY ml.token_id"""),
                            params).scalars().all()
    else:
        toks = conn.execute(text("""
            SELECT ml.token_id FROM market_links ml LEFT JOIN races ra ON ra.id = ml.race_id
            LEFT JOIN events e ON e.id = ra.event_id LEFT JOIN competitions co ON co.id = ml.competition_id
            WHERE NOT ml.closed AND ml.exchange = 'polymarket'
              AND (ml.prediction <> 'unmodeled' OR (ml.race_id IS NOT NULL AND co.code = :f1))
              AND (ml.race_id IS NULL OR e.status <> 'completed')"""),
            dict(f1=sports.load("f1")["competition"]["code"])).scalars().all()
    rows = []
    with httpx.Client(base_url=CLOB, timeout=30) as c:
        for i in range(0, len(toks), 100):
            r = http.post(c, "/books", json=[{"token_id": t} for t in toks[i:i + 100]])
            for b in (r.json() if r.status_code == 200 else []):
                bids, asks = _levels(b.get("bids"), True), _levels(b.get("asks"), False)
                rows.append(dict(token_id=b["asset_id"], ts=datetime.fromtimestamp(int(b["timestamp"]) / 1000, tz=timezone.utc),
                                 best_bid=bids[0][0] if bids else None, best_ask=asks[0][0] if asks else None,
                                 bids=bids[:depth], asks=asks[:depth]))
    if rows:
        session.execute(pg_insert(m.MarketBookSnapshot).values(rows).on_conflict_do_nothing())
        session.commit()
    return len(rows)
