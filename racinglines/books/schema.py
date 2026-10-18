"""
The three sportsbook schemas (docs/sportsbook/index.md), validated the way frames/schema.py validates a frame:

    load_event(path)    events/<sport>/<season>-<round>-<slug>.toml   one event, no venue, no prices, no entrants
    load_venue(path)    venues/<code>.toml                            a venue's odds format and title rules
    load_book(path)     books/<venue>/<date>-<event>.toml             one capture of a venue's lines
    load(path)          whichever of the three the file is (by its top-level table)

Each returns the parsed dict, or raises BookError listing every problem found, each naming the field. The checks:
required sections and fields, odds that can be odds (decimal > 1; American an integer of at least 100 either way),
UTC times (ISO 8601 ending in Z, or "unknown"), every market kind in the kinds registry or among the props
(models/props.PROP_KINDS), every line either mapped to a market key or `"unmapped"`, every rule a
compilable regex. A kind the registry has but no model prices yet still passes: pricing says so, not loading.

Market keys are `(kind, subject, params)` in the registry's vocabulary: a mapped line's `market` table carries `kind`
plus `driver` / `team` / `opponent` / `line` / `side` as the kind needs. A combo (same-game parlay, e.g. "Race Winner
and Fastest Lap") is `kind = "combo"` with `legs = [...]` (two or more such tables) and `void_leg = "void_all" |
"drop_leg"` (default void_all), priced and settled by racinglines/markets/combos.py. Selections resolve to the entry list through
an exact alias table at pricing time, never here and never fuzzily.

Optional fields read by `racinglines book` (books/slips.py, docs/sportsbook/slips.md): `[book] sport`; `sport`, `event`,
`race_id`, `driver_id`, `opponent_id` and a leg's own `odds` on any market table or combo leg (a slip across events and
sports; a combo whose legs all carry odds may leave out the line's odds); a line's `id`; an exact `[aliases]` table. Read by
`racinglines book export` (books/ledger.py): a line's `stake` (the bet placed, in book.currency, a number above 0) and
`placed_utc` (when it was placed).

Odds presentations (`venue.odds`, `book.odds`; names provisional until the owner checks them, docs/todo.md):

    decimal       1.85          European sportsbooks: the payout per unit staked, stake included
    american      -118 / +150   US sportsbooks: stake to win 100, or the win on a 100 stake
    fractional    "7/4"         UK sportsbooks: the win per unit staked, as a string
    cents         37            Kalshi: the price of a $1 YES contract, in cents (1 to 99)
    dollars       0.37          Polymarket and OG.com: the price of a $1 share, in dollars (0 to 1)
    prob          0.37          a plain probability: our fair values, and a book quoted as one

`to_prob(value, fmt)` is the implied probability of a quote (the venue's vig still in it); `present(prob, fmt)` writes
a probability in a venue's presentation, so a fair value can sit next to the quote in the book's own units.
"""
from fractions import Fraction


import re
import tomllib
from pathlib import Path

from racinglines.markets import kinds as K
from racinglines.models.props import PROP_KINDS

KNOWN_KINDS = frozenset(K.KINDS) | frozenset(PROP_KINDS)
VENUE_KINDS = ("exchange", "sportsbook")
ODDS_FORMATS = ("decimal", "american", "fractional", "cents", "dollars", "prob")
SOURCES = ("screenshot", "paste", "api")
SUBJECTS = ("driver", "team", "pair", "line", "none", "title_driver", "title_team")
SIDES = ("yes", "no", "over", "under")
UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$")
EVENT_KEY = re.compile(r"^\d{4}-\d{2}$")                     # the launch-spec key, e.g. "2026-17"


class BookError(ValueError):
    """A file that doesn't match its schema. `problems` lists each one, naming the field."""

    def __init__(self, path, problems):
        super().__init__(f"{path}: " + "; ".join(problems))
        self.problems = list(problems)


def _read(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def _need(d, section, fields, problems):
    """Every field present and non-empty under `section`; returns the section (or {})."""
    sec = d.get(section)
    if not isinstance(sec, dict):
        problems.append(f"[{section}] missing")
        return {}
    for f in fields:
        if sec.get(f) in (None, ""):
            problems.append(f"{section}.{f} missing")
    return sec


def _utc(value, field, problems, allow_unknown=True):
    if value == "unknown" and allow_unknown:
        return
    if not isinstance(value, str) or not UTC.match(value):
        problems.append(f"{field}: {value!r} is not a UTC time like 2026-10-11T12:00Z" + (' or "unknown"' if allow_unknown else ""))


FRACTIONAL = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def check_odds(value, fmt):
    """None if `value` can be odds in `fmt`, else the reason."""
    if fmt == "fractional":
        m = FRACTIONAL.match(value) if isinstance(value, str) else None
        return None if m and int(m[1]) > 0 and int(m[2]) > 0 else f'fractional odds {value!r} must be a string like "7/4"'
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{value!r} is not a number"
    if fmt == "decimal":
        return None if value > 1 else f"decimal odds {value} must be above 1"
    if fmt == "american":
        if value != int(value) or abs(value) < 100:
            return f"American odds {value} must be an integer of at least 100 either way"
        return None
    if fmt == "cents":
        return None if value == int(value) and 1 <= value <= 99 else f"cents {value} must be an integer from 1 to 99"
    if fmt in ("dollars", "prob"):
        return None if 0 < value < 1 else f"{fmt} {value} must be between 0 and 1"
    return f"unknown odds format {fmt!r}"


def to_prob(value, fmt):
    """The implied probability of a quote in `fmt` (the venue's margin still in it). Raises ValueError on a bad quote."""
    bad = check_odds(value, fmt)
    if bad:
        raise ValueError(bad)
    if fmt == "decimal":
        return 1.0 / float(value)
    if fmt == "american":
        v = float(value)
        return (-v / (100.0 - v)) if v < 0 else 100.0 / (v + 100.0)
    if fmt == "fractional":
        m = FRACTIONAL.match(value)
        return int(m[2]) / (int(m[1]) + int(m[2]))
    if fmt == "cents":
        return float(value) / 100.0
    return float(value)                                      # dollars, prob


def present(prob, fmt):
    """A probability written in a venue's presentation: a decimal to two places, American odds as an int (never
    between -100 and +100), a fractional string with a denominator up to 20, cents as an int, dollars and prob to
    two places. Raises ValueError outside (0, 1)."""
    if not 0 < prob < 1:
        raise ValueError(f"probability {prob} must be between 0 and 1")
    if fmt == "decimal":
        return round(1.0 / prob, 2)
    if fmt == "american":
        return int(round(-100.0 * prob / (1.0 - prob))) if prob >= 0.5 else int(round(100.0 * (1.0 - prob) / prob))
    if fmt == "fractional":
        f = Fraction(1.0 / prob - 1.0).limit_denominator(20)
        return f"{f.numerator}/{f.denominator}"
    if fmt == "cents":
        return int(min(99, max(1, round(prob * 100))))
    if fmt in ("dollars", "prob"):
        return round(prob, 2)
    raise ValueError(f"unknown odds format {fmt!r}")


COMBO = "combo"
VOID_LEG = ("void_all", "drop_leg")


def _combo(m, field, problems, fmt=None):
    """A combo (same-game parlay) line: `legs`, at least two market tables of single kinds (no combo inside a combo),
    and `void_leg` ("void_all" by default: a void leg voids the slip; "drop_leg": the rest decide). Priced and settled by
    racinglines/markets/combos.py once the legs' names resolve to entrants."""
    legs = m.get("legs")
    if not isinstance(legs, list) or len(legs) < 2:
        problems.append(f"{field}.legs: a combo needs a list of at least two legs")
        return
    if m.get("void_leg", "void_all") not in VOID_LEG:
        problems.append(f"{field}.void_leg: {m.get('void_leg')!r} not in {VOID_LEG}")
    for i, leg in enumerate(legs):
        if isinstance(leg, dict) and leg.get("kind") == COMBO:
            problems.append(f"{field}.legs[{i}].kind: a combo can't hold a combo")
            continue
        if leg == "unmapped":
            problems.append(f"{field}.legs[{i}]: a combo's legs must all be mapped")
            continue
        _market(leg, f"{field}.legs[{i}]", problems, fmt)


def _where(m, field, problems, fmt=None):
    """The optional fields that place a market (or a combo's leg) on an event other than the book's, by exact keys
    (racinglines/books/slips.py): `sport` (a sports/<code>.toml), `event` (the launch-spec key) or `race_id` (the DB's
    races.id), and a leg's own `odds` in the book's format (a parlay priced from its legs' odds)."""
    from racinglines import sports as SP
    if "sport" in m and m["sport"] not in SP.SPORT_CODES:
        problems.append(f"{field}.sport: {m['sport']!r} not in {SP.SPORT_CODES}")
    if "event" in m and not (isinstance(m["event"], str) and EVENT_KEY.match(m["event"])):
        problems.append(f"{field}.event: {m['event']!r} is not like 2026-17")
    if "race_id" in m and (isinstance(m["race_id"], bool) or not isinstance(m["race_id"], int)):
        problems.append(f"{field}.race_id: the DB race id, an integer")
    for f in ("driver_id", "opponent_id"):
        if f in m and (isinstance(m[f], bool) or not isinstance(m[f], int)):
            problems.append(f"{field}.{f}: an athlete id, an integer")
    if "odds" in m and fmt in ODDS_FORMATS:
        bad = check_odds(m["odds"], fmt)
        if bad:
            problems.append(f"{field}.odds: {bad}")


def _market(m, field, problems, fmt=None):
    """A line's market key: "unmapped", a table with a known kind and the fields that kind's subject needs, or a combo
    (kind = "combo", its legs each such a table)."""
    if m == "unmapped":
        return
    if not isinstance(m, dict):
        problems.append(f'{field}: must be "unmapped" or a table with a kind')
        return
    kind = m.get("kind")
    _where(m, field, problems, fmt)
    if kind == COMBO:
        _combo(m, field, problems, fmt)
        return
    if kind not in KNOWN_KINDS:
        problems.append(f"{field}.kind: {kind!r} is not a known kind")
        return
    if "side" in m and m["side"] not in SIDES:
        problems.append(f"{field}.side: {m['side']!r} not in {SIDES}")
    if "line" in m and (isinstance(m["line"], bool) or not isinstance(m["line"], (int, float))):
        problems.append(f"{field}.line: {m['line']!r} is not a number")
    k = K.KINDS.get(kind)
    if k is not None and k.spec is not None:            # a declarative kind: its spec says what a selection names
        spec = k.spec["payoff"]
        need = {"driver": ("driver",), "team": ("team",), "field": ()}[spec["subject"]] + \
            (("line",) if spec.get("compare") == "over" else ())
        for f in need:
            if f not in m and f"{f}_id" not in m:
                problems.append(f"{field}: kind {kind} " + ("needs a line" if f == "line" else f"names a {f}"))
        return
    payoff = k.payoff if k is not None else None
    driver = "driver" in m or "driver_id" in m
    opponent = "opponent" in m or "opponent_id" in m
    if payoff in ("top_n", "stage_top_n", "indicator", "mover") and not driver:
        problems.append(f"{field}: kind {kind} names a driver")
    if payoff == "group_top" and "team" not in m:
        problems.append(f"{field}: kind {kind} names a team")
    if payoff == "h2h" and not (driver and opponent):
        problems.append(f"{field}: kind {kind} names a driver and an opponent")


def validate_event(d, path="event"):
    problems = []
    ev = _need(d, "event", ("sport", "season", "round", "key", "name", "race_id", "format"), problems)
    if ev:
        if not isinstance(ev.get("season"), int):
            problems.append("event.season: must be an integer year")
        if not isinstance(ev.get("round"), int):
            problems.append("event.round: must be an integer")
        if not (isinstance(ev.get("key"), str) and EVENT_KEY.match(ev["key"])):
            problems.append(f"event.key: {ev.get('key')!r} is not like 2026-17")
        rid = ev.get("race_id")
        if not (rid == "unknown" or (isinstance(rid, int) and not isinstance(rid, bool))):
            problems.append('event.race_id: the DB race id (an integer) or "unknown"')
    sessions = d.get("sessions")
    if not isinstance(sessions, dict) or not sessions:
        problems.append("[sessions] missing")
    else:
        for name, when in sessions.items():
            _utc(when, f"sessions.{name}", problems)
    mk = d.get("markets")
    if not isinstance(mk, dict) or not isinstance(mk.get("kinds"), list) or not mk["kinds"]:
        problems.append("markets.kinds: a non-empty list of kinds")
    else:
        for k in mk["kinds"]:
            if k not in KNOWN_KINDS:
                problems.append(f"markets.kinds: {k!r} is not a known kind")
        for k, lines in (mk.get("lines") or {}).items():
            if k not in mk["kinds"]:
                problems.append(f"markets.lines.{k}: not in markets.kinds")
            if not isinstance(lines, list) or not all(isinstance(x, (int, float)) for x in lines):
                problems.append(f"markets.lines.{k}: a list of numbers")
    if problems:
        raise BookError(path, problems)
    return d


def validate_venue(d, path="venue"):
    problems = []
    v = _need(d, "venue", ("code", "name", "kind", "odds", "currency"), problems)
    if v:
        if v.get("kind") not in VENUE_KINDS:
            problems.append(f"venue.kind: {v.get('kind')!r} not in {VENUE_KINDS}")
        if v.get("odds") not in ODDS_FORMATS:
            problems.append(f"venue.odds: {v.get('odds')!r} not in {ODDS_FORMATS}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", str(v.get("code", ""))):
            problems.append("venue.code: lowercase letters, digits and underscores")
    rules = d.get("rules")
    if not isinstance(rules, list) or not rules:
        problems.append("[[rules]] missing")
    else:
        for i, r in enumerate(rules):
            f = f"rules[{i}]"
            for need in ("title", "kind", "subject"):
                if r.get(need) in (None, ""):
                    problems.append(f"{f}.{need} missing")
            try:
                re.compile(r.get("title") or "")
            except re.error as ex:
                problems.append(f"{f}.title: not a regex ({ex})")
            if r.get("kind") not in KNOWN_KINDS:
                problems.append(f"{f}.kind: {r.get('kind')!r} is not a known kind")
            if r.get("subject") not in SUBJECTS:
                problems.append(f"{f}.subject: {r.get('subject')!r} not in {SUBJECTS}")
    if problems:
        raise BookError(path, problems)
    return d


def validate_book(d, path="book"):
    problems = []
    b = _need(d, "book", ("venue", "event", "captured_utc", "source", "odds", "currency"), problems)
    fmt = b.get("odds")
    if b:
        _utc(b.get("captured_utc"), "book.captured_utc", problems, allow_unknown=False)
        if b.get("source") not in SOURCES:
            problems.append(f"book.source: {b.get('source')!r} not in {SOURCES}")
        if fmt not in ODDS_FORMATS:
            problems.append(f"book.odds: {fmt!r} not in {ODDS_FORMATS}")
        if not (isinstance(b.get("event"), str) and EVENT_KEY.match(b["event"])):
            problems.append(f"book.event: {b.get('event')!r} is not like 2026-17")
        from racinglines import sports as SP
        if "sport" in b and b["sport"] not in SP.SPORT_CODES:
            problems.append(f"book.sport: {b['sport']!r} not in {SP.SPORT_CODES}")
        if "image" in b and not (isinstance(b["image"], str) and b["image"]):
            problems.append("book.image: the screenshot's path, a string (relative to the book file)")
    aliases = d.get("aliases", {})
    if not isinstance(aliases, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()):
        problems.append('aliases: a table of exact names, "venue name" = "display name"')
    lines = d.get("lines")
    if not isinstance(lines, list) or not lines:
        problems.append("[[lines]] missing")
    else:
        ids = set()
        for i, ln in enumerate(lines):
            f = f"lines[{i}]"
            m = ln.get("market")
            # a parlay may leave its odds to its legs: the payout is the product of the legs' odds
            legs_priced = isinstance(m, dict) and m.get("kind") == COMBO and isinstance(m.get("legs"), list) and \
                all(isinstance(x, dict) and "odds" in x for x in m["legs"])
            for need in ("title", "selection", "odds", "market"):
                if need == "odds" and legs_priced:
                    continue
                if need not in ln or ln[need] in (None, ""):
                    problems.append(f"{f}.{need} missing")
            if "odds" in ln and fmt in ODDS_FORMATS:
                bad = check_odds(ln["odds"], fmt)
                if bad:
                    problems.append(f"{f}.odds: {bad}")
            if "stake" in ln and (isinstance(ln["stake"], bool) or not isinstance(ln["stake"], (int, float))
                                  or ln["stake"] <= 0):
                problems.append(f"{f}.stake: {ln['stake']!r} must be a number above 0, in book.currency")
            if "placed_utc" in ln:
                _utc(ln["placed_utc"], f"{f}.placed_utc", problems, allow_unknown=False)
            if "id" in ln:
                if not isinstance(ln["id"], str) or not ln["id"] or ln["id"] in ids:
                    problems.append(f"{f}.id: {ln['id']!r} must be a unique non-empty string")
                ids.add(ln.get("id"))
            if "market" in ln:
                _market(ln["market"], f"{f}.market", problems, fmt)
            if "box" in ln:
                bx = ln["box"]
                if not (isinstance(bx, list) and len(bx) == 4 and all(isinstance(v, (int, float)) and
                                                                     not isinstance(v, bool) and v >= 0 for v in bx)):
                    problems.append(f"{f}.box: [x, y, w, h], four non-negative numbers (pixels on the screenshot)")
    if problems:
        raise BookError(path, problems)
    return d


VALIDATORS = {"event": validate_event, "venue": validate_venue, "book": validate_book}


def load_event(path):
    return validate_event(_read(path), path)


def load_venue(path):
    return validate_venue(_read(path), path)


def load_book(path):
    return validate_book(_read(path), path)


def load(path):
    """Validate whichever schema the file's top-level table says it is ([event], [venue] or [book])."""
    d = _read(path)
    found = [t for t in VALIDATORS if t in d]
    if len(found) != 1:
        raise BookError(path, [f"expected exactly one of [event], [venue], [book]; found {found or 'none'}"])
    return VALIDATORS[found[0]](d, path)


def unmapped(book):
    """The lines a validated book leaves without a market key."""
    return [ln for ln in book["lines"] if ln["market"] == "unmapped"]


EXAMPLES = Path(__file__).resolve().parents[2] / "docs" / "sportsbook" / "schemas"
