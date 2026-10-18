"""
Reference data: the sports, leagues, competitions and categories the project
knows about, plus canonical venue names. The values come from the sport
schemas (sports/*.toml, see racinglines/sports.py); `racinglines db seed` (and every
ingest) upserts these, so adding a new sport or league starts with a schema.
"""

from racinglines import sports as _S

_SCHEMAS = [_S.load(c) for c in _S.SPORT_CODES]

SPORTS = {                                     # code: (name, result_kind)
    s["sport"]["code"]: (s["sport"]["name"], s["sport"]["result_kind"]) for s in _SCHEMAS}

LEAGUES = {                                    # code: (name, organizer)
    s["league"]["code"]: (s["league"]["name"], s["league"]["organizer"]) for s in _SCHEMAS}

COMPETITIONS = {                               # code: league, sport, name, categories {code: (name, gender, age_group)}
    s["competition"]["code"]: dict(league=s["league"]["code"], sport=s["sport"]["code"], name=s["competition"]["name"],
                                   categories={k: tuple(v) for k, v in s["competition"]["categories"].items()})
    for s in _SCHEMAS}

# F1 circuits (FastF1 "Location" slug) that run on public roads: slower, walls, fewer overtakes
STREET_CIRCUITS = set(_S.load("f1")["tracks"]["street"])

# the downhill schema's venues and rounds (empty where the install has no mtb_dh schema: the public release is F1 only)
_DH = _S.load("mtb_dh") if "mtb_dh" in _S.SPORT_CODES else {}

# canonical venue slug: (display name, country)
VENUES = {slug: tuple(v) for slug, v in _DH.get("venues", {}).items()}

# other spellings seen in source data -> canonical slug
VENUE_ALIASES = dict(_DH.get("venue_aliases", {}))


def canonical_venue(slug):
    """A venue slug as the database stores it (mont-ste-anne -> mont-sainte-anne); unknown slugs unchanged."""
    return VENUE_ALIASES.get(slug, slug)

# source-data round labels -> running order within a race
ROUND_ORDER = dict(_DH.get("rounds", {}).get("order", {}))
