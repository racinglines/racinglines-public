"""
Account roles and access tiers, in one place.

    admin   everything, plus the activity log, user management and the database explorer
    pro     every maker and taker strategy, the private book, and the Lab (backtests, Edge Finder,
            diagnostics, replays)
    basic   "Your picks": a blend of BASIC_PICKS taker strategies drawn per account from profiles.TAKER_TOP
            (basic_members), on Markets / Strategy / Positions / Live; never which strategy made a pick, no fair
            values or edges (a 1-3 star rating instead, pick_stars), no book, no Lab

`pro` and `basic` were called `maker` and `taker` until 2026-09-30. Rows written before then still carry the old
spelling (no migration in that change): `canonical()` maps them, every user dict the app builds goes through it,
and `db_roles()` spells both forms for SQL `role IN (...)` filters. A one-line migration
(`UPDATE users SET role = 'pro' WHERE role = 'maker'`, same for taker -> basic) retires the aliases; until it runs,
nothing here breaks on either spelling. Strategy names (`maker` / `update` / `hold` ... in candidates and sweeps)
are not roles and are untouched.
"""

ROLES = ("admin", "pro", "basic")
LEGACY = {"maker": "pro", "taker": "basic"}       # old role value -> current role
PRO = ("admin", "pro")                            # the tier that sees fair values, the book and the Lab
BASIC = ("basic",)
ANY = ROLES

# A basic account runs "Your picks": BASIC_PICKS strategies from profiles.TAKER_TOP, drawn once per account from
# BASIC_SEED and the account id (stable: never re-rolled at login, nothing stored until an admin assigns it), each at
# weight 1/BASIC_PICKS. Change BASIC_SEED to re-draw every account. Pro and admin may run any candidate or blend.
BASIC_PICKS = 3
BASIC_SEED = "taker-top-2026-09-30"
BASIC_NAME = "Your picks"
# Profiles basic accounts ran before the draw (codes): an account that still has one keeps working (role edits are
# allowed) until it is re-assigned; a new assignment must be its own draw.
LEGACY_BASIC_PROFILES = ("A",)
# Pick rating for basic accounts: the entry's edge over its strategy's minimum edge -> stars, as (ratio, stars)
STARS = ((2.0, 3), (1.5, 2), (0.0, 1))
# What a basic account never sees on a signal (and on a signal's detail): which strategy made it, our numbers
BASIC_HIDDEN = ("fair", "edge", "candidate_id", "run_id")
BASIC_DETAIL_KEEP = ("followed", "backfill", "venue")


def canonical(role):
    """The current spelling of a stored role ('maker' -> 'pro', 'taker' -> 'basic'; anything else unchanged)."""
    return LEGACY.get(role, role)


def db_roles(*roles):
    """Every stored spelling of these roles, for SQL `role IN (...)`: db_roles('pro') -> ('pro', 'maker')."""
    out = []
    for r in roles:
        out.append(r)
        out.extend(old for old, new in LEGACY.items() if new == r)
    return tuple(out)


def is_pro(user):
    return canonical(user["role"]) in PRO


def is_basic(user):
    return canonical(user["role"]) == "basic"


def basic_members(user_id):
    """The BASIC_PICKS codes (from profiles.TAKER_TOP, in its rank order) a basic account's picks come from.
    Deterministic per account id and BASIC_SEED."""
    import random

    from racinglines.pipelines import profiles as PF
    drawn = random.Random(f"{BASIC_SEED}:{user_id}").sample(PF.TAKER_TOP, BASIC_PICKS)
    return tuple(sorted(drawn, key=PF.TAKER_TOP.index))


def basic_profile(conn, user_id):
    """A basic account's profile: the blend of its basic_members at 1/BASIC_PICKS each, named BASIC_NAME."""
    from racinglines.pipelines import profiles as PF
    return PF.combo(conn, [(code, 1.0 / BASIC_PICKS) for code in basic_members(user_id)], BASIC_NAME, basic=True)


def basic_profile_names():
    """The profile names a basic account may be assigned (its own draw, always named BASIC_NAME)."""
    return (BASIC_NAME,)


def _is_basic_draw(profile, user_id):
    from racinglines.pipelines import profiles as PF
    if not PF.is_combo(profile) or profile.get("name") != BASIC_NAME:
        return False
    codes = tuple(m["code"] for m in profile["members"])
    if user_id is not None:
        return set(codes) == set(basic_members(user_id))
    return len(codes) == BASIC_PICKS and all(c in PF.TAKER_TOP for c in codes)


def is_alpha_style(profile):
    """One of the alpha layout's five styles (profiles.ALPHA_STYLES) as tuned: a single strategy saved from
    /strategy/pick with its `base` code, no tweak, and that code's own settings (sizing may be any)."""
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import sweep_settings as SS
    code = (profile or {}).get("base")
    if code not in PF.ALPHA_STYLES or PF.is_combo(profile) or profile.get("adjusted") or not profile.get("settings"):
        return False
    key = lambda st: SS.Settings.from_dict(st, strict=False).key                     # noqa: E731
    return key(profile["settings"]) == key(PF.PROFILES[code]["settings"])


def allowed_profile(role, profile, user_id=None, legacy_ok=False):
    """May an account with this role run this strategy profile (a dict from profiles.load / basic_profile, or None)?
    Admin and pro: any. Basic: only its own draw (basic_profile; with user_id, checked against basic_members).
    legacy_ok: also a profile in LEGACY_BASIC_PROFILES, which a basic account already running one keeps.
    Under the alpha layout (web/layout.alpha) a basic account may also run one of the five styles (is_alpha_style).
    While profiles.MAKERS_OPEN is off (Alpha), a maker profile is admin-only."""
    from racinglines.pipelines import profiles as PF
    if profile is not None and not PF.MAKERS_OPEN and canonical(role) != "admin" and PF.is_maker(profile):
        return False
    if profile is None or canonical(role) in PRO:
        return True
    if _is_basic_draw(profile, user_id):
        return True
    from racinglines.web import layout
    if layout.alpha() and is_alpha_style(profile):
        return True
    if legacy_ok:
        from racinglines.pipelines import profiles as PF
        return profile.get("name") in {PF.PROFILES[c]["name"] for c in LEGACY_BASIC_PROFILES if c in PF.PROFILES}
    return False


# ---------------------------------------------------------------------------
# What a basic account sees of its picks
# ---------------------------------------------------------------------------

def _min_edge(settings, kind):
    """A taker's minimum edge for a market kind (the h2h threshold and per-kind map, as the taker applies them)."""
    from racinglines.markets.strategies import taker_weekend as RB
    from racinglines.pipelines import sweep_settings as SS
    st = SS.Settings.from_dict(settings or {}, strict=False)
    p = RB.TakerParams(min_edge=st["min_edge"], min_edge_h2h=st["min_edge_h2h"],
                       min_edge_by_kind=tuple(SS.parse_map(st["min_edge_by_kind"]).items()))
    return RB.params_for(kind, p).min_edge


def pick_stars(edge, min_edge):
    """1-3 stars from an entry's edge relative to its strategy's minimum edge (>= 2x: 3, >= 1.5x: 2, else 1);
    None without an edge."""
    if edge is None or not min_edge or edge != edge:            # edge != edge: NaN
        return None
    ratio = abs(float(edge)) / float(min_edge) + 1e-9          # 0.15 / 0.10 is 1.4999...: 2 stars
    return next(n for lo, n in STARS if ratio >= lo)


def signal_stars(signal, profile=None):
    """A buy signal's stars, against the minimum edge of the strategy that made it: the blend member named in
    detail.member, else the account's profile. None for exits, quotes and fills."""
    from racinglines.pipelines import profiles as PF
    if signal.get("action") != "buy":
        return None
    code = (signal.get("detail") or {}).get("member")
    member = PF.PROFILES.get(code) if code else None
    settings = member["settings"] if member and not PF.is_combo(member) else (profile or {}).get("settings")
    return pick_stars(signal.get("edge"), _min_edge(settings, signal.get("kind")))


def basic_signal(signal, profile=None):
    """A signal as a basic account sees it: named BASIC_NAME, no strategy, member or candidate, no fair value or
    edge (detail keeps only followed / backfill / venue), with `stars` (signal_stars, computed here)."""
    out = {k: v for k, v in signal.items() if k not in BASIC_HIDDEN}
    out.update(profile=BASIC_NAME, strategy="taker", stars=signal_stars(signal, profile))
    out["detail"] = {k: v for k, v in (signal.get("detail") or {}).items() if k in BASIC_DETAIL_KEEP}
    return out


def basic_position(position):
    """A paper position as a basic account sees it: no candidate (which strategy held it)."""
    return {k: v for k, v in position.items() if k != "candidate_id"}


def basic_row(row):
    """A track-record or phase row (story.track_record, story.phases) for a basic account: the strategy renamed."""
    if row.get("profile") == "Private book":
        return dict(row)
    return dict(row, profile=BASIC_NAME, strategy="taker")


def basic_view_profile(profile):
    """The profile dict the pages show a basic account: BASIC_NAME, a taker, its follow rate and bankroll only."""
    if profile is None:
        return None
    if is_alpha_style(profile):                     # a style picked by name (alpha): its name is all it shows
        return dict(name=profile["name"], strategy="taker", basic=True,
                    **{k: profile[k] for k in ("follow_rate", "bankroll") if k in profile})
    return dict(name=BASIC_NAME, strategy="taker", basic=True,
                **{k: profile[k] for k in ("follow_rate", "bankroll") if k in profile})
