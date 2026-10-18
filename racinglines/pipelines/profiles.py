"""
Strategy profiles: a Lab candidate (model x strategy x settings) assigned to a user, whose live paper
signals the signal engine computes (racinglines/pipelines/signals.py).

The two recommended by the params-4h cloud search (data/runs/search/params-4h/REPORT.md, section 4):

    A  core taker: the update taker on gridq+pretrain+reset, edges of 10 pts or more (5 on head-to-head),
       no pre-weekend stage.                                                    (candidate #09 + min_edge_h2h)
    C  maker sleeve: the conservative maker on gbm, 25-share quotes, out of markets where the model
       disagrees with the price by more than 5 pts.                              (candidate #24)

    K  Kalshi maker: the conservative maker tuned on Kalshi's own 2025-26 tape (docs/kalshi-history.md,
       sweeps/kalshi-maker-k.toml); the settings below are the sweep's winner under the params-4h
       held-out rule. The candidate carries venue = "kalshi": it is judged on and quotes Kalshi.

    racinglines strategy profiles f1                      # list, and create A / C / K as Lab candidates if missing
    racinglines strategy profiles f1 --assign-demo        # demo taker -> A, demo maker -> C
    racinglines strategy profiles f1 --monitors           # lab-<id>: one account per Lab taker candidate (backup first)
    racinglines strategy profiles f1 --assign-demo --venue kalshi   # demo maker -> K, as its Kalshi profile (explicit;
                                                           # nothing reads it until the signal engine runs Kalshi)

    T1..T10  the taker shortlist (TAKER_TOP, ranked; source "taker-resweep"): what pro accounts pick from, and
             the pool a basic account's three picks are drawn from (web/roles.basic_members).
    TB       a blend (COMBOS): members [(code, weight)] instead of one strategy; each member runs as itself with
             its stakes scaled by its weight (signals.compute_all). Not a Lab candidate.

A user's profile lives in users.prefs["strategy_profile"] with the full settings, so deleting the
candidate doesn't break the assignment. A Kalshi profile lives beside it in prefs["strategy_profile_kalshi"]
(PREF_KALSHI), so the Polymarket profile and everything reading it are untouched.
"""

import json

from sqlalchemy import text

from racinglines.pipelines import sweep_settings as SS

LIVE_STAGES = ["after FP1", "after FP2", "after FP3", "after SQ", "after Sprint", "after Quali"]
K_SETTINGS = {"variant": "gbm", "max_disagree": 0.10, "size": 25, "maker_min_volume_24h": 400, "venue": "kalshi"}
K_WHY = ("kalshi-maker-k search (full grid + 16k confirmations), conservative maker on Kalshi's tape with its maker "
         "fee, 2c quotes, only in markets with $400+ traded in the prior 24 h. Robust in both seasons (held-out rule, "
         "maker noise floor 350) and confirmed at 16k sims: 2026 +490 (Sharpe 0.62, max DD 499; +1,360 vs Kalshi "
         "baseline, +609 vs C on Kalshi; 16k +453), 2025 +659 (1.07, 504; +846 vs baseline, +150 vs C; 16k +741).")
PROFILES = {
    "A": dict(name="A · core taker (update)", strategy="update",
              settings={"variant": "gridq+pretrain+reset", "min_edge": 0.10, "min_edge_h2h": 0.05,
                        "taker_stages": LIVE_STAGES},
              why="params-4h #09 + head-to-head at 5 pts. Robust in both seasons: 2026 +1,232 (Sharpe 1.53, "
                  "max DD 249), 2025 +1,237 (1.35, 306), both without the h2h change; noise range "
                  "+1,270 ± 67 / +1,020 ± 149. defaults-2026-10 (16 + 23 weekends, 16k): 2026 +1,376 (1.27, 249), 2025 +1,432 (1.55, 233)."),
    "C": dict(name="C · maker sleeve (gbm)", strategy="maker",
              settings={"variant": "gbm", "max_disagree": 0.05, "size": 25},
              why="params-4h #24. Profitable in both seasons with the smallest drawdowns: 2026 +653 "
                  "(Sharpe 1.78, max DD 279), 2025 +835 (1.63, 176); noise range +717 ± 72 / +881 ± 63."),
    "K": dict(name="K · Kalshi maker", strategy="maker", venue="kalshi",
              settings=K_SETTINGS,
              why=K_WHY),
}

# The taker shortlist from the taker re-sweep (source "taker-resweep"): pro accounts pick any of these; a basic account
# gets picks from BASIC_PICKS of them, drawn per account (web/roles.basic_members). Data only: reorder TAKER_TOP or
# rewrite a `why` here and nothing else moves. Numbers: data/runs/search/taker-resweep/REPORT.md.
TAKER_SOURCE = "taker-resweep"
_T1 = {"variant": "gridq+pretrain+reset", "min_edge": 0.10, "min_edge_h2h": 0.05, "taker_stages": LIVE_STAGES}
_T7 = dict(_T1, coherence_tol_by_kind="race_podium=0.35")
TAKER_PROFILES = {
    "T1": dict(name="T1 · update, core (A's settings)", strategy="update", settings=dict(_T1),
               why="taker-resweep rank 2 (= profile A). Robust: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +1,376 (Sharpe 1.27, max DD 249), 2025 +1,432 (1.55, 233)."),
    "T2": dict(name="T2 · update, every stage", strategy="update",
               settings={"variant": "gridq+pretrain+reset", "min_edge": 0.10, "min_edge_h2h": 0.05},
               why="taker-resweep rank 3: A with pre-weekend entry. Robust, not better than A: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +1,244 (1.11, 289), 2025 +1,400 (1.53, 239)."),
    "T3": dict(name="T3 · update, gridq+pretrain", strategy="update",
               settings={"variant": "gridq+pretrain", "min_edge": 0.10},
               why="taker-resweep rank 4: gridq+pretrain, 10-pt edge. Robust: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +983 (0.86, 287), 2025 +786 (0.83, 446); 2026 rests on one weekend."),
    "T4": dict(name="T4 · update, gbm", strategy="update", settings={"variant": "gbm", "min_edge": 0.10},
               why="taker-resweep rank 5: gbm, 10-pt edge. Robust: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +659 (0.72, 406), 2025 +841 (0.79, 448)."),
    "T5": dict(name="T5 · update, 8-pt edges", strategy="update",
               settings={"variant": "gridq+pretrain+reset", "min_edge": 0.08},
               why="taker-resweep rank 6: 8-pt edge, more trades. Robust: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +860 (0.79, 399), 2025 +816 (0.81, 412)."),
    "T6": dict(name="T6 · update, $200 volume floor", strategy="update", settings=dict(_T1, min_volume_24h=200),
               why="taker-resweep rank 8: A in markets with $200+ traded in 24 h. Robust, smallest drawdowns: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +689 (0.72, 260), 2025 +608 (0.78, 225)."),
    "T7": dict(name="T7 · update, loose podium coherence", strategy="update", settings=dict(_T7),
               why="taker-resweep rank 1: A with podium coherence 0.35. Robust but within noise of A: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +1,258 (1.17, 249), 2025 +1,456 (1.58, 213)."),
    "T8": dict(name="T8 · stage-aware (early, $25 volume floor)", strategy="early", settings=dict(_T1, min_volume_24h=25),
               why="defaults-2026-10: A's settings on the stage-aware taker at a $25 volume floor. 2026-led: 16k 2026 +3,145 (1.65, 231), 2025 +757 (1.36, 424); +438 / +211 over the $50 floor, +473 / +233 and +475 / +224 at seeds 1 and 2 (seedcheck-2026-10)."),
    "T9": dict(name="T9 · stage-aware, loose podium coherence", strategy="early", settings=dict(_T7),
               why="taker-resweep rank 9: T8 with podium coherence 0.35. 2026-led: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +2,794 (1.62, 214), 2025 +570 (1.00, 412)."),
    "T10": dict(name="T10 · update, baseline model", strategy="update",
                settings={"variant": "baseline", "min_edge": 0.10},
                why="taker-resweep rank 7: the baseline model at a 10-pt edge. Robust: defaults-2026-10 (16 + 23 weekends, 16k): 2026 +728 (0.55, 504), 2025 +757 (0.71, 656)."),
}
TAKER_TOP = ("T7", "T1", "T2", "T3", "T4", "T5", "T10", "T6", "T9", "T8")     # ranked, best first (worse season per weekend, 16k)
# The alpha layout's five styles (web/layout.alpha, web/strategy_pick.py; owner, 2026-10-12 19:23-19:25Z: "down select
# from strategies we have previously validated ... human readable names / descriptions. For a first-timer ... variety
# rather than best performance"; library/process/2026-10-12-alpha-lab-strategies.md). Users see the label, the blurb
# and the backtest, never the code or the `why`. `backtest` is (2026, 2025) season P&L in dollars at the default sizing,
# copied from each code's `why` above (tests/test_alpha_styles.py checks they match); `in_sample`: tuned on the same
# seasons it is scored on (web/edge.py flags `early`). Data only: reorder ALPHA_ORDER or reword a blurb here.
ALPHA_STYLES = {
    "T1": dict(label="Steady", backtest=(1376, 1432),
               blurb="Buys only when our price beats the market by 10 points (5 on head-to-heads), during the weekend's sessions."),
    "T2": dict(label="Head start", backtest=(1244, 1400),
               blurb="Steady, but it may also buy before the weekend begins."),
    "T8": dict(label="Early bird", backtest=(3145, 757), in_sample=True,
               blurb="Trades practice and the sprint, then stops before final practice and qualifying. It's busier and swings more."),
    "T6": dict(label="Big markets only", backtest=(689, 608),
               blurb="Steady, but only where $200+ traded that day. It has the smallest drawdowns."),
    "T5": dict(label="Hair trigger", backtest=(860, 816),
               blurb="Buys on an 8-point edge instead of 10, so it trades a lot more."),
}
ALPHA_ORDER = ("T1", "T2", "T8", "T6", "T5")
# Blends (combos): `members` [(code, weight)] instead of one strategy. Each member runs as itself with its stakes
# scaled by its weight (signals.compute_all); its signals carry detail.member. A blend is not a Lab candidate.
COMBOS = {
    "TB": dict(name="TB · blended taker (update + stage-aware)", members=[("T1", 0.5), ("T8", 0.5)],
               why="taker-resweep recommended blend: half T1, half T8 (same model, two entry rules). defaults-2026-10 (16 + 23 weekends, 16k): 2026 +2,261, 2025 +1,094 with T8 at its $25 floor (seedcheck-2026-10, default seed). Chosen on both seasons, so no held-out season."),
}
PROFILES.update(TAKER_PROFILES)
PROFILES.update(COMBOS)
DEMO = {"taker": "A", "maker": "C"}
DEMO_KALSHI = {"maker": "K"}        # the demo maker's Kalshi profile, assigned only by --assign-demo --venue kalshi
# Share of recommendations each demo account follows, by the kind of strategy it runs (code only, not in the UI):
# the basic demo (`taker`) takes about a third of its taker picks; the pro demo (`maker`) fills every pick of a taker
# strategy when it runs one (its maker strategy has no follow rate). Unset = follows everything.
DEMO_FOLLOW = {"taker": {"taker": 0.33}, "maker": {"taker": 1.0}}
DEMO_BANKROLL = {"maker": 10_000.0, "taker": 1_000.0}   # starting bankroll ($), from BANKROLL_SINCE
BANKROLL_SINCE = "2025-01-01"
# the demo maker's saved Edge Finder: the setups it ran, newest first (M1 = the baseline maker, also the benchmark)
DEMO_EDGE_FINDER = {"maker": [("C", "maker"), ("M3", "maker"), ("M2", "maker"), ("M1", "maker")]}
DEMO_EDGE_YEAR = 2026
# Maker strategies for users. Closed for the Alpha on 2026-10-11 22:29Z ("For the Alpha Fantasy Launch we can remove
# maker strategies completely"), open again on 2026-10-12 05:11Z (owner: "include maker strategies, we will see how it
# goes"). Closed means: no account but admin can be assigned a maker profile (web/roles.allowed_profile), the picker
# doesn't offer one (open_codes), the web pages hide maker strategies (web/edge.SHOWN_KEYS) and the signal engine skips
# any maker profile outside the demo accounts (signals.run_all). Nothing is deleted either way.
MAKERS_OPEN = True


def is_maker(profile):
    """A maker profile (a PROFILES entry or an assigned profile): one strategy that isn't a taker mode. A blend is a
    maker if any member is."""
    if not profile:
        return False
    if is_combo(profile):
        return any(is_maker(PROFILES.get(m[0]) if isinstance(m, (list, tuple)) else m.get("member") or PROFILES.get(m["code"]))
                   for m in profile["members"])
    from racinglines.pipelines import weekend_sweep as WS
    return profile.get("strategy", "update") not in WS.TAKER_MODES


def open_codes(codes):
    """The codes users may pick from, in order: makers drop out while MAKERS_OPEN is off."""
    return tuple(c for c in codes if MAKERS_OPEN or not is_maker(PROFILES.get(c)))


PREF = "strategy_profile"
PREF_KALSHI = "strategy_profile_kalshi"


def is_combo(profile):
    """A blend: a profile (a PROFILES entry or an assigned profile dict) with `members` instead of one strategy."""
    return bool(profile and profile.get("members"))


def follow_rate(username, strategy):
    """The demo follow rate for this account running this strategy (DEMO_FOLLOW), or None."""
    from racinglines.pipelines import weekend_sweep as WS
    kind = "taker" if strategy in WS.TAKER_MODES else "maker"
    return (DEMO_FOLLOW.get(username) or {}).get(kind)


# A fantasy player's sizing choice for the strategy they follow (owner, 2026-10-12 19:52Z and 19:53Z: "expose some
# dynamic sizing options like half or quarter kelly"): `sizing` = "standard" (the bankroll-scaled linear rule,
# fantasy/sizing.py; the default) or "kelly" with `kelly` = the fraction, 0.10 to 1.00. Stored on the user's assigned
# profile (users.prefs, pref()); read only by the live signal pass for registered players (signals.for_player), so
# the demo accounts, backtests and sweeps never see it. No profile's own settings change.
SIZINGS = ("standard", "kelly")
KELLY_RANGE = (0.10, 1.00)


def kelly_value(sizing, kelly):
    """The Kelly fraction a (sizing, kelly) choice means, or None for standard sizing: anything but sizing "kelly"
    with a number from 0.10 to 1.00 is standard."""
    if sizing != "kelly" or isinstance(kelly, bool):
        return None
    try:
        k = float(kelly)
    except (TypeError, ValueError):
        return None
    return k if KELLY_RANGE[0] <= k <= KELLY_RANGE[1] else None


def sizing_of(profile):
    """A profile's Kelly fraction (its `sizing` = "kelly" and `kelly` in KELLY_RANGE), or None = standard sizing."""
    profile = profile or {}
    return kelly_value(profile.get("sizing"), profile.get("kelly"))


def set_sizing(conn, user_id, sizing, kelly=None, venue="polymarket", sport=None):
    """Save a user's sizing choice on their assigned profile (the picker's form fields `sizing` = "standard" | "kelly"
    and `kelly` = 0.10 to 1.00). Raises ValueError for anything else, or when the user has no profile for that venue
    and sport."""
    sport = sport or DEFAULT_SPORT
    if sizing not in SIZINGS:
        raise ValueError(f"sizing {sizing!r}: one of {', '.join(SIZINGS)}")
    k = kelly_value(sizing, kelly)
    if sizing == "kelly" and k is None:
        raise ValueError(f"kelly {kelly!r}: a fraction from {KELLY_RANGE[0]:.2f} to {KELLY_RANGE[1]:.2f}")
    prof = of_user(conn, user_id, venue=venue, sport=sport)
    if prof is None:
        raise ValueError("no strategy profile to size")
    prof = {k_: v for k_, v in prof.items() if k_ != "kelly"}
    assign(conn, user_id, dict(prof, sizing=sizing, **({"kelly": k} if k is not None else {})), venue=venue,
           sport=sport)


DEFAULT_SPORT = "f1"                    # a profile with no `sport` (every one before other sports had signals)


def pref(venue="polymarket", sport=DEFAULT_SPORT):
    """The users.prefs key a sport and venue's profile lives under: F1 as before (Polymarket: the profile itself,
    another venue strategy_profile_<venue>); another sport strategy_profile_<sport>_<venue>
    (pipelines/sport_signals.py)."""
    if sport != DEFAULT_SPORT:
        return f"{PREF}_{sport}_{venue}"
    return PREF if venue == "polymarket" else f"{PREF}_{venue}"

# The demo accounts' track record (pipelines/demo_history.py): backtest replays of real weekends, shown as
# what each account ran. Both accounts follow the same walk-forward rule (pipelines/story.py: switch to the
# Edge Finder's P&L leader mid-season past a margin, commit to the most consistent setup within the noise
# floor before a season), each on its own pool of setups fixed in advance. The maker tried three setups in
# 2025 and has run C since the first race of 2026; the taker started on the defaults and moved to the
# grid-aware model with the reset at a 10-pt edge after 2025 round 8, and kept it (taker-resweep report,
# section 6: the rule never picked a blend).
HISTORY_PROFILES = {
    "M1": dict(name="M1 · launch maker (baseline)", strategy="maker", settings={"variant": "baseline"},
               why="2025 rounds 1-8: no market history yet, so the defaults: the conservative maker on the "
                   "baseline model."),
    "M2": dict(name="M2 · grid-aware maker, 10-pt filter", strategy="maker",
               settings={"variant": "gridq+pretrain", "max_disagree": 0.10},
               why="2025 rounds 9-16: the Edge Finder's P&L leader over the season's first 7 weekends "
                   "(+$697): the grid-aware model, out of markets 10+ pts from the model."),
    "M3": dict(name="M3 · gbm maker, tight filter", strategy="maker",
               settings={"variant": "gbm", "max_disagree": 0.07},
               why="2025 rounds 17-24: the Edge Finder's P&L leader over 15 weekends (+$1,134): the "
                   "gradient-boosted model, quoting only where it agrees with the market within 7 pts."),
    "TW1": dict(name="TW1 · launch taker (defaults)", strategy="update", settings={"variant": "baseline"},
                why="2025 rounds 1-8: no market history yet, so the defaults: the update taker on the baseline "
                    "model at a 5-pt edge."),
    "TW2": dict(name="TW2 · grid-aware taker, 10-pt edge", strategy="update",
                settings={"variant": "gridq+pretrain+reset", "min_edge": 0.10},
                why="From 2025 round 9: the Edge Finder's P&L leader over the season's first 8 weekends (+$327, "
                    "update taker); still the leader after 16 (+$1,216) and the most consistent setup within "
                    "the $400 noise floor of the leader before 2026, so kept."),
}
HISTORY = {   # username -> [(profile code, year, first round, last round)]
    "maker": [("M1", 2025, 1, 8), ("M2", 2025, 9, 16), ("M3", 2025, 17, 24), ("C", 2026, 1, 99)],
    "taker": [("TW1", 2025, 1, 8), ("TW2", 2025, 9, 99), ("TW2", 2026, 1, 99)],
}


def _candidates(conn):
    return conn.execute(text("""SELECT id, params FROM model_runs WHERE kind = 'candidate' ORDER BY id""")).all()


def ensure_candidates(conn, history=False):
    """Create A, C and K (and with history, M1-M3) as Lab candidates unless a candidate of the same name
    exists. {code: candidate id}."""
    from racinglines.pipelines.search import add_candidate
    have = {p.get("name"): i for i, p in _candidates(conn)}
    out = {}
    for code, pr in {**PROFILES, **(HISTORY_PROFILES if history else {})}.items():
        if is_combo(pr):                                  # a blend isn't a candidate: its members are
            continue
        year, source = (2025, "demo-history") if code in HISTORY_PROFILES else \
            (2026, "kalshi-maker-k" if code == "K" else TAKER_SOURCE if code in TAKER_PROFILES else "params-4h")
        out[code] = have.get(pr["name"]) or add_candidate(
            conn, pr["name"], SS.Settings.from_dict(pr["settings"]), year, pr["strategy"], why=pr["why"],
            source=source, venue=pr.get("venue"))
    return out


def load(conn, ref):
    """A profile from a candidate id, a candidate name, or a code (A, C, M1-M3).
    -> dict(candidate_id, name, strategy, settings[, venue]) (venue: a Kalshi candidate's exchange)."""
    ref = str(ref).strip()
    if is_combo(PROFILES.get(ref.upper())):
        pr = PROFILES[ref.upper()]
        return combo(conn, pr["members"], pr["name"])
    rows = _candidates(conn)
    hit = None
    codes = {**PROFILES, **HISTORY_PROFILES}
    if ref.isdigit():
        hit = next(((i, p) for i, p in rows if i == int(ref)), None)
    else:
        name = codes[ref.upper()]["name"] if ref.upper() in codes else ref
        hit = next(((i, p) for i, p in reversed(rows) if p.get("name") == name), None)
    if hit is None:
        raise ValueError(f"no candidate {ref!r} (racinglines strategy profiles f1 creates A and C)")
    i, p = hit
    sport = p.get("sport") or DEFAULT_SPORT          # another sport's candidate: its own sweep settings
    if sport != DEFAULT_SPORT:
        from racinglines.pipelines import season_sweep as SW
        cls = SW.settings_class(sport)
    else:
        cls = SS.Settings
    return dict(candidate_id=i, name=p["name"], strategy=p["strategy"],
                settings=cls.from_dict(p["settings"], strict=False).to_json(),
                **({"venue": p["venue"]} if p.get("venue") else {}), **({"sport": sport} if sport != DEFAULT_SPORT else {}))


def combo(conn, members, name, **extra):
    """A blend's profile dict: dict(name, strategy, settings, candidate_id=None, members=[dict(code, weight, member)]).
    strategy and settings are the first member's, so every reader that expects one strategy (the Markets page's
    calls, the season P&L line) sees a taker; only the signal engine and the backfill run each member."""
    ids = ensure_candidates(conn)
    ms = [dict(code=code, weight=float(w), member=load(conn, ids[code])) for code, w in members]
    return dict(name=name, strategy=ms[0]["member"]["strategy"], settings=ms[0]["member"]["settings"],
                candidate_id=None, members=ms, **extra)


def assign(conn, user_id, profile, venue="polymarket", sport=DEFAULT_SPORT):
    """Set (profile dict from load) or clear (None) a user's strategy profile; venue="kalshi" sets the
    user's Kalshi profile (pref("kalshi")) and leaves the Polymarket one alone; another sport's profile is stored
    with its `sport` under pref(venue, sport)."""
    key = pref(venue, sport)
    if profile is not None and sport != DEFAULT_SPORT:
        profile = dict(profile, sport=sport)
    if profile is not None and "sizing" not in profile:     # a new strategy keeps the player's sizing choice
        had = sizing_of(of_user(conn, user_id, venue=venue, sport=sport))
        if had is not None:
            profile = dict(profile, sizing="kelly", kelly=had)
    conn.execute(text(f"""UPDATE users SET prefs = CASE WHEN CAST(:v AS jsonb) IS NULL
                            THEN coalesce(prefs, '{{}}'::jsonb) - '{key}'
                            ELSE coalesce(prefs, '{{}}'::jsonb) || jsonb_build_object('{key}', CAST(:v AS jsonb)) END
                          WHERE id = :u"""), dict(u=user_id, v=None if profile is None else json.dumps(profile)))


def of_user(conn, user_id, venue="polymarket", sport=DEFAULT_SPORT):
    p = conn.execute(text(f"SELECT prefs->'{pref(venue, sport)}' FROM users WHERE id = :u"), dict(u=user_id)).scalar()
    return p or None


def assigned(conn, venue="polymarket", sport=DEFAULT_SPORT):
    """[(user_id, username, role, profile)] for every active user with a profile (for that sport, on that venue)."""
    key = pref(venue, sport)
    rows = conn.execute(text(f"""SELECT id, username, role, prefs->'{key}' FROM users
                                WHERE active AND prefs ? '{key}' ORDER BY id""")).all()
    from racinglines.web.roles import canonical
    return [(i, u, canonical(r), p) for i, u, r, p in rows]


MONITOR_PREFIX = "lab-"           # lab-<candidate id>: one hidden account per taker candidate (web/accounts reserves it)


def taker_candidates(conn):
    """[(candidate id, name)] of every F1 taker candidate in the Lab (the strategy is a taker mode; makers and other
    sports' candidates are left out)."""
    from racinglines.pipelines import weekend_sweep as WS
    return [(i, p["name"]) for i, p in _candidates(conn)
            if p.get("strategy") in WS.TAKER_MODES and (p.get("sport") or DEFAULT_SPORT) == DEFAULT_SPORT]


def ensure_monitors(conn):
    """One account per taker candidate, `lab-<id>`, assigned that candidate as its Polymarket profile, so the live
    signal pass paper-trades every Lab taker and each keeps its own track record (owner, 2026-10-17: "takers only",
    from Austin through the end of the year). The accounts can't sign in (a random password nobody holds) and have no
    fantasy grant, so they are off the leaderboard and trade at the strategy's own sizing. Idempotent: an existing
    account keeps its row and is re-assigned its candidate. -> [(username, candidate id, created)]"""
    import secrets
    from racinglines.web import users as U
    out = []
    for cid, name in taker_candidates(conn):
        username = f"{MONITOR_PREFIX}{cid}"
        uid = conn.execute(text("SELECT id FROM users WHERE username = :u"), dict(u=username)).scalar()
        created = uid is None
        if created:
            uid = conn.execute(text("""INSERT INTO users (username, display_name, role, password_hash, active)
                                       VALUES (:u, :d, 'basic', :h, true) RETURNING id"""),
                               dict(u=username, d=f"Lab monitor: {name}"[:120],
                                    h=U.hash_password(secrets.token_urlsafe(32)))).scalar()
        assign(conn, uid, load(conn, cid))
        out.append((username, cid, created))
    return out


def assign_demo(conn, venue="polymarket"):
    """Demo taker -> A, demo maker -> C (their Polymarket profiles, with the demo bankrolls and saved views).
    venue="kalshi": demo maker -> K as its Kalshi profile only; the Polymarket assignments don't move."""
    ids = ensure_candidates(conn)
    done = {}
    if venue != "polymarket":
        for username, code in DEMO_KALSHI.items():
            uid = conn.execute(text("SELECT id FROM users WHERE username = :u"), dict(u=username)).scalar()
            if uid is not None:
                assign(conn, uid, dict(load(conn, ids[code]), venue=venue), venue=venue)
                done[username] = PROFILES[code]["name"]
        return done
    for username, code in DEMO.items():
        uid = conn.execute(text("SELECT id FROM users WHERE username = :u"), dict(u=username)).scalar()
        if uid is not None:
            prof = load(conn, ids[code])
            if follow_rate(username, prof["strategy"]) is not None:
                prof["follow_rate"] = follow_rate(username, prof["strategy"])
            if username in DEMO_BANKROLL:
                prof["bankroll"] = dict(start=DEMO_BANKROLL[username], since=BANKROLL_SINCE)
            assign(conn, uid, prof)
            if username in DEMO_EDGE_FINDER:                      # the account's saved baseline view
                codes = {**PROFILES, **HISTORY_PROFILES}
                refs = []
                for ef_code, strategy in DEMO_EDGE_FINDER[username]:
                    st = SS.Settings.from_dict(codes[ef_code]["settings"])
                    refs.append([st["variant"] if not st.changed().keys() - {"variant"} else "cfg:" + st.key, strategy])
                conn.execute(text("""UPDATE users SET prefs = coalesce(prefs, '{}'::jsonb)
                                        || jsonb_build_object('edge_finder', CAST(:e AS jsonb), 'edge_year', CAST(:y AS int))
                                      WHERE id = :u"""), dict(e=json.dumps(refs), y=DEMO_EDGE_YEAR, u=uid))
            done[username] = PROFILES[code]["name"]
    return done
