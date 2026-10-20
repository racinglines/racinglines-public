"""
Follow a strategy: the one function behind /strategy/pick (web/strategy_pick.pick_save) and the MCP tools
fantasy_strategy and fantasy_follow (mcp/tools.py), so a pick made from either writes the same profile row (owner,
2026-10-14: every surface; reports/2026-10-14-follow-strategy-roadmap.md).

    options(user)            the five alpha style cards (profiles.ALPHA_STYLES, in ALPHA_ORDER)
    current(conn, user)      the profile the account follows and the race-weekend lock, if one is on
    follow(conn, user, code) follow a style (or a classic code, see below), or stop with code "none"

The gates, in the page's order and with its words: demo accounts keep their strategy, only F1 on Polymarket (F1_SLOT,
the one slot the VM's signal timer runs), no change while a weekend's signal window is open (open_weekend), sizing in
profiles.SIZINGS with a Kelly fraction in KELLY_RANGE on its 5% step, a code the account may pick, and
roles.allowed_profile last. A refusal raises FollowError(reason, message); nothing is written then.

Which codes: under the alpha layout (web/layout.alpha) the five styles and "none", plus, for an admin, every code
the classic page offers (classic_codes); with the alpha layout off, the classic page's codes. The MCP tool always
asks for the alpha set (alpha=True).

`user` is dict(id, username, role) with the role a request runs as (web/layout.role: basic reads as pro under the
alpha layout), as web/app.py puts it on request.state.user.
"""

import time

import pandas as pd

from racinglines.pipelines import profiles as PF
from racinglines.pipelines import sweep_settings as SS
from racinglines.web import demo
from racinglines.web import layout as L
from racinglines.web import roles as R
from racinglines.web.ranges import UI_RANGES

F1_SLOT = "f1:polymarket"
PRE = "pre-weekend"
# what a pro account picks from on the classic page, in this order (TAKER_TOP is ranked), maker C included
# (owner, 2026-10-12 05:11Z: "include maker strategies, we will see how it goes")
PRO_CODES = (*PF.TAKER_TOP, "TB", "C")
# The settings a user may move, per kind of strategy: (setting, label, low, high, step, unit), the range from the web
# forms' caps (web/ranges.py: what has been backtested). Probabilities show as points (x100). Anything else stays as
# the strategy has it.
_T = {"taker": (("min_edge", "Minimum edge to take a price", 0.01, "pts"),
                ("min_edge_h2h", "Minimum edge on head-to-head markets", 0.01, "pts"),
                ("min_volume_24h", "Only markets with this much traded in the last 24 h", 50, "$")),
      "maker": (("max_disagree", "Stop quoting where the model and the market differ by more than", 0.01, "pts"),
                ("size", "Shares per quote", 5, "shares"))}
TUNABLE = {k: tuple((n, label, *UI_RANGES[n], step, unit) for n, label, step, unit in v) for k, v in _T.items()}
ALPHA_TWEAKS = ("min_edge", "min_volume_24h")       # the alpha tweak drawer's two sliders, with TUNABLE's caps
# Sizing for a followed strategy (owner, 2026-10-12 19:53Z): Standard (stakes scaled to the player's F$1,000) or Kelly at
# a fraction. The allowed values and their meaning live in profiles (SIZINGS, KELLY_RANGE); the signal engine reads
# them through profiles.sizing_of (fantasy/sizing.py).
SIZINGS = PF.SIZINGS
KELLY = dict(lo=PF.KELLY_RANGE[0], hi=PF.KELLY_RANGE[1], step=0.05, default=0.25)

REASONS = ("demo_account", "slot_not_open", "weekend_locked", "bad_sizing", "bad_code", "not_allowed")
_AUTO = object()                                    # follow(weekend=...): look it up with open_weekend(now)


class FollowError(ValueError):
    """A refused pick: reason (one of REASONS) and the page's own message."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason, self.message = reason, message


# --- the strategies -------------------------------------------------------------------------------------------------

def kind(profile):
    from racinglines.pipelines import weekend_sweep as WS
    return "taker" if profile.get("strategy", "update") in WS.TAKER_MODES else "maker"


def tunables(pr):
    """The sliders and the pre-weekend toggle for one strategy (a PROFILES entry), at its own values."""
    if PF.is_combo(pr):
        return [], None
    st = SS.Settings.from_dict(pr["settings"], strict=False)
    k = kind(pr)
    sl = []
    for name, label, lo, hi, step, unit in TUNABLE[k]:
        v = st[name] if st[name] is not None else st["min_edge"] if name == "min_edge_h2h" else lo
        v = min(max(float(v), lo), hi)
        sl.append(dict(name=name, label=label, lo=lo, hi=hi, step=step, unit=unit, value=v, base=v))
    pre = (PRE in (st["taker_stages"] or ())) if k == "taker" else None
    return sl, pre


def style_name(code, adjusted=False):
    return PF.ALPHA_STYLES[code]["label"] + (" · adjusted" if adjusted else "")


def display_name(profile):
    """What the alpha page calls a running profile: its style's label when it came from one, else its name."""
    if not profile:
        return None
    code = profile.get("base")
    if code in PF.ALPHA_STYLES and not PF.is_combo(profile):
        return style_name(code, bool(profile.get("adjusted")))
    return profile.get("name")


def options(user):
    """The five style cards, in ALPHA_ORDER: [dict(code, label, blurb, backtest, roi, in_sample, kind, sliders)]; kind is
    maker or taker (the card's M/T badge, owner 2026-10-14); sliders are
    the drawer's two at the style's own values, for pro and admin only (a basic account runs a style as tuned). roi:
    each season's backtest P&L as a return on the F$1,000 grant (standard sizing stakes a F$1,000 balance as the
    backtests did); the card shows only that, never dollars (owner, 2026-10-13)."""
    from racinglines.web.accounts import SIGNUP_GRANT
    out = []
    for code in PF.ALPHA_ORDER:
        st = PF.ALPHA_STYLES[code]
        sl = [x for x in tunables(PF.PROFILES[code])[0] if x["name"] in ALPHA_TWEAKS] if not R.is_basic(user) else []
        out.append(dict(code=code, label=st["label"], blurb=st["blurb"], backtest=st["backtest"],
                        roi=tuple(v / SIGNUP_GRANT for v in st["backtest"]), in_sample=bool(st.get("in_sample")),
                        kind=kind(PF.PROFILES[code]), sliders=sl))
    return out


def classic_codes(user):
    """The codes the classic page offers this account: its own draw ("basic") for a basic account, else PRO_CODES."""
    return {"basic"} if R.is_basic(user) else set(PRO_CODES)


def allowed_codes(user, alpha=None):
    """The codes this account may follow ("none", stop, is always allowed): the five styles under the alpha layout (an
    admin also every classic code), else the classic page's."""
    if L.alpha() if alpha is None else alpha:
        return set(PF.ALPHA_ORDER) | (classic_codes(user) if R.canonical(user["role"]) == "admin" else set())
    return classic_codes(user)


# --- the race-weekend lock ------------------------------------------------------------------------------------------

_SCHED = {}                                 # year -> (loaded at, schedule): the page reads it, so cache it an hour


def _schedule(year):
    from racinglines.pipelines import weekend_sweep as WS
    hit = _SCHED.get(year)
    if hit is None or time.time() - hit[0] > 3600:
        hit = _SCHED[year] = (time.time(), WS.schedule(year))
    return hit[1]


def open_weekend(now=None, schedule=None):
    """The F1 weekend whose signal window is open at `now` (naive UTC; default now), else None. The signal engine
    replays a weekend's stages so far with the profile's current settings (pipelines/signals.compute), so a change
    inside the window would rewrite that weekend's earlier entries with hindsight: saves wait until it closes
    (Core review, 2026-10-11). Same window as the engine: signals.in_weekend (race start - WEEKEND_LEAD to + WEEKEND_TAIL).
    No schedule (FastF1 unreachable and nothing saved): None, logged; the engine can't run then either.
    schedule: year -> schedule (default _schedule)."""
    from racinglines.pipelines import signals as SG
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    try:
        _, w = SG.pick_event((schedule or _schedule)(now.year), now)
    except Exception as ex:                         # noqa: BLE001
        print(f"strategy_pick: no F1 schedule for {now.year}: {ex}", flush=True)
        return None
    return w if w is not None and SG.in_weekend(w, now) else None


def reopens(w):
    """When a weekend's lock lifts: race start + signals.WEEKEND_TAIL."""
    from racinglines.pipelines import signals as SG
    return w["race_start"] + SG.WEEKEND_TAIL


def lock_note(w):
    return (f"Strategy changes are paused during {w['name']}: they open again about a day after the race "
            f"({reopens(w):%a %d %b, %H:%M} UTC), so a change never rewrites this weekend's calls.")


# --- sizing and tweaks ----------------------------------------------------------------------------------------------

def check_sizing(sizing=None, kelly=None):
    """(sizing, kelly) validated: sizing in SIZINGS (default standard), kelly a fraction in [0.10, 1.00] (default 0.25),
    kept on its 5% step. Raises FollowError("bad_sizing", the page's message)."""
    sizing = sizing or "standard"
    if sizing not in SIZINGS:
        raise FollowError("bad_sizing", "pick Standard or Kelly sizing.")
    try:
        k = KELLY["default"] if kelly in (None, "") else float(kelly)
    except (TypeError, ValueError):
        raise FollowError("bad_sizing", "the Kelly fraction must be a number.") from None
    if not (KELLY["lo"] - 1e-9 <= k <= KELLY["hi"] + 1e-9):
        raise FollowError("bad_sizing", "the Kelly fraction must be between 10% and 100%.")
    return sizing, round(round(k / KELLY["step"]) * KELLY["step"], 2)


def adjusted(code, tweaks):
    """A strategy's settings with `tweaks` ({setting: value} for its sliders, and "pre_weekend" for its pre-weekend
    toggle, left as the strategy has it when absent; other keys ignored), clamped to TUNABLE and stepped;
    -> (settings, {what moved}), {} when nothing moved. The classic page always passes pre_weekend (its checkbox sends
    nothing when unticked: strategy_pick._classic_form); the alpha drawer and MCP never do, so a style followed with no
    tweak saves as tuned (fixed 2026-10-14: T2 and T5, whose stages include pre-weekend, saved as "· adjusted")."""
    base = PF.PROFILES[code]
    st = dict(base["settings"])
    sl, pre = tunables(base)
    moved = {}
    for s in sl:
        raw = tweaks.get(s["name"])
        if raw in (None, ""):
            continue
        v = min(max(float(raw), s["lo"]), s["hi"])
        v = round(v / s["step"]) * s["step"]
        v = round(v, 4) if s["unit"] == "pts" else float(v)
        if abs(v - s["value"]) > 1e-9:
            st[s["name"]] = v
            moved[s["name"]] = v
    if pre is not None and tweaks.get("pre_weekend") is not None:   # absent: as the strategy has it (the alpha drawer)
        want = tweaks.get("pre_weekend") in (True, "1")
        if want != pre:
            stages = list(SS.Settings.from_dict(base["settings"], strict=False)["taker_stages"] or SS.STAGES)
            stages = [PRE, *[x for x in stages if x != PRE]] if want else [x for x in stages if x != PRE]
            st["taker_stages"] = [x for x in SS.STAGES if x in stages]
            moved["pre_weekend"] = want
    SS.Settings.from_dict(st)                          # strict: a value the engine wouldn't take fails here, not later
    return st, moved


# --- read and follow ------------------------------------------------------------------------------------------------

def summary(profile):
    """A saved profile as the MCP tools show it: dict(base, name, sizing, kelly, adjusted, rolled), or None."""
    if not profile:
        return None
    return dict(base=profile.get("base"), name=display_name(profile), sizing=profile.get("sizing") or "standard",
                kelly=float(profile.get("kelly") or KELLY["default"]), adjusted=profile.get("adjusted") or None,
                rolled=bool(profile.get("rolled")))


def saved_message(new):
    """The page's line after a save (new: the profile follow returned, None for a stop)."""
    if new is None:
        return "Stopped: nothing trades for this account until you pick again."
    return (f"Now running {new['name']}. Its calls start when the next race weekend's signals open, four days before "
            "the race.")


def current(conn, user, now=None, weekend=_AUTO):
    """What the account follows and whether it may change it now: dict(profile, lock, demo); profile is
    dict(base, name, sizing, kelly, adjusted, rolled) or None (nothing follows), lock dict(weekend, reopens, message)
    or None (between weekends)."""
    w = open_weekend(now) if weekend is _AUTO else weekend
    lock = dict(weekend=w["name"], reopens=reopens(w).isoformat(), message=lock_note(w)) if w else None
    return dict(profile=summary(PF.of_user(conn, user["id"])), lock=lock, demo=demo.is_demo(user))


def follow(conn, user, code, sizing="standard", kelly=0.25, tweaks=None, rolled=False, slot=F1_SLOT, alpha=None,
           weekend=_AUTO, now=None):
    """Follow `code` (an alpha style, a classic code where allowed, see the docstring) or stop ("none") on the
    account's F1 Polymarket profile, as /strategy/pick saves it. sizing/kelly apply to an alpha style; tweaks
    ({setting: value}, see adjusted) to a single strategy; rolled marks a tweaked profile as the dice's. alpha: the
    code set (None: the layout's). weekend: the open weekend if the caller looked it up (default open_weekend(now)).
    -> the profile saved, None for a stop. Raises FollowError, before anything is written."""
    if demo.is_demo(user):
        raise FollowError("demo_account", "The demo accounts keep their strategy.")
    if slot != F1_SLOT:                             # only F1 on Polymarket can be picked in the Alpha
        raise FollowError("slot_not_open", "only Formula 1 on Polymarket can be picked for now.")
    w = open_weekend(now) if weekend is _AUTO else weekend
    if w is not None:                               # a pick, a slider, a dice roll or a stop: all wait for the window
        raise FollowError("weekend_locked", lock_note(w))
    alpha = L.alpha() if alpha is None else alpha
    size = check_sizing(sizing, kelly) if alpha and code in PF.ALPHA_STYLES else None
    if code == "none":
        new = None
    elif code not in allowed_codes(user, alpha):
        raise FollowError("bad_code", "pick one of the strategies listed.")
    elif code == "basic":
        new = R.basic_profile(conn, user["id"])
    elif PF.is_combo(PF.PROFILES[code]):
        new = PF.load(conn, code)
    else:
        new = PF.load(conn, PF.ensure_candidates(conn)[code])
        settings, moved = adjusted(code, tweaks or {})
        if moved:
            new = dict(new, name=f"{PF.PROFILES[code]['name']} · adjusted", settings=SS.Settings.from_dict(
                settings, strict=False).to_json(), adjusted=moved, **({"rolled": True} if rolled else {}))
    if new is not None:
        new = dict(new, base=code)
    if new is not None and size is not None:        # an alpha style: its name and the player's sizing
        new = dict(new, name=style_name(code, bool(new.get("adjusted"))), sizing=size[0], kelly=size[1])
    if not R.allowed_profile(user["role"], new, user_id=user["id"]):
        raise FollowError("not_allowed", "that strategy isn't open to this account.")
    PF.assign(conn, user["id"], new)
    return new
