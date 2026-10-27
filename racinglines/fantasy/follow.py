"""
Follow a strategy: the one function behind /strategy/pick (web/strategy_pick.pick_save) and the MCP tools
fantasy_strategy and fantasy_follow (mcp/tools.py), so a pick made from either writes the same profile row (owner,
2026-10-14: every surface; reports/2026-10-14-follow-strategy-roadmap.md).

    options(user)            the five alpha style cards (profiles.ALPHA_STYLES, in ALPHA_ORDER)
    current(conn, user)      the profile the account follows and the race-weekend lock, if one is on
    follow(conn, user, code) follow a style (or a classic code, see below), or stop with code "none"
    venues_of / save_venues  the exchanges the account opts its followed strategy into (VENUES, users.prefs)

The gates, in the page's order and with its words: demo accounts keep their strategy, only F1 on Polymarket (F1_SLOT,
the one slot the VM's signal timer runs), no change from lights out until the window closes a day later
(open_weekend), sizing in
profiles.SIZINGS with a Kelly fraction in KELLY_RANGE on its 5% step, a code the account may pick, and
roles.allowed_profile last. A refusal raises FollowError(reason, message); nothing is written then.

Which codes: under the alpha layout (web/layout.alpha) the five styles and "none", plus, for an admin, every code
the classic page offers (classic_codes); with the alpha layout off, the classic page's codes. The MCP tool always
asks for the alpha set (alpha=True).

`user` is dict(id, username, role) with the role a request runs as (web/layout.role: basic reads as pro under the
alpha layout), as web/app.py puts it on request.state.user.
"""

import json
import time

import pandas as pd
from sqlalchemy import text

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

# The exchanges a followed strategy can run on (owner, 2026-10-21: the follow popup's checkboxes). Only Polymarket runs
# today (F1_SLOT); Kalshi and ProphetX are opt-ins kept in users.prefs[PREF_VENUES], so an account that ticked one is
# ready when that slot opens. ProphetX is an admin-only venue until it goes public. (key, label, live, note)
VENUES = (("polymarket", "Polymarket", True, ""),
          ("kalshi", "Kalshi", False, "coming soon"),
          ("prophetx", "ProphetX", False, "coming soon"))
PREF_VENUES = "follow_venues"

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
    for code in PF.followable():
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
        return set(PF.followable()) | (classic_codes(user) if R.canonical(user["role"]) == "admin" else set())
    return classic_codes(user)


# --- the race-weekend lock ------------------------------------------------------------------------------------------

_SCHED = {}                                 # year -> (loaded at, schedule): the page reads it, so cache it an hour


def _schedule(year):
    from racinglines.pipelines import weekend_sweep as WS
    hit = _SCHED.get(year)
    if hit is None or time.time() - hit[0] > 3600:
        hit = _SCHED[year] = (time.time(), WS.schedule(year))
    return hit[1]


def locks_file():
    """Where the admin's lock overrides live (/admin/locks): a JSON file under the data folder, written only by that
    page, so no migration and a deploy never touches it."""
    from racinglines import paths
    return paths.DATA / "fantasy" / "pick_locks.json"


def lock_overrides():
    """{event_key: dict(closes, reopens, off)} as the admin saved them (/admin/locks; owner, 2026-10-21: "UI for the
    admin to override/ edit upcoming locks"); closes/reopens are naive-UTC ISO strings or None (the default), off
    True: no lock at all for that weekend. {} without the file or if it can't be read."""
    import json
    try:
        return json.loads(locks_file().read_text())
    except (OSError, ValueError):
        return {}


def save_lock_override(event_key, closes=None, reopens_at=None, off=False):
    """Set (or, with nothing given, clear) one weekend's override; -> the overrides saved."""
    import json
    o = lock_overrides()
    row = dict(closes=_iso(closes), reopens=_iso(reopens_at), off=bool(off))
    if row == dict(closes=None, reopens=None, off=False):
        o.pop(event_key, None)
    else:
        o[event_key] = row
    f = locks_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(o, indent=1, sort_keys=True))
    tmp.replace(f)
    return o


def _iso(t):
    """A time as naive-UTC ISO (None for none): the form lock_overrides stores."""
    if t in (None, ""):
        return None
    t = pd.Timestamp(t)
    return (t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t).isoformat()


def open_weekend(now=None, schedule=None):
    """The F1 weekend whose picks are locked at `now` (naive UTC; default now), else None. The signal engine replays a
    weekend's stages so far with the profile's current settings (pipelines/signals.compute); a pick made inside the
    weekend carries `since` (its time), and the engine trades only the stages priced after it for that player
    (signals.for_player), so a pick never bets on a session that already happened. Picks therefore stay open right up
    to lights out (owner, 2026-10-21: "right up to lights out ... just no betting on events that already happened")
    and close from cutoff(w) (the race start) until reopens(w) (race start + signals.WEEKEND_TAIL), unless the admin
    moved either time or turned the weekend's lock off (/admin/locks, lock_overrides).
    No schedule (FastF1 unreachable and nothing saved): None, logged; the engine can't run then either.
    schedule: year -> schedule (default _schedule)."""
    from racinglines.pipelines import signals as SG
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    try:
        _, w = SG.pick_event((schedule or _schedule)(now.year), now)
    except Exception as ex:                         # noqa: BLE001
        print(f"strategy_pick: no F1 schedule for {now.year}: {ex}", flush=True)
        return None
    if w is None or lock_overrides().get(w.get("event_key"), {}).get("off"):
        return None
    return w if cutoff(w) <= now < reopens(w) else None


def cutoff(w):
    """When a weekend's picks close: lights out (its race start), or the admin's time (lock_overrides)."""
    t = lock_overrides().get(w.get("event_key"), {}).get("closes")
    return pd.Timestamp(t) if t else w["race_start"]


def default_reopens(w):
    """When a weekend's lock lifts by default: race start + signals.WEEKEND_TAIL."""
    from racinglines.pipelines import signals as SG
    return w["race_start"] + SG.WEEKEND_TAIL


def reopens(w):
    """When a weekend's lock lifts: default_reopens, or the admin's time (lock_overrides)."""
    t = lock_overrides().get(w.get("event_key"), {}).get("reopens")
    return pd.Timestamp(t) if t else default_reopens(w)


def lock_note(w):
    return (f"Strategy picks closed for {w['name']} at lights out: they open again about a day after the race "
            f"({reopens(w):%a %d %b, %H:%M} UTC).")


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
    if new is not None:                             # since: the engine trades only stages priced after the pick
        new = dict(new, base=code, since=(pd.Timestamp(now) if now is not None
                                          else pd.Timestamp.now(tz="UTC").tz_localize(None)).isoformat())
    if new is not None and size is not None:        # an alpha style: its name and the player's sizing
        new = dict(new, name=style_name(code, bool(new.get("adjusted"))), sizing=size[0], kelly=size[1])
    if not R.allowed_profile(user["role"], new, user_id=user["id"]):
        raise FollowError("not_allowed", "that strategy isn't open to this account.")
    old = PF.of_user(conn, user["id"])
    new = kept_since(old, new)
    carry_trades(conn, user["id"], old, new)
    PF.assign(conn, user["id"], new)
    return new


def carry_trades(conn, user_id, old, new):
    """A switch to another strategy (another candidate) mid-weekend: the old strategy's followed trades in every
    event the player still has open Polymarket positions in move to the new strategy's candidate, and the old rows
    are dropped, so the next pass rebuilds them as the new strategy's kept positions (signals.kept_positions) and
    they settle with it. Without this the old rows stayed open forever: nothing else settles a strategy nobody runs
    (Mexico spec rule 4: a change "never deletes trades already made"). A blend, a stop or the same candidate: no-op.
    -> the event keys moved"""
    a, b = (old or {}).get("candidate_id"), (new or {}).get("candidate_id")
    if a is None or b is None or a == b:
        return []
    evs = conn.execute(text("""SELECT DISTINCT event_key FROM paper_positions WHERE user_id = :u AND candidate_id = :a
                               AND venue = 'polymarket' AND outcome IS NULL AND (yes_shares <> 0 OR no_shares <> 0)"""),
                       dict(u=user_id, a=a)).scalars().all()
    if not evs:
        return []
    conn.execute(text("""UPDATE strategy_signals s SET candidate_id = :b
                         WHERE s.user_id = :u AND s.candidate_id = :a AND s.event_key = ANY(:e)
                           AND s.detail->>'followed' = 'true' AND coalesce(s.detail->>'venue', 'polymarket') = 'polymarket'
                           AND NOT EXISTS (SELECT 1 FROM strategy_signals x WHERE x.user_id = s.user_id
                                           AND x.candidate_id = :b AND x.market_key = s.market_key
                                           AND x.dedupe IS NOT DISTINCT FROM s.dedupe AND x.action = s.action
                                           AND x.side IS NOT DISTINCT FROM s.side)"""),
                 dict(u=user_id, a=a, b=b, e=list(evs)))
    conn.execute(text("""DELETE FROM paper_positions WHERE user_id = :u AND candidate_id = :a AND venue = 'polymarket'
                         AND event_key = ANY(:e)"""), dict(u=user_id, a=a, e=list(evs)))
    return list(evs)


def kept_since(old, new):
    """new, with old's `since` when new is the same strategy saved again (same code, settings and sizing): an
    identical re-save never moves the player's pick time, so it never closes a stage that was already open (Mexico
    spec rule 4, owner 2026-10-25: whycanti's re-save at 05:41Z Sunday reset `since`). A changed strategy keeps
    the new time: it trades from the next stage on, and the trades made before it stay (signals.kept_positions)."""
    if not old or not new or not old.get("since"):
        return new
    same = {k: v for k, v in old.items() if k != "since"} == {k: v for k, v in new.items() if k != "since"}
    return dict(new, since=old["since"]) if same else new


ASSIGN_LEAD = pd.Timedelta(minutes=1)       # an assigned pick counts this long before the pre-race stage (strict rule)


def auto_assign(conn, w, now, seed=None, echo=print):
    """Mexico spec rule 3 (owner, 2026-10-26 01:02Z): once the weekend's pre-race stage (core/stages.pre_race) has
    come and until lights out, every fantasy player with no strategy is assigned one of PF.followable() at random
    (deterministic per weekend and player), picked ASSIGN_LEAD before the pre-race stage, so it trades that stage
    only. Skipped: a player already following one, one who joined (their grant) after the pre-race stage, and one
    who stopped their strategy during this weekend (a strategy_pick audit row since signals opened). Each
    assignment writes a strategy_pick audit row (via=auto) and one data_changes note per pass. -> [(user_id,
    username, code)]"""
    from racinglines.fantasy import season_demo as SD
    from racinglines.pipelines import signals as SG
    pre = (w or {}).get("pre_race")
    now = pd.Timestamp(now)
    if pre is None or not (pre[1] <= now < cutoff(w)):
        return []
    codes = PF.followable()
    opened = w["race_start"] - SG.WEEKEND_LEAD
    out = []
    for uid, name in SD.players(conn):
        if PF.of_user(conn, uid) is not None:
            continue
        joined = conn.execute(text("""SELECT min(created_at) FROM accounts.fantasy_ledger
                                      WHERE user_id = :u AND ref_table IS NULL"""), dict(u=uid)).scalar()
        if joined is None or pd.Timestamp(joined).tz_convert("UTC").tz_localize(None) >= pre[1]:
            continue
        if conn.execute(text("""SELECT EXISTS (SELECT 1 FROM activity_log WHERE user_id = :u
                                AND action = 'strategy_pick' AND ts >= CAST(:t AS timestamp) AT TIME ZONE 'UTC')"""),
                        dict(u=uid, t=opened.isoformat())).scalar():
            continue
        user = dict(conn.execute(text("SELECT id, username, role FROM users WHERE id = :u"), dict(u=uid)).mappings().one())
        code = codes[int(SD._u(seed or w["event_key"], uid) * len(codes))]
        new = follow(conn, user, code, alpha=True, weekend=None, now=pre[1] - ASSIGN_LEAD)
        conn.execute(text("""INSERT INTO activity_log (user_id, username, role, action, detail)
                             VALUES (:u, :n, :r, 'strategy_pick', CAST(:d AS jsonb))"""),
                     dict(u=uid, n=name, r=user["role"], d=json.dumps(dict(old=None, new=new["name"], via="auto",
                                                                             event_key=w["event_key"]))))
        out.append((uid, name, code))
        echo(f"  auto-assigned {name}: {code} ({new['name']})")
    if out:
        conn.execute(text("""INSERT INTO data_changes (kind, sport, summary, detail, by)
                             VALUES ('note', 'f1', :s, CAST(:d AS jsonb), 'signals timer')"""),
                     dict(s=f"{w['event_key']} pre-race: {len(out)} fantasy players with no strategy assigned one "
                            f"at random (fantasy/follow.auto_assign, Mexico spec rule 3)",
                          d=json.dumps(dict(players=[dict(user_id=u, username=n, code=c) for u, n, c in out]))))
    return out


def venues_of(conn, user_id):
    """The exchanges the account opted into: [key], Polymarket always first (the one that runs)."""
    from sqlalchemy import text
    got = conn.execute(text("SELECT prefs->CAST(:k AS text) FROM users WHERE id = :u"), dict(k=PREF_VENUES, u=user_id)).scalar()
    return ["polymarket"] + [v for v, *_ in VENUES[1:] if v in (got or [])]


def save_venues(conn, user_id, venues):
    """Store the account's opt-ins (unknown keys dropped, Polymarket always on) in users.prefs; -> the list saved."""
    import json

    from sqlalchemy import text
    keep = ["polymarket"] + [v for v, *_ in VENUES[1:] if v in set(venues or ())]
    conn.execute(text("""UPDATE users SET prefs = coalesce(prefs, '{}'::jsonb) || jsonb_build_object(CAST(:k AS text), CAST(:v AS jsonb))
                         WHERE id = :u"""), dict(k=PREF_VENUES, v=json.dumps(keep), u=user_id))
    return keep
