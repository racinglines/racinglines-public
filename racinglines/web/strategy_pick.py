"""
/strategy/pick: an account picks the strategy it paper-trades (owner, 2026-10-11 22:04Z: "no default strategy until they
select one ... defaults + sensible toggles/sliders"). Sign-up assigns none, so until a user picks here nothing trades
for them; this page is the user-facing side of what /admin/users/<id> does (admin.admin_user_profile).

    basic        "Your picks" (roles.basic_profile: its own draw of BASIC_PICKS from the taker shortlist), or none
    pro, admin   the taker shortlist (profiles.TAKER_TOP, ranked), the recommended blend (COMBOS) and maker C (open to
                 users since 2026-10-12 05:11Z), with the few settings worth a slider
                 (TUNABLE) on a single strategy; a blend runs as tuned

The page shows every sport and exchange a strategy could point at (admin.profile_slots; owner, 2026-10-11 22:30Z:
"pointing a strat at a particular sport"), to admins only, and only F1 on Polymarket can be picked: it is the one slot the VM's signal
timer runs (deploy/vm/systemd/racinglines-signals.service), and the library's strategies carry F1's stages and kinds,
so on another sport they would fail or change (library/core/2026-10-11-strategies-per-sport.md). The others show as
"coming soon" until Core adds per-sport strategies.

Under the alpha layout (web/layout.alpha; owner, 2026-10-12 19:23-19:54Z; library/process/2026-10-12-alpha-lab-strategies.md)
the page is the five style cards instead (profiles.ALPHA_STYLES, by name, for basic and pro alike; no makers, blend or
other codes, and a save outside the five is refused unless the account is an admin), a sizing choice (Standard, or
Kelly at a fraction), and one closed "tweak" drawer with two sliders (edge needed, market volume) and the dice for pro
and admin; any tweak is untested, scored live from the next race weekend. alpha_picker() is the one switch: with it off
the page is the classic one above, unchanged.

The alpha page and the follow popup (_site_notice.html, app.follow_prompt) also send the exchange checkboxes
(`venues`, with `venues_form` set): Polymarket runs, Kalshi and ProphetX are opt-ins saved with follow.save_venues.
Saving goes through fantasy/follow.follow (the MCP tool fantasy_follow's path too) and profiles.assign (F1 on Polymarket,
the slot the signal engine runs). The demo accounts keep the
profiles assign_demo gives them: they see the page but can't save. The dice (owner, 2026-10-11 22:20Z: "let the user input
RNG'ed values ... high temperature / lots of variation once we have a few dozen users") roll the sliders in the browser
(app.js), as far from the strategy's own settings as the spread allows; a rolled save is marked `rolled` on the profile. Split out of web/app.py, which imports this module.
"""

from fastapi import Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from racinglines.fantasy import follow as FW
from racinglines.pipelines import profiles as PF
from racinglines.pipelines import sweep_settings as SS
from racinglines.web import demo
from racinglines.web import layout as L
from racinglines.web import roles as R
from racinglines.web.app import ANY, allow, app, audit, check_csrf, get_engine, render

# The follow logic lives in fantasy/follow.py (the MCP tools use it too); these names stay here for the page and its tests.
PRO_CODES = FW.PRO_CODES
DEFAULT = PF.TAKER_TOP[0]                   # preselected when the account runs nothing yet: the shortlist's first
PRE = FW.PRE
TUNABLE = FW.TUNABLE
F1_SLOT = FW.F1_SLOT

# --- the alpha picker (alpha_picker) ---
ALPHA_TWEAKS = FW.ALPHA_TWEAKS                      # the tweak drawer's two sliders, with TUNABLE's caps
TWEAK = "tweak:"                                    # the drawer's field prefix: applied to the card picked
SIZINGS = FW.SIZINGS
KELLY = FW.KELLY
KELLY_PICKS = (("Quarter", 0.25), ("Half", 0.50), ("Full", 1.00))
# the label beside the Kelly slider, "nitty to degen", one emoji per tier (owner, 19:53-19:55Z): (highest fraction, label)
KELLY_LABELS = ((0.25, "Nitty ☕"), (0.50, "Shark 🦈"), (0.80, "Spicy 🌡️"), (1.00, "Degen 🤠"))


def alpha_picker():
    """The one switch between the alpha picker and the classic page: the alpha layout (web/layout.alpha)."""
    return L.alpha()


def kelly_label(f):
    return next(lab for top, lab in KELLY_LABELS if f <= top + 1e-9)


style_name = FW.style_name
display_name = FW.display_name
alpha_options = FW.options
kind = FW.kind
tunables = FW.tunables
lock_note = FW.lock_note


def sizing_of(form):
    """(sizing, kelly) from the form, validated (follow.check_sizing). Raises ValueError with a message for the page."""
    return FW.check_sizing(form.get("sizing"), form.get("kelly"))


def slots(user):
    """The sport tabs: admins see every sport and exchange a strategy could point at (F1 on Polymarket first, the only
    one open; see the docstring); everyone else sees none, as F1 on Polymarket is all they can pick in the Alpha (the
    owner hid MotoGP for it, 2026-10-11 22:59Z). -> [dict(key, label, open)]."""
    from racinglines.web.admin import profile_slots
    if R.canonical(user["role"]) != "admin":
        return []
    return [dict(s, open=s["key"] == F1_SLOT) for s in profile_slots()]


def _schedule(year):
    return FW._schedule(year)


def open_weekend(now=None):
    """The F1 weekend whose signal window is open at `now`, else None (follow.open_weekend, with this module's
    _schedule)."""
    return FW.open_weekend(now, schedule=_schedule)


def options(user):
    """The strategies this account may pick: [dict(code, name, kind, why, sliders, pre)]."""
    if R.is_basic(user):
        return [dict(code="basic", name=R.BASIC_NAME, kind="taker", sliders=[], pre=None,
                     why=f"{R.BASIC_PICKS} taker strategies from our ranked shortlist, drawn for your account and blended "
                         "evenly. You see each pick and how strong it is.")]
    out = []
    for code in PRO_CODES:
        pr = PF.PROFILES[code]
        sl, pre = tunables(pr)
        out.append(dict(code=code, name=pr["name"], kind="taker" if PF.is_combo(pr) else kind(pr), why=pr["why"],
                        sliders=sl, pre=pre, rank=PF.TAKER_TOP.index(code) + 1 if code in PF.TAKER_TOP else None))
    return out


def current_code(profile, opts):
    """The option the account runs now, by name (an adjusted profile keeps its base code), or None."""
    if not profile:
        return None
    return next((o["code"] for o in opts if profile.get("base") == o["code"] or profile.get("name") == o["name"]), None)


@app.get("/strategy/pick", response_class=HTMLResponse, dependencies=[allow(*ANY)])
def pick_page(request: Request, msg: str = ""):
    me = request.state.user
    opts = options(me)
    with get_engine().connect() as c:
        profile = PF.of_user(c, me["id"])
    now = current_code(profile, opts)
    for o in opts:                                 # an adjusted profile: its sliders start where the user left them
        if o["code"] == now and profile.get("adjusted"):
            st = SS.Settings.from_dict(profile["settings"], strict=False)
            for sl in o["sliders"]:
                if st[sl["name"]] is not None:
                    sl["value"] = min(max(float(st[sl["name"]]), sl["lo"]), sl["hi"])
            if o["pre"] is not None:
                o["pre"] = PRE in (st["taker_stages"] or ())
    w = open_weekend()
    if alpha_picker():
        return _alpha_page(request, me, profile, w, msg)
    return render(request, "strategy_pick.html", opts=opts, profile=profile, now=now,
                  chosen=now or (opts[0]["code"] if R.is_basic(me) else DEFAULT), locked=demo.is_demo(me) or bool(w),
                  demo_locked=demo.is_demo(me), weekend_note=lock_note(w) if w else "", msg=msg, basic=R.is_basic(me),
                  slots=slots(me))


def _alpha_page(request, me, profile, w, msg):
    opts = alpha_options(me)
    now = profile.get("base") if profile and profile.get("base") in PF.ALPHA_STYLES and not PF.is_combo(profile) else None
    chosen = now or PF.ALPHA_ORDER[0]
    drawer = next(o for o in opts if o["code"] == chosen)["sliders"]
    if now and profile.get("adjusted"):            # the drawer starts where the user left it
        st = SS.Settings.from_dict(profile["settings"], strict=False)
        drawer = [dict(x, value=min(max(float(st[x["name"]]), x["lo"]), x["hi"])) if st[x["name"]] is not None else x
                  for x in drawer]
    k = float((profile or {}).get("kelly") or KELLY["default"])
    return render(request, "strategy_pick_alpha.html", opts=opts, profile=profile, running=display_name(profile),
                  now=now, chosen=chosen, drawer=drawer, sizing=(profile or {}).get("sizing") or "standard", kelly=k,
                  kelly_cfg=KELLY, kelly_picks=KELLY_PICKS, kelly_labels=KELLY_LABELS, kelly_label=kelly_label(k),
                  locked=demo.is_demo(me) or bool(w), demo_locked=demo.is_demo(me),
                  weekend_note=lock_note(w) if w else "", msg=msg, basic=R.is_basic(me), venues=_venues(me))


def _venues(me):
    """The exchange checkboxes: [dict(key, label, live, note, on)] with the account's opt-ins ticked."""
    with get_engine().connect() as c:
        on = FW.venues_of(c, me["id"])
    return [dict(key=k, label=lab, live=live, note=note, on=k in on) for k, lab, live, note in FW.VENUES]


def _alpha_form(form, user):
    """The drawer's fields ("tweak:<setting>", moved from "tweak_from:<setting>") as follow's tweaks; a basic account's
    tweaks are ignored (it runs a style as tuned)."""
    if R.is_basic(user):
        return {}
    out = {}
    for n in ALPHA_TWEAKS:
        v, was = form.get(TWEAK + n), form.get("tweak_from:" + n)
        try:
            if v not in (None, "") and (was in (None, "") or abs(float(v) - float(was)) > 1e-9):
                out[n] = v
        except ValueError:
            continue
    return out


def _classic_form(code, form):
    """The classic page's fields for one code ("<code>:<setting>", and its pre-weekend checkbox, which sends nothing
    when unticked) as follow's tweaks."""
    out = {k.split(":", 1)[1]: v for k, v in form.items() if k.startswith(f"{code}:")}
    return dict(out, pre_weekend=form.get(f"{code}:pre_weekend") == "1")


def _adjusted(code, form):
    """A strategy's settings with the form's sliders and pre-weekend toggle for it (fields named "<code>:<setting>"),
    clamped to TUNABLE; -> (settings, {what moved}), {} when nothing moved (follow.adjusted)."""
    return FW.adjusted(code, _classic_form(code, form))


@app.post("/strategy/pick", dependencies=[Depends(check_csrf), allow(*ANY)])
async def pick_save(request: Request, code: str = Form(""), rolled: str = Form(""), slot: str = Form(F1_SLOT)):
    me, form, alpha = request.state.user, await request.form(), alpha_picker()
    tweaks = _alpha_form(form, me) if alpha and code in PF.ALPHA_STYLES else _classic_form(code, form)
    try:
        with get_engine().begin() as c:
            old = PF.of_user(c, me["id"])
            new = FW.follow(c, me, code, sizing=form.get("sizing"), kelly=form.get("kelly"), tweaks=tweaks,
                            rolled=rolled == "1", slot=slot, alpha=alpha, weekend=open_weekend())
            if form.get("venues_form") and new is not None:   # the exchange checkboxes (popup and alpha page)
                FW.save_venues(c, me["id"], form.getlist("venues"))
    except FW.FollowError as ex:
        return RedirectResponse("/strategy/pick?msg=" + ("" if ex.reason == "demo_account" else "Error: ") + ex.message,
                                status_code=303)
    audit(request, "strategy_pick", old=(old or {}).get("name"), new=(new or {}).get("name"),
          adjusted=(new or {}).get("adjusted"), rolled=bool((new or {}).get("rolled")))
    return RedirectResponse(("/strategy/pick" if new is None else "/strategy") + "?msg=" + FW.saved_message(new),
                            status_code=303)
