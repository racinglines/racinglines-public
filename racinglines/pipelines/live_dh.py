"""
Live downhill final: poll UCI's live timing (ChronoRace), simulate the rest of the final, and re-quote.

    racinglines strategy live mtb_dh --slug 20260925_mtb --final 3 --quali 2,91 --conditions "clear, rutted"

Every --interval seconds:
  1. **Feed.** The final's live JSON: who has finished (time), who is on course (latest split), who is
     still to start (start order), DNF / DNS.
  2. **Pace.** Each rider's time blends this weekend's qualifying (best run, sessions put on one scale)
     with the season model's pre-final view (its podium-probability rank mapped onto the field's
     qualifying times; SEASON_WEIGHT). Riders who were mathematically safe to qualify (the season model's
     make-the-final probability in the top quarter of the field, fading to none at the median) may have
     cruised: when their run came out slower than their season rank implies, that gap becomes extra
     uncertainty (wider spread, up to SAFE_SIGMA_CAP) rather than a faster expected time; a fast run is
     genuine pace. The time is
     scaled by how the final is running against qualifying so far (median final/qualifying ratio of the
     finishers, starting from a small prior). A rider on course is projected from the latest split with
     the split-to-finish ratio (finishers of the final, else qualifying). Noise shrinks with the distance
     left; track conditions ("rutted", "wet") widen it and raise the crash / DNF risk.
  3. **Monte Carlo.** N_SIMS simulated finishes -> each rider's probability of every top rank: P(1st),
     P(2nd), P(3rd), top 3, top 5, top 10.
  4. **Maker.** The demo maker quotes YES on "wins" and "podium" for every rider: fair +- a half-spread,
     wider while the rider is on course and just before they start; no quote once the outcome is certain.
     Every quote is appended to the history.
  5. **Taker.** The demo taker's picks (picks.json, chosen once; see make_picks) are marked against the
     quotes and settled when the final is over.

Files: data/runs/live/<slug>_<key>/
    latest.json        the newest snapshot (the web app's Live tab reads it)
    history.jsonl      one line per poll: counts, every market's fair value and quote
    picks.json         the demo taker's picks
    meta.json          the model inputs and parameters (qualifying times, split ratios, season priors,
                       conditions), written on the first poll and whenever they change
    book.json          the maker's private book: position and cash per market, crowd totals, the seeds
    crowd.jsonl        every anonymous crowd fill, per poll, with that poll's random seed
    raw/<ts>.json.gz   every raw feed response, as received      } kept permanently, so the final can be
    snaps/<ts>.json.gz every full snapshot (rank probabilities)   } replayed or re-scored later
A demo experiment: the quotes are simulated, nothing is traded anywhere.
"""

import fcntl
import json
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from racinglines import paths

from racinglines import sports
from racinglines.markets import crowd as C
from racinglines.markets import quoting as Q
from racinglines.pipelines import live as LV

_LIVE = sports.load("mtb_dh")["live"]          # the sport's live settings (sports/mtb_dh.toml [live])
FEED = _LIVE["sources"]["live"]["url"]
MODEL_VERSION = 2               # v2 (27 Sep 2026, mid-final): track trend by start order, shared track shock,
                                # recent-finisher split ratios, quotes pulled after a rider's last split
N_SIMS = 20_000
TREND_RECENCY = 0.85            # weight of a finisher k slots before the latest: TREND_RECENCY ** k
TREND_PRIOR_N = 8.0             # the trend slope is shrunk toward flat with the weight of this many runs
TREND_BLEND = 0.5               # riders still to come: this share of the trend, the rest the whole final's level
                                # (start order is reverse qualifying rank, so the slope also carries regression to
                                # the mean of slow qualifiers, not only track wear)
RECENT_SPLITS = 6               # split-to-finish ratios from this many most recent finishers
RANKS = 10
SEASON_WEIGHT = 0.4             # share of the pace from the season model for a rider who had to push in qualifying
SAFE_SIGMA_CAP = 0.012          # a safe rider's slow qualifying adds up to this much log-time spread (they may have cruised)
PRIOR_FACTOR = 1.01             # a final runs ~1% slower than qualifying before any evidence (tired track)
SIGMA_START = 0.016             # log-time s.d. of a rider still to start
DNF_START = 0.05                # crash / DNF probability of a rider still to start
COND = {"rutted": (0.004, 0.02), "wet": (0.010, 0.04), "dry": (0.0, 0.0), "clear": (0.0, 0.0)}
# quoting, the crowd and the cadence: data, in sports/mtb_dh.toml [live]
HALF_SPREAD = _LIVE["quoting"]["half_spread"]
WIDEN_ON_COURSE = _LIVE["quoting"]["widen_on_course"]      # the rider is on course: information arrives fast
WIDEN_NEXT = _LIVE["quoting"]["widen_next"]                # next few to start
MARKETS = dict(_LIVE["markets"]["names"])
# the private book's crowd: CROWD anonymous takers, each hitting a quoted market with probability CROWD_P per
# poll, on a random side; each taker has an event budget of $10-200 and bets part of what's left each time;
# anonymous, tracked as one group
CROWD = _LIVE["crowd"]["takers"]
CROWD_P = _LIVE["crowd"]["p"]
CROWD_BUDGET = tuple(_LIVE["crowd"]["budget"])   # $ each taker may bet over the whole event, log-uniform (a private beta, play money)
CROWD_BET = tuple(_LIVE["crowd"]["bet"])         # each bet: this share of what the taker has left (at least MIN_BET)
MIN_BET = _LIVE["crowd"]["min_bet"]
LATE_WHEN_LEFT = _LIVE["poll"]["late_when_left"]  # the late window opens once the fourth-to-last rider has started (<= 3 to start):
LATE_INTERVAL = _LIVE["poll"]["late_interval_s"]  # polls every 2 s, and every taker gets a fresh $LATE_CAP for the window
LATE_CAP = _LIVE["crowd"]["late_cap"]
LATE_PACE = _LIVE["crowd"]["late_pace"]           # the late window trades at the push pace: 50x the normal crowd rate
WAIT_INTERVAL = 300                                # before the final's start list: a look every 5 minutes, a minute
SOON_INTERVAL = 60                                 # once a qualifying session's start list is on the timing feed,
                                                   # and BASE_INTERVAL while a qualifying session is running
BASE_INTERVAL = _LIVE["poll"]["interval_s"]       # CROWD_P is per BASE_INTERVAL seconds; faster polls scale it down (same rate)
MAX_POS = _LIVE["quoting"]["max_pos"]             # the maker's shares per market, either way
SKEW = _LIVE["quoting"]["skew"]                   # quotes lean against inventory: shift = -SKEW x half-spread x inventory / MAX_POS
CROWD_SEED = _LIVE["crowd"]["seed"]


EVENT_NAMES = {"20260925_mtb": "Whistler DH final (private book)",
               "20261002_mtb": "Lake Placid DH final (private book)"}


def event_name(slug):
    """A private-book event's display name (its positions have no events row)."""
    return EVENT_NAMES.get(slug, slug)


def book_curve(slug, maker=True):
    """The private book's P&L through the event (all of its finals): pipelines/live.py book_curve."""
    return LV.book_curve(slug, maker)


def run_name(slug, key):
    """The run folder's name (data/runs/live/<slug>_<key>)."""
    return f"{slug}_{key}"


def outdir(slug, key):
    return LV.folder(run_name(slug, key))


def fetch(slug, key, session=None):
    import requests

    from racinglines.sources import http
    s = session or requests.Session()
    r = http.get(s, FEED.format(slug=slug, key=key), headers={"User-Agent": "racinglines/0.1"}, timeout=30)
    r.raise_for_status()
    d = r.json()
    return d if isinstance(d, dict) else {}


def quali_best(slug, keys, session=None):
    """{bib: best qualifying time ms, on the first session's scale} and the qualifying split-to-finish
    ratios (median per split). Later sessions can run in different conditions (Whistler's Q2 ran ~2% slower
    than Q1), so each is rescaled to the first with the median time ratio of the riders who finished both."""
    return quali_from([fetch(slug, k, session) for k in keys])


def quali_from(feeds):
    """quali_best() on qualifying feeds already fetched."""
    sessions, ratios = [], {}
    for d in feeds:
        t = {}
        for r in d.get("Results") or []:
            if r.get("Status") != "Finished" or not r.get("RaceTime"):
                continue
            t[r["RaceNr"]] = r["RaceTime"]
            for i, x in enumerate(r.get("Times") or []):
                if x.get("RaceTime"):
                    ratios.setdefault(i, []).append(r["RaceTime"] / x["RaceTime"])
        sessions.append(t)
    best = dict(sessions[0]) if sessions else {}
    for t in sessions[1:]:
        both = [best[n] / t[n] for n in t if n in best and 0.9 < best[n] / t[n] < 1.1]
        scale = float(np.median(both)) if len(both) >= 5 else 1.0
        for n, x in t.items():
            best[n] = min(best.get(n, 1e12), x * scale)
    return best, {i: float(np.median(v)) for i, v in ratios.items() if v}


def parse(d):
    """Riders of the final: dict(bib, name, nation, uci_rank, status, time, splits, order)."""
    riders = d.get("Riders") or {}
    nxt = {n: i for i, n in enumerate(d.get("NextToStart") or [])}
    on = {r["RaceNr"]: r for r in d.get("OnTrack") or []}
    out = []
    for r in d.get("Results") or []:
        n = r["RaceNr"]
        rd = riders.get(str(n), {})
        live = on.get(n, r)
        status = live.get("Status") or r.get("Status")
        splits = [t.get("RaceTime") for t in (live.get("Times") or r.get("Times") or []) if t.get("RaceTime")]
        out.append(dict(bib=n, name=rd.get("PrintName") or str(n), nation=rd.get("Nation"), team=rd.get("UciTeamName"),
                        uci_rank=rd.get("UciRank"), status=status, start=r.get("ExpectedStartTime"),
                        time=r.get("RaceTime") if status == "Finished" else None, splits=splits,
                        split_pos=[t.get("Position") for t in (live.get("Times") or [])],
                        sort=r.get("SortOrder"), next=nxt.get(n)))
    order = sorted(out, key=lambda x: (x["start"] is None, x["start"] or 0))
    for i, x in enumerate(order):
        x["slot"] = i                                                   # start order
    return out


def _rider_category(rider):
    """Category text from ChronoRace: 'Men Elite', 'ME', etc. Returns a normalised lower-case string."""
    if not isinstance(rider, dict):
        return ""
    for key in ("Category", "CategoryName", "CompetitionCategory", "CategoryCode", "CategoryId", "Name"):
        value = rider.get(key)
        if value is not None:
            return str(value).lower().strip()
    return ""


def _men_elite_rider(rider):
    cat = _rider_category(rider)
    return cat in {"me", "men elite", "elite men", "male elite", "menelite", "male-elite"} or (
        ("men" in cat or "male" in cat) and "elite" in cat)


def poll_interval(feed, default=BASE_INTERVAL, full=LATE_INTERVAL):
    """Raise the live poll rate whenever a Men's Elite rider is on-track or next to start; keep the base cadence otherwise."""
    if not isinstance(feed, dict):
        return default
    riders = feed.get("Riders") or {}
    for key in ("OnTrack", "NextToStart"):
        for row in feed.get(key) or []:
            if isinstance(row, dict):
                bib = row.get("RaceNr")
            else:
                bib = row
            rider = riders.get(str(bib), riders.get(bib))
            if rider is not None and _men_elite_rider(rider):
                return full
    return default


def track_trend(fin, qbest):
    """(ratio(slot) function, s.d. of the ratio for the next riders): how the final runs against qualifying,
    by start slot. A recency-weighted line through the clean runs (crashes and big mistakes left out), its
    slope shrunk toward flat; None until three clean runs."""
    pts = [(r["slot"], r["time"] / qbest[r["bib"]]) for r in fin if r["bib"] in qbest and r.get("slot") is not None]
    pts = [(x, y) for x, y in pts if 0.85 < y < 1.12]
    if len(pts) < 3:
        return None, None
    x = np.array([p[0] for p in pts], float)
    y = np.log([p[1] for p in pts])
    keep = np.ones(len(x), bool)
    for _ in range(3):                                   # robust: drop runs far off the line (mistakes), refit
        w = TREND_RECENCY ** (x.max() - x) * keep
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        sxx = float(np.sum(w * (x - xm) ** 2))
        slope = float(np.sum(w * (x - xm) * (y - ym)) / (sxx + TREND_PRIOR_N * max(float(np.var(x)), 1.0)))
        resid = y - (ym + slope * (x - xm))
        sd = max(float(np.sqrt(np.average(resid ** 2, weights=w))), 0.008)
        new = np.abs(resid) <= 2.5 * sd
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    w = TREND_RECENCY ** (x.max() - x) * keep
    n_eff = float(w.sum() ** 2 / np.sum(w ** 2))
    lo, hi = float(y.min()) - 0.02, float(y.max()) + 0.02

    def ratio(slot):
        return float(np.exp(np.clip(ym + slope * (slot - xm), lo, hi)))
    return ratio, sd / math.sqrt(max(n_eff, 1.0)) + 0.003


def conditions_noise(cond):
    extra_s, extra_d = 0.0, 0.0
    for word, (s, dnf) in COND.items():
        if word in (cond or "").lower():
            extra_s, extra_d = extra_s + s, extra_d + dnf
    return extra_s, extra_d


def season_prior(slug, category="ME"):
    """{rider name: (podium probability, make-the-final probability)} from the latest downhill forecast for
    this event's race."""
    from sqlalchemy import text

    from racinglines.db.config import get_engine
    try:
        with get_engine().connect() as c:
            rows = c.execute(text("""
                SELECT a.display_name, rp.podium_prob, rp.make_final_prob FROM race_predictions rp
                JOIN athletes a ON a.id = rp.athlete_id JOIN races ra ON ra.id = rp.race_id
                JOIN events e ON e.id = ra.event_id JOIN categories ca ON ca.id = ra.category_id
                WHERE e.source_key = :k AND ca.code = :c AND rp.model_run_id = (
                    SELECT max(rp2.model_run_id) FROM race_predictions rp2 WHERE rp2.race_id = ra.id)"""),
                dict(k=slug, c=category)).all()
        return {n: (float(p or 0), float(mf or 0)) for n, p, mf in rows}
    except Exception:                                  # noqa: BLE001  (no database: qualifying only)
        return {}


def safety(riders, prior):
    """{bib: 0..1}: how safe each rider was to qualify, by the season model's make-the-final probability
    within this field (top quarter = 1, fading to 0 at the median)."""
    mf = {r["bib"]: prior[r["name"]][1] for r in riders if r["name"] in prior}
    if len(mf) < 4:
        return {}
    v = np.array(sorted(mf.values()))
    med, q3 = float(np.quantile(v, 0.5)), float(np.quantile(v, 0.75))
    return {b: float(np.clip((x - med) / max(q3 - med, 1e-9), 0, 1)) for b, x in mf.items()}


def blended_pace(riders, qbest, prior):
    """({bib: base time ms}, {bib: extra log-time s.d.}): qualifying blended with the season model's rank
    (mapped onto the field's qualifying times); a safe rider's slow qualifying adds spread, not speed."""
    field = [r for r in riders if r["bib"] in qbest]
    qtimes = sorted(qbest[r["bib"]] for r in field)
    if not prior or not field:
        return {r["bib"]: qbest[r["bib"]] for r in field}, {}
    ranked = sorted(field, key=lambda r: -prior.get(r["name"], (0.0, 0.0))[0])
    model_t = {r["bib"]: qtimes[i] for i, r in enumerate(ranked)}
    safe = safety(riders, prior)
    out, extra = {}, {}
    for r in field:
        b = r["bib"]
        out[b] = (1 - SEASON_WEIGHT) * qbest[b] + SEASON_WEIGHT * model_t[b]
        # cruising only makes a run slower: a safe rider whose run came out slower than their season rank
        # implies is less known, not faster; a fast run is genuine pace whoever sets it
        gap = math.log(qbest[b] / model_t[b]) if qbest[b] > model_t[b] else 0.0
        extra[b] = min(SAFE_SIGMA_CAP, 0.5 * safe.get(b, 0.0) * gap)
        r["safety"], r["extra_sigma"] = safe.get(b, 0.0), extra[b]
    return out, extra


def simulate(riders, qbest, qratio, cond="", n=N_SIMS, seed=7, prior=None):
    """Rank probabilities for every rider. Returns (rows, factor)."""
    rng = np.random.default_rng(seed)
    extra_s, extra_d = conditions_noise(cond)
    fin = [r for r in riders if r["status"] == "Finished" and r["time"]]
    # how the final runs against qualifying: the median of clean runs (crashes and big mistakes, > 12% off,
    # are left out), shrunk toward the prior with the weight of two runs
    ratios = [r["time"] / qbest[r["bib"]] for r in fin if r["bib"] in qbest]
    clean = [x for x in ratios if 0.85 < x < 1.12]
    k = len(clean)
    factor = (PRIOR_FACTOR * 2 + float(np.median(clean)) * k) / (2 + k) if k else PRIOR_FACTOR
    trend, trend_sd = track_trend(fin, qbest)                            # v2: the track changes through the final
    shock = rng.normal(0, trend_sd, n) if trend_sd else np.zeros(n)      # one shared track shock per simulation
    # split -> finish ratios from the final's finishers, else qualifying
    fratio = {}
    clean_fin = [r for r in fin if r["bib"] not in qbest or 0.85 < r["time"] / qbest[r["bib"]] < 1.12]
    clean_fin = sorted(clean_fin, key=lambda r: r.get("slot") or 0)[-RECENT_SPLITS:]     # v2: recent finishers
    for i in range(6):
        v = [r["time"] / r["splits"][i] for r in clean_fin if len(r["splits"]) > i and r["splits"][i]]
        fratio[i] = float(np.median(v)) if len(v) >= 3 else qratio.get(i)
    slowest = max(qbest.values()) if qbest else 240_000
    base, extra_sig = blended_pace(riders, qbest, prior or {})
    times = np.full((n, len(riders)), np.inf)
    for j, r in enumerate(riders):
        st = r["status"]
        if st == "Finished" and r["time"]:
            times[:, j] = r["time"]
        elif st in ("DNF", "DNS", "DSQ"):
            continue
        elif st == "InRace" and r["splits"] and fratio.get(len(r["splits"]) - 1):
            i = len(r["splits"]) - 1
            mu = r["splits"][i] * fratio[i]
            left = max(0.05, 1 - r["splits"][i] / mu)                 # share of the run still to go
            t = mu * np.exp(rng.normal(0, (SIGMA_START + extra_s) * math.sqrt(left), n))
            dnf = rng.random(n) < (DNF_START + extra_d) * left
            times[:, j] = np.where(dnf, np.inf, t)
        else:                                                           # still to start (or on course, no split yet)
            f = (trend(r["slot"]) ** TREND_BLEND * factor ** (1 - TREND_BLEND)
                 if trend and r.get("slot") is not None else factor)
            mu = base.get(r["bib"], slowest * 1.02) * f
            t = mu * np.exp(rng.normal(0, SIGMA_START + extra_s + extra_sig.get(r["bib"], 0.0), n) + shock)
            dnf = rng.random(n) < DNF_START + extra_d
            times[:, j] = np.where(dnf, np.inf, t)
    order = np.argsort(times, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(1, len(riders) + 1)
    finished_any = np.isfinite(times)
    rows = []
    for j, r in enumerate(riders):
        rk = np.where(finished_any[:, j], ranks[:, j], 999)
        p = {f"p{i}": float((rk == i).mean()) for i in range(1, RANKS + 1)}
        rows.append(dict(r, **p, top3=float((rk <= 3).mean()), top5=float((rk <= 5).mean()),
                         top10=float((rk <= 10).mean()), p_finish=float(finished_any[:, j].mean()),
                         expected=None if not np.isfinite(times[:, j]).any() else float(np.median(times[:, j][np.isfinite(times[:, j])]))))
    simulate.trend = None if not trend else dict(next_ratio=trend(max((r.get("slot") or 0) for r in riders)),
                                                  sd=trend_sd)
    return rows, factor


def quote(fair, rider, n_next=3, inv=0.0):
    """The maker's YES quote on one market: (bid, ask) in cents-rounded dollars, or (None, None). Leans
    against inventory; stops adding to a position at MAX_POS (markets/quoting.py). Downhill's half-spread
    widens while the rider is on course and just before they start."""
    if fair is None or fair <= Q.LO or fair >= Q.HI:
        return None, None
    hs = HALF_SPREAD
    if rider["status"] == "InRace":
        if MODEL_VERSION >= 2 and len(rider.get("splits") or []) >= 4:   # past the last split: the result is all but
            return None, None                                             # known, so only informed takers would trade
        hs += WIDEN_ON_COURSE
    elif rider.get("next") is not None and rider["next"] < n_next:
        hs += WIDEN_NEXT
    if MODEL_VERSION >= 2 and rider["status"] not in ("Finished", "InRace"):
        hs += 0.01                                                        # the track is changing under them
    return Q.quote(fair, hs, inv, MAX_POS, SKEW, tidy=False)


def crowd_params():
    """The crowd's settings (markets/crowd.py), from this module's constants."""
    return C.Params(takers=CROWD, p=CROWD_P, budget=CROWD_BUDGET, bet=CROWD_BET, min_bet=MIN_BET, max_pos=MAX_POS,
                    seed=CROWD_SEED, late_cap=LATE_CAP, late_pace=LATE_PACE)


def load_book(out):
    return C.load_book(out, crowd_params())


def crowd_fills(quotes, book, rng, intensity=1.0, pot="left"):
    """One poll of the anonymous crowd against the maker's quotes (intensity x the normal hit rate). Mutates
    book; returns the fills (markets/crowd.py)."""
    return C.fills(quotes, book, rng, crowd_params(), intensity=intensity, pot=pot)


PRIVATE, MAKER_USER, TAKER_USER = C.PRIVATE, C.MAKER_USER, C.TAKER_USER


def _by_key(outcomes):
    """{(bib, market): yes} -> {"bib:market": yes}."""
    return {f"{b}:{m}": v for (b, m), v in outcomes.items()}


def sync_positions(slug, rows, book, picks, fair, outcomes):
    """Write the private book into paper_positions (venue 'private', event_key = the event slug): the demo
    maker's position per market (the crowd's fills and the demo taker's picks) and the demo taker's picks.
    Replaced on every update, so the Positions page shows them live and settled."""
    names = {r["bib"]: r["name"] for r in rows}

    def describe(k):
        b, m = k.split(":")
        return f"dh_{m}", names.get(int(b), b)
    C.sync_positions(slug, book, picks, describe, fair, _by_key(outcomes))


def crowd_results(book, fair, outcomes, spotlight_seed=20260927):
    """Each anonymous taker's P&L (marked to fair; settled where known), summarised as a group, plus one
    randomly chosen taker's bets."""
    return C.crowd_results(book, fair, _by_key(outcomes), spotlight_seed)


def book_pnl(book, fair, outcomes, picks=()):
    """The maker's P&L, marked to fair (settled where the outcome is known): total, vs the crowd, vs the demo taker."""
    return C.book_pnl(book, fair, _by_key(outcomes), picks)


def settle(rows, done):
    """Outcomes once the final is over: {(bib, market): bool}."""
    if not done:
        return {}
    fin = sorted([r for r in rows if r["status"] == "Finished" and r["time"]], key=lambda r: r["time"])
    rank = {r["bib"]: i + 1 for i, r in enumerate(fin)}
    return {(r["bib"], m): (rank.get(r["bib"], 999) <= (1 if m == "win" else 3)) for r in rows for m in MARKETS}


def quali_table(d):
    """One qualifying feed as rows for the Live page: rank, bib, name, nation, time (ms), gap (ms), status."""
    riders = d.get("Riders") or {}
    res = d.get("Results") or []
    fin = sorted([r for r in res if r.get("Status") == "Finished" and r.get("RaceTime")], key=lambda r: r["RaceTime"])
    rest = [r for r in res if r not in fin]
    lead = fin[0]["RaceTime"] if fin else None
    out = []
    for i, r in enumerate(fin + rest):
        rd = riders.get(str(r["RaceNr"]), {})
        done = i < len(fin)
        out.append(dict(rank=i + 1 if done else None, bib=r["RaceNr"], name=rd.get("PrintName") or str(r["RaceNr"]),
                        nation=rd.get("Nation"), time=r["RaceTime"] if done else None,
                        gap=r["RaceTime"] - lead if done else None, status=r.get("Status")))
    return out


def provisional(feeds, qbest):
    """The final's riders as qualifying leaves them, before ChronoRace publishes the start list: everyone with
    a qualifying time, starting in reverse qualifying order (the fastest last), all still to start. Nobody is
    "next": who makes the final and the real order are ChronoRace's start list's to say."""
    names = {}
    for d in feeds:
        for b, rd in (d.get("Riders") or {}).items():
            names.setdefault(int(b) if str(b).isdigit() else b, rd)
    order = sorted(qbest, key=lambda b: -qbest[b])                    # slowest first: the final's start order
    out = []
    for i, b in enumerate(order):
        rd = names.get(b, {})
        out.append(dict(bib=b, name=rd.get("PrintName") or str(b), nation=rd.get("Nation"), team=rd.get("UciTeamName"),
                        uci_rank=rd.get("UciRank"), status="NA", start=None, time=None, splits=[], split_pos=[],
                        sort=None, next=None, slot=i))
    return out


def running(d):
    """A qualifying session under way: a rider in ChronoRace's OnTrack whose result isn't final yet, anyone in
    NextToStart, or an InRace status. OnTrack alone isn't enough: timed training kept its last riders there after
    they had all finished (Lake Placid, 3 Oct)."""
    res = d.get("Results") or []
    status = {r.get("RaceNr"): r.get("Status") for r in res}
    on = [x.get("RaceNr") if isinstance(x, dict) else x for x in d.get("OnTrack") or []]
    return bool(any(status.get(n) not in ("Finished", "DNF", "DNS", "DSQ") for n in on) or d.get("NextToStart")
                or "InRace" in status.values())


def settled(d):
    """A qualifying feed that won't change: times in and nobody on course or next to start. Riders left without a
    time then didn't start or finish, whatever their status says (a DNS can stay 'NA')."""
    return not running(d) and any(r.get("Status") == "Finished" for r in d.get("Results") or [])


def waiting(slug, key, d, cond, out, quali_keys=(), session=None):
    """The final's feed has no riders yet (ChronoRace publishes the start list after qualifying): write a
    latest.json so the Live page lists the event, with the qualifying results so far and the model's pre-final
    odds from them (reverse qualifying order, nobody started), and leave the book, the crowd, the positions and
    the final's qualifying cache alone: no quotes until the final's own feed has riders. Without this an empty
    feed counted 0 riders to start and opened the late window (Lake Placid, 3 Oct 2026).
    Requests: a settled session (Q1 once it's over) is read once per process and kept; a session not started or
    under way is read every look. The next look is BASE_INTERVAL while one is running, SOON_INTERVAL while one's
    start list is on the feed but nobody has started, else WAIT_INTERVAL; unchanged qualifying times reuse the
    last simulation."""
    cache = waiting.__dict__.setdefault("cache", {})
    kept = cache.setdefault("settled", {})
    feeds, read = [], []
    for k in quali_keys:
        f = kept.get((slug, k))
        if f is None:
            f = fetch(slug, k, session)
            read.append(k)
            if settled(f):
                kept[(slug, k)] = f
        feeds.append(f)
    live = [k for k, f in zip(quali_keys, feeds) if running(f)]
    soon = [k for k, f in zip(quali_keys, feeds) if f.get("Results") and not running(f) and not settled(f)]   # start list up
    interval = BASE_INTERVAL if live else SOON_INTERVAL if soon else WAIT_INTERVAL
    qbest, qratio = quali_from(feeds)
    if "prior" not in cache:
        cache["prior"] = season_prior(slug)
    state = json.dumps(sorted(qbest.items()))
    if cache.get("state") != (slug, state):
        cache["rows"] = simulate(provisional(feeds, qbest), qbest, qratio, cond, prior=cache["prior"])[0] if qbest else []
        cache["state"] = (slug, state)
    rows = cache["rows"]
    book = load_book(out)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snap = dict(ts=now, slug=slug, key=key, round=d.get("DisplayName") or event_name(slug), conditions=cond,
                waiting=("ChronoRace has no start list for this final yet; odds are the model's from qualifying, "
                         "in reverse qualifying order, with no quotes until the final's start list is out"
                         if rows else "ChronoRace has no start list for this final and no qualifying times yet"),
                qualifying=[dict(key=k, title=f.get("DisplayName") or f"Qualifying {k}", rows=quali_table(f),
                                 running=k in live) for k, f in zip(quali_keys, feeds) if f.get("Results")],
                qualifying_live=live, fetched=read,
                model_version=MODEL_VERSION, counts={"NA": len(rows)} if rows else {}, done=False, riders=rows,
                quotes=[], maker_pnl=book_pnl(book, {}, {}),
                betting=dict(closed=False, closed_at=None, last_call=False, to_start=None, late=bool(book.get("late")),
                             late_at=book.get("late_at"), interval=interval, late_cap=LATE_CAP),
                crowd=dict(book["crowd"], takers=CROWD, last_fills=0,
                           active=sum(1 for x, b in zip(book["left"], book["budget"]) if x < b - 0.005),
                           left=round(sum(book["left"]), 2), results=crowd_results(book, {}, {})),
                outcomes=[], poll_interval=interval)
    tmp = out / "latest.json.tmp"
    tmp.write_text(json.dumps(snap, default=str))
    tmp.replace(out / "latest.json")
    return snap


def update(slug, key, quali_keys, cond="", session=None, interval=BASE_INTERVAL):
    """One poll: feed -> simulation -> quotes -> files. Returns the snapshot."""
    import gzip
    d = fetch(slug, key, session)
    fetched = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = outdir(slug, key)
    riders = parse(d)
    if not riders:                                                    # no start list yet: the book waits untouched
        return waiting(slug, key, d, cond, out, quali_keys, session)
    (out / "raw").mkdir(exist_ok=True)
    (out / "snaps").mkdir(exist_ok=True)
    with gzip.open(out / "raw" / f"{fetched}.json.gz", "wt") as f:                 # the feed, as received
        json.dump(d, f)
    import hashlib
    cache = update.__dict__.setdefault("cache", {})                   # per process: nothing old is re-processed
    if "q" not in cache:
        cache["q"] = quali_best(slug, quali_keys, session)            # qualifying: fixed for the final
        cache["prior"] = season_prior(slug)                           # the season model's pre-final view: fixed
    qbest, qratio = cache["q"]
    prior = cache["prior"]
    # the timing state that pricing depends on; an unchanged feed reuses the last simulation
    effective_interval = poll_interval(d, default=interval, full=LATE_INTERVAL)
    state = hashlib.sha1(json.dumps([(r["bib"], r["status"], r["time"], r["splits"]) for r in riders]).encode()).hexdigest()
    meta = dict(slug=slug, key=key, quali_keys=list(quali_keys), conditions=cond,
                params=dict(MODEL_VERSION=MODEL_VERSION, TREND_RECENCY=TREND_RECENCY, TREND_PRIOR_N=TREND_PRIOR_N,
                            TREND_BLEND=TREND_BLEND,
                            RECENT_SPLITS=RECENT_SPLITS, N_SIMS=N_SIMS, SIGMA_START=SIGMA_START, DNF_START=DNF_START, PRIOR_FACTOR=PRIOR_FACTOR,
                            SEASON_WEIGHT=SEASON_WEIGHT, SAFE_SIGMA_CAP=SAFE_SIGMA_CAP, COND=COND, HALF_SPREAD=HALF_SPREAD,
                            WIDEN_ON_COURSE=WIDEN_ON_COURSE, WIDEN_NEXT=WIDEN_NEXT, seed=7, CROWD=CROWD,
                            CROWD_P=CROWD_P, CROWD_BUDGET=CROWD_BUDGET, CROWD_BET=CROWD_BET, MIN_BET=MIN_BET,
                            LATE_WHEN_LEFT=LATE_WHEN_LEFT,
                            LATE_INTERVAL=LATE_INTERVAL, LATE_CAP=LATE_CAP, LATE_PACE=LATE_PACE,
                            MAX_POS=MAX_POS, SKEW=SKEW),
                qbest={str(k): v for k, v in qbest.items()}, qratio={str(k): v for k, v in qratio.items()},
                prior={k: list(v) for k, v in prior.items()})
    blob = json.dumps(meta, sort_keys=True, default=str)
    if not (out / "meta.json").exists() or (out / "meta.json").read_text() != blob:
        (out / "meta.json").write_text(blob)
        (out / f"meta_{fetched}.json").write_text(blob)                               # every version kept
    if cache.get("state") == state and cache.get("rows") is not None:
        rows, factor = json.loads(cache["rows"]), cache["factor"]       # nothing new on the timing screen
        changed = False
    else:
        rows, factor = simulate(riders, qbest, qratio, cond, prior=prior)
        cache.update(state=state, rows=json.dumps(rows), factor=factor)
        changed = True
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}}
    done = not any(r["status"] in ("InRace", "NA", "Waiting") for r in rows) and counts.get("Finished", 0) > 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    book = load_book(out)
    picks = json.loads((out / "picks.json").read_text()) if (out / "picks.json").exists() else []
    pick_inv = {}
    for p in picks:                                                     # the demo taker's YES: the maker is short
        pick_inv[f"{p['bib']}:{p['market']}"] = pick_inv.get(f"{p['bib']}:{p['market']}", 0.0) - p["shares"]
    quotes = []
    for r in rows:
        r["quali"] = qbest.get(r["bib"])
        for m, fair in (("win", r["p1"]), ("podium", r["top3"])):
            k = f"{r['bib']}:{m}"
            inv = book["markets"].get(k, {}).get("inv", 0.0) + pick_inv.get(k, 0.0)
            bid, ask = quote(fair, r, inv=inv)
            quotes.append(dict(bib=r["bib"], market=m, fair=round(fair, 4), bid=bid, ask=ask, inv=round(inv, 2)))
    outcomes = settle(rows, done)
    # The anonymous crowd trades on every individually valid quote until the next poll (seeded: replayable).
    # Existing books closed by the retired global lockout are reopened while the final is still running.
    seed = int(datetime.now(timezone.utc).timestamp())
    to_start = sum(1 for r in rows if r["status"] in ("NA", "Waiting"))
    if book.get("closed") and not done:
        book.pop("closed", None)
        book.pop("closed_at", None)
    if to_start <= LATE_WHEN_LEFT and not book.get("late"):             # the late window opens: fresh $100 caps
        book["late"], book["late_at"] = True, now
        book["late_left"] = [LATE_CAP] * CROWD
    pot = "late_left" if book.get("late") else "left"
    rate = effective_interval / BASE_INTERVAL                              # same crowd rate per second at any poll rate
    if book.get("late"):
        rate *= LATE_PACE                                                # late window: the high-volume push pace
    if done:
        fills = []
    else:
        fills = crowd_fills(quotes, book, np.random.default_rng(seed), intensity=rate, pot=pot)
    book["seeds"].append(seed)
    (out / "book.json").write_text(json.dumps(book))
    with (out / "crowd.jsonl").open("a") as f:
        # the rate is logged so a replay never has to re-derive it from today's constants (docs/live-events.md)
        f.write(json.dumps(dict(ts=now, seed=seed, intensity=rate, fills=fills, last_call=None)) + "\n")
    fairmap = {f"{q['bib']}:{q['market']}": q["fair"] for q in quotes}
    pnl = book_pnl(book, fairmap, outcomes, picks)
    cres = crowd_results(book, fairmap, outcomes)
    try:
        if changed or fills or not cache.get("synced"):                 # only when the book or the prices moved
            sync_positions(slug, rows, book, picks, fairmap, outcomes)
            cache["synced"] = True
    except Exception as ex:                          # noqa: BLE001  never stop the live loop for the database
        print(f"positions sync failed: {ex}", flush=True)
    snap = dict(ts=now, slug=slug, key=key, round=d.get("DisplayName"), conditions=cond, factor=factor,
                model_version=MODEL_VERSION, trend=getattr(simulate, "trend", None),
                counts=counts, done=done, riders=rows, quotes=quotes, maker_pnl=pnl,
                betting=dict(closed=False, closed_at=None, last_call=False,
                             to_start=to_start, late=bool(book.get("late")), late_at=book.get("late_at"),
                             interval=effective_interval, late_cap=LATE_CAP),
                crowd=dict(book["crowd"], takers=CROWD, last_fills=len(fills),
                           active=sum(1 for x, b in zip(book["left"], book["budget"]) if x < b - 0.005),
                           left=round(sum(book["left"]), 2), results=cres),
                outcomes=[dict(bib=b, market=m, yes=v) for (b, m), v in outcomes.items()],
                poll_interval=effective_interval)
    with gzip.open(out / "snaps" / f"{fetched}.json.gz", "wt") as f:
        json.dump(snap, f, default=str)
    tmp = out / "latest.json.tmp"
    tmp.write_text(json.dumps(snap, default=str))
    tmp.replace(out / "latest.json")
    with (out / "history.jsonl").open("a") as f:
        f.write(json.dumps(dict(ts=now, counts=counts, fair={f"{q['bib']}:{q['market']}": q["fair"] for q in quotes},
                                quotes={f"{q['bib']}:{q['market']}": [q["bid"], q["ask"]] for q in quotes})) + "\n")
    return snap


@contextmanager
def lock(slug, key):
    """One poll loop per final: a non-blocking lock on the run folder, the F1 step's (cli/live.py lock()). Yields
    True holding it, False when another process holds it. The OS drops the lock with the process that took it,
    so a killed loop leaves nothing stale to clean up; the `.lock` file itself stays (it's empty)."""
    f = (outdir(slug, key) / ".lock").open("w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def run(slug, key, quali_keys, cond="", interval=BASE_INTERVAL, minutes=0, echo=print):
    """Poll the final until it is over (or `minutes`), holding the run folder's lock() the whole time: a second
    loop for the same final (a restart that didn't stop the first) says so and returns None instead of polling
    alongside it (Whistler 22:38-22:39, docs/live-events.md)."""
    with lock(slug, key) as ok:
        if not ok:
            echo(f"{run_name(slug, key)}: another loop is polling this final (its lock is held); not started")
            return None
        return _run(slug, key, quali_keys, cond, interval=interval, minutes=minutes, echo=echo)


def _run(slug, key, quali_keys, cond="", interval=BASE_INTERVAL, minutes=0, echo=print):
    import requests
    s = requests.Session()
    t0 = time.time()
    late = False
    while True:
        snap = None
        try:
            snap = update(slug, key, quali_keys, cond, s, interval=LATE_INTERVAL if late else interval)
            if snap.get("waiting"):                                     # no start list yet: the base pace
                late = False
                live_q = ", ".join(snap.get("qualifying_live") or []) or "none running"
                echo(f"{snap['ts']} waiting: {len(snap['riders'])} riders from qualifying (read {snap.get('fetched')}; "
                     f"{live_q}); next look in {snap['poll_interval']} s")
            else:
                cur_interval = int((snap.get("betting") or {}).get("interval", interval))
                if cur_interval == LATE_INTERVAL and not late:               # a Men's Elite rider is on track or next to start
                    late = True
                    echo(f"{snap['ts']} Men Elite on track: polling every {LATE_INTERVAL} s")
                elif cur_interval != LATE_INTERVAL and late and not (snap.get("betting") or {}).get("late"):
                    late = False
                if not late and (snap.get("betting") or {}).get("late"):      # relaunch at the late-window pace
                    late = True
                    echo(f"{snap['ts']} late window: polling every {LATE_INTERVAL} s, fresh ${LATE_CAP:.0f} caps")
                lead = max(snap["riders"], key=lambda r: r["p1"])
                echo(f"{snap['ts']} {snap['counts']} favourite {lead['name']} {lead['p1']:.1%}"
                     + (" · FINAL OVER" if snap["done"] else ""))
                if snap["done"]:
                    return snap
        except Exception as ex:                      # noqa: BLE001  keep polling through feed hiccups
            echo(f"error: {ex}")
        if minutes and time.time() - t0 > minutes * 60:
            return None
        time.sleep(snap["poll_interval"] if snap is not None and snap.get("waiting") else LATE_INTERVAL if late else interval)


def step(spec, now=None, echo=print, **_):
    """Adapter interface: one poll of the final named by a launch spec ([feed] slug, final, quali, conditions).
    Live only (the feed is the present): now is ignored. A final that is over is left alone (its run folder is
    the record the replay reads)."""
    f = spec["feed"]
    snap, _, _ = load(f["slug"], str(f["final"]))
    if snap and snap.get("done"):
        echo(f"{f['slug']} final {f['final']}: over; nothing to do")
        return None
    s = update(f["slug"], str(f["final"]), [str(k) for k in f.get("quali", [])], f.get("conditions", ""),
               interval=spec["live"]["poll"]["interval_s"])
    echo(f"{s['ts']} {s['counts']} maker P&L {s['maker_pnl']['total']:+.2f}" + (" · FINAL OVER" if s["done"] else ""))
    return s


def run_spec(spec, minutes=0, echo=print):
    """Adapter: poll the spec's final until it is over (run(), under the run folder's lock)."""
    f = spec["feed"]
    return run(f["slug"], str(f["final"]), [str(k) for k in f.get("quali", [])], f.get("conditions", ""),
               interval=spec["live"]["poll"]["interval_s"], minutes=minutes, echo=echo)


# ---------------------------------------------------------------------------
# The demo taker's picks: chosen once, at the maker's quotes of that moment
# ---------------------------------------------------------------------------

def make_picks(slug, key, picks, stake=25.0):
    """picks: [dict(name (substring of the rider's name), market 'win'|'podium', side 'YES'|'NO', hype 1-5,
    why)]. Each is taken at the maker's current ask (YES) or 1 - bid (NO), for `stake` dollars. Written to
    picks.json with the time and price; never re-priced."""
    out = outdir(slug, key)
    snap = json.loads((out / "latest.json").read_text())
    by = {r["bib"]: r for r in snap["riders"]}
    qs = {(q["bib"], q["market"]): q for q in snap["quotes"]}
    done = []
    for p in picks:
        r = next((r for r in snap["riders"] if p["name"].lower() in r["name"].lower()), None)
        if r is None:
            raise ValueError(f"pick {p['name']!r}: no rider in the final matches that name")
        q = qs[(r["bib"], p["market"])]
        price = q["ask"] if p["side"] == "YES" else (1 - q["bid"] if q["bid"] is not None else None)
        if price is None:
            continue
        done.append(dict(p, bib=r["bib"], rider=r["name"], price=round(price, 2), stake=stake,
                         shares=round(stake / price, 2), fair_then=q["fair"], ts=snap["ts"], status_then=by[r["bib"]]["status"]))
    (out / "picks.json").write_text(json.dumps(done, indent=1))
    return done


def load(slug, key):
    return LV.load(run_name(slug, key))


def _finals():
    """The recorded downhill finals, newest first: [(slug, key)]."""
    return [tuple(e["run"].rsplit("_", 1)) for e in LV.events() if e["sport"] == "mtb_dh"]


def latest():
    """(slug, key) of the most recently updated final, however old (replay), or None."""
    f = _finals()
    return f[0] if f else None


def state(max_age_h=6):
    """'live' while a final is running (updated within max_age_h hours and not over), 'replay' once it is
    over (or stale), None when nothing was ever recorded (pipelines/live.py state, for downhill finals)."""
    cur = latest()
    return None if cur is None else LV.state(run_name(*cur), max_age_h)


def snap_times(slug, key):
    """The saved snapshots' times (their file names, e.g. 20260927T223221), oldest first."""
    return LV.snap_times(run_name(slug, key))


def load_at(slug, key, t):
    """load() as of one snapshot (the last one at or before t, a snap_times name): the snapshot, the picks,
    the history up to it."""
    return LV.load_at(run_name(slug, key), t)


def book_opened(slug, key):
    """When the private book that ran opened: its first crowd poll's time (ISO), or None."""
    return LV.book_opened(run_name(slug, key))


def book_at(slug, key, ts=None):
    """The maker's private book rebuilt from the crowd's fills up to ts (ISO; None = all): {market: dict(inv,
    cash)} (the maker's side; the same as book.json at the end) and the crowd polls up to ts, newest last."""
    return LV.book_at(run_name(slug, key), ts)


def current(max_age_h=12):
    """(slug, key) of the most recently updated live final, if updated within max_age_h hours."""
    cur = latest()
    if cur is None or time.time() - (outdir(*cur) / "latest.json").stat().st_mtime > max_age_h * 3600:
        return None
    return cur


# ---------------------------------------------------------------------------
# The Live tab's body (templates/live_mtb_dh.html)
# ---------------------------------------------------------------------------

def view(run, snap, picks, hist, mode, maker):
    """What the downhill body needs, for the maker or the taker: the leaderboard, riders on course, up next,
    the rank-probability grid with quotes, the taker's picks, the win-probability chart and the private book
    (rebuilt from the crowd's fills up to this snapshot)."""
    import pandas as pd

    from racinglines.web.viz import line_chart
    out = {}
    riders = snap["riders"]
    qs = {(q["bib"], q["market"]): q for q in snap["quotes"]}
    prev = hist[-2]["quotes"] if len(hist) >= 2 else {}
    outcomes = {(o["bib"], o["market"]): o["yes"] for o in snap.get("outcomes") or []}
    fin = sorted([r for r in riders if r["status"] == "Finished" and r["time"]], key=lambda r: r["time"])
    lead = fin[0]["time"] if fin else None
    for i, r in enumerate(fin):
        r["rank"], r["gap"] = i + 1, (r["time"] - lead) if lead else None
    on = [r for r in riders if r["status"] == "InRace"]
    nxt = sorted([r for r in riders if r.get("next") is not None], key=lambda r: r["next"])[:6]
    grid = sorted([r for r in riders if r["top10"] > 0.001 or r["status"] == "Finished"],
                  key=lambda r: (-r["top3"], -r["top10"]))[:16]
    grid += [r for r in fin[:3] if r not in grid]                        # the current leaders always show
    grid.sort(key=lambda r: (-r["top3"], -r["top10"]))
    for r in grid:
        for m in MARKETS:
            q = qs.get((r["bib"], m), {})
            old = prev.get(f"{r['bib']}:{m}") or [None, None]
            r[f"{m}_bid"], r[f"{m}_ask"] = q.get("bid"), q.get("ask")
            r[f"{m}_move"] = (None if q.get("ask") is None or old[1] is None else
                              ("up" if q["ask"] > old[1] else "down" if q["ask"] < old[1] else None))
    by = {r["bib"]: r for r in riders}
    # the taker's picks: each YES bought at the maker's ask; the maker holds the other side
    rows_, stake = [], 0.0
    for p in picks:
        q = qs.get((p["bib"], p["market"]), {})
        res = outcomes.get((p["bib"], p["market"]))
        if res is not None:
            now, state = (1.0 if res else 0.0), ("won" if res else "lost")
        else:
            mid = [x for x in (q.get("bid"), q.get("ask")) if x is not None]
            now, state = (sum(mid) / len(mid) if mid else q.get("fair")), "live"
        value = p["shares"] * now if now is not None else None
        fair = q.get("fair") if res is None else float(res)
        rows_.append(dict(p, now=now, value=value, pnl=None if value is None else value - p["stake"], state=state,
                          fair=fair, maker_pnl=None if fair is None else p["stake"] - p["shares"] * fair,
                          rider_status=by.get(p["bib"], {}).get("status")))
        stake += p["stake"]
    tot = dict(stake=stake, value=sum(r["value"] or 0 for r in rows_), pnl=sum(r["pnl"] or 0 for r in rows_),
               maker_pnl=sum(r["maker_pnl"] or 0 for r in rows_),
               won=sum(r["state"] == "won" for r in rows_), lost=sum(r["state"] == "lost" for r in rows_))
    chart = None
    if maker and len(hist) >= 2:
        top = sorted(riders, key=lambda r: -r["p1"])[:5]
        # the rider on course (else the next to start) is always drawn, dashed, even at zero
        cur_r = next((r for r in riders if r["status"] == "InRace"), None) or next(
            (r for r in sorted(riders, key=lambda r: r["next"] if r.get("next") is not None else 999)
             if r.get("next") is not None), None)
        if cur_r and cur_r not in top:
            top.append(cur_r)
        series = {r["name"]: [(pd.Timestamp(h["ts"]), h["fair"].get(f"{r['bib']}:win", 0.0)) for h in hist]
                  for r in top}
        labels = {r["name"]: r["name"] + (" (on course)" if r["status"] == "InRace" else " (next)" if r is cur_r else "")
                  for r in top}
        chart = line_chart(series, labels, money=False, h=190, highlight={cur_r["name"]} if cur_r else ())
    # the private book: the anonymous crowd, as a group, and the maker's biggest positions
    # rebuilt from the crowd's fills up to this snapshot (live: all of them; the same as book.json)
    markets, polls = LV.book_at(run, snap["ts"] if mode == "replay" else None)
    book = markets if polls and snap.get("maker_pnl") else None
    positions, recent = [], []
    if book:
        fair = {f"{q['bib']}:{q['market']}": q["fair"] for q in snap["quotes"]}
        for k, mk in markets.items():
            b_, m_ = k.split(":")
            o = outcomes.get((int(b_), m_))
            v = float(o) if o is not None else fair.get(k, 0.0)
            positions.append(dict(rider=by.get(int(b_), {}).get("name", b_), market=m_, inv=mk["inv"],
                                  pnl=mk["cash"] + mk["inv"] * v, value=v, settled=o is not None))
        positions.sort(key=lambda x: -abs(x["inv"]))
        for e in reversed(polls[-6:]):
            for f in e["fills"][:4]:
                recent.append(dict(f, ts=e["ts"][11:19], rider=by.get(f["bib"], {}).get("name", f["bib"])))
    out.update(by_bib={r["bib"]: r["name"] for r in riders})
    out.update(book=book, positions=positions[:10], recent=recent[:12], crowd=snap.get("crowd"),
               mpnl=snap.get("maker_pnl"))
    out.update(fin=fin, on=on, nxt=nxt, grid=grid, picks=rows_, tot=tot, win_chart=chart,
               age=int((pd.Timestamp.now(tz="UTC") - pd.Timestamp(snap["ts"])).total_seconds()))
    return out


# ---------------------------------------------------------------------------
# Re-pricing a final from its logged raw feed (adapter: reprice)
# ---------------------------------------------------------------------------

def reprice(slug, key, label="current", every=1, n_sims=N_SIMS, echo=print):
    """Re-run the pricing over every logged raw feed response (raw/<ts>.json.gz), with this code and the
    final's recorded inputs (meta.json: qualifying on one scale, split ratios, season priors, conditions):
    no network, no crowd, no quotes. Writes reprice/<label>.jsonl (ts, fair per market) in the run folder and
    returns dict(rows, scorecard=dict(live, repriced)) scored on the same updates against the result."""
    import gzip

    from racinglines.pipelines import live_report as R
    out = outdir(slug, key)
    meta = json.loads((out / "meta.json").read_text())
    qbest = {int(k): v for k, v in meta["qbest"].items()}
    qratio = {int(k): v for k, v in meta["qratio"].items()}
    prior = {k: tuple(v) for k, v in meta.get("prior", {}).items()}
    raws = sorted((out / "raw").glob("*.json.gz"))[::max(1, every)]
    rows, last = [], None
    for i, f in enumerate(raws):
        with gzip.open(f, "rt") as fh:
            riders = parse(json.load(fh))
        state = json.dumps([(r["bib"], r["status"], r["time"], r["splits"]) for r in riders])
        if state != last:                                  # an unchanged feed: the same prices
            sim, _ = simulate(riders, qbest, qratio, meta.get("conditions", ""), n=n_sims, prior=prior)
            fair = {f"{r['bib']}:{m}": round(r[col], 4) for r in sim for m, col in (("win", "p1"), ("podium", "top3"))}
            last = state
        stamp = f.name.split(".")[0]
        rows.append(dict(ts=f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}+00:00",
                         stamp=stamp, fair=fair))
        if i % 100 == 0:
            echo(f"  {i + 1}/{len(raws)} {stamp}")
    (out / "reprice").mkdir(exist_ok=True)
    with (out / "reprice" / f"{label}.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    snap, _, _ = load(slug, key)
    oc = R.outcomes_of(snap)
    live_rows = []
    for r in rows:                                         # what was quoted live at the same feed responses
        p = out / "snaps" / f"{r['stamp']}.json.gz"
        if p.exists():
            with gzip.open(p, "rt") as fh:
                s = json.load(fh)
            live_rows.append(dict(ts=r["ts"], fair={f"{q['bib']}:{q['market']}": q["fair"] for q in s["quotes"]}))

    def mean(sc):
        by = {}
        for x in sc:
            d = by.setdefault(x["kind"], [0, 0.0, 0.0])
            d[0] += 1
            d[1] += x["brier"]
            d[2] += x["logloss"]
        return {k: dict(updates=n, brier=b / n, logloss=ll / n) for k, (n, b, ll) in by.items()}
    have = {r["ts"] for r in live_rows}
    return dict(rows=rows, scorecard=dict(live=mean(R.scorecard(live_rows, oc)),
                                          repriced=mean(R.scorecard([r for r in rows if r["ts"] in have], oc))))
