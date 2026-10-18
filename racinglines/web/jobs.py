"""
Model runs launched from the web app.

Each job type is a CLI command with a few typed knobs; the Lab page renders the
form from this catalog, so every sport gets the same UI. Jobs run one at a time
in a background thread as a subprocess (the same code path as the command line),
stream progress into the `jobs` table and link to the model run they saved.

Guard rails:
  - forecasts launched here are saved as kind='scenario': live prices only change
    when a maker promotes one (see promote());
  - backtests and diagnostics use the same as-of pricing as the CLI (no leakage
    by construction), whatever the knobs.
"""

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, text

from racinglines import exchanges, sports
from racinglines.db import models as m
from racinglines.db.config import get_engine, get_session
from racinglines.pipelines import model_vs_market as MVM
from racinglines.pipelines import season_strategy as SE
from racinglines.pipelines import sweep_settings as SS

ROOT = Path(__file__).resolve().parents[2]      # the repository
LOG_MAX = 20_000


@dataclass
class Knob:
    name: str
    label: str
    type: str = "float"          # float | int | choice | text | datetime | event | years | legs (combo legs, JSON) | calib
    default: object = None
    min: float | None = None
    max: float | None = None
    choices: list = field(default_factory=list)
    help: str = ""


ANY_SPORT = "any"     # a job type whose `sport` knob picks the sport (the shared backtest core)


@dataclass
class JobType:
    code: str
    sport: str            # a sport code, or ANY_SPORT
    label: str
    what: str
    knobs: list
    argv: callable
    minutes: str = ""
    venues: tuple = (None,)   # the exchanges it trades or scores against (the Lab launcher's columns); None: model only

    def sport_of(self, params):
        return params.get("sport", self.sport) if self.sport == ANY_SPORT else self.sport


def modeled_sports():
    """Sports whose schema names a pricing model (sports/<code>.toml [sport] pricing_model): each one runs through
    the backtest core."""
    return sorted((code for code in sports.SPORT_CODES if sports.load(code)["sport"].get("pricing_model")),
                  key=lambda code: sports.load(code)["sport"]["display_order"])


def sport_name(code):
    s = sports.load(code)
    return s["competition"].get("display_name") or s["sport"]["name"]


def venue_name(code):
    """'polymarket' -> 'Polymarket', 'og' -> 'OG.com' (exchanges/<code>.toml), None -> 'Model only'."""
    if code is None:
        return "Model only"
    if code in exchanges.CODES:
        return exchanges.load(code)["exchange"]["name"]
    from racinglines.markets import venues as V
    return next((v.name for v in V.VENUES if v.code == code), code)


def sport_venues(code):
    """The exchanges that list a sport: its schema's [markets] venues and every exchange schema naming it."""
    listed = [v for v in sports.load(code).get("markets", {}).get("venues", []) if v != "private"]
    return listed + [x for x in exchanges.CODES if code in exchanges.sports(x) and x not in listed]


# model variants offered for the sweep: each switch alone, and the combinations worth comparing
MODEL_CHOICES = ["baseline", "grid", "gridq", "pretrain", "gbm", "tail", "reset", "gridq+pretrain", "gridq+reset",
                 "pretrain+reset", "gridq+pretrain+reset", "gridq+pretrain+tail"]


# the switches a variant combines (models/variants.py), offered as checkboxes in the sweep form; fastlap
# and flpos both draw the fastest lap, so only one of them can be on
EXCLUSIVE_SWITCHES = ("fastlap", "flpos")


def variant_switches():
    """[(switch, description)] in variants.SWITCHES order, baseline left out (it is no switch at all)."""
    from racinglines.models import variants as V
    return [(k, V.DESCRIPTION.get(k, "")) for k in V.SWITCHES if k != "baseline"]


def _sweep_argv(p):
    st = SS.Settings.from_dict(p.get("settings") or {})
    return ["-m", "racinglines", "strategy", "sweep", "f1", "--variant", st["variant"], "--year", str(p.get("year", 2026)),
            "--no-fetch", "--save", *st.argv()]


def _sport_sweep_argv(sport):
    def argv(p, out):
        from racinglines.pipelines import season_sweep as SW

        settings = SW.settings_class(sport).from_dict(p.get("settings") or {})
        return ["-m", "racinglines", "strategy", "sweep", sport, "--year", str(p.get("year", 2026)), "--save", *settings.argv()]
    return argv


def _walk_forward_argv(p, out):
    return (["-m", "racinglines", "backtest", "walk-forward", p["sport"], "--save", "--out-dir", str(Path(out).with_suffix(""))]
            + (["--model", "global"] if p["model"] == "global" else [])
            + (["--seasons", *p["seasons"].split()] if p["seasons"] else [])
            + (["--venue", p["venue"]] if p["venue"] != "none" else []))


def live_variant():
    """The model variant the live book prices with: sports/f1.toml [live.sources] live's profile
    (pipelines/profiles.py)."""
    from racinglines import sports
    from racinglines.pipelines import profiles as PF
    prof = sports.load("f1")["live"]["sources"]["live"].get("profile", "C")
    return PF.PROFILES[prof]["settings"].get("variant", "baseline")


def taker_variant():
    """The model variant the core taker prices with (pipelines/profiles.py profile A): sportsbook combos are taker
    bets, so they default to it, the variant of the Singapore stage runs the picks used."""
    from racinglines.pipelines import profiles as PF
    return PF.PROFILES["A"]["settings"].get("variant", "baseline")


def combo_variants():
    """The variants the f1_combo job offers: the taker's plus flpos first (the default), then the live book's plus
    flpos, then the taker's with fastlap, then flpos alone."""
    out = []
    for v in (f"{taker_variant()}+flpos", f"{live_variant()}+flpos", f"{taker_variant()}+fastlap", "flpos"):
        v = v.removeprefix("baseline+")
        if v not in out:
            out.append(v)
    return out


def _combo_argv(p):
    return (["-m", "racinglines", "model", "combo", "f1", "--variant", p["variant"], "--half-life", str(p["half_life"]),
             "--event", p["event"], "--legs", p["legs"], "--sims", str(p["sims"]), "--save"]
            + ([] if p["cutoff"] == "now" else ["--cutoff", p["cutoff"]])
            + (["--no-track"] if p["track"] == "off" else [])
            + (["--calib-power", p["calib_power"]] if p.get("calib_power") else []))


def _f1_common(p, command):
    return (["-m", "racinglines", "model", command, "f1", "--half-life", str(p["half_life"])]
            + (["--calib-power", p["calib_power"]] if p.get("calib_power") else []))


def _setting(name):
    return next(s for s in SS.SETTINGS if s.name == name)


# F1 knobs that are also sweep settings take their default and range from the schema (pipelines/sweep_settings.py), so
# the Lab, the sweep flags and MCP can't drift apart; simulation counts keep each job's own default (a backtest prices
# many races, a forecast one season)
HALF_LIFE = Knob("half_life", _setting("half_life_days").label, "float", _setting("half_life_days").default,
                 _setting("half_life_days").min, _setting("half_life_days").max,
                 help="How fast old races fade from the pace models.")


def _sims(default, label="Simulations"):
    return Knob("sims", label, "int", default, _setting("sims").min, _setting("sims").max)
CALIB = Knob("calib_power", _setting("calib_power").label, "calib", None, help=_setting("calib_power").help)
F1_TRACK = Knob("track", "Track/sector features", "choice", "on", choices=["on", "off"],
                help="Use the sector-type car model at each circuit.")

CATALOG = {j.code: j for j in [
    JobType("walk_forward", ANY_SPORT, "Walk-forward backtest (model vs market)", "Price every past event of the sport "
            "from data before it, through the shared backtest core, and score the model per market kind; with an "
            "exchange, that exchange's price at the same moment is scored beside it on the markets linked by exact keys.",
            [Knob("sport", "Sport", "choice", "f1", choices=modeled_sports()),
             Knob("venue", "Score against", "choice", "none", choices=["none", *MVM.VENUES],
                  help="An exchange's linked markets, priced at the same moment as the model."),
             Knob("model", "Model", "choice", "own", choices=["own", "global"],
                  help="own: the sport's pricing model; global: the sport-agnostic results model."),
             Knob("seasons", "Seasons", "years", "", help="Blank: every season the model can score.")],
            _walk_forward_argv, "a few minutes per season", venues=(None, *MVM.VENUES)),
    JobType("f1_backtest", "f1", "Backtest", "Price past races as of just before qualifying and just before the race, "
            "then score against results and the grid-only baseline.",
            [Knob("races", "Last N races", "int", 30, 3, 200, help="Fewer races = faster."),
             HALF_LIFE,
             Knob("sims", "Simulations per race", "int", 2000, 200, 10000),
             Knob("track", "Track/sector features", "choice", "both", choices=["both", "on", "off"],
                  help="'both' runs twice to compare.")],
            lambda p, out: _f1_common(p, "backtest") + ["--races", str(p["races"]), "--sims", str(p["sims"]),
                                            "--track", p["track"], "--start-year", "2021", "--out", out, "--save"],
            "~1 min per 30 races"),
    JobType("f1_scenario", "f1", "Forward forecast (scenario)", "Price every remaining race and the championships "
            "with data up to now. Saved as a scenario; promote it to make it the live fair prices.",
            [Knob("label", "Scenario name", "text", "my scenario"), HALF_LIFE,
             _sims(10000), F1_TRACK],
            lambda p, out: _f1_common(p, "forecast") + ["--sims", str(p["sims"]), "--save", "--scenario", p["label"]]
            + (["--no-track"] if p["track"] == "off" else []),
            "~1-2 min"),
    JobType("f1_diagnostic", "f1", "Event diagnostic", "Price one past race as of a cutoff (e.g. the evening before), "
            "then compare with Polymarket at that time and the result.",
            [Knob("event", "Event", "event"), Knob("cutoff", "As of (UTC)", "datetime"), HALF_LIFE,
             _sims(10000), F1_TRACK, CALIB],
            lambda p, out: _f1_common(p, "diagnostic") + ["--event", p["event"], "--cutoff", p["cutoff"],
                                            "--sims", str(p["sims"]), "--save"]
            + (["--no-track"] if p["track"] == "off" else []),
            "~1 min", venues=("polymarket",)),
    JobType("f1_combo", "f1", "Combo (same-game parlay) prices", "Price one event (past or upcoming) once as of a "
            "cutoff and read each combo's fair value from the same simulations, with every leg's marginal, their "
            "product and the lift; pole and fastest-lap legs are flagged when the simulation's correlation is off "
            "history. Saved as a model run (kind 'combo') with the prices in its metrics.",
            [Knob("event", "Event", "event"),
             Knob("cutoff", "As of (UTC)", "datetime", "now", help="YYYY-MM-DDTHH:MM, or now"),
             Knob("legs", "Legs (JSON)", "legs",
                  help='A list of legs, a list of combos, or {"name": [legs]}; a leg is {"kind": "race_win", "driver": '
                       '"Max Verstappen"} with "driver" / "athlete" / "team" / "pair" / "line" / "side" as the kind '
                       'needs (markets/combos.py)'),
             Knob("variant", "Model variant", "choice", combo_variants()[0], choices=combo_variants(),
                  help="the live book's variant plus flpos (the fastest lap from the simulated result) by default"),
             HALF_LIFE, _sims(10000), F1_TRACK, CALIB],
            lambda p, out: _combo_argv(p),
            "~1 min"),
    JobType("f1_sweep", "f1", "Edge Finder sweep (every weekend of the season)", "Every raced weekend of the "
            "season, priced with these settings before any running and after each session, then traded on "
            "Polymarket by every taker and maker strategy and settled on the result. The run joins the Edge Finder. "
            "Stages already priced with the same model settings and data are reused.",
            [Knob("year", "Season", "choice", "2026", choices=["2026", "2025"]),
             Knob("settings", "Settings", "sweep_settings", None)],
            lambda p, out: _sweep_argv(p),
            "a few minutes per season (GBM longer)", venues=SS.VENUES),
    JobType("f1_season_strategy", "f1", "Season strategy (championships)", "Replay the default championship-market "
            "strategy through the season on the exchange's recorded prices: as-of season forecasts pre-season and after "
            "every race, rebalance, settle eliminated markets, mark the rest. Forecasts already computed are reused.",
            [Knob("venue", "Exchange", "choice", "polymarket", choices=list(SE.VENUES),
                  help="Kalshi and OG.com replay their stored tape."),
             Knob("min_edge", "Min edge to act (prob.)", "float", 0.03, 0.005, 0.5),
             Knob("stake_per_edge", "Stake per unit edge ($)", "float", 500, 10, 10000),
             Knob("max_stake", "Max stake per market ($)", "float", 150, 1, 10000),
             Knob("capital", "Capital cap ($)", "float", 1500, 50, 100000)],
            lambda p, out: ["-m", "racinglines", "strategy", "season-strategy", "f1", "--no-fetch", "--save", "--min-edge", str(p["min_edge"]),
                            "--stake-per-edge", str(p["stake_per_edge"]), "--max-stake", str(p["max_stake"]),
                            "--capital", str(p["capital"]), "--venue", p.get("venue", "polymarket")],
            "~1 min once forecasts exist", venues=SE.VENUES),
    JobType("dh_scenario", "mtb_dh", "Forward forecast (scenario)", "Simulate the remaining downhill rounds and the "
            "championship. Saved as a scenario; promote it to make it the live fair prices.",
            [Knob("label", "Scenario name", "text", "my scenario"),
             Knob("half_life", "Recency half-life (days)", "float", 240, 20, 2000),
             Knob("junior_weight", "Weight of junior results", "float", 0.25, 0, 1),
             _sims(10000)],
            lambda p, out: ["-m", "racinglines", "model", "forecast", "mtb_dh", "--db", "--save", "--scenario", p["label"],
                            "--half-life-days", str(p["half_life"]), "--junior-weight", str(p["junior_weight"]),
                            "--sims", str(p["sims"]), "--out-dir", str(Path(out).parent.parent / "mtb_dh" / "season")],
            "~1 min"),
    JobType("dh_backtest", "mtb_dh", "Backtest", "Walk forward through past downhill seasons, predicting each round "
            "from earlier ones only.",
            [Knob("half_life", "Recency half-life (days)", "float", 240, 20, 2000),
             Knob("junior_weight", "Weight of junior results", "float", 0.25, 0, 1),
             Knob("sims", "Simulations", "int", 2000, 200, 10000)],
            lambda p, out: ["-m", "racinglines", "model", "backtest", "mtb_dh", "--db", "--save", "--half-life-days", str(p["half_life"]),
                            "--junior-weight", str(p["junior_weight"]), "--sims", str(p["sims"]),
                            "--out-dir", str(Path(out).parent.parent / "mtb_dh" / "backtests")],
            "a few min"),
]}

from racinglines.pipelines import season_sweep as _SEASON_SWEEP

for _sport in sports.SPORT_CODES:
    if _sport == "f1" or not _SEASON_SWEEP.supports(_sport):
        continue
    _settings = _SEASON_SWEEP.settings_class(_sport)
    _venues = _SEASON_SWEEP.venues(_sport)
    _code = f"{_sport}_sweep"
    CATALOG[_code] = JobType(
        _code, _sport, f"{sport_name(_sport)} strategy sweep",
        "Replay the season on the selected exchange's stored prices, then calculate every taker and maker strategy's "
        "simulated P&L weekend by weekend. Saved runs appear in Edge Finder.",
        [Knob("year", "Season", "choice", "2026", choices=["2026", "2025"]),
         Knob("settings", "Settings", "sport_sweep_settings", None)],
        _sport_sweep_argv(_sport), "a few minutes per season", venues=_venues)


def launcher():
    """The Lab launcher: a row per sport with a pricing model, a column per exchange any of them is listed on (and one
    for the model alone), each cell the job types that run for that sport there. -> (venues, rows); a row is
    dict(code, name, cells: {venue: None (not listed there) or [job types]}). Read from the schemas and this catalog."""
    codes = modeled_sports()
    listed = {code: sport_venues(code) for code in codes}
    venues = [None] + [v for v in dict.fromkeys(v for code in codes for v in listed[code])]
    out = []
    for code in codes:
        cells = {v: [jt for jt in CATALOG.values() if jt.sport in (code, ANY_SPORT) and v in jt.venues]
                 if v is None or v in listed[code] else None for v in venues}
        out.append(dict(code=code, name=sport_name(code), cells=cells))
    return venues, out


def parse(job_type, form):
    """Validate a submitted form against the job's knobs. Returns params or raises ValueError."""
    out = {}
    for k in job_type.knobs:
        if k.type in ("sweep_settings", "sport_sweep_settings"):
            settings_cls = SS.Settings
            if k.type == "sport_sweep_settings":
                from racinglines.pipelines import season_sweep as SW

                settings_cls = SW.settings_class(job_type.sport)
            out[k.name] = parse_sweep_settings(form, settings_cls)
            continue
        raw = (form.get(k.name) or "").strip()
        if raw == "" and k.default is not None:
            raw = str(k.default)
        if k.type in ("float", "int"):
            v = float(raw) if k.type == "float" else int(float(raw))
            if (k.min is not None and v < k.min) or (k.max is not None and v > k.max):
                raise ValueError(f"{k.label} must be between {k.min:g} and {k.max:g}")
            out[k.name] = v
        elif k.type == "choice":
            if raw not in k.choices:
                raise ValueError(f"{k.label}: pick one of {k.choices}")
            out[k.name] = raw
        elif k.type == "datetime" and k.default == "now" and raw == "now":
            out[k.name] = raw
        elif k.type == "legs":
            out[k.name] = parse_legs(raw)
        elif k.type == "calib":                                  # off / walk_forward / a power (models/calibration.py)
            from racinglines.models import calibration as CAL
            out[k.name] = CAL.parse(raw)
        elif k.type == "datetime":
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", raw):
                raise ValueError(f"{k.label}: use YYYY-MM-DDTHH:MM")
            out[k.name] = raw
        elif k.type == "years":
            if raw and not re.fullmatch(r"\d{4}( \d{4})*", " ".join(raw.replace(",", " ").split())):
                raise ValueError(f"{k.label}: seasons as years, e.g. 2025 2026")
            out[k.name] = " ".join(raw.replace(",", " ").split())
        elif k.type == "event":
            if not re.fullmatch(r"\d{4}-\d{1,2}", raw):
                raise ValueError(f"{k.label}: pick an event")
            out[k.name] = raw
        else:
            if not raw or len(raw) > 60 or not re.fullmatch(r"[\w .,'()-]+", raw):
                raise ValueError(f"{k.label}: letters, numbers and spaces, up to 60 characters")
            out[k.name] = raw
    return out


def parse_legs(raw):
    """A combo job's legs (combo_job.parse_combos' JSON), checked without the entry list: each leg's kind can be
    priced per simulation and names what it needs (a driver counts as an athlete). Returns compact JSON."""
    import json
    from racinglines.markets import combos as C
    from racinglines.models import combo_job as CJ
    if not raw or len(raw) > 4000:
        raise ValueError("Legs: JSON, up to 4000 characters")
    combos = CJ.parse_combos(raw)
    for name, legs in combos:
        stand_in = [dict(leg, athlete=leg.get("athlete", leg.get("driver")),
                         pair=leg.get("pair")) if isinstance(leg, dict) else leg for leg in legs]
        stand_in = [{k: v for k, v in leg.items() if v is not None and k != "driver"} if isinstance(leg, dict) else leg
                    for leg in stand_in]
        try:
            C.check_legs(stand_in)
        except ValueError as e:
            raise ValueError(f"Legs ({name}): {e}") from None
    return json.dumps({n: legs for n, legs in combos} if len(combos) > 1 or combos[0][0] != "combo 1"
                      else combos[0][1], separators=(",", ":"))


def parse_sweep_settings(form, settings_cls=SS.Settings):
    """{name: value} of the settings that differ from the defaults (validated), from the Lab form: one
    field per setting; booleans are checkboxes (with a hidden 'false' before each), multi-choice
    settings are checkbox groups."""
    getlist = form.getlist if hasattr(form, "getlist") else (
        lambda k: [] if form.get(k) in (None, "") else (form[k] if isinstance(form[k], list) else [form[k]]))
    d = {}
    for s in settings_cls.SPEC:
        if s.type == "multi":
            if form.get(f"{s.name}__present") is None:
                continue
            d[s.name] = [v for v in getlist(s.name) if v]
        elif s.type == "bool":
            vals = getlist(s.name)
            if vals:
                d[s.name] = "true" in vals
        else:
            raw = (form.get(s.name) or "").strip()
            if raw != "":
                d[s.name] = raw
    st = settings_cls.from_dict(d)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in st.changed().items()}


def submit(job_type, params, user_id, engine=None):
    """Queue a job (the caller has validated `params` with parse()). `engine`: a database other than the default one."""
    from sqlalchemy.orm import sessionmaker
    with (sessionmaker(engine, expire_on_commit=False)() if engine is not None else get_session()) as s:
        job = m.Job(kind=job_type.code, sport=job_type.sport_of(params), params=params, user_id=user_id, status="queued")
        s.add(job)
        s.commit()
        job_id = job.id
    _wake.set()
    return job_id


def promote(session, run_id):
    """Make a scenario run the live forecast for its competition/category."""
    run = session.get(m.ModelRun, run_id)
    if run is None or run.kind not in ("scenario", "forecast"):
        raise ValueError("only forecast scenarios can be promoted")
    run.kind = "forecast"
    run.params = dict(run.params or {}, promoted_at=datetime.now(timezone.utc).isoformat())
    session.commit()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_wake = threading.Event()
_started = False


def start_worker(reset_running=True):
    """Start the worker thread. `reset_running`: mark jobs still 'running' as failed first (the web app on start: those
    were its own subprocesses, gone with the restart); the MCP server passes False, since a running job may be the web app's."""
    global _started
    if _started:
        return
    _started = True
    if reset_running:
        with get_engine().begin() as c:
            c.execute(text("""UPDATE jobs SET status = 'failed', finished_at = now(),
                              progress = 'interrupted: the app restarted' WHERE status = 'running'"""))
    threading.Thread(target=_loop, daemon=True, name="jobs").start()


def _loop():
    while True:
        try:
            if not _run_next():
                _wake.wait(timeout=5)
                _wake.clear()
        except Exception as ex:  # noqa: BLE001  keep the worker alive
            print(f"jobs worker error: {ex}", file=sys.stderr)
            time.sleep(5)


def _run_next():
    with get_session() as s:
        job = s.scalars(select(m.Job).where(m.Job.status == "queued").order_by(m.Job.id)
                        .with_for_update(skip_locked=True).limit(1)).first()
        if job is None:
            return False
        jt = CATALOG[job.kind]
        from racinglines import paths
        out = paths.runs("jobs") / f"job_{job.id}.csv"
        argv = [sys.executable] + jt.argv(job.params, str(out))
        job.argv, job.status, job.started_at, job.progress = argv[1:], "running", datetime.now(timezone.utc), "starting"
        s.commit()
        job_id = job.id
    log, last_write, status, run_id = [], 0.0, "failed", None
    try:
        # the job's log is shown as text (the Lab, get_job): Rich layout and emojis without escape codes (cli/tty.py)
        p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env=dict(os.environ, NO_COLOR="1"))
        progress = "running"
        for line in p.stdout:
            line = line.rstrip()
            if line.startswith("INFO ") or not line:
                continue
            log.append(line)
            if line.startswith("progress "):
                progress = line[9:][:200]
            mt = re.search(r"Saved (?:[\w-]+ )*run (\d+)", line)
            if mt:
                run_id = int(mt.group(1))
            if time.time() - last_write > 1.5:
                _update(job_id, progress=progress, log="\n".join(log)[-LOG_MAX:])
                last_write = time.time()
        rc = p.wait()
        status = "done" if rc == 0 else "failed"
        progress = "finished" if rc == 0 else f"exit code {rc}"
    except Exception as ex:  # noqa: BLE001
        log.append(f"error: {ex}")
        progress = f"error: {ex}"[:200]
    _update(job_id, status=status, progress=progress, finished_at=datetime.now(timezone.utc),
            result_run_id=run_id, log="\n".join(log)[-LOG_MAX:])
    return True


def _update(job_id, **values):
    with get_session() as s:
        job = s.get(m.Job, job_id)
        for k, v in values.items():
            setattr(job, k, v)
        s.commit()
