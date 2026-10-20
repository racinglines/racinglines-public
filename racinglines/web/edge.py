"""
Edge Finder: the Lab's headline view. A user's chosen combos (a sweep configuration x a strategy),
each read from the latest full-season sweep saved with exactly that configuration (every weekend of
the season raced so far): headline cards, a full-season recap, and P&L per weekend side by side.

A configuration is a complete set of sweep settings (racinglines/pipelines/sweep_settings.py). A combo
refers to one as "cfg:<settings key>", or by a model variant's name, meaning that variant with every
other setting at its default. The selection is stored per user in users.prefs["edge_finder"] as
[[ref, strategy], ...] (default: the baseline with every strategy), the season in prefs["edge_year"].

Nothing here simulates: it only reads saved runs. Candidates (kind='candidate') are named
configurations x strategy that the Lab's Edge Finder sweep form can start from.
"""

import math

import pandas as pd
from sqlalchemy import bindparam, text

from racinglines.models import evaluate as EV
from racinglines.pipelines import profiles as PF
from racinglines.pipelines import sweep_settings as SS
from racinglines.web import prefs as P

STRATEGY_LABEL = dict(EV.STRATEGIES)
STRATEGY_KEYS = [k for k, _ in EV.STRATEGIES]
DEFAULT = [("baseline", k) for k in STRATEGY_KEYS]
MAX_COMBOS = 30
IN_SAMPLE = {"early"}          # rules drawn from the sweep they're scored on
YEARS = (2026, 2025)
# The benchmark every combo is measured against, pinned so it can't be removed or replaced: the
# conservative maker replay (quote fair ± 2c, conservative fills) on the baseline model at default
# settings. Other configurations never replace it.
BENCHMARK_STRATEGY = "maker"


def label(strategy):
    return STRATEGY_LABEL[strategy].replace("Taker: ", "").replace("Maker: ", "Maker, ")


def ref_key(ref):
    """Settings key of a combo's configuration reference, or None if it isn't one."""
    if isinstance(ref, str) and ref.startswith("cfg:") and len(ref) == 16:
        return ref[4:]
    try:
        return SS.Settings.from_dict({"variant": ref}).key
    except (ValueError, TypeError):
        return None


def valid(ref, strategy):
    return strategy in STRATEGY_LABEL and ref_key(ref) is not None


def combos(conn, user_id):
    """The user's combos, or the default."""
    got = P.get(conn, user_id).get("edge_finder")
    if got is None:
        return list(DEFAULT)
    return [(v, s) for v, s in got if valid(v, s)]


def save(conn, user_id, cs):
    out = list(dict.fromkeys((v, s) for v, s in cs if valid(v, s)))[:MAX_COMBOS]
    P.put(conn, user_id, "edge_finder", [list(c) for c in out])
    return out


def run_sport(params):
    """The sport a sweep run priced: its params' `sport`, or F1 for the runs saved before sweeps recorded it."""
    return (params or {}).get("sport") or "f1"


def run_venue(params):
    """The exchange a sweep run traded: its settings' `venue`, or Polymarket while unset."""
    return (params or {}).get("venue") or SS.venue_of((params or {}).get("settings") or {})


def settings_class(sport):
    from racinglines.pipelines import season_sweep as SW

    return SW.settings_class(sport)


def _resolve(ref, cfgs, sport):
    """Resolve the user's combo reference against this sport's settings schema."""
    match = next((key for key, cfg in cfgs.items() if cfg["ref"] == ref), None)
    if match is not None:
        return match
    if isinstance(ref, str) and ref.startswith("cfg:"):
        return ref_key(ref)
    try:
        return settings_class(sport).from_dict({"variant": ref}).key
    except (ValueError, TypeError):
        return None


def scope(conn, user_id):
    """The user's Edge Finder filter (sport, venue): F1 unless another sport is chosen, venue None for all; stored in
    prefs["edge_scope"]."""
    got = P.get(conn, user_id).get("edge_scope") or {}
    return got.get("sport") or "f1", got.get("venue") or None


def scopes(conn, year=2026):
    """(sports, venues) that have a sweep of `year`, for the filter's choices (sorted, F1 and Polymarket first)."""
    rows = conn.execute(text("SELECT params FROM model_runs WHERE kind = 'sweep' AND (params->>'year')::int = :y"),
                        dict(y=year)).scalars().all()
    sp = {run_sport(r) for r in rows}
    ve = {run_venue(r) for r in rows}
    return (sorted(sp, key=lambda x: (x != "f1", x)), sorted(ve, key=lambda x: (x != SS.DEFAULT_VENUE, x)))


def year(conn, user_id):
    y = P.get(conn, user_id).get("edge_year")
    return y if y in YEARS else YEARS[0]


def apply(cs, action, ref="", strategy=""):
    """New combo list after one edit. add_model: the configuration x every strategy already shown (all
    strategies if none); add_strategy: the strategy x every configuration already shown (baseline if none)."""
    cs = list(cs)
    same = lambda a, b: ref_key(a) == ref_key(b)                                    # noqa: E731
    has = lambda r, s: any(same(v, r) and t == s for v, t in cs)                     # noqa: E731
    if action == "reset":
        return list(DEFAULT)
    if action == "clear":
        return []
    if action in ("add", "remove", "toggle") and not valid(ref, strategy):
        raise ValueError("unknown configuration or strategy")
    if action == "add" or (action == "toggle" and not has(ref, strategy)):
        return cs + [(ref, strategy)] if not has(ref, strategy) else cs
    if action in ("remove", "toggle"):
        return [c for c in cs if not (same(c[0], ref) and c[1] == strategy)]
    if action == "add_model":
        if ref_key(ref) is None:
            raise ValueError("unknown configuration")
        strategies = list(dict.fromkeys(s for _, s in cs)) or STRATEGY_KEYS
        return cs + [(ref, s) for s in strategies if not has(ref, s)]
    if action == "remove_model":
        return [c for c in cs if not same(c[0], ref)]
    if action == "add_strategy":
        if strategy not in STRATEGY_KEYS:
            raise ValueError("unknown strategy")
        models = list(dict.fromkeys(v for v, _ in cs)) or ["baseline"]
        return cs + [(v, strategy) for v in models if not has(v, strategy)]
    raise ValueError(f"unknown action {action!r}")


# ---------------------------------------------------------------------------------------------------
# saved sweeps -> configurations
# ---------------------------------------------------------------------------------------------------

def configs(conn, year=2026, sport="f1", venue=None):
    """{settings key: dict(key, ref, settings, label, run_id, weekends, data_key)} for every
    configuration with a full-season sweep of `year` (as many weekends as the most complete sweep of
    that season on the same venue); the latest run per configuration. One sport at a time (F1 by default, also for
    None), since a configuration's key doesn't name its sport; venue: only that exchange's sweeps (None: all)."""
    sport = sport or "f1"
    rows = conn.execute(text("""SELECT id, params, jsonb_array_length(coalesce(metrics->'weekends', '[]'::jsonb)) n
                                FROM model_runs WHERE kind = 'sweep' AND (params->>'year')::int = :y ORDER BY id"""),
                        dict(y=year)).fetchall()
    rows = [r for r in rows if run_sport(r[1]) == (sport or "f1") and (venue is None or run_venue(r[1]) == venue)]
    venue_of = run_venue
    full = {}                          # per venue: Kalshi listed a 2025 weekend Polymarket didn't (Imola)
    for r in rows:
        if not ((r[1] or {}).get("rounds") and "settings" in (r[1] or {})):
            full[venue_of(r[1])] = max(full.get(venue_of(r[1]), 0), r[2])
    out = {}
    for rid, params, n in rows:
        if n < full.get(venue_of(params), 0) or n == 0 or ((params or {}).get("rounds") and "settings" in (params or {})):
            continue                                   # partial seasons (explicit rounds) never count
        try:
            st = settings_class(sport).from_run_params(params)
        except ValueError:
            continue
        plain = set(st.changed()) <= {"variant"}                   # only the model variant differs
        ref = st.get("variant", "baseline") if plain else "cfg:" + st.key
        out[st.key] = dict(key=st.key, ref=ref, settings=st, label=st.label(),
                           run_id=int(rid), weekends=int(n), data_key=(params or {}).get("data_key"),
                           venue=venue_of(params))
    return out


def _run(conn, rid):
    r = conn.execute(text("SELECT id, params, metrics FROM model_runs WHERE id = :i"), dict(i=rid)).fetchone()
    return dict(id=int(r[0]), params=r[1] or {}, metrics=r[2] or {}) if r else None


def clv_of(metrics, strategy):
    """A strategy's closing-line value from a saved sweep's metrics.clv (pipelines/clv.py sweep_summary): its
    by_strategy row, or None (a sweep saved before CLV was, or one whose closes could not be read)."""
    rows = ((metrics or {}).get("clv") or {}).get("by_strategy") or []
    return next((r for r in rows if r.get("strategy") == strategy), None)


def strategy_rows(metrics):
    """A saved sweep's strategies side by side for its run page: P&L, amount bought or filled, weekends up (totals)
    and closing-line value (metrics.clv): bets, decided, mean CLV in points and percent, share positive."""
    out = []
    for s, t in ((metrics or {}).get("totals") or {}).items():
        c = clv_of(metrics, s) or {}
        out.append(dict(strategy=s, pnl=t.get("pnl"), bought=t.get("bought"), weekends_up=t.get("weekends_up"),
                        weekends=t.get("weekends"), bets=c.get("bets"), decided=c.get("decided"),
                        mean_clv=c.get("mean_clv"), mean_clv_pct=c.get("mean_clv_pct"),
                        positive_share=c.get("positive_share")))
    return out


def recap(weekends, strategy):
    """Full-season stats of one strategy from a sweep's per-weekend rows."""
    ws = sorted(weekends or [], key=lambda w: (w.get("round") or 0))
    pnl = pd.Series([w.get(f"{strategy}_pnl") for w in ws], dtype=float)
    traded = pnl.dropna()
    maker = strategy.startswith("maker")
    vol = sum((w.get(f"{strategy}_notional" if maker else f"{strategy}_bought") or 0) for w in ws)
    total = float(traded.sum())
    eq = traded.cumsum()
    dd = float((eq - eq.cummax().clip(lower=0)).min()) if len(eq) else 0.0
    sd = float(traded.std()) if len(traded) > 1 else float("nan")
    out = dict(pnl=total, volume=vol, roi=total / vol if vol else None, weekends=len(traded),
               up=int((traded > 0).sum()), avg=float(traded.mean()) if len(traded) else None, sd=sd,
               best=float(traded.max()) if len(traded) else None, worst=float(traded.min()) if len(traded) else None,
               drawdown=dd, consistency=(float(traded.mean()) / sd * math.sqrt(len(traded))
                                         if len(traded) > 1 and sd and sd > 0 else None), fills=None, markout=None)
    if maker:
        fills = [w.get(f"{strategy}_fills") for w in ws if w.get(f"{strategy}_fills") is not None]
        mo = [(w.get(f"{strategy}_markout_60m"), w.get(f"{strategy}_fills") or 0) for w in ws
              if w.get(f"{strategy}_markout_60m") is not None]
        out.update(fills=int(sum(fills)) if fills else None,
                   markout=(sum(m * f for m, f in mo) / sum(f for _, f in mo)) if mo and sum(f for _, f in mo) else None)
    return out


def demo_pools(conn):
    """Saved F1 Polymarket results for the demo maker/taker decision pools and named history phases."""
    from racinglines.pipelines import profiles as PF
    from racinglines.pipelines import story

    years = (2025, 2026)
    configs_by_year = {year: configs(conn, year, "f1", "polymarket") for year in years}
    wanted_keys = {SS.Settings.from_dict(settings).key for settings, _ in story.POOL + story.TAKER_POOL}
    wanted_keys.add(SS.Settings.from_dict(PF.TAKER_PROFILES["T1"]["settings"]).key)
    run_ids = sorted({cfg["run_id"] for cfgs in configs_by_year.values()
                      for key, cfg in cfgs.items() if key in wanted_keys})
    metrics_by_id = {}
    if run_ids:
        stmt = text("SELECT id, metrics FROM model_runs WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True))
        metrics_by_id = {int(r[0]): (r[1] or {}) for r in conn.execute(stmt, dict(ids=run_ids)).all()}

    profiles = {**PF.PROFILES, **PF.HISTORY_PROFILES}
    pool_specs = (
        ("Demo maker", "maker", story.POOL,
         "The fixed maker pool behind M1, M2, M3 and C. These are saved Polymarket F1 season sweeps, not runs launched by opening Lab."),
        ("Demo taker", "taker", story.TAKER_POOL,
         "The fixed taker pool behind TW1 and TW2. T1/A is shown separately: it is a recommended profile, not one of these walk-forward pool rows."),
    )
    out = []
    for name, username, pool, description in pool_specs:
        phase_by_combo = {}
        phases = []
        for code, year, first_round, last_round in PF.HISTORY.get(username, []):
            profile = profiles.get(code)
            if profile is None:
                continue
            settings = SS.Settings.from_dict(profile["settings"])
            rounds = f"rounds {first_round}–{last_round}" if last_round < 99 else f"round {first_round}–season end"
            phase = dict(code=code, name=profile["name"], period=f"{year} · {rounds}",
                         strategy=profile["strategy"], settings=settings.label(), why=profile["why"])
            phases.append(phase)
            phase_by_combo.setdefault((settings.key, profile["strategy"]), []).append(phase)

        rows = []
        for settings_dict, strategies in pool:
            settings = SS.Settings.from_dict(settings_dict)
            cfgs = {year: configs_by_year[year].get(settings.key) for year in years}
            for strategy in strategies:
                results = {}
                for year in years:
                    cfg = cfgs[year]
                    if cfg is None:
                        results[year] = None
                        continue
                    metrics = metrics_by_id.get(cfg["run_id"], {})
                    results[year] = dict(recap(metrics.get("weekends"), strategy),
                                         run_id=cfg["run_id"], config=cfg["label"])
                rows.append(dict(settings=settings.label(), strategy=strategy,
                                 phases=phase_by_combo.get((settings.key, strategy), []), results=results))

        decision_checks = []
        for index, decision in enumerate(story.decisions(conn, taker=username == "taker")):
            expected = phases[min(index, len(phases) - 1)] if phases else None
            required = decision["known"] if decision["known"] not in (0, 99) else \
                (story.SEASON_WEEKENDS if decision["known"] == 99 else 0)
            saved_evidence = all(row["results"][2025] and row["results"][2025]["weekends"] >= required
                                 for row in rows) if required else True
            complete_pool = decision["known"] == 0 or (
                decision["candidates"] == len(rows) and saved_evidence)
            checked = dict(decision, expected_phase=expected, pool_complete=complete_pool, history_matches=None)
            if decision["table"] and expected and complete_pool:
                expected_profile = profiles[expected["code"]]
                expected_key = SS.Settings.from_dict(expected_profile["settings"]).key
                checked["history_matches"] = (
                    SS.Settings.from_dict(decision["chosen_settings"]).key == expected_key
                    and decision["chosen_strategy"] == expected_profile["strategy"])
            decision_checks.append(checked)

        out.append(dict(name=name, username=username, description=description, phases=phases, rows=rows,
                        pool_size=len(rows), decision_checks=decision_checks))

    taker = next(p for p in out if p["username"] == "taker")
    t1 = PF.TAKER_PROFILES["T1"]
    t1_settings = SS.Settings.from_dict(t1["settings"])
    taker["reference"] = dict(name=t1["name"], why=t1["why"], strategy=t1["strategy"],
                              settings=t1_settings.label(), results={
                                  year: (dict(recap(metrics_by_id.get(cfg["run_id"], {}).get("weekends"),
                                                    t1["strategy"]),
                                              run_id=cfg["run_id"], config=cfg["label"])
                                         if (cfg := configs_by_year[year].get(t1_settings.key)) else None)
                                  for year in years})
    return out


def cumulative_curve(weekends, strategy, dates):
    """Cumulative saved weekend P&L in event-date order; never infer missing event dates."""
    key = f"{strategy}_pnl"
    traded = [w for w in weekends or [] if w.get(key) is not None]
    missing = [w["event_key"] for w in traded if w["event_key"] not in dates]
    if missing:
        raise ValueError(f"saved sweep events have no matching calendar date: {', '.join(missing)}")
    rows = sorted(((pd.Timestamp(dates[w["event_key"]]), float(w[key])) for w in traded),
                  key=lambda x: x[0])
    if not rows:
        return [], 0.0
    points = [(rows[0][0] - pd.Timedelta(seconds=1), 0.0)]
    total = 0.0
    for ts, pnl in rows:
        total += pnl
        points.append((ts, total))
    return points, total


def build(conn, cs, year=2026, sport="f1", venue=None):
    """Everything the Edge Finder template shows, from saved runs only (sport / venue: the filter; venue None for all)."""
    sport = sport or "f1"
    cs = [c for c in cs if c[1] in STRATEGY_KEYS]
    cfgs = configs(conn, year, sport, venue)
    default_key = settings_class(sport).from_dict().key
    runs = {}

    def run_of(key):
        c = cfgs.get(key)
        if c and c["run_id"] not in runs:
            runs[c["run_id"]] = _run(conn, c["run_id"])
        return runs.get(c["run_id"]) if c else None

    def baseline_for(run):
        """The default-settings sweep priced from the same data (else the latest one)."""
        dk = (run or {}).get("params", {}).get("data_key")
        same = conn.execute(text("""SELECT id, params FROM model_runs WHERE kind = 'sweep' AND (params->>'year')::int = :y
                                    AND coalesce(params->>'sport', 'f1') = :sport
                                    AND params->>'data_key' = :d AND params->>'settings_key' = :k ORDER BY id DESC LIMIT 1"""),
                            dict(y=year, sport=sport, d=dk or "", k=default_key)).fetchone() if dk else None
        if same:
            if same[0] not in runs:
                runs[same[0]] = _run(conn, same[0])
            return runs[same[0]], True
        return run_of(default_key), False

    bench_run = run_of(default_key)
    bench = None
    if bench_run and (t := (bench_run["metrics"].get("totals") or {}).get(BENCHMARK_STRATEGY)):
        bench = dict(label=label(BENCHMARK_STRATEGY), run_id=bench_run["id"], pnl=t["pnl"], volume=t["bought"],
                     up=t["weekends_up"], weekends=t["weekends"])
    cards, columns, missing, recaps = [], [], [], []
    for ref, s in cs:
        key = _resolve(ref, cfgs, sport)
        cfg = cfgs.get(key)
        run = run_of(key)
        t = ((run or {}).get("metrics", {}).get("totals") or {}).get(s)
        name = cfg["label"] if cfg else (ref if not str(ref).startswith("cfg:") else ref)
        if not t:
            missing.append(dict(ref=ref, name=name, strategy=s, label=label(s),
                                settings=cfg["settings"].to_json() if cfg else None))
            continue
        base, same_data = baseline_for(run)
        bt = ((base or {}).get("metrics", {}).get("totals") or {}).get(s)
        is_bench = key == default_key and s == BENCHMARK_STRATEGY
        rc = recap(run["metrics"].get("weekends"), s)
        maker = s.startswith("maker")
        card = dict(ref=ref, key=key, name=name, strategy=s, label=label(s), pnl=t["pnl"], volume=t["bought"],
                    volume_label="filled" if maker else "bought", up=t["weekends_up"], weekends=t["weekends"],
                    run_id=run["id"], in_sample=s in IN_SAMPLE, settings=cfg["settings"].to_json(),
                    vs_base=(t["pnl"] - bt["pnl"]) if bt and key != default_key else None, same_data=same_data,
                    vs_bench=(t["pnl"] - bench["pnl"]) if bench and not is_bench else None, is_bench=is_bench,
                    clv=clv_of(run["metrics"], s))
        recaps.append(dict(card, **rc))
        if not is_bench:
            cards.append(card)
        columns.append((key, s, run["id"], name))
    events = {}
    for key, s, rid, _ in columns:
        for w in runs[rid]["metrics"].get("weekends") or []:
            e = events.setdefault(w["event_key"], dict(event=w["event"], event_key=w["event_key"], round=w.get("round"),
                                                       format=w.get("format"), last_run_id=w.get("last_run_id"), cells={}))
            e["cells"][key, s] = w.get(f"{s}_pnl")
    weekends = sorted(events.values(), key=lambda e: (e["round"] is None, e["round"] or 0, e["event_key"]))
    for e in weekends:
        e["cells"] = [e["cells"].get((k, s)) for k, s, _, _ in columns]
    totals = [sum((e["cells"][i] or 0) for e in weekends) for i in range(len(columns))]
    best = max((c["pnl"] for c in cards), default=None)
    for c in cards:
        c["best"] = len(cards) > 1 and c["pnl"] == best
    recaps.sort(key=lambda r: -r["pnl"])
    first = next((runs[rid] for _, _, rid, _ in columns), None)
    return dict(benchmark=bench, cards=cards, recaps=recaps, weekends=weekends, totals=totals,
                columns=[dict(name=n, strategy=s, label=label(s)) for _, s, _, n in columns], missing=missing,
                detail=first, detail_name=next((n for _, _, _, n in columns), None), year=year,
                n_weekends=max((c["weekends"] for c in cfgs.values()), default=0),
                configs=sorted(cfgs.values(), key=lambda c: (c["key"] != default_key, c["label"])))


def candidates(conn):
    rows = conn.execute(text("""SELECT id, created_at, params FROM model_runs WHERE kind = 'candidate'
                                ORDER BY id DESC""")).fetchall()
    out = []
    for rid, created, p in rows:
        try:
            st = SS.Settings.from_dict(p.get("settings") or {})
        except ValueError:
            continue
        out.append(dict(id=int(rid), created_at=created, name=p.get("name") or st.label(), strategy=p.get("strategy"),
                        strategy_label=label(p["strategy"]) if p.get("strategy") in STRATEGY_LABEL else p.get("strategy"),
                        year=p.get("year"), why=p.get("why", ""), run_id=p.get("run_id"), source=p.get("source", ""),
                        settings=st.to_json(), ref=st["variant"] if set(st.changed()) <= {"variant"} else "cfg:" + st.key,
                        label=st.label()))
    return out


def swept_variants(conn, year=2026):
    """Model variants with a default-settings full-season sweep for `year` (for the '+ Edge Finder' buttons)."""
    return sorted({c["settings"]["variant"] for c in configs(conn, year).values()
                   if set(c["settings"].changed()) <= {"variant"}})
