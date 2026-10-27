"""
`racinglines eval compare RUN_A RUN_B`: two saved runs scored against the same outcomes, paired by event
(docs/engine-roadmap.md phase E5; docs/novig-and-model-bakeoff.md Part 2, "paired by event with ±2 standard errors").

It reads the prediction records of each run (racinglines/db/records.py, written with RACINGLINES_PREDICTION_RECORDS=1),
and works in four steps, each a function of plain frames so it runs on any two runs of any sport:

    pair(a, b)               inner join on (event_id, stage, kind, subject, params); the rows only one run has are
                             counted and reported, never dropped silently
    outcomes(paired, ...)    y per paired row from the official result, through the walk-forward's own settlement
                             (markets.kinds.settle on the sport's pricing model's `results`); a row that can't settle
                             is counted and left out
    compare(paired_y)        log loss and Brier per row (the clip of core/calibration.py), the mean per event for A and
                             for B, and B - A per event
    summarize(per_event)     per kind, per kind x season and pooled over seasons: the mean of the per-event
                             differences, its standard error over events (std, ddof 1, over sqrt(n)), and whether
                             0 lies outside ±2 SE

The score is "lower is better", so a negative difference (B - A) means B is the better forecast.

`diagnose(paired_y, results_for)` adds the pass test's three checks (docs/backtest-core.md, "Prediction records"),
on the same paired, settled rows:

    buckets      reliability: six fixed buckets of the price (edges 0.05, 0.15, 0.30, 0.50, 0.70), per kind, stage and season
                 and pooled, for A and B side by side: n, mean priced, mean happened, the gap, and a flag when the gap
                 is over 5 points on at least 30 rows
                 `pass_top2` marks, per run, the two highest-priced buckets with n >= 30: True when both gaps are within 5 points
    favourites   per event and kind (race_win, race_podium, race_top10; kinds with params are skipped), the row with the
                 highest fair in each run: its mean price against how often it happened
    pole         race_win only: the entrant first in the event's qualifying classification (the results frame's
                 qual_position, from the same results_for as the settlement): how often it won against each run's mean win price
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from racinglines.core import calibration as CAL

KEY = ["event_id", "stage", "kind", "subject", "params"]
SCORES = (("logloss", "log loss"), ("brier", "Brier"))
SUMMARY_COLUMNS = ["kind", "stage", "season", "n_events", "n_rows",
                   "logloss_a", "logloss_b", "logloss_diff", "logloss_se", "logloss_outside",
                   "brier_a", "brier_b", "brier_diff", "brier_se", "brier_outside"]


BUCKET_EDGES = (0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.0001)           # [lo, hi), the last includes 1: core/calibration.py's bins
BUCKET_LABELS = ("<0.05", "0.05-0.15", "0.15-0.30", "0.30-0.50", "0.50-0.70", ">=0.70")
FLAG_GAP, FLAG_N = 0.05, 30                                            # a bucket is flagged when |happened - priced| > 5 points on n >= 30
BUCKET_COLUMNS = ["kind", "stage", "season", "bucket"] + [f"{c}_{s}" for s in "ab" for c in
                                                         ("n", "priced", "happened", "gap", "flag", "pass_top2")]
FAVOURITE_KINDS = ("race_win", "race_podium", "race_top10")            # subject kinds only: kinds whose rows carry params are skipped
FAVOURITE_COLUMNS = ["kind", "stage", "season", "n", "priced_a", "happened_a", "gap_a", "priced_b", "happened_b", "gap_b",
                     "same_favourite"]
POLE_COLUMNS = ["kind", "stage", "season", "n_events", "no_qualifying", "not_priced", "pole_won", "priced_a", "gap_a",
                "priced_b", "gap_b"]


# --- pairing -------------------------------------------------------------------------------------

def _canon(p):
    """params as one canonical JSON string (sorted keys), whether the record holds a string, a dict or nothing."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "{}"
    return json.dumps(json.loads(p) if isinstance(p, str) else p, sort_keys=True)


def _keyed(df, label):
    df = df.copy()
    df["params"] = df["params"].map(_canon)
    df["event_id"] = df["event_id"].astype(str)
    df["subject"] = df["subject"].astype(str)
    df["stage"] = df["stage"].astype(str)
    dup = df.duplicated(KEY, keep=False)
    if dup.any():
        raise ValueError(f"run {label}: {int(dup.sum())} records share an (event, stage, kind, subject, params) key; "
                         f"the first is {df.loc[dup, KEY].iloc[0].to_dict()}")
    return df


def pair(records_a, records_b):
    """(paired, counts): the records both runs have, side by side, and the counts of what was left unpaired.
    paired: sport, season, event_id, event, stage, kind, subject, params, fair_a, fair_b, se_a, se_b.
    counts: n_a, n_b, paired, only_a, only_b (rows), and the same split by kind in `only_a_kinds` / `only_b_kinds`."""
    a, b = _keyed(records_a, "A"), _keyed(records_b, "B")
    cols = ["fair", "se"]
    m = a.merge(b[KEY + cols], on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"]
    paired = both[["sport", "season", "event_id", "event", "stage", "kind", "subject", "params",
                   "fair_a", "fair_b", "se_a", "se_b"]].reset_index(drop=True)
    paired = paired.assign(season=pd.to_numeric(paired["season"]).astype("Int64"))     # the outer join made it a float
    only_b = b.merge(a[KEY], on=KEY, how="left", indicator=True)
    only_b = only_b[only_b["_merge"] == "left_only"]
    only_a = m[m["_merge"] == "left_only"]
    counts = dict(n_a=len(a), n_b=len(b), paired=len(paired), only_a=len(only_a), only_b=len(only_b),
                  only_a_kinds=only_a["kind"].value_counts().to_dict(), only_b_kinds=only_b["kind"].value_counts().to_dict())
    return paired, counts


# --- outcomes ------------------------------------------------------------------------------------

def settle_rows(paired, results_for):
    """The paired frame with a `y` column (1.0 / 0.0, NaN where the result can't settle the row). results_for(sport,
    event_id) -> the official result in markets.kinds.settle's shape (columns athlete_id, position, status, ...) or None.
    The settlement is markets.kinds.settle, the call core/walk_forward.event_rows makes: nothing is re-implemented."""
    from racinglines.markets import kinds as K
    y = pd.Series(np.nan, index=paired.index, dtype=float)
    for (sport, event_id), g in paired.groupby(["sport", "event_id"], sort=False, dropna=False):
        res = results_for(sport, event_id)
        if res is None or res.empty:
            continue
        ids = {str(x): x for x in res["athlete_id"]}
        for i, kind, subject, params in zip(g.index, g["kind"], g["subject"], g["params"]):
            p = json.loads(params)
            if "team" in p:                           # a top team: the group key is the subject, the athlete none
                ok = K.settle(kind, None, p, res)
            else:
                ok = K.settle(kind, ids.get(subject, subject), p or None, res)
            if ok is not None:
                y[i] = float(bool(ok))
    return paired.assign(y=y)


def db_results(engine_url=None):
    """results_for(sport, event_id) reading the official result as the walk-forward does: the sport's pricing model
    loads its data once (model.load), `events` names the event, `results` is the official classification, marked with
    the sport's classified rule (markets.payoffs.mark_sport)."""
    from racinglines.markets import payoffs as P
    from racinglines.models import race_model as RM
    cache = {}

    def loaded(sport):
        if sport not in cache:
            model = RM.get(sport)
            data = model.load(engine_url)
            cache[sport] = (model, data, {str(ev.id): ev for ev in model.events(data, model.Settings.from_dict({}), None)})
        return cache[sport]

    def results_for(sport, event_id):
        model, data, events = loaded(sport)
        if event_id not in events:
            return None
        return P.mark_sport(model.results(data, events[event_id]), getattr(model, "sport", None))

    return results_for


# --- scores --------------------------------------------------------------------------------------

def compare(paired_y):
    """Per (kind, stage, season, event): n rows, the mean log loss and Brier of A and of B, and B - A. Rows with no y
    are left out here (settle_rows marks them; run() reports their number)."""
    d = paired_y.dropna(subset=["y"])
    cols = ["kind", "stage", "season", "event_id", "event", "n", "logloss_a", "logloss_b", "logloss_diff",
            "brier_a", "brier_b", "brier_diff"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    y = d["y"].to_numpy(float)
    out = d[["kind", "stage", "season", "event_id", "event"]].copy()
    for s in "ab":
        p = d[f"fair_{s}"].to_numpy(float)
        q = np.clip(p, CAL.EPS, 1 - CAL.EPS)
        out[f"logloss_{s}"] = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        out[f"brier_{s}"] = (p - y) ** 2
    out["n"] = 1
    g = out.groupby(["kind", "stage", "season", "event_id", "event"], sort=True, dropna=False)
    ev = g.agg(n=("n", "sum"), **{f"{m}_{s}": (f"{m}_{s}", "mean") for m in ("logloss", "brier") for s in "ab"}).reset_index()
    for m, _ in SCORES:
        ev[f"{m}_diff"] = ev[f"{m}_b"] - ev[f"{m}_a"]
    return ev[cols]


def _summ(g):
    row = dict(n_events=len(g), n_rows=int(g["n"].sum()))
    for m, _ in SCORES:
        d = g[f"{m}_diff"].to_numpy(float)
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        mean = float(d.mean())
        row.update({f"{m}_a": float(g[f"{m}_a"].mean()), f"{m}_b": float(g[f"{m}_b"].mean()), f"{m}_diff": mean,
                    f"{m}_se": se, f"{m}_outside": bool(np.isfinite(se) and abs(mean) > 2 * se)})
    return row


def summarize(per_event):
    """Per (kind, stage, season) and pooled over seasons (season = "all"): the mean of the per-event differences, its
    standard error over events (paired by event) and whether 0 is outside ±2 SE. Events weigh equally."""
    if per_event.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    for (kind, stage), g in per_event.groupby(["kind", "stage"], sort=True):
        for season, h in g.groupby("season", sort=True, dropna=False):
            rows.append(dict(kind=kind, stage=stage, season=str(season), **_summ(h)))
        rows.append(dict(kind=kind, stage=stage, season="all", **_summ(g)))
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


# --- the pass test's diagnostics ---------------------------------------------------------------

def _groups(d):
    """(kind, stage, season label, frame) for every kind x stage x season and pooled over seasons ("all")."""
    for (kind, stage), g in d.groupby(["kind", "stage"], sort=True):
        for season, h in g.groupby("season", sort=True, dropna=False):
            yield kind, stage, str(season), h
        yield kind, stage, "all", g


def _buckets(p, y):
    """The six buckets of one run's prices: n, priced, happened, gap (happened - priced), flag. core/calibration.reliability
    does the binning; an empty bucket keeps its row (n 0, the rest NaN)."""
    got = {round(r["lo"], 6): r for r in CAL.reliability(p, y, BUCKET_EDGES)}
    out = []
    for lo in BUCKET_EDGES[:-1]:
        r = got.get(round(lo, 6))
        if r is None:
            out.append(dict(n=0, priced=np.nan, happened=np.nan, gap=np.nan, flag=False, pass_top2=None))
            continue
        gap = r["observed"] - r["predicted"]
        out.append(dict(n=r["n"], priced=r["predicted"], happened=r["observed"], gap=gap,
                        flag=bool(abs(gap) > FLAG_GAP and r["n"] >= FLAG_N), pass_top2=None))
    top = sorted((i for i, r in enumerate(out) if r["n"] >= FLAG_N), key=lambda i: out[i]["priced"])[-2:]
    if len(top) == 2:                                                  # fewer than two buckets with n >= 30: no verdict (None)
        ok = all(abs(out[i]["gap"]) <= FLAG_GAP for i in top)
        for i in top:
            out[i]["pass_top2"] = ok
    return out


def buckets(paired_y):
    """Reliability buckets per (kind, stage, season) and pooled, run A and run B side by side (each run's rows fall in
    the bucket of its own price, so n differs). Only settled rows."""
    d = paired_y.dropna(subset=["y"])
    rows = []
    for kind, stage, season, g in _groups(d):
        per = {s: _buckets(g[f"fair_{s}"].to_numpy(float), g["y"].to_numpy(float)) for s in "ab"}
        for i, label in enumerate(BUCKET_LABELS):
            rows.append(dict(kind=kind, stage=stage, season=season, bucket=label,
                             **{f"{c}_{s}": per[s][i][c] for s in "ab" for c in ("n", "priced", "happened", "gap", "flag", "pass_top2")}))
    out = pd.DataFrame(rows, columns=BUCKET_COLUMNS)
    for s in "ab":
        out[f"pass_top2_{s}"] = out[f"pass_top2_{s}"].astype(object)
    return out


def top_bucket_pass(buckets_df):
    """One line per (kind, stage, season) under the bucket table: for each run, "pass" or "fail" with the gaps of its two
    highest-priced buckets that have n >= 30 (both within 5 points = pass), or "n/a" when fewer than two buckets qualify.
    Returns a list of dicts: kind, stage, season, and per run `verdict` (True / False / None) and `gaps` ([(bucket, gap)])."""
    out = []
    for (kind, stage, season), g in buckets_df.groupby(["kind", "stage", "season"], sort=False):
        row = dict(kind=kind, stage=stage, season=season)
        for s in "ab":
            sel = g[g[f"pass_top2_{s}"].notna()]
            row[f"verdict_{s}"] = bool(sel[f"pass_top2_{s}"].iloc[0]) if len(sel) else None
            row[f"gaps_{s}"] = [(r["bucket"], r[f"gap_{s}"]) for _, r in sel.iloc[::-1].iterrows()]
        out.append(row)
    return out


def favourites(paired_y):
    """Per (event, stage, kind), the settled row with the highest fair in each run (a tie goes to the lower subject): the
    mean price of those favourites against how often they happened, per kind x stage x season and pooled; `same_favourite`
    is the share of events where both runs name the same row. Only the subject kinds (FAVOURITE_KINDS: race_win, race_podium,
    race_top10); kinds whose rows carry params (race_h2h, exact place, ...) are skipped. For race_win this is the
    "favourite wins" check."""
    d = paired_y.dropna(subset=["y"])
    d = d[d["kind"].isin(FAVOURITE_KINDS)].sort_values(["subject", "params"], kind="stable")
    if d.empty:
        return pd.DataFrame(columns=FAVOURITE_COLUMNS)
    keys = ["kind", "stage", "season", "event_id"]
    top = {s: d.loc[d.groupby(keys, sort=True, dropna=False)[f"fair_{s}"].idxmax()].set_index(keys) for s in "ab"}
    ev = pd.DataFrame({f"{c}_{s}": top[s][col] for s in "ab" for c, col in
                       (("priced", f"fair_{s}"), ("happened", "y"), ("subject", "subject"), ("params", "params"))})
    ev["same"] = (ev["subject_a"] == ev["subject_b"]) & (ev["params_a"] == ev["params_b"])
    ev = ev.reset_index()
    rows = []
    for kind, stage, season, g in _groups(ev):
        row = dict(kind=kind, stage=stage, season=season, n=len(g), same_favourite=float(g["same"].mean()))
        for s in "ab":
            row.update({f"priced_{s}": float(g[f"priced_{s}"].mean()), f"happened_{s}": float(g[f"happened_{s}"].mean())})
            row[f"gap_{s}"] = row[f"happened_{s}"] - row[f"priced_{s}"]
        rows.append(row)
    return pd.DataFrame(rows, columns=FAVOURITE_COLUMNS)


def pole(paired_y, results_for):
    """race_win only. The pole-sitter is the entrant with qual_position == 1 in results_for(sport, event_id) (the frame
    the settlement reads, markets.payoffs.stage_col("qual")): per kind x stage x season and pooled, the fraction of events
    the pole-sitter won, and each run's mean win price for the pole-sitter. An event whose result has no qualifying order
    is skipped and counted (`no_qualifying`); so is one whose pole-sitter has no settled race_win row (`not_priced`)."""
    from racinglines.markets import payoffs as P
    col = P.stage_col("qual")
    d = paired_y[(paired_y["kind"] == "race_win")].dropna(subset=["y"])
    rows = []
    cache = {}
    for (sport, event_id), g in d.groupby(["sport", "event_id"], sort=False, dropna=False):
        res = results_for(sport, event_id)
        sitter = None
        if res is not None and not res.empty and col in res:
            first = res.loc[res[col] == 1, "athlete_id"]
            sitter = str(first.iloc[0]) if len(first) else None
        cache[(sport, event_id)] = sitter
    ev = []
    for (sport, event_id, stage, season), g in d.groupby(["sport", "event_id", "stage", "season"], sort=False, dropna=False):
        sitter = cache[(sport, event_id)]
        hit = g[g["subject"] == sitter] if sitter is not None else g.iloc[:0]
        ev.append(dict(kind="race_win", stage=stage, season=season, no_qualifying=sitter is None,
                       not_priced=sitter is not None and hit.empty, y=float(hit["y"].iloc[0]) if len(hit) else np.nan,
                       fair_a=float(hit["fair_a"].iloc[0]) if len(hit) else np.nan,
                       fair_b=float(hit["fair_b"].iloc[0]) if len(hit) else np.nan))
    if not ev:
        return pd.DataFrame(columns=POLE_COLUMNS)
    ev = pd.DataFrame(ev)
    for kind, stage, season, g in _groups(ev):
        used = g.dropna(subset=["y"])
        row = dict(kind=kind, stage=stage, season=season, n_events=len(used), no_qualifying=int(g["no_qualifying"].sum()),
                   not_priced=int(g["not_priced"].sum()), pole_won=float(used["y"].mean()) if len(used) else np.nan)
        for s in "ab":
            row[f"priced_{s}"] = float(used[f"fair_{s}"].mean()) if len(used) else np.nan
            row[f"gap_{s}"] = row["pole_won"] - row[f"priced_{s}"]
        rows.append(row)
    return pd.DataFrame(rows, columns=POLE_COLUMNS)


def diagnose(paired_y, results_for):
    """The three pass-test tables, from the paired frame settle_rows returns: dict(buckets, favourites, pole)."""
    return dict(buckets=buckets(paired_y), favourites=favourites(paired_y), pole=pole(paired_y, results_for))


def format_diagnostics(diag, run_a, run_b):
    def f(x, spec=".3f"):
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)

    def table(df, head, cells):
        stages = df["stage"].nunique() if len(df) else 0
        h = [head[0]] + (["stage"] if stages > 1 else []) + head[1:]
        out = ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in h) + "|"]
        for r in df.to_dict("records"):
            c = cells(r)
            out.append("| " + " | ".join([c[0]] + ([r["stage"]] if stages > 1 else []) + c[1:]) + " |")
        return out

    b, fv, po = diag["buckets"], diag["favourites"], diag["pole"]
    lines = ["", f"### Reliability buckets (A = run {run_a}, B = run {run_b})", "",
             f"gap = happened - priced; `!` flags |gap| > {FLAG_GAP:g} on n >= {FLAG_N}. Each run's rows sit in the bucket of its own price.", ""]
    lines += table(b, ["kind", "season", "bucket", "n A", "priced A", "happened A", "gap A", "n B", "priced B", "happened B", "gap B"],
                   lambda r: [r["kind"], r["season"], r["bucket"]] + [x for s in "ab" for x in (
                       str(r[f"n_{s}"]), f(r[f"priced_{s}"]), f(r[f"happened_{s}"]),
                       (f(r[f"gap_{s}"], "+.3f") + (" !" if r[f"flag_{s}"] else "")))])
    lines += ["", "Top-bucket pass (the two highest-priced buckets with n >= %d, both |gap| <= %g):" % (FLAG_N, FLAG_GAP), ""]
    for t in top_bucket_pass(b):
        parts = []
        for s, name in (("a", "A"), ("b", "B")):
            v = t[f"verdict_{s}"]
            gaps = ", ".join(f"{bk} {f(g, '+.3f')}" for bk, g in t[f"gaps_{s}"])
            parts.append(f"{name} " + ("n/a (fewer than 2 buckets with n >= %d)" % FLAG_N if v is None else
                                       ("pass" if v else "fail") + f" ({gaps})"))
        lines.append(f"- {t['kind']}" + (f" {t['stage']}" if b["stage"].nunique() > 1 else "") + f" {t['season']}: " + "; ".join(parts))
    lines += ["", "### Favourite wins (the highest fair per event and kind)", "",
              "Subject kinds only (%s); kinds whose rows carry params, such as race_h2h, are skipped." % ", ".join(FAVOURITE_KINDS), ""]
    lines += table(fv, ["kind", "season", "n", "priced A", "happened A", "gap A", "priced B", "happened B", "gap B", "same favourite"],
                   lambda r: [r["kind"], r["season"], str(r["n"]), f(r["priced_a"]), f(r["happened_a"]), f(r["gap_a"], "+.3f"),
                              f(r["priced_b"]), f(r["happened_b"]), f(r["gap_b"], "+.3f"), f(r["same_favourite"], ".2f")])
    lines += ["", "### Pole-sitter win rate (race_win; the pole-sitter is first in the qualifying classification)", ""]
    if len(po):
        lines += table(po, ["kind", "season", "n events", "no qualifying", "not priced", "pole won", "priced A", "gap A", "priced B", "gap B"],
                       lambda r: [r["kind"], r["season"], str(r["n_events"]), str(r["no_qualifying"]), str(r["not_priced"]),
                                  f(r["pole_won"]), f(r["priced_a"]), f(r["gap_a"], "+.3f"), f(r["priced_b"]), f(r["gap_b"], "+.3f")])
    else:
        lines.append("No settled race_win rows.")
    return "\n".join(lines)


# --- the report ----------------------------------------------------------------------------------

def format_markdown(summary, run_a, run_b, counts=None, unsettled=0):
    stages = summary["stage"].nunique() if len(summary) else 0
    head = ["kind"] + (["stage"] if stages > 1 else []) + ["season", "n events", "logloss A", "logloss B", "diff ± 2 SE",
                                                           "Brier A", "Brier B", "diff ± 2 SE"]
    lines = [f"Run A = {run_a}, run B = {run_b}. diff = B - A; lower is better, so a negative diff favours B. "
             f"`*` marks a diff whose ±2 SE (over events, paired) excludes 0.", "",
             "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]

    def dd(r, m):
        se = r[f"{m}_se"]
        star = "*" if r[f"{m}_outside"] else ""
        return f"{r[f'{m}_diff']:+.4f} ± {2 * se:.4f}{star}" if np.isfinite(se) else f"{r[f'{m}_diff']:+.4f} (1 event)"

    for r in summary.to_dict("records"):
        cells = [r["kind"]] + ([r["stage"]] if stages > 1 else []) + [
            r["season"], str(r["n_events"]), f"{r['logloss_a']:.4f}", f"{r['logloss_b']:.4f}", dd(r, "logloss"),
            f"{r['brier_a']:.4f}", f"{r['brier_b']:.4f}", dd(r, "brier")]
        lines.append("| " + " | ".join(cells) + " |")
    if counts:
        lines += ["", f"Paired {counts['paired']} rows (A has {counts['n_a']}, B has {counts['n_b']}); "
                      f"only in A: {counts['only_a']}{_kinds(counts['only_a_kinds'])}; "
                      f"only in B: {counts['only_b']}{_kinds(counts['only_b_kinds'])}; "
                      f"paired but not settled (left out of the scores): {unsettled}."]
    return "\n".join(lines)


def _kinds(d):
    return " (" + ", ".join(f"{k} {v}" for k, v in d.items()) + ")" if d else ""


def run(run_a, run_b, kinds=None, stage=None, out_dir=None, echo=print, results_for=None, engine_url=None, write=True,
        diagnostics=False):
    """Compare two saved runs on their prediction records. Returns dict(paired, per_event, summary, counts, unsettled).
    results_for: how a result is read (default: the sport's pricing model on engine_url, db_results). write=False
    (the MCP tool) writes nothing; otherwise compare.csv (per event) and compare.md (summary) go to out_dir, by default
    data/runs/eval/<run_a>-vs-<run_b>/. diagnostics adds the pass test's tables (diagnose()): printed after the
    summary, appended to compare.md, written as buckets.csv, favourites.csv and pole.csv, and returned under those keys."""
    from racinglines import paths
    from racinglines.db import records as REC
    frames = []
    for label, rid in (("A", run_a), ("B", run_b)):
        try:
            df = REC.read(rid)
        except FileNotFoundError:
            raise ValueError(f"run {rid} ({label}) has no prediction records in {REC.records_dir()}; run it with "
                             f"--save and {REC.SWITCH}=1") from None
        if kinds:
            df = df[df["kind"].isin(kinds)]
        if stage:
            df = df[df["stage"] == stage]
        frames.append(df)
    paired, counts = pair(*frames)
    if not len(paired):
        echo(f"No records pair: run {run_a} has {counts['n_a']} rows and run {run_b} {counts['n_b']} after the filters, "
             f"and none share an (event, stage, kind, subject, params) key.")
    results_for = results_for or db_results(engine_url)
    paired_y = settle_rows(paired, results_for) if len(paired) else paired.assign(y=np.nan)
    unsettled = int(paired_y["y"].isna().sum())
    per_event = compare(paired_y)
    summary = summarize(per_event)
    md = format_markdown(summary, run_a, run_b, counts, unsettled)
    echo(md)
    diag = {}
    if diagnostics:
        diag = diagnose(paired_y, results_for)
        dmd = format_diagnostics(diag, run_a, run_b)
        echo(dmd)
        md += "\n" + dmd
    if write:
        d = Path(out_dir) if out_dir else paths.runs("eval", f"{run_a}-vs-{run_b}", mkdir=False)
        d.mkdir(parents=True, exist_ok=True)
        per_event.to_csv(d / "compare.csv", index=False)
        (d / "compare.md").write_text(md + "\n")
        for name, df in diag.items():
            df.to_csv(d / f"{name}.csv", index=False)
        echo(f"\nFiles -> {d}/")
    return dict(paired=paired_y, per_event=per_event, summary=summary, counts=counts, unsettled=unsettled, **diag)
