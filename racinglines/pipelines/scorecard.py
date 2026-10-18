"""
The pricing scorecard of an exchange weekend (docs/paper-trading.md, validation rule 2): at every stage
cutoff of the weekend (core/stages.py), the model's fair value of every linked market kind scored against
the result and against the venue's mid at that cutoff, traded or not.

    racinglines strategy scorecard f1 --event 2026-15 --venue both
    racinglines strategy scorecard f1 --all --year 2025 --venue polymarket

Nothing is priced here: the fair values are the stored stage runs (kind='diagnostic', one per stage, as
the sweep and the live signals store them, chosen by the model settings' `model_key`). The venue's mid
is the recorded price history (markets/store.py, read through markets/venue_replay.py, so a price older
than six hours is no price and a multi-outcome group whose prices don't sum near its target is skipped,
as in the sweep). The result is read last, to settle (markets/kinds.py). Per stage and market kind:

    n, brier, logloss            the model on every market with a fair value and a result
    paired, brier_model, brier_venue, logloss_model, logloss_venue, gap
                                 the model and the venue on the markets both priced (gap: mean |fair - mid|)

A market kind counts only while it is open at the stage (pole markets close when qualifying starts).
Written under data/runs/f1/scorecard/: <event>_<venue>.md / .csv (per stage and kind) and _markets.csv
(every scored market), or <year>_<venue>.md / .csv for a season (per weekend, and per kind and stage
pooled over the weekends).
"""

import math
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from racinglines import paths
from racinglines.core import stages as STG
from racinglines.db import reads as D
from racinglines.markets import private_book as house
from racinglines.markets import venue_replay as VR
from racinglines.pipelines import weekend_sweep as WS

VENUES = ("polymarket", "kalshi")
EPS = 1e-4
STAGE_ORDER = ["pre-weekend", "after FP1", "after SQ", "after Sprint", "after FP2", "after FP3", "after Quali"]
EMPTY = ["stage", "cutoff", "kind", "subject", "token_id", "fair", "mid", "coherent", "open", "y"]
COLS = ["stage", "kind", "n", "brier", "logloss", "paired", "brier_model", "brier_venue", "logloss_model",
        "logloss_venue", "gap"]


def default_model_key(variant="baseline"):
    """The model_key of the default sweep settings (what `f1 sweep` and the live signals price with)."""
    from racinglines.pipelines import sweep_settings as SS
    return SS.Settings.from_dict({"variant": variant}).model_key


def stage_runs(conn, event_key, stages, model_key):
    """[(label, cutoff, run_id)] for the schedule's stages that have a stored stage run (the latest one per
    cutoff when several were priced from different data)."""
    have = pd.read_sql(text("""SELECT id, params->>'cutoff' AS cutoff FROM model_runs
                               WHERE kind = 'diagnostic' AND params ? 'sweep_stage' AND params->>'model_key' = :m
                                 AND params->>'event_key' = :k ORDER BY id"""), conn, params=dict(m=model_key, k=event_key))
    by = {str(pd.Timestamp(c)): int(i) for i, c in zip(have["id"], have["cutoff"])}
    return [(lab, cut, by[str(pd.Timestamp(cut))]) for lab, cut in stages if str(pd.Timestamp(cut)) in by]


def venue_links(conn, race_id, venue):
    """The race's linked markets on the venue, one per market (Polymarket: the first token of each
    condition; Kalshi: the YES contract), for every kind the model prices."""
    kinds = [k for k in D.PREDICTION_KINDS if k.startswith("race_")]
    if venue == "kalshi":
        return VR.Kalshi.links(conn, race_id, kinds)
    links = pd.read_sql(text("""SELECT ml.*, a.display_name AS athlete FROM market_links ml
                                LEFT JOIN athletes a ON a.id = ml.athlete_id
                                WHERE ml.race_id = :r AND ml.prediction = ANY(:k) AND ml.exchange = 'polymarket'
                                ORDER BY ml.id"""), conn, params=dict(r=race_id, k=kinds))
    return links.drop_duplicates("condition_id", keep="first")


def _subject(link):
    s = link.get("athlete") or (link.get("params") or {}).get("team") or link.get("group_title") or link.get("outcome")
    if link["prediction"] == "race_h2h" and link.get("question"):
        s = f"{link['outcome']} ({link['question'].split(': ')[-1]})"
    return s


def weekend_rows(conn, w, runs, venue):
    """One row per (stage, market): fair, the venue's mid (as recorded), whether its multi-outcome group was
    coherent then (score() takes the mid only when it was), the outcome, and whether the kind is open at the stage.
    w: a weekend from weekend_sweep.schedule; runs: [(label, cutoff, run_id)]."""
    rid = WS._race_id(conn, w["event_key"])
    if rid is None or not runs:
        return pd.DataFrame(columns=EMPTY)
    links = venue_links(conn, rid, venue)
    if not len(links):
        return pd.DataFrame(columns=EMPTY)
    start, end = runs[0][1] - timedelta(hours=1), w["race_start"]
    cls = VR.Kalshi if venue == "kalshi" else VR.Polymarket
    ven = cls(conn, links, start, end, WS.GROUP_TARGET, WS.COHERENCE_TOL, WS.STALE)
    cache, rows = {}, []
    for lab, cutoff, run_id in runs:
        coherent = {k: ven.coherent(k, cutoff) for k in WS.GROUP_TARGET}
        for link in ven.markets():
            kind = link["prediction"]
            fair = D.model_prob(conn, link, cache, run_id=run_id)[0]
            mid = ven.price(link["token_id"], cutoff)
            if mid is not None and not (0 < mid < 1):
                mid = None
            rows.append(dict(stage=lab, cutoff=cutoff, kind=kind, subject=_subject(link), token_id=link["token_id"],
                             fair=fair, mid=mid, coherent=coherent.get(kind, True),
                             open=STG.is_open(kind, cutoff, w.get("closes") or {})))
    df = pd.DataFrame(rows)
    res = house.race_outcomes(conn, rid)                 # read last: settlement only
    y = {link["token_id"]: ven.resolve(link, res) for link in ven.markets()}
    df["y"] = df["token_id"].map(lambda t: None if y.get(t) is None else float(y[t]))
    return df


def _score(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    q = np.clip(p, EPS, 1 - EPS)
    return float(((p - y) ** 2).mean()), float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def score(rows, by=("stage", "kind")):
    """Per group in `by`: the model against the result on every market with a fair value and a result
    (n, brier, logloss), and model vs venue on the markets both priced (paired, brier_model, brier_venue,
    logloss_model, logloss_venue, gap). Closed kinds are left out."""
    rows = pd.DataFrame(rows)
    out = []
    if not len(rows):
        return pd.DataFrame(columns=[*by, *COLS[2:]])
    rows = rows[rows["open"].astype(bool) & rows["fair"].notna() & rows["y"].notna()]
    rows = rows.assign(mid=rows["mid"].where(rows["coherent"].astype(bool)))    # an incoherent group: no mid
    for key, g in (rows.groupby(list(by), sort=False) if by else [((), rows)]):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key), n=len(g))
        r["brier"], r["logloss"] = _score(g["fair"], g["y"])
        h = g[g["mid"].notna()]
        r["paired"] = len(h)
        if len(h):
            r["brier_model"], r["logloss_model"] = _score(h["fair"], h["y"])
            r["brier_venue"], r["logloss_venue"] = _score(h["mid"], h["y"])
            r["gap"] = float((h["fair"] - h["mid"]).abs().mean())
        else:
            r.update(brier_model=None, logloss_model=None, brier_venue=None, logloss_venue=None, gap=None)
        out.append(r)
    df = pd.DataFrame(out)
    if "stage" in by:
        df["_o"] = df["stage"].map({s: i for i, s in enumerate(STAGE_ORDER)}).fillna(99)
        df = df.sort_values(["_o", *[c for c in by if c != "stage"]]).drop(columns="_o")
    elif by:
        df = df.sort_values(list(by))
    return df.reset_index(drop=True)


def _f(x, d=4):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def format_table(df, first=("stage", "kind")):
    """A markdown table of score() rows (or the season's pooled rows)."""
    head = [*first, "n", "Brier", "Log loss", "paired", "Brier model", "Brier venue", "Log loss model",
            "Log loss venue", "|fair − mid|"]
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in df.to_dict("records"):
        cells = [str(r[c]) for c in first] + [str(int(r["n"])), _f(r["brier"]), _f(r["logloss"]), str(int(r["paired"])),
                                              _f(r.get("brier_model")), _f(r.get("brier_venue")),
                                              _f(r.get("logloss_model")), _f(r.get("logloss_venue")), _f(r.get("gap"), 3)]
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def format_text(df, first=("stage", "kind")):
    cols = [*first, "n", "brier", "logloss", "paired", "brier_model", "brier_venue", "logloss_model", "logloss_venue", "gap"]
    return df[[c for c in cols if c in df]].to_string(index=False, float_format="{:.4f}".format, na_rep="–")


def notes(rows, venue):
    """What the venue's side of the table leaves out: kinds it never priced, and the stages where a
    multi-outcome group's prices didn't sum near its target (an empty or stale book), so its mid isn't scored."""
    out = []
    if not len(rows):
        return out
    live = rows[rows["open"].astype(bool)]
    unpriced = [k for k, g in live.groupby("kind", sort=True) if g["mid"].isna().all()]
    if unpriced:
        out.append(f"no {venue} price recorded for {', '.join(unpriced)}")
    bad = live[~live["coherent"].astype(bool) & live["mid"].notna()]
    if len(bad):
        skipped = bad.groupby("kind", sort=True)["stage"].unique()
        out.append("mid not scored where the group's prices didn't sum near its target (an empty or stale book): "
                   + "; ".join(f"{k} at {', '.join(v)}" for k, v in skipped.items()))
    return out


def weekend(conn, w, venue, model_key):
    """One weekend on one venue: dict(event_key, venue, runs, rows, scores, note)."""
    runs = stage_runs(conn, w["event_key"], w["stages"], model_key)
    missing = [lab for lab, _ in w["stages"] if lab not in {r[0] for r in runs}]
    rows = weekend_rows(conn, w, runs, venue)
    note = []
    if missing:
        note.append(f"no stage run for {', '.join(missing)} (model_key {model_key}; run `racinglines strategy sweep f1` or `racinglines strategy signals f1` first)")
    if not len(rows) and runs:
        note.append(f"no {venue} markets linked")
    note += notes(rows, venue)
    return dict(event_key=w["event_key"], name=w["name"], venue=venue, runs=runs, rows=rows,
                scores=score(rows) if len(rows) else score([]), note="; ".join(note))


def _source_note(venue):
    """The compact source note (racinglines/attribution.py): F1's results feed, the venue, the stored stage runs."""
    from racinglines import attribution
    return attribution.note("f1", [venue], model=True)["text"] + "."


def markdown(res, model_key):
    sc = res["scores"]
    L = [f"# Pricing scorecard: {res['event_key']} {res['name']} on {res['venue']}", "",
         f"The model's fair value (stage runs, model_key `{model_key}`) at each stage's cutoff, scored against the "
         f"result and against {res['venue']}'s mid at that cutoff (Brier and log loss, lower is better; 0.25 is a coin "
         "flip on a 50/50 market). `n`: markets with a fair value and a result; `paired`: those the venue also priced, "
         "where model and venue are scored side by side; `|fair − mid|`: their mean disagreement.", ""]
    if res["note"]:
        L += [f"Note: {res['note']}.", ""]
    if res["runs"]:
        L += ["Stages: " + ", ".join(f"{lab} ({pd.Timestamp(c):%d %b %H:%M} UTC, run #{i})" for lab, c, i in res["runs"]), ""]
    if len(sc):
        L += [format_table(sc), ""]
        pooled = score(res["rows"], by=("kind",))
        L += ["All stages pooled (a market counts once per stage):", "", format_table(pooled, first=("kind",)), ""]
    L += ["", _source_note(res["venue"])]
    return "\n".join(L)


def write(res, model_key, outdir=None):
    """Write <event>_<venue>.md / .csv / _markets.csv. Returns {name: path}."""
    out = outdir or paths.runs("f1", "scorecard")
    base = f"{res['event_key']}_{res['venue']}"
    files = {}
    (out / f"{base}.md").write_text(markdown(res, model_key) + "\n")
    files["md"] = out / f"{base}.md"
    res["scores"].to_csv(out / f"{base}.csv", index=False)
    files["csv"] = out / f"{base}.csv"
    if len(res["rows"]):
        res["rows"].to_csv(out / f"{base}_markets.csv", index=False)
        files["markets"] = out / f"{base}_markets.csv"
    return files


def season(conn, year, venue, model_key, rounds=None, echo=print):
    """Every raced weekend of a season on one venue: dict(weekends DataFrame, by_stage, by_kind, rows, results)."""
    sched = WS.schedule(year, rounds)
    results, rows, per = [], [], []
    for rnd, w in sched.items():
        r = weekend(conn, w, venue, model_key)
        if not r["runs"]:
            echo(f"  {w['event_key']} {w['name']}: {r['note']}")
            continue
        results.append(r)
        if len(r["rows"]):
            rows.append(r["rows"].assign(event_key=w["event_key"]))
        s = score(r["rows"], by=()) if len(r["rows"]) else pd.DataFrame()
        row = dict(event_key=w["event_key"], event=w["name"], stages=len(r["runs"]),
                   n=int(s["n"].iloc[0]) if len(s) else 0, brier=s["brier"].iloc[0] if len(s) else None,
                   logloss=s["logloss"].iloc[0] if len(s) else None, paired=int(s["paired"].iloc[0]) if len(s) else 0,
                   brier_model=s["brier_model"].iloc[0] if len(s) else None, brier_venue=s["brier_venue"].iloc[0] if len(s) else None,
                   logloss_model=s["logloss_model"].iloc[0] if len(s) else None,
                   logloss_venue=s["logloss_venue"].iloc[0] if len(s) else None, gap=s["gap"].iloc[0] if len(s) else None)
        per.append(row)
        echo(f"  {w['event_key']} {w['name']}: {row['n']} markets, {row['paired']} paired · Brier model "
             f"{_f(row['brier_model'])} vs {venue} {_f(row['brier_venue'])}" + (f" · {r['note']}" if r["note"] else ""))
    allrows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return dict(year=year, venue=venue, weekends=pd.DataFrame(per), rows=allrows, results=results,
                by_stage=score(allrows) if len(allrows) else score([]),
                by_kind=score(allrows, by=("kind",)) if len(allrows) else score([], by=("kind",)))


def season_markdown(out, model_key):
    L = [f"# Pricing scorecard: {out['year']} season on {out['venue']}", "",
         f"Every raced weekend with stage runs (model_key `{model_key}`) and {out['venue']} markets: the model's fair "
         f"value at each stage's cutoff against the result and against the venue's mid at that cutoff. Brier and log "
         "loss, lower is better. `n`: markets with a fair value and a result, summed over the weekend's stages; "
         "`paired`: those the venue also priced, scored side by side; `|fair − mid|`: their mean disagreement.", ""]
    if len(out["weekends"]):
        L += ["## Per weekend (all stages and kinds pooled)", "", format_table(out["weekends"], first=("event_key", "event")), ""]
    if len(out["by_kind"]):
        L += ["## Per market kind, all stages and weekends pooled", "", format_table(out["by_kind"], first=("kind",)), ""]
    if len(out["by_stage"]):
        L += ["## Per stage and market kind, all weekends pooled", "", format_table(out["by_stage"]), ""]
    L += ["", _source_note(out["venue"])]
    return "\n".join(L)


def write_season(out, model_key, outdir=None):
    d = outdir or paths.runs("f1", "scorecard")
    base = f"{out['year']}_{out['venue']}"
    (d / f"{base}.md").write_text(season_markdown(out, model_key) + "\n")
    out["weekends"].to_csv(d / f"{base}_weekends.csv", index=False)
    out["by_stage"].to_csv(d / f"{base}.csv", index=False)
    files = dict(md=d / f"{base}.md", csv=d / f"{base}.csv", weekends=d / f"{base}_weekends.csv")
    if len(out["rows"]):
        out["rows"].to_csv(d / f"{base}_markets.csv", index=False)
        files["markets"] = d / f"{base}_markets.csv"
    return files
