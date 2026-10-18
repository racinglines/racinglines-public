"""
Practice-pace prior: update the model's qualifying and race pace with what the
weekend's practice sessions showed, like downhill's timed-training prior.

Training rows (one per past event x driver), all as of 1 minute before the event's
first practice session, so the model's prediction contains nothing from that weekend:
    qp_pre, rp_pre           the model's predicted qualifying / race-pace deficit
    best_k, long_k           practice one-lap and long-run deficits after the first k sessions
    q_def, r_def             what happened (qualifying deficit, race-pace deficit)

Blend (fit per k, only on events whose race finished before the cutoff, recency-weighted):
    q_def ~ a + b * qp_pre + c * best_k          -> new qp, and a smaller grid noise
    r_def ~ a + b * rp_pre + c * long_k          -> new rp
With fewer than MIN_EVENTS training events for a k, the model is left unchanged.

Practice fastest lap (package C12b; model.PRACTICE_FASTEST, variant "practicefast", off by default; decision log
2026-10-07 "F1 practice fastest", provisional): who sets the fastest lap of FP1, FP2 and FP3 (Polymarket's "Practice
N fastest lap"). Each session's order of best laps is drawn as the qualifying order is, qp plus the qualifying noise,
plus PRACTICE_SIGMA of practice noise (simulate); once the session has run, its real order (classification). The
kinds race_fp1_fastest .. race_fp3_fastest (markets/kinds.toml) price stage_rank["fp<n>"] == 1 and settle on the
results' fp<n>_position: the session's order of best laps (deleted laps don't count), from the stored laps
(classification for the model's frames, outcomes for the database); no laps stored for the session: undecided.
score() is its walk-forward: each session priced one minute before it starts, scored on who was fastest.
"""

import numpy as np
import pandas as pd

from racinglines.models.inputs import laps as M

USE_PRACTICE = True
MIN_EVENTS = 6
CLIP = 0.05      # deficits beyond 5% are crashes / aborted laps, not pace
ONE_MIN = pd.Timedelta(minutes=1)


def session_features(prac, event_id):
    """(k, best, long): sessions seen, and each driver's best one-lap deficit / mean long-run deficit."""
    p = prac[prac["event_id"] == event_id] if prac is not None and len(prac) else None
    if p is None or p.empty:
        return 0, pd.Series(dtype=float), pd.Series(dtype=float)
    k = p["round"].nunique()
    return (k, p.groupby("athlete_id")["best_def"].min().clip(upper=CLIP),
            p.groupby("athlete_id")["long_def"].mean().clip(upper=CLIP))


def training_rows(meas, use_track=True):
    """One pass over events with practice data (see module docstring)."""
    from racinglines.models.inputs.laps_pricing import entry_list
    if meas.practice is None or meas.practice.empty:
        return pd.DataFrame()
    first = meas.practice.groupby("event_id")["session_ts"].min()
    rows = []
    for ev in meas.drivers.drop_duplicates("event_id").itertuples():
        if ev.event_id not in first.index:
            continue
        cutoff = first[ev.event_id] - ONE_MIN
        v = meas.view(cutoff)
        snap = M.snapshot(v.drivers, v.sectors, v.xmap if use_track else {}, cutoff, use_track)
        if not snap.team_q:
            continue
        tf = M.venue_track_features(v.prof, ev.venue, cutoff)
        e = M.predict_paces(snap, entry_list(meas, ev.event_id), tf)[["athlete_id", "qp", "rp"]]
        e = e.rename(columns={"qp": "qp_pre", "rp": "rp_pre"})
        p = meas.practice[meas.practice["event_id"] == ev.event_id].sort_values("session_ts")
        order = list(dict.fromkeys(p["round"]))
        for k in range(1, len(order) + 1):
            pk = p[p["round"].isin(order[:k])]
            e[f"best_{k}"] = e["athlete_id"].map(pk.groupby("athlete_id")["best_def"].min()).clip(upper=CLIP)
            e[f"long_{k}"] = e["athlete_id"].map(pk.groupby("athlete_id")["long_def"].mean()).clip(upper=CLIP)
        out = meas.drivers[meas.drivers["event_id"] == ev.event_id][["athlete_id", "q_def", "r_def"]]
        e = e.merge(out, on="athlete_id", how="left")
        e["event_id"], e["r_ts"], e["start_date"] = ev.event_id, ev.r_ts, ev.start_date
        rows.append(e)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit(d, x_pre, x_prac, y, before):
    d = d[d[x_prac].notna() & d[y].notna() & d[x_pre].notna()]
    if d["event_id"].nunique() < MIN_EVENTS:
        return None
    w = M._weights(d["start_date"], before) + 0.05
    X = np.column_stack([np.ones(len(d)), d[x_pre], d[x_prac]])
    Y = d[y].clip(upper=CLIP).to_numpy()
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * W[:, None], Y * W, rcond=None)
    sd = float(np.sqrt(np.average((Y - X @ coef) ** 2, weights=w)))
    return coef, sd


def fit(train, before):
    """{('q'|'r', k): (coef[a, b, c], resid_sd)} from events whose race finished before `before`."""
    if train is None or not len(train) or not USE_PRACTICE:
        return {}
    t = train[train["r_ts"] + M.RACE_DONE < before]
    out = {}
    for k in (1, 2, 3, 4):
        if f"best_{k}" not in t:
            continue
        q = _fit(t, "qp_pre", f"best_{k}", "q_def", before)
        r = _fit(t, "rp_pre", f"long_{k}", "r_def", before)
        if q:
            out[("q", k)] = q
        if r:
            out[("r", k)] = r
    return out


def apply(e, prac_view, event_id, blend):
    """Blend e's qp / rp with this weekend's practice (only sessions in the as-of view).
    Returns (e, sigma_q or None)."""
    k, best, long_ = session_features(prac_view, event_id)
    if not k or not blend:
        return e, None
    e = e.copy()
    sigma_q = None
    kq = max((kk for (w, kk) in blend if w == "q" and kk <= k), default=None)
    if kq is not None and len(best):
        (a, b, c), sd = blend[("q", kq)]
        x = e["athlete_id"].map(best).fillna(best.median())
        e["qp"] = a + b * e["qp"] + c * x
        sigma_q = sd
    kr = max((kk for (w, kk) in blend if w == "r" and kk <= k), default=None)
    lr = long_.dropna()
    if kr is not None and len(lr) >= 5:
        (a, b, c), _ = blend[("r", kr)]
        x = e["athlete_id"].map(lr).fillna(lr.median())
        e["rp"] = a + b * e["rp"] + c * x
    return e, sigma_q


# --- practice fastest lap -----------------------------------------------------------------------------------------

def classification(prac, event_id=None):
    """Each practice session's order of best laps from practice_measurements rows (best_def per event x session x
    driver): event_id, round, athlete_id, position (competition rank: a tie shares the place). The sessions in
    model.PRACTICE_CLASSIFIED only; one event with `event_id`."""
    cols = ["event_id", "round", "athlete_id", "position"]
    if prac is None or not len(prac):
        return pd.DataFrame(columns=cols)
    p = prac[prac["round"].isin(M.PRACTICE_CLASSIFIED) & prac["best_def"].notna()]
    if event_id is not None:
        p = p[p["event_id"] == event_id]
    p = p.drop_duplicates(["event_id", "round", "athlete_id"])
    return p.assign(position=p.groupby(["event_id", "round"])["best_def"].rank(method="min"))[cols]


def with_positions(res, cls):
    """The result frame `res` with a <session>_position column per practice session in `cls` (classification's
    rows for one event): the order kinds.settle reads for race_fp<n>_fastest. A driver without a lap in a session that
    ran has no position there (settles NO); a session with no rows gets no column (undecidable)."""
    out = res.copy()
    for session, g in cls.groupby("round", sort=False):
        out[f"{session}_position"] = out["athlete_id"].map(g.set_index("athlete_id")["position"]).astype(float)
    return out


def simulate(meas, view, event_id, e, fm, rng, n_sims):
    """{session: (n_sims, n) order of best laps} for the weekend's practice sessions (model.PRACTICE_CLASSIFIED that
    the event's rows name): the real order once the session has ended before the view's cutoff (a driver without a
    lap after every driver with one), else drawn: qp + the qualifying noise (sigma_q, rho_q) + PRACTICE_SIGMA. Every
    draw is on a side stream seeded from `rng` without advancing it, so no other price moves."""
    have = set(meas.res.loc[meas.res["event_id"] == event_id, "round"])
    done = classification(view.practice, event_id)
    teams = e["team_key"].astype(str).to_numpy()
    qp = e["qp"].to_numpy(float)[None, :]
    n, out = len(e), {}
    for i, session in enumerate(M.PRACTICE_CLASSIFIED):
        if session not in have:
            continue
        real = done[done["round"] == session]
        if len(real):
            pos = e["athlete_id"].map(real.set_index("athlete_id")["position"]).astype(float).fillna(n + 1).to_numpy()
            rank = np.empty(n)
            rank[np.argsort(pos, kind="stable")] = np.arange(1, n + 1)
            rank = np.where(pos == 1, 1, rank)                  # a tie for the fastest lap: every tied driver is first
            out[session] = np.tile(rank, (n_sims, 1))
            continue
        side = M._side_rng(rng, M.PRACTICE_SEED + i)
        q = qp + M._noise(side, fm.sigma_q, fm.rho_q, teams, n_sims) + side.normal(0, M.PRACTICE_SIGMA, (n_sims, n))
        out[session] = M.ranks(q).astype(float)
    return out


def outcomes(conn, race_id):
    """Per athlete, each stored practice session's order of best laps for a race in the database: athlete_id and
    <session>_position (competition rank of the driver's fastest non-deleted lap time), the sessions in
    model.PRACTICE_CLASSIFIED that have laps. Just athlete_id when none has: the race_fp<n>_fastest kinds are then
    undecidable. The settlement frame (private_book.race_outcomes) does not merge it yet (settlement parity, C5)."""
    from sqlalchemy import text
    rows = pd.read_sql(text("""
        SELECT r.athlete_id, ro.kind, min(l.lap_time_ms) AS best
        FROM laps l JOIN results r ON r.id = l.result_id JOIN rounds ro ON ro.id = r.round_id
        WHERE ro.race_id = :r AND ro.kind = ANY(:k) AND l.lap_time_ms > 0 AND NOT coalesce(l.deleted, false)
              AND r.athlete_id IS NOT NULL
        GROUP BY r.athlete_id, ro.kind"""), conn, params=dict(r=race_id, k=list(M.PRACTICE_CLASSIFIED)))
    out = pd.DataFrame(dict(athlete_id=pd.Series(sorted(set(rows["athlete_id"])), dtype="int64")))
    for session in M.PRACTICE_CLASSIFIED:
        g = rows[rows["kind"] == session]
        if len(g):
            pos = g.set_index("athlete_id")["best"].rank(method="min")
            out[f"{session}_position"] = out["athlete_id"].map(pos).astype(float)
    return out


def score(meas, hist, n_sims=2000, seed=0, events=None, use_track=True):
    """The practice-fastest walk-forward: every stored practice session (model.PRACTICE_CLASSIFIED with laps) priced
    by price_race one minute before it starts, with PRACTICE_FASTEST on, and scored on who set its fastest lap. One
    row per event x session: n (drivers priced), fastest (athlete id), p_fastest (the model's fair for them), logloss
    (-log of it, floored at 1e-4), uniform (log n), brier (over every driver), favourite and its fair. `events`: event
    ids to score (default every event with practice laps). One seeded generator for the run, in date order."""
    from racinglines.models.inputs import laps_pricing as run
    rng = np.random.default_rng(seed)
    cls = classification(meas.practice)
    order = meas.drivers.drop_duplicates("event_id").sort_values("r_ts")["event_id"].tolist()
    rows = []
    old = M.PRACTICE_FASTEST
    M.PRACTICE_FASTEST = True
    try:
        for event_id in order:
            if events is not None and event_id not in set(events):
                continue
            starts = meas.sessions(event_id)
            for session in M.PRACTICE_CLASSIFIED:
                real = cls[(cls["event_id"] == event_id) & (cls["round"] == session)]
                start = starts.get(session)
                if real.empty or start is None or pd.isna(start):
                    continue
                cutoff = start - ONE_MIN
                if not (hist["r_ts"] + M.RACE_DONE < cutoff).any():
                    continue
                _, ex = run.price_race(meas, hist, cutoff, event_id, n_sims=n_sims, rng=rng, use_track=use_track)
                ids = ex["entrants"]["athlete_id"].tolist()
                fair = (ex["sim"]["practice"][session] == 1).mean(0)
                y = np.array([float(a in set(real.loc[real["position"] == 1, "athlete_id"])) for a in ids])
                if y.sum() == 0:
                    continue                              # the fastest driver isn't in the entry list
                p = float(fair[y == 1].sum())
                rows.append(dict(event_id=event_id, session=session, cutoff=str(cutoff), n=len(ids),
                                 fastest=int(np.array(ids)[y == 1][0]), p_fastest=p,
                                 logloss=float(-np.log(max(p, 1e-4))), uniform=float(np.log(len(ids))),
                                 brier=float(np.mean((fair - y) ** 2)), favourite=int(ids[int(np.argmax(fair))]),
                                 p_favourite=float(fair.max())))
    finally:
        M.PRACTICE_FASTEST = old
    return pd.DataFrame(rows)
