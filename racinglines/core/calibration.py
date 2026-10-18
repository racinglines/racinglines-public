"""
Calibration of probability forecasts, for any sport and any source of probabilities (our model or an
exchange's prices): reliability bins and the proper scores, from (probability, outcome) pairs.

    reliability(p, y)            per probability bin: n, mean predicted, observed frequency, z
    scores(p, y)                 n, Brier, log loss and ECE (the bins' weighted mean |predicted - observed|)
    table(rows, sources, by)     both, for several probability columns side by side (e.g. fair and price)

Bins are [lo, hi); the last one includes 1. A z beyond about ±2 is a bin the forecast gets wrong by more
than its binomial noise.
"""

import numpy as np
import pandas as pd

BINS = (0, .02, .05, .1, .2, .35, .5, .7, .9, 1.0001)
EPS = 1e-4


def reliability(p, y, bins=BINS):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.digitize(p, bins) - 1
    out = []
    for i in range(len(bins) - 1):
        sel = idx == i
        if not sel.any():
            continue
        m, o = float(p[sel].mean()), float(y[sel].mean())
        se = np.sqrt(max(m * (1 - m), 1e-6) / sel.sum())
        out.append(dict(lo=bins[i], hi=min(bins[i + 1], 1.0), n=int(sel.sum()), predicted=m, observed=o,
                        z=(o - m) / se))
    return out


def scores(p, y, bins=BINS):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return dict(n=0, brier=None, logloss=None, ece=None)
    q = np.clip(p, EPS, 1 - EPS)
    rel = reliability(p, y, bins)
    return dict(n=len(p), brier=float(((p - y) ** 2).mean()),
                logloss=float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()),
                ece=float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rel) / len(p)))


def table(rows, sources, by=(), outcome="y", bins=BINS):
    """rows: DataFrame with an outcome column and one probability column per source.
    Returns (summary, bins): one summary row per group in `by` and source (n, Brier, log loss, ECE), and
    the reliability bins of each. Rows missing a source's probability are left out of that source only."""
    rows = pd.DataFrame(rows)
    summary, rel = [], []
    groups = rows.groupby(list(by), sort=True) if by else [((), rows)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        tag = dict(zip(by, key))
        for src in sources:
            h = g[g[src].notna() & g[outcome].notna()]
            summary.append(dict(tag, source=src, **scores(h[src], h[outcome], bins)))
            rel += [dict(tag, source=src, **r) for r in reliability(h[src], h[outcome], bins)]
    return pd.DataFrame(summary), pd.DataFrame(rel)
