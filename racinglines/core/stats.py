"""Small numerical helpers shared across models and scoring."""

import numpy as np


def ranks(x):
    """Per-row 1-based ranks along axis 1 of an (n_sims, n) array (lower value = better
    rank). Non-finite values (didn't run / didn't finish) get rank inf."""
    n_sims, n = x.shape
    order = np.argsort(x, axis=1)
    r = np.empty((n_sims, n))
    np.put_along_axis(r, order, np.broadcast_to(np.arange(1, n + 1, dtype=float), (n_sims, n)), axis=1)
    return np.where(np.isfinite(x), r, np.inf)


def brier(p, y):
    """Mean squared error of probabilities p against 0/1 outcomes y."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def to_ms(seconds):
    """Seconds -> integer milliseconds (None for missing)."""
    import pandas as pd
    return None if seconds is None or pd.isna(seconds) else int(round(float(seconds) * 1000))


def last_at(ts, px, t, max_age):
    """Last value of the series (ts sorted ascending, px aligned) at or before t, or None
    if there is none or it is older than max_age."""
    i = np.searchsorted(ts, t, side="right") - 1
    if i < 0 or (t - ts[i]) > max_age:
        return None
    return float(px[i])
