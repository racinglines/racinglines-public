"""Small helpers the sport adapters share."""

import numpy as np
import pandas as pd

from racinglines.frames.schema import SCHEMAS


def gap_to_leader(c):
    """`gap_ms` for a classifications frame: each row's time_ms minus the time_ms of the P1 row of its
    (event_id, session); null where either is missing."""
    lead = c[c["position"] == 1].groupby(["event_id", "session"])["time_ms"].first()
    at = pd.MultiIndex.from_frame(c[["event_id", "session"]])
    return c["time_ms"] - lead.reindex(at).to_numpy(dtype=float, na_value=np.nan)


DTYPES = {"id": object, "str": object, "object": object, "int": "int64", "number": "float64", "bool": bool,
          "datetime": "datetime64[ns]"}


def empty(name):
    """An empty frame `name`: the declared columns in order, each typed for its dtype family (so it validates and
    concatenates like a full frame; a bare `DataFrame(columns=...)` is all object)."""
    return pd.DataFrame({c.name: pd.Series([], dtype=DTYPES[c.dtype]) for c in SCHEMAS[name].columns})


def shape(name, df):
    """`df` as frame `name`: exactly the declared columns in the declared order, rows sorted by the key with a fresh
    index (so a frame is the same however its source happened to be ordered). A frame with no rows comes back as
    `empty(name)`, typed."""
    sc = SCHEMAS[name]
    if df.empty:
        return empty(name)
    return df[list(sc.column_names)].sort_values(list(sc.key), kind="stable").reset_index(drop=True)
