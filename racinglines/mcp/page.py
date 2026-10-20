"""
One envelope for every tabular result the MCP server returns, so no call can flood a chat client's context:

    {"columns": [...], "rows": [{...}, ...], "total": 1187, "offset": 0, "limit": 50,
     "truncated": true, "next_offset": 50, "note": "50 of 1,187 rows; pass offset=50 for the next page"}

- rows are capped (DEFAULT_LIMIT per call, MAX_LIMIT at most; `limit` argument), `total` always says how many exist;
- a page over MAX_BYTES (RACINGLINES_MCP_MAX_BYTES) is cut to the rows that fit and says so;
- values are JSON-safe: floats rounded to 4 decimals, NaN -> null, timestamps ISO-8601, numpy -> Python;
- long text and JSON columns are previewed (PREVIEW_CHARS) with their full size, so a wide JSONB column
  (params, metrics, detail, log) never lands whole in a list; the `get_*` tool for one row returns it whole.
"""

import json
import math
import os
from datetime import date, datetime, time
from decimal import Decimal

import numpy as np
import pandas as pd

DEFAULT_LIMIT = 50
MAX_LIMIT = 500
MAX_BYTES = int(os.environ.get("RACINGLINES_MCP_MAX_BYTES", "24000"))
PREVIEW_CHARS = 200
FLOAT_DIGITS = 4


def clamp(limit, default=DEFAULT_LIMIT, top=MAX_LIMIT):
    """A caller's `limit` -> an int in [1, top] (None -> default)."""
    if limit is None:
        return default
    try:
        return max(1, min(int(limit), top))
    except (TypeError, ValueError):
        return default


def plain(v, preview=True):
    """One value -> something json.dumps accepts and a reader can use."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating, Decimal)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, FLOAT_DIGITS)
    if isinstance(v, (pd.Timestamp, datetime)):
        if pd.isna(v):
            return None
        return v.isoformat(timespec="seconds" if isinstance(v, datetime) else "auto")
    if isinstance(v, (date, time)):
        return v.isoformat()
    if isinstance(v, pd.Timedelta):
        return str(v)
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, dict):
        if preview:
            s = json.dumps(v, default=str)
            if len(s) > PREVIEW_CHARS:
                return dict(_preview=s[:PREVIEW_CHARS] + "...", _size=len(s), _keys=list(v)[:20])
        return {str(k): plain(x, preview) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        v = list(v)
        if preview and len(v) > 20:
            return dict(_preview=[plain(x, preview) for x in v[:20]], _size=len(v))
        return [plain(x, preview) for x in v]
    if isinstance(v, str):
        if preview and len(v) > PREVIEW_CHARS:
            return v[:PREVIEW_CHARS] + f"... [{len(v)} chars]"
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def record(d, preview=True):
    return {str(k): plain(v, preview) for k, v in dict(d).items()}


def _size(obj):
    return len(json.dumps(obj, default=str))


def page(rows, *, limit=None, offset=0, columns=None, total=None, note=None, max_bytes=None):
    """Rows (a DataFrame or a list of dicts) -> the envelope. `total` defaults to len(rows); pass it when the
    rows are already one page of a larger result (the query did the LIMIT/OFFSET)."""
    limit = clamp(limit)
    offset = max(0, int(offset or 0))
    max_bytes = max_bytes or MAX_BYTES
    if isinstance(rows, pd.DataFrame):
        df = rows
        if columns:
            keep = [c for c in columns if c in df.columns]
            missing = [c for c in columns if c not in df.columns]
            df = df[keep] if keep else df.iloc[:, :0]
        else:
            missing = []
        cols = list(df.columns)
        n = len(df) if total is None else int(total)
        sl = df.iloc[offset:offset + limit] if total is None else df.iloc[:limit]
        recs = [record(r) for r in sl.to_dict("records")]
    else:
        rows = list(rows)
        missing = []
        if columns:
            rows = [{k: r.get(k) for k in columns} for r in rows]
        cols = list(rows[0]) if rows else list(columns or [])
        n = len(rows) if total is None else int(total)
        sl = rows[offset:offset + limit] if total is None else rows[:limit]
        recs = [record(r) for r in sl]
    cut = False
    while recs and _size(recs) > max_bytes:
        recs = recs[: max(1, len(recs) * 3 // 4)] if len(recs) > 1 else []
        cut = True
    if cut and not recs:
        recs, cut = [], True
    shown = len(recs)
    truncated = offset + shown < n
    out = dict(columns=cols, rows=recs, total=n, offset=offset, limit=limit, truncated=truncated,
               next_offset=(offset + shown) if truncated else None)
    notes = []
    if cut:
        notes.append(f"cut to {shown} rows to stay under {max_bytes:,} bytes; ask for fewer columns or a smaller limit")
    if truncated:
        notes.append(f"{shown} of {n:,} rows; pass offset={offset + shown} for the next page")
    elif n:
        notes.append(f"all {n:,} rows")
    if missing:
        notes.append(f"unknown columns ignored: {', '.join(missing)}")
    if note:
        notes.insert(0, note)
    out["note"] = "; ".join(notes)
    return out


def describe(df, name="rows"):
    """A summary instead of rows: count, columns with types, numeric ranges, top values of short text columns."""
    if df is None or not len(df):
        return dict(count=0, columns=[])
    cols = []
    for c in df.columns:
        s = df[c]
        info = dict(name=str(c), dtype=str(s.dtype), nulls=int(s.isna().sum()))
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info.update(min=plain(s.min()), max=plain(s.max()), mean=plain(s.mean()))
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.update(min=plain(s.min()), max=plain(s.max()))
        elif s.dtype == object:
            vc = s.dropna().astype(str).value_counts()
            if len(vc) <= 12:
                info["values"] = {str(k): int(v) for k, v in vc.items()}
            else:
                info["distinct"] = int(len(vc))
        cols.append(info)
    return dict(count=int(len(df)), columns=cols)
