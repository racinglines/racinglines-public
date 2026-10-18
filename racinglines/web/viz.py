"""Server-rendered SVG charts (no JavaScript): exchange price lines with our fair values."""

import warnings
from typing import TypedDict

import numpy as np
import pandas as pd

COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#d97706", "#0891b2"]


class ChartTick(TypedDict, total=False):
    """One axis label or marker: x (xticks, markers) or y (yticks) in SVG coordinates, the text shown, and
    (markers only) which end of the text to anchor at that point."""
    x: float
    y: float
    label: str
    anchor: str              # markers only: "start" | "end"


class ChartLine(TypedDict, total=False):
    """One series on the chart (`price_chart`: one venue's price; `line_chart`: one line of a P&L curve)."""
    label: str
    color: str
    points: str              # "x,y x,y ..." ready for an SVG <polyline>
    fair_y: float | None     # price_chart only: our fair value's y, for the dashed reference line
    dash: bool               # line_chart only: a highlighted / secondary series drawn dashed


class ChartData(TypedDict, total=False):
    """The `chart` template macro's input (racinglines/web/templates/_macros.html): every producer (price_chart,
    line_chart, and views.py's per-venue wrappers) returns exactly this shape, or None when there's nothing to
    draw. A new venue's chart is a new dict of this shape, nothing template-side."""
    w: int
    h: int
    pad: int
    lines: list[ChartLine]
    yticks: list[ChartTick]
    xticks: list[ChartTick]
    markers: list[ChartTick]
    zero_y: float | None    # line_chart only: the y of value 0, for the zero line
    venue: str              # set by callers (e.g. views._race_chart) when the chart isn't Polymarket's


def _normalize_series_points(key, pts):
    """Best-effort timestamp normalization for chart inputs.

    We keep the page from crashing even when a series is temporarily backwards or malformed: valid timestamps
    are sorted before plotting, invalid timestamps are left in place with a warning, and malformed items are also
    warned about without raising.
    """
    if not pts:
        return []
    try:
        out = []
        for item in pts:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(f"chart series {key!r} contains a malformed point: {item!r}")
            ts, value = item
            try:
                ts = pd.to_datetime(ts, utc=True)
            except (TypeError, ValueError):
                raise TypeError(f"chart series {key!r} has a non-timestamp x-value: {ts!r}") from None
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise TypeError(f"chart series {key!r} has a non-numeric y-value: {value!r}") from None
            out.append((ts, value))
    except TypeError as ex:
        warnings.warn(f"Could not sort incoming chart data for {key!r}: {ex}; skipping this series to keep the page alive.",
                      UserWarning, stacklevel=2)
        return []

    if len(out) >= 2 and not all(out[i][0] <= out[i + 1][0] for i in range(len(out) - 1)):
        warnings.warn(f"Incoming chart data for {key!r} is not in timestamp order; sorting before plotting.",
                      UserWarning, stacklevel=2)
        out = sorted(out, key=lambda item: item[0])
    return out


def _normalize_markers(markers):
    out = []
    for item in markers:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            warnings.warn(f"Could not sort incoming chart markers: {item!r}; skipping them to keep the page alive.",
                          UserWarning, stacklevel=2)
            return []
        ts, label = item
        try:
            ts = pd.to_datetime(ts, utc=True)
        except (TypeError, ValueError):
            warnings.warn(f"Could not sort incoming chart markers for {label!r}: {ts!r} is not a timestamp; "
                          "skipping them to keep the page alive.", UserWarning, stacklevel=2)
            return []
        out.append((ts, label))
    if len(out) >= 2 and not all(out[i][0] <= out[i + 1][0] for i in range(len(out) - 1)):
        warnings.warn("Incoming chart markers are not in timestamp order; sorting before plotting.", UserWarning,
                      stacklevel=2)
        out = sorted(out, key=lambda item: item[0])
    return [(ts, label) for ts, label in out]


def price_chart(series, labels, fairs=None, markers=(), w=900, h=220, pad=34) -> ChartData | None:
    """series: {key: [(ts, price), ...]}; labels/fairs: {key: ...}; markers: [(ts, label)].
    Returns a ChartData dict for the `chart` template macro, or None if there's nothing to draw."""
    series = {k: _normalize_series_points(k, v) for k, v in series.items()}
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return None
    t0 = min(v[0][0] for v in series.values())
    t1 = max(v[-1][0] for v in series.values())
    span = max((t1 - t0).total_seconds(), 1)
    top = max(max(p for _, p in v) for v in series.values())
    top = max(top, max((f for f in (fairs or {}).values() if f is not None), default=0))
    ymax = min(1.0, max(0.1, round(top * 1.15 + 0.05, 1)))
    plot_h = h - 16

    def x(t):
        return pad + (w - pad - 4) * (t - t0).total_seconds() / span

    def y(p):
        return 4 + (plot_h - 8) * (1 - min(p, ymax) / ymax)

    lines = []
    for i, (k, pts) in enumerate(series.items()):
        f = (fairs or {}).get(k)
        lines.append(dict(label=labels.get(k, k), color=COLORS[i % len(COLORS)],
                          points=" ".join(f"{x(t):.1f},{y(p):.1f}" for t, p in pts),
                          fair_y=round(y(f), 1) if f is not None else None))
    yticks = [dict(y=round(y(v), 1), label=f"{v * 100:.0f}%") for v in [ymax * i / 4 for i in range(5)]]
    days = pd.date_range(pd.Timestamp(t0).ceil("D"), t1, freq="D")
    step = max(1, len(days) // 6)
    xticks = [dict(x=round(x(d), 1), label=d.strftime("%b %d")) for d in days[::step]]
    if len(xticks) < 2:
        hrs = pd.date_range(pd.Timestamp(t0).ceil("6h"), t1, freq="6h")
        xticks = [dict(x=round(x(d), 1), label=d.strftime("%d %H:%M")) for d in hrs]
    markers = _normalize_markers(markers)
    mk = [dict(x=round(x(t), 1), label=lab, anchor="end" if x(t) > w - 80 else "start") for t, lab in markers if t0 <= t <= t1]
    return dict(w=w, h=h, pad=pad, lines=lines, yticks=yticks, xticks=xticks, markers=mk)


def line_chart(series, labels=None, w=900, h=200, pad=48, money=True, include_zero=True, markers=(), highlight=()) -> ChartData | None:
    """series: {key: [(ts, value), ...]} -> a ChartData dict for the `chart` macro, with a $ axis and a zero line.
    include_zero=False fits the axis to the data (e.g. a bankroll); markers: [(ts, label)] dashed verticals."""
    series = {k: _normalize_series_points(k, v) for k, v in series.items()}
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return None
    t0 = min(v[0][0] for v in series.values())
    t1 = max(v[-1][0] for v in series.values())
    span = max((t1 - t0).total_seconds(), 1)
    lo = min(min(y for _, y in v) for v in series.values())
    hi = max(max(y for _, y in v) for v in series.values())
    if include_zero:
        lo, hi = min(0.0, lo), max(0.0, hi)
    rng = (hi - lo) or 1.0
    lo, hi = lo - 0.08 * rng, hi + 0.08 * rng
    plot_h = h - 16

    def x(t):
        return pad + (w - pad - 4) * (t - t0).total_seconds() / span

    def y(v):
        return 4 + (plot_h - 8) * (1 - (v - lo) / (hi - lo))

    lines = [dict(label=(labels or {}).get(k, k), color=COLORS[i % len(COLORS)], fair_y=None, dash=k in highlight,
                  points=" ".join(f"{x(t):.1f},{y(v):.1f}" for t, v in pts)) for i, (k, pts) in enumerate(series.items())]
    ticks = np.linspace(lo, hi, 5)
    yticks = [dict(y=round(y(v), 1), label=(f"${v:,.0f}" if money else f"{v:.2f}")) for v in ticks]
    fmt = "%b %d"
    if span <= 86400:                                   # a day or less (a private book): clock times
        freq = "15min" if span <= 3 * 3600 else "1h"
        months, fmt = pd.date_range(pd.Timestamp(t0).ceil(freq), t1, freq=freq), "%H:%M"
    else:
        months = pd.date_range(pd.Timestamp(t0).ceil("D"), t1, freq="MS")
        if len(months) < 2:
            months = pd.date_range(pd.Timestamp(t0).ceil("D"), t1, freq="7D")
    xticks = [dict(x=round(x(d), 1), label=d.strftime(fmt)) for d in months]
    markers = _normalize_markers(markers)
    mk = [dict(x=round(x(t), 1), label=lab, anchor="end" if x(t) > w * 0.8 else "start")
          for t, lab in markers if t0 <= t <= t1]
    return dict(w=w, h=h, pad=pad, lines=lines, yticks=yticks, xticks=xticks, markers=mk,
                zero_y=round(y(0.0), 1) if lo <= 0 <= hi else None)
