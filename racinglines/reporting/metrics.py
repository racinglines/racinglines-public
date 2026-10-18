"""Dependency-free summary metrics used by generated reports."""

import math


def curve_stats(pnls):
    """Return per-season P&L metrics and the cumulative curve for weekend returns."""
    cumulative, peak, drawdown, total = [], 0.0, 0.0, 0.0
    for pnl in pnls:
        total += pnl
        cumulative.append(total)
        peak = max(peak, total)
        drawdown = max(drawdown, peak - total)
    count = len(pnls)
    mean = total / count if count else 0.0
    standard_deviation = math.sqrt(sum((pnl - mean) ** 2 for pnl in pnls) / (count - 1)) if count > 1 else 0.0
    return dict(pnl=total, weekends=count, weekends_up=sum(pnl > 0 for pnl in pnls), max_drawdown=drawdown,
                sharpe=(mean / standard_deviation * math.sqrt(24)) if standard_deviation else 0.0,
                mean_weekend=mean, sd_weekend=standard_deviation), cumulative