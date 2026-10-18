"""
Template formatting helpers: money, dates, probabilities, market kinds and short column labels.
Registered as Jinja filters and globals by web/app.py.
"""

import math
from datetime import date, datetime, timezone
from numbers import Real

import pandas as pd


MONEY_COLS = {"pnl", "spread_pnl", "markout_60m", "markout_5m", "model_edge", "worst_case", "cash", "taker_pnl",
              "staked", "stake", "payout", "ev", "pnl_if_yes", "pnl_if_no", "cost", "proceeds", "notional", "settled_pnl",
              "paper_pnl", "replay_pnl", "kalshi_pnl", "worst"}
SIGNED_MONEY_COLS = {"pnl", "spread_pnl", "markout_60m", "markout_5m", "model_edge", "taker_pnl", "pnl_if_yes",
                     "pnl_if_no", "ev", "settled_pnl", "paper_pnl", "replay_pnl", "kalshi_pnl"}   # P&L-like: shown with a sign


def money(v, sign=False, cents=True):
    """-302.97 -> '-$302.97'; sign=True adds '+' to positives; cents=False keeps whole dollars ('-$303')."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = "-" if v < 0 else ("+" if sign and v > 0 else "")
    return f"{s}${abs(v):,.{2 if cents else 0}f}"


def _when(value):
    """One date format: 'YYYY-MM-DD HH:MM UTC', or 'YYYY-MM-DD' when there is no time part (tz-aware values go to UTC)."""
    if isinstance(value, pd.Timestamp) and pd.isna(value):
        return ""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%d %H:%M UTC") if (value.hour or value.minute) else value.strftime("%Y-%m-%d")
    return value.strftime("%Y-%m-%d")                                     # a plain date


def fmt(value, col=""):
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
        return ""
    if isinstance(value, (datetime, date)):                                # pd.Timestamp is a datetime
        return _when(value)
    if col in MONEY_COLS and isinstance(value, Real) and not isinstance(value, bool):
        return money(value, sign=col in SIGNED_MONEY_COLS)
    if isinstance(value, float):
        if col.endswith("_prob") or col in ("model_prob", "edge", "quoted_share", "mean_clv_pct", "positive_share"):
            return f"{value:.1%}"
        if col.endswith("time_s"):
            mins, secs = divmod(value, 60)
            return f"{int(mins)}:{secs:06.3f}" if mins else f"{secs:.3f}"
        if value.is_integer() and abs(value) < 1e6:
            return f"{int(value)}"
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)



def kind(k):
    """'race_win' -> 'win', 'race_top10' -> 'top10', 'race_make_final' -> 'make final'."""
    if k is None:
        return ""
    k = str(k).removeprefix("race_")
    return ("DH " + k.removeprefix("dh_") if k.startswith("dh_") else k).replace("_", " ")


def when(v):
    """fmt for an ISO string ('2026-09-27T14:05:00'; no zone means UTC)."""
    if not v:
        return ""
    t = pd.Timestamp(v)
    return fmt(t.tz_localize("UTC") if t.tzinfo is None else t)


# Short column headers for the `table` macro; a key not listed gets `_` -> space and a capital first letter.
LABELS = {"markets_made": "Markets made", "bets_against": "Bets against", "bets_placed": "Bets placed", "staked": "Staked",
          "last_seen": "Last seen", "created_at": "Created", "pnl": "P&L", "pnl_if_yes": "P&L if yes", "pnl_if_no": "P&L if no",
          "ev": "EV", "id": "ID", "fair_prob": "Fair", "yes_price": "Yes", "no_price": "No",
          "win_prob": "Win", "podium_prob": "Podium", "top10_prob": "Top 10", "make_final_prob": "Makes final",
          "exp_points": "Exp. points", "actual_final_pos": "Final pos.", "time_s": "Time", "start_date": "Date",
          "series_round": "Round", "source_key": "Source", "last_race": "Last race", "brier_model": "Brier (model)",
          "brier_polymarket": "Brier (Polymarket)", "logloss_model": "Log loss (model)",
          "logloss_polymarket": "Log loss (Polymarket)", "run_id": "Run", "spread_pnl": "Spread P&L",
          "markout_60m": "Markout 60m", "model_edge": "Model edge", "worst_case": "Worst case", "quoted_share": "Quoted",
          "half_spread": "Half spread", "max_disagree": "Max disagree", "market_steps": "Market steps",
          "taker_pnl": "Taker P&L", "model_prob": "Model prob", "best_bid": "Best bid", "best_ask": "Best ask",
          "exchange_order_id": "Exchange order", "ts": "Time", "qty": "Qty", "mid": "Mid", "fair": "Fair"}
