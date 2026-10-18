"""An event's tape and a maker replay's settings as JSON, for the Rust/WebAssembly replay in wasm/ (the browser's
Maker replay on the diagnostics page, and scripts/bench). The Python replay (maker_replay.replay) stays the
reference; tests/test_wasm_replay.py checks the two agree on the same JSON.

Only what a replay reads goes in: per market its kind, subject, fairs per stage, outcome (read after the replay for
P&L), minute mids and trades (YES perspective). No order books, so the "queue" fill rule stays server-side."""

import math

from racinglines.markets import settlement_rules as SR


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _outcome(o):
    """Market.outcome: None (undecided), True / False, a payout in [0, 1], "void" or "fair"."""
    if o is None or o == SR.FAIR:
        return None if o is None else "fair"
    if o == SR.VOID:
        return "void"
    if isinstance(o, bool) or type(o).__name__ == "bool_":
        return bool(o)
    return float(o)


def event_json(data):
    """maker_replay.load_event's dict as the wasm replay's Event."""
    return dict(
        stages=[dict(run_id=int(s["run_id"]), start=int(s["start"]), end=int(s["end"]),
                     session_end=bool(s.get("session_end", True))) for s in data["stages"]],
        qual_start=None if data.get("qual_start") is None else int(data["qual_start"]),
        markets=[dict(kind=m.kind, subject=m.subject or "", outcome=_outcome(m.outcome),
                      fairs={str(k): _num(v) for k, v in m.fairs.items()},
                      mid_ts=m.mid_ts.astype("int64").tolist(), mid_px=m.mid_px.astype(float).tolist(),
                      tr_ts=m.tr_ts.astype("int64").tolist(), tr_px=m.tr_px.astype(float).tolist(),
                      tr_sz=m.tr_sz.astype(float).tolist(), tr_buy=[bool(x) for x in m.tr_buy])
                 for m in data["markets"]])


def params_json(p):
    """maker_replay.Params as the wasm replay's Params (same field names; fill "touch" or "through")."""
    if p.fill not in ("touch", "through"):
        raise ValueError(f"the wasm replay has fill rules touch and through, not {p.fill!r}")
    return dict(half_spread=p.half_spread, tick=p.tick, size=p.size, max_pos=p.max_pos, max_capital=p.max_capital,
                skew=p.skew, step_min=p.step_min, pull_min=p.pull_min, min_volume_24h=p.min_volume_24h,
                price_band=list(p.price_band), max_disagree=p.max_disagree, fill=p.fill,
                flatten_before_qual=p.flatten_before_qual, taker_cost=p.taker_cost, info_skew=p.info_skew,
                info_tau_h=p.info_tau_h, half_spread_by_kind=dict(p.half_spread_by_kind or {}),
                maker_fee=p.maker_fee)
