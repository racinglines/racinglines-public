"""
What the web forms let a user type for a numeric setting (owner, 2026-10-11 22:29Z: "make sure the dynamic range of all
input sliders is reasonable. err on the side of a sensible cap rather than a gigantic range"). The strategy caps follow
what our sweeps have backtested (library/core/2026-10-11-strategies-per-sport.md: min edge 0.03-0.15, 24 h volume
floor up to 400, disagreement cut-off from 0.05); defaults are unchanged.

The bounds in sweep_settings.Setting and jobs.Knob are what the engine accepts (the CLI and the MCP tools keep them);
these are tighter caps for the Lab's forms and the event diagnostic's replay sliders only. Every cap holds the setting's default and every library strategy's
value (profiles.PROFILES). A saved value outside a cap (an older run, a CLI sweep) widens that one input to include it,
so a form never refuses to submit what it was loaded with.
"""

# name -> (low, high)
UI_RANGES = {
    # model
    "sims": (1000, 16000), "half_life_days": (30, 365), "half_life": (30, 365), "ridge_team": (0, 10),
    "ridge_slope": (0, 20), "driver_prior_n": (0, 10), "finish_rho_scale": (0, 2), "reset_weight": (0, 1),
    "races": (5, 60), "junior_weight": (0, 1),
    # taker (fantasy accounts start with 1,000)
    "min_edge": (0.03, 0.15), "min_edge_h2h": (0.03, 0.15), "stake_per_edge": (50, 1000), "max_stake": (10, 200),
    "cost": (0, 0.03), "thin_edge_mult": (1, 5), "bankroll": (500, 10000), "kelly": (0.05, 1.0),
    "max_deployed": (100, 5000), "capital": (200, 5000),
    # markets
    "min_volume_24h": (0, 400),
    # maker (hidden while profiles.MAKERS_OPEN is off)
    "half_spread": (0.005, 0.10), "size": (10, 100), "max_pos": (50, 1000), "skew": (0, 3), "max_disagree": (0.05, 0.15),
    "maker_min_volume_24h": (0, 1000), "info_skew": (0, 5), "widen": (1, 3),
    # event diagnostic replay sliders (diag.html) besides the above
    "max_capital": (100, 5000), "pull_min": (0, 120),
}


def ui_range(name, value=None, lo=None, hi=None):
    """(min, max) for a form input: the UI cap within the engine's (lo, hi), widened to hold `value`; else the engine's own (lo, hi)."""
    if name not in UI_RANGES:
        return lo, hi
    a, b = UI_RANGES[name]
    a, b = (a if lo is None else max(a, lo)), (b if hi is None else min(b, hi))     # never past what the engine takes
    try:
        v = float(value)
    except (TypeError, ValueError):
        return a, b
    return min(a, v), max(b, v)
