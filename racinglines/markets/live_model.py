"""
Which model prices the board: the live default's model key per competition (owner, 2026-10-25).

The signal step saves a live stage run for every profile it runs (signals.price_stages_now), so for one stage of one
race there are several live runs with different model keys (today the default b684991c76b3, profiles A/T1/T8, and gbm
e64f1e7206b5, profiles C/K/T4). The board (markets/venues.pricing_run, so /markets, MCP list_markets and book slips)
reads only the live default's runs; before this it took the newest run of any key, so whichever profile finished last
priced the board (Austin 2026 after qualifying: gbm run 12281, saved two minutes after the default's 12280).

Changing a key here changes live prices: it is a deliberate change with a decision-log row and a data_changes entry,
never a side effect. A competition with no entry keeps the newest live stage run.
"""

LIVE_MODEL_KEY = {
    "f1_wdc": "b684991c76b3",          # the F1 live default: profile A's model (A, T1 and T8 share it)
}


def live_model_key(competition):
    """The model key whose live runs price `competition`'s board, or None (any live stage run)."""
    return LIVE_MODEL_KEY.get(competition)
