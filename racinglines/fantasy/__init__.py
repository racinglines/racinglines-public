"""Fantasy trading for players: F$ only, never real money (docs/fantasy-trading.md).

    exchange.py   the first-cut manual trade: immediate-or-cancel against the exchanges' recorded order books
    ledger.py     the tables (accounts schema), balances, positions and the leaderboard

Nothing here imports an order path (markets/polymarket/trade.py, markets/kalshi/trade.py) or calls a venue: prices
come from the recorder's market_book_snapshots rows (tests/test_fantasy_exchange.py checks both).
"""
