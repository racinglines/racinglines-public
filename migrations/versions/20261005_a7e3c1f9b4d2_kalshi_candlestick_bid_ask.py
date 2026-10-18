"""kalshi candlestick bid/ask columns for accurate taker replay

Revision ID: a7e3c1f9b4d2
Revises: f8a2c4e7b5d1
Create Date: 2026-10-05 06:25:00

Adds bid and ask columns to market_price_history to store candlestick bid/ask closes
separately from the (potentially spiky) last-trade price. Enables accurate taker replay
fills against Kalshi order books.

Backward compatible: price column retained for existing queries; new code reads bid/ask.
Re-pull of 2025-26 Kalshi history required to populate these columns.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a7e3c1f9b4d2'
down_revision: Union[str, None] = 'f8a2c4e7b5d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add bid and ask columns (nullable for backward compatibility during re-pull)
    op.add_column('market_price_history', sa.Column('bid', sa.Float(), nullable=True))
    op.add_column('market_price_history', sa.Column('ask', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('market_price_history', 'ask')
    op.drop_column('market_price_history', 'bid')
