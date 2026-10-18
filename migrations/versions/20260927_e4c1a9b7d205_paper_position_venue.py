"""paper_positions.venue: polymarket (paper trading on the exchange) or private (a maker's private book)

Revision ID: e4c1a9b7d205
Revises: d2b8e5a1c3f7
Create Date: 2026-09-27 23:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'e4c1a9b7d205'
down_revision: Union[str, None] = 'd2b8e5a1c3f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('paper_positions', sa.Column('venue', sa.String(20), nullable=False, server_default='polymarket'))


def downgrade() -> None:
    op.drop_column('paper_positions', 'venue')
