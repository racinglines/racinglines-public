"""market_links.first_seen_at (when sync first saw the token on the exchange; drives "new" badges and alerts)

Revision ID: 9c4d2e8f1a63
Revises: 5b1e0c7d2a41
Create Date: 2026-09-27 18:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '9c4d2e8f1a63'
down_revision: Union[str, None] = '5b1e0c7d2a41'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('market_links', sa.Column('first_seen_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('market_links', 'first_seen_at')
