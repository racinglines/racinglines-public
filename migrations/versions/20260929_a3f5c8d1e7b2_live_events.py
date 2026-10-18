"""live_events: every settled live private-book event (its dates, the book's P&L, the crowd's totals)

Revision ID: a3f5c8d1e7b2
Revises: e4c1a9b7d205
Create Date: 2026-09-29 04:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'a3f5c8d1e7b2'
down_revision: Union[str, None] = 'e4c1a9b7d205'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'live_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('run', sa.String(60), nullable=False, unique=True),
        sa.Column('sport', sa.String(20), nullable=False),
        sa.Column('event_key', sa.String(40), nullable=False, index=True),
        sa.Column('title', sa.Text()),
        sa.Column('opened_at', sa.DateTime(timezone=True)),
        sa.Column('settled_at', sa.DateTime(timezone=True)),
        sa.Column('maker_pnl', sa.Float()),
        sa.Column('crowd_pnl', sa.Float()),
        sa.Column('taker_pnl', sa.Float()),
        sa.Column('fills', sa.Integer()),
        sa.Column('volume', sa.Float()),
        sa.Column('detail', postgresql.JSONB()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('live_events')
