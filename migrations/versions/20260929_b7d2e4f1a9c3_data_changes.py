"""data_changes: the data change log (ingests that changed anything, athlete merges, notes)

Revision ID: b7d2e4f1a9c3
Revises: a3f5c8d1e7b2
Create Date: 2026-09-29 05:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'b7d2e4f1a9c3'
down_revision: Union[str, None] = 'a3f5c8d1e7b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'data_changes',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column('sport', sa.String(20)),
        sa.Column('kind', sa.String(30), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('detail', postgresql.JSONB()),
        sa.Column('by', sa.String(80)),
    )


def downgrade() -> None:
    op.drop_table('data_changes')
