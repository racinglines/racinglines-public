"""user prefs (per-user UI settings, e.g. the Lab's Edge Finder combos)

Revision ID: 5b1e0c7d2a41
Revises: 81022900a936
Create Date: 2026-09-27 12:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '5b1e0c7d2a41'
down_revision: Union[str, None] = '81022900a936'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('prefs', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'prefs')
