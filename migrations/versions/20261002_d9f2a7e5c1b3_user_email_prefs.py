"""user_email_prefs: email delivery settings per user (exchanges, sports interests)

Revision ID: d9f2a7e5c1b3
Revises: c8e3f6a2d4b1
Create Date: 2026-10-02 21:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'd9f2a7e5c1b3'
down_revision: Union[str, None] = 'c8e3f6a2d4b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'user_email_prefs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=254), nullable=True),
        sa.Column('exchanges', postgresql.ARRAY(sa.String()), nullable=False, server_default=sa.text("'{polymarket}'")),
        sa.Column('sports', postgresql.ARRAY(sa.String()), nullable=False, server_default=sa.text("'{f1_wdc}'")),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_user_email_prefs_user_id'), 'user_email_prefs', ['user_id'], unique=False)
    op.create_index(op.f('ix_user_email_prefs_email'), 'user_email_prefs', ['email'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_user_email_prefs_email'), table_name='user_email_prefs')
    op.drop_index(op.f('ix_user_email_prefs_user_id'), table_name='user_email_prefs')
    op.drop_table('user_email_prefs')
