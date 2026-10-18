"""strategy_signals and paper_positions (live paper signals of users' strategy profiles)

Revision ID: c7a1f4e2b9d0
Revises: 9c4d2e8f1a63
Create Date: 2026-09-27 20:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c7a1f4e2b9d0'
down_revision: Union[str, None] = '9c4d2e8f1a63'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'strategy_signals',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('candidate_id', sa.Integer(), nullable=True),
        sa.Column('profile', sa.String(120), nullable=False),
        sa.Column('strategy', sa.String(20), nullable=False),
        sa.Column('race_id', sa.Integer(), sa.ForeignKey('races.id', ondelete='SET NULL'), nullable=True),
        sa.Column('event_key', sa.String(20), nullable=False),
        sa.Column('market_key', sa.String(100), nullable=False),
        sa.Column('kind', sa.String(30), nullable=False),
        sa.Column('subject', sa.Text(), nullable=True),
        sa.Column('stage', sa.String(20), nullable=False),
        sa.Column('dedupe', sa.String(40), nullable=False),
        sa.Column('action', sa.String(12), nullable=False),
        sa.Column('side', sa.String(8), nullable=False),
        sa.Column('shares', sa.Float(), nullable=True),
        sa.Column('limit_price', sa.Float(), nullable=True),
        sa.Column('fair', sa.Float(), nullable=True),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('edge', sa.Float(), nullable=True),
        sa.Column('heat', sa.Integer(), nullable=True),
        sa.Column('target_cost', sa.Float(), nullable=True),
        sa.Column('status', sa.String(14), nullable=False, server_default='new'),
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('model_runs.id', ondelete='SET NULL'), nullable=True),
        sa.Column('signal_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('alerted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('detail', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.UniqueConstraint('user_id', 'candidate_id', 'market_key', 'dedupe', 'action', 'side'),
    )
    op.create_index('ix_strategy_signals_user_id', 'strategy_signals', ['user_id'])
    op.create_table(
        'paper_positions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('candidate_id', sa.Integer(), nullable=True),
        sa.Column('race_id', sa.Integer(), sa.ForeignKey('races.id', ondelete='SET NULL'), nullable=True),
        sa.Column('event_key', sa.String(20), nullable=False),
        sa.Column('market_key', sa.String(100), nullable=False),
        sa.Column('kind', sa.String(30), nullable=False),
        sa.Column('subject', sa.Text(), nullable=True),
        sa.Column('yes_shares', sa.Float(), nullable=False, server_default='0'),
        sa.Column('no_shares', sa.Float(), nullable=False, server_default='0'),
        sa.Column('cash', sa.Float(), nullable=False, server_default='0'),
        sa.Column('mark', sa.Float(), nullable=True),
        sa.Column('outcome', sa.Boolean(), nullable=True),
        sa.Column('bid', sa.Float(), nullable=True),
        sa.Column('ask', sa.Float(), nullable=True),
        sa.Column('quote_state', sa.String(30), nullable=True),
        sa.UniqueConstraint('user_id', 'candidate_id', 'market_key'),
    )
    op.create_index('ix_paper_positions_user_id', 'paper_positions', ['user_id'])


def downgrade() -> None:
    op.drop_table('paper_positions')
    op.drop_table('strategy_signals')
