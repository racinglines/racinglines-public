"""market_disagreements: the cross-venue disagreement log (Polymarket vs Kalshi, per outcome and tick)

Revision ID: c8e3f6a2d4b1
Revises: b7d2e4f1a9c3
Create Date: 2026-09-29 06:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'c8e3f6a2d4b1'
down_revision: Union[str, None] = 'b7d2e4f1a9c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'market_disagreements',
        sa.Column('pm_token', sa.String(100), primary_key=True),
        sa.Column('kalshi_token', sa.String(100), primary_key=True),
        sa.Column('ts', sa.DateTime(timezone=True), primary_key=True),
        sa.Column('competition_id', sa.Integer(), sa.ForeignKey('competitions.id'), nullable=False),
        sa.Column('race_id', sa.Integer(), sa.ForeignKey('races.id', ondelete='CASCADE')),
        sa.Column('kind', sa.String(30), nullable=False),
        sa.Column('athlete_id', sa.Integer(), sa.ForeignKey('athletes.id', ondelete='CASCADE')),
        sa.Column('team', sa.String(40)),
        sa.Column('opponent_id', sa.Integer()),
        sa.Column('n', sa.Integer()),
        sa.Column('pm_mid', sa.Float()),
        sa.Column('pm_bid', sa.Float()),
        sa.Column('pm_ask', sa.Float()),
        sa.Column('kalshi_mid', sa.Float()),
        sa.Column('kalshi_bid', sa.Float()),
        sa.Column('kalshi_ask', sa.Float()),
        sa.Column('fair', sa.Float()),
        sa.Column('run_id', sa.Integer(), sa.ForeignKey('model_runs.id', ondelete='SET NULL')),
        sa.Column('pm_edge', sa.Float()),
        sa.Column('pm_side', sa.String(4)),
        sa.Column('kalshi_edge', sa.Float()),
        sa.Column('kalshi_side', sa.String(4)),
        sa.Column('gap', sa.Float()),
        sa.Column('gap_net', sa.Float()),
        sa.Column('pm_vol24', sa.Float()),
        sa.Column('kalshi_vol24', sa.Float()),
    )
    op.create_index('ix_market_disagreements_competition_ts', 'market_disagreements', ['competition_id', 'ts'])
    op.create_index('ix_market_disagreements_race_ts', 'market_disagreements', ['race_id', 'ts'])


def downgrade() -> None:
    op.drop_index('ix_market_disagreements_race_ts', table_name='market_disagreements')
    op.drop_index('ix_market_disagreements_competition_ts', table_name='market_disagreements')
    op.drop_table('market_disagreements')
