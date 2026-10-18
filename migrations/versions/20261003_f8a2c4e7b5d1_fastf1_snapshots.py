"""FastF1 live session snapshots and driver positions

Revision ID: f8a2c4e7b5d1
Revises: d9f4b1c7e5a2
Create Date: 2026-10-03 12:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'f8a2c4e7b5d1'
down_revision: Union[str, None] = 'd9f4b1c7e5a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'fastf1_session_snapshots',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('year', sa.Integer(), nullable=False),
        sa.Column('round', sa.Integer(), nullable=False),
        sa.Column('session_type', sa.String(20), nullable=False),  # fp1, fp2, fp3, sq, sprint, qual, race
        sa.Column('status', sa.String(20), nullable=False),  # not_started, ongoing, completed, paused
        sa.Column('lap_count', sa.Integer()),
        sa.Column('time_remaining', sa.Integer()),  # seconds
        sa.Column('laps_remaining', sa.Integer()),
        sa.Column('flag', sa.String(20)),  # green, yellow, red, chequered
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index('ix_fastf1_snapshots_year_round', 'fastf1_session_snapshots', ['year', 'round'])
    op.create_index('ix_fastf1_snapshots_session', 'fastf1_session_snapshots', ['year', 'round', 'session_type'])

    op.create_table(
        'fastf1_driver_positions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('snapshot_id', sa.Integer(), sa.ForeignKey('fastf1_session_snapshots.id', ondelete='CASCADE'), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('driver_number', sa.Integer(), nullable=False),
        sa.Column('driver_name', sa.String(100), nullable=False),
        sa.Column('team', sa.String(100)),
        sa.Column('gap_to_leader', sa.Float()),  # seconds
        sa.Column('last_lap_time', sa.Float()),  # seconds
        sa.Column('best_lap_time', sa.Float()),  # seconds
        sa.Column('status', sa.String(20)),  # on_track, pitted, out, dnf
        sa.Column('lap_count', sa.Integer()),
    )
    op.create_index('ix_fastf1_positions_snapshot', 'fastf1_driver_positions', ['snapshot_id'])


def downgrade() -> None:
    op.drop_index('ix_fastf1_positions_snapshot', table_name='fastf1_driver_positions')
    op.drop_table('fastf1_driver_positions')
    op.drop_index('ix_fastf1_snapshots_session', table_name='fastf1_session_snapshots')
    op.drop_index('ix_fastf1_snapshots_year_round', table_name='fastf1_session_snapshots')
    op.drop_table('fastf1_session_snapshots')
