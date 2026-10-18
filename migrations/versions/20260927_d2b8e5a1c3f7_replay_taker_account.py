"""polymarket-takers system account: the replay counterparty, separate from the demo taker

Replay fills recorded from a diagnostic page used to be bets by the demo `taker` login. They are
Polymarket's takers, not our demo taker (which paper trades its own strategy profile): this creates
the `polymarket-takers` system account (inactive, no login) and moves those bets to it.

Revision ID: d2b8e5a1c3f7
Revises: c7a1f4e2b9d0
Create Date: 2026-09-27 22:00:00
"""
from typing import Sequence, Union

from alembic import op

revision: str = 'd2b8e5a1c3f7'
down_revision: Union[str, None] = 'c7a1f4e2b9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

REPLAY = "Polymarket taker trade%filled the replayed maker"


def upgrade() -> None:
    op.execute("""INSERT INTO users (username, password_hash, role, active, display_name)
                  SELECT 'polymarket-takers', '!no-login', 'taker', false, 'Polymarket takers (replay counterparty)'
                  WHERE NOT EXISTS (SELECT 1 FROM users WHERE username = 'polymarket-takers')""")
    op.execute(f"""UPDATE house_bets SET taker_id = (SELECT id FROM users WHERE username = 'polymarket-takers'),
                                        counterparty = 'polymarket-takers'
                   WHERE note LIKE '{REPLAY}'
                     AND (taker_id IS NULL OR taker_id = (SELECT id FROM users WHERE username = 'taker'))""")


def downgrade() -> None:
    op.execute(f"""UPDATE house_bets SET taker_id = (SELECT id FROM users WHERE username = 'taker'), counterparty = 'taker'
                   WHERE taker_id = (SELECT id FROM users WHERE username = 'polymarket-takers') AND note LIKE '{REPLAY}'""")
