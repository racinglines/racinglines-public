"""user MCP tokens for API access (bearer token auth with hashing)

Revision ID: d9f4b1c7e5a2
Revises: d9f2a7e5c1b3
Create Date: 2026-10-02 21:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'd9f4b1c7e5a2'
down_revision: Union[str, None] = 'd9f2a7e5c1b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('mcp_tokens', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'mcp_tokens')
