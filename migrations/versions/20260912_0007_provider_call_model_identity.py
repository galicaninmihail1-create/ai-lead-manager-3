"""Persist provider-reported model identity.

Revision ID: 20260912_0007
Revises: 20260912_0006
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260912_0007"
down_revision: str | None = "20260912_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "provider_calls",
        sa.Column("provider_model", sa.String(length=200), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("provider_calls", "provider_model")
