"""Constrain supported customer identity types.

Revision ID: 20260911_0003
Revises: 20260911_0002
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260911_0003"
down_revision: str | None = "20260911_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_customer_identities_supported_type",
        "customer_identities",
        "identity_type IN ('email', 'phone', 'external_user_id')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_customer_identities_supported_type",
        "customer_identities",
        type_="check",
    )
