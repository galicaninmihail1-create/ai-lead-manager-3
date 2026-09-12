"""Add persistent tenant API keys.

Revision ID: 20260912_0008
Revises: 20260912_0007
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260912_0008"
down_revision: str | None = "20260912_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tenant_api_keys",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("key_prefix", sa.String(12), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column(
            "scopes",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("status", sa.String(10), server_default="ACTIVE", nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("status IN ('ACTIVE', 'REVOKED')", name="ck_tenant_api_keys_status"),
        sa.CheckConstraint("length(key_prefix) = 12", name="ck_tenant_api_keys_prefix_length"),
        sa.CheckConstraint("length(key_hash) = 64", name="ck_tenant_api_keys_hash_length"),
        sa.CheckConstraint(
            "(status = 'ACTIVE' AND revoked_at IS NULL) OR "
            "(status = 'REVOKED' AND revoked_at IS NOT NULL)",
            name="ck_tenant_api_keys_revocation_state",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_prefix", name="uq_tenant_api_keys_prefix"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_tenant_api_keys_id_tenant"),
    )
    op.create_index(
        "ix_tenant_api_keys_tenant_status", "tenant_api_keys", ["tenant_id", "status"]
    )


def downgrade() -> None:
    op.drop_table("tenant_api_keys")
