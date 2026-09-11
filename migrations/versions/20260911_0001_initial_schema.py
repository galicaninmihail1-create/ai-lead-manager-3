"""Create initial multi-tenant schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260911_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table("tenants", sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False), sa.Column("slug", sa.String(100), nullable=False),
        *timestamps(), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_tenants_slug", "tenants", ["slug"], unique=True)
    op.create_table("customers", sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False), sa.Column("name", sa.String(200), nullable=False),
        sa.Column("email", sa.String(320)), sa.Column("phone", sa.String(50)), *timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_customers_id_tenant"),
        sa.UniqueConstraint("tenant_id", "email", name="uq_customers_tenant_email"))
    op.create_index("ix_customers_tenant_created", "customers", ["tenant_id", "created_at"])
    op.create_table("conversations", sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False), sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("channel", sa.String(50), nullable=False), sa.Column("external_id", sa.String(255)),
        sa.Column("status", sa.String(50), nullable=False), *timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["customer_id", "tenant_id"], ["customers.id", "customers.tenant_id"],
                                name="fk_conversations_customer_tenant", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("id", "tenant_id", name="uq_conversations_id_tenant"))
    op.create_index("ix_conversations_tenant_customer", "conversations", ["tenant_id", "customer_id"])
    op.create_table("leads", sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False), sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False), sa.Column("score", sa.Integer()),
        sa.Column("source", sa.String(100)), sa.Column("notes", sa.Text()), *timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["customer_id", "tenant_id"], ["customers.id", "customers.tenant_id"],
                                name="fk_leads_customer_tenant", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"), sa.UniqueConstraint("tenant_id", "customer_id", name="uq_leads_tenant_customer"))
    op.create_index("ix_leads_tenant_status_created", "leads", ["tenant_id", "status", "created_at"])
    op.create_table("messages", sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False), sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("direction", sa.String(20), nullable=False), sa.Column("body", sa.Text(), nullable=False),
        sa.Column("external_id", sa.String(255)), *timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id", "tenant_id"], ["conversations.id", "conversations.tenant_id"],
                                name="fk_messages_conversation_tenant", ondelete="CASCADE"), sa.PrimaryKeyConstraint("id"))
    op.create_index("ix_messages_tenant_conversation_created", "messages",
                    ["tenant_id", "conversation_id", "created_at"])


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_table("leads")
    op.drop_table("conversations")
    op.drop_table("customers")
    op.drop_table("tenants")
