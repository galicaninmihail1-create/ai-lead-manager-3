"""Add idempotent inbound core.

Revision ID: 20260911_0002
Revises: 20260911_0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260911_0002"
down_revision: str | None = "20260911_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "customer_identities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("identity_type", sa.String(length=30), nullable=False),
        sa.Column("normalized_value", sa.String(length=500), nullable=False),
        sa.Column("original_value", sa.String(length=500), nullable=True),
        *timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["customer_id", "tenant_id"],
            ["customers.id", "customers.tenant_id"],
            name="fk_customer_identities_customer_tenant",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "identity_type",
            "normalized_value",
            name="uq_customer_identities_tenant_type_value",
        ),
    )
    op.create_index(
        "ix_customer_identities_tenant_customer",
        "customer_identities",
        ["tenant_id", "customer_id"],
    )
    op.create_unique_constraint(
        "uq_conversations_tenant_channel_external",
        "conversations",
        ["tenant_id", "channel", "external_id"],
    )
    op.create_unique_constraint("uq_messages_id_tenant", "messages", ["id", "tenant_id"])
    op.add_column("leads", sa.Column("conversation_id", sa.Uuid(), nullable=True))
    op.create_unique_constraint("uq_leads_id_tenant", "leads", ["id", "tenant_id"])
    op.create_foreign_key(
        "fk_leads_conversation_tenant",
        "leads",
        "conversations",
        ["conversation_id", "tenant_id"],
        ["id", "tenant_id"],
    )
    op.create_table(
        "inbound_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("external_event_id", sa.String(length=255), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("customer_id", sa.Uuid(), nullable=True),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("message_id", sa.Uuid(), nullable=True),
        sa.Column("lead_id", sa.Uuid(), nullable=True),
        *timestamps(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["customer_id", "tenant_id"],
            ["customers.id", "customers.tenant_id"],
            name="fk_inbound_events_customer_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "tenant_id"],
            ["conversations.id", "conversations.tenant_id"],
            name="fk_inbound_events_conversation_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["message_id", "tenant_id"],
            ["messages.id", "messages.tenant_id"],
            name="fk_inbound_events_message_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            name="fk_inbound_events_lead_tenant",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "source",
            "external_event_id",
            name="uq_inbound_events_tenant_source_external",
        ),
    )
    op.create_index(
        "ix_inbound_events_tenant_status_received",
        "inbound_events",
        ["tenant_id", "status", "received_at"],
    )


def downgrade() -> None:
    op.drop_table("inbound_events")
    op.drop_constraint("fk_leads_conversation_tenant", "leads", type_="foreignkey")
    op.drop_constraint("uq_leads_id_tenant", "leads", type_="unique")
    op.drop_column("leads", "conversation_id")
    op.drop_constraint("uq_messages_id_tenant", "messages", type_="unique")
    op.drop_constraint(
        "uq_conversations_tenant_channel_external", "conversations", type_="unique"
    )
    op.drop_table("customer_identities")
