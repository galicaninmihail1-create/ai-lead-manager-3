"""Add sales workflow, approvals, and audit history.

Revision ID: 20260911_0005
Revises: 20260911_0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260911_0005"
down_revision: str | None = "20260911_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def created_at() -> sa.Column:
    return sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


def upgrade() -> None:
    op.execute("UPDATE leads SET status = upper(status)")
    op.create_check_constraint(
        "ck_leads_lifecycle_status",
        "leads",
        "status IN ('NEW', 'QUALIFIED', 'CONTACTED', 'MEETING', 'WON', 'LOST')",
    )
    op.create_unique_constraint(
        "uq_next_actions_id_tenant", "next_action_recommendations", ["id", "tenant_id"]
    )
    op.create_table(
        "lead_status_history",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("lead_id", sa.Uuid(), nullable=False),
        sa.Column("from_status", sa.String(50), nullable=True),
        sa.Column("to_status", sa.String(50), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("actor_type", sa.String(50), nullable=False),
        created_at(),
        sa.CheckConstraint(
            "to_status IN ('NEW', 'QUALIFIED', 'CONTACTED', 'MEETING', 'WON', 'LOST')",
            name="ck_lead_status_history_to_status",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["lead_id", "tenant_id"], ["leads.id", "leads.tenant_id"], name="fk_lead_status_history_lead_tenant", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_lead_status_history_tenant_lead_created", "lead_status_history", ["tenant_id", "lead_id", "created_at"])
    op.create_table(
        "sales_tasks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("lead_id", sa.Uuid(), nullable=False),
        sa.Column("recommendation_id", sa.Uuid(), nullable=True),
        sa.Column("task_type", sa.String(100), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        created_at(),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("status IN ('OPEN', 'DONE', 'CANCELLED')", name="ck_sales_tasks_status"),
        sa.CheckConstraint("priority >= 1 AND priority <= 3", name="ck_sales_tasks_priority"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["lead_id", "tenant_id"], ["leads.id", "leads.tenant_id"], name="fk_sales_tasks_lead_tenant", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["recommendation_id", "tenant_id"], ["next_action_recommendations.id", "next_action_recommendations.tenant_id"], name="fk_sales_tasks_recommendation_tenant"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_sales_tasks_id_tenant"),
        sa.UniqueConstraint("tenant_id", "recommendation_id", name="uq_sales_tasks_tenant_recommendation"),
    )
    op.create_index("ix_sales_tasks_tenant_lead_status", "sales_tasks", ["tenant_id", "lead_id", "status"])
    op.create_table(
        "response_drafts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("lead_id", sa.Uuid(), nullable=False),
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=True),
        created_at(),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("status IN ('APPROVAL_REQUIRED', 'APPROVED', 'REJECTED')", name="ck_response_drafts_status"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["lead_id", "tenant_id"], ["leads.id", "leads.tenant_id"], name="fk_response_drafts_lead_tenant", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["analysis_id", "tenant_id"], ["lead_analyses.id", "lead_analyses.tenant_id"], name="fk_response_drafts_analysis_tenant", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_response_drafts_id_tenant"),
    )
    op.create_index("ix_response_drafts_tenant_lead_created", "response_drafts", ["tenant_id", "lead_id", "created_at"])
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("actor_type", sa.String(50), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(100), nullable=False),
        sa.Column("entity_id", sa.Uuid(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        created_at(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_tenant_entity_created", "audit_events", ["tenant_id", "entity_type", "entity_id", "created_at"])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("response_drafts")
    op.drop_table("sales_tasks")
    op.drop_table("lead_status_history")
    op.drop_constraint("uq_next_actions_id_tenant", "next_action_recommendations", type_="unique")
    op.drop_constraint("ck_leads_lifecycle_status", "leads", type_="check")
    op.execute("UPDATE leads SET status = lower(status)")
