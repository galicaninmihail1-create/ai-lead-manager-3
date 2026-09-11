"""Add explainable lead qualification history.

Revision ID: 20260911_0004
Revises: 20260911_0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260911_0004"
down_revision: str | None = "20260911_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "lead_analyses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("lead_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("schema_version", sa.String(length=50), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("intent", sa.String(length=100), nullable=True),
        sa.Column("task", sa.Text(), nullable=True),
        sa.Column("pain_points", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("budget_amount", sa.Numeric(precision=14, scale=2), nullable=True),
        sa.Column("budget_currency", sa.String(length=3), nullable=True),
        sa.Column("deadline", sa.String(length=100), nullable=True),
        sa.Column("objections", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("urgency", sa.String(length=30), nullable=True),
        sa.Column("decision_readiness", sa.String(length=30), nullable=True),
        sa.Column("ai_summary", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("raw_output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        created_at(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            name="fk_lead_analyses_lead_tenant",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_lead_analyses_id_tenant"),
    )
    op.create_index(
        "ix_lead_analyses_tenant_lead_created",
        "lead_analyses",
        ["tenant_id", "lead_id", "created_at"],
    )
    op.create_table(
        "score_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("lead_id", sa.Uuid(), nullable=False),
        sa.Column("analysis_id", sa.Uuid(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.String(length=10), nullable=False),
        sa.Column("ruleset_version", sa.String(length=50), nullable=False),
        created_at(),
        sa.CheckConstraint("score >= 0 AND score <= 100", name="ck_score_runs_score_range"),
        sa.CheckConstraint(
            "temperature IN ('HOT', 'WARM', 'COLD')", name="ck_score_runs_temperature"
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            name="fk_score_runs_lead_tenant",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["analysis_id", "tenant_id"],
            ["lead_analyses.id", "lead_analyses.tenant_id"],
            name="fk_score_runs_analysis_tenant",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_score_runs_id_tenant"),
        sa.UniqueConstraint("tenant_id", "analysis_id", name="uq_score_runs_tenant_analysis"),
    )
    op.create_index(
        "ix_score_runs_tenant_lead_created",
        "score_runs",
        ["tenant_id", "lead_id", "created_at"],
    )
    op.create_table(
        "score_factors",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("score_run_id", sa.Uuid(), nullable=False),
        sa.Column("factor_name", sa.String(length=100), nullable=False),
        sa.Column("raw_value", sa.Text(), nullable=False),
        sa.Column("contribution", sa.Integer(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        created_at(),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["score_run_id", "tenant_id"],
            ["score_runs.id", "score_runs.tenant_id"],
            name="fk_score_factors_run_tenant",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_score_factors_tenant_run", "score_factors", ["tenant_id", "score_run_id"]
    )
    op.create_table(
        "next_action_recommendations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("lead_id", sa.Uuid(), nullable=False),
        sa.Column("score_run_id", sa.Uuid(), nullable=False),
        sa.Column("action_type", sa.String(length=100), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        created_at(),
        sa.CheckConstraint("priority >= 1 AND priority <= 3", name="ck_next_actions_priority"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            name="fk_next_actions_lead_tenant",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["score_run_id", "tenant_id"],
            ["score_runs.id", "score_runs.tenant_id"],
            name="fk_next_actions_run_tenant",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_next_actions_tenant_lead_created",
        "next_action_recommendations",
        ["tenant_id", "lead_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("next_action_recommendations")
    op.drop_table("score_factors")
    op.drop_table("score_runs")
    op.drop_table("lead_analyses")
