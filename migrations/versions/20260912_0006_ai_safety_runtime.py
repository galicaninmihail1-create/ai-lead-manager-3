"""Add safe AI execution and cost controls.

Revision ID: 20260912_0006
Revises: 20260911_0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260912_0006"
down_revision: str | None = "20260911_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def timestamps() -> tuple[sa.Column, sa.Column]:
    return (
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def upgrade() -> None:
    op.create_table(
        "tenant_ai_policies",
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("mode", sa.String(10), server_default="MOCK", nullable=False),
        sa.Column("qualification_provider", sa.String(100), nullable=False),
        sa.Column("qualification_model", sa.String(100), nullable=False),
        sa.Column("draft_provider", sa.String(100), nullable=False),
        sa.Column("draft_model", sa.String(100), nullable=False),
        sa.Column("monthly_limit_microusd", sa.BigInteger(), nullable=False),
        sa.Column("per_call_limit_microusd", sa.BigInteger(), nullable=False),
        sa.Column("max_input_tokens", sa.Integer(), nullable=False),
        sa.Column("max_output_tokens", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column(
            "allow_unknown_cost", sa.Boolean(), server_default=sa.false(), nullable=False
        ),
        *timestamps(),
        sa.CheckConstraint("mode IN ('MOCK', 'LIVE')", name="ck_tenant_ai_policies_mode"),
        sa.CheckConstraint(
            "monthly_limit_microusd >= 0 AND per_call_limit_microusd >= 0",
            name="ck_tenant_ai_policies_cost_limits",
        ),
        sa.CheckConstraint(
            "max_input_tokens > 0 AND max_output_tokens > 0 AND max_attempts > 0",
            name="ck_tenant_ai_policies_execution_limits",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("tenant_id"),
    )
    op.create_table(
        "ai_operations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("operation_type", sa.String(30), nullable=False),
        sa.Column("lead_id", sa.Uuid(), nullable=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("pricing_version", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("estimated_max_cost_microusd", sa.BigInteger(), nullable=False),
        sa.Column("reserved_cost_microusd", sa.BigInteger(), nullable=False),
        sa.Column("actual_cost_microusd", sa.BigInteger(), nullable=True),
        sa.Column("result_analysis_id", sa.Uuid(), nullable=True),
        sa.Column("result_draft_id", sa.Uuid(), nullable=True),
        sa.Column("error_code", sa.String(100), nullable=True),
        *timestamps(),
        sa.CheckConstraint(
            "operation_type IN ('QUALIFICATION', 'DRAFT')", name="ck_ai_operations_type"
        ),
        sa.CheckConstraint(
            "status IN ('RESERVED', 'CALLING', 'SUCCEEDED', 'FAILED', 'UNKNOWN')",
            name="ck_ai_operations_status",
        ),
        sa.CheckConstraint(
            "estimated_max_cost_microusd >= 0 AND reserved_cost_microusd >= 0 "
            "AND (actual_cost_microusd IS NULL OR actual_cost_microusd >= 0)",
            name="ck_ai_operations_costs",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            name="fk_ai_operations_lead_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["result_analysis_id", "tenant_id"],
            ["lead_analyses.id", "lead_analyses.tenant_id"],
            name="fk_ai_operations_analysis_tenant",
        ),
        sa.ForeignKeyConstraint(
            ["result_draft_id", "tenant_id"],
            ["response_drafts.id", "response_drafts.tenant_id"],
            name="fk_ai_operations_draft_tenant",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "tenant_id", name="uq_ai_operations_id_tenant"),
        sa.UniqueConstraint(
            "tenant_id",
            "operation_type",
            "idempotency_key",
            name="uq_ai_operations_tenant_type_key",
        ),
    )
    op.create_index(
        "ix_ai_operations_tenant_created_status",
        "ai_operations",
        ["tenant_id", "created_at", "status"],
    )
    op.create_table(
        "provider_calls",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.Uuid(), nullable=False),
        sa.Column("operation_id", sa.Uuid(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("provider_request_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("error_kind", sa.String(100), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_tokens", sa.Integer(), nullable=True),
        sa.Column("actual_cost_microusd", sa.BigInteger(), nullable=True),
        sa.Column("cost_known", sa.Boolean(), nullable=False),
        sa.Column("finish_reason", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("attempt > 0", name="ck_provider_calls_attempt"),
        sa.CheckConstraint(
            "status IN ('SUCCEEDED', 'FAILED', 'UNKNOWN')",
            name="ck_provider_calls_status",
        ),
        sa.CheckConstraint(
            "(latency_ms IS NULL OR latency_ms >= 0) "
            "AND (input_tokens IS NULL OR input_tokens >= 0) "
            "AND (output_tokens IS NULL OR output_tokens >= 0) "
            "AND (cached_tokens IS NULL OR cached_tokens >= 0) "
            "AND (actual_cost_microusd IS NULL OR actual_cost_microusd >= 0)",
            name="ck_provider_calls_metrics",
        ),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["operation_id", "tenant_id"],
            ["ai_operations.id", "ai_operations.tenant_id"],
            name="fk_provider_calls_operation_tenant",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "operation_id", "attempt", name="uq_provider_calls_operation_attempt"
        ),
    )
    op.create_index(
        "ix_provider_calls_tenant_operation",
        "provider_calls",
        ["tenant_id", "operation_id"],
    )


def downgrade() -> None:
    op.drop_table("provider_calls")
    op.drop_table("ai_operations")
    op.drop_table("tenant_ai_policies")
