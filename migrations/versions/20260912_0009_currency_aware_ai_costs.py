"""Make AI cost accounting currency-aware.

Revision ID: 20260912_0009
Revises: 20260912_0008
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260912_0009"
down_revision: str | None = "20260912_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_tenant_ai_policies_cost_limits", "tenant_ai_policies", type_="check"
    )
    op.alter_column(
        "tenant_ai_policies",
        "monthly_limit_microusd",
        new_column_name="monthly_limit_micros",
    )
    op.alter_column(
        "tenant_ai_policies",
        "per_call_limit_microusd",
        new_column_name="per_call_limit_micros",
    )
    op.add_column(
        "tenant_ai_policies",
        sa.Column(
            "budget_currency",
            sa.String(length=3),
            nullable=False,
            server_default="USD",
        ),
    )
    op.create_check_constraint(
        "ck_tenant_ai_policies_cost_limits",
        "tenant_ai_policies",
        "monthly_limit_micros >= 0 AND per_call_limit_micros >= 0",
    )
    op.create_check_constraint(
        "ck_tenant_ai_policies_budget_currency",
        "tenant_ai_policies",
        "budget_currency IN ('USD', 'RUB')",
    )

    op.drop_constraint("ck_ai_operations_costs", "ai_operations", type_="check")
    op.alter_column(
        "ai_operations",
        "estimated_max_cost_microusd",
        new_column_name="estimated_max_cost_micros",
    )
    op.alter_column(
        "ai_operations", "reserved_cost_microusd", new_column_name="reserved_cost_micros"
    )
    op.alter_column(
        "ai_operations", "actual_cost_microusd", new_column_name="actual_cost_micros"
    )
    op.add_column(
        "ai_operations",
        sa.Column("cost_currency", sa.String(length=3), nullable=True),
    )
    op.execute("UPDATE ai_operations SET cost_currency = 'USD'")
    op.alter_column("ai_operations", "cost_currency", nullable=False)
    op.create_check_constraint(
        "ck_ai_operations_costs",
        "ai_operations",
        "estimated_max_cost_micros >= 0 AND reserved_cost_micros >= 0 "
        "AND (actual_cost_micros IS NULL OR actual_cost_micros >= 0)",
    )
    op.create_check_constraint(
        "ck_ai_operations_currency",
        "ai_operations",
        "cost_currency IN ('USD', 'RUB')",
    )

    op.drop_constraint("ck_provider_calls_metrics", "provider_calls", type_="check")
    op.alter_column(
        "provider_calls", "actual_cost_microusd", new_column_name="actual_cost_micros"
    )
    op.add_column(
        "provider_calls",
        sa.Column("cost_currency", sa.String(length=3), nullable=True),
    )
    op.add_column(
        "provider_calls",
        sa.Column("provider_reported_cost_micros", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "provider_calls",
        sa.Column("provider_reported_cost_currency", sa.String(length=3), nullable=True),
    )
    op.execute("UPDATE provider_calls SET cost_currency = 'USD'")
    op.alter_column("provider_calls", "cost_currency", nullable=False)
    op.create_check_constraint(
        "ck_provider_calls_metrics",
        "provider_calls",
        "(latency_ms IS NULL OR latency_ms >= 0) "
        "AND (input_tokens IS NULL OR input_tokens >= 0) "
        "AND (output_tokens IS NULL OR output_tokens >= 0) "
        "AND (cached_tokens IS NULL OR cached_tokens >= 0) "
        "AND (actual_cost_micros IS NULL OR actual_cost_micros >= 0) "
        "AND (provider_reported_cost_micros IS NULL "
        "OR provider_reported_cost_micros >= 0)",
    )
    op.create_check_constraint(
        "ck_provider_calls_currency",
        "provider_calls",
        "cost_currency IN ('USD', 'RUB')",
    )
    op.create_check_constraint(
        "ck_provider_calls_reported_cost",
        "provider_calls",
        "(provider_reported_cost_micros IS NULL "
        "AND provider_reported_cost_currency IS NULL) OR "
        "(provider_reported_cost_micros IS NOT NULL "
        "AND provider_reported_cost_currency IN ('USD', 'RUB'))",
    )


def downgrade() -> None:
    op.drop_constraint("ck_provider_calls_reported_cost", "provider_calls", type_="check")
    op.drop_constraint("ck_provider_calls_currency", "provider_calls", type_="check")
    op.drop_constraint("ck_provider_calls_metrics", "provider_calls", type_="check")
    op.drop_column("provider_calls", "provider_reported_cost_currency")
    op.drop_column("provider_calls", "provider_reported_cost_micros")
    op.drop_column("provider_calls", "cost_currency")
    op.alter_column(
        "provider_calls", "actual_cost_micros", new_column_name="actual_cost_microusd"
    )
    op.create_check_constraint(
        "ck_provider_calls_metrics",
        "provider_calls",
        "(latency_ms IS NULL OR latency_ms >= 0) "
        "AND (input_tokens IS NULL OR input_tokens >= 0) "
        "AND (output_tokens IS NULL OR output_tokens >= 0) "
        "AND (cached_tokens IS NULL OR cached_tokens >= 0) "
        "AND (actual_cost_microusd IS NULL OR actual_cost_microusd >= 0)",
    )

    op.drop_constraint("ck_ai_operations_currency", "ai_operations", type_="check")
    op.drop_constraint("ck_ai_operations_costs", "ai_operations", type_="check")
    op.drop_column("ai_operations", "cost_currency")
    op.alter_column(
        "ai_operations", "actual_cost_micros", new_column_name="actual_cost_microusd"
    )
    op.alter_column(
        "ai_operations", "reserved_cost_micros", new_column_name="reserved_cost_microusd"
    )
    op.alter_column(
        "ai_operations",
        "estimated_max_cost_micros",
        new_column_name="estimated_max_cost_microusd",
    )
    op.create_check_constraint(
        "ck_ai_operations_costs",
        "ai_operations",
        "estimated_max_cost_microusd >= 0 AND reserved_cost_microusd >= 0 "
        "AND (actual_cost_microusd IS NULL OR actual_cost_microusd >= 0)",
    )

    op.drop_constraint(
        "ck_tenant_ai_policies_budget_currency", "tenant_ai_policies", type_="check"
    )
    op.drop_constraint(
        "ck_tenant_ai_policies_cost_limits", "tenant_ai_policies", type_="check"
    )
    op.drop_column("tenant_ai_policies", "budget_currency")
    op.alter_column(
        "tenant_ai_policies",
        "per_call_limit_micros",
        new_column_name="per_call_limit_microusd",
    )
    op.alter_column(
        "tenant_ai_policies",
        "monthly_limit_micros",
        new_column_name="monthly_limit_microusd",
    )
    op.create_check_constraint(
        "ck_tenant_ai_policies_cost_limits",
        "tenant_ai_policies",
        "monthly_limit_microusd >= 0 AND per_call_limit_microusd >= 0",
    )
