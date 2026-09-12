"""Add exact provider-reported decimal cost.

Revision ID: 20260912_0010
Revises: 20260912_0009
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260912_0010"
down_revision: str | None = "20260912_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_provider_calls_reported_cost", "provider_calls", type_="check")
    op.add_column(
        "provider_calls",
        sa.Column(
            "provider_reported_cost_decimal",
            sa.Numeric(precision=32, scale=18),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "ck_provider_calls_reported_cost",
        "provider_calls",
        "(provider_reported_cost_decimal IS NULL "
        "AND provider_reported_cost_micros IS NULL "
        "AND provider_reported_cost_currency IS NULL) OR "
        "(provider_reported_cost_decimal IS NOT NULL "
        "AND provider_reported_cost_decimal >= 0 "
        "AND provider_reported_cost_micros IS NOT NULL "
        "AND provider_reported_cost_currency IS NOT NULL "
        "AND provider_reported_cost_currency IN ('USD', 'RUB'))",
    )


def downgrade() -> None:
    op.drop_constraint("ck_provider_calls_reported_cost", "provider_calls", type_="check")
    op.drop_column("provider_calls", "provider_reported_cost_decimal")
    op.create_check_constraint(
        "ck_provider_calls_reported_cost",
        "provider_calls",
        "(provider_reported_cost_micros IS NULL "
        "AND provider_reported_cost_currency IS NULL) OR "
        "(provider_reported_cost_micros IS NOT NULL "
        "AND provider_reported_cost_currency IN ('USD', 'RUB'))",
    )
