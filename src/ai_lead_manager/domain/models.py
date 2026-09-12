from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import TypeDecorator


class _ExactDecimal(TypeDecorator[Decimal]):
    """Exact NUMERIC in PostgreSQL with a lossless SQLite test-backend variant."""

    impl = Numeric(32, 18)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String(33))
        return dialect.type_descriptor(Numeric(32, 18))

    def process_bind_param(self, value: Decimal | None, dialect):
        if value is None:
            return None
        if not isinstance(value, Decimal):
            raise TypeError("Exact decimal database values must use Decimal")
        return format(value, "f") if dialect.name == "sqlite" else value

    def process_result_value(self, value, dialect) -> Decimal | None:
        if value is None:
            return None
        return value if isinstance(value, Decimal) else Decimal(value)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Tenant(Base, TimestampMixin):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)


class TenantAPIKey(Base, TimestampMixin):
    __tablename__ = "tenant_api_keys"
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE', 'REVOKED')", name="ck_tenant_api_keys_status"),
        CheckConstraint("length(key_prefix) = 12", name="ck_tenant_api_keys_prefix_length"),
        CheckConstraint("length(key_hash) = 64", name="ck_tenant_api_keys_hash_length"),
        CheckConstraint(
            "(status = 'ACTIVE' AND revoked_at IS NULL) OR "
            "(status = 'REVOKED' AND revoked_at IS NOT NULL)",
            name="ck_tenant_api_keys_revocation_state",
        ),
        UniqueConstraint("key_prefix", name="uq_tenant_api_keys_prefix"),
        UniqueConstraint("id", "tenant_id", name="uq_tenant_api_keys_id_tenant"),
        Index("ix_tenant_api_keys_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(12), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(10), default="ACTIVE", nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Customer(Base, TimestampMixin):
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("id", "tenant_id", name="uq_customers_id_tenant"),
        UniqueConstraint("tenant_id", "email", name="uq_customers_tenant_email"),
        Index("ix_customers_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    phone: Mapped[str | None] = mapped_column(String(50))


class CustomerIdentity(Base, TimestampMixin):
    __tablename__ = "customer_identities"
    __table_args__ = (
        ForeignKeyConstraint(
            ["customer_id", "tenant_id"],
            ["customers.id", "customers.tenant_id"],
            ondelete="CASCADE",
            name="fk_customer_identities_customer_tenant",
        ),
        UniqueConstraint(
            "tenant_id",
            "identity_type",
            "normalized_value",
            name="uq_customer_identities_tenant_type_value",
        ),
        CheckConstraint(
            "identity_type IN ('email', 'phone', 'external_user_id')",
            name="ck_customer_identities_supported_type",
        ),
        Index("ix_customer_identities_tenant_customer", "tenant_id", "customer_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    identity_type: Mapped[str] = mapped_column(String(30), nullable=False)
    normalized_value: Mapped[str] = mapped_column(String(500), nullable=False)
    original_value: Mapped[str | None] = mapped_column(String(500))


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["customer_id", "tenant_id"],
            ["customers.id", "customers.tenant_id"],
            ondelete="CASCADE",
            name="fk_conversations_customer_tenant",
        ),
        UniqueConstraint("id", "tenant_id", name="uq_conversations_id_tenant"),
        UniqueConstraint(
            "tenant_id", "channel", "external_id", name="uq_conversations_tenant_channel_external"
        ),
        Index("ix_conversations_tenant_customer", "tenant_id", "customer_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    channel: Mapped[str] = mapped_column(String(50), default="mock", nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False)


class Message(Base, TimestampMixin):
    __tablename__ = "messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id", "tenant_id"],
            ["conversations.id", "conversations.tenant_id"],
            ondelete="CASCADE",
            name="fk_messages_conversation_tenant",
        ),
        UniqueConstraint("id", "tenant_id", name="uq_messages_id_tenant"),
        Index("ix_messages_tenant_conversation_created", "tenant_id", "conversation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    direction: Mapped[str] = mapped_column(String(20), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255))


class Lead(Base, TimestampMixin):
    __tablename__ = "leads"
    __table_args__ = (
        ForeignKeyConstraint(
            ["customer_id", "tenant_id"],
            ["customers.id", "customers.tenant_id"],
            ondelete="CASCADE",
            name="fk_leads_customer_tenant",
        ),
        ForeignKeyConstraint(
            ["conversation_id", "tenant_id"],
            ["conversations.id", "conversations.tenant_id"],
            name="fk_leads_conversation_tenant",
        ),
        UniqueConstraint("id", "tenant_id", name="uq_leads_id_tenant"),
        UniqueConstraint("tenant_id", "customer_id", name="uq_leads_tenant_customer"),
        CheckConstraint(
            "status IN ('NEW', 'QUALIFIED', 'CONTACTED', 'MEETING', 'WON', 'LOST')",
            name="ck_leads_lifecycle_status",
        ),
        Index("ix_leads_tenant_status_created", "tenant_id", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    conversation_id: Mapped[uuid.UUID | None]
    status: Mapped[str] = mapped_column(String(50), default="NEW", nullable=False)
    score: Mapped[int | None]
    source: Mapped[str | None] = mapped_column(String(100))
    notes: Mapped[str | None] = mapped_column(Text)


class InboundEvent(Base, TimestampMixin):
    __tablename__ = "inbound_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["customer_id", "tenant_id"],
            ["customers.id", "customers.tenant_id"],
            name="fk_inbound_events_customer_tenant",
        ),
        ForeignKeyConstraint(
            ["conversation_id", "tenant_id"],
            ["conversations.id", "conversations.tenant_id"],
            name="fk_inbound_events_conversation_tenant",
        ),
        ForeignKeyConstraint(
            ["message_id", "tenant_id"],
            ["messages.id", "messages.tenant_id"],
            name="fk_inbound_events_message_tenant",
        ),
        ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            name="fk_inbound_events_lead_tenant",
        ),
        UniqueConstraint(
            "tenant_id",
            "source",
            "external_event_id",
            name="uq_inbound_events_tenant_source_external",
        ),
        Index("ix_inbound_events_tenant_status_received", "tenant_id", "status", "received_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    external_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="received", nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    customer_id: Mapped[uuid.UUID | None]
    conversation_id: Mapped[uuid.UUID | None]
    message_id: Mapped[uuid.UUID | None]
    lead_id: Mapped[uuid.UUID | None]


class LeadAnalysis(Base):
    __tablename__ = "lead_analyses"
    __table_args__ = (
        ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            ondelete="CASCADE",
            name="fk_lead_analyses_lead_tenant",
        ),
        UniqueConstraint("id", "tenant_id", name="uq_lead_analyses_id_tenant"),
        Index("ix_lead_analyses_tenant_lead_created", "tenant_id", "lead_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    lead_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    intent: Mapped[str | None] = mapped_column(String(100))
    task: Mapped[str | None] = mapped_column(Text)
    pain_points: Mapped[list] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    budget_amount: Mapped[float | None] = mapped_column(Numeric(14, 2))
    budget_currency: Mapped[str | None] = mapped_column(String(3))
    deadline: Mapped[str | None] = mapped_column(String(100))
    objections: Mapped[list] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    urgency: Mapped[str | None] = mapped_column(String(30))
    decision_readiness: Mapped[str | None] = mapped_column(String(30))
    ai_summary: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Numeric(4, 3))
    raw_output: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ScoreRun(Base):
    __tablename__ = "score_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            ondelete="CASCADE",
            name="fk_score_runs_lead_tenant",
        ),
        ForeignKeyConstraint(
            ["analysis_id", "tenant_id"],
            ["lead_analyses.id", "lead_analyses.tenant_id"],
            ondelete="CASCADE",
            name="fk_score_runs_analysis_tenant",
        ),
        UniqueConstraint("id", "tenant_id", name="uq_score_runs_id_tenant"),
        UniqueConstraint("tenant_id", "analysis_id", name="uq_score_runs_tenant_analysis"),
        CheckConstraint("score >= 0 AND score <= 100", name="ck_score_runs_score_range"),
        CheckConstraint(
            "temperature IN ('HOT', 'WARM', 'COLD')", name="ck_score_runs_temperature"
        ),
        Index("ix_score_runs_tenant_lead_created", "tenant_id", "lead_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    lead_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    analysis_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    score: Mapped[int] = mapped_column(nullable=False)
    temperature: Mapped[str] = mapped_column(String(10), nullable=False)
    ruleset_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ScoreFactor(Base):
    __tablename__ = "score_factors"
    __table_args__ = (
        ForeignKeyConstraint(
            ["score_run_id", "tenant_id"],
            ["score_runs.id", "score_runs.tenant_id"],
            ondelete="CASCADE",
            name="fk_score_factors_run_tenant",
        ),
        Index("ix_score_factors_tenant_run", "tenant_id", "score_run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    score_run_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    factor_name: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_value: Mapped[str] = mapped_column(Text, nullable=False)
    contribution: Mapped[int] = mapped_column(nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class NextActionRecommendation(Base):
    __tablename__ = "next_action_recommendations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            ondelete="CASCADE",
            name="fk_next_actions_lead_tenant",
        ),
        ForeignKeyConstraint(
            ["score_run_id", "tenant_id"],
            ["score_runs.id", "score_runs.tenant_id"],
            ondelete="CASCADE",
            name="fk_next_actions_run_tenant",
        ),
        UniqueConstraint("id", "tenant_id", name="uq_next_actions_id_tenant"),
        CheckConstraint("priority >= 1 AND priority <= 3", name="ck_next_actions_priority"),
        Index("ix_next_actions_tenant_lead_created", "tenant_id", "lead_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    lead_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    score_run_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class LeadStatusHistory(Base):
    __tablename__ = "lead_status_history"
    __table_args__ = (
        ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            ondelete="CASCADE",
            name="fk_lead_status_history_lead_tenant",
        ),
        CheckConstraint(
            "to_status IN ('NEW', 'QUALIFIED', 'CONTACTED', 'MEETING', 'WON', 'LOST')",
            name="ck_lead_status_history_to_status",
        ),
        Index("ix_lead_status_history_tenant_lead_created", "tenant_id", "lead_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    lead_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(50))
    to_status: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    actor_type: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SalesTask(Base, TimestampMixin):
    __tablename__ = "sales_tasks"
    __table_args__ = (
        ForeignKeyConstraint(["lead_id", "tenant_id"], ["leads.id", "leads.tenant_id"], ondelete="CASCADE", name="fk_sales_tasks_lead_tenant"),
        ForeignKeyConstraint(["recommendation_id", "tenant_id"], ["next_action_recommendations.id", "next_action_recommendations.tenant_id"], name="fk_sales_tasks_recommendation_tenant"),
        UniqueConstraint("tenant_id", "recommendation_id", name="uq_sales_tasks_tenant_recommendation"),
        UniqueConstraint("id", "tenant_id", name="uq_sales_tasks_id_tenant"),
        CheckConstraint("status IN ('OPEN', 'DONE', 'CANCELLED')", name="ck_sales_tasks_status"),
        CheckConstraint("priority >= 1 AND priority <= 3", name="ck_sales_tasks_priority"),
        Index("ix_sales_tasks_tenant_lead_status", "tenant_id", "lead_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    lead_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    recommendation_id: Mapped[uuid.UUID | None]
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="OPEN", nullable=False)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ResponseDraft(Base, TimestampMixin):
    __tablename__ = "response_drafts"
    __table_args__ = (
        ForeignKeyConstraint(["lead_id", "tenant_id"], ["leads.id", "leads.tenant_id"], ondelete="CASCADE", name="fk_response_drafts_lead_tenant"),
        ForeignKeyConstraint(["analysis_id", "tenant_id"], ["lead_analyses.id", "lead_analyses.tenant_id"], ondelete="CASCADE", name="fk_response_drafts_analysis_tenant"),
        UniqueConstraint("id", "tenant_id", name="uq_response_drafts_id_tenant"),
        CheckConstraint("status IN ('APPROVAL_REQUIRED', 'APPROVED', 'REJECTED')", name="ck_response_drafts_status"),
        Index("ix_response_drafts_tenant_lead_created", "tenant_id", "lead_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    lead_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    analysis_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="APPROVAL_REQUIRED", nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rejected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_tenant_entity_created", "tenant_id", "entity_type", "entity_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    actor_type: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    event_metadata: Mapped[dict] = mapped_column("metadata", JSON().with_variant(JSONB, "postgresql"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TenantAIPolicy(Base, TimestampMixin):
    __tablename__ = "tenant_ai_policies"
    __table_args__ = (
        CheckConstraint("mode IN ('MOCK', 'LIVE')", name="ck_tenant_ai_policies_mode"),
        CheckConstraint(
            "monthly_limit_micros >= 0 AND per_call_limit_micros >= 0",
            name="ck_tenant_ai_policies_cost_limits",
        ),
        CheckConstraint(
            "budget_currency IN ('USD', 'RUB')",
            name="ck_tenant_ai_policies_budget_currency",
        ),
        CheckConstraint(
            "max_input_tokens > 0 AND max_output_tokens > 0 AND max_attempts > 0",
            name="ck_tenant_ai_policies_execution_limits",
        ),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    mode: Mapped[str] = mapped_column(String(10), default="MOCK", server_default="MOCK")
    qualification_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    qualification_model: Mapped[str] = mapped_column(String(100), nullable=False)
    draft_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    draft_model: Mapped[str] = mapped_column(String(100), nullable=False)
    monthly_limit_micros: Mapped[int] = mapped_column(BigInteger, nullable=False)
    per_call_limit_micros: Mapped[int] = mapped_column(BigInteger, nullable=False)
    budget_currency: Mapped[str] = mapped_column(
        String(3), default="USD", server_default="USD", nullable=False
    )
    max_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    allow_unknown_cost: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )


class AIOperation(Base, TimestampMixin):
    __tablename__ = "ai_operations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["lead_id", "tenant_id"],
            ["leads.id", "leads.tenant_id"],
            name="fk_ai_operations_lead_tenant",
        ),
        ForeignKeyConstraint(
            ["result_analysis_id", "tenant_id"],
            ["lead_analyses.id", "lead_analyses.tenant_id"],
            name="fk_ai_operations_analysis_tenant",
        ),
        ForeignKeyConstraint(
            ["result_draft_id", "tenant_id"],
            ["response_drafts.id", "response_drafts.tenant_id"],
            name="fk_ai_operations_draft_tenant",
        ),
        UniqueConstraint("id", "tenant_id", name="uq_ai_operations_id_tenant"),
        UniqueConstraint(
            "tenant_id",
            "operation_type",
            "idempotency_key",
            name="uq_ai_operations_tenant_type_key",
        ),
        CheckConstraint(
            "operation_type IN ('QUALIFICATION', 'DRAFT')",
            name="ck_ai_operations_type",
        ),
        CheckConstraint(
            "status IN ('RESERVED', 'CALLING', 'SUCCEEDED', 'FAILED', 'UNKNOWN')",
            name="ck_ai_operations_status",
        ),
        CheckConstraint(
            "estimated_max_cost_micros >= 0 AND reserved_cost_micros >= 0 "
            "AND (actual_cost_micros IS NULL OR actual_cost_micros >= 0)",
            name="ck_ai_operations_costs",
        ),
        CheckConstraint("cost_currency IN ('USD', 'RUB')", name="ck_ai_operations_currency"),
        Index("ix_ai_operations_tenant_created_status", "tenant_id", "created_at", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    operation_type: Mapped[str] = mapped_column(String(30), nullable=False)
    lead_id: Mapped[uuid.UUID | None]
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    pricing_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    estimated_max_cost_micros: Mapped[int] = mapped_column(BigInteger, nullable=False)
    reserved_cost_micros: Mapped[int] = mapped_column(BigInteger, nullable=False)
    actual_cost_micros: Mapped[int | None] = mapped_column(BigInteger)
    cost_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    result_analysis_id: Mapped[uuid.UUID | None]
    result_draft_id: Mapped[uuid.UUID | None]
    error_code: Mapped[str | None] = mapped_column(String(100))


class ProviderCall(Base):
    __tablename__ = "provider_calls"
    __table_args__ = (
        ForeignKeyConstraint(
            ["operation_id", "tenant_id"],
            ["ai_operations.id", "ai_operations.tenant_id"],
            ondelete="CASCADE",
            name="fk_provider_calls_operation_tenant",
        ),
        UniqueConstraint("operation_id", "attempt", name="uq_provider_calls_operation_attempt"),
        CheckConstraint(
            "status IN ('SUCCEEDED', 'FAILED', 'UNKNOWN')",
            name="ck_provider_calls_status",
        ),
        CheckConstraint("attempt > 0", name="ck_provider_calls_attempt"),
        CheckConstraint(
            "(latency_ms IS NULL OR latency_ms >= 0) "
            "AND (input_tokens IS NULL OR input_tokens >= 0) "
            "AND (output_tokens IS NULL OR output_tokens >= 0) "
            "AND (cached_tokens IS NULL OR cached_tokens >= 0) "
            "AND (actual_cost_micros IS NULL OR actual_cost_micros >= 0) "
            "AND (provider_reported_cost_micros IS NULL "
            "OR provider_reported_cost_micros >= 0)",
            name="ck_provider_calls_metrics",
        ),
        CheckConstraint("cost_currency IN ('USD', 'RUB')", name="ck_provider_calls_currency"),
        CheckConstraint(
            "(provider_reported_cost_decimal IS NULL AND "
            "provider_reported_cost_micros IS NULL AND "
            "provider_reported_cost_currency IS NULL) OR "
            "(provider_reported_cost_decimal IS NOT NULL AND "
            "provider_reported_cost_decimal >= 0 AND "
            "provider_reported_cost_micros IS NOT NULL AND "
            "provider_reported_cost_currency IS NOT NULL AND "
            "provider_reported_cost_currency IN ('USD', 'RUB'))",
            name="ck_provider_calls_reported_cost",
        ),
        Index("ix_provider_calls_tenant_operation", "tenant_id", "operation_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    operation_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_model: Mapped[str | None] = mapped_column(String(200))
    provider_request_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer)
    error_kind: Mapped[str | None] = mapped_column(String(100))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_tokens: Mapped[int | None] = mapped_column(Integer)
    actual_cost_micros: Mapped[int | None] = mapped_column(BigInteger)
    cost_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    provider_reported_cost_decimal: Mapped[Decimal | None] = mapped_column(
        _ExactDecimal()
    )
    provider_reported_cost_micros: Mapped[int | None] = mapped_column(BigInteger)
    provider_reported_cost_currency: Mapped[str | None] = mapped_column(String(3))
    cost_known: Mapped[bool] = mapped_column(Boolean, nullable=False)
    finish_reason: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
