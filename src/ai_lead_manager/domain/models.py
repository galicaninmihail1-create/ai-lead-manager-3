from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


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
        Index("ix_leads_tenant_status_created", "tenant_id", "status", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    conversation_id: Mapped[uuid.UUID | None]
    status: Mapped[str] = mapped_column(String(50), default="new", nullable=False)
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
