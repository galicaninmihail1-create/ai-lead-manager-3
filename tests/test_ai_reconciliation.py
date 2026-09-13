import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from ai_lead_manager.application.ai.money import Currency
from ai_lead_manager.application.ai.reconciliation import (
    AIReconciliationError,
    AIReconciliationErrorCode,
    AIReconciliationEvidence,
    AIReconciliationService,
)
from ai_lead_manager.domain.models import (
    AIOperation,
    AuditEvent,
    Base,
    Conversation,
    Customer,
    Lead,
    LeadAnalysis,
    NextActionRecommendation,
    ProviderCall,
    ScoreFactor,
    ScoreRun,
    Tenant,
    TenantAIPolicy,
)
from ai_lead_manager.infrastructure.ai.repositories import SqlAlchemyAIRepositoryFactory

RECONCILED_AT = datetime(2026, 9, 13, 14, 0, tzinfo=UTC)


@pytest.fixture
def session_factory(tmp_path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(f"sqlite:///{tmp_path / 'reconciliation.db'}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    yield factory
    engine.dispose()


def evidence(**updates) -> AIReconciliationEvidence:
    values = {
        "source": "polza_history",
        "external_generation_id": "gen_test_reconciliation",
        "external_status": "failed",
        "confirmed_cost_micros": 0,
        "confirmed_cost_currency": Currency.RUB,
        "usage_available": False,
        "external_attempts_count": 4,
        "external_created_at": datetime(2026, 9, 13, 13, 12, 57, tzinfo=UTC),
        "external_completed_at": datetime(2026, 9, 13, 13, 13, 4, tzinfo=UTC),
    }
    values.update(updates)
    return AIReconciliationEvidence(**values)


def seed_unknown(factory, *, call_count: int = 1):
    tenant_id, customer_id, conversation_id, lead_id, operation_id = (
        uuid.uuid4() for _ in range(5)
    )
    with factory.begin() as session:
        session.add(Tenant(id=tenant_id, name="Reconciliation", slug=f"r-{tenant_id.hex}"))
        session.add(
            TenantAIPolicy(
                tenant_id=tenant_id,
                enabled=True,
                mode="LIVE",
                qualification_provider="polza",
                qualification_model="lead-structured-v1",
                draft_provider="polza",
                draft_model="lead-structured-v1",
                monthly_limit_micros=10_000_000,
                per_call_limit_micros=1_000_000,
                budget_currency="RUB",
                max_input_tokens=1_000,
                max_output_tokens=100,
                max_attempts=1,
                allow_unknown_cost=False,
            )
        )
        session.flush()
        session.add(Customer(id=customer_id, tenant_id=tenant_id, name="Customer"))
        session.flush()
        session.add(
            Conversation(
                id=conversation_id,
                tenant_id=tenant_id,
                customer_id=customer_id,
                channel="test",
            )
        )
        session.flush()
        session.add(
            Lead(
                id=lead_id,
                tenant_id=tenant_id,
                customer_id=customer_id,
                conversation_id=conversation_id,
            )
        )
        session.flush()
        session.add(
            AIOperation(
                id=operation_id,
                tenant_id=tenant_id,
                lead_id=lead_id,
                operation_type="QUALIFICATION",
                idempotency_key="reconcile-key",
                request_hash="a" * 64,
                provider="polza",
                model="lead-structured-v1",
                pricing_version="polza-price-v1",
                status="UNKNOWN",
                estimated_max_cost_micros=20_484,
                reserved_cost_micros=20_484,
                actual_cost_micros=0,
                cost_currency="RUB",
                error_code="provider_5xx",
            )
        )
        session.flush()
        for attempt in range(1, call_count + 1):
            session.add(
                ProviderCall(
                    tenant_id=tenant_id,
                    operation_id=operation_id,
                    attempt=attempt,
                    provider="polza",
                    model="lead-structured-v1",
                    status="UNKNOWN",
                    http_status=502,
                    error_kind="provider_5xx",
                    latency_ms=7_849,
                    actual_cost_micros=None,
                    cost_currency="RUB",
                    cost_known=False,
                )
            )
    return tenant_id, lead_id, operation_id


def service(factory) -> AIReconciliationService:
    return AIReconciliationService(
        SqlAlchemyAIRepositoryFactory(factory), clock=lambda: RECONCILED_AT
    )


def reconcile(factory, tenant_id, lead_id, operation_id):
    return service(factory).reconcile_failed_qualification(
        tenant_id, operation_id, lead_id, "reconcile-key", evidence()
    )


def test_unknown_failed_zero_cost_reconciliation_is_atomic_and_auditable(
    session_factory,
) -> None:
    tenant_id, lead_id, operation_id = seed_unknown(session_factory)
    with session_factory() as session:
        before = session.scalar(
            select(func.sum(AIOperation.actual_cost_micros + AIOperation.reserved_cost_micros))
        )
        assert before == 20_484

    result = reconcile(session_factory, tenant_id, lead_id, operation_id)
    assert result.status == "FAILED"
    assert result.confirmed_cost_micros == 0

    with session_factory() as session:
        operation = session.get(AIOperation, operation_id)
        calls = list(session.scalars(select(ProviderCall).where(ProviderCall.operation_id == operation_id)))
        assert operation.status == "FAILED"
        assert operation.reserved_cost_micros == 0
        assert operation.actual_cost_micros == 0
        assert operation.error_code == "provider_5xx"
        assert len(calls) == 1
        call = calls[0]
        assert call.status == "FAILED"
        assert call.attempt == 1
        assert call.cost_known is True
        assert call.actual_cost_micros == 0
        assert call.provider_reported_cost_decimal == Decimal("0")
        assert call.provider_reported_cost_micros == 0
        assert call.provider_reported_cost_currency == "RUB"
        assert (call.input_tokens, call.output_tokens, call.cached_tokens) == (None, None, None)
        assert call.http_status == 502
        assert call.error_kind == "provider_5xx"
        assert session.scalar(select(func.count()).select_from(ProviderCall)) == 1
        assert session.scalar(select(func.count()).select_from(LeadAnalysis)) == 0
        assert session.scalar(select(func.count()).select_from(ScoreRun)) == 0
        assert session.scalar(select(func.count()).select_from(ScoreFactor)) == 0
        assert session.scalar(select(func.count()).select_from(NextActionRecommendation)) == 0
        spend = session.scalar(
            select(func.sum(AIOperation.actual_cost_micros + AIOperation.reserved_cost_micros))
        )
        assert spend == 0
        audit = session.scalar(
            select(AuditEvent).where(AuditEvent.action == "ai.operation_reconciled")
        )
        assert audit.event_metadata == {
            "operation_id": str(operation_id),
            "operation_type": "QUALIFICATION",
            "reconciliation_source": "polza_history",
            "external_generation_id": "gen_test_reconciliation",
            "external_status": "failed",
            "confirmed_cost": {"amount_micros": 0, "currency": "RUB"},
            "usage_available": False,
            "external_attempts_count": 4,
            "external_created_at": "2026-09-13T13:12:57+00:00",
            "external_completed_at": "2026-09-13T13:13:04+00:00",
            "previous_operation_status": "UNKNOWN",
            "previous_provider_call_status": "UNKNOWN",
            "reconciled_at": "2026-09-13T14:00:00+00:00",
        }


def test_second_reconciliation_is_rejected_without_mutation(session_factory) -> None:
    tenant_id, lead_id, operation_id = seed_unknown(session_factory)
    reconcile(session_factory, tenant_id, lead_id, operation_id)
    with pytest.raises(AIReconciliationError) as captured:
        reconcile(session_factory, tenant_id, lead_id, operation_id)
    assert captured.value.code == AIReconciliationErrorCode.ALREADY_RECONCILED
    with session_factory() as session:
        operation = session.get(AIOperation, operation_id)
        assert (operation.status, operation.reserved_cost_micros, operation.actual_cost_micros) == (
            "FAILED",
            0,
            0,
        )
        assert session.scalar(
            select(func.count()).select_from(AuditEvent).where(
                AuditEvent.action == "ai.operation_reconciled"
            )
        ) == 1


@pytest.mark.parametrize("mismatch", ["lead", "key", "provider", "model", "operation_status", "currency", "call_status"])
def test_mismatched_state_rolls_back(session_factory, mismatch: str) -> None:
    tenant_id, lead_id, operation_id = seed_unknown(session_factory)
    expected_lead = lead_id
    expected_key = "reconcile-key"
    with session_factory.begin() as session:
        operation = session.get(AIOperation, operation_id)
        call = session.scalar(select(ProviderCall).where(ProviderCall.operation_id == operation_id))
        if mismatch == "lead":
            expected_lead = uuid.uuid4()
        elif mismatch == "key":
            expected_key = "wrong-key"
        elif mismatch == "provider":
            operation.provider = "other"
        elif mismatch == "model":
            operation.model = "other"
        elif mismatch == "operation_status":
            operation.status = "CALLING"
        elif mismatch == "currency":
            operation.cost_currency = "USD"
            call.cost_currency = "USD"
        elif mismatch == "call_status":
            call.status = "FAILED"

    with pytest.raises(AIReconciliationError):
        service(session_factory).reconcile_failed_qualification(
            tenant_id, operation_id, expected_lead, expected_key, evidence()
        )
    with session_factory() as session:
        operation = session.get(AIOperation, operation_id)
        assert operation.reserved_cost_micros == 20_484
        assert session.scalar(
            select(func.count()).select_from(AuditEvent).where(
                AuditEvent.action == "ai.operation_reconciled"
            )
        ) == 0


@pytest.mark.parametrize("call_count", [0, 2])
def test_requires_exactly_one_provider_call(session_factory, call_count: int) -> None:
    tenant_id, lead_id, operation_id = seed_unknown(session_factory, call_count=call_count)
    with pytest.raises(AIReconciliationError) as captured:
        reconcile(session_factory, tenant_id, lead_id, operation_id)
    assert captured.value.code == AIReconciliationErrorCode.INVALID_STATE


def test_wrong_tenant_is_not_found(session_factory) -> None:
    tenant_id, lead_id, operation_id = seed_unknown(session_factory)
    other_tenant, _, _ = seed_unknown(session_factory)
    with pytest.raises(AIReconciliationError) as captured:
        service(session_factory).reconcile_failed_qualification(
            other_tenant, operation_id, lead_id, "reconcile-key", evidence()
        )
    assert captured.value.code == AIReconciliationErrorCode.NOT_FOUND
    with session_factory() as session:
        assert session.get(AIOperation, operation_id).status == "UNKNOWN"


@pytest.mark.parametrize(
    "updates",
    [
        {"source": "unsupported"},
        {"external_status": "completed"},
        {"confirmed_cost_micros": -1},
        {"confirmed_cost_currency": Currency.USD},
        {"usage_available": True},
        {"external_attempts_count": 0},
        {
            "external_created_at": datetime(2026, 9, 13, 13, 13, 5, tzinfo=UTC),
            "external_completed_at": datetime(2026, 9, 13, 13, 13, 4, tzinfo=UTC),
        },
    ],
)
def test_unsupported_evidence_is_rejected(updates) -> None:
    with pytest.raises(ValidationError):
        evidence(**updates)
