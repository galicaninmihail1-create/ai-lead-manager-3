import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.providers import MockQualificationProvider
from ai_lead_manager.application.qualification import QualificationService
from ai_lead_manager.application.schemas import InboundHttpRequest, TenantCreate
from ai_lead_manager.application.services import TenantService
from ai_lead_manager.domain.exceptions import EntityNotFoundError, QualificationUnavailableError
from ai_lead_manager.domain.models import (
    Conversation,
    Customer,
    Lead,
    LeadAnalysis,
    NextActionRecommendation,
    ScoreFactor,
    ScoreRun,
)
from ai_lead_manager.infrastructure.database import get_session
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyQualificationRepository,
    SqlAlchemyTenantRepository,
    SqlAlchemyUnitOfWork,
)


def create_tenant(session: Session, slug: str) -> uuid.UUID:
    return TenantService(SqlAlchemyTenantRepository(session), SqlAlchemyUnitOfWork(session)).create(
        TenantCreate(name=slug, slug=slug)
    ).id


def create_lead(session: Session, tenant_id: uuid.UUID) -> uuid.UUID:
    result = InboundIngestionService(
        SqlAlchemyInboundRepository(session, tenant_id), SqlAlchemyUnitOfWork(session)
    ).ingest(
        InboundHttpRequest(
            tenant_id=tenant_id,
            source="web-form",
            external_event_id="qualification-event",
            external_thread_id="qualification-thread",
            name="Анна",
            email="anna@example.com",
            phone="+7 999 123-45-67",
            message=(
                "Нужно срочно автоматизировать входящие заявки, мы теряем клиентов. "
                "Бюджет 300 000 рублей, хотим внедрить за 2 недели."
            ),
        )
    )
    return result.lead_id


def qualification_service(session: Session, tenant_id: uuid.UUID) -> QualificationService:
    return QualificationService(
        tenant_id,
        SqlAlchemyQualificationRepository(session, tenant_id),
        SqlAlchemyUnitOfWork(session),
        MockQualificationProvider(),
    )


def test_successful_qualification_persists_explainable_history(session: Session) -> None:
    tenant_id = create_tenant(session, "qualification-tenant")
    lead_id = create_lead(session, tenant_id)

    result = qualification_service(session, tenant_id).qualify(lead_id)

    assert result.score == 100
    assert result.temperature == "HOT"
    assert result.analysis.budget_amount == 300_000
    assert result.analysis.intent == "purchase"
    assert sum(factor.contribution for factor in result.score_factors) == result.score
    assert result.next_best_action.action_type == "schedule_call"
    assert session.scalar(select(func.count()).select_from(LeadAnalysis)) == 1
    assert session.scalar(select(func.count()).select_from(ScoreRun)) == 1
    assert session.scalar(select(func.count()).select_from(ScoreFactor)) == 7
    assert session.scalar(select(func.count()).select_from(NextActionRecommendation)) == 1
    assert session.get(Lead, lead_id).score == 100


def test_requalification_preserves_history(session: Session) -> None:
    tenant_id = create_tenant(session, "history-tenant")
    lead_id = create_lead(session, tenant_id)
    first = qualification_service(session, tenant_id).qualify(lead_id)
    second = qualification_service(session, tenant_id).qualify(lead_id)

    assert first.analysis_id != second.analysis_id
    assert first.score_run_id != second.score_run_id
    assert session.scalar(select(func.count()).select_from(LeadAnalysis)) == 2
    assert session.scalar(select(func.count()).select_from(ScoreRun)) == 2
    assert session.scalar(select(func.count()).select_from(ScoreFactor)) == 14
    assert session.scalar(select(func.count()).select_from(NextActionRecommendation)) == 2


def test_foreign_tenant_cannot_qualify_lead(session: Session) -> None:
    tenant_a = create_tenant(session, "qualification-a")
    tenant_b = create_tenant(session, "qualification-b")
    lead_id = create_lead(session, tenant_b)

    with pytest.raises(EntityNotFoundError):
        qualification_service(session, tenant_a).qualify(lead_id)


def test_missing_inbound_message_is_rejected_without_history(session: Session) -> None:
    tenant_id = create_tenant(session, "no-message-tenant")
    customer = Customer(tenant_id=tenant_id, name="No messages")
    session.add(customer)
    session.flush()
    conversation = Conversation(
        tenant_id=tenant_id, customer_id=customer.id, channel="manual", status="open"
    )
    session.add(conversation)
    session.flush()
    lead = Lead(
        tenant_id=tenant_id,
        customer_id=customer.id,
        conversation_id=conversation.id,
        status="NEW",
    )
    session.add(lead)
    session.commit()

    with pytest.raises(QualificationUnavailableError):
        qualification_service(session, tenant_id).qualify(lead.id)
    assert session.scalar(select(func.count()).select_from(LeadAnalysis)) == 0


def test_qualification_api(session: Session) -> None:
    from ai_lead_manager.main import app

    tenant_id = create_tenant(session, "qualification-api")
    lead_id = create_lead(session, tenant_id)
    app.dependency_overrides[get_session] = lambda: session
    try:
        response = TestClient(app).post(
            f"/api/v1/leads/{lead_id}/qualify", json={"tenant_id": str(tenant_id)}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["lead_id"] == str(lead_id)
    assert response.json()["score"] == 100
