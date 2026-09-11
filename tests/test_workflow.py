import socket
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_lead_manager.application.drafts import DraftContext, MockDraftProvider
from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.providers import MockQualificationProvider
from ai_lead_manager.application.qualification import QualificationService
from ai_lead_manager.application.schemas import InboundHttpRequest, TenantCreate
from ai_lead_manager.application.services import TenantService
from ai_lead_manager.application.workflow import SalesWorkflowService
from ai_lead_manager.domain.exceptions import (
    EntityNotFoundError,
    InvalidWorkflowTransitionError,
    TenantScopeError,
    WorkflowPrerequisiteError,
)
from ai_lead_manager.domain.models import (
    AuditEvent,
    Lead,
    LeadStatusHistory,
    ResponseDraft,
    SalesTask,
)
from ai_lead_manager.infrastructure.database import get_session
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyQualificationRepository,
    SqlAlchemyTenantRepository,
    SqlAlchemyUnitOfWork,
    SqlAlchemyWorkflowRepository,
)


def tenant(session: Session, slug: str) -> uuid.UUID:
    return TenantService(SqlAlchemyTenantRepository(session), SqlAlchemyUnitOfWork(session)).create(
        TenantCreate(name=slug, slug=slug)
    ).id


def qualified_lead(session: Session, tenant_id: uuid.UUID, event: str = "workflow-event") -> uuid.UUID:
    inbound = InboundIngestionService(
        SqlAlchemyInboundRepository(session, tenant_id), SqlAlchemyUnitOfWork(session)
    ).ingest(
        InboundHttpRequest(
            tenant_id=tenant_id,
            source="web-form",
            external_event_id=event,
            name="Анна",
            email="anna@example.com",
            phone="+7 999 123-45-67",
            message="Нужно срочно автоматизировать заявки. Бюджет 300 000 рублей, срок за 2 недели.",
        )
    )
    QualificationService(
        tenant_id,
        SqlAlchemyQualificationRepository(session, tenant_id),
        SqlAlchemyUnitOfWork(session),
        MockQualificationProvider(),
    ).qualify(inbound.lead_id)
    return inbound.lead_id


def workflow(session: Session, tenant_id: uuid.UUID) -> SalesWorkflowService:
    return SalesWorkflowService(
        tenant_id,
        SqlAlchemyWorkflowRepository(session, tenant_id),
        SqlAlchemyUnitOfWork(session),
        MockDraftProvider(),
    )


def count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_valid_lifecycle_and_history_preservation(session: Session) -> None:
    tenant_id = tenant(session, "lifecycle")
    lead_id = qualified_lead(session, tenant_id)
    service = workflow(session, tenant_id)

    for status in ("QUALIFIED", "CONTACTED", "MEETING", "WON"):
        result = service.transition_lead(lead_id, status, f"Move to {status}", "manager")
        assert result.changed is True
        assert result.status == status

    histories = list(
        session.scalars(
            select(LeadStatusHistory).where(LeadStatusHistory.lead_id == lead_id).order_by(LeadStatusHistory.created_at)
        )
    )
    assert [(item.from_status, item.to_status) for item in histories] == [
        ("NEW", "QUALIFIED"),
        ("QUALIFIED", "CONTACTED"),
        ("CONTACTED", "MEETING"),
        ("MEETING", "WON"),
    ]


def test_valid_lifecycle_can_end_as_lost(session: Session) -> None:
    tenant_id = tenant(session, "lifecycle-lost")
    lead_id = qualified_lead(session, tenant_id, "lifecycle-lost-event")
    service = workflow(session, tenant_id)

    for status in ("QUALIFIED", "CONTACTED", "MEETING", "LOST"):
        service.transition_lead(lead_id, status, None, "manager")

    assert session.get(Lead, lead_id).status == "LOST"


def test_invalid_transition_rolls_back_without_history(session: Session) -> None:
    tenant_id = tenant(session, "invalid-transition")
    lead_id = qualified_lead(session, tenant_id)
    audit_before = count(session, AuditEvent)

    with pytest.raises(InvalidWorkflowTransitionError):
        workflow(session, tenant_id).transition_lead(lead_id, "MEETING", None, "manager")

    assert session.get(Lead, lead_id).status == "NEW"
    assert count(session, LeadStatusHistory) == 0
    assert count(session, AuditEvent) == audit_before


def test_task_creation_is_idempotent_and_completion_rules(session: Session) -> None:
    tenant_id = tenant(session, "tasks")
    lead_id = qualified_lead(session, tenant_id)
    service = workflow(session, tenant_id)
    first = service.create_task_from_recommendation(lead_id, "manager")
    replay = service.create_task_from_recommendation(lead_id, "manager")

    assert first.created is True
    assert replay.created is False
    assert replay.task_id == first.task_id
    assert count(session, SalesTask) == 1
    done = service.complete_task(first.task_id, "manager")
    assert done.status == "DONE"
    assert service.complete_task(first.task_id, "manager").status == "DONE"
    with pytest.raises(InvalidWorkflowTransitionError):
        service.cancel_task(first.task_id, "manager")


def test_cancelled_task_cannot_be_completed(session: Session) -> None:
    tenant_id = tenant(session, "cancel-task")
    lead_id = qualified_lead(session, tenant_id)
    service = workflow(session, tenant_id)
    task = service.create_task_from_recommendation(lead_id, "manager")
    assert service.cancel_task(task.task_id, "manager").status == "CANCELLED"
    assert service.cancel_task(task.task_id, "manager").status == "CANCELLED"
    with pytest.raises(InvalidWorkflowTransitionError):
        service.complete_task(task.task_id, "manager")


def test_draft_generation_approval_and_rejection_rules(session: Session) -> None:
    tenant_id = tenant(session, "drafts")
    lead_id = qualified_lead(session, tenant_id)
    service = workflow(session, tenant_id)
    approved = service.generate_draft(lead_id, "manager")
    assert approved.status == "APPROVAL_REQUIRED"
    assert approved.content.startswith("Здравствуйте, Анна!")
    assert service.approve_draft(approved.draft_id, "manager").status == "APPROVED"
    assert service.approve_draft(approved.draft_id, "manager").status == "APPROVED"
    with pytest.raises(InvalidWorkflowTransitionError):
        service.reject_draft(approved.draft_id, "manager")

    rejected = service.generate_draft(lead_id, "manager")
    assert service.reject_draft(rejected.draft_id, "manager").status == "REJECTED"
    with pytest.raises(InvalidWorkflowTransitionError):
        service.approve_draft(rejected.draft_id, "manager")
    assert count(session, ResponseDraft) == 2


def test_draft_decision_requires_manager_and_rolls_back(session: Session) -> None:
    tenant_id = tenant(session, "human-approval")
    lead_id = qualified_lead(session, tenant_id, "human-approval-event")
    service = workflow(session, tenant_id)
    draft = service.generate_draft(lead_id, "system")
    audit_before = count(session, AuditEvent)

    with pytest.raises(WorkflowPrerequisiteError):
        service.approve_draft(draft.draft_id, "system")

    stored = session.get(ResponseDraft, draft.draft_id)
    assert stored.status == "APPROVAL_REQUIRED"
    assert stored.approved_at is None
    assert count(session, AuditEvent) == audit_before


def test_mock_draft_is_deterministic_and_offline(monkeypatch) -> None:
    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    context = DraftContext(
        customer_name="Анна",
        task="Автоматизировать заявки",
        summary="Готовы к внедрению",
        budget_amount=300_000,
        budget_currency="RUB",
        deadline="2 недели",
    )
    provider = MockDraftProvider()
    assert provider.generate(context) == provider.generate(context)


def test_tenant_isolation_for_workflow_entities(session: Session) -> None:
    tenant_a = tenant(session, "workflow-a")
    tenant_b = tenant(session, "workflow-b")
    lead_b = qualified_lead(session, tenant_b)
    service_b = workflow(session, tenant_b)
    task_b = service_b.create_task_from_recommendation(lead_b, "manager")
    draft_b = service_b.generate_draft(lead_b, "manager")
    service_a = workflow(session, tenant_a)

    with pytest.raises(EntityNotFoundError):
        service_a.create_task_from_recommendation(lead_b, "manager")
    with pytest.raises(EntityNotFoundError):
        service_a.complete_task(task_b.task_id, "manager")
    with pytest.raises(EntityNotFoundError):
        service_a.approve_draft(draft_b.draft_id, "manager")
    with pytest.raises(EntityNotFoundError):
        service_a.transition_lead(lead_b, "QUALIFIED", None, "manager")
    with pytest.raises(TenantScopeError):
        service_a.repository.add_audit(
            AuditEvent(
                tenant_id=tenant_b,
                actor_type="system",
                action="cross-tenant",
                entity_type="lead",
                entity_id=lead_b,
                event_metadata={},
            )
        )


def test_important_actions_are_audited(session: Session) -> None:
    tenant_id = tenant(session, "audit")
    lead_id = qualified_lead(session, tenant_id)
    service = workflow(session, tenant_id)
    task = service.create_task_from_recommendation(lead_id, "manager")
    service.complete_task(task.task_id, "manager")
    draft = service.generate_draft(lead_id, "manager")
    service.approve_draft(draft.draft_id, "manager")
    service.transition_lead(lead_id, "QUALIFIED", "Qualified by manager", "manager")

    actions = set(session.scalars(select(AuditEvent.action).where(AuditEvent.tenant_id == tenant_id)))
    assert {
        "qualification.completed",
        "task.created",
        "task.done",
        "draft.created",
        "draft.approved",
        "lead.status_changed",
    } <= actions


def test_workflow_api_endpoints(session: Session) -> None:
    from ai_lead_manager.main import app

    tenant_id = tenant(session, "workflow-api")
    lead_id = qualified_lead(session, tenant_id)
    body = {"tenant_id": str(tenant_id), "actor_type": "manager"}
    app.dependency_overrides[get_session] = lambda: session
    client = TestClient(app)
    try:
        task = client.post(f"/api/v1/leads/{lead_id}/tasks/from-recommendation", json=body)
        replay = client.post(f"/api/v1/leads/{lead_id}/tasks/from-recommendation", json=body)
        draft = client.post(f"/api/v1/leads/{lead_id}/drafts", json=body)
        approval = client.post(f"/api/v1/drafts/{draft.json()['draft_id']}/approve", json=body)
        transition = client.post(
            f"/api/v1/leads/{lead_id}/status",
            json={**body, "status": "QUALIFIED", "reason": "API test"},
        )
    finally:
        app.dependency_overrides.clear()

    assert task.status_code == 200 and task.json()["created"] is True
    assert replay.status_code == 200 and replay.json()["created"] is False
    assert approval.status_code == 200 and approval.json()["status"] == "APPROVED"
    assert transition.status_code == 200 and transition.json()["status"] == "QUALIFIED"
