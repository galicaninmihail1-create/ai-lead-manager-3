import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.normalization import normalize_email, normalize_phone
from ai_lead_manager.application.schemas import InboundHttpRequest, TenantCreate
from ai_lead_manager.application.services import TenantService
from ai_lead_manager.domain.exceptions import EntityNotFoundError
from ai_lead_manager.domain.models import (
    Conversation,
    Customer,
    CustomerIdentity,
    InboundEvent,
    Lead,
    Message,
)
from ai_lead_manager.infrastructure.database import get_session
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyTenantRepository,
    SqlAlchemyUnitOfWork,
)
from ai_lead_manager.main import app


def tenant(session: Session, slug: str) -> uuid.UUID:
    return TenantService(SqlAlchemyTenantRepository(session), SqlAlchemyUnitOfWork(session)).create(
        TenantCreate(name=slug, slug=slug)
    ).id


def request(tenant_id: uuid.UUID, event_id: str = "evt-1", **changes) -> InboundHttpRequest:
    values = {
        "tenant_id": tenant_id,
        "source": "web-form",
        "external_event_id": event_id,
        "external_user_id": " User-42 ",
        "external_thread_id": "thread-1",
        "name": "Ada",
        "email": "ADA@EXAMPLE.COM",
        "phone": "+1 (202) 555-0100",
        "message": "Please contact me",
        "metadata": {"campaign": "spring"},
    }
    values.update(changes)
    return InboundHttpRequest(**values)


def service(session: Session, tenant_id: uuid.UUID) -> InboundIngestionService:
    return InboundIngestionService(
        SqlAlchemyInboundRepository(session, tenant_id), SqlAlchemyUnitOfWork(session)
    )


def count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_new_event_creates_complete_inbound_graph(session: Session) -> None:
    tenant_id = tenant(session, "tenant-a")
    result = service(session, tenant_id).ingest(request(tenant_id))

    assert result.duplicate is False
    assert result.lead_created is True
    assert count(session, InboundEvent) == 1
    assert count(session, Customer) == 1
    assert count(session, CustomerIdentity) == 3
    assert count(session, Conversation) == 1
    assert count(session, Message) == 1
    assert count(session, Lead) == 1
    event = session.get(InboundEvent, result.event_id)
    assert event is not None and event.status == "processed" and event.processed_at is not None


def test_replay_is_idempotent(session: Session) -> None:
    tenant_id = tenant(session, "tenant-a")
    first = service(session, tenant_id).ingest(request(tenant_id))
    replay = service(session, tenant_id).ingest(request(tenant_id))

    assert replay.duplicate is True
    assert replay.event_id == first.event_id
    assert replay.customer_id == first.customer_id
    assert replay.message_id == first.message_id
    assert replay.lead_id == first.lead_id
    assert count(session, InboundEvent) == 1
    assert count(session, Customer) == 1
    assert count(session, Message) == 1
    assert count(session, Lead) == 1


def test_existing_customer_conversation_and_lead_are_reused(session: Session) -> None:
    tenant_id = tenant(session, "tenant-a")
    first = service(session, tenant_id).ingest(request(tenant_id))
    second = service(session, tenant_id).ingest(request(tenant_id, "evt-2"))

    assert second.customer_id == first.customer_id
    assert second.conversation_id == first.conversation_id
    assert second.lead_id == first.lead_id
    assert second.message_id != first.message_id
    assert second.lead_created is False
    assert count(session, Customer) == 1
    assert count(session, CustomerIdentity) == 3
    assert count(session, Conversation) == 1
    assert count(session, Message) == 2
    assert count(session, Lead) == 1


def test_identity_normalization() -> None:
    assert normalize_email("  Ada@Example.COM ") == "ada@example.com"
    assert normalize_phone(" +1 (202) 555-0100 ") == "+12025550100"
    assert normalize_phone("8 999 123-45-67") == "89991234567"
    with pytest.raises(ValueError):
        normalize_phone("+1-CALL-ME")


def test_same_email_is_isolated_between_tenants(session: Session) -> None:
    tenant_a = tenant(session, "tenant-a")
    tenant_b = tenant(session, "tenant-b")
    result_a = service(session, tenant_a).ingest(request(tenant_a, external_user_id=None))
    result_b = service(session, tenant_b).ingest(request(tenant_b, external_user_id=None))

    assert result_a.customer_id != result_b.customer_id
    assert count(session, Customer) == 2
    assert count(session, CustomerIdentity) == 4


def test_invalid_tenant_rolls_back_event(session: Session) -> None:
    missing = uuid.uuid4()
    with pytest.raises(EntityNotFoundError):
        service(session, missing).ingest(request(missing))
    assert count(session, InboundEvent) == 0


def test_http_endpoint_uses_validated_body_tenant(session: Session) -> None:
    tenant_id = tenant(session, "tenant-api")
    app.dependency_overrides[get_session] = lambda: session
    try:
        response = TestClient(app).post(
            "/api/v1/inbound/http", json=request(tenant_id).model_dump(mode="json")
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["tenant_id"] == str(tenant_id)
    assert response.json()["duplicate"] is False

    app.dependency_overrides[get_session] = lambda: session
    try:
        replay = TestClient(app).post(
            "/api/v1/inbound/http", json=request(tenant_id).model_dump(mode="json")
        )
    finally:
        app.dependency_overrides.clear()
    assert replay.status_code == 200
    assert replay.json()["duplicate"] is True


def test_inbound_event_cannot_reference_other_tenant_customer(session: Session) -> None:
    tenant_a = tenant(session, "event-tenant-a")
    tenant_b = tenant(session, "event-tenant-b")
    customer_b = Customer(tenant_id=tenant_b, name="B")
    session.add(customer_b)
    session.commit()
    session.add(
        InboundEvent(
            tenant_id=tenant_a,
            source="test",
            external_event_id="cross-tenant",
            raw_payload={},
            status="processed",
            customer_id=customer_b.id,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_unsupported_identity_type_is_rejected_by_database(session: Session) -> None:
    tenant_id = tenant(session, "identity-type-tenant")
    customer = Customer(tenant_id=tenant_id, name="Customer")
    session.add(customer)
    session.flush()
    session.add(
        CustomerIdentity(
            tenant_id=tenant_id,
            customer_id=customer.id,
            identity_type="username",
            normalized_value="ada",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


@pytest.mark.parametrize(
    ("factory", "constraint"),
    [
        (
            lambda a, b, customer, conversation: CustomerIdentity(
                tenant_id=a, customer_id=customer, identity_type="email", normalized_value="x@example.com"
            ),
            "identity",
        ),
        (
            lambda a, b, customer, conversation: Conversation(
                tenant_id=a, customer_id=customer, channel="test", status="open"
            ),
            "conversation",
        ),
        (
            lambda a, b, customer, conversation: Message(
                tenant_id=a, conversation_id=conversation, direction="inbound", body="test"
            ),
            "message",
        ),
        (
            lambda a, b, customer, conversation: Lead(
                tenant_id=a, customer_id=customer, conversation_id=conversation, status="NEW"
            ),
            "lead",
        ),
    ],
)
def test_cross_tenant_references_are_rejected(
    session: Session, factory, constraint: str
) -> None:
    tenant_a = tenant(session, f"tenant-a-{constraint}")
    tenant_b = tenant(session, f"tenant-b-{constraint}")
    customer_b = Customer(tenant_id=tenant_b, name="B")
    session.add(customer_b)
    session.flush()
    conversation_b = Conversation(
        tenant_id=tenant_b, customer_id=customer_b.id, channel="test", status="open"
    )
    session.add(conversation_b)
    session.commit()

    session.add(factory(tenant_a, tenant_b, customer_b.id, conversation_b.id))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
