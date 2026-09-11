import os
import uuid

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.schemas import InboundHttpRequest
from ai_lead_manager.domain.models import (
    Customer,
    CustomerIdentity,
    InboundEvent,
    Lead,
    Message,
    Tenant,
)
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyUnitOfWork,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1", reason="PostgreSQL integration tests are opt-in"
)


def test_postgres_idempotency_and_tenant_constraints() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    tenant_a = uuid.uuid4()
    tenant_b = uuid.uuid4()
    slug_suffix = tenant_a.hex

    with Session(engine, expire_on_commit=False) as session:
        try:
            session.add_all(
                [
                    Tenant(id=tenant_a, name="PG integration A", slug=f"pg-a-{slug_suffix}"),
                    Tenant(id=tenant_b, name="PG integration B", slug=f"pg-b-{slug_suffix}"),
                ]
            )
            session.commit()
            request = InboundHttpRequest(
                tenant_id=tenant_a,
                source="pytest-postgres",
                external_event_id=f"event-{slug_suffix}",
                email="same@example.com",
                message="PostgreSQL integration test",
            )
            service = InboundIngestionService(
                SqlAlchemyInboundRepository(session, tenant_a), SqlAlchemyUnitOfWork(session)
            )
            first = service.ingest(request)
            replay = service.ingest(request)

            assert replay.duplicate is True
            assert replay.event_id == first.event_id
            assert session.scalar(
                select(func.count()).select_from(InboundEvent).where(InboundEvent.tenant_id == tenant_a)
            ) == 1
            assert session.scalar(
                select(func.count()).select_from(Message).where(Message.tenant_id == tenant_a)
            ) == 1
            assert session.scalar(
                select(func.count()).select_from(Lead).where(Lead.tenant_id == tenant_a)
            ) == 1

            customer = session.get(Customer, first.customer_id)
            session.add(
                CustomerIdentity(
                    tenant_id=tenant_b,
                    customer_id=customer.id,
                    identity_type="email",
                    normalized_value="cross@example.com",
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            session.add(
                InboundEvent(
                    tenant_id=tenant_a,
                    source=request.source,
                    external_event_id=request.external_event_id,
                    raw_payload={},
                    status="received",
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()
        finally:
            session.execute(delete(Tenant).where(Tenant.id.in_([tenant_a, tenant_b])))
            session.commit()
