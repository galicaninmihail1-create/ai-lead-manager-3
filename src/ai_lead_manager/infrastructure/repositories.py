from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_lead_manager.domain.exceptions import ConcurrentInboundEventError, TenantScopeError
from ai_lead_manager.domain.models import (
    Conversation,
    Customer,
    CustomerIdentity,
    InboundEvent,
    Lead,
    Message,
    Tenant,
)


class SqlAlchemyUnitOfWork:
    def __init__(self, session: Session) -> None:
        self.session = session

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()


class SqlAlchemyTenantRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, tenant: Tenant) -> Tenant:
        self.session.add(tenant)
        self.session.flush()
        return tenant

    def get(self, tenant_id: uuid.UUID) -> Tenant | None:
        return self.session.get(Tenant, tenant_id)


class TenantScopedRepository[T: (Customer, Lead)]:
    model: type[T]

    def __init__(self, session: Session, tenant_id: uuid.UUID) -> None:
        self.session = session
        self.tenant_id = tenant_id

    def add(self, entity: T) -> T:
        if entity.tenant_id != self.tenant_id:
            raise TenantScopeError("Entity tenant does not match repository tenant scope")
        self.session.add(entity)
        self.session.flush()
        return entity

    def get(self, entity_id: uuid.UUID) -> T | None:
        statement = select(self.model).where(
            self.model.id == entity_id,
            self.model.tenant_id == self.tenant_id,
        )
        return self.session.scalar(statement)

    def list(self) -> list[T]:
        statement = select(self.model).where(self.model.tenant_id == self.tenant_id)
        return list(self.session.scalars(statement))


class SqlAlchemyCustomerRepository(TenantScopedRepository[Customer]):
    model = Customer


class SqlAlchemyLeadRepository(TenantScopedRepository[Lead]):
    model = Lead


class SqlAlchemyInboundRepository:
    def __init__(self, session: Session, tenant_id: uuid.UUID) -> None:
        self.session = session
        self.tenant_id = tenant_id

    def _add[T](self, entity: T) -> T:
        if getattr(entity, "tenant_id", None) != self.tenant_id:
            raise TenantScopeError("Entity tenant does not match repository tenant scope")
        self.session.add(entity)
        self.session.flush()
        return entity

    def tenant_exists(self) -> bool:
        return self.session.get(Tenant, self.tenant_id) is not None

    def get_event(self, source: str, external_event_id: str) -> InboundEvent | None:
        return self.session.scalar(
            select(InboundEvent).where(
                InboundEvent.tenant_id == self.tenant_id,
                InboundEvent.source == source,
                InboundEvent.external_event_id == external_event_id,
            )
        )

    def add_event(self, event: InboundEvent) -> InboundEvent:
        try:
            return self._add(event)
        except IntegrityError as error:
            raise ConcurrentInboundEventError from error

    def find_customers(self, identities: list[tuple[str, str]]) -> list[Customer]:
        found: dict[uuid.UUID, Customer] = {}
        for identity_type, normalized_value in identities:
            statement = (
                select(Customer)
                .join(CustomerIdentity, CustomerIdentity.customer_id == Customer.id)
                .where(
                    Customer.tenant_id == self.tenant_id,
                    CustomerIdentity.tenant_id == self.tenant_id,
                    CustomerIdentity.identity_type == identity_type,
                    CustomerIdentity.normalized_value == normalized_value,
                )
            )
            customer = self.session.scalar(statement)
            if customer is not None:
                found[customer.id] = customer
        return list(found.values())

    def add_customer(self, customer: Customer) -> Customer:
        return self._add(customer)

    def existing_identity_keys(self, customer_id: uuid.UUID) -> set[tuple[str, str]]:
        statement = select(CustomerIdentity.identity_type, CustomerIdentity.normalized_value).where(
            CustomerIdentity.tenant_id == self.tenant_id,
            CustomerIdentity.customer_id == customer_id,
        )
        return set(self.session.execute(statement).tuples())

    def add_identity(self, identity: CustomerIdentity) -> CustomerIdentity:
        return self._add(identity)

    def find_conversation(self, source: str, external_thread_id: str) -> Conversation | None:
        return self.session.scalar(
            select(Conversation).where(
                Conversation.tenant_id == self.tenant_id,
                Conversation.channel == source,
                Conversation.external_id == external_thread_id,
            )
        )

    def add_conversation(self, conversation: Conversation) -> Conversation:
        return self._add(conversation)

    def add_message(self, message: Message) -> Message:
        return self._add(message)

    def find_lead(self, customer_id: uuid.UUID) -> Lead | None:
        return self.session.scalar(
            select(Lead).where(Lead.tenant_id == self.tenant_id, Lead.customer_id == customer_id)
        )

    def add_lead(self, lead: Lead) -> Lead:
        return self._add(lead)
