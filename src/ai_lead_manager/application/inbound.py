from datetime import UTC, datetime

from ai_lead_manager.application.normalization import (
    normalize_email,
    normalize_external_user_id,
    normalize_phone,
)
from ai_lead_manager.application.ports import InboundRepository, UnitOfWork
from ai_lead_manager.application.schemas import InboundHttpRequest, InboundHttpResponse
from ai_lead_manager.domain.exceptions import (
    ConcurrentInboundEventError,
    EntityNotFoundError,
    IdentityConflictError,
)
from ai_lead_manager.domain.models import (
    Conversation,
    Customer,
    CustomerIdentity,
    InboundEvent,
    Lead,
    Message,
)


class InboundIngestionService:
    def __init__(self, repository: InboundRepository, uow: UnitOfWork) -> None:
        self.repository = repository
        self.uow = uow

    def ingest(self, data: InboundHttpRequest) -> InboundHttpResponse:
        try:
            return self._ingest(data)
        except Exception:
            self.uow.rollback()
            raise

    def _ingest(self, data: InboundHttpRequest) -> InboundHttpResponse:
        if data.tenant_id != self.repository.tenant_id:
            raise ValueError("Request and repository tenant scopes differ")
        if not self.repository.tenant_exists():
            raise EntityNotFoundError("Tenant was not found")

        existing = self.repository.get_event(data.source, data.external_event_id)
        if existing is not None:
            return self._response(existing, duplicate=True, lead_created=False)

        event = InboundEvent(
            tenant_id=data.tenant_id,
            source=data.source,
            external_event_id=data.external_event_id,
            raw_payload=data.model_dump(mode="json"),
            status="received",
        )
        try:
            self.repository.add_event(event)
        except ConcurrentInboundEventError:
            self.uow.rollback()
            winner = self.repository.get_event(data.source, data.external_event_id)
            if winner is None:
                raise
            return self._response(winner, duplicate=True, lead_created=False)

        identities = self._identities(data)
        customers = self.repository.find_customers([(kind, normalized) for kind, normalized, _ in identities])
        customer_ids = {customer.id for customer in customers}
        if len(customer_ids) > 1:
            raise IdentityConflictError("Provided identities belong to different customers")

        if customers:
            customer = customers[0]
        else:
            customer = self.repository.add_customer(
                Customer(
                    tenant_id=data.tenant_id,
                    name=data.name or "Unknown customer",
                    email=normalize_email(str(data.email)) if data.email else None,
                    phone=normalize_phone(data.phone) if data.phone else None,
                )
            )

        existing_keys = self.repository.existing_identity_keys(customer.id)
        for identity_type, normalized, original in identities:
            if (identity_type, normalized) not in existing_keys:
                self.repository.add_identity(
                    CustomerIdentity(
                        tenant_id=data.tenant_id,
                        customer_id=customer.id,
                        identity_type=identity_type,
                        normalized_value=normalized,
                        original_value=original,
                    )
                )

        conversation = None
        if data.external_thread_id:
            conversation = self.repository.find_conversation(data.source, data.external_thread_id)
            if conversation is not None and conversation.customer_id != customer.id:
                raise IdentityConflictError("Thread belongs to a different customer")
        if conversation is None:
            conversation = self.repository.add_conversation(
                Conversation(
                    tenant_id=data.tenant_id,
                    customer_id=customer.id,
                    channel=data.source,
                    external_id=data.external_thread_id,
                    status="open",
                )
            )

        message = self.repository.add_message(
            Message(
                tenant_id=data.tenant_id,
                conversation_id=conversation.id,
                direction="inbound",
                body=data.message,
                external_id=data.external_event_id,
            )
        )
        lead = self.repository.find_lead(customer.id)
        lead_created = lead is None
        if lead is None:
            lead = self.repository.add_lead(
                Lead(
                    tenant_id=data.tenant_id,
                    customer_id=customer.id,
                    conversation_id=conversation.id,
                    status="new",
                    source=data.source,
                )
            )

        event.customer_id = customer.id
        event.conversation_id = conversation.id
        event.message_id = message.id
        event.lead_id = lead.id
        event.status = "processed"
        event.processed_at = datetime.now(UTC)
        self.uow.commit()
        return self._response(event, duplicate=False, lead_created=lead_created)

    @staticmethod
    def _identities(data: InboundHttpRequest) -> list[tuple[str, str, str]]:
        identities: list[tuple[str, str, str]] = []
        if data.external_user_id:
            identities.append(
                ("external_user_id", normalize_external_user_id(data.external_user_id), data.external_user_id)
            )
        if data.email:
            identities.append(("email", normalize_email(str(data.email)), str(data.email)))
        if data.phone:
            identities.append(("phone", normalize_phone(data.phone), data.phone))
        return identities

    @staticmethod
    def _response(event: InboundEvent, duplicate: bool, lead_created: bool) -> InboundHttpResponse:
        if not all((event.customer_id, event.conversation_id, event.message_id, event.lead_id)):
            raise RuntimeError("Processed event has an incomplete result")
        return InboundHttpResponse(
            event_id=event.id,
            duplicate=duplicate,
            tenant_id=event.tenant_id,
            customer_id=event.customer_id,
            conversation_id=event.conversation_id,
            message_id=event.message_id,
            lead_id=event.lead_id,
            lead_created=lead_created,
        )
