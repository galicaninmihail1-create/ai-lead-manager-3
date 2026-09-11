from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_lead_manager.domain.exceptions import ConcurrentInboundEventError, TenantScopeError
from ai_lead_manager.domain.models import (
    AIOperation,
    AuditEvent,
    Conversation,
    Customer,
    CustomerIdentity,
    InboundEvent,
    Lead,
    LeadAnalysis,
    LeadStatusHistory,
    Message,
    NextActionRecommendation,
    ResponseDraft,
    SalesTask,
    ScoreFactor,
    ScoreRun,
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


class SqlAlchemyQualificationRepository:
    def __init__(self, session: Session, tenant_id: uuid.UUID) -> None:
        self.session = session
        self.tenant_id = tenant_id

    def _add[T](self, entity: T) -> T:
        if getattr(entity, "tenant_id", None) != self.tenant_id:
            raise TenantScopeError("Entity tenant does not match repository tenant scope")
        self.session.add(entity)
        self.session.flush()
        return entity

    def get_lead(self, lead_id: uuid.UUID) -> Lead | None:
        return self.session.scalar(
            select(Lead).where(Lead.id == lead_id, Lead.tenant_id == self.tenant_id)
        )

    def get_customer(self, customer_id: uuid.UUID) -> Customer | None:
        return self.session.scalar(
            select(Customer).where(Customer.id == customer_id, Customer.tenant_id == self.tenant_id)
        )

    def get_inbound_messages(self, lead: Lead) -> list[Message]:
        if lead.tenant_id != self.tenant_id or lead.conversation_id is None:
            return []
        statement = (
            select(Message)
            .where(
                Message.tenant_id == self.tenant_id,
                Message.conversation_id == lead.conversation_id,
                Message.direction == "inbound",
            )
            .order_by(Message.created_at, Message.id)
        )
        return list(self.session.scalars(statement))

    def add_analysis(self, analysis: LeadAnalysis) -> LeadAnalysis:
        return self._add(analysis)

    def add_score_run(self, score_run: ScoreRun) -> ScoreRun:
        return self._add(score_run)

    def add_score_factor(self, factor: ScoreFactor) -> ScoreFactor:
        return self._add(factor)

    def add_recommendation(
        self, recommendation: NextActionRecommendation
    ) -> NextActionRecommendation:
        return self._add(recommendation)

    def add_audit(self, event: AuditEvent) -> AuditEvent:
        return self._add(event)

    def get_analysis(self, analysis_id: uuid.UUID) -> LeadAnalysis | None:
        return self.session.scalar(
            select(LeadAnalysis).where(
                LeadAnalysis.id == analysis_id,
                LeadAnalysis.tenant_id == self.tenant_id,
            )
        )

    def get_score_run_for_analysis(self, analysis_id: uuid.UUID) -> ScoreRun | None:
        return self.session.scalar(
            select(ScoreRun).where(
                ScoreRun.analysis_id == analysis_id,
                ScoreRun.tenant_id == self.tenant_id,
            )
        )

    def get_score_factors(self, score_run_id: uuid.UUID) -> list[ScoreFactor]:
        return list(
            self.session.scalars(
                select(ScoreFactor)
                .where(
                    ScoreFactor.score_run_id == score_run_id,
                    ScoreFactor.tenant_id == self.tenant_id,
                )
                .order_by(ScoreFactor.created_at, ScoreFactor.id)
            )
        )

    def get_recommendation_for_score_run(
        self, score_run_id: uuid.UUID
    ) -> NextActionRecommendation | None:
        return self.session.scalar(
            select(NextActionRecommendation).where(
                NextActionRecommendation.score_run_id == score_run_id,
                NextActionRecommendation.tenant_id == self.tenant_id,
            )
        )

    def get_ai_operation_for_update(self, operation_id: uuid.UUID) -> AIOperation | None:
        return self.session.scalar(
            select(AIOperation)
            .where(
                AIOperation.id == operation_id,
                AIOperation.tenant_id == self.tenant_id,
            )
            .with_for_update()
        )


class SqlAlchemyWorkflowRepository:
    def __init__(self, session: Session, tenant_id: uuid.UUID) -> None:
        self.session = session
        self.tenant_id = tenant_id

    def _add[T](self, entity: T) -> T:
        if getattr(entity, "tenant_id", None) != self.tenant_id:
            raise TenantScopeError("Entity tenant does not match repository tenant scope")
        self.session.add(entity)
        self.session.flush()
        return entity

    def get_lead(self, lead_id: uuid.UUID) -> Lead | None:
        return self.session.scalar(select(Lead).where(Lead.id == lead_id, Lead.tenant_id == self.tenant_id))

    def get_customer(self, customer_id: uuid.UUID) -> Customer | None:
        return self.session.scalar(select(Customer).where(Customer.id == customer_id, Customer.tenant_id == self.tenant_id))

    def latest_recommendation(self, lead_id: uuid.UUID) -> NextActionRecommendation | None:
        return self.session.scalar(select(NextActionRecommendation).where(NextActionRecommendation.tenant_id == self.tenant_id, NextActionRecommendation.lead_id == lead_id).order_by(NextActionRecommendation.created_at.desc(), NextActionRecommendation.id.desc()))

    def latest_analysis(self, lead_id: uuid.UUID) -> LeadAnalysis | None:
        return self.session.scalar(select(LeadAnalysis).where(LeadAnalysis.tenant_id == self.tenant_id, LeadAnalysis.lead_id == lead_id).order_by(LeadAnalysis.created_at.desc(), LeadAnalysis.id.desc()))

    def task_for_recommendation(self, recommendation_id: uuid.UUID) -> SalesTask | None:
        return self.session.scalar(select(SalesTask).where(SalesTask.tenant_id == self.tenant_id, SalesTask.recommendation_id == recommendation_id))

    def get_task(self, task_id: uuid.UUID) -> SalesTask | None:
        return self.session.scalar(select(SalesTask).where(SalesTask.id == task_id, SalesTask.tenant_id == self.tenant_id))

    def get_draft(self, draft_id: uuid.UUID) -> ResponseDraft | None:
        return self.session.scalar(select(ResponseDraft).where(ResponseDraft.id == draft_id, ResponseDraft.tenant_id == self.tenant_id))

    def add_task(self, task: SalesTask) -> SalesTask:
        return self._add(task)

    def add_draft(self, draft: ResponseDraft) -> ResponseDraft:
        return self._add(draft)

    def add_status_history(self, history: LeadStatusHistory) -> LeadStatusHistory:
        return self._add(history)

    def add_audit(self, event: AuditEvent) -> AuditEvent:
        return self._add(event)

    def get_ai_operation_for_update(self, operation_id: uuid.UUID) -> AIOperation | None:
        return self.session.scalar(
            select(AIOperation)
            .where(
                AIOperation.id == operation_id,
                AIOperation.tenant_id == self.tenant_id,
            )
            .with_for_update()
        )
