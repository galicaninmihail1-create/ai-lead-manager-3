import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ai_lead_manager.application.ai.money import Currency, CurrencyMismatchError, Money
from ai_lead_manager.domain.exceptions import TenantScopeError
from ai_lead_manager.domain.models import AIOperation, AuditEvent, ProviderCall, TenantAIPolicy


class SqlAlchemyAIExecutionRepository:
    def __init__(self, session: Session, tenant_id: uuid.UUID) -> None:
        self.session = session
        self.tenant_id = tenant_id

    def _add[T](self, entity: T) -> T:
        if getattr(entity, "tenant_id", None) != self.tenant_id:
            raise TenantScopeError("Entity tenant does not match AI repository tenant scope")
        self.session.add(entity)
        self.session.flush()
        return entity

    def get_policy_for_update(self) -> TenantAIPolicy | None:
        return self.session.scalar(
            select(TenantAIPolicy)
            .where(TenantAIPolicy.tenant_id == self.tenant_id)
            .with_for_update()
        )

    def get_operation_by_key_for_update(
        self, operation_type: str, idempotency_key: str
    ) -> AIOperation | None:
        return self.session.scalar(
            select(AIOperation)
            .where(
                AIOperation.tenant_id == self.tenant_id,
                AIOperation.operation_type == operation_type,
                AIOperation.idempotency_key == idempotency_key,
            )
            .with_for_update()
        )

    def get_operation_for_update(self, operation_id: uuid.UUID) -> AIOperation | None:
        return self.session.scalar(
            select(AIOperation)
            .where(
                AIOperation.id == operation_id,
                AIOperation.tenant_id == self.tenant_id,
            )
            .with_for_update()
        )

    def get_provider_calls_for_update(self, operation_id: uuid.UUID) -> list[ProviderCall]:
        return list(
            self.session.scalars(
                select(ProviderCall)
                .where(
                    ProviderCall.operation_id == operation_id,
                    ProviderCall.tenant_id == self.tenant_id,
                )
                .order_by(ProviderCall.attempt, ProviderCall.id)
                .with_for_update()
            )
        )

    def add_operation(self, operation: AIOperation) -> AIOperation:
        return self._add(operation)

    def add_provider_call(self, call: ProviderCall) -> ProviderCall:
        return self._add(call)

    def current_period_spend(self, period_start: datetime, currency: Currency) -> Money:
        currencies = set(
            self.session.scalars(
                select(AIOperation.cost_currency)
                .where(
                    AIOperation.tenant_id == self.tenant_id,
                    AIOperation.created_at >= period_start,
                )
                .distinct()
            )
        )
        if currencies - {currency.value}:
            raise CurrencyMismatchError(
                "Current budget period contains AI operations in another currency"
            )
        total = self.session.scalar(
            select(
                func.coalesce(
                    func.sum(
                        func.coalesce(AIOperation.actual_cost_micros, 0)
                        + AIOperation.reserved_cost_micros
                    ),
                    0,
                )
            ).where(
                AIOperation.tenant_id == self.tenant_id,
                AIOperation.created_at >= period_start,
            )
        )
        return Money(amount_micros=int(total or 0), currency=currency)

    def add_audit(self, event: AuditEvent) -> AuditEvent:
        return self._add(event)


class SqlAlchemyAIRepositoryFactory:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    @contextmanager
    def transaction(
        self, tenant_id: uuid.UUID
    ) -> Iterator[SqlAlchemyAIExecutionRepository]:
        with self.session_factory() as session, session.begin():
            yield SqlAlchemyAIExecutionRepository(session, tenant_id)
