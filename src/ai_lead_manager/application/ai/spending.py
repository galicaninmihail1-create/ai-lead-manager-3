from dataclasses import dataclass
from datetime import UTC, datetime

from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    BudgetExceededError,
    UnknownCostError,
)
from ai_lead_manager.application.ai.money import Currency, CurrencyMismatchError, Money
from ai_lead_manager.application.ai.ports import AIExecutionRepository
from ai_lead_manager.domain.models import TenantAIPolicy


@dataclass(frozen=True, slots=True)
class SpendingReservation:
    amount: Money
    period_spend_before: Money


class SpendingGuard:
    @staticmethod
    def period_start(now: datetime | None = None) -> datetime:
        current = now or datetime.now(UTC)
        return current.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def reserve(
        self,
        repository: AIExecutionRepository,
        policy: TenantAIPolicy,
        amount: Money,
        *,
        now: datetime | None = None,
    ) -> SpendingReservation:
        try:
            budget_currency = Currency(policy.budget_currency)
        except ValueError as error:
            raise UnknownCostError(
                AIErrorKind.UNKNOWN_COST, "Tenant AI budget currency is unsupported"
            ) from error
        if amount.currency != budget_currency:
            raise UnknownCostError(
                AIErrorKind.UNKNOWN_COST,
                "Model pricing currency does not match the tenant AI budget currency",
            )
        per_call_limit = Money(
            amount_micros=policy.per_call_limit_micros, currency=budget_currency
        )
        monthly_limit = Money(
            amount_micros=policy.monthly_limit_micros, currency=budget_currency
        )
        if amount > per_call_limit:
            raise BudgetExceededError(
                AIErrorKind.BUDGET_EXCEEDED,
                "Worst-case AI cost exceeds the tenant per-call limit",
            )
        try:
            current = repository.current_period_spend(self.period_start(now), budget_currency)
            projected = current + amount
        except CurrencyMismatchError as error:
            raise UnknownCostError(
                AIErrorKind.UNKNOWN_COST,
                "Current AI spend currency does not match the tenant budget currency",
            ) from error
        if projected > monthly_limit:
            raise BudgetExceededError(
                AIErrorKind.BUDGET_EXCEEDED,
                "Worst-case AI cost exceeds the tenant monthly hard limit",
            )
        return SpendingReservation(
            amount=amount,
            period_spend_before=current,
        )
