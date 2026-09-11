from dataclasses import dataclass
from datetime import UTC, datetime

from ai_lead_manager.application.ai.contracts import AIErrorKind, BudgetExceededError
from ai_lead_manager.application.ai.ports import AIExecutionRepository
from ai_lead_manager.domain.models import TenantAIPolicy


@dataclass(frozen=True, slots=True)
class SpendingReservation:
    amount_microusd: int
    period_spend_before_microusd: int


class SpendingGuard:
    @staticmethod
    def period_start(now: datetime | None = None) -> datetime:
        current = now or datetime.now(UTC)
        return current.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def reserve(
        self,
        repository: AIExecutionRepository,
        policy: TenantAIPolicy,
        amount_microusd: int,
        *,
        now: datetime | None = None,
    ) -> SpendingReservation:
        if amount_microusd < 0:
            raise ValueError("Reservation cannot be negative")
        if amount_microusd > policy.per_call_limit_microusd:
            raise BudgetExceededError(
                AIErrorKind.BUDGET_EXCEEDED,
                "Worst-case AI cost exceeds the tenant per-call limit",
            )
        current = repository.current_period_spend(self.period_start(now))
        if current + amount_microusd > policy.monthly_limit_microusd:
            raise BudgetExceededError(
                AIErrorKind.BUDGET_EXCEEDED,
                "Worst-case AI cost exceeds the tenant monthly hard limit",
            )
        return SpendingReservation(
            amount_microusd=amount_microusd,
            period_spend_before_microusd=current,
        )
