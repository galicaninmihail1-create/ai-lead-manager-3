import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIProviderError,
    ProviderDisabledError,
)


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(slots=True)
class CircuitBreaker:
    failure_threshold: int = 3
    reset_after_seconds: float = 60.0
    clock: Callable[[], float] = time.monotonic
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None

    def before_call(self) -> None:
        if self.state == CircuitState.OPEN:
            assert self.opened_at is not None
            if self.clock() - self.opened_at >= self.reset_after_seconds:
                self.state = CircuitState.HALF_OPEN
            else:
                raise ProviderDisabledError(
                    AIErrorKind.PROVIDER_DISABLED, "Provider circuit is open"
                )

    def record_success(self) -> None:
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self, error: AIProviderError) -> None:
        operational = {
            AIErrorKind.TIMEOUT_CONNECT,
            AIErrorKind.TIMEOUT_READ,
            AIErrorKind.TIMEOUT_TOTAL,
            AIErrorKind.RATE_LIMIT,
            AIErrorKind.TRANSPORT,
            AIErrorKind.PROVIDER_5XX,
            AIErrorKind.AMBIGUOUS_RESULT,
        }
        if error.kind not in operational:
            return
        self.consecutive_failures += 1
        if self.state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = self.clock()


class CircuitBreakerRegistry:
    def __init__(self, failure_threshold: int = 3, reset_after_seconds: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_seconds = reset_after_seconds
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}

    def get(self, provider: str, model: str) -> CircuitBreaker:
        key = (provider, model)
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(
                failure_threshold=self.failure_threshold,
                reset_after_seconds=self.reset_after_seconds,
            )
        return self._breakers[key]
