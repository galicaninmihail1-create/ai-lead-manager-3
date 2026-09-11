from dataclasses import dataclass

from ai_lead_manager.application.ai.contracts import AIErrorKind, AIProviderError


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    max_retry_after_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")

    def should_retry(self, error: AIProviderError, attempt: int) -> bool:
        if attempt >= self.max_attempts or error.ambiguous:
            return False
        if error.kind == AIErrorKind.TIMEOUT_CONNECT:
            return True
        if error.kind == AIErrorKind.RATE_LIMIT:
            return True
        if error.kind in {AIErrorKind.PROVIDER_5XX, AIErrorKind.TRANSPORT}:
            return error.safe_to_retry
        return False

    def delay_seconds(self, error: AIProviderError, attempt: int) -> float:
        if error.kind == AIErrorKind.RATE_LIMIT and error.retry_after_seconds is not None:
            return min(max(error.retry_after_seconds, 0), self.max_retry_after_seconds)
        return min(float(2 ** (attempt - 1)), self.max_retry_after_seconds)
