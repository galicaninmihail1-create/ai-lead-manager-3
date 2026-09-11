from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class AIExecutionMode(StrEnum):
    MOCK = "MOCK"
    LIVE = "LIVE"


class AIOperationType(StrEnum):
    QUALIFICATION = "QUALIFICATION"
    DRAFT = "DRAFT"


class AIOperationStatus(StrEnum):
    RESERVED = "RESERVED"
    CALLING = "CALLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ProviderCallStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class AIErrorKind(StrEnum):
    TIMEOUT_CONNECT = "timeout_connect"
    TIMEOUT_READ = "timeout_read"
    TIMEOUT_TOTAL = "timeout_total"
    RATE_LIMIT = "rate_limit"
    TRANSPORT = "transport"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    PROVIDER_5XX = "provider_5xx"
    MALFORMED_OUTPUT = "malformed_output"
    SCHEMA_VALIDATION = "schema_validation"
    AMBIGUOUS_RESULT = "ambiguous_result"
    PROVIDER_DISABLED = "provider_disabled"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNKNOWN_COST = "unknown_cost"


class AIUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)


class ProviderMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    provider_model: str | None = Field(default=None, min_length=1, max_length=200)
    provider_request_id: str | None = Field(default=None, max_length=255)
    finish_reason: str | None = Field(default=None, max_length=100)


class ProviderTimeouts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    connect_seconds: float = Field(gt=0)
    read_seconds: float = Field(gt=0)
    total_seconds: float = Field(gt=0)


class StructuredAIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_type: AIOperationType
    schema_name: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=50)
    prompt_version: str = Field(min_length=1, max_length=50)
    payload: dict[str, Any]
    estimated_input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    timeouts: ProviderTimeouts | None = None


class StructuredAIResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output: dict[str, Any]
    usage: AIUsage | None = None
    metadata: ProviderMetadata
    latency_ms: int | None = Field(default=None, ge=0)


class AIProviderError(Exception):
    def __init__(
        self,
        kind: AIErrorKind,
        message: str,
        *,
        http_status: int | None = None,
        retry_after_seconds: float | None = None,
        safe_to_retry: bool = False,
        ambiguous: bool = False,
        usage: AIUsage | None = None,
        provider_request_id: str | None = None,
        cost_known: bool = True,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds
        self.safe_to_retry = safe_to_retry
        self.ambiguous = ambiguous
        self.usage = usage
        self.provider_request_id = provider_request_id
        self.cost_known = cost_known


class AIExecutionError(Exception):
    def __init__(self, kind: AIErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class ProviderDisabledError(AIExecutionError):
    pass


class BudgetExceededError(AIExecutionError):
    pass


class UnknownCostError(AIExecutionError):
    pass


class AIIdempotencyConflictError(AIExecutionError):
    pass


class AIOperationNotReplayableError(AIExecutionError):
    pass


class AIFinalizationError(AIExecutionError):
    pass


class StructuredAIProvider(Protocol):
    provider_name: str
    model_name: str
    execution_mode: AIExecutionMode

    def execute(self, request: StructuredAIRequest) -> StructuredAIResult: ...


@dataclass(frozen=True, slots=True)
class AIExecutionOutcome[ResultT: BaseModel]:
    operation_id: uuid.UUID
    value: ResultT | None
    replayed: bool
    provider: str
    model: str
    result_analysis_id: uuid.UUID | None = None
    result_draft_id: uuid.UUID | None = None
