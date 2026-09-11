from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ai_lead_manager.application.ai.circuit import CircuitBreakerRegistry
from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIExecutionError,
    AIExecutionMode,
    AIExecutionOutcome,
    AIFinalizationError,
    AIIdempotencyConflictError,
    AIOperationNotReplayableError,
    AIOperationStatus,
    AIOperationType,
    AIProviderError,
    AIUsage,
    BudgetExceededError,
    ProviderCallStatus,
    ProviderDisabledError,
    ProviderTimeouts,
    StructuredAIRequest,
)
from ai_lead_manager.application.ai.ports import AIRepositoryFactory
from ai_lead_manager.application.ai.pricing import ModelPricing, PricingCatalog
from ai_lead_manager.application.ai.registry import ProviderRegistry
from ai_lead_manager.application.ai.retry import RetryPolicy
from ai_lead_manager.application.ai.spending import SpendingGuard
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.config import Settings
from ai_lead_manager.domain.models import AIOperation, AuditEvent, ProviderCall, TenantAIPolicy

ResultT = TypeVar("ResultT", bound=BaseModel)


class AIExecutor:
    def __init__(
        self,
        settings: Settings,
        repository_factory: AIRepositoryFactory,
        providers: ProviderRegistry,
        pricing: PricingCatalog,
        *,
        spending_guard: SpendingGuard | None = None,
        circuit_breakers: CircuitBreakerRegistry | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.repository_factory = repository_factory
        self.providers = providers
        self.pricing = pricing
        self.spending_guard = spending_guard or SpendingGuard()
        self.circuit_breakers = circuit_breakers or CircuitBreakerRegistry()
        self.sleeper = sleeper

    def execute(
        self,
        tenant: TenantContext,
        operation_type: AIOperationType,
        lead_id: uuid.UUID | None,
        idempotency_key: str,
        request_hash: str,
        request: StructuredAIRequest,
        response_type: type[ResultT],
    ) -> AIExecutionOutcome[ResultT]:
        self._validate_identity(idempotency_key, request_hash)
        if request.operation_type != operation_type:
            raise ValueError("Request operation type does not match executor operation type")

        budget_error: BudgetExceededError | None = None
        with self.repository_factory.transaction(tenant.tenant_id) as repository:
            policy = repository.get_policy_for_update()
            existing = repository.get_operation_by_key_for_update(
                operation_type.value, idempotency_key
            )
            if existing is not None:
                return self._existing_outcome(existing, request_hash)

            self._validate_policy(tenant, policy)
            assert policy is not None
            provider_name, model_name = self._route(policy, operation_type)
            provider = self.providers.get(provider_name, model_name)
            if provider.execution_mode.value != policy.mode:
                raise ProviderDisabledError(
                    AIErrorKind.PROVIDER_DISABLED,
                    "Registered provider mode does not match tenant policy",
                )
            circuit = self.circuit_breakers.get(provider_name, model_name)
            circuit.before_call()
            price = self.pricing.get(provider_name, model_name)
            attempts = min(policy.max_attempts, self.settings.ai_max_attempts)
            max_output = min(
                request.max_output_tokens,
                policy.max_output_tokens,
                self.settings.ai_max_output_tokens,
            )
            if request.estimated_input_tokens > min(
                policy.max_input_tokens, self.settings.ai_max_input_tokens
            ):
                raise AIExecutionError(
                    AIErrorKind.INVALID_REQUEST, "AI input exceeds the configured token limit"
                )
            estimate = price.worst_case_cost(
                request.estimated_input_tokens, max_output, attempts
            )
            bounded_request = request.model_copy(
                update={
                    "max_output_tokens": max_output,
                    "timeouts": ProviderTimeouts(
                        connect_seconds=self.settings.ai_connect_timeout_seconds,
                        read_seconds=self.settings.ai_read_timeout_seconds,
                        total_seconds=self.settings.ai_total_timeout_seconds,
                    ),
                }
            )
            try:
                reservation = self.spending_guard.reserve(repository, policy, estimate)
            except BudgetExceededError as error:
                budget_error = error
                repository.add_audit(
                    AuditEvent(
                        tenant_id=tenant.tenant_id,
                        actor_type="system",
                        action="ai.budget_rejected",
                        entity_type="tenant",
                        entity_id=tenant.tenant_id,
                        event_metadata={
                            "operation_type": operation_type.value,
                            "estimated_max_cost_microusd": estimate,
                        },
                    )
                )
            else:
                operation = repository.add_operation(
                    AIOperation(
                        tenant_id=tenant.tenant_id,
                        operation_type=operation_type.value,
                        lead_id=lead_id,
                        idempotency_key=idempotency_key,
                        request_hash=request_hash,
                        provider=provider_name,
                        model=model_name,
                        pricing_version=price.version,
                        status=AIOperationStatus.CALLING.value,
                        estimated_max_cost_microusd=estimate,
                        reserved_cost_microusd=reservation.amount_microusd,
                        actual_cost_microusd=0,
                    )
                )
                repository.add_audit(
                    self._audit(
                        tenant,
                        "ai.operation_reserved",
                        operation,
                        {
                            "estimated_max_cost_microusd": estimate,
                            "period_spend_before_microusd": reservation.period_spend_before_microusd,
                        },
                    )
                )
                repository.add_audit(
                    self._audit(tenant, "ai.execution_started", operation, {})
                )
                operation_id = operation.id

        if budget_error is not None:
            raise budget_error

        retry_policy = RetryPolicy(max_attempts=attempts)
        for attempt in range(1, attempts + 1):
            try:
                result = provider.execute(bounded_request)
                if result.usage is None:
                    error = AIProviderError(
                        AIErrorKind.UNKNOWN_COST,
                        "Provider did not return usage required for cost accounting",
                        ambiguous=True,
                        cost_known=False,
                        provider_request_id=result.metadata.provider_request_id,
                    )
                    self._record_error(tenant, operation_id, attempt, operation, price, error, True)
                    raise error
                self._validate_provider_identity(
                    result.metadata.provider,
                    result.metadata.model,
                    operation,
                    result.usage,
                    result.metadata.provider_request_id,
                )
                try:
                    structured = response_type.model_validate(result.output)
                except ValidationError as validation_error:
                    error = AIProviderError(
                        AIErrorKind.SCHEMA_VALIDATION,
                        "Provider output did not match the required schema",
                        usage=result.usage,
                        provider_request_id=result.metadata.provider_request_id,
                    )
                    self._record_error(tenant, operation_id, attempt, operation, price, error, True)
                    raise error from validation_error
                actual_cost = price.actual_cost(result.usage)
                try:
                    self._record_success(
                        tenant, operation_id, attempt, operation, result, actual_cost
                    )
                except Exception as persistence_error:
                    self._mark_unknown_best_effort(
                        tenant, operation_id, "provider_result_persistence_failed"
                    )
                    raise AIFinalizationError(
                        AIErrorKind.AMBIGUOUS_RESULT,
                        "Provider succeeded but its result could not be persisted",
                    ) from persistence_error
                circuit.record_success()
                return AIExecutionOutcome(
                    operation_id=operation_id,
                    value=structured,
                    replayed=False,
                    provider=provider_name,
                    model=model_name,
                )
            except AIProviderError as error:
                if error.kind in {AIErrorKind.SCHEMA_VALIDATION, AIErrorKind.UNKNOWN_COST}:
                    raise
                terminal = not retry_policy.should_retry(error, attempt)
                self._record_error(tenant, operation_id, attempt, operation, price, error, terminal)
                if terminal:
                    circuit.record_failure(error)
                    raise
                self.sleeper(retry_policy.delay_seconds(error, attempt))

        raise AssertionError("Retry loop exhausted without a terminal outcome")

    def mark_unknown(
        self, tenant: TenantContext, operation_id: uuid.UUID, error_code: str
    ) -> None:
        with self.repository_factory.transaction(tenant.tenant_id) as repository:
            operation = repository.get_operation_for_update(operation_id)
            if operation is None or operation.status == AIOperationStatus.SUCCEEDED.value:
                return
            operation.status = AIOperationStatus.UNKNOWN.value
            operation.error_code = error_code
            repository.add_audit(
                self._audit(tenant, "ai.operation_unknown", operation, {"error_code": error_code})
            )

    def _mark_unknown_best_effort(
        self, tenant: TenantContext, operation_id: uuid.UUID, error_code: str
    ) -> None:
        try:
            self.mark_unknown(tenant, operation_id, error_code)
        except Exception:
            # The committed CALLING state remains fail-closed and is reconciled manually.
            pass

    @staticmethod
    def _validate_identity(idempotency_key: str, request_hash: str) -> None:
        if not idempotency_key.strip() or len(idempotency_key) > 200:
            raise ValueError("A bounded non-empty idempotency key is required")
        if len(request_hash) != 64 or any(character not in "0123456789abcdef" for character in request_hash):
            raise ValueError("request_hash must be a lowercase SHA-256 digest")

    def _validate_policy(
        self, tenant: TenantContext, policy: TenantAIPolicy | None
    ) -> None:
        if self.settings.ai_kill_switch:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED, "Global AI kill switch is active"
            )
        if policy is None or not policy.enabled:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED, "Tenant AI policy is missing or disabled"
            )
        if policy.mode == AIExecutionMode.LIVE.value:
            if not tenant.trusted:
                raise ProviderDisabledError(
                    AIErrorKind.PROVIDER_DISABLED,
                    "LIVE AI requires an explicitly trusted tenant context",
                )
            if (
                self.settings.ai_execution_mode != AIExecutionMode.LIVE.value
                or not self.settings.allow_live_ai
            ):
                raise ProviderDisabledError(
                    AIErrorKind.PROVIDER_DISABLED,
                    "LIVE AI is not enabled by global fail-closed settings",
                )
        elif policy.mode != AIExecutionMode.MOCK.value:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED, "Tenant AI mode is not supported"
            )

    @staticmethod
    def _route(
        policy: TenantAIPolicy, operation_type: AIOperationType
    ) -> tuple[str, str]:
        if operation_type == AIOperationType.QUALIFICATION:
            return policy.qualification_provider, policy.qualification_model
        return policy.draft_provider, policy.draft_model

    @staticmethod
    def _existing_outcome(
        operation: AIOperation, request_hash: str
    ) -> AIExecutionOutcome:
        if operation.request_hash != request_hash:
            raise AIIdempotencyConflictError(
                AIErrorKind.INVALID_REQUEST,
                "Idempotency key was already used with a different request hash",
            )
        if operation.status == AIOperationStatus.SUCCEEDED.value:
            return AIExecutionOutcome(
                operation_id=operation.id,
                value=None,
                replayed=True,
                provider=operation.provider,
                model=operation.model,
                result_analysis_id=operation.result_analysis_id,
                result_draft_id=operation.result_draft_id,
            )
        raise AIOperationNotReplayableError(
            AIErrorKind.AMBIGUOUS_RESULT,
            f"Existing operation is {operation.status}; it will not be executed again",
        )

    @staticmethod
    def _validate_provider_identity(
        actual_provider: str,
        actual_model: str,
        operation: AIOperation,
        usage: AIUsage,
        provider_request_id: str | None,
    ) -> None:
        if (actual_provider, actual_model) != (operation.provider, operation.model):
            raise AIProviderError(
                AIErrorKind.INVALID_REQUEST,
                "Provider response identity does not match the reserved operation",
                usage=usage,
                provider_request_id=provider_request_id,
            )

    def _record_success(
        self,
        tenant: TenantContext,
        operation_id: uuid.UUID,
        attempt: int,
        reserved_operation: AIOperation,
        result,
        actual_cost: int,
    ) -> None:
        with self.repository_factory.transaction(tenant.tenant_id) as repository:
            operation = repository.get_operation_for_update(operation_id)
            if operation is None:
                raise RuntimeError("Reserved AI operation disappeared")
            repository.add_provider_call(
                ProviderCall(
                    tenant_id=tenant.tenant_id,
                    operation_id=operation.id,
                    attempt=attempt,
                    provider=reserved_operation.provider,
                    model=reserved_operation.model,
                    provider_request_id=result.metadata.provider_request_id,
                    status=ProviderCallStatus.SUCCEEDED.value,
                    latency_ms=result.latency_ms,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    cached_tokens=result.usage.cached_tokens,
                    actual_cost_microusd=actual_cost,
                    cost_known=True,
                    finish_reason=result.metadata.finish_reason,
                )
            )
            operation.actual_cost_microusd = (operation.actual_cost_microusd or 0) + actual_cost
            operation.reserved_cost_microusd = max(
                operation.estimated_max_cost_microusd - operation.actual_cost_microusd, 0
            )
            repository.add_audit(
                self._audit(
                    tenant,
                    "ai.provider_attempt_succeeded",
                    operation,
                    {"attempt": attempt, "actual_cost_microusd": actual_cost},
                )
            )

    def _record_error(
        self,
        tenant: TenantContext,
        operation_id: uuid.UUID,
        attempt: int,
        reserved_operation: AIOperation,
        price: ModelPricing,
        error: AIProviderError,
        terminal: bool,
    ) -> None:
        with self.repository_factory.transaction(tenant.tenant_id) as repository:
            operation = repository.get_operation_for_update(operation_id)
            if operation is None:
                raise RuntimeError("Reserved AI operation disappeared")
            actual_cost = price.actual_cost(error.usage) if error.usage is not None else 0
            cost_known = error.cost_known
            call_status = (
                ProviderCallStatus.UNKNOWN.value
                if error.ambiguous or not cost_known
                else ProviderCallStatus.FAILED.value
            )
            repository.add_provider_call(
                ProviderCall(
                    tenant_id=tenant.tenant_id,
                    operation_id=operation.id,
                    attempt=attempt,
                    provider=reserved_operation.provider,
                    model=reserved_operation.model,
                    provider_request_id=error.provider_request_id,
                    status=call_status,
                    http_status=error.http_status,
                    error_kind=error.kind.value,
                    input_tokens=error.usage.input_tokens if error.usage else None,
                    output_tokens=error.usage.output_tokens if error.usage else None,
                    cached_tokens=error.usage.cached_tokens if error.usage else None,
                    actual_cost_microusd=actual_cost if cost_known else None,
                    cost_known=cost_known,
                )
            )
            if cost_known:
                operation.actual_cost_microusd = (operation.actual_cost_microusd or 0) + actual_cost
                operation.reserved_cost_microusd = max(
                    operation.estimated_max_cost_microusd - operation.actual_cost_microusd, 0
                )
            if terminal:
                unknown = error.ambiguous or not cost_known or error.kind in {
                    AIErrorKind.TIMEOUT_READ,
                    AIErrorKind.TIMEOUT_TOTAL,
                    AIErrorKind.AMBIGUOUS_RESULT,
                }
                operation.status = (
                    AIOperationStatus.UNKNOWN.value if unknown else AIOperationStatus.FAILED.value
                )
                operation.error_code = error.kind.value
                if not unknown:
                    operation.reserved_cost_microusd = 0
            repository.add_audit(
                self._audit(
                    tenant,
                    "ai.provider_attempt_failed",
                    operation,
                    {"attempt": attempt, "error_kind": error.kind.value, "terminal": terminal},
                )
            )
            if terminal:
                repository.add_audit(
                    self._audit(
                        tenant,
                        "ai.operation_unknown" if operation.status == "UNKNOWN" else "ai.operation_failed",
                        operation,
                        {"error_code": error.kind.value},
                    )
                )

    @staticmethod
    def _audit(
        tenant: TenantContext,
        action: str,
        operation: AIOperation,
        metadata: dict,
    ) -> AuditEvent:
        return AuditEvent(
            tenant_id=tenant.tenant_id,
            actor_type="system",
            action=action,
            entity_type="ai_operation",
            entity_id=operation.id,
            event_metadata={
                "operation_id": str(operation.id),
                "operation_type": operation.operation_type,
                "tenant_context_source": tenant.source,
                **metadata,
            },
        )
