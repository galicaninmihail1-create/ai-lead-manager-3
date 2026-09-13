import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_lead_manager.application.ai.contracts import (
    AIOperationStatus,
    AIOperationType,
    ProviderCallStatus,
)
from ai_lead_manager.application.ai.money import POSTGRES_BIGINT_MAX, Currency
from ai_lead_manager.application.ai.ports import AIRepositoryFactory
from ai_lead_manager.domain.models import AuditEvent

MICROS_PER_RUB = Decimal(1_000_000)


class AIReconciliationErrorCode(StrEnum):
    NOT_FOUND = "not_found"
    ALREADY_RECONCILED = "already_reconciled"
    INVALID_STATE = "invalid_state"
    EVIDENCE_MISMATCH = "evidence_mismatch"


class AIReconciliationError(ValueError):
    def __init__(self, code: AIReconciliationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class AIReconciliationEvidence(BaseModel):
    """Safe external evidence for the single supported failed-outcome path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["polza_history"]
    external_generation_id: str = Field(min_length=1, max_length=255)
    external_status: Literal["failed"]
    confirmed_cost_micros: int = Field(ge=0, le=POSTGRES_BIGINT_MAX, strict=True)
    confirmed_cost_currency: Literal[Currency.RUB]
    usage_available: Literal[False]
    external_attempts_count: int = Field(ge=1, strict=True)
    external_created_at: datetime
    external_completed_at: datetime

    @model_validator(mode="after")
    def validate_timeline(self) -> "AIReconciliationEvidence":
        if self.external_created_at.tzinfo is None or self.external_completed_at.tzinfo is None:
            raise ValueError("External reconciliation timestamps must be timezone-aware")
        if self.external_completed_at < self.external_created_at:
            raise ValueError("External completion cannot precede creation")
        return self


class AIReconciliationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation_id: uuid.UUID
    provider_call_id: uuid.UUID
    status: Literal["FAILED"]
    confirmed_cost_micros: int
    confirmed_cost_currency: Currency


class AIReconciliationService:
    """Reconcile one externally verified failed Polza qualification atomically."""

    def __init__(
        self,
        repository_factory: AIRepositoryFactory,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.repository_factory = repository_factory
        self.clock = clock

    def reconcile_failed_qualification(
        self,
        tenant_id: uuid.UUID,
        operation_id: uuid.UUID,
        expected_lead_id: uuid.UUID,
        expected_idempotency_key: str,
        evidence: AIReconciliationEvidence,
    ) -> AIReconciliationResult:
        reconciled_at = self.clock()
        if reconciled_at.tzinfo is None:
            raise ValueError("Reconciliation clock must return a timezone-aware datetime")

        with self.repository_factory.transaction(tenant_id) as repository:
            policy = repository.get_policy_for_update()
            operation = repository.get_operation_for_update(operation_id)
            if operation is None:
                raise AIReconciliationError(
                    AIReconciliationErrorCode.NOT_FOUND,
                    "AI operation was not found in the tenant scope",
                )
            if operation.status == AIOperationStatus.FAILED.value:
                raise AIReconciliationError(
                    AIReconciliationErrorCode.ALREADY_RECONCILED,
                    "AI operation is already failed/reconciled",
                )
            self._validate_operation(
                operation,
                policy,
                expected_lead_id,
                expected_idempotency_key,
                evidence,
            )
            calls = repository.get_provider_calls_for_update(operation.id)
            if len(calls) != 1:
                raise AIReconciliationError(
                    AIReconciliationErrorCode.INVALID_STATE,
                    "Reconciliation requires exactly one persisted provider call",
                )
            call = calls[0]
            self._validate_call(call, operation, evidence)

            previous_operation_status = operation.status
            previous_provider_call_status = call.status
            operation.status = AIOperationStatus.FAILED.value
            operation.reserved_cost_micros = 0
            operation.actual_cost_micros = evidence.confirmed_cost_micros
            call.status = ProviderCallStatus.FAILED.value
            call.cost_known = True
            call.actual_cost_micros = evidence.confirmed_cost_micros
            call.cost_currency = evidence.confirmed_cost_currency.value
            call.provider_reported_cost_decimal = (
                Decimal(evidence.confirmed_cost_micros) / MICROS_PER_RUB
            )
            call.provider_reported_cost_micros = evidence.confirmed_cost_micros
            call.provider_reported_cost_currency = evidence.confirmed_cost_currency.value
            repository.add_audit(
                AuditEvent(
                    tenant_id=tenant_id,
                    actor_type="system",
                    action="ai.operation_reconciled",
                    entity_type="ai_operation",
                    entity_id=operation.id,
                    event_metadata={
                        "operation_id": str(operation.id),
                        "operation_type": operation.operation_type,
                        "reconciliation_source": evidence.source,
                        "external_generation_id": evidence.external_generation_id,
                        "external_status": evidence.external_status,
                        "confirmed_cost": {
                            "amount_micros": evidence.confirmed_cost_micros,
                            "currency": evidence.confirmed_cost_currency.value,
                        },
                        "usage_available": evidence.usage_available,
                        "external_attempts_count": evidence.external_attempts_count,
                        "external_created_at": evidence.external_created_at.isoformat(),
                        "external_completed_at": evidence.external_completed_at.isoformat(),
                        "previous_operation_status": previous_operation_status,
                        "previous_provider_call_status": previous_provider_call_status,
                        "reconciled_at": reconciled_at.isoformat(),
                    },
                )
            )
            return AIReconciliationResult(
                operation_id=operation.id,
                provider_call_id=call.id,
                status="FAILED",
                confirmed_cost_micros=evidence.confirmed_cost_micros,
                confirmed_cost_currency=evidence.confirmed_cost_currency,
            )

    @staticmethod
    def _validate_operation(operation, policy, expected_lead_id, expected_key, evidence) -> None:
        invalid = (
            policy is None
            or operation.operation_type != AIOperationType.QUALIFICATION.value
            or operation.status != AIOperationStatus.UNKNOWN.value
            or operation.lead_id != expected_lead_id
            or operation.idempotency_key != expected_key
            or operation.provider != "polza"
            or operation.model != "lead-structured-v1"
            or operation.cost_currency != Currency.RUB.value
            or policy.budget_currency != Currency.RUB.value
            or policy.qualification_provider != operation.provider
            or policy.qualification_model != operation.model
            or operation.result_analysis_id is not None
            or operation.result_draft_id is not None
            or operation.actual_cost_micros not in {None, 0}
        )
        if invalid:
            raise AIReconciliationError(
                AIReconciliationErrorCode.EVIDENCE_MISMATCH,
                "Operation does not match the supported reconciliation contract",
            )
        if evidence.confirmed_cost_currency.value != operation.cost_currency:
            raise AIReconciliationError(
                AIReconciliationErrorCode.EVIDENCE_MISMATCH,
                "Confirmed cost currency does not match the operation",
            )

    @staticmethod
    def _validate_call(call, operation, evidence) -> None:
        invalid = (
            call.tenant_id != operation.tenant_id
            or call.operation_id != operation.id
            or call.status != ProviderCallStatus.UNKNOWN.value
            or call.attempt != 1
            or call.provider != operation.provider
            or call.model != operation.model
            or call.cost_currency != evidence.confirmed_cost_currency.value
            or call.cost_known
            or call.actual_cost_micros is not None
            or any(
                value is not None
                for value in (call.input_tokens, call.output_tokens, call.cached_tokens)
            )
            or any(
                value is not None
                for value in (
                    call.provider_reported_cost_decimal,
                    call.provider_reported_cost_micros,
                    call.provider_reported_cost_currency,
                )
            )
        )
        if invalid:
            raise AIReconciliationError(
                AIReconciliationErrorCode.EVIDENCE_MISMATCH,
                "Provider call does not match the supported reconciliation contract",
            )
