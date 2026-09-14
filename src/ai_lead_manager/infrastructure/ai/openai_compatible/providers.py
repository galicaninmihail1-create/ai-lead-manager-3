import re
from typing import Any

from pydantic import ValidationError

from ai_lead_manager.application.ai.context import estimate_structured_tokens
from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIExecutionMode,
    AIOperationType,
    AIProviderError,
    StructuredAIRequest,
    StructuredAIResult,
)
from ai_lead_manager.application.drafts import DraftContext, DraftResult
from ai_lead_manager.application.providers import QualificationContext, StructuredQualification
from ai_lead_manager.infrastructure.ai.openai_compatible.config import (
    ResolvedOpenAICompatibleModel,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.parsing import (
    parse_structured_chat_completion,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.prompts import (
    DraftPromptBuilder,
    PromptBundle,
    QualificationPromptBuilder,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.transport import (
    OpenAICompatibleHTTPTransport,
)


class _OpenAICompatibleStructuredProvider:
    execution_mode = AIExecutionMode.LIVE
    schema_version = "1"

    def __init__(
        self,
        resolved: ResolvedOpenAICompatibleModel,
        transport: OpenAICompatibleHTTPTransport,
    ) -> None:
        self.provider_name = resolved.provider.alias
        self.model_name = resolved.model.alias
        self.upstream_model = resolved.model.upstream_model
        self.provider_model_allowlist = resolved.model.provider_model_allowlist
        self.pricing_version = resolved.model.pricing_version
        self.require_parameters = resolved.provider.require_parameters
        self.billing_currency = (
            resolved.model.billing_currency.value
            if resolved.model.billing_currency is not None
            else None
        )
        self.credential_configured = False
        self._transport = transport

    def _request_payload(
        self,
        request: StructuredAIRequest,
        prompt: PromptBundle,
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if request.prompt_version != prompt.version:
            raise AIProviderError(
                AIErrorKind.INVALID_REQUEST,
                "AI request prompt version does not match the provider adapter",
            )
        if request.schema_version != self.schema_version:
            raise AIProviderError(
                AIErrorKind.INVALID_REQUEST,
                "AI request schema version does not match the provider adapter",
            )
        payload = {
            "max_tokens": request.max_output_tokens,
            "messages": prompt.as_messages(),
            "model": self.upstream_model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
            "temperature": 0,
        }
        if self.require_parameters:
            payload["provider"] = {"require_parameters": True}
        return payload

    def _validate_provider_model(
        self, result: StructuredAIResult
    ) -> StructuredAIResult:
        if result.metadata.provider_model not in self.provider_model_allowlist:
            raise AIProviderError(
                AIErrorKind.UNKNOWN_COST,
                "Provider returned a model outside the server allowlist",
                usage=result.usage,
                provider_request_id=result.metadata.provider_request_id,
                provider_model=result.metadata.provider_model,
                finish_reason=result.metadata.finish_reason,
                latency_ms=result.latency_ms,
                ambiguous=True,
                cost_known=False,
                provider_reported_cost=result.provider_reported_cost,
                exact_provider_reported_cost=result.exact_provider_reported_cost,
            )
        return result


class OpenAICompatibleQualificationProvider(_OpenAICompatibleStructuredProvider):
    operation_type = AIOperationType.QUALIFICATION
    operation_type = AIOperationType.QUALIFICATION
    prompt_builder = QualificationPromptBuilder()
    prompt_version = prompt_builder.version

    def _qualification_payload(self, request: StructuredAIRequest) -> dict[str, Any]:
        if request.operation_type != AIOperationType.QUALIFICATION:
            raise AIProviderError(
                AIErrorKind.INVALID_REQUEST,
                "Qualification provider received the wrong operation type",
            )
        try:
            context = QualificationContext.model_validate(request.payload)
        except ValidationError as error:
            raise AIProviderError(
                AIErrorKind.INVALID_REQUEST,
                "Qualification context failed validation",
            ) from error
        prompt = self.prompt_builder.build(context)
        payload = self._request_payload(
            request,
            prompt,
            schema_name="structured_qualification_v1",
            schema=StructuredQualification.model_json_schema(),
        )
        return payload

    def estimate_input_tokens(self, request: StructuredAIRequest) -> int:
        return max(
            request.estimated_input_tokens,
            estimate_structured_tokens(self._qualification_payload(request)),
        )

    def execute(self, request: StructuredAIRequest) -> StructuredAIResult:
        payload = self._qualification_payload(request)
        wire = self._transport.post_chat_completion(payload)
        return self._validate_provider_model(
            parse_structured_chat_completion(
                wire,
                provider_alias=self.provider_name,
                model_alias=self.model_name,
                result_type=StructuredQualification,
            )
        )


class OpenAICompatibleDraftProvider(_OpenAICompatibleStructuredProvider):
    operation_type = AIOperationType.DRAFT
    operation_type = AIOperationType.DRAFT
    prompt_builder = DraftPromptBuilder()
    prompt_version = prompt_builder.version

    def execute(self, request: StructuredAIRequest) -> StructuredAIResult:
        if request.operation_type != AIOperationType.DRAFT:
            raise AIProviderError(
                AIErrorKind.INVALID_REQUEST,
                "Draft provider received the wrong operation type",
            )
        try:
            context = DraftContext.model_validate(request.payload)
        except ValidationError as error:
            raise AIProviderError(
                AIErrorKind.INVALID_REQUEST,
                "Draft context failed validation",
            ) from error
        prompt = self.prompt_builder.build(context)
        payload = self._request_payload(
            request,
            prompt,
            schema_name="response_draft_v1",
            schema=DraftResult.model_json_schema(),
        )
        wire = self._transport.post_chat_completion(payload)
        result = self._validate_provider_model(
            parse_structured_chat_completion(
                wire,
                provider_alias=self.provider_name,
                model_alias=self.model_name,
                result_type=DraftResult,
            )
        )
        if re.search(r"[А-Яа-яЁё]", result.output["content"]) is None:
            raise AIProviderError(
                AIErrorKind.SCHEMA_VALIDATION,
                "Provider draft is not Russian-language content",
                usage=result.usage,
                provider_request_id=result.metadata.provider_request_id,
                provider_model=result.metadata.provider_model,
                finish_reason=result.metadata.finish_reason,
                latency_ms=result.latency_ms,
                cost_known=result.usage is not None,
                provider_reported_cost=result.provider_reported_cost,
                exact_provider_reported_cost=result.exact_provider_reported_cost,
            )
        return result
