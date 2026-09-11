import re
from typing import Any

from pydantic import ValidationError

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
        return {
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

    def _validate_provider_model(
        self, result: StructuredAIResult
    ) -> StructuredAIResult:
        if result.metadata.provider_model not in self.provider_model_allowlist:
            raise AIProviderError(
                AIErrorKind.UNKNOWN_COST,
                "Provider returned a model outside the server allowlist",
                usage=result.usage,
                provider_request_id=result.metadata.provider_request_id,
                ambiguous=True,
                cost_known=False,
            )
        return result


class OpenAICompatibleQualificationProvider(_OpenAICompatibleStructuredProvider):
    prompt_builder = QualificationPromptBuilder()
    prompt_version = prompt_builder.version

    def execute(self, request: StructuredAIRequest) -> StructuredAIResult:
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
                cost_known=result.usage is not None,
            )
        return result
