import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIProviderError,
    AIUsage,
    ProviderMetadata,
    StructuredAIResult,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.transport import OpenAIWireResponse


class _PromptTokenDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    cached_tokens: int = Field(default=0, ge=0)


class _WireUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    prompt_tokens_details: _PromptTokenDetails | None = None


def _request_id(payload: dict[str, Any], header_request_id: str | None) -> str | None:
    if header_request_id is not None:
        return header_request_id
    candidate = payload.get("id")
    if not isinstance(candidate, str):
        return None
    candidate = candidate.strip()
    if (
        not candidate
        or len(candidate) > 255
        or re.fullmatch(r"[A-Za-z0-9._:-]+", candidate) is None
    ):
        return None
    return candidate


def _usage(
    payload: dict[str, Any], provider_request_id: str | None, latency_ms: int
) -> AIUsage | None:
    raw_usage = payload.get("usage")
    if raw_usage is None:
        return None
    try:
        parsed = _WireUsage.model_validate(raw_usage)
    except ValidationError as error:
        raise AIProviderError(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider returned malformed usage metadata",
            ambiguous=True,
            provider_request_id=provider_request_id,
            latency_ms=latency_ms,
            cost_known=False,
        ) from error
    cached = parsed.prompt_tokens_details.cached_tokens if parsed.prompt_tokens_details else 0
    if cached > parsed.prompt_tokens:
        raise AIProviderError(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider returned inconsistent usage metadata",
            ambiguous=True,
            provider_request_id=provider_request_id,
            latency_ms=latency_ms,
            cost_known=False,
        )
    return AIUsage(
        input_tokens=parsed.prompt_tokens,
        output_tokens=parsed.completion_tokens,
        cached_tokens=cached,
    )


def _output_error(
    kind: AIErrorKind,
    message: str,
    usage: AIUsage | None,
    provider_request_id: str | None,
    *,
    ambiguous: bool = False,
    provider_model: str | None = None,
    finish_reason: str | None = None,
    latency_ms: int | None = None,
) -> AIProviderError:
    return AIProviderError(
        kind,
        message,
        usage=usage,
        provider_request_id=provider_request_id,
        provider_model=provider_model,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
        ambiguous=ambiguous,
        cost_known=usage is not None,
    )


def parse_structured_chat_completion[ResultT: BaseModel](
    wire: OpenAIWireResponse,
    *,
    provider_alias: str,
    model_alias: str,
    result_type: type[ResultT],
) -> StructuredAIResult:
    payload = wire.payload
    provider_request_id = _request_id(payload, wire.provider_request_id)
    usage = _usage(payload, provider_request_id, wire.latency_ms)
    provider_model = payload.get("model")
    if not isinstance(provider_model, str) or not provider_model.strip():
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider response omitted its model identifier",
            usage,
            provider_request_id,
            latency_ms=wire.latency_ms,
        )

    choices = payload.get("choices")
    normalized_provider_model = provider_model.strip()
    if not isinstance(choices, list) or not choices:
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider returned an empty or invalid choices collection",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            latency_ms=wire.latency_ms,
        )
    if len(choices) != 1:
        raise _output_error(
            AIErrorKind.AMBIGUOUS_RESULT,
            "Provider returned multiple structured outputs",
            usage,
            provider_request_id,
            ambiguous=True,
            provider_model=normalized_provider_model,
            latency_ms=wire.latency_ms,
        )
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("index") not in {0, None}:
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider returned an unexpected choice shape",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            latency_ms=wire.latency_ms,
        )
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider response omitted its finish reason",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            latency_ms=wire.latency_ms,
        )
    if finish_reason == "length":
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider output was truncated",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        )

    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider returned an unexpected message shape",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        )
    refusal = message.get("refusal")
    refused = refusal is not None and refusal != "" and refusal is not False
    if finish_reason == "content_filter" or refused:
        raise _output_error(
            AIErrorKind.INVALID_REQUEST,
            "Provider refused the structured request",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        )
    if finish_reason != "stop":
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider returned an unsupported finish reason",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider returned empty structured content",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        )
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider structured content was not valid JSON",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        ) from error
    if not isinstance(decoded, dict):
        raise _output_error(
            AIErrorKind.MALFORMED_OUTPUT,
            "Provider structured content was not a JSON object",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        )
    try:
        validated = result_type.model_validate(decoded)
    except ValidationError as error:
        raise _output_error(
            AIErrorKind.SCHEMA_VALIDATION,
            "Provider structured content failed schema validation",
            usage,
            provider_request_id,
            provider_model=normalized_provider_model,
            finish_reason=finish_reason,
            latency_ms=wire.latency_ms,
        ) from error

    return StructuredAIResult(
        output=validated.model_dump(mode="json"),
        usage=usage,
        metadata=ProviderMetadata(
            provider=provider_alias,
            model=model_alias,
            provider_model=normalized_provider_model,
            provider_request_id=provider_request_id,
            finish_reason=finish_reason,
        ),
        latency_ms=wire.latency_ms,
    )
