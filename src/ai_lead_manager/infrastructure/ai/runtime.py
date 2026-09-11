import json
from functools import lru_cache

from ai_lead_manager.application.ai.context import estimate_tokens
from ai_lead_manager.application.ai.contracts import (
    AIUsage,
    ProviderMetadata,
    StructuredAIRequest,
    StructuredAIResult,
)
from ai_lead_manager.application.ai.executor import AIExecutor
from ai_lead_manager.application.ai.pricing import default_mock_pricing_catalog
from ai_lead_manager.application.ai.registry import ProviderRegistry
from ai_lead_manager.application.drafts import DraftContext, MockDraftProvider
from ai_lead_manager.application.providers import MockQualificationProvider, QualificationContext
from ai_lead_manager.config import get_settings
from ai_lead_manager.infrastructure.ai.recording import RecordingStructuredProvider
from ai_lead_manager.infrastructure.ai.repositories import SqlAlchemyAIRepositoryFactory
from ai_lead_manager.infrastructure.database import SessionFactory


def _usage(request: StructuredAIRequest, output: dict) -> AIUsage:
    return AIUsage(
        input_tokens=request.estimated_input_tokens,
        output_tokens=estimate_tokens(json.dumps(output, ensure_ascii=False, sort_keys=True)),
    )


def _qualification(request: StructuredAIRequest) -> StructuredAIResult:
    output = MockQualificationProvider().analyze(
        QualificationContext.model_validate(request.payload)
    ).model_dump(mode="json")
    return StructuredAIResult(
        output=output,
        usage=_usage(request, output),
        metadata=ProviderMetadata(
            provider="mock",
            model="deterministic-keyword-v1",
            provider_request_id="offline-qualification",
            finish_reason="stop",
        ),
        latency_ms=0,
    )


def _draft(request: StructuredAIRequest) -> StructuredAIResult:
    output = MockDraftProvider().generate(
        DraftContext.model_validate(request.payload)
    ).model_dump(mode="json")
    return StructuredAIResult(
        output=output,
        usage=_usage(request, output),
        metadata=ProviderMetadata(
            provider="mock",
            model="deterministic-template-v1",
            provider_request_id="offline-draft",
            finish_reason="stop",
        ),
        latency_ms=0,
    )


@lru_cache
def get_provider_registry() -> ProviderRegistry:
    return ProviderRegistry(
        [
            RecordingStructuredProvider(
                "mock", "deterministic-keyword-v1", _qualification
            ),
            RecordingStructuredProvider(
                "mock", "deterministic-template-v1", _draft
            ),
        ]
    )


@lru_cache
def get_ai_executor() -> AIExecutor:
    return AIExecutor(
        get_settings(),
        SqlAlchemyAIRepositoryFactory(SessionFactory),
        get_provider_registry(),
        default_mock_pricing_catalog(),
    )
