import json
from collections.abc import Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from ai_lead_manager.application.ai.context import estimate_tokens
from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIOperationType,
    AIUsage,
    ProviderMetadata,
    StructuredAIRequest,
    StructuredAIResult,
    UnknownCostError,
)
from ai_lead_manager.application.ai.executor import AIExecutor
from ai_lead_manager.application.ai.pricing import (
    ModelPricing,
    PricingCatalog,
    default_mock_pricing_catalog,
)
from ai_lead_manager.application.ai.registry import ProviderRegistry
from ai_lead_manager.application.drafts import DraftContext, MockDraftProvider
from ai_lead_manager.application.providers import (
    MockQualificationProvider,
    QualificationContext,
)
from ai_lead_manager.config import Settings
from ai_lead_manager.infrastructure.ai.openai_compatible.config import (
    OpenAICompatibleProviderCatalog,
    OpenAICompatibleProviderConfig,
    ProviderCapability,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.credentials import (
    CredentialResolver,
    EnvironmentCredentialResolver,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.factory import (
    OpenAICompatibleProviderFactory,
)
from ai_lead_manager.infrastructure.ai.recording import RecordingStructuredProvider
from ai_lead_manager.infrastructure.ai.repositories import SqlAlchemyAIRepositoryFactory

ClientFactory = Callable[[OpenAICompatibleProviderConfig], httpx.Client | None]


class ServerModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9._-]+$")
    model: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9._-]+$")
    version: str = Field(min_length=1, max_length=50)
    input_microusd_per_million_tokens: int = Field(ge=0)
    output_microusd_per_million_tokens: int = Field(ge=0)
    cached_microusd_per_million_tokens: int = Field(ge=0)

    def to_domain(self) -> ModelPricing:
        return ModelPricing(**self.model_dump())


def parse_server_pricing(raw_configuration: str) -> list[ModelPricing]:
    try:
        raw = json.loads(raw_configuration)
        if not isinstance(raw, list):
            raise ValueError("Pricing configuration must be a JSON list")
        return [ServerModelPricing.model_validate(item).to_domain() for item in raw]
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
        raise UnknownCostError(
            AIErrorKind.UNKNOWN_COST,
            "Server-controlled AI pricing configuration is invalid",
        ) from error


def _usage(request: StructuredAIRequest, output: dict) -> AIUsage:
    return AIUsage(
        input_tokens=request.estimated_input_tokens,
        output_tokens=estimate_tokens(json.dumps(output, ensure_ascii=False, sort_keys=True)),
    )


def _mock_qualification(request: StructuredAIRequest) -> StructuredAIResult:
    output = MockQualificationProvider().analyze(
        QualificationContext.model_validate(request.payload)
    ).model_dump(mode="json")
    return StructuredAIResult(
        output=output,
        usage=_usage(request, output),
        metadata=ProviderMetadata(
            provider="mock",
            model="deterministic-keyword-v1",
            provider_model="deterministic-keyword-v1",
            provider_request_id="offline-qualification",
            finish_reason="stop",
        ),
        latency_ms=0,
    )


def _mock_draft(request: StructuredAIRequest) -> StructuredAIResult:
    output = MockDraftProvider().generate(
        DraftContext.model_validate(request.payload)
    ).model_dump(mode="json")
    return StructuredAIResult(
        output=output,
        usage=_usage(request, output),
        metadata=ProviderMetadata(
            provider="mock",
            model="deterministic-template-v1",
            provider_model="deterministic-template-v1",
            provider_request_id="offline-draft",
            finish_reason="stop",
        ),
        latency_ms=0,
    )


def build_provider_registry(
    settings: Settings,
    *,
    credential_resolver: CredentialResolver | None = None,
    client_factory: ClientFactory | None = None,
) -> ProviderRegistry:
    providers = [
        RecordingStructuredProvider(
            "mock",
            "deterministic-keyword-v1",
            _mock_qualification,
            operation_type=AIOperationType.QUALIFICATION,
            prompt_version=MockQualificationProvider.prompt_version,
            schema_version=MockQualificationProvider.schema_version,
            pricing_version="mock-pricing-v1",
        ),
        RecordingStructuredProvider(
            "mock",
            "deterministic-template-v1",
            _mock_draft,
            operation_type=AIOperationType.DRAFT,
            prompt_version=MockDraftProvider.prompt_version,
            schema_version="1",
            pricing_version="mock-pricing-v1",
        ),
    ]
    catalog = OpenAICompatibleProviderCatalog.from_json(
        settings.ai_openai_compatible_providers_json
    )
    factory = OpenAICompatibleProviderFactory(
        catalog,
        credential_resolver or EnvironmentCredentialResolver(),
        client_factory=client_factory,
    )
    for provider_config in catalog.providers:
        for model in provider_config.models:
            if ProviderCapability.QUALIFICATION in model.capabilities:
                providers.append(factory.qualification(provider_config.alias, model.alias))
            if ProviderCapability.DRAFT in model.capabilities:
                providers.append(factory.draft(provider_config.alias, model.alias))
    return ProviderRegistry(providers)


def build_pricing_catalog(settings: Settings) -> PricingCatalog:
    return default_mock_pricing_catalog().combined_with(
        parse_server_pricing(settings.ai_model_pricing_json)
    )


def build_ai_executor(
    settings: Settings,
    session_factory: sessionmaker[Session],
    *,
    credential_resolver: CredentialResolver | None = None,
    client_factory: ClientFactory | None = None,
) -> AIExecutor:
    return AIExecutor(
        settings,
        SqlAlchemyAIRepositoryFactory(session_factory),
        build_provider_registry(
            settings,
            credential_resolver=credential_resolver,
            client_factory=client_factory,
        ),
        build_pricing_catalog(settings),
    )
