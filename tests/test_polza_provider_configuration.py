import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from ai_lead_manager.application.ai.contracts import AIOperationType, AIUsage
from ai_lead_manager.application.ai.money import Currency, Money
from ai_lead_manager.config import Settings
from ai_lead_manager.infrastructure.ai.composition import (
    build_pricing_catalog,
    build_provider_registry,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.config import (
    OpenAICompatibleProviderCatalog,
    OpenAICompatibleProviderConfig,
    ProviderCapability,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.credentials import (
    EnvironmentCredentialResolver,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.providers import (
    OpenAICompatibleDraftProvider,
    OpenAICompatibleQualificationProvider,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.transport import (
    OpenAICompatibleHTTPTransport,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLZA_CONFIGURATION_PATH = PROJECT_ROOT / "config" / "polza-provider.json"
POLZA_PRICING_PATH = PROJECT_ROOT / "config" / "polza-pricing-20260912-r1.json"
POLZA_PRICING_VERSION = "polza-gpt41mini-rub-20260912-r1"
OFFLINE_TEST_SECRET = "offline-polza-registration-test-secret"


def polza_configuration_json() -> str:
    return POLZA_CONFIGURATION_PATH.read_text(encoding="utf-8")


def polza_provider_config() -> OpenAICompatibleProviderConfig:
    catalog = OpenAICompatibleProviderCatalog.from_json(polza_configuration_json())
    assert len(catalog.providers) == 1
    return catalog.providers[0]


def test_polza_configuration_keeps_identities_separate() -> None:
    catalog = OpenAICompatibleProviderCatalog.from_json(polza_configuration_json())
    qualification = catalog.resolve(
        "polza", "lead-structured-v1", ProviderCapability.QUALIFICATION
    )
    draft = catalog.resolve("polza", "lead-structured-v1", ProviderCapability.DRAFT)

    assert qualification == draft
    assert qualification.provider.alias == "polza"
    assert qualification.model.alias == "lead-structured-v1"
    assert qualification.model.upstream_model == "openai/gpt-4.1-mini"
    assert qualification.provider.api_key_env_var == "POLZA_AI_API_KEY"
    assert not hasattr(qualification.provider, "api_key")
    assert qualification.model.capabilities == {
        ProviderCapability.QUALIFICATION,
        ProviderCapability.DRAFT,
        ProviderCapability.STRICT_STRUCTURED_OUTPUT,
    }
    assert OFFLINE_TEST_SECRET not in json.dumps(
        qualification.model_dump(mode="json"), sort_keys=True
    )


def test_polza_registers_existing_adapters_without_http() -> None:
    requests: list[httpx.Request] = []

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, request=request)

    settings = Settings(
        _env_file=None,
        ai_openai_compatible_providers_json=polza_configuration_json(),
    )

    with httpx.Client(
        transport=httpx.MockTransport(unexpected_request),
        trust_env=False,
    ) as client:
        registry = build_provider_registry(
            settings,
            credential_resolver=EnvironmentCredentialResolver(
                {"POLZA_AI_API_KEY": OFFLINE_TEST_SECRET}
            ),
            client_factory=lambda _config: client,
        )

        qualification = registry.get(
            "polza", "lead-structured-v1", AIOperationType.QUALIFICATION
        )
        draft = registry.get("polza", "lead-structured-v1", AIOperationType.DRAFT)

        assert isinstance(qualification, OpenAICompatibleQualificationProvider)
        assert isinstance(draft, OpenAICompatibleDraftProvider)
        assert registry.route(
            AIOperationType.QUALIFICATION, "polza", "lead-structured-v1"
        ).pricing_version == POLZA_PRICING_VERSION
        assert requests == []


def test_polza_immutable_pricing_snapshot_matches_route_and_exact_rub_rates() -> None:
    settings = Settings(
        _env_file=None,
        ai_model_pricing_json=POLZA_PRICING_PATH.read_text(encoding="utf-8"),
    )
    price = build_pricing_catalog(settings).get("polza", "lead-structured-v1")
    route = polza_provider_config().models[0]

    assert price.version == route.pricing_version == POLZA_PRICING_VERSION
    assert price.currency is Currency.RUB
    assert price.input_price_micros_per_million_tokens == 48_540_000
    assert price.output_price_micros_per_million_tokens == 194_160_000
    assert price.cached_price_micros_per_million_tokens == 48_540_000
    assert price.actual_cost(AIUsage(input_tokens=1_000_000, output_tokens=0)) == Money(
        amount_micros=48_540_000, currency=Currency.RUB
    )
    assert price.actual_cost(AIUsage(input_tokens=0, output_tokens=1_000_000)) == Money(
        amount_micros=194_160_000, currency=Currency.RUB
    )


@pytest.mark.parametrize(
    "base_url",
    ["https://polza.ai/api", "https://polza.ai/api/"],
)
def test_polza_chat_completions_url_is_canonical(base_url: str) -> None:
    raw_config = polza_provider_config().model_dump(mode="json")
    raw_config["base_url"] = base_url
    config = OpenAICompatibleProviderConfig.model_validate(raw_config)

    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("URL inspection must not issue an HTTP request")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        transport = OpenAICompatibleHTTPTransport(
            config,
            SecretStr(OFFLINE_TEST_SECRET),
            client=client,
        )

        assert transport.chat_completions_url == (
            "https://polza.ai/api/v1/chat/completions"
        )


def test_chat_completions_path_rejects_duplicate_slashes() -> None:
    raw_config = polza_provider_config().model_dump(mode="json")
    raw_config["chat_completions_path"] = "//v1/chat/completions"

    with pytest.raises(ValueError, match="duplicate slashes"):
        OpenAICompatibleProviderConfig.model_validate(raw_config)
