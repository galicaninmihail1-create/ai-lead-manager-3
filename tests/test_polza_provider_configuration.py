import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from ai_lead_manager.application.ai.contracts import (
    AIOperationType,
    AIUsage,
    ProviderDisabledError,
)
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
POLZA_COMBINED_PRICING_PATH = PROJECT_ROOT / "config" / "polza-pricing-20260914-r1.json"
POLZA_CONSERVATIVE_PRICING_PATH = (
    PROJECT_ROOT / "config" / "polza-pricing-20260914-r2.json"
)
POLZA_PRICING_VERSION = "polza-gpt41mini-rub-20260912-r1"
QWEN_MODEL_ALIAS = "lead-structured-qwen-v1"
QWEN_UPSTREAM_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
QWEN_HISTORICAL_PRICING_VERSION = "polza-qwen3-30ba3b-rub-20260914-r1"
QWEN_PRICING_VERSION = "polza-qwen3-30ba3b-rub-20260914-r2"
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
        qwen_qualification = registry.get(
            "polza", QWEN_MODEL_ALIAS, AIOperationType.QUALIFICATION
        )
        draft = registry.get("polza", "lead-structured-v1", AIOperationType.DRAFT)

        assert isinstance(qualification, OpenAICompatibleQualificationProvider)
        assert isinstance(qwen_qualification, OpenAICompatibleQualificationProvider)
        assert isinstance(draft, OpenAICompatibleDraftProvider)
        assert registry.route(
            AIOperationType.QUALIFICATION, "polza", "lead-structured-v1"
        ).pricing_version == POLZA_PRICING_VERSION
        assert registry.route(
            AIOperationType.QUALIFICATION, "polza", QWEN_MODEL_ALIAS
        ).pricing_version == QWEN_PRICING_VERSION
        assert requests == []


def test_polza_qwen_qualification_route_is_allowlisted_without_expanding_draft() -> None:
    catalog = OpenAICompatibleProviderCatalog.from_json(polza_configuration_json())
    resolved = catalog.resolve(
        "polza", QWEN_MODEL_ALIAS, ProviderCapability.QUALIFICATION
    )

    assert resolved.provider.alias == "polza"
    assert resolved.model.alias == QWEN_MODEL_ALIAS
    assert resolved.model.upstream_model == QWEN_UPSTREAM_MODEL
    assert resolved.model.provider_model_allowlist == {QWEN_UPSTREAM_MODEL}
    assert resolved.model.pricing_version == QWEN_PRICING_VERSION
    assert resolved.model.billing_currency is Currency.RUB
    assert resolved.model.capabilities == {
        ProviderCapability.QUALIFICATION,
        ProviderCapability.STRICT_STRUCTURED_OUTPUT,
    }
    with pytest.raises(ProviderDisabledError):
        catalog.resolve("polza", QWEN_MODEL_ALIAS, ProviderCapability.DRAFT)


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


def test_qwen_historical_pricing_snapshot_remains_immutable() -> None:
    settings = Settings(
        _env_file=None,
        ai_model_pricing_json=POLZA_COMBINED_PRICING_PATH.read_text(encoding="utf-8"),
    )
    qwen = build_pricing_catalog(settings).get("polza", QWEN_MODEL_ALIAS)

    assert qwen.version == QWEN_HISTORICAL_PRICING_VERSION
    assert qwen.input_price_micros_per_million_tokens == 5_677_945
    assert qwen.output_price_micros_per_million_tokens == 22_764_843
    assert qwen.cached_price_micros_per_million_tokens == 5_677_945


def test_qwen_pricing_snapshot_covers_observed_polza_cost_and_budget() -> None:
    settings = Settings(
        _env_file=None,
        ai_model_pricing_json=POLZA_CONSERVATIVE_PRICING_PATH.read_text(
            encoding="utf-8"
        ),
    )
    catalog = build_pricing_catalog(settings)
    qwen = catalog.get("polza", QWEN_MODEL_ALIAS)
    old = catalog.get("polza", "lead-structured-v1")
    route = next(
        model for model in polza_provider_config().models if model.alias == QWEN_MODEL_ALIAS
    )

    assert qwen.version == route.pricing_version == QWEN_PRICING_VERSION
    assert qwen.currency is Currency.RUB
    assert qwen.input_price_micros_per_million_tokens == 9_482_167
    assert qwen.output_price_micros_per_million_tokens == 38_017_287
    assert qwen.cached_price_micros_per_million_tokens == 9_482_167
    assert qwen.actual_cost(
        AIUsage(input_tokens=217, output_tokens=100, cached_tokens=0)
    ) == Money(amount_micros=5_860, currency=Currency.RUB)
    assert qwen.worst_case_cost(647, 400, 1) == Money(
        amount_micros=21_342, currency=Currency.RUB
    )
    assert qwen.worst_case_cost(1_000, 400, 1).amount_micros < 1_000_000
    assert old.version == POLZA_PRICING_VERSION
    assert old.input_price_micros_per_million_tokens == 48_540_000
    assert old.output_price_micros_per_million_tokens == 194_160_000


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
