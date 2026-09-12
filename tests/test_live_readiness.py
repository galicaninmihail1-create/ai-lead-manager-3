import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from test_live_ai_integration import FakeOpenAIHTTP, live_settings, runtime_for, seed_live_lead

from ai_lead_manager.application.ai.contracts import AIOperationType
from ai_lead_manager.application.ai.readiness import LiveAIReadinessService
from ai_lead_manager.application.api_keys import APIKeyScope
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.domain.models import AIOperation, LeadAnalysis, ProviderCall, TenantAIPolicy
from ai_lead_manager.infrastructure.ai.composition import build_ai_executor
from ai_lead_manager.infrastructure.ai.openai_compatible.credentials import (
    EnvironmentCredentialResolver,
)

ROOT = Path(__file__).resolve().parents[1]
POLZA_PROVIDER = "polza"
POLZA_MODEL = "lead-structured-v1"


def polza_runtime(
    session: Session,
    environment: dict[str, str],
    *,
    pricing_version: str | None = None,
    **overrides,
):
    pricing_json = (
        ROOT / "config" / "polza-pricing-20260912-r1.json"
    ).read_text(encoding="utf-8")
    if pricing_version is not None:
        pricing_json = pricing_json.replace(
            "polza-gpt41mini-rub-20260912-r1", pricing_version
        )
    values = {
        "ai_execution_mode": "LIVE",
        "allow_live_ai": True,
        "ai_openai_compatible_providers_json": (
            ROOT / "config" / "polza-provider.json"
        ).read_text(encoding="utf-8"),
        "ai_model_pricing_json": pricing_json,
    }
    values.update(overrides)
    settings = live_settings(**values)
    created_clients = []

    def client_factory(_config):
        created_clients.append(True)
        raise AssertionError("readiness must not create an HTTP client")

    executor = build_ai_executor(
        settings,
        sessionmaker(bind=session.get_bind(), expire_on_commit=False),
        credential_resolver=EnvironmentCredentialResolver(environment),
        client_factory=client_factory,
    )
    return settings, executor, created_clients


def seed_polza_policy(session: Session, slug: str) -> tuple[uuid.UUID, uuid.UUID]:
    tenant_id, lead_id = seed_live_lead(
        session, slug=slug, budget_currency="RUB", monthly_limit=1_000_000_000
    )
    policy = session.get(TenantAIPolicy, tenant_id)
    policy.qualification_provider = POLZA_PROVIDER
    policy.qualification_model = POLZA_MODEL
    policy.draft_provider = POLZA_PROVIDER
    policy.draft_model = POLZA_MODEL
    session.commit()
    return tenant_id, lead_id


def codes(result) -> set[str]:
    return {blocker.code for blocker in result.blockers}


def assert_zero_side_effects(session: Session, fake_http: FakeOpenAIHTTP) -> None:
    assert session.scalar(select(func.count()).select_from(AIOperation)) == 0
    assert session.scalar(select(func.count()).select_from(ProviderCall)) == 0
    assert fake_http.call_count == 0


def test_ready_qualification_exact_preview_and_zero_side_effects(session: Session) -> None:
    fake_http = FakeOpenAIHTTP()
    settings = live_settings()
    executor = runtime_for(session, fake_http, settings)
    tenant_id, lead_id = seed_live_lead(session, slug="readiness-qualification")
    result = LiveAIReadinessService(session, executor, settings).check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )

    price = executor.pricing.get(result.provider_alias, result.model_alias)
    policy = session.get(TenantAIPolicy, tenant_id)
    expected = price.worst_case_cost(
        result.estimated_input_tokens,
        min(policy.max_output_tokens, settings.ai_max_output_tokens),
        min(policy.max_attempts, settings.ai_max_attempts),
    )
    assert result.ready is True
    assert result.estimated_max_cost == expected
    assert result.estimated_max_cost.currency.value == "USD"
    assert result.remaining_monthly_budget.currency.value == "USD"
    assert result.per_call_limit.currency.value == "USD"
    assert result.credential_configured is True
    assert_zero_side_effects(session, fake_http)


def test_ready_draft_and_operation_scope(session: Session) -> None:
    fake_http = FakeOpenAIHTTP()
    settings = live_settings()
    executor = runtime_for(session, fake_http, settings)
    tenant_id, lead_id = seed_live_lead(session, slug="readiness-draft")
    session.add(
        LeadAnalysis(
            tenant_id=tenant_id,
            lead_id=lead_id,
            provider="setup",
            model="setup",
            schema_version="1",
            prompt_version="1",
            task="Автоматизация заявок",
            pain_points=[],
            objections=[],
            ai_summary="Требуется подготовить ответ",
        )
    )
    session.commit()
    service = LiveAIReadinessService(session, executor, settings)

    ready = service.check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.DRAFT,
        lead_id,
        frozenset({APIKeyScope.AI_DRAFT}),
    )
    denied = service.check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.DRAFT,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )
    assert ready.ready is True and ready.required_scope == "ai:draft"
    assert "missing_scope" in codes(denied)
    assert_zero_side_effects(session, fake_http)


def test_readiness_returns_explicit_rub_money(session: Session) -> None:
    fake_http = FakeOpenAIHTTP()
    settings = live_settings(pricing_currency="RUB")
    executor = runtime_for(session, fake_http, settings)
    tenant_id, lead_id = seed_live_lead(
        session,
        slug="readiness-rub",
        budget_currency="RUB",
    )

    result = LiveAIReadinessService(session, executor, settings).check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )

    assert result.ready is True
    assert result.model_dump(mode="json")["estimated_max_cost"] == {
        "amount_micros": result.estimated_max_cost.amount_micros,
        "currency": "RUB",
    }
    assert result.remaining_monthly_budget.currency.value == "RUB"
    assert result.per_call_limit.currency.value == "RUB"
    assert_zero_side_effects(session, fake_http)


def test_collects_safe_blockers_budget_boundaries_and_cross_tenant(session: Session) -> None:
    fake_http = FakeOpenAIHTTP()
    settings = live_settings(ai_kill_switch=True, allow_live_ai=False)
    executor = runtime_for(session, fake_http, settings)
    tenant_id, lead_id = seed_live_lead(
        session, slug="readiness-blockers", per_call_limit=0
    )
    other_tenant, _ = seed_live_lead(session, slug="readiness-other")
    service = LiveAIReadinessService(session, executor, settings)
    result = service.check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset(),
    )
    foreign = service.check(
        TenantContext.trusted_context(other_tenant),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )
    assert {"missing_scope", "live_disabled", "kill_switch_active", "per_call_limit_exceeded"}.issubset(codes(result))
    assert "entity_not_found" in codes(foreign)
    assert_zero_side_effects(session, fake_http)


def test_untrusted_missing_and_disabled_policy(session: Session) -> None:
    fake_http = FakeOpenAIHTTP()
    settings = live_settings()
    executor = runtime_for(session, fake_http, settings)
    tenant_id, lead_id = seed_live_lead(session, slug="readiness-policy")
    service = LiveAIReadinessService(session, executor, settings)
    untrusted = service.check(
        TenantContext.untrusted(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )
    session.get(TenantAIPolicy, tenant_id).enabled = False
    session.commit()
    disabled = service.check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )
    missing = service.check(
        TenantContext.trusted_context(uuid.uuid4()),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )
    assert codes(untrusted) == {"untrusted_tenant"}
    assert "policy_disabled" in codes(disabled)
    assert codes(missing) == {"policy_missing"}
    assert_zero_side_effects(session, fake_http)


@pytest.mark.parametrize("environment", [{}, {"POLZA_AI_API_KEY": "   "}])
def test_polza_readiness_blocks_missing_or_blank_credential_without_side_effects(
    session: Session, environment: dict[str, str]
) -> None:
    settings, executor, created_clients = polza_runtime(session, environment)
    tenant_id, lead_id = seed_polza_policy(session, f"polza-missing-{len(environment)}")

    result = LiveAIReadinessService(session, executor, settings).check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )

    assert result.ready is False
    assert result.credential_configured is False
    assert "credential_missing" in codes(result)
    assert created_clients == []
    assert session.scalar(select(func.count()).select_from(AIOperation)) == 0
    assert session.scalar(select(func.count()).select_from(ProviderCall)) == 0


def test_polza_readiness_is_safe_ready_preview_with_rub_costs(session: Session) -> None:
    marker = "configured-placeholder"
    settings, executor, created_clients = polza_runtime(
        session, {"POLZA_AI_API_KEY": marker}
    )
    tenant_id, lead_id = seed_polza_policy(session, "polza-ready-preview")

    result = LiveAIReadinessService(session, executor, settings).check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )

    serialized = result.model_dump_json()
    assert result.ready is True
    assert result.credential_configured is True
    assert result.estimated_max_cost.currency.value == "RUB"
    assert result.remaining_monthly_budget.currency.value == "RUB"
    assert marker not in serialized
    assert created_clients == []
    assert session.scalar(select(func.count()).select_from(AIOperation)) == 0
    assert session.scalar(select(func.count()).select_from(ProviderCall)) == 0


def test_polza_readiness_blocks_pricing_version_and_currency_mismatch(
    session: Session,
) -> None:
    settings, executor, created_clients = polza_runtime(
        session,
        {"POLZA_AI_API_KEY": "configured-placeholder"},
        pricing_version="wrong-version",
    )
    tenant_id, lead_id = seed_polza_policy(session, "polza-pricing-mismatch")
    wrong_version = LiveAIReadinessService(session, executor, settings).check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )

    valid_settings, valid_executor, valid_created_clients = polza_runtime(
        session, {"POLZA_AI_API_KEY": "configured-placeholder"}
    )
    session.get(TenantAIPolicy, tenant_id).budget_currency = "USD"
    session.commit()
    currency = LiveAIReadinessService(session, valid_executor, valid_settings).check(
        TenantContext.trusted_context(tenant_id),
        AIOperationType.QUALIFICATION,
        lead_id,
        frozenset({APIKeyScope.AI_QUALIFY}),
    )

    assert "pricing_unknown" in codes(wrong_version)
    assert "currency_mismatch" in codes(currency)
    assert created_clients == []
    assert valid_created_clients == []
    assert session.scalar(select(func.count()).select_from(AIOperation)) == 0
    assert session.scalar(select(func.count()).select_from(ProviderCall)) == 0
