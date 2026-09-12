import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_live_ai_integration import FakeOpenAIHTTP, live_settings, runtime_for, seed_live_lead

from ai_lead_manager.application.ai.contracts import AIOperationType
from ai_lead_manager.application.ai.readiness import LiveAIReadinessService
from ai_lead_manager.application.api_keys import APIKeyScope
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.domain.models import AIOperation, LeadAnalysis, ProviderCall, TenantAIPolicy


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
    assert result.estimated_max_cost_microusd == expected
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
