import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_lead_manager.application.ai.context import (
    BoundedContextBuilder,
    estimate_structured_tokens,
)
from ai_lead_manager.application.ai.contracts import AIOperationType
from ai_lead_manager.application.ai.executor import AIExecutor
from ai_lead_manager.application.ai.money import Currency, Money
from ai_lead_manager.application.api_keys import APIKeyScope
from ai_lead_manager.application.drafts import DraftContext
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.config import Settings
from ai_lead_manager.domain.models import (
    AIOperation,
    Customer,
    Lead,
    LeadAnalysis,
    Message,
    TenantAIPolicy,
)


class ReadinessBlocker(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str
    description: str


class LiveAIReadinessResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    ready: bool
    blockers: list[ReadinessBlocker]
    provider_alias: str | None = None
    model_alias: str | None = None
    pricing_version: str | None = None
    credential_configured: bool = False
    estimated_max_cost: Money | None = None
    remaining_monthly_budget: Money | None = None
    per_call_limit: Money | None = None
    required_scope: str
    estimated_input_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    idempotency_required: bool = True


class LiveAIReadinessService:
    def __init__(self, session: Session, executor: AIExecutor, settings: Settings) -> None:
        self.session = session
        self.executor = executor
        self.settings = settings

    def check(
        self,
        tenant: TenantContext,
        operation: AIOperationType,
        lead_id: uuid.UUID,
        granted_scopes: frozenset[APIKeyScope],
    ) -> LiveAIReadinessResult:
        required = (
            APIKeyScope.AI_QUALIFY
            if operation == AIOperationType.QUALIFICATION
            else APIKeyScope.AI_DRAFT
        )
        blockers: list[ReadinessBlocker] = []

        def block(code: str, description: str) -> None:
            blockers.append(ReadinessBlocker(code=code, description=description))

        if not tenant.trusted:
            block("untrusted_tenant", "Trusted tenant authentication is required")
            return LiveAIReadinessResult(ready=False, blockers=blockers, required_scope=required)
        if required not in granted_scopes:
            block("missing_scope", "The authenticated key lacks the required operation scope")

        policy = self.session.get(TenantAIPolicy, tenant.tenant_id)
        if policy is None:
            block("policy_missing", "Tenant AI policy is missing")
            return LiveAIReadinessResult(ready=False, blockers=blockers, required_scope=required)
        if not policy.enabled:
            block("policy_disabled", "Tenant AI policy is disabled")
        if policy.mode != "LIVE":
            block("live_disabled", "Tenant policy does not permit LIVE execution")
        if self.settings.ai_execution_mode != "LIVE" or not self.settings.allow_live_ai:
            block("live_disabled", "Global LIVE execution is disabled")
        if self.settings.ai_kill_switch:
            block("kill_switch_active", "Global AI kill switch is active")

        try:
            budget_currency = Currency(policy.budget_currency)
        except ValueError:
            budget_currency = None
            block("currency_mismatch", "Tenant AI budget currency is unsupported")

        provider, model = (
            (policy.qualification_provider, policy.qualification_model)
            if operation == AIOperationType.QUALIFICATION
            else (policy.draft_provider, policy.draft_model)
        )
        route = None
        try:
            route = self.executor.providers.route(operation, provider, model)
            if route.execution_mode.value != "LIVE":
                block("provider_not_allowed", "Provider route is not LIVE-enabled")
        except Exception:
            block("provider_not_allowed", "Provider/model/operation route is not allowlisted")

        price = None
        try:
            price = self.executor.pricing.get(provider, model)
            if route is not None and route.pricing_version != price.version:
                block("pricing_unknown", "Pricing version does not match the provider route")
                price = None
            elif budget_currency is None or price.currency != budget_currency:
                block(
                    "currency_mismatch",
                    "Model pricing currency does not match the tenant AI budget currency",
                )
                price = None
        except Exception:
            block("pricing_unknown", "Server-controlled model pricing is unavailable")

        lead = self.session.scalar(
            select(Lead).where(Lead.id == lead_id, Lead.tenant_id == tenant.tenant_id)
        )
        estimated_tokens = None
        if lead is None:
            block("entity_not_found", "Target lead was not found in the authenticated tenant")
        elif operation == AIOperationType.QUALIFICATION:
            messages = list(
                self.session.scalars(
                    select(Message.body)
                    .where(
                        Message.tenant_id == tenant.tenant_id,
                        Message.conversation_id == lead.conversation_id,
                        Message.direction == "inbound",
                    )
                    .order_by(Message.created_at, Message.id)
                )
            )
            try:
                bounded = BoundedContextBuilder(
                    self.settings.ai_max_messages,
                    self.settings.ai_max_input_chars,
                    self.settings.ai_max_input_tokens,
                ).build(messages)
                estimated_tokens = bounded.estimated_tokens
                if not bounded.messages:
                    block("context_too_large", "No usable bounded qualification context")
            except Exception:
                block("context_too_large", "Qualification context cannot be bounded")
        else:
            customer = self.session.scalar(
                select(Customer).where(
                    Customer.id == lead.customer_id, Customer.tenant_id == tenant.tenant_id
                )
            )
            analysis = self.session.scalar(
                select(LeadAnalysis)
                .where(
                    LeadAnalysis.tenant_id == tenant.tenant_id,
                    LeadAnalysis.lead_id == lead.id,
                )
                .order_by(LeadAnalysis.created_at.desc(), LeadAnalysis.id.desc())
            )
            if customer is None or analysis is None:
                block("prerequisite_missing", "Draft requires customer and qualification analysis")
            else:
                payload = DraftContext(
                    customer_name=customer.name,
                    task=analysis.task,
                    summary=analysis.ai_summary,
                    budget_amount=(
                        float(analysis.budget_amount)
                        if analysis.budget_amount is not None
                        else None
                    ),
                    budget_currency=analysis.budget_currency,
                    deadline=analysis.deadline,
                ).model_dump(mode="json")
                estimated_tokens = estimate_structured_tokens(payload)

        input_limit = min(policy.max_input_tokens, self.settings.ai_max_input_tokens)
        output_limit = min(policy.max_output_tokens, self.settings.ai_max_output_tokens)
        if estimated_tokens is not None and estimated_tokens > input_limit:
            block("context_too_large", "Estimated input exceeds the configured token limit")
        attempts = min(policy.max_attempts, self.settings.ai_max_attempts)
        estimate = None
        if price is not None and estimated_tokens is not None:
            estimate = price.worst_case_cost(estimated_tokens, output_limit, attempts)
        period_start = self.executor.spending_guard.period_start(datetime.now(UTC))
        operation_currencies = set(
            self.session.scalars(
                select(AIOperation.cost_currency)
                .where(
                    AIOperation.tenant_id == tenant.tenant_id,
                    AIOperation.created_at >= period_start,
                )
                .distinct()
            )
        )
        remaining = None
        per_call_limit = None
        if budget_currency is not None:
            mismatched = operation_currencies - {budget_currency.value}
            if mismatched:
                block(
                    "currency_mismatch",
                    "Current AI spend currency does not match the tenant budget currency",
                )
            else:
                spent = int(
                    self.session.scalar(
                        select(
                            func.coalesce(
                                func.sum(
                                    func.coalesce(AIOperation.actual_cost_micros, 0)
                                    + AIOperation.reserved_cost_micros
                                ),
                                0,
                            )
                        ).where(
                            AIOperation.tenant_id == tenant.tenant_id,
                            AIOperation.created_at >= period_start,
                        )
                    )
                    or 0
                )
                remaining = Money(
                    amount_micros=max(policy.monthly_limit_micros - spent, 0),
                    currency=budget_currency,
                )
            per_call_limit = Money(
                amount_micros=policy.per_call_limit_micros,
                currency=budget_currency,
            )
        if estimate is not None and per_call_limit is not None:
            if estimate > per_call_limit:
                block("per_call_limit_exceeded", "Worst-case cost exceeds the per-call limit")
            if remaining is not None and estimate > remaining:
                block("monthly_budget_exceeded", "Worst-case cost exceeds remaining monthly budget")

        return LiveAIReadinessResult(
            ready=not blockers,
            blockers=blockers,
            provider_alias=provider,
            model_alias=model,
            pricing_version=price.version if price is not None else None,
            credential_configured=route is not None and route.execution_mode.value == "LIVE",
            estimated_max_cost=estimate,
            remaining_monthly_budget=remaining,
            per_call_limit=per_call_limit,
            required_scope=required.value,
            estimated_input_tokens=estimated_tokens,
            max_input_tokens=input_limit,
            max_output_tokens=output_limit,
        )
