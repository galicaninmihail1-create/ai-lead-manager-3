import uuid

from ai_lead_manager.application.ai.context import BoundedContextBuilder, canonical_request_hash
from ai_lead_manager.application.ai.contracts import (
    AIExecutionOutcome,
    AIOperationStatus,
    AIOperationType,
    StructuredAIRequest,
)
from ai_lead_manager.application.ai.executor import AIExecutor
from ai_lead_manager.application.ports import QualificationRepository, UnitOfWork
from ai_lead_manager.application.providers import (
    QualificationContext,
    QualificationProvider,
    StructuredQualification,
)
from ai_lead_manager.application.schemas import (
    AnalysisSummaryView,
    NextActionView,
    QualificationResponse,
    ScoreFactorView,
)
from ai_lead_manager.application.scoring import ScoreContext, calculate_score, recommend_next_action
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.config import get_settings
from ai_lead_manager.domain.exceptions import EntityNotFoundError, QualificationUnavailableError
from ai_lead_manager.domain.models import (
    AuditEvent,
    LeadAnalysis,
    NextActionRecommendation,
    ScoreFactor,
    ScoreRun,
)


class QualificationService:
    def __init__(
        self,
        tenant: TenantContext | uuid.UUID,
        repository: QualificationRepository,
        uow: UnitOfWork,
        provider: QualificationProvider,
        ai_executor: AIExecutor | None = None,
        context_builder: BoundedContextBuilder | None = None,
    ) -> None:
        self.tenant_context = (
            tenant
            if isinstance(tenant, TenantContext)
            else TenantContext.untrusted(tenant, "legacy_application_call")
        )
        self.tenant_id = self.tenant_context.tenant_id
        self.repository = repository
        self.uow = uow
        self.provider = provider
        self.ai_executor = ai_executor
        settings = get_settings()
        self.context_builder = context_builder or BoundedContextBuilder(
            settings.ai_max_messages,
            settings.ai_max_input_chars,
            settings.ai_max_input_tokens,
        )
        self.max_output_tokens = settings.ai_max_output_tokens

    def qualify(
        self, lead_id: uuid.UUID, idempotency_key: str | None = None
    ) -> QualificationResponse:
        try:
            return self._qualify(lead_id, idempotency_key)
        except Exception:
            self.uow.rollback()
            raise

    def _qualify(
        self, lead_id: uuid.UUID, idempotency_key: str | None
    ) -> QualificationResponse:
        lead = self.repository.get_lead(lead_id)
        if lead is None:
            raise EntityNotFoundError("Lead was not found in the current tenant")
        customer = self.repository.get_customer(lead.customer_id)
        if customer is None:
            raise EntityNotFoundError("Lead customer was not found in the current tenant")
        raw_messages = [
            message.body for message in self.repository.get_inbound_messages(lead) if message.body.strip()
        ]
        bounded = self.context_builder.build(raw_messages)
        if not bounded.messages:
            raise QualificationUnavailableError("Lead has no usable inbound messages")

        provider_context = QualificationContext(
            customer_name=customer.name,
            has_email=bool(customer.email),
            has_phone=bool(customer.phone),
            messages=list(bounded.messages),
        )
        execution: AIExecutionOutcome[StructuredQualification] | None = None
        if idempotency_key is not None and self.ai_executor is not None:
            payload = provider_context.model_dump(mode="json")
            request = StructuredAIRequest(
                operation_type=AIOperationType.QUALIFICATION,
                schema_name="StructuredQualification",
                schema_version=self.provider.schema_version,
                prompt_version=self.provider.prompt_version,
                payload=payload,
                estimated_input_tokens=bounded.estimated_tokens,
                max_output_tokens=self.max_output_tokens,
            )
            request_hash = canonical_request_hash(
                {
                    "lead_id": str(lead.id),
                    "schema_name": request.schema_name,
                    "schema_version": request.schema_version,
                    "prompt_version": request.prompt_version,
                    "payload": payload,
                }
            )
            self.uow.commit()
            execution = self.ai_executor.execute(
                self.tenant_context,
                AIOperationType.QUALIFICATION,
                lead.id,
                idempotency_key,
                request_hash,
                request,
                StructuredQualification,
            )
            if execution.replayed:
                if execution.result_analysis_id is None:
                    raise QualificationUnavailableError(
                        "Succeeded AI operation has no qualification result"
                    )
                return self._historical_response(
                    lead.id, execution.result_analysis_id, execution.operation_id
                )
            assert execution.value is not None
            provider_result = execution.value
            provider_name = execution.provider
            model_name = execution.model
        else:
            provider_result = StructuredQualification.model_validate(
                self.provider.analyze(provider_context)
            )
            provider_name = self.provider.provider_name
            model_name = self.provider.model_name

        analysis = self.repository.add_analysis(
            LeadAnalysis(
                tenant_id=self.tenant_id,
                lead_id=lead.id,
                provider=provider_name,
                model=model_name,
                schema_version=self.provider.schema_version,
                prompt_version=self.provider.prompt_version,
                intent=provider_result.intent,
                task=provider_result.task,
                pain_points=provider_result.pain_points,
                budget_amount=provider_result.budget_amount,
                budget_currency=provider_result.budget_currency,
                deadline=provider_result.deadline,
                objections=provider_result.objections,
                urgency=provider_result.urgency,
                decision_readiness=provider_result.decision_readiness,
                ai_summary=provider_result.summary,
                confidence=provider_result.confidence,
                raw_output=provider_result.model_dump(mode="json"),
            )
        )
        score = calculate_score(
            ScoreContext(
                analysis=provider_result,
                has_email=bool(customer.email),
                has_phone=bool(customer.phone),
            )
        )
        score_run = self.repository.add_score_run(
            ScoreRun(
                tenant_id=self.tenant_id,
                lead_id=lead.id,
                analysis_id=analysis.id,
                score=score.score,
                temperature=score.temperature,
                ruleset_version=score.ruleset_version,
            )
        )
        factors = [
            self.repository.add_score_factor(
                ScoreFactor(
                    tenant_id=self.tenant_id,
                    score_run_id=score_run.id,
                    **factor.model_dump(),
                )
            )
            for factor in score.factors
        ]
        next_action = recommend_next_action(score)
        recommendation = self.repository.add_recommendation(
            NextActionRecommendation(
                tenant_id=self.tenant_id,
                lead_id=lead.id,
                score_run_id=score_run.id,
                **next_action.model_dump(),
            )
        )
        lead.score = score.score
        self.repository.add_audit(
            AuditEvent(
                tenant_id=self.tenant_id,
                actor_type="system",
                action="qualification.completed",
                entity_type="lead",
                entity_id=lead.id,
                event_metadata={
                    "analysis_id": str(analysis.id),
                    "score_run_id": str(score_run.id),
                    "score": score.score,
                    "temperature": score.temperature,
                },
            )
        )
        if execution is not None:
            operation = self.repository.get_ai_operation_for_update(execution.operation_id)
            if operation is None or operation.status != AIOperationStatus.CALLING.value:
                raise QualificationUnavailableError("AI operation cannot be finalized")
            operation.status = AIOperationStatus.SUCCEEDED.value
            operation.reserved_cost_microusd = 0
            operation.result_analysis_id = analysis.id
            self.repository.add_audit(
                AuditEvent(
                    tenant_id=self.tenant_id,
                    actor_type="system",
                    action="ai.operation_finalized",
                    entity_type="ai_operation",
                    entity_id=operation.id,
                    event_metadata={
                        "operation_id": str(operation.id),
                        "result_analysis_id": str(analysis.id),
                    },
                )
            )
        try:
            self.uow.commit()
        except Exception:
            self.uow.rollback()
            if execution is not None and self.ai_executor is not None:
                self.ai_executor.mark_unknown(
                    self.tenant_context, execution.operation_id, "domain_finalization_failed"
                )
            raise
        return self._response(
            lead,
            analysis,
            score_run,
            factors,
            recommendation,
            execution.operation_id if execution else None,
            False,
        )

    def _historical_response(
        self, lead_id: uuid.UUID, analysis_id: uuid.UUID, operation_id: uuid.UUID
    ) -> QualificationResponse:
        lead = self.repository.get_lead(lead_id)
        analysis = self.repository.get_analysis(analysis_id)
        score_run = self.repository.get_score_run_for_analysis(analysis_id)
        if lead is None or analysis is None or score_run is None:
            raise QualificationUnavailableError("Persisted qualification history is incomplete")
        factors = self.repository.get_score_factors(score_run.id)
        recommendation = self.repository.get_recommendation_for_score_run(score_run.id)
        if recommendation is None:
            raise QualificationUnavailableError("Persisted recommendation is missing")
        return self._response(
            lead, analysis, score_run, factors, recommendation, operation_id, True
        )

    def _response(
        self,
        lead,
        analysis: LeadAnalysis,
        score_run: ScoreRun,
        factors: list[ScoreFactor],
        recommendation: NextActionRecommendation,
        operation_id: uuid.UUID | None,
        replayed: bool,
    ) -> QualificationResponse:
        return QualificationResponse(
            tenant_id=self.tenant_id,
            lead_id=lead.id,
            analysis_id=analysis.id,
            score_run_id=score_run.id,
            score=score_run.score,
            temperature=score_run.temperature,
            ruleset_version=score_run.ruleset_version,
            score_factors=[ScoreFactorView.model_validate(factor, from_attributes=True) for factor in factors],
            next_best_action=NextActionView.model_validate(recommendation, from_attributes=True),
            analysis=AnalysisSummaryView(
                intent=analysis.intent,
                task=analysis.task,
                pain_points=analysis.pain_points,
                budget_amount=float(analysis.budget_amount) if analysis.budget_amount is not None else None,
                budget_currency=analysis.budget_currency,
                deadline=analysis.deadline,
                objections=analysis.objections,
                urgency=analysis.urgency,
                confidence=float(analysis.confidence) if analysis.confidence is not None else None,
                summary=analysis.ai_summary,
            ),
            ai_operation_id=operation_id,
            replayed=replayed,
        )
