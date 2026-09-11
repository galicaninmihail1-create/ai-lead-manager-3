import uuid

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
from ai_lead_manager.domain.exceptions import EntityNotFoundError, QualificationUnavailableError
from ai_lead_manager.domain.models import (
    LeadAnalysis,
    NextActionRecommendation,
    ScoreFactor,
    ScoreRun,
)


class QualificationService:
    def __init__(
        self,
        tenant_id: uuid.UUID,
        repository: QualificationRepository,
        uow: UnitOfWork,
        provider: QualificationProvider,
    ) -> None:
        self.tenant_id = tenant_id
        self.repository = repository
        self.uow = uow
        self.provider = provider

    def qualify(self, lead_id: uuid.UUID) -> QualificationResponse:
        try:
            return self._qualify(lead_id)
        except Exception:
            self.uow.rollback()
            raise

    def _qualify(self, lead_id: uuid.UUID) -> QualificationResponse:
        lead = self.repository.get_lead(lead_id)
        if lead is None:
            raise EntityNotFoundError("Lead was not found in the current tenant")
        customer = self.repository.get_customer(lead.customer_id)
        if customer is None:
            raise EntityNotFoundError("Lead customer was not found in the current tenant")
        messages = [message.body.strip() for message in self.repository.get_inbound_messages(lead) if message.body.strip()]
        if not messages:
            raise QualificationUnavailableError("Lead has no usable inbound messages")

        provider_result = StructuredQualification.model_validate(
            self.provider.analyze(
                QualificationContext(
                customer_name=customer.name,
                has_email=bool(customer.email),
                has_phone=bool(customer.phone),
                messages=messages,
                )
            )
        )
        analysis = self.repository.add_analysis(
            LeadAnalysis(
                tenant_id=self.tenant_id,
                lead_id=lead.id,
                provider=self.provider.provider_name,
                model=self.provider.model_name,
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
        self.uow.commit()
        return QualificationResponse(
            tenant_id=self.tenant_id,
            lead_id=lead.id,
            analysis_id=analysis.id,
            score_run_id=score_run.id,
            score=score.score,
            temperature=score.temperature,
            ruleset_version=score.ruleset_version,
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
        )
