import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from ai_lead_manager.application.drafts import DraftContext, DraftProvider, DraftResult
from ai_lead_manager.application.ports import UnitOfWork, WorkflowRepository
from ai_lead_manager.application.schemas import (
    LeadStatusResponse,
    ResponseDraftResponse,
    SalesTaskResponse,
)
from ai_lead_manager.domain.exceptions import (
    EntityNotFoundError,
    InvalidWorkflowTransitionError,
    WorkflowPrerequisiteError,
)
from ai_lead_manager.domain.models import (
    AuditEvent,
    Lead,
    LeadStatusHistory,
    ResponseDraft,
    SalesTask,
)

ALLOWED_TRANSITIONS = {
    "NEW": {"QUALIFIED"},
    "QUALIFIED": {"CONTACTED"},
    "CONTACTED": {"MEETING"},
    "MEETING": {"WON", "LOST"},
    "WON": set(),
    "LOST": set(),
}


class SalesWorkflowService:
    def __init__(
        self,
        tenant_id: uuid.UUID,
        repository: WorkflowRepository,
        uow: UnitOfWork,
        draft_provider: DraftProvider,
    ) -> None:
        self.tenant_id = tenant_id
        self.repository = repository
        self.uow = uow
        self.draft_provider = draft_provider

    def _execute[T](self, operation: Callable[[], T]) -> T:
        try:
            result = operation()
            self.uow.commit()
            return result
        except Exception:
            self.uow.rollback()
            raise

    def create_task_from_recommendation(
        self, lead_id: uuid.UUID, actor_type: str
    ) -> SalesTaskResponse:
        return self._execute(lambda: self._create_task(lead_id, actor_type))

    def _create_task(self, lead_id: uuid.UUID, actor_type: str) -> SalesTaskResponse:
        self._lead(lead_id)
        recommendation = self.repository.latest_recommendation(lead_id)
        if recommendation is None:
            raise WorkflowPrerequisiteError("Lead has no qualification recommendation")
        existing = self.repository.task_for_recommendation(recommendation.id)
        if existing is not None:
            return self._task_response(existing, created=False)
        task = self.repository.add_task(
            SalesTask(
                tenant_id=self.tenant_id,
                lead_id=lead_id,
                recommendation_id=recommendation.id,
                task_type=recommendation.action_type,
                title=self._task_title(recommendation.action_type),
                description=recommendation.reason,
                priority=recommendation.priority,
                status="OPEN",
            )
        )
        self._audit(actor_type, "task.created", "sales_task", task.id, {"lead_id": str(lead_id)})
        return self._task_response(task, created=True)

    def generate_draft(self, lead_id: uuid.UUID, actor_type: str) -> ResponseDraftResponse:
        return self._execute(lambda: self._generate_draft(lead_id, actor_type))

    def _generate_draft(self, lead_id: uuid.UUID, actor_type: str) -> ResponseDraftResponse:
        lead = self._lead(lead_id)
        customer = self.repository.get_customer(lead.customer_id)
        analysis = self.repository.latest_analysis(lead_id)
        if customer is None or analysis is None:
            raise WorkflowPrerequisiteError("Lead requires a customer and qualification analysis")
        result = DraftResult.model_validate(
            self.draft_provider.generate(
                DraftContext(
                    customer_name=customer.name,
                    task=analysis.task,
                    summary=analysis.ai_summary,
                    budget_amount=float(analysis.budget_amount) if analysis.budget_amount is not None else None,
                    budget_currency=analysis.budget_currency,
                    deadline=analysis.deadline,
                )
            )
        )
        draft = self.repository.add_draft(
            ResponseDraft(
                tenant_id=self.tenant_id,
                lead_id=lead.id,
                analysis_id=analysis.id,
                provider=self.draft_provider.provider_name,
                model=self.draft_provider.model_name,
                prompt_version=self.draft_provider.prompt_version,
                content=result.content,
                status="APPROVAL_REQUIRED",
            )
        )
        self._audit(actor_type, "draft.created", "response_draft", draft.id, {"lead_id": str(lead.id)})
        return self._draft_response(draft)

    def approve_draft(self, draft_id: uuid.UUID, actor_type: str) -> ResponseDraftResponse:
        return self._execute(lambda: self._set_draft_status(draft_id, "APPROVED", actor_type))

    def reject_draft(self, draft_id: uuid.UUID, actor_type: str) -> ResponseDraftResponse:
        return self._execute(lambda: self._set_draft_status(draft_id, "REJECTED", actor_type))

    def _set_draft_status(
        self, draft_id: uuid.UUID, target: str, actor_type: str
    ) -> ResponseDraftResponse:
        if actor_type != "manager":
            raise WorkflowPrerequisiteError("Draft approval or rejection requires a manager")
        draft = self.repository.get_draft(draft_id)
        if draft is None:
            raise EntityNotFoundError("Draft was not found in the current tenant")
        if draft.status == target:
            return self._draft_response(draft)
        if draft.status != "APPROVAL_REQUIRED":
            raise InvalidWorkflowTransitionError(f"Cannot change draft from {draft.status} to {target}")
        now = datetime.now(UTC)
        draft.status = target
        if target == "APPROVED":
            draft.approved_at = now
        else:
            draft.rejected_at = now
        self._audit(actor_type, f"draft.{target.lower()}", "response_draft", draft.id, {})
        return self._draft_response(draft)

    def complete_task(self, task_id: uuid.UUID, actor_type: str) -> SalesTaskResponse:
        return self._execute(lambda: self._set_task_status(task_id, "DONE", actor_type))

    def cancel_task(self, task_id: uuid.UUID, actor_type: str) -> SalesTaskResponse:
        return self._execute(lambda: self._set_task_status(task_id, "CANCELLED", actor_type))

    def _set_task_status(self, task_id: uuid.UUID, target: str, actor_type: str) -> SalesTaskResponse:
        task = self.repository.get_task(task_id)
        if task is None:
            raise EntityNotFoundError("Task was not found in the current tenant")
        if task.status == target:
            return self._task_response(task, created=False)
        if task.status != "OPEN":
            raise InvalidWorkflowTransitionError(f"Cannot change task from {task.status} to {target}")
        task.status = target
        task.completed_at = datetime.now(UTC) if target == "DONE" else None
        self._audit(actor_type, f"task.{target.lower()}", "sales_task", task.id, {})
        return self._task_response(task, created=False)

    def transition_lead(
        self, lead_id: uuid.UUID, target: str, reason: str | None, actor_type: str
    ) -> LeadStatusResponse:
        return self._execute(lambda: self._transition_lead(lead_id, target, reason, actor_type))

    def _transition_lead(
        self, lead_id: uuid.UUID, target: str, reason: str | None, actor_type: str
    ) -> LeadStatusResponse:
        lead = self._lead(lead_id)
        previous = lead.status
        if previous == target:
            return LeadStatusResponse(
                lead_id=lead.id,
                previous_status=previous,
                status=target,
                changed=False,
                history_id=None,
            )
        if target not in ALLOWED_TRANSITIONS.get(previous, set()):
            raise InvalidWorkflowTransitionError(f"Transition {previous} -> {target} is not allowed")
        history = self.repository.add_status_history(
            LeadStatusHistory(
                tenant_id=self.tenant_id,
                lead_id=lead.id,
                from_status=previous,
                to_status=target,
                reason=reason,
                actor_type=actor_type,
            )
        )
        lead.status = target
        self._audit(actor_type, "lead.status_changed", "lead", lead.id, {"from": previous, "to": target})
        return LeadStatusResponse(
            lead_id=lead.id,
            previous_status=previous,
            status=target,
            changed=True,
            history_id=history.id,
        )

    def _lead(self, lead_id: uuid.UUID) -> Lead:
        lead = self.repository.get_lead(lead_id)
        if lead is None:
            raise EntityNotFoundError("Lead was not found in the current tenant")
        return lead

    def _audit(
        self, actor_type: str, action: str, entity_type: str, entity_id: uuid.UUID, metadata: dict
    ) -> None:
        self.repository.add_audit(
            AuditEvent(
                tenant_id=self.tenant_id,
                actor_type=actor_type,
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                event_metadata=metadata,
            )
        )

    @staticmethod
    def _task_title(action_type: str) -> str:
        return {
            "schedule_call": "Связаться с лидом и согласовать звонок",
            "clarify_requirements": "Уточнить требования лида",
            "collect_context": "Собрать дополнительный контекст",
        }.get(action_type, "Обработать рекомендацию по лиду")

    @staticmethod
    def _task_response(task: SalesTask, created: bool) -> SalesTaskResponse:
        return SalesTaskResponse(
            task_id=task.id,
            lead_id=task.lead_id,
            recommendation_id=task.recommendation_id,
            task_type=task.task_type,
            title=task.title,
            description=task.description,
            priority=task.priority,
            status=task.status,
            created=created,
        )

    @staticmethod
    def _draft_response(draft: ResponseDraft) -> ResponseDraftResponse:
        return ResponseDraftResponse(
            draft_id=draft.id,
            lead_id=draft.lead_id,
            analysis_id=draft.analysis_id,
            content=draft.content,
            status=draft.status,
            approved_at=draft.approved_at,
            rejected_at=draft.rejected_at,
        )
