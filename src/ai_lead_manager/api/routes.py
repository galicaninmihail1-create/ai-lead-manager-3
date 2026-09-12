import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from sqlalchemy.orm import Session

from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIExecutionError,
    AIProviderError,
    BudgetExceededError,
)
from ai_lead_manager.application.drafts import MockDraftProvider
from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.providers import MockQualificationProvider
from ai_lead_manager.application.qualification import QualificationService
from ai_lead_manager.application.schemas import (
    InboundHttpRequest,
    InboundHttpResponse,
    LeadStatusRequest,
    LeadStatusResponse,
    QualificationRequest,
    QualificationResponse,
    ResponseDraftResponse,
    SalesTaskResponse,
    TenantActorRequest,
)
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.application.workflow import SalesWorkflowService
from ai_lead_manager.domain.exceptions import (
    EntityNotFoundError,
    IdentityConflictError,
    InvalidWorkflowTransitionError,
    QualificationUnavailableError,
    WorkflowPrerequisiteError,
)
from ai_lead_manager.infrastructure.ai.runtime import get_ai_executor
from ai_lead_manager.infrastructure.database import get_session
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyQualificationRepository,
    SqlAlchemyUnitOfWork,
    SqlAlchemyWorkflowRepository,
)

router = APIRouter()

IdempotencyHeader = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=200,
        pattern=r"^[\x21-\x7e]+$",
    ),
]


def resolve_idempotency_key(body_key: str | None, header_key: str | None) -> str | None:
    if body_key is not None and header_key is not None and body_key != header_key:
        raise HTTPException(status_code=409, detail={"code": AIErrorKind.INVALID_REQUEST.value})
    return header_key or body_key


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/inbound/http", response_model=InboundHttpResponse, status_code=201)
def ingest_http(
    request: InboundHttpRequest,
    response: Response,
    session: Annotated[Session, Depends(get_session)],
) -> InboundHttpResponse:
    service = InboundIngestionService(
        SqlAlchemyInboundRepository(session, request.tenant_id),
        SqlAlchemyUnitOfWork(session),
    )
    try:
        result = service.ingest(request)
        if result.duplicate:
            response.status_code = 200
        return result
    except EntityNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except IdentityConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/api/v1/leads/{lead_id}/qualify", response_model=QualificationResponse)
def qualify_lead(
    lead_id: uuid.UUID,
    request: QualificationRequest,
    session: Annotated[Session, Depends(get_session)],
    idempotency_key: IdempotencyHeader = None,
) -> QualificationResponse:
    service = QualificationService(
        TenantContext.untrusted(request.tenant_id),
        SqlAlchemyQualificationRepository(session, request.tenant_id),
        SqlAlchemyUnitOfWork(session),
        MockQualificationProvider(),
        get_ai_executor(),
    )
    try:
        return service.qualify(
            lead_id,
            resolve_idempotency_key(request.idempotency_key, idempotency_key),
        )
    except EntityNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except QualificationUnavailableError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (AIExecutionError, AIProviderError) as error:
        raise ai_http_error(error) from error


def workflow_service(session: Session, tenant_id: uuid.UUID) -> SalesWorkflowService:
    return SalesWorkflowService(
        TenantContext.untrusted(tenant_id),
        SqlAlchemyWorkflowRepository(session, tenant_id),
        SqlAlchemyUnitOfWork(session),
        MockDraftProvider(),
        get_ai_executor(),
    )


def ai_http_error(error: AIExecutionError | AIProviderError) -> HTTPException:
    if isinstance(error, BudgetExceededError):
        status = 402
    elif error.kind == AIErrorKind.RATE_LIMIT:
        status = 429
    elif error.kind == AIErrorKind.PROVIDER_DISABLED:
        status = 503
    elif error.kind in {AIErrorKind.INVALID_REQUEST, AIErrorKind.AMBIGUOUS_RESULT}:
        status = 409
    else:
        status = 502
    return HTTPException(status_code=status, detail={"code": error.kind.value})


def workflow_result[T](operation: Callable[[], T]) -> T:
    try:
        return operation()
    except EntityNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (InvalidWorkflowTransitionError, WorkflowPrerequisiteError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (AIExecutionError, AIProviderError) as error:
        raise ai_http_error(error) from error


@router.post("/api/v1/leads/{lead_id}/tasks/from-recommendation", response_model=SalesTaskResponse)
def create_task(
    lead_id: uuid.UUID,
    request: TenantActorRequest,
    session: Annotated[Session, Depends(get_session)],
) -> SalesTaskResponse:
    service = workflow_service(session, request.tenant_id)
    return workflow_result(lambda: service.create_task_from_recommendation(lead_id, request.actor_type))


@router.post("/api/v1/leads/{lead_id}/drafts", response_model=ResponseDraftResponse)
def generate_draft(
    lead_id: uuid.UUID,
    request: TenantActorRequest,
    session: Annotated[Session, Depends(get_session)],
    idempotency_key: IdempotencyHeader = None,
) -> ResponseDraftResponse:
    service = workflow_service(session, request.tenant_id)
    return workflow_result(
        lambda: service.generate_draft(
            lead_id,
            request.actor_type,
            resolve_idempotency_key(request.idempotency_key, idempotency_key),
        )
    )


@router.post("/api/v1/drafts/{draft_id}/approve", response_model=ResponseDraftResponse)
def approve_draft(
    draft_id: uuid.UUID,
    request: TenantActorRequest,
    session: Annotated[Session, Depends(get_session)],
) -> ResponseDraftResponse:
    service = workflow_service(session, request.tenant_id)
    return workflow_result(lambda: service.approve_draft(draft_id, request.actor_type))


@router.post("/api/v1/drafts/{draft_id}/reject", response_model=ResponseDraftResponse)
def reject_draft(
    draft_id: uuid.UUID,
    request: TenantActorRequest,
    session: Annotated[Session, Depends(get_session)],
) -> ResponseDraftResponse:
    service = workflow_service(session, request.tenant_id)
    return workflow_result(lambda: service.reject_draft(draft_id, request.actor_type))


@router.post("/api/v1/tasks/{task_id}/complete", response_model=SalesTaskResponse)
def complete_task(
    task_id: uuid.UUID,
    request: TenantActorRequest,
    session: Annotated[Session, Depends(get_session)],
) -> SalesTaskResponse:
    service = workflow_service(session, request.tenant_id)
    return workflow_result(lambda: service.complete_task(task_id, request.actor_type))


@router.post("/api/v1/tasks/{task_id}/cancel", response_model=SalesTaskResponse)
def cancel_task(
    task_id: uuid.UUID,
    request: TenantActorRequest,
    session: Annotated[Session, Depends(get_session)],
) -> SalesTaskResponse:
    service = workflow_service(session, request.tenant_id)
    return workflow_result(lambda: service.cancel_task(task_id, request.actor_type))


@router.post("/api/v1/leads/{lead_id}/status", response_model=LeadStatusResponse)
def transition_lead_status(
    lead_id: uuid.UUID,
    request: LeadStatusRequest,
    session: Annotated[Session, Depends(get_session)],
) -> LeadStatusResponse:
    service = workflow_service(session, request.tenant_id)
    return workflow_result(
        lambda: service.transition_lead(
            lead_id, request.status, request.reason, request.actor_type
        )
    )
