import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ai_lead_manager.application.inbound import InboundIngestionService
from ai_lead_manager.application.providers import MockQualificationProvider
from ai_lead_manager.application.qualification import QualificationService
from ai_lead_manager.application.schemas import (
    InboundHttpRequest,
    InboundHttpResponse,
    QualificationRequest,
    QualificationResponse,
)
from ai_lead_manager.domain.exceptions import (
    EntityNotFoundError,
    IdentityConflictError,
    QualificationUnavailableError,
)
from ai_lead_manager.infrastructure.database import get_session
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyInboundRepository,
    SqlAlchemyQualificationRepository,
    SqlAlchemyUnitOfWork,
)

router = APIRouter()


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
) -> QualificationResponse:
    service = QualificationService(
        request.tenant_id,
        SqlAlchemyQualificationRepository(session, request.tenant_id),
        SqlAlchemyUnitOfWork(session),
        MockQualificationProvider(),
    )
    try:
        return service.qualify(lead_id)
    except EntityNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except QualificationUnavailableError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
