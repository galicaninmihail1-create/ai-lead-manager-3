import uuid
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai_lead_manager.application.api_keys import (
    APIKeyAuthenticationError,
    APIKeyConfigurationError,
    APIKeyScope,
    TenantAPIKeyCredentialService,
    TenantAPIKeyService,
    VerifiedAPIKey,
)
from ai_lead_manager.application.tenant_context import TenantContext
from ai_lead_manager.config import get_settings
from ai_lead_manager.domain.models import AuditEvent, TenantAIPolicy
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyTenantAPIKeyRepository,
    SqlAlchemyUnitOfWork,
)

bearer_scheme = HTTPBearer(auto_error=False)
BearerCredentials = Annotated[
    HTTPAuthorizationCredentials | None,
    Security(bearer_scheme),
]


def authentication_error() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail={"code": "invalid_credentials"},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _is_live_tenant(session: Session, tenant_id: uuid.UUID) -> bool:
    return session.scalar(
        select(TenantAIPolicy.mode).where(TenantAIPolicy.tenant_id == tenant_id)
    ) == "LIVE"


def _verify(
    session: Session,
    tenant_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials,
) -> VerifiedAPIKey:
    pepper = get_settings().tenant_api_key_pepper
    service = TenantAPIKeyService(
        tenant_id,
        SqlAlchemyTenantAPIKeyRepository(session, tenant_id),
        SqlAlchemyUnitOfWork(session),
        TenantAPIKeyCredentialService(
            pepper.get_secret_value() if pepper is not None else None
        ),
    )
    try:
        return service.verify(credentials.credentials)
    except (APIKeyAuthenticationError, APIKeyConfigurationError) as error:
        raise authentication_error() from error


def trusted_context_for_scope(
    session: Session,
    body_tenant_id: uuid.UUID,
    credentials: HTTPAuthorizationCredentials | None,
    required_scope: APIKeyScope,
) -> TenantContext:
    if credentials is None:
        if _is_live_tenant(session, body_tenant_id):
            raise authentication_error()
        return TenantContext.untrusted(body_tenant_id)

    verified = _verify(session, body_tenant_id, credentials)
    if required_scope not in verified.scopes:
        session.add(
            AuditEvent(
                tenant_id=verified.tenant_id,
                actor_type="system",
                action="tenant_api_key.scope_denied",
                entity_type="tenant_api_key",
                entity_id=verified.key_id,
                event_metadata={
                    "key_prefix": verified.key_prefix,
                    "required_scope": required_scope.value,
                },
            )
        )
        session.commit()
        raise HTTPException(status_code=403, detail={"code": "insufficient_scope"})
    session.add(
        AuditEvent(
            tenant_id=verified.tenant_id,
            actor_type="system",
            action="tenant_api_key.authenticated",
            entity_type="tenant_api_key",
            entity_id=verified.key_id,
            event_metadata={
                "key_prefix": verified.key_prefix,
                "scope": required_scope.value,
            },
        )
    )
    session.commit()
    return TenantContext.trusted_context(
        verified.tenant_id, f"tenant_api_key:{verified.key_id}"
    )
