import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai_lead_manager.api import auth
from ai_lead_manager.api.auth import BearerCredentials, trusted_context_for_scope
from ai_lead_manager.application.api_keys import (
    APIKeyCreate,
    APIKeyScope,
    TenantAPIKeyCredentialService,
    TenantAPIKeyService,
)
from ai_lead_manager.application.schemas import TenantCreate
from ai_lead_manager.application.services import TenantService
from ai_lead_manager.config import Settings
from ai_lead_manager.domain.models import (
    AIOperation,
    AuditEvent,
    ProviderCall,
    TenantAIPolicy,
    TenantAPIKey,
)
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyTenantAPIKeyRepository,
    SqlAlchemyTenantRepository,
    SqlAlchemyUnitOfWork,
)

PEPPER = "stage5c1b-test-only-pepper"


class AuthBody(BaseModel):
    tenant_id: uuid.UUID


def seed_tenant(session: Session, slug: str) -> uuid.UUID:
    tenant = TenantService(
        SqlAlchemyTenantRepository(session), SqlAlchemyUnitOfWork(session)
    ).create(TenantCreate(name=slug, slug=slug))
    session.add(
        TenantAIPolicy(
            tenant_id=tenant.id,
            enabled=True,
            mode="LIVE",
            qualification_provider="mock",
            qualification_model="deterministic-keyword-v1",
            draft_provider="mock",
            draft_model="deterministic-template-v1",
            monthly_limit_microusd=1_000_000,
            per_call_limit_microusd=100_000,
            max_input_tokens=8_000,
            max_output_tokens=1_000,
            max_attempts=1,
            allow_unknown_cost=False,
        )
    )
    session.commit()
    return tenant.id


def issue_key(
    session: Session,
    tenant_id: uuid.UUID,
    scopes: set[str],
    *,
    expires_at=None,
):
    service = TenantAPIKeyService(
        tenant_id,
        SqlAlchemyTenantAPIKeyRepository(session, tenant_id),
        SqlAlchemyUnitOfWork(session),
        TenantAPIKeyCredentialService(PEPPER),
    )
    return service, service.create(
        APIKeyCreate(label="HTTP test", scopes=scopes, expires_at=expires_at)
    )


def auth_client(session: Session, monkeypatch) -> TestClient:
    monkeypatch.setattr(
        auth,
        "get_settings",
        lambda: Settings(_env_file=None, tenant_api_key_pepper=PEPPER),
    )
    app = FastAPI()

    @app.post("/qualify")
    def qualify(body: AuthBody, credentials: BearerCredentials = None):
        context = trusted_context_for_scope(
            session, body.tenant_id, credentials, APIKeyScope.AI_QUALIFY
        )
        return {"tenant_id": context.tenant_id, "trusted": context.trusted}

    @app.post("/draft")
    def draft(body: AuthBody, credentials: BearerCredentials = None):
        context = trusted_context_for_scope(
            session, body.tenant_id, credentials, APIKeyScope.AI_DRAFT
        )
        return {"tenant_id": context.tenant_id, "trusted": context.trusted}

    return TestClient(app)


def assert_no_ai_records(session: Session) -> None:
    assert session.scalar(select(func.count()).select_from(AIOperation)) == 0
    assert session.scalar(select(func.count()).select_from(ProviderCall)) == 0


def test_valid_bearer_creates_trusted_context_and_updates_last_used(
    session: Session, monkeypatch
) -> None:
    tenant_id = seed_tenant(session, "http-auth-valid")
    _service, created = issue_key(session, tenant_id, {"ai:qualify"})
    client = auth_client(session, monkeypatch)

    response = client.post(
        "/qualify",
        headers={"Authorization": f"Bearer {created.api_key}"},
        json={"tenant_id": str(tenant_id)},
    )

    assert response.status_code == 200
    assert response.json() == {"tenant_id": str(tenant_id), "trusted": True}
    stored = session.get(TenantAPIKey, created.metadata.id)
    session.refresh(stored)
    assert stored.last_used_at is not None
    assert_no_ai_records(session)


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic dXNlcjpwYXNz", "Bearer malformed", f"Bearer alm_sk_ffffffffffff_{'A' * 43}"],
)
def test_missing_basic_malformed_and_unknown_are_generic_401(
    session: Session, monkeypatch, authorization: str | None
) -> None:
    tenant_id = seed_tenant(session, f"http-auth-failure-{uuid.uuid4().hex}")
    client = auth_client(session, monkeypatch)
    headers = {"Authorization": authorization} if authorization else {}

    response = client.post(
        "/qualify", headers=headers, json={"tenant_id": str(tenant_id)}
    )

    assert response.status_code == 401
    assert response.json() == {"detail": {"code": "invalid_credentials"}}
    assert response.headers["www-authenticate"] == "Bearer"
    assert authorization is None or authorization not in response.text
    assert_no_ai_records(session)


def test_wrong_revoked_expired_and_failed_auth_do_not_update_last_used(
    session: Session, monkeypatch
) -> None:
    tenant_id = seed_tenant(session, "http-auth-inactive")
    service, active = issue_key(session, tenant_id, {"ai:qualify"})
    _expired_service, expired = issue_key(
        session,
        tenant_id,
        {"ai:qualify"},
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    service.revoke(active.metadata.id)
    client = auth_client(session, monkeypatch)
    wrong = active.api_key[:-1] + ("A" if active.api_key[-1] != "A" else "B")

    for key in (wrong, active.api_key, expired.api_key):
        response = client.post(
            "/qualify",
            headers={"Authorization": f"Bearer {key}"},
            json={"tenant_id": str(tenant_id)},
        )
        assert response.status_code == 401

    for key_id in (active.metadata.id, expired.metadata.id):
        stored = session.get(TenantAPIKey, key_id)
        session.refresh(stored)
        assert stored.last_used_at is None
    assert_no_ai_records(session)


def test_scope_guard_accepts_mapped_scope_and_denies_before_ai_records(
    session: Session, monkeypatch
) -> None:
    tenant_id = seed_tenant(session, "http-auth-scopes")
    _qualify_service, qualify = issue_key(session, tenant_id, {"ai:qualify"})
    _draft_service, draft = issue_key(session, tenant_id, {"ai:draft"})
    client = auth_client(session, monkeypatch)

    assert client.post(
        "/qualify",
        headers={"Authorization": f"Bearer {qualify.api_key}"},
        json={"tenant_id": str(tenant_id)},
    ).status_code == 200
    denied = client.post(
        "/draft",
        headers={"Authorization": f"Bearer {qualify.api_key}"},
        json={"tenant_id": str(tenant_id)},
    )
    assert denied.status_code == 403
    assert denied.json() == {"detail": {"code": "insufficient_scope"}}
    assert client.post(
        "/draft",
        headers={"Authorization": f"Bearer {draft.api_key}"},
        json={"tenant_id": str(tenant_id)},
    ).status_code == 200
    assert_no_ai_records(session)


def test_client_tenant_override_and_cross_tenant_key_fail_closed(
    session: Session, monkeypatch
) -> None:
    tenant_a = seed_tenant(session, "http-auth-tenant-a")
    tenant_b = seed_tenant(session, "http-auth-tenant-b")
    _service, key_a = issue_key(session, tenant_a, {"ai:qualify", "ai:draft"})
    client = auth_client(session, monkeypatch)

    response = client.post(
        "/qualify",
        headers={"Authorization": f"Bearer {key_a.api_key}"},
        json={"tenant_id": str(tenant_b)},
    )

    assert response.status_code == 401
    assert_no_ai_records(session)


def test_auth_audit_is_sanitized_and_does_not_spam_invalid_attempts(
    session: Session, monkeypatch
) -> None:
    tenant_id = seed_tenant(session, "http-auth-audit")
    _service, created = issue_key(session, tenant_id, {"ai:qualify"})
    client = auth_client(session, monkeypatch)
    client.post(
        "/qualify",
        headers={"Authorization": f"Bearer {created.api_key}"},
        json={"tenant_id": str(tenant_id)},
    )
    client.post(
        "/draft",
        headers={"Authorization": f"Bearer {created.api_key}"},
        json={"tenant_id": str(tenant_id)},
    )
    client.post(
        "/qualify",
        headers={"Authorization": "Bearer malformed"},
        json={"tenant_id": str(tenant_id)},
    )

    audits = list(
        session.scalars(select(AuditEvent).where(AuditEvent.tenant_id == tenant_id))
    )
    payload = json.dumps([event.event_metadata for event in audits])
    assert [event.action for event in audits].count("tenant_api_key.authenticated") == 1
    assert [event.action for event in audits].count("tenant_api_key.scope_denied") == 1
    assert created.api_key not in payload and PEPPER not in payload
