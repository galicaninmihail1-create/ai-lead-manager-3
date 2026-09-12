import json
import os
import re
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ai_lead_manager.application.api_keys import (
    APIKeyAuthenticationError,
    APIKeyConfigurationError,
    APIKeyCreate,
    APIKeyScope,
    TenantAPIKeyCredentialService,
    TenantAPIKeyService,
)
from ai_lead_manager.application.schemas import TenantCreate
from ai_lead_manager.application.services import TenantService
from ai_lead_manager.domain.exceptions import EntityNotFoundError
from ai_lead_manager.domain.models import AuditEvent, Tenant, TenantAPIKey
from ai_lead_manager.infrastructure.repositories import (
    SqlAlchemyTenantAPIKeyRepository,
    SqlAlchemyTenantRepository,
    SqlAlchemyUnitOfWork,
)

PEPPER = "test-only-pepper-that-is-never-a-production-secret"


def make_service(
    session: Session,
    tenant_id: uuid.UUID,
    *,
    pepper: str | None = PEPPER,
    clock=lambda: datetime.now(UTC),
) -> TenantAPIKeyService:
    return TenantAPIKeyService(
        tenant_id,
        SqlAlchemyTenantAPIKeyRepository(session, tenant_id),
        SqlAlchemyUnitOfWork(session),
        TenantAPIKeyCredentialService(pepper),
        clock=clock,
    )


def seed_tenant(session: Session, slug: str) -> uuid.UUID:
    return TenantService(
        SqlAlchemyTenantRepository(session), SqlAlchemyUnitOfWork(session)
    ).create(TenantCreate(name=slug, slug=slug)).id


def create_key(service: TenantAPIKeyService, **overrides):
    values = {"label": "Production operator", "scopes": {"ai:qualify"}}
    values.update(overrides)
    return service.create(APIKeyCreate.model_validate(values))


def test_key_format_entropy_uniqueness_and_parser() -> None:
    credentials = TenantAPIKeyCredentialService(PEPPER)
    generated = [credentials.generate() for _ in range(100)]

    assert len({item.presented_key for item in generated}) == 100
    assert len({item.prefix for item in generated}) == 100
    assert all(
        re.fullmatch(r"alm_sk_[0-9a-f]{12}_[A-Za-z0-9_-]{43}", item.presented_key)
        for item in generated
    )
    assert credentials.parse(generated[0].presented_key).prefix == generated[0].prefix

    for malformed in ("", "alm_sk_short_secret", "ALM_SK_abcdefabcdef_x", " alm_sk_x_y"):
        with pytest.raises(APIKeyAuthenticationError, match="Invalid tenant API key"):
            credentials.parse(malformed)


def test_create_verify_one_time_secret_and_sanitized_audit(session: Session) -> None:
    tenant_id = seed_tenant(session, "api-key-create")
    service = make_service(session, tenant_id)
    created = create_key(service, scopes={"ai:qualify", "ai:draft"})
    raw_key = created.api_key
    verified = service.verify(raw_key)
    stored = session.scalar(select(TenantAPIKey).where(TenantAPIKey.id == created.metadata.id))
    audits = list(
        session.scalars(select(AuditEvent).where(AuditEvent.tenant_id == tenant_id))
    )

    assert verified.tenant_id == tenant_id
    assert verified.key_id == stored.id
    assert verified.scopes == {APIKeyScope.AI_QUALIFY, APIKeyScope.AI_DRAFT}
    assert stored.key_hash != raw_key and raw_key not in stored.key_hash
    assert stored.last_used_at is not None
    assert raw_key not in repr(created)
    assert PEPPER not in repr(service.credentials)
    assert "api_key" not in verified.model_dump()
    assert "key_hash" not in created.metadata.model_dump()
    assert raw_key not in json.dumps([event.event_metadata for event in audits])
    assert PEPPER not in json.dumps([event.event_metadata for event in audits])
    assert [event.action for event in audits] == ["tenant_api_key.created"]


def test_wrong_unknown_malformed_revoked_and_expired_are_generic(session: Session) -> None:
    now = datetime(2026, 9, 12, tzinfo=UTC)
    tenant_id = seed_tenant(session, "api-key-failures")
    service = make_service(session, tenant_id, clock=lambda: now)
    active = create_key(service)
    expired = create_key(service, label="Expired", expires_at=now - timedelta(seconds=1))

    failures = [
        active.api_key[:-1] + ("A" if active.api_key[-1] != "A" else "B"),
        f"alm_sk_ffffffffffff_{'A' * 43}",
        "malformed",
        expired.api_key,
    ]
    service.revoke(active.metadata.id)
    failures.append(active.api_key)

    for presented in failures:
        with pytest.raises(APIKeyAuthenticationError) as captured:
            service.verify(presented)
        assert str(captured.value) == "Invalid tenant API key"
        assert presented not in repr(captured.value)
        assert PEPPER not in repr(captured.value)


def test_revocation_immediate_multiple_keys_and_metadata_has_no_secret(session: Session) -> None:
    tenant_id = seed_tenant(session, "api-key-multiple")
    service = make_service(session, tenant_id)
    first = create_key(service, label="First")
    second = create_key(service, label="Second", scopes={"ai:draft"})

    assert service.verify(first.api_key).key_id == first.metadata.id
    assert service.verify(second.api_key).key_id == second.metadata.id
    revoked = service.revoke(first.metadata.id)
    assert revoked.status == "REVOKED" and revoked.revoked_at is not None
    with pytest.raises(APIKeyAuthenticationError):
        service.verify(first.api_key)
    assert service.verify(second.api_key).key_id == second.metadata.id

    metadata = service.list()
    serialized = json.dumps([item.model_dump(mode="json") for item in metadata])
    assert len(metadata) == 2
    assert first.api_key not in serialized and second.api_key not in serialized
    assert "key_hash" not in serialized


def test_scopes_are_strict_and_missing_pepper_fails_closed(session: Session) -> None:
    with pytest.raises(ValidationError):
        APIKeyCreate(label="Unknown", scopes={"admin"})
    with pytest.raises(ValidationError):
        APIKeyCreate(label="Empty", scopes=set())

    tenant_id = seed_tenant(session, "api-key-pepper")
    for pepper in (None, "", "   "):
        service = make_service(session, tenant_id, pepper=pepper)
        with pytest.raises(APIKeyConfigurationError) as created:
            create_key(service)
        assert PEPPER not in repr(created.value)
        with pytest.raises(APIKeyConfigurationError):
            service.verify(f"alm_sk_ffffffffffff_{'A' * 43}")
    assert service.list() == []


def test_tenants_are_isolated_for_verify_list_and_revoke(session: Session) -> None:
    first_tenant = seed_tenant(session, "api-key-tenant-a")
    second_tenant = seed_tenant(session, "api-key-tenant-b")
    first_service = make_service(session, first_tenant)
    second_service = make_service(session, second_tenant)
    first = create_key(first_service)
    second = create_key(second_service)

    with pytest.raises(APIKeyAuthenticationError):
        second_service.verify(first.api_key)
    with pytest.raises(EntityNotFoundError):
        second_service.revoke(first.metadata.id)
    assert [item.id for item in first_service.list()] == [first.metadata.id]
    assert [item.id for item in second_service.list()] == [second.metadata.id]


postgres_only = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="PostgreSQL integration tests are opt-in",
)


@postgres_only
def test_postgres_tenant_api_key_constraints_and_cascade() -> None:
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    tenant_id = None
    try:
        with factory() as session:
            tenant_id = seed_tenant(session, f"pg-api-key-{uuid.uuid4().hex}")
            valid = TenantAPIKey(
                tenant_id=tenant_id,
                label="Valid",
                key_prefix="0123456789ab",
                key_hash="a" * 64,
                scopes=["ai:qualify"],
                status="ACTIVE",
            )
            session.add(valid)
            session.commit()
            valid_id = valid.id

            invalid_rows = [
                TenantAPIKey(
                    tenant_id=tenant_id,
                    label="Bad status",
                    key_prefix="1123456789ab",
                    key_hash="b" * 64,
                    scopes=["ai:qualify"],
                    status="INVALID",
                ),
                TenantAPIKey(
                    tenant_id=tenant_id,
                    label="Bad prefix",
                    key_prefix="short",
                    key_hash="c" * 64,
                    scopes=["ai:qualify"],
                    status="ACTIVE",
                ),
                TenantAPIKey(
                    tenant_id=tenant_id,
                    label="Bad hash",
                    key_prefix="2123456789ab",
                    key_hash="short",
                    scopes=["ai:qualify"],
                    status="ACTIVE",
                ),
                TenantAPIKey(
                    tenant_id=uuid.uuid4(),
                    label="Unknown tenant",
                    key_prefix="3123456789ab",
                    key_hash="d" * 64,
                    scopes=["ai:qualify"],
                    status="ACTIVE",
                ),
            ]
            for row in invalid_rows:
                session.add(row)
                with pytest.raises(IntegrityError):
                    session.commit()
                session.rollback()

            session.execute(delete(Tenant).where(Tenant.id == tenant_id))
            session.commit()
            assert session.scalar(
                select(TenantAPIKey).where(TenantAPIKey.id == valid_id)
            ) is None
            tenant_id = None
    finally:
        if tenant_id is not None:
            with factory.begin() as session:
                session.execute(delete(Tenant).where(Tenant.id == tenant_id))
        engine.dispose()
