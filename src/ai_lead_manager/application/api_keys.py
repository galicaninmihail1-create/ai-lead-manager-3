from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_lead_manager.application.ports import TenantAPIKeyRepository, UnitOfWork
from ai_lead_manager.domain.exceptions import EntityNotFoundError
from ai_lead_manager.domain.models import AuditEvent, TenantAPIKey

KEY_PREFIX_BYTES = 6
KEY_SECRET_BYTES = 32
KEY_PATTERN = re.compile(r"\Aalm_sk_([0-9a-f]{12})_([A-Za-z0-9_-]{43})\Z")


class APIKeyScope(StrEnum):
    AI_QUALIFY = "ai:qualify"
    AI_DRAFT = "ai:draft"


class APIKeyAuthenticationError(Exception):
    def __init__(self) -> None:
        super().__init__("Invalid tenant API key")


class APIKeyConfigurationError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Tenant API key service is not configured")


class APIKeyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=200)
    scopes: frozenset[APIKeyScope] = Field(min_length=1)
    expires_at: datetime | None = None

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label must not be blank")
        return cleaned

    @field_validator("expires_at")
    @classmethod
    def normalize_expiration(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        return value


class APIKeyMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    label: str
    key_prefix: str
    scopes: frozenset[APIKeyScope]
    status: str
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None


class CreatedAPIKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: str = Field(repr=False)
    metadata: APIKeyMetadata


class VerifiedAPIKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    key_id: uuid.UUID
    tenant_id: uuid.UUID
    key_prefix: str
    scopes: frozenset[APIKeyScope]


@dataclass(frozen=True, repr=False)
class ParsedAPIKey:
    prefix: str
    presented_key: str


class TenantAPIKeyCredentialService:
    def __init__(self, pepper: str | None) -> None:
        self._pepper = pepper.strip() if pepper is not None else ""

    def _pepper_bytes(self) -> bytes:
        if not self._pepper:
            raise APIKeyConfigurationError()
        return self._pepper.encode("utf-8")

    def ensure_configured(self) -> None:
        self._pepper_bytes()

    def generate(self) -> ParsedAPIKey:
        prefix = secrets.token_hex(KEY_PREFIX_BYTES)
        secret = secrets.token_urlsafe(KEY_SECRET_BYTES)
        return ParsedAPIKey(prefix, f"alm_sk_{prefix}_{secret}")

    def parse(self, presented_key: str) -> ParsedAPIKey:
        match = KEY_PATTERN.fullmatch(presented_key)
        if match is None:
            raise APIKeyAuthenticationError()
        return ParsedAPIKey(match.group(1), presented_key)

    def digest(self, presented_key: str) -> str:
        return hmac.new(
            self._pepper_bytes(), presented_key.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def matches(self, presented_key: str, expected_digest: str) -> bool:
        return hmac.compare_digest(self.digest(presented_key), expected_digest)


class TenantAPIKeyService:
    def __init__(
        self,
        tenant_id: uuid.UUID,
        repository: TenantAPIKeyRepository,
        uow: UnitOfWork,
        credentials: TenantAPIKeyCredentialService,
        *,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        self.tenant_id = tenant_id
        self.repository = repository
        self.uow = uow
        self.credentials = credentials
        self.clock = clock

    def create(self, data: APIKeyCreate) -> CreatedAPIKey:
        if not self.repository.tenant_exists():
            raise EntityNotFoundError("Tenant was not found")
        generated = self.credentials.generate()
        entity = self.repository.add(
            TenantAPIKey(
                tenant_id=self.tenant_id,
                label=data.label,
                key_prefix=generated.prefix,
                key_hash=self.credentials.digest(generated.presented_key),
                scopes=sorted(scope.value for scope in data.scopes),
                status="ACTIVE",
                expires_at=data.expires_at,
            )
        )
        self._audit("tenant_api_key.created", entity)
        self.uow.commit()
        return CreatedAPIKey(api_key=generated.presented_key, metadata=self._metadata(entity))

    def verify(self, presented_key: str) -> VerifiedAPIKey:
        try:
            self.credentials.ensure_configured()
            parsed = self.credentials.parse(presented_key)
            entity = self.repository.get_by_prefix(parsed.prefix)
            now = self.clock()
            valid = (
                entity is not None
                and entity.status == "ACTIVE"
                and entity.revoked_at is None
                and (entity.expires_at is None or entity.expires_at > now)
                and self.credentials.matches(parsed.presented_key, entity.key_hash)
            )
        except APIKeyConfigurationError:
            raise
        except Exception as error:
            raise APIKeyAuthenticationError() from error
        if not valid or entity is None:
            raise APIKeyAuthenticationError()
        entity.last_used_at = now
        self.uow.commit()
        return VerifiedAPIKey(
            key_id=entity.id,
            tenant_id=entity.tenant_id,
            key_prefix=entity.key_prefix,
            scopes=frozenset(APIKeyScope(scope) for scope in entity.scopes),
        )

    def revoke(self, key_id: uuid.UUID) -> APIKeyMetadata:
        entity = self.repository.get(key_id)
        if entity is None:
            raise EntityNotFoundError("Tenant API key was not found in the current tenant")
        if entity.status != "REVOKED":
            entity.status = "REVOKED"
            entity.revoked_at = self.clock()
            self._audit("tenant_api_key.revoked", entity)
            self.uow.commit()
        return self._metadata(entity)

    def list(self) -> list[APIKeyMetadata]:
        return [self._metadata(entity) for entity in self.repository.list()]

    def _audit(self, action: str, entity: TenantAPIKey) -> None:
        self.repository.add_audit(
            AuditEvent(
                tenant_id=self.tenant_id,
                actor_type="system",
                action=action,
                entity_type="tenant_api_key",
                entity_id=entity.id,
                event_metadata={
                    "key_prefix": entity.key_prefix,
                    "scopes": list(entity.scopes),
                },
            )
        )

    @staticmethod
    def _metadata(entity: TenantAPIKey) -> APIKeyMetadata:
        return APIKeyMetadata(
            id=entity.id,
            tenant_id=entity.tenant_id,
            label=entity.label,
            key_prefix=entity.key_prefix,
            scopes=frozenset(APIKeyScope(scope) for scope in entity.scopes),
            status=entity.status,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            last_used_at=entity.last_used_at,
            expires_at=entity.expires_at,
            revoked_at=entity.revoked_at,
        )
