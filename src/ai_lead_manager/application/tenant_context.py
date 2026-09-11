import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Tenant identity plus the trust decision made by the transport layer."""

    tenant_id: uuid.UUID
    trusted: bool
    source: str

    @classmethod
    def untrusted(cls, tenant_id: uuid.UUID, source: str = "request_body") -> "TenantContext":
        return cls(tenant_id=tenant_id, trusted=False, source=source)

    @classmethod
    def trusted_context(
        cls, tenant_id: uuid.UUID, source: str = "authenticated_principal"
    ) -> "TenantContext":
        return cls(tenant_id=tenant_id, trusted=True, source=source)
