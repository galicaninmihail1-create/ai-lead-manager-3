class EntityNotFoundError(Exception):
    """Requested entity does not exist in the current tenant."""


class TenantScopeError(ValueError):
    """An operation attempted to escape its tenant scope."""


class ConcurrentInboundEventError(Exception):
    """A concurrent request inserted the same inbound event first."""


class IdentityConflictError(Exception):
    """Provided identities resolve to incompatible customers."""


class QualificationUnavailableError(ValueError):
    """A lead cannot be qualified from the available source facts."""
