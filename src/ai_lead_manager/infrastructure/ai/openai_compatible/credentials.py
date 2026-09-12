import os
from collections.abc import Mapping
from typing import Protocol

from pydantic import SecretStr

from ai_lead_manager.application.ai.contracts import AIErrorKind, ProviderDisabledError


class CredentialResolver(Protocol):
    def is_configured(self, reference: str) -> bool: ...

    def resolve(self, reference: str) -> SecretStr: ...


class EnvironmentCredentialResolver:
    """Resolves a server-owned environment reference at the infrastructure boundary."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def is_configured(self, reference: str) -> bool:
        raw_secret = self._environment.get(reference)
        return raw_secret is not None and bool(raw_secret.strip())

    def resolve(self, reference: str) -> SecretStr:
        raw_secret = self._environment.get(reference)
        if not self.is_configured(reference):
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "Provider credential is unavailable",
            )
        assert raw_secret is not None
        return SecretStr(raw_secret.strip())
