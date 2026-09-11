import os
from collections.abc import Mapping
from typing import Protocol

from pydantic import SecretStr

from ai_lead_manager.application.ai.contracts import AIErrorKind, ProviderDisabledError


class CredentialResolver(Protocol):
    def resolve(self, reference: str) -> SecretStr: ...


class EnvironmentCredentialResolver:
    """Resolves a server-owned environment reference at the infrastructure boundary."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def resolve(self, reference: str) -> SecretStr:
        raw_secret = self._environment.get(reference)
        if raw_secret is None or not raw_secret.strip():
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "Provider credential is unavailable",
            )
        return SecretStr(raw_secret.strip())
