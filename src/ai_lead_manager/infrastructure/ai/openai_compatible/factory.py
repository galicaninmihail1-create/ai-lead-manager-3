from collections.abc import Callable
from typing import Any

import httpx

from ai_lead_manager.infrastructure.ai.openai_compatible.config import (
    OpenAICompatibleProviderCatalog,
    OpenAICompatibleProviderConfig,
    ProviderCapability,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.credentials import CredentialResolver
from ai_lead_manager.infrastructure.ai.openai_compatible.providers import (
    OpenAICompatibleDraftProvider,
    OpenAICompatibleQualificationProvider,
)
from ai_lead_manager.infrastructure.ai.openai_compatible.transport import (
    OpenAICompatibleHTTPTransport,
    OpenAIWireResponse,
)

ClientFactory = Callable[[OpenAICompatibleProviderConfig], httpx.Client | None]


class _DeferredOpenAICompatibleTransport:
    """Resolves secrets and creates HTTP state only for an actual provider call."""

    def __init__(
        self,
        config: OpenAICompatibleProviderConfig,
        credentials: CredentialResolver,
        client_factory: ClientFactory,
    ) -> None:
        self.config = config
        self.credentials = credentials
        self.client_factory = client_factory

    def post_chat_completion(self, payload: dict[str, Any]) -> OpenAIWireResponse:
        transport = OpenAICompatibleHTTPTransport(
            self.config,
            self.credentials.resolve(self.config.api_key_env_var),
            client=self.client_factory(self.config),
        )
        try:
            return transport.post_chat_completion(payload)
        finally:
            transport.close()


class OpenAICompatibleProviderFactory:
    """Builds allowlisted LIVE adapters without exposing credentials to application code."""

    def __init__(
        self,
        catalog: OpenAICompatibleProviderCatalog,
        credentials: CredentialResolver,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.catalog = catalog
        self.credentials = credentials
        self.client_factory = client_factory or (lambda _config: None)

    def qualification(
        self, provider_alias: str, model_alias: str
    ) -> OpenAICompatibleQualificationProvider:
        resolved = self.catalog.resolve(
            provider_alias, model_alias, ProviderCapability.QUALIFICATION
        )
        provider = OpenAICompatibleQualificationProvider(
            resolved,
            self._transport(resolved.provider),
        )
        provider.credential_configured = self.credentials.is_configured(
            resolved.provider.api_key_env_var
        )
        return provider

    def draft(
        self, provider_alias: str, model_alias: str
    ) -> OpenAICompatibleDraftProvider:
        resolved = self.catalog.resolve(provider_alias, model_alias, ProviderCapability.DRAFT)
        provider = OpenAICompatibleDraftProvider(
            resolved,
            self._transport(resolved.provider),
        )
        provider.credential_configured = self.credentials.is_configured(
            resolved.provider.api_key_env_var
        )
        return provider

    def _transport(
        self, config: OpenAICompatibleProviderConfig
    ) -> _DeferredOpenAICompatibleTransport:
        return _DeferredOpenAICompatibleTransport(
            config, self.credentials, self.client_factory
        )
