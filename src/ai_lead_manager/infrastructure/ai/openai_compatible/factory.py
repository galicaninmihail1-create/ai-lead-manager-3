from collections.abc import Callable

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
)

ClientFactory = Callable[[OpenAICompatibleProviderConfig], httpx.Client | None]


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
        return OpenAICompatibleQualificationProvider(
            resolved,
            self._transport(resolved.provider),
        )

    def draft(
        self, provider_alias: str, model_alias: str
    ) -> OpenAICompatibleDraftProvider:
        resolved = self.catalog.resolve(provider_alias, model_alias, ProviderCapability.DRAFT)
        return OpenAICompatibleDraftProvider(
            resolved,
            self._transport(resolved.provider),
        )

    def _transport(
        self, config: OpenAICompatibleProviderConfig
    ) -> OpenAICompatibleHTTPTransport:
        credential = self.credentials.resolve(config.api_key_env_var)
        return OpenAICompatibleHTTPTransport(
            config,
            credential,
            client=self.client_factory(config),
        )
