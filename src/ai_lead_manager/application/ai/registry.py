from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    ProviderDisabledError,
    StructuredAIProvider,
)


class ProviderRegistry:
    def __init__(self, providers: list[StructuredAIProvider]) -> None:
        self._providers = {(provider.provider_name, provider.model_name): provider for provider in providers}
        if len(self._providers) != len(providers):
            raise ValueError("Duplicate provider/model registration")

    def get(self, provider: str, model: str) -> StructuredAIProvider:
        implementation = self._providers.get((provider, model))
        if implementation is None:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                f"Provider/model is not enabled: {provider}/{model}",
            )
        return implementation
