from dataclasses import dataclass

from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIExecutionMode,
    AIOperationType,
    ProviderDisabledError,
    StructuredAIProvider,
)


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    operation_type: AIOperationType
    provider: str
    model: str
    execution_mode: AIExecutionMode
    prompt_version: str
    schema_version: str
    pricing_version: str | None
    billing_currency: str | None
    credential_configured: bool | None
    enforce_request_contract: bool


class ProviderRegistry:
    def __init__(self, providers: list[StructuredAIProvider]) -> None:
        self._providers: dict[
            tuple[AIOperationType | None, str, str], StructuredAIProvider
        ] = {}
        for provider in providers:
            operation_type = getattr(provider, "operation_type", None)
            key = (operation_type, provider.provider_name, provider.model_name)
            if key in self._providers:
                raise ValueError("Duplicate provider/model/operation registration")
            self._providers[key] = provider

    def get(
        self,
        provider: str,
        model: str,
        operation_type: AIOperationType | None = None,
    ) -> StructuredAIProvider:
        implementation = self._providers.get((operation_type, provider, model))
        if implementation is None and operation_type is not None:
            implementation = self._providers.get((None, provider, model))
        if implementation is None:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "Requested provider/model/operation is not enabled",
            )
        return implementation

    def route(
        self, operation_type: AIOperationType, provider: str, model: str
    ) -> ProviderRoute:
        implementation = self.get(provider, model, operation_type)
        prompt_version = getattr(implementation, "prompt_version", None)
        schema_version = getattr(implementation, "schema_version", None)
        if not prompt_version or not schema_version:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "Registered provider is missing its request contract",
            )
        return ProviderRoute(
            operation_type=operation_type,
            provider=provider,
            model=model,
            execution_mode=implementation.execution_mode,
            prompt_version=prompt_version,
            schema_version=schema_version,
            pricing_version=getattr(implementation, "pricing_version", None),
            billing_currency=getattr(implementation, "billing_currency", None),
            credential_configured=getattr(implementation, "credential_configured", None),
            enforce_request_contract=getattr(implementation, "operation_type", None)
            is not None,
        )
