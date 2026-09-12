import json
import re
from enum import StrEnum

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ai_lead_manager.application.ai.contracts import AIErrorKind, ProviderDisabledError
from ai_lead_manager.application.ai.money import Currency


class ProviderCapability(StrEnum):
    QUALIFICATION = "qualification"
    DRAFT = "draft"
    STRICT_STRUCTURED_OUTPUT = "strict_structured_output"


class OpenAIHTTPTimeoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    connect_seconds: float = Field(default=3.0, gt=0, le=30)
    read_seconds: float = Field(default=15.0, gt=0, le=120)
    write_seconds: float = Field(default=10.0, gt=0, le=120)
    pool_seconds: float = Field(default=3.0, gt=0, le=30)


class OpenAICompatibleModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    alias: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9._-]+$")
    upstream_model: str = Field(min_length=1, max_length=200)
    accepted_provider_models: frozenset[str] = Field(default_factory=frozenset)
    pricing_version: str = Field(min_length=1, max_length=50)
    billing_currency: Currency | None = None
    capabilities: frozenset[ProviderCapability] = Field(min_length=1)

    @field_validator("accepted_provider_models")
    @classmethod
    def validate_provider_model_names(cls, value: frozenset[str]) -> frozenset[str]:
        if any(not item or len(item) > 200 for item in value):
            raise ValueError("Accepted provider model identifiers must be bounded")
        return value

    @property
    def provider_model_allowlist(self) -> frozenset[str]:
        return self.accepted_provider_models | {self.upstream_model}


class OpenAICompatibleProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    alias: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9._-]+$")
    base_url: AnyHttpUrl
    api_key_env_var: str = Field(min_length=1, max_length=200)
    models: tuple[OpenAICompatibleModelConfig, ...] = Field(min_length=1)
    timeouts: OpenAIHTTPTimeoutConfig = Field(default_factory=OpenAIHTTPTimeoutConfig)
    chat_completions_path: str = Field(default="/v1/chat/completions", pattern=r"^/[A-Za-z0-9_./-]+$")
    max_response_bytes: int = Field(default=1_000_000, ge=1_024, le=10_000_000)
    safe_retry_http_statuses: frozenset[int] = Field(default_factory=frozenset)

    @field_validator("base_url")
    @classmethod
    def require_https(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.scheme != "https":
            raise ValueError("OpenAI-compatible LIVE providers require HTTPS")
        if value.username or value.password or value.query or value.fragment:
            raise ValueError(
                "Provider base URL cannot contain credentials, query parameters, or fragments"
            )
        return value

    @field_validator("api_key_env_var")
    @classmethod
    def validate_env_reference(cls, value: str) -> str:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", value) is None:
            raise ValueError("Credential reference must be an uppercase environment variable name")
        return value

    @field_validator("chat_completions_path")
    @classmethod
    def reject_duplicate_path_separators(cls, value: str) -> str:
        if "//" in value:
            raise ValueError("Chat completions path cannot contain duplicate slashes")
        return value

    @field_validator("safe_retry_http_statuses")
    @classmethod
    def validate_safe_statuses(cls, value: frozenset[int]) -> frozenset[int]:
        if any(status < 500 or status > 599 for status in value):
            raise ValueError("Only explicitly reviewed 5xx statuses can be marked retry-safe")
        return value

    @model_validator(mode="after")
    def unique_model_aliases(self) -> "OpenAICompatibleProviderConfig":
        aliases = [model.alias for model in self.models]
        if len(set(aliases)) != len(aliases):
            raise ValueError("Model aliases must be unique within a provider")
        return self


class ResolvedOpenAICompatibleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: OpenAICompatibleProviderConfig
    model: OpenAICompatibleModelConfig


class OpenAICompatibleProviderCatalog:
    def __init__(self, providers: tuple[OpenAICompatibleProviderConfig, ...]) -> None:
        self._providers = {provider.alias: provider for provider in providers}
        if len(self._providers) != len(providers):
            raise ValueError("Provider aliases must be unique")

    @property
    def providers(self) -> tuple[OpenAICompatibleProviderConfig, ...]:
        return tuple(self._providers.values())

    @classmethod
    def from_json(cls, raw_configuration: str) -> "OpenAICompatibleProviderCatalog":
        try:
            raw = json.loads(raw_configuration)
            if not isinstance(raw, list):
                raise ValueError("Provider configuration must be a JSON list")
            providers = tuple(OpenAICompatibleProviderConfig.model_validate(item) for item in raw)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as error:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "OpenAI-compatible provider configuration is invalid",
            ) from error
        return cls(providers)

    def resolve(
        self,
        provider_alias: str,
        model_alias: str,
        capability: ProviderCapability,
    ) -> ResolvedOpenAICompatibleModel:
        provider = self._providers.get(provider_alias)
        if provider is None:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "Requested provider alias is not server-allowlisted",
            )
        model = next((item for item in provider.models if item.alias == model_alias), None)
        if model is None:
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "Requested model alias is not server-allowlisted",
            )
        required = {capability, ProviderCapability.STRICT_STRUCTURED_OUTPUT}
        if not required.issubset(model.capabilities):
            raise ProviderDisabledError(
                AIErrorKind.PROVIDER_DISABLED,
                "Requested model does not provide the required capability",
            )
        return ResolvedOpenAICompatibleModel(provider=provider, model=model)
