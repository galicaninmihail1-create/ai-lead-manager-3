from dataclasses import dataclass

from ai_lead_manager.application.ai.contracts import AIErrorKind, AIUsage, UnknownCostError

MICROUSD_PER_USD = 1_000_000


@dataclass(frozen=True, slots=True)
class ModelPricing:
    provider: str
    model: str
    version: str
    input_microusd_per_million_tokens: int
    output_microusd_per_million_tokens: int
    cached_microusd_per_million_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.input_microusd_per_million_tokens,
            self.output_microusd_per_million_tokens,
            self.cached_microusd_per_million_tokens,
        )
        if not self.provider or not self.model or not self.version or any(value < 0 for value in values):
            raise ValueError("Model pricing must be named, versioned, and non-negative")

    @staticmethod
    def _charge(tokens: int, rate: int) -> int:
        if tokens < 0:
            raise ValueError("Token counts cannot be negative")
        return (tokens * rate + MICROUSD_PER_USD - 1) // MICROUSD_PER_USD

    def actual_cost(self, usage: AIUsage) -> int:
        uncached = max(usage.input_tokens - usage.cached_tokens, 0)
        return (
            self._charge(uncached, self.input_microusd_per_million_tokens)
            + self._charge(usage.cached_tokens, self.cached_microusd_per_million_tokens)
            + self._charge(usage.output_tokens, self.output_microusd_per_million_tokens)
        )

    def worst_case_cost(
        self, estimated_input_tokens: int, max_output_tokens: int, max_attempts: int
    ) -> int:
        if max_attempts < 1:
            raise ValueError("At least one attempt must be reserved")
        per_attempt = self._charge(
            estimated_input_tokens, self.input_microusd_per_million_tokens
        ) + self._charge(max_output_tokens, self.output_microusd_per_million_tokens)
        return per_attempt * max_attempts


class PricingCatalog:
    def __init__(self, prices: list[ModelPricing]) -> None:
        self._prices = {(price.provider, price.model): price for price in prices}
        if len(self._prices) != len(prices):
            raise ValueError("Duplicate provider/model pricing entry")

    def get(self, provider: str, model: str) -> ModelPricing:
        price = self._prices.get((provider, model))
        if price is None:
            raise UnknownCostError(
                AIErrorKind.UNKNOWN_COST,
                f"No server-controlled price for provider={provider!r}, model={model!r}",
            )
        return price


def default_mock_pricing_catalog() -> PricingCatalog:
    return PricingCatalog(
        [
            ModelPricing(
                provider="mock",
                model="deterministic-keyword-v1",
                version="mock-pricing-v1",
                input_microusd_per_million_tokens=1_000_000,
                output_microusd_per_million_tokens=2_000_000,
                cached_microusd_per_million_tokens=250_000,
            ),
            ModelPricing(
                provider="mock",
                model="deterministic-template-v1",
                version="mock-pricing-v1",
                input_microusd_per_million_tokens=1_000_000,
                output_microusd_per_million_tokens=2_000_000,
                cached_microusd_per_million_tokens=250_000,
            ),
        ]
    )
