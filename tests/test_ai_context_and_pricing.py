import pytest

from ai_lead_manager.application.ai.circuit import CircuitBreaker, CircuitState
from ai_lead_manager.application.ai.context import (
    BoundedContextBuilder,
    canonical_request_hash,
    estimate_tokens,
)
from ai_lead_manager.application.ai.contracts import (
    AIErrorKind,
    AIProviderError,
    AIUsage,
    ProviderDisabledError,
    UnknownCostError,
)
from ai_lead_manager.application.ai.money import Currency, CurrencyMismatchError, Money
from ai_lead_manager.application.ai.pricing import ModelPricing, PricingCatalog


def test_bounded_context_keeps_newest_messages_deterministically() -> None:
    builder = BoundedContextBuilder(max_messages=2, max_characters=12, max_input_tokens=20)
    messages = ["old-message", "middle", "newest-message"]

    first = builder.build(messages)
    second = builder.build(messages)

    assert first == second
    assert first.truncated is True
    assert first.messages[-1].endswith("newest-message"[-12:])
    assert len(first.messages) <= 2
    assert first.character_count <= 12
    assert first.estimated_tokens <= 20


def test_context_token_limit_truncates_oversized_unicode_message() -> None:
    result = BoundedContextBuilder(10, 10_000, 5).build(["важный контекст " * 50])
    assert result.truncated is True
    assert result.estimated_tokens <= 5
    assert result.messages


def test_request_hash_is_canonical() -> None:
    assert canonical_request_hash({"b": 2, "a": 1}) == canonical_request_hash(
        {"a": 1, "b": 2}
    )
    assert len(canonical_request_hash({"message": "Привет"})) == 64


def test_integer_cost_arithmetic_is_exact() -> None:
    pricing = ModelPricing(
        "mock", "model", "v1", Currency.USD, 1_000_000, 2_000_000, 250_000
    )
    usage = AIUsage(input_tokens=10, output_tokens=3, cached_tokens=4)

    assert pricing.actual_cost(usage) == Money(amount_micros=13, currency=Currency.USD)
    assert pricing.worst_case_cost(10, 5, 2) == Money(
        amount_micros=40, currency=Currency.USD
    )
    assert isinstance(pricing.actual_cost(usage).amount_micros, int)


def test_rub_pricing_and_cached_token_arithmetic_are_exact() -> None:
    pricing = ModelPricing(
        "polza-compatible",
        "rub-model",
        "rub-v1",
        Currency.RUB,
        1_500_001,
        2_500_003,
        500_007,
    )

    assert pricing.actual_cost(
        AIUsage(input_tokens=1_000_000, output_tokens=3, cached_tokens=250_000)
    ) == Money(amount_micros=1_250_011, currency=Currency.RUB)
    assert pricing.worst_case_cost(10, 5, 2) == Money(
        amount_micros=58, currency=Currency.RUB
    )


def test_money_never_combines_or_compares_different_currencies() -> None:
    usd = Money(amount_micros=1, currency=Currency.USD)
    rub = Money(amount_micros=1, currency=Currency.RUB)

    with pytest.raises(CurrencyMismatchError):
        _ = usd + rub
    with pytest.raises(CurrencyMismatchError):
        _ = usd < rub
    with pytest.raises(CurrencyMismatchError):
        _ = usd == rub


def test_money_rejects_float_amounts() -> None:
    with pytest.raises(ValueError):
        Money(amount_micros=1.5, currency=Currency.RUB)


def test_unknown_price_fails_closed() -> None:
    with pytest.raises(UnknownCostError):
        PricingCatalog([]).get("unknown", "expensive-model")


def test_token_estimator_is_deterministic_and_positive() -> None:
    assert estimate_tokens("abc") == estimate_tokens("abc")
    assert estimate_tokens("") == 1


def test_circuit_breaker_opens_and_allows_one_half_open_probe() -> None:
    now = 10.0
    breaker = CircuitBreaker(
        failure_threshold=2,
        reset_after_seconds=5,
        clock=lambda: now,
    )
    error = AIProviderError(AIErrorKind.TRANSPORT, "offline", safe_to_retry=True)

    breaker.record_failure(error)
    breaker.record_failure(error)
    assert breaker.state == CircuitState.OPEN
    with pytest.raises(ProviderDisabledError):
        breaker.before_call()

    now = 16.0
    breaker.before_call()
    assert breaker.state == CircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
