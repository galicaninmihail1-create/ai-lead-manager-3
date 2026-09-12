from decimal import ROUND_DOWN, Decimal, localcontext

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
from ai_lead_manager.application.ai.money import (
    POSTGRES_BIGINT_MAX,
    Currency,
    CurrencyMismatchError,
    Money,
)
from ai_lead_manager.application.ai.pricing import ModelPricing, PricingCatalog
from ai_lead_manager.application.ai.provider_cost import ExactProviderCost


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


@pytest.mark.parametrize(
    ("usage", "expected_micros"),
    [
        (AIUsage(input_tokens=0, output_tokens=0), 0),
        (AIUsage(input_tokens=1, output_tokens=0), 49),
        (AIUsage(input_tokens=12_345, output_tokens=0), 599_227),
        (AIUsage(input_tokens=10, output_tokens=0, cached_tokens=10), 486),
        (AIUsage(input_tokens=17, output_tokens=3, cached_tokens=5), 1_409),
    ],
)
def test_polza_pricing_uses_integer_ceiling_per_token_class(
    usage: AIUsage, expected_micros: int
) -> None:
    pricing = ModelPricing(
        "polza",
        "lead-structured-v1",
        "polza-gpt41mini-rub-20260912-r1",
        Currency.RUB,
        48_540_000,
        194_160_000,
        48_540_000,
    )

    cost = pricing.actual_cost(usage)
    assert cost == Money(amount_micros=expected_micros, currency=Currency.RUB)
    assert isinstance(cost.amount_micros, int)


def test_cached_tokens_cannot_exceed_input_tokens() -> None:
    with pytest.raises(ValueError, match="Cached tokens cannot exceed input tokens"):
        AIUsage(input_tokens=2, output_tokens=0, cached_tokens=3)


def test_money_and_pricing_reject_values_above_postgres_bigint() -> None:
    Money(amount_micros=POSTGRES_BIGINT_MAX, currency=Currency.RUB)
    with pytest.raises(ValueError):
        Money(amount_micros=POSTGRES_BIGINT_MAX + 1, currency=Currency.RUB)
    with pytest.raises(ValueError, match="fit PostgreSQL BIGINT"):
        ModelPricing("provider", "model", "v1", Currency.RUB, POSTGRES_BIGINT_MAX + 1, 0, 0)


def test_cost_result_cannot_exceed_postgres_bigint() -> None:
    pricing = ModelPricing(
        "provider", "model", "v1", Currency.RUB, POSTGRES_BIGINT_MAX, 0, 0
    )
    with pytest.raises(ValueError):
        pricing.actual_cost(AIUsage(input_tokens=1_000_001, output_tokens=0))


@pytest.mark.parametrize(
    ("raw", "expected_micros"),
    [
        (Decimal("0"), 0),
        (Decimal("0.000001"), 1),
        (Decimal("0.00000001"), 1),
        (Decimal("1"), 1_000_000),
        (Decimal("0.04131306"), 41_314),
        (Decimal("0.123456789012345678"), 123_457),
        (Decimal("9000000000000"), 9_000_000_000_000_000_000),
        ("4.131306e-2", 41_314),
    ],
)
def test_exact_provider_cost_normalizes_up_without_float(
    raw: Decimal | str, expected_micros: int
) -> None:
    cost = ExactProviderCost.parse(raw, Currency.RUB)

    assert cost.normalized_money == Money(
        amount_micros=expected_micros, currency=Currency.RUB
    )
    assert isinstance(cost.amount, Decimal)


@pytest.mark.parametrize(
    "raw",
    [
        Decimal("0.1234567890123456789"),
        Decimal("100000000000000"),
        Decimal("-0.01"),
        Decimal("NaN"),
        Decimal("Infinity"),
        "not-a-number",
        0.04131306,
        True,
    ],
)
def test_exact_provider_cost_rejects_invalid_or_unbounded_values(raw: object) -> None:
    with pytest.raises(ValueError):
        ExactProviderCost.parse(raw, Currency.RUB)  # type: ignore[arg-type]


def test_exact_provider_cost_rejects_normalized_bigint_overflow() -> None:
    with pytest.raises(ValueError, match="PostgreSQL BIGINT"):
        ExactProviderCost.parse(9_223_372_036_855, Currency.RUB)


def test_exact_provider_cost_ignores_ambient_decimal_rounding() -> None:
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        normalized = ExactProviderCost.parse(
            Decimal("0.04131306"), Currency.RUB
        ).normalized_money

    assert normalized.amount_micros == 41_314


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
