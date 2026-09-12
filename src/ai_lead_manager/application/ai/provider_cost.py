from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from ai_lead_manager.application.ai.money import POSTGRES_BIGINT_MAX, Currency, Money

MAX_PROVIDER_COST_INTEGER_DIGITS = 14
MAX_PROVIDER_COST_FRACTIONAL_DIGITS = 18
_MICROS_PER_UNIT = 1_000_000
_DECIMAL_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)


@dataclass(frozen=True, slots=True)
class ExactProviderCost:
    """Exact provider-reported cost plus its conservative micro-unit normalization."""

    amount: Decimal
    currency: Currency

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal) or isinstance(self.amount, bool):
            raise ValueError("Provider cost amount must be an exact Decimal")
        if not isinstance(self.currency, Currency):
            raise ValueError("Provider cost currency must be supported")
        if not self.amount.is_finite():
            raise ValueError("Provider cost amount must be finite")
        if self.amount < 0:
            raise ValueError("Provider cost amount must be nonnegative")

        exponent = self.amount.as_tuple().exponent
        fractional_digits = max(-exponent, 0)
        integer_digits = max(self.amount.adjusted() + 1, 0) if self.amount else 0
        if fractional_digits > MAX_PROVIDER_COST_FRACTIONAL_DIGITS:
            raise ValueError("Provider cost exceeds NUMERIC(32,18) fractional precision")
        if integer_digits > MAX_PROVIDER_COST_INTEGER_DIGITS:
            raise ValueError("Provider cost exceeds NUMERIC(32,18) integer precision")

        # Validate the future persisted normalization at construction time.
        _ = self.normalized_money

    @classmethod
    def parse(cls, value: Decimal | int | str, currency: Currency) -> ExactProviderCost:
        if isinstance(value, bool) or isinstance(value, float):
            raise ValueError("Provider cost must not use bool or binary float")
        if isinstance(value, Decimal):
            amount = value
        elif type(value) is int:
            amount = Decimal(value)
        elif isinstance(value, str) and _DECIMAL_PATTERN.fullmatch(value) is not None:
            try:
                amount = Decimal(value)
            except InvalidOperation as error:
                raise ValueError("Provider cost is not a valid decimal") from error
        else:
            raise ValueError("Provider cost is not a valid decimal")
        return cls(amount=amount, currency=currency)

    @property
    def normalized_money(self) -> Money:
        numerator, denominator = self.amount.as_integer_ratio()
        scaled_numerator = numerator * _MICROS_PER_UNIT
        amount_micros = (scaled_numerator + denominator - 1) // denominator
        if amount_micros > POSTGRES_BIGINT_MAX:
            raise ValueError("Normalized provider cost exceeds PostgreSQL BIGINT")
        return Money(amount_micros=amount_micros, currency=self.currency)

    @property
    def canonical_amount(self) -> str:
        return format(self.amount, "f")
