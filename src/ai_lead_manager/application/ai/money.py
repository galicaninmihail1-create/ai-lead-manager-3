from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Currency(StrEnum):
    USD = "USD"
    RUB = "RUB"


class CurrencyMismatchError(ValueError):
    pass


class Money(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    amount_micros: int = Field(ge=0, strict=True)
    currency: Currency

    def _require_same_currency(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(
                f"Cannot combine {self.currency.value} and {other.currency.value} amounts"
            )

    def __add__(self, other: object) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return Money(
            amount_micros=self.amount_micros + other.amount_micros,
            currency=self.currency,
        )

    def __sub__(self, other: object) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return Money(
            amount_micros=self.amount_micros - other.amount_micros,
            currency=self.currency,
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_micros < other.amount_micros

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_micros == other.amount_micros

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_micros <= other.amount_micros

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_micros > other.amount_micros

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._require_same_currency(other)
        return self.amount_micros >= other.amount_micros

    @classmethod
    def zero(cls, currency: Currency) -> Money:
        return cls(amount_micros=0, currency=currency)
