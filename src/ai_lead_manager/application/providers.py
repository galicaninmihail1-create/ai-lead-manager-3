import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class QualificationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_name: str
    has_email: bool
    has_phone: bool
    messages: list[str] = Field(min_length=1)


class StructuredQualification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["purchase", "exploration", "support", "unknown"]
    task: str | None = None
    pain_points: list[str]
    budget_amount: float | None = Field(default=None, ge=0)
    budget_currency: str | None = Field(default=None, min_length=3, max_length=3)
    deadline: str | None = None
    objections: list[str]
    urgency: Literal["high", "medium", "low"]
    decision_readiness: Literal["ready", "considering", "researching"]
    summary: str
    confidence: float = Field(ge=0, le=1)


class QualificationProvider(Protocol):
    provider_name: str
    model_name: str
    schema_version: str
    prompt_version: str

    def analyze(self, context: QualificationContext) -> StructuredQualification: ...


class MockQualificationProvider:
    """Deterministic local extractor with no network capability."""

    provider_name = "mock"
    model_name = "deterministic-keyword-v1"
    schema_version = "1"
    prompt_version = "mock-qualification-v1"

    def analyze(self, context: QualificationContext) -> StructuredQualification:
        text = " ".join(context.messages)
        lowered = text.lower()
        budget_amount, budget_currency = self._budget(lowered)
        deadline = self._deadline(lowered)
        automation = any(word in lowered for word in ("автомат", "automation", "заявк", "lead"))
        pain_points = []
        if any(word in lowered for word in ("теряем", "потер", "не успева", "медленно")):
            pain_points.append("Потери или медленная обработка входящих обращений")
        if automation:
            pain_points.append("Ручная обработка входящих обращений")

        purchase_words = ("нужно", "хочу", "готов", "ищем", "купить", "внедрить", "need", "buy")
        intent = "purchase" if any(word in lowered for word in purchase_words) else "exploration"
        task = "Автоматизировать обработку входящих обращений" if automation else None
        objections = [
            label
            for marker, label in (
                ("дорого", "Сомнение в стоимости"),
                ("сомнева", "Неуверенность в решении"),
                ("риск", "Опасение по рискам"),
            )
            if marker in lowered
        ]
        if any(word in lowered for word in ("срочно", "urgent", "как можно скорее")):
            urgency = "high"
        elif deadline is not None:
            urgency = "medium"
        else:
            urgency = "low"
        if intent == "purchase" and budget_amount is not None and deadline is not None:
            readiness = "ready"
        elif intent == "purchase":
            readiness = "considering"
        else:
            readiness = "researching"

        facts = [task or "Требуется уточнить задачу"]
        if budget_amount is not None:
            facts.append(f"бюджет {budget_amount:g} {budget_currency}")
        if deadline:
            facts.append(f"срок: {deadline}")
        return StructuredQualification(
            intent=intent,
            task=task,
            pain_points=pain_points,
            budget_amount=budget_amount,
            budget_currency=budget_currency,
            deadline=deadline,
            objections=objections,
            urgency=urgency,
            decision_readiness=readiness,
            summary="; ".join(facts),
            confidence=0.9 if automation else 0.65,
        )

    @staticmethod
    def _budget(text: str) -> tuple[float | None, str | None]:
        match = re.search(
            r"(?:бюджет\D{0,15})?(\d[\d ]{0,12})\s*(тыс(?:яч[аи]?)?|руб(?:лей|ля)?|₽|usd|доллар\w*)",
            text,
        )
        if not match:
            return None, None
        amount = float(match.group(1).replace(" ", ""))
        unit = match.group(2)
        if unit.startswith("тыс"):
            amount *= 1000
        currency = "USD" if unit == "usd" or unit.startswith("доллар") else "RUB"
        return amount, currency

    @staticmethod
    def _deadline(text: str) -> str | None:
        match = re.search(r"(?:за|в течение|срок\D{0,10})\s*(\d+)\s*(дн\w*|недел\w*|месяц\w*)", text)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        for phrase in ("до конца месяца", "как можно скорее"):
            if phrase in text:
                return phrase
        return None
