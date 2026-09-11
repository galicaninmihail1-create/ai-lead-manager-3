import pytest

from ai_lead_manager.application.providers import StructuredQualification
from ai_lead_manager.application.scoring import (
    ScoreContext,
    calculate_score,
    recommend_next_action,
    temperature_for,
)


def analysis(**changes) -> StructuredQualification:
    values = {
        "intent": "purchase",
        "task": "Автоматизировать заявки",
        "pain_points": ["Ручная обработка"],
        "budget_amount": 300_000,
        "budget_currency": "RUB",
        "deadline": "2 недели",
        "objections": [],
        "urgency": "high",
        "decision_readiness": "ready",
        "summary": "Готовый лид",
        "confidence": 0.9,
    }
    values.update(changes)
    return StructuredQualification(**values)


@pytest.mark.parametrize(
    ("score", "temperature"),
    [(0, "COLD"), (39, "COLD"), (40, "WARM"), (69, "WARM"), (70, "HOT"), (100, "HOT")],
)
def test_temperature_exact_boundaries(score: int, temperature: str) -> None:
    assert temperature_for(score) == temperature


@pytest.mark.parametrize("score", [-1, 101])
def test_temperature_rejects_out_of_range_score(score: int) -> None:
    with pytest.raises(ValueError):
        temperature_for(score)


def test_hot_score_is_reproducible_from_factors() -> None:
    result = calculate_score(ScoreContext(analysis=analysis(), has_email=True, has_phone=True))
    assert result.score == 100
    assert result.temperature == "HOT"
    assert sum(factor.contribution for factor in result.factors) == result.score
    assert recommend_next_action(result).action_type == "schedule_call"


def test_warm_score_and_action() -> None:
    result = calculate_score(
        ScoreContext(
            analysis=analysis(
                budget_amount=None,
                budget_currency=None,
                urgency="medium",
                decision_readiness="considering",
            ),
            has_email=True,
        )
    )
    assert result.temperature == "WARM"
    assert recommend_next_action(result).action_type == "clarify_requirements"


def test_cold_score_and_action() -> None:
    result = calculate_score(
        ScoreContext(
            analysis=analysis(
                intent="unknown",
                task=None,
                pain_points=[],
                budget_amount=None,
                budget_currency=None,
                deadline=None,
                urgency="low",
                decision_readiness="researching",
                objections=["Стоимость", "Риск", "Срок"],
            )
        )
    )
    assert 0 <= result.score <= 39
    assert result.temperature == "COLD"
    assert sum(factor.contribution for factor in result.factors) == result.score
    assert recommend_next_action(result).action_type == "collect_context"
