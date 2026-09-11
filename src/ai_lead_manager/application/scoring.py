from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_lead_manager.application.providers import StructuredQualification

RULESET_VERSION = "qualification-v1"


class ScoreContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: StructuredQualification
    has_email: bool = False
    has_phone: bool = False


class CalculatedFactor(BaseModel):
    factor_name: str
    raw_value: str
    contribution: int
    explanation: str


class ScoreResult(BaseModel):
    score: int = Field(ge=0, le=100)
    temperature: Literal["HOT", "WARM", "COLD"]
    ruleset_version: str = RULESET_VERSION
    factors: list[CalculatedFactor]


class NextAction(BaseModel):
    action_type: str
    reason: str
    priority: int = Field(ge=1, le=3)


def temperature_for(score: int) -> Literal["HOT", "WARM", "COLD"]:
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    if score >= 70:
        return "HOT"
    if score >= 40:
        return "WARM"
    return "COLD"


def calculate_score(context: ScoreContext) -> ScoreResult:
    analysis = context.analysis
    def factor(name: str, raw: str, contribution: int, explanation: str) -> CalculatedFactor:
        return CalculatedFactor(
            factor_name=name,
            raw_value=raw,
            contribution=contribution,
            explanation=explanation,
        )

    factors = [
        factor("budget_quality", str(analysis.budget_amount or "missing"), 20 if analysis.budget_amount is not None else 0, "Explicit budget: 20 points."),
        factor("urgency", analysis.urgency, {"high": 20, "medium": 10, "low": 0}[analysis.urgency], "High: 20; medium: 10; low: 0."),
        factor("intent_strength", analysis.intent, {"purchase": 20, "exploration": 10, "support": 0, "unknown": 0}[analysis.intent], "Purchase: 20; exploration: 10."),
        factor("problem_clarity", f"task={bool(analysis.task)},pain_points={len(analysis.pain_points)}", 15 if analysis.task and analysis.pain_points else 8 if analysis.task else 0, "Task with pain points: 15; task alone: 8."),
        factor("decision_readiness", analysis.decision_readiness, {"ready": 15, "considering": 8, "researching": 0}[analysis.decision_readiness], "Ready: 15; considering: 8."),
        factor("objection_risk", str(len(analysis.objections)), -min(15, len(analysis.objections) * 5), "Each objection: -5, capped at -15."),
        factor("contact_completeness", f"email={context.has_email},phone={context.has_phone}", 10 if context.has_email and context.has_phone else 5 if context.has_email or context.has_phone else 0, "Both contacts: 10; one: 5."),
    ]
    total = sum(factor.contribution for factor in factors)
    if total < 0:
        factors.append(factor("score_floor", str(total), -total, "Score cannot be below zero."))
        total = 0
    return ScoreResult(score=total, temperature=temperature_for(total), factors=factors)


def recommend_next_action(result: ScoreResult) -> NextAction:
    if result.temperature == "HOT":
        return NextAction(action_type="schedule_call", reason="Высокий score: подтверждены намерение и готовность к решению.", priority=1)
    if result.temperature == "WARM":
        return NextAction(action_type="clarify_requirements", reason="Нужно уточнить бюджет, срок или детали задачи.", priority=2)
    return NextAction(action_type="collect_context", reason="Недостаточно признаков готовности; нужно собрать контекст.", priority=3)
