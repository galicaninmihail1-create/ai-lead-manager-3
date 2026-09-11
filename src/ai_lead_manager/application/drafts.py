from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class DraftContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_name: str
    task: str | None
    summary: str | None
    budget_amount: float | None
    budget_currency: str | None
    deadline: str | None


class DraftResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=20_000)


class DraftProvider(Protocol):
    provider_name: str
    model_name: str
    prompt_version: str

    def generate(self, context: DraftContext) -> DraftResult: ...


class MockDraftProvider:
    """Offline deterministic draft generator. It never sends the generated text."""

    provider_name = "mock"
    model_name = "deterministic-template-v1"
    prompt_version = "sales-draft-v1"

    def generate(self, context: DraftContext) -> DraftResult:
        task = context.task or "вашу задачу"
        details = []
        if context.budget_amount is not None and context.budget_currency:
            details.append(f"бюджет {context.budget_amount:g} {context.budget_currency}")
        if context.deadline:
            details.append(f"срок {context.deadline}")
        suffix = f" Учтём: {', '.join(details)}." if details else ""
        return DraftResult(
            content=(
                f"Здравствуйте, {context.customer_name}! Спасибо за обращение. "
                f"Мы готовы обсудить, как решить задачу «{task}»."
                f"{suffix} Предлагаю согласовать короткий звонок с менеджером."
            )
        )
