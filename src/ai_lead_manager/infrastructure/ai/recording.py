from collections.abc import Callable

from ai_lead_manager.application.ai.contracts import (
    AIExecutionMode,
    AIOperationType,
    AIProviderError,
    StructuredAIRequest,
    StructuredAIResult,
)


class RecordingStructuredProvider:
    """Offline provider double that records physical calls and can replay scripted outcomes."""

    execution_mode = AIExecutionMode.MOCK

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        handler: Callable[[StructuredAIRequest], StructuredAIResult],
        scripted_outcomes: list[StructuredAIResult | AIProviderError] | None = None,
        execution_mode: AIExecutionMode = AIExecutionMode.MOCK,
        operation_type: AIOperationType | None = None,
        prompt_version: str = "test-v1",
        schema_version: str = "1",
        pricing_version: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.handler = handler
        self.scripted_outcomes = list(scripted_outcomes or [])
        self.requests: list[StructuredAIRequest] = []
        self.execution_mode = execution_mode
        self.operation_type = operation_type
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.pricing_version = pricing_version

    @property
    def physical_call_count(self) -> int:
        return len(self.requests)

    def execute(self, request: StructuredAIRequest) -> StructuredAIResult:
        self.requests.append(request)
        if self.scripted_outcomes:
            outcome = self.scripted_outcomes.pop(0)
            if isinstance(outcome, AIProviderError):
                raise outcome
            return outcome
        return self.handler(request)
