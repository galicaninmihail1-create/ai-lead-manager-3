from typing import Protocol

from pydantic import BaseModel


class LeadQualification(BaseModel):
    score: int
    summary: str


class LeadQualificationProvider(Protocol):
    """Boundary for a future AI or rules-based qualification provider."""

    def qualify(self, context: str) -> LeadQualification: ...


class MockLeadQualificationProvider:
    """Deterministic local provider; performs no network or paid API calls."""

    def qualify(self, context: str) -> LeadQualification:
        return LeadQualification(
            score=50 if context.strip() else 0,
            summary="Mock qualification for local development",
        )

