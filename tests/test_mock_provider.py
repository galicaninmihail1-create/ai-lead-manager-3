from ai_lead_manager.application.providers import MockLeadQualificationProvider


def test_mock_provider_is_deterministic_and_local() -> None:
    result = MockLeadQualificationProvider().qualify("Interested in the product")

    assert result.score == 50
    assert result.summary == "Mock qualification for local development"
