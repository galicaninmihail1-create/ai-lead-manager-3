import socket

from ai_lead_manager.application.providers import MockQualificationProvider, QualificationContext


def context(message: str) -> QualificationContext:
    return QualificationContext(
        customer_name="Анна",
        has_email=True,
        has_phone=True,
        messages=[message],
    )


def test_mock_provider_is_deterministic_and_structured() -> None:
    message = (
        "Нужно срочно автоматизировать входящие заявки, теряем клиентов. "
        "Бюджет 300 000 рублей, хотим внедрить за 2 недели."
    )
    provider = MockQualificationProvider()

    first = provider.analyze(context(message))
    second = provider.analyze(context(message))

    assert first == second
    assert first.intent == "purchase"
    assert first.budget_amount == 300_000
    assert first.budget_currency == "RUB"
    assert first.deadline == "2 недели"
    assert first.urgency == "high"
    assert first.decision_readiness == "ready"
    assert first.pain_points


def test_mock_provider_performs_no_network_calls(monkeypatch) -> None:
    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    result = MockQualificationProvider().analyze(context("Хочу автоматизировать заявки"))
    assert result.intent == "purchase"
