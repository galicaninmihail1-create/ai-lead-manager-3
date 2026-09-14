from fastapi.testclient import TestClient

from ai_lead_manager.main import create_app


def test_web_ui_serves_lead_form_and_existing_api_flow_targets() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert 'id="lead-form"' in response.text
    assert "/api/v1/inbound/http" in response.text
    assert "/qualify" in response.text
    assert "Создать и квалифицировать" in response.text
    assert "ID аккаунта" in response.text
    assert "Готовность к решению" in response.text
    assert "schedule_call: 'Назначить звонок'" in response.text
    assert "formatConfidence" in response.text
    assert "formatBudget" in response.text
