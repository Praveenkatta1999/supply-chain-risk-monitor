from fastapi.testclient import TestClient

from scrm.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_brief_is_stubbed_with_run_id():
    body = {"site_ids": ["SUP-01"], "date_range": {"start": "2026-09-01", "end": "2026-09-07"}}
    response = client.post("/brief", json=body)
    assert response.status_code == 501
    assert response.json()["detail"]["run_id"]
