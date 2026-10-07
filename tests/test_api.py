from datetime import UTC, datetime

from fastapi.testclient import TestClient

from scrm.agents import orchestrator
from scrm.api import app
from scrm.schemas import RiskBrief

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


BODY = {"site_ids": ["P05"], "date_range": {"start": "2026-09-01", "end": "2026-09-07"}}


def test_brief_returns_orchestrator_result(monkeypatch, date_range):
    async def fake_run(request, ctx):
        return RiskBrief(
            run_id=ctx.run_id,
            generated_at=datetime.now(UTC),
            date_range=request.date_range,
            site_ids=request.site_ids,
            items=[],
            markdown="# Brief",
        )

    monkeypatch.setattr(orchestrator, "run", fake_run)
    response = client.post("/brief", json=BODY)
    assert response.status_code == 200
    assert response.json()["markdown"] == "# Brief"


def test_unknown_site_is_a_400_with_run_id(monkeypatch):
    async def fake_run(request, ctx):
        raise ValueError("unknown site_ids: ['P05']")

    monkeypatch.setattr(orchestrator, "run", fake_run)
    response = client.post("/brief", json=BODY)
    assert response.status_code == 400
    assert response.json()["detail"]["run_id"]
