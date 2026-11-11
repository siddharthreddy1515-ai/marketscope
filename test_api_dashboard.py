
from fastapi.testclient import TestClient
import main


def test_analyze_returns_research_data(monkeypatch):

    async def fake_workflow(company_name):
        return {
            "status": "completed",
            "report": "# Example report",
            "research_attempts": 1,
            "sources": [
                {
                    "title": "Example article",
                    "url": "https://example.com/article",
                    "content": "Example company announcement.",
                    "published_at": None,
                }
            ],
            "source_stats": {
                "returned": 1,
                "accepted": 1,
                "discarded": 0,
            },
            "research_warnings": [],
        }

    monkeypatch.setattr(
        main, "run_workflow", fake_workflow
    )

    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        json={"company_name": "Microsoft"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["report"] == "# Example report"
    assert data["research_attempts"] == 1
    assert len(data["sources"]) == 1
    assert data["source_stats"]["accepted"] == 1


def test_failed_research_returns_error(monkeypatch):

    async def fake_workflow(company_name):
        return {
            "status": "failed",
            "error_code": "RESEARCH_EXHAUSTED",
            "report": None,
            "sources": [],
        }

    monkeypatch.setattr(
        main, "run_workflow", fake_workflow
    )

    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        json={"company_name": "Microsoft"},
    )

    assert response.status_code == 502
