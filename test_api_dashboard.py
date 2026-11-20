
import pytest
from fastapi.testclient import TestClient
import main


def test_analyze_returns_research_data(monkeypatch):

    async def fake_workflow(company_name):
        return {
            "status": "completed",
            "analysis_status": "complete",
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


@pytest.mark.parametrize("analysis_status", ["complete", "partial", "insufficient_evidence"])
def test_analysis_status_and_warnings_reach_api(monkeypatch, analysis_status):
    warning = "Some generated findings were excluded because they could not be validated."
    warnings = [warning] if analysis_status == "partial" else []

    async def fake_workflow(company_name):
        return {
            "status": "completed", "analysis_status": analysis_status,
            "analysis_warnings": warnings, "report": "# Acme Report",
            "research_warnings": ["Research coverage was limited."],
        }

    monkeypatch.setattr(main, "run_workflow", fake_workflow)
    response = TestClient(main.app).post("/api/v1/analyze", json={"company_name": "Acme"})
    assert response.status_code == 200
    assert response.json()["analysis_status"] == analysis_status
    assert response.json()["analysis_warnings"] == warnings
    assert response.json()["research_warnings"] == ["Research coverage was limited."]


def test_malformed_analysis_returns_safe_specific_error(monkeypatch):
    async def fake_workflow(company_name):
        return {"status": "failed", "error_code": "ANALYSIS_MALFORMED",
                "report": None, "insights": "Raw model output"}

    monkeypatch.setattr(main, "run_workflow", fake_workflow)
    response = TestClient(main.app).post("/api/v1/analyze", json={"company_name": "Acme"})
    assert response.status_code == 502
    assert response.json() == {"detail": "The analysis response could not be read. Please try again."}


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


def test_executive_summary_reaches_api(monkeypatch):

    async def fake_workflow(company_name):
        return {
            "status": "completed",
            "analysis_status": "complete",
            "report": "# Acme Report",
            "research_attempts": 1,
            "sources": [
                {
                    "title": "Acme Announcement",
                    "url": "https://example.com/acme",
                    "content": "Acme launched a product.",
                    "published_at": None,
                }
            ],
            "source_stats": {
                "returned": 1,
                "accepted": 1,
                "discarded": 0,
            },
            "research_warnings": [],
            "insight_items": [],
            "executive_summary": {
                "text": "Acme announced a product launch.",
                "sources": [
                    {
                        "title": "Acme Announcement",
                        "url": "https://example.com/acme",
                    }
                ],
            },
        }

    monkeypatch.setattr(
        main,
        "run_workflow",
        fake_workflow,
    )

    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        json={"company_name": "Acme"},
    )

    assert response.status_code == 200

    summary = response.json()["executive_summary"]

    assert summary["text"] == (
        "Acme announced a product launch."
    )

    assert summary["sources"][0]["url"] == (
        "https://example.com/acme"
    )


def test_source_sentiment_reaches_api(monkeypatch):

    async def fake_workflow(company_name):
        return {
            "status": "completed",
            "analysis_status": "complete",
            "report": "# Acme Report",
            "research_attempts": 1,
            "sources": [
                {
                    "title": "Acme announcement",
                    "url": "https://example.com/acme",
                    "content": "Acme launched a product.",
                    "published_at": None,
                }
            ],
            "source_stats": {
                "returned": 1,
                "accepted": 1,
                "discarded": 0,
            },
            "research_warnings": [],
            "insight_items": [],
            "executive_summary": None,
            "source_sentiments": [
                {
                    "source_id": "S1",
                    "label": "Positive",
                    "title": "Acme announcement",
                    "url": "https://example.com/acme",
                }
            ],
            "sentiment_distribution": {
                "positive": 1,
                "neutral": 0,
                "negative": 0,
                "assessed": 1,
                "unassessed": 0,
                "total_sources": 1,
            },
        }

    monkeypatch.setattr(
        main,
        "run_workflow",
        fake_workflow,
    )

    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        json={"company_name": "Acme"},
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["source_sentiments"][0]["label"]
        == "Positive"
    )

    distribution = data["sentiment_distribution"]

    assert distribution["positive"] == 1
    assert distribution["assessed"] == 1
    assert distribution["total_sources"] == 1


def test_competitors_reach_api(monkeypatch):

    async def fake_workflow(company_name):
        return {
            "status": "completed",
            "analysis_status": "complete",
            "report": "# Acme Report",
            "research_attempts": 1,
            "sources": [
                {
                    "title": "Acme product announcement",
                    "url": "https://example.com/product",
                    "content": "Acme launched a product.",
                    "published_at": None,
                }
            ],
            "source_stats": {
                "returned": 1,
                "accepted": 1,
                "discarded": 0,
            },
            "research_warnings": [],
            "insight_items": [],
            "executive_summary": None,
            "source_sentiments": [],
            "sentiment_distribution": None,
            "competitors": [
                {
                    "name": "Northstar Analytics",
                    "overlap": (
                        "Both companies sell analytics "
                        "dashboards to small businesses."
                    ),
                    "sources": [
                        {
                            "title": "Acme competitor comparison",
                            "url": "https://example.com/comparison",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(
        main,
        "run_workflow",
        fake_workflow,
    )

    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        json={"company_name": "Acme"},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["competitors"]) == 1

    competitor = data["competitors"][0]

    assert competitor["name"] == "Northstar Analytics"

    assert "analytics dashboards" in competitor["overlap"]

    assert competitor["sources"][0]["url"] == (
        "https://example.com/comparison"
    )
