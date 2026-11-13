
import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
import workflow


def fake_result():
    return {
        "status": "completed",
        "report": "# Linear Report",
        "research_attempts": 1,
        "sources": [],
        "source_stats": {
            "returned": 0,
            "accepted": 0,
            "discarded": 0,
        },
        "research_warnings": [],
        "insight_items": [],
        "executive_summary": None,
        "source_sentiments": [],
        "sentiment_distribution": {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "assessed": 0,
            "unassessed": 0,
            "total_sources": 0,
        },
        "competitors": [],
    }


def test_request_normalizes_company_website():
    request = main.AnalyzeRequest(
        company_name="Linear",
        official_domain=(
            "https://www.linear.app/changelog"
        ),
    )

    assert request.official_domain == "linear.app"


def test_company_website_is_optional():
    request = main.AnalyzeRequest(
        company_name="Linear"
    )

    assert request.official_domain is None


def test_invalid_website_is_rejected():
    with pytest.raises(ValidationError):
        main.AnalyzeRequest(
            company_name="Linear",
            official_domain="ftp://linear.app",
        )


def test_api_forwards_official_domain(monkeypatch):
    calls = []

    async def fake_workflow(
        company_name,
        official_domain=None,
    ):
        calls.append(
            (company_name, official_domain)
        )
        return fake_result()

    monkeypatch.setattr(
        main,
        "run_workflow",
        fake_workflow,
    )

    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        json={
            "company_name": "Linear",
            "official_domain": (
                "https://www.linear.app/changelog"
            ),
        },
    )

    assert response.status_code == 200, response.text

    assert calls == [
        ("Linear", "linear.app")
    ]


def test_name_only_api_remains_compatible(
    monkeypatch,
):
    calls = []

    # Deliberately accepts only one argument.
    # Existing callers should keep working.
    async def fake_workflow(company_name):
        calls.append(company_name)
        return fake_result()

    monkeypatch.setattr(
        main,
        "run_workflow",
        fake_workflow,
    )

    client = TestClient(main.app)

    response = client.post(
        "/api/v1/analyze",
        json={"company_name": "Linear"},
    )

    assert response.status_code == 200, response.text
    assert calls == ["Linear"]


def test_workflow_receives_normalized_domain(
    monkeypatch,
):
    received = {}

    class FakeGraph:
        async def ainvoke(
            self,
            state,
            config=None,
        ):
            received.update(state)

            return {
                "sources": [
                    {
                        "title": "Linear product update",
                        "url": "https://linear.app/changelog",
                        "content": "Linear launched features.",
                        "published_at": None,
                    }
                ],
                "report": "# Linear Report",
                "research_attempts": 1,
            }

    monkeypatch.setattr(
        workflow,
        "app_graph",
        FakeGraph(),
    )

    result = asyncio.run(
        workflow.run_workflow(
            "Linear",
            official_domain=(
                "https://www.linear.app/changelog"
            ),
        )
    )

    assert result["status"] == "completed"

    assert received["official_domain"] == (
        "linear.app"
    )
