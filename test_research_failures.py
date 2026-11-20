
import asyncio
import json

import pytest
import workflow


@pytest.mark.parametrize(
    "payload",
    [
        {"results": []},
        {
            "results": [
                {
                    "title": "Empty article",
                    "url": "https://example.com/article",
                    "content": "",
                }
            ]
        },
    ],
)
def test_research_retries_without_usable_evidence(
    monkeypatch, tmp_path, payload
):
    # Create isolated fixture data for this test.
    fixture = tmp_path / "search.json"
    fixture.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    # Never call the real Tavily API.
    monkeypatch.setenv("MOCK_TAVILY", "true")
    monkeypatch.setattr(
        workflow,
        "mock_path",
        lambda company: str(fixture),
    )

    result = asyncio.run(
        workflow.research_node({
            "company_name": "Acme",
            "retry_count": 0,
        })
    )

    # No evidence means research must not succeed.
    assert result["retry_count"] == 1
    assert not result.get("search_results")
    assert not result.get("sources")


def test_research_accepts_partial_results(monkeypatch, tmp_path):
    payload = {
        "results": [
            {
                "title": "Launch",
                "url": "https://example.com/a",
                "content": "Acme launched a product.",
            },
            {
                "title": "Expansion",
                "url": "https://example.com/b",
                "content": "Acme expanded internationally.",
            },
            {
                "url": "https://example.com/c",
                "content": "",
            },
            {
                "url": "",
                "content": "Missing source URL.",
            },
            {
                "url": "invalid-url",
                "content": "Invalid source URL.",
            },
        ]
    }

    fixture = tmp_path / "partial.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv("MOCK_TAVILY", "true")
    monkeypatch.setattr(
        workflow, "mock_path", lambda company: str(fixture)
    )

    result = asyncio.run(
        workflow.research_node({
            "company_name": "Acme",
            "retry_count": 0,
        })
    )

    assert result["retry_count"] == 0
    assert len(result["sources"]) == 2
    assert result["source_stats"] == {
        "returned": 5,
        "accepted": 2,
        "discarded": 3,
    }
    assert result["research_warnings"]
    assert "Acme launched" in result["search_results"]
    assert "Acme expanded" in result["search_results"]
    assert "Missing source URL" not in result["search_results"]
    assert "Invalid source URL" not in result["search_results"]
