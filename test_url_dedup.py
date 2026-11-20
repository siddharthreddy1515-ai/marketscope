
import asyncio
import json

import workflow


def test_tracking_parameters_are_removed():
    url_a = (
        "https://news.com/article?id=123"
        "&utm_source=google"
    )

    url_b = (
        "https://news.com/article?utm_medium=email"
        "&id=123"
    )

    url_c = "https://news.com/article?id=456"

    # Tracking parameters should not affect identity.
    assert (
        workflow.canonical_source_url(url_a)
        == workflow.canonical_source_url(url_b)
    )

    # Meaningful parameters must remain significant.
    assert (
        workflow.canonical_source_url(url_a)
        != workflow.canonical_source_url(url_c)
    )


def test_research_removes_duplicate_sources(
    monkeypatch, tmp_path
):
    payload = {
        "results": [
            {
                "title": "Article A",
                "url": (
                    "https://news.com/article?id=123"
                    "&utm_source=google"
                ),
                "content": "Acme announced a product.",
            },
            {
                "title": "Duplicate Article A",
                "url": (
                    "https://news.com/article"
                    "?utm_medium=email&id=123"
                ),
                "content": "Duplicate coverage of Acme.",
            },
            {
                "title": "Article B",
                "url": "https://news.com/article?id=456",
                "content": "Acme opened a new office.",
            },
        ]
    }

    fixture = tmp_path / "duplicates.json"
    fixture.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

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

    assert len(result["sources"]) == 2

    assert result["source_stats"] == {
        "returned": 3,
        "accepted": 2,
        "discarded": 1,
    }

    # Original URLs must remain unchanged.
    assert (
        result["sources"][0]["url"]
        == payload["results"][0]["url"]
    )

    # Duplicate content must not reach the LLM.
    assert (
        "Duplicate coverage"
        not in result["search_results"]
    )

    assert any(
        "duplicate" in warning.lower()
        for warning in result["research_warnings"]
    )
