
import asyncio
import json

import workflow


def run_node(state):
    return asyncio.run(
        workflow.competitor_research_node(state)
    )


def initial_state():
    return {
        "company_name": "Acme",
        "competitor_sources": [],
        "competitor_research_attempts": 0,
        "competitor_research_warnings": [],
    }


def test_competitor_query_targets_market_competition():
    query = workflow.build_competitor_query("Acme")

    assert "Acme" in query
    assert (
        "competitor" in query.lower()
        or "alternative" in query.lower()
    )


def test_competitor_research_uses_one_search_and_filters_sources(
    monkeypatch,
):
    monkeypatch.setenv("MOCK_TAVILY", "false")

    class FakeSearch:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, payload):
            self.calls.append(payload)

            return {
                "results": [
                    {
                        "title": "Acme competing products",
                        "url": "https://example.com/rivals",
                        "content": (
                            "An article comparing Acme "
                            "with rival products."
                        ),
                        "published_date": "2026-10-01",
                    },
                    {
                        "title": "Duplicate tracking URL",
                        "url": (
                            "https://example.com/rivals"
                            "?utm_source=newsletter"
                        ),
                        "content": "Duplicate article.",
                    },
                    {
                        "title": "Empty article",
                        "url": "https://example.com/empty",
                        "content": "",
                    },
                    {
                        "title": "Invalid URL",
                        "url": "javascript:alert(1)",
                        "content": "Invalid source.",
                    },
                ]
            }

    fake = FakeSearch()

    monkeypatch.setattr(
        workflow,
        "competitor_search_tool",
        fake,
    )

    result = run_node(initial_state())

    # Exactly one search call.
    assert len(fake.calls) == 1

    competitor_sources = result["competitor_sources"]

    # Only one structurally valid unique source.
    assert len(competitor_sources) == 1

    assert competitor_sources[0]["url"] == (
        "https://example.com/rivals"
    )

    assert competitor_sources[0]["published_at"] == (
        "2026-10-01"
    )

    # Competitor research must not overwrite
    # the existing ordinary news sources.
    assert "sources" not in result

    assert result["competitor_research_attempts"] == 1


def test_competitor_search_failure_is_non_blocking(
    monkeypatch,
):
    monkeypatch.setenv("MOCK_TAVILY", "false")

    class FailingSearch:
        async def ainvoke(self, payload):
            raise RuntimeError("Simulated Tavily failure")

    monkeypatch.setattr(
        workflow,
        "competitor_search_tool",
        FailingSearch(),
    )

    result = run_node(initial_state())

    assert result["competitor_sources"] == []

    assert result["competitor_research_attempts"] == 1

    assert result["competitor_research_warnings"]

    # Failure should not replace existing
    # company research state.
    assert "sources" not in result


def test_mock_competitor_research_never_calls_tavily(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("MOCK_TAVILY", "true")
    monkeypatch.chdir(tmp_path)

    mock_dir = tmp_path / "mocks"
    mock_dir.mkdir()

    fixture = mock_dir / "Acme_competitors.json"

    fixture.write_text(
        json.dumps({
            "results": [
                {
                    "title": "Mock market comparison",
                    "url": "https://example.com/comparison",
                    "content": (
                        "Mock comparison of competing "
                        "analytics products."
                    ),
                }
            ]
        }),
        encoding="utf-8",
    )

    class NoExternalSearch:
        async def ainvoke(self, payload):
            raise AssertionError(
                "Tavily must not run in mock mode"
            )

    monkeypatch.setattr(
        workflow,
        "competitor_search_tool",
        NoExternalSearch(),
    )

    result = run_node(initial_state())

    assert len(result["competitor_sources"]) == 1

    assert result["competitor_research_attempts"] == 1


def test_competitor_research_cannot_repeat_a_completed_search(
    monkeypatch,
):
    monkeypatch.setenv("MOCK_TAVILY", "false")

    class NoSecondSearch:
        async def ainvoke(self, payload):
            raise AssertionError(
                "Competitor research called twice"
            )

    monkeypatch.setattr(
        workflow,
        "competitor_search_tool",
        NoSecondSearch(),
    )

    state = initial_state()

    state["competitor_research_attempts"] = 1

    state["competitor_sources"] = [
        {
            "title": "Earlier competitor evidence",
            "url": "https://example.com/existing",
            "content": "Previously retrieved evidence.",
            "published_at": None,
        }
    ]

    result = run_node(state)

    assert result["competitor_research_attempts"] == 1
    assert len(result["competitor_sources"]) == 1

