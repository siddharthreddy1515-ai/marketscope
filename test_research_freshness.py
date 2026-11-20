
import asyncio
from datetime import date

import workflow


def test_company_research_date_policy():
    assert workflow.default_analysis_window(date(2026, 11, 12)) == {
        "start": "2026-08-14", "end": "2026-11-12",
    }


def test_first_news_search_uses_recent_window_after_profile_search(
    monkeypatch,
):
    monkeypatch.setenv("MOCK_TAVILY", "false")

    calls = []

    class FakeSearch:
        def __init__(self, label):
            self.label = label

        async def ainvoke(self, arguments):
            calls.append((self.label, arguments))

            number = len(calls)

            return {
                "results": [
                    {
                        "title": f"Acme development {number}",
                        "url": (
                            f"https://example.com/news/{number}"
                        ),
                        "content": (
                            f"Acme announced development {number}."
                        ),
                        "published_date": "2026-09-01",
                    }
                ]
            }

    monkeypatch.setattr(
        workflow,
        "search_tool",
        FakeSearch("initial"),
    )

    monkeypatch.setattr(
        workflow,
        "expanded_search_tool",
        FakeSearch("expanded"),
    )

    initial_state = {
        "company_name": "Acme",
        "analysis_window": {"start": "2026-08-14", "end": "2026-11-12"},
        "sources": [],
        "research_attempts": 0,
        "retry_count": 0,
        "enrichment_attempted": False,
        "research_warnings": [],
        "source_stats": {
            "returned": 0,
            "accepted": 0,
            "discarded": 0,
        },
    }

    first = asyncio.run(
        workflow.research_node(initial_state)
    )

    # Only one source was found, below SOURCE_TARGET.
    assert len(first["sources"]) == 1

    assert workflow.check_research({
        **initial_state,
        **first,
    }) == "retry"

    second = asyncio.run(
        workflow.research_node({
            **initial_state,
            **first,
        })
    )

    assert len(calls) == 2

    assert calls[0][0] == "initial"
    assert calls[1][0] == "expanded"

    assert calls[0][1] == {
        "query": "Acme company products services",
        "topic": "general",
    }

    assert calls[1][1]["topic"] == "news"
    assert calls[1][1]["start_date"] == initial_state["analysis_window"]["start"]

    # Earlier evidence must be preserved.
    assert len(second["sources"]) == 2

    # One enrichment attempt is sufficient.
    assert workflow.check_research({
        **initial_state,
        **first,
        **second,
    }) == "continue"
