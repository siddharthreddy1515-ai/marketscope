
import asyncio
from datetime import date, timedelta

import workflow


def test_company_research_date_policy():
    today = date(2026, 11, 12)

    assert workflow.company_news_start_date(
        1, today=today
    ) == "2026-08-14"

    assert workflow.company_news_start_date(
        2, today=today
    ) == "2025-11-12"

    assert workflow.company_news_start_date(
        3, today=today
    ) == "2025-11-12"


def test_incomplete_research_broadens_date_window(
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

    today = date.today()

    assert calls[0][1]["start_date"] == (
        today - timedelta(days=90)
    ).isoformat()

    assert calls[1][1]["start_date"] == (
        today - timedelta(days=365)
    ).isoformat()

    # Earlier evidence must be preserved.
    assert len(second["sources"]) == 2

    # One enrichment attempt is sufficient.
    assert workflow.check_research({
        **initial_state,
        **first,
        **second,
    }) == "continue"
