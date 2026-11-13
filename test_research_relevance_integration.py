
import asyncio

import workflow


def make_source(title, url, content):
    return {
        "title": title,
        "url": url,
        "content": content,
        "published_date": "2026-10-15",
    }


def test_independent_business_article_without_domain():
    article = make_source(
        "Linear releases new issue tracking features",
        "https://tech.example.com/linear-update",
        (
            "Linear, the issue tracking software platform "
            "for development teams, announced new "
            "project management features."
        ),
    )

    result = workflow.classify_company_evidence(
        article,
        company_name="Linear",
        official_domain="linear.app",
    )

    # An independent article can be relevant
    # without explicitly printing linear.app.
    assert result == "relevant"


def test_research_counts_only_relevant_sources(
    monkeypatch,
):
    monkeypatch.setenv("MOCK_TAVILY", "false")

    calls = []

    first_results = [
        make_source(
            "Linear introduces software development features",
            "https://linear.app/changelog/mobile-diffs",
            "Linear released new features for software teams.",
        ),
        make_source(
            "Nanobiotechnology market forecast",
            "https://example.com/nanotechnology",
            "Analysis of the nanobiotechnology industry.",
        ),
        make_source(
            "Apple TSMC partnership",
            "https://example.com/semiconductors",
            "Apple and TSMC announced chip manufacturing plans.",
        ),
        make_source(
            "Linear announces funding",
            "https://example.com/uncertain",
            "Linear announced funding.",
        ),
    ]

    second_results = [
        make_source(
            "Linear issue tracking software update",
            "https://example.com/linear-release",
            (
                "Linear (linear.app), the issue tracking "
                "software platform, released new capabilities."
            ),
        ),
        make_source(
            "Linear launches new developer tools",
            "https://linear.app/changelog/developer-tools",
            "Linear introduced new tools for software developers.",
        ),
    ]

    class FakeSearch:
        def __init__(self, name, results):
            self.name = name
            self.results = results

        async def ainvoke(self, arguments):
            calls.append((self.name, arguments))
            return {"results": self.results}

    monkeypatch.setattr(
        workflow,
        "search_tool",
        FakeSearch("initial", first_results),
    )

    monkeypatch.setattr(
        workflow,
        "expanded_search_tool",
        FakeSearch("expanded", second_results),
    )

    initial = {
        "company_name": "Linear",
        "official_domain": "linear.app",
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
        workflow.research_node(initial)
    )

    # Four returned articles, only one relevant.
    assert len(first["sources"]) == 1
    assert first["source_stats"]["returned"] == 4
    assert first["source_stats"]["accepted"] == 1
    assert first["source_stats"]["discarded"] == 3

    state_after_first = {
        **initial,
        **first,
    }

    assert workflow.check_research(
        state_after_first
    ) == "retry"

    second = asyncio.run(
        workflow.research_node(state_after_first)
    )

    # The existing fallback retrieves two more.
    assert len(second["sources"]) == 3

    urls = {
        source["url"]
        for source in second["sources"]
    }

    assert "https://example.com/nanotechnology" not in urls
    assert "https://example.com/semiconductors" not in urls
    assert "https://example.com/uncertain" not in urls

    assert workflow.check_research({
        **state_after_first,
        **second,
    }) == "continue"

    # No additional search beyond existing fallback.
    assert len(calls) == 2
    assert calls[0][0] == "initial"
    assert calls[1][0] == "expanded"

    # Company identity must influence retrieval.
    assert "linear.app" in calls[0][1]["query"]

    # Existing date policy remains active.
    assert "start_date" in calls[0][1]
    assert "start_date" in calls[1][1]
