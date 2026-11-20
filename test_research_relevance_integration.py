
import asyncio

import pytest

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
            "Linear issue tracking features",
            "https://linear.app/product",
            "Linear combines issue tracking and project management in its software platform.",
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

    # The first search can find evergreen facts. Dated news is still available
    # through the existing bounded enrichment path.
    assert calls[0][1]["topic"] == "general"
    assert "start_date" not in calls[0][1]
    assert calls[1][1]["topic"] == "news"
    assert "start_date" in calls[1][1]


def test_general_search_retains_official_company_basics(monkeypatch):
    monkeypatch.setenv("MOCK_TAVILY", "false")

    class FakeSearch:
        async def ainvoke(self, request):
            assert request == {
                "query": "Acme acme.example company products services",
                "topic": "general",
            }
            return {"results": [
                {"title": "Acme", "url": "https://acme.example/about",
                 "content": "Acme makes electric motorcycles for city riders."},
                {"title": "Other Acme", "url": "https://example.com/other",
                 "content": "Acme makes industrial washing machines."},
            ]}

    monkeypatch.setattr(workflow, "search_tool", FakeSearch())
    result = asyncio.run(workflow.research_node({
        "company_name": "Acme",
        "official_domain": "acme.example",
    }))
    assert [source["url"] for source in result["sources"]] == ["https://acme.example/about"]


@pytest.mark.parametrize("with_news", [False, True])
def test_last_search_targets_official_pages_when_product_detail_is_missing(monkeypatch, with_news):
    monkeypatch.setenv("MOCK_TAVILY", "false")
    calls = []
    profile = {"title": "Acme company profile", "url": "https://directory.example/acme",
               "content": "Acme (acme.example) provides payroll software services."}
    product = {"title": "Acme Payroll", "url": "https://acme.example/products/payroll",
               "content": "Acme handles wages and tax filings."}
    announcement = {"title": "Acme announces an expansion", "url": "https://publisher.example/acme-expansion",
                    "content": "Acme announces new payroll services for employers."}

    class FakeSearch:
        async def ainvoke(self, request):
            calls.append(request)
            if request["topic"] == "news" and with_news:
                return {"results": [announcement]}
            return {"results": [product if "include_domains" in request else profile]}

    monkeypatch.setattr(workflow, "search_tool", FakeSearch())
    monkeypatch.setattr(workflow, "expanded_search_tool", FakeSearch())
    state = {"company_name": "Acme", "official_domain": "acme.example"}
    for _ in range(3):
        state.update(asyncio.run(workflow.research_node(state)))
        if workflow.check_research(state) == "continue":
            break

    assert [request["topic"] for request in calls] == ["general", "news", "general"]
    assert calls[-1]["include_domains"] == ["acme.example"]
    expected_sources = [profile, announcement, product] if with_news else [profile, product]
    assert [source["url"] for source in state["sources"]] == [source["url"] for source in expected_sources]


def test_official_news_alone_does_not_satisfy_product_detail():
    state = {
        "company_name": "Acme", "official_domain": "acme.example", "research_attempts": 2,
        "enrichment_attempted": True,
        "sources": [make_source("Acme launches a new product", "https://acme.example/news/launch",
                                "Acme launches a new product.")],
    }
    assert workflow.check_research(state) == "retry"
    state["official_detail_attempted"] = True
    assert workflow.check_research(state) == "continue"


def test_news_request_uses_matched_profile_and_preserves_sparse_sources(monkeypatch):
    monkeypatch.setenv("MOCK_TAVILY", "false")
    profile = make_source("Aster", "https://tavily.com/knowledge/company/123",
                          "Website: www.aster.example\nBusiness areas: Cargo Bicycles, Parts")
    calls = []

    class EmptyNews:
        async def ainvoke(self, arguments):
            calls.append(arguments)
            return {"results": []}

    monkeypatch.setattr(workflow, "expanded_search_tool", EmptyNews())
    state = {"company_name": "Aster", "official_domain": "aster.example", "sources": [profile],
             "research_attempts": 1, "analysis_window": {"start": "2026-08-22", "end": "2026-11-20"}}
    result = asyncio.run(workflow.research_node(state))
    assert calls == [{"query": '"Aster" aster.example Cargo Bicycles recent business news',
                      "topic": "news", "start_date": "2026-08-22", "exclude_domains": ["tavily.com"]}]
    assert result["sources"] == [profile]
    assert result["research_attempts"] == 2
    assert workflow.check_research({**state, **result}) == "retry"


def test_news_request_does_not_borrow_a_mismatched_profiles_industry():
    profile = make_source("Aster", "https://tavily.com/knowledge/company/123",
                          "Website: another.example\nBusiness areas: Freight")
    request = workflow.news_search_request("Aster", "aster.example", [profile], "2026-08-22", offsite=True)
    assert request["query"] == workflow.build_query("Aster")
    assert request["exclude_domains"] == ["aster.example", "tavily.com"]
    own_request = workflow.news_search_request("Tavily", "tavily.com", [profile], "2026-08-22")
    assert "exclude_domains" not in own_request


def test_follow_up_looks_beyond_the_supplied_site_without_losing_basics(monkeypatch):
    monkeypatch.setenv("MOCK_TAVILY", "false")
    company_pages = [
        make_source("Acme Products", "https://acme.example/products", "Acme provides payroll services."),
        make_source("Acme Support", "https://acme.example/support", "Acme helps current customers."),
        make_source("Acme Company Overview", "https://directory.example/acme",
                    "Acme (acme.example) runs payroll software for employers."),
    ]
    reporting = make_source(
        "Acme launches a new payroll service", "https://trade-publication.example/acme-launch",
        "Acme launched a new payroll service for customers.",
    )

    class FakeSearch:
        async def ainvoke(self, request):
            if request["topic"] == "general":
                return {"results": company_pages}
            assert request["exclude_domains"] == ["acme.example"]
            assert "acme.example" not in request["query"]
            return {"results": [reporting]}

    monkeypatch.setattr(workflow, "search_tool", FakeSearch())
    monkeypatch.setattr(workflow, "expanded_search_tool", FakeSearch())
    state = {"company_name": "Acme", "official_domain": "acme.example"}
    state.update(asyncio.run(workflow.research_node(state)))
    assert len(state["sources"]) == 3
    assert workflow.check_research(state) == "retry"
    state.update(asyncio.run(workflow.research_node(state)))
    assert [source["url"] for source in state["sources"]] == [page["url"] for page in [*company_pages, reporting]]
    assert workflow.check_research(state) == "continue"


@pytest.mark.parametrize("website", [None, "acme.example"])
def test_last_empty_search_recovers_company_basics(monkeypatch, website):
    monkeypatch.setenv("MOCK_TAVILY", "false")
    calls = []
    product = make_source("Acme Payroll", "https://acme.example/products/payroll", "Acme handles wages and tax filings.")

    class FakeSearch:
        async def ainvoke(self, request):
            calls.append(request)
            return {"results": [product] if len(calls) == 3 else []}

    monkeypatch.setattr(workflow, "search_tool", FakeSearch())
    state = {"company_name": "Acme", "official_domain": website}
    for _ in range(workflow.MAX_RESEARCH_CALLS):
        assert workflow.check_research(state) == "retry"
        state.update(asyncio.run(workflow.research_node(state)))

    assert [call["topic"] for call in calls] == ["general", "news", "general"]
    assert calls[-1]["query"] != calls[0]["query"]
    assert calls[-1].get("include_domains") == ([website] if website else None)
    assert [source["url"] for source in state["sources"]] == [product["url"]]
    assert workflow.check_research(state) == "continue"
