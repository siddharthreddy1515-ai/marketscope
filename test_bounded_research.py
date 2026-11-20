
import asyncio
from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler
import pytest

import workflow


def source(name, url):
    return {
        "title": name,
        "url": url,
        "content": f"Acme announced {name}.",
    }


def setup_fake_services(monkeypatch, responses):
    calls = []
    response_iterator = iter(responses)

    class FakeSearch:
        def __init__(self, limit):
            self.limit = limit

        async def ainvoke(self, payload):
            calls.append(self.limit)

            try:
                result = next(response_iterator)
            except StopIteration:
                raise AssertionError(
                    "Unexpected additional search request"
                )

            if isinstance(result, Exception):
                raise result

            return result

    class FakeLLM:
        async def ainvoke(self, *args, **kwargs):
            return SimpleNamespace(
                content="- Acme announced a development. [S1]"
            )

    monkeypatch.setenv("MOCK_TAVILY", "false")
    monkeypatch.setattr(
        workflow, "search_tool", FakeSearch(5)
    )
    monkeypatch.setattr(
        workflow,
        "expanded_search_tool",
        FakeSearch(10),
        raising=False,
    )
    monkeypatch.setattr(
        workflow, "llm_analysis", FakeLLM()
    )
    monkeypatch.setattr(
        workflow, "langfuse_handler", BaseCallbackHandler()
    )

    return calls


def test_stop_after_three_unique_sources(monkeypatch):
    responses = [
        {
            "results": [
                source("A", "https://example.com/a"),
                source("B", "https://example.com/b"),
                source("C", "https://example.com/c"),
            ]
        }
    ]

    calls = setup_fake_services(monkeypatch, responses)

    result = asyncio.run(workflow.run_workflow("Acme"))

    assert result["status"] == "completed"
    assert len(result["sources"]) == 3
    assert calls == [5]


def test_enrich_until_three_unique_sources(monkeypatch):
    responses = [
        {
            "results": [
                source("A", "https://example.com/a"),
                source("B", "https://example.com/b"),
                {"url": "", "content": "Invalid source"},
            ]
        },
        {
            "results": [
                source(
                    "A duplicate",
                    "https://example.com/a?utm_source=email",
                ),
                source("C", "https://example.com/c"),
            ]
        },
    ]

    calls = setup_fake_services(monkeypatch, responses)

    result = asyncio.run(workflow.run_workflow("Acme"))

    assert result["status"] == "completed"
    assert len(result["sources"]) == 3
    assert calls == [5, 10]


def test_preserve_sources_if_enrichment_fails(monkeypatch):
    responses = [
        {
            "results": [
                source("A", "https://example.com/a"),
                source("B", "https://example.com/b"),
            ]
        },
        RuntimeError("Tavily temporarily unavailable"),
    ]

    calls = setup_fake_services(monkeypatch, responses)

    result = asyncio.run(workflow.run_workflow("Acme"))

    assert result["status"] == "completed"
    assert len(result["sources"]) == 2
    assert calls == [5, 10]


def test_background_pages_do_not_stop_search_early(monkeypatch):
    first = [
        {"title": "Terms of Service | Acme", "url": "https://example.com/terms",
         "content": "Acme offers restaurant delivery services."},
        {"title": "Acme company profile", "url": "https://directory.example/acme",
         "content": "Acme provides delivery services."},
        {"title": "Acme Products", "url": "https://example.com/products",
         "content": "Acme customers can order meals from local restaurants."},
    ]
    change = source("a new delivery service", "https://example.com/announcements/new-service")
    calls = setup_fake_services(monkeypatch, [{"results": first}, {"results": [change]}])

    result = asyncio.run(workflow.run_workflow("Acme"))

    assert calls == [5, 10]
    assert [item["url"] for item in result["sources"]] == [item["url"] for item in [*first, change]]
    assert result["research_coverage"]["development_sources"] == 1


@pytest.mark.parametrize("follow_up_succeeds", [True, False])
def test_preview_dominated_news_uses_one_focused_search(monkeypatch, follow_up_succeeds):
    profile = source("ordering services", "https://acme.example/about")
    full = {"title": "Acme and Green Grocer launch Dallas drone deliveries",
            "url": "https://news.example/acme-grocery",
            "content": "Acme and Green Grocer launched drone deliveries in Dallas for grocery orders."}
    repeat = {**full, "url": "https://www.google.com/url?q=duplicate-preview"}
    lead = {"title": "Acme tests return delivery with Kite Pharmacy",
            "url": "https://www.google.com/url?q=pharmacy-preview",
            "content": "Kite Pharmacy is testing a return delivery service through Acme"}
    article = {**lead, "url": "https://retail.example/pharmacy",
               "content": "Kite Pharmacy is testing a return delivery service through Acme in Denver."}
    last = {"results": [article]} if follow_up_succeeds else RuntimeError("Search failed")
    calls = setup_fake_services(monkeypatch, [{"results": [profile]}, {"results": [full, repeat, lead]}, last])
    requests = []
    expanded = workflow.expanded_search_tool.ainvoke

    async def capture(request):
        requests.append(request)
        return await expanded(request)

    monkeypatch.setattr(workflow.expanded_search_tool, "ainvoke", capture)
    result = asyncio.run(workflow.run_workflow("Acme"))
    assert calls == [5, 10, 10]
    assert "Kite Pharmacy" in requests[-1]["query"]
    assert "Green Grocer" not in requests[-1]["query"]
    assert requests[-1]["start_date"] == requests[0]["start_date"]
    assert requests[-1]["exclude_domains"] == ["google.com", "tavily.com"]
    expected = [profile, full, repeat, lead, *([article] if follow_up_succeeds else [])]
    assert [s["url"] for s in result["sources"]] == [s["url"] for s in expected]
