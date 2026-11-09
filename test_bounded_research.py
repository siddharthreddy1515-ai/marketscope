
import asyncio
from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler

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
                content="- Mocked strategic insight."
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
