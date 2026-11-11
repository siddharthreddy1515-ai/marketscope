"""Tests for source-to-insight mapping, without live API calls."""
import asyncio
from types import SimpleNamespace

import workflow


def fixture_sources():
    return [
        {
            "title": "Company A partnership",
            "url": "https://example.com/partnership",
            "content": "Acme announced a partnership.",
            "published_at": "2026-11-01",
        },
        {
            "title": "Company A product",
            "url": "https://example.com/product",
            "content": "Acme released a product.",
            "published_at": "2026-11-02",
        },
    ]


def test_mapping_uses_only_retrieved_urls():
    raw = (
        "- Acme announced a partnership. [S1]\n"
        "- Acme released a product. [S2] [S1] [S2]"
    )
    items = workflow.parse_cited_insights(raw, fixture_sources())
    assert len(items) == 2
    assert items[0]["text"] == "Acme announced a partnership."
    assert items[0]["sources"] == [
        {"title": "Company A partnership", "url": "https://example.com/partnership"}
    ]
    assert [link["url"] for link in items[1]["sources"]] == [
        "https://example.com/product", "https://example.com/partnership"
    ]


def test_mapping_does_not_display_unknown_or_uncited_insights():
    raw = (
        "- Unsupported claim without citations.\n"
        "- Claim using a fictional source. [S99]\n"
        "- Claim mixing a true and fictional source. [S1] [S99]"
    )
    assert workflow.parse_cited_insights(raw, fixture_sources()) == []


def test_analysis_node_uses_one_llm_call(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            assert "[S1]" in messages[0].content
            return SimpleNamespace(content="- Acme announced a partnership. [S1]")

    fake = FakeLLM()
    monkeypatch.setattr(workflow, "llm_analysis", fake)
    state = {"company_name": "Acme", "sources": fixture_sources()}
    result = asyncio.run(workflow.analysis_node(state))

    assert fake.calls == 1
    assert len(result["insight_items"]) == 1
    assert result["insight_items"][0]["sources"][0]["url"] == "https://example.com/partnership"
