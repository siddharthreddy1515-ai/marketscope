
"""
Signal Breakdown tests.

These tests make no real Tavily,
OpenAI, or DeepEval requests.
"""

import asyncio
from types import SimpleNamespace

from main import InsightItemResponse
import workflow


def sources():
    return [
        {
            "title": "Acme partners",
            "url": "https://example.com/partner",
            "content": "Acme signed a partnership.",
            "published_at": None,
        },
        {
            "title": "Acme launches",
            "url": "https://example.com/launch",
            "content": "Acme launched a product.",
            "published_at": None,
        },
    ]


def test_categories_and_citations_are_mapped():
    raw = (
        "- [M&A / Partnerships] "
        "Acme signed a partnership. [S1]\n"
        "- [Product & Innovation] "
        "Acme launched a product. [S2]"
    )

    items = workflow.parse_cited_insights(
        raw,
        sources(),
    )

    assert [
        item["category"]
        for item in items
    ] == [
        "M&A / Partnerships",
        "Product & Innovation",
    ]

    assert [
        item["text"]
        for item in items
    ] == [
        "Acme signed a partnership.",
        "Acme launched a product.",
    ]

    assert (
        items[0]["sources"][0]["url"]
        == sources()[0]["url"]
    )


def test_unknown_or_missing_category_falls_back_to_other():
    raw = (
        "- [Mysterious Category] "
        "Acme signed a partnership. [S1]\n"
        "- Legacy uncategorized insight. [S2]"
    )

    items = workflow.parse_cited_insights(
        raw,
        sources(),
    )

    assert [
        item["category"]
        for item in items
    ] == [
        "Other",
        "Other",
    ]


def test_missing_or_invalid_source_never_creates_an_insight():
    raw = (
        "- [Product & Innovation] An uncited claim.\n"
        "- [Hiring & Talent] Claim with invalid ID. [S99]"
    )

    assert workflow.parse_cited_insights(
        raw,
        sources(),
    ) == []


def test_api_schema_accepts_category_and_validates_unknown_values():
    item = InsightItemResponse(
        text="Acme signed a partnership.",
        category="M&A / Partnerships",
        sources=[],
    )

    assert item.category == "M&A / Partnerships"

    unknown = InsightItemResponse(
        text="Legacy claim",
        category="Sales!!!",
        sources=[],
    )

    assert unknown.category == "Other"

    previous = InsightItemResponse(
        text="Earlier response",
        sources=[],
    )

    assert previous.category == "Other"


def test_analysis_keeps_single_llm_call(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1

            assert (
                "M&A / Partnerships"
                in messages[0].content
            )

            return SimpleNamespace(
                content=(
                    "- [M&A / Partnerships] "
                    "Acme signed a partnership. [S1]"
                )
            )

    fake = FakeLLM()

    monkeypatch.setattr(
        workflow,
        "llm_analysis",
        fake,
    )

    result = asyncio.run(
        workflow.analysis_node({
            "company_name": "Acme",
            "sources": sources(),
        })
    )

    assert fake.calls == 1

    assert (
        result["insight_items"][0]["category"]
        == "M&A / Partnerships"
    )
