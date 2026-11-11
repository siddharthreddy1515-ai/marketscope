
import asyncio
from types import SimpleNamespace

import workflow


def sample_sources():
    return [
        {
            "title": "Acme Partnership Announcement",
            "url": "https://example.com/partnership",
            "content": "Acme announced a technology partnership.",
            "published_at": None,
        },
        {
            "title": "Acme Product Launch",
            "url": "https://example.com/product",
            "content": "Acme introduced a new software product.",
            "published_at": None,
        },
    ]


def test_summary_maps_supporting_sources():
    raw = (
        "SUMMARY: Acme's recent developments include "
        "a partnership and a product launch. [S1] [S2]\n"
        "- [M&A / Partnerships] Acme signed a partnership. [S1]\n"
        "- [Product & Innovation] Acme launched software. [S2]"
    )

    summary = workflow.parse_cited_summary(
        raw,
        sample_sources(),
    )

    assert summary is not None
    assert summary["text"] == (
        "Acme's recent developments include "
        "a partnership and a product launch."
    )

    assert len(summary["sources"]) == 2
    assert summary["sources"][0]["url"] == (
        "https://example.com/partnership"
    )
    assert summary["sources"][1]["url"] == (
        "https://example.com/product"
    )


def test_summary_requires_valid_citations():
    invalid = "SUMMARY: Acme dominates its market. [S99]"
    uncited = "SUMMARY: Acme dominates its market."

    assert workflow.parse_cited_summary(
        invalid, sample_sources()
    ) is None

    assert workflow.parse_cited_summary(
        uncited, sample_sources()
    ) is None


def test_summary_uses_existing_analysis_call(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1

            return SimpleNamespace(
                content=(
                    "SUMMARY: Acme announced a partnership "
                    "and launched software. [S1] [S2]\n"
                    "- [M&A / Partnerships] "
                    "Acme signed a partnership. [S1]\n"
                    "- [Product & Innovation] "
                    "Acme launched software. [S2]"
                )
            )

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        workflow, "llm_analysis", fake_llm
    )

    result = asyncio.run(
        workflow.analysis_node({
            "company_name": "Acme",
            "sources": sample_sources(),
        })
    )

    assert fake_llm.calls == 1
    assert len(result["insight_items"]) == 2

    summary = result["executive_summary"]
    assert summary is not None
    assert len(summary["sources"]) == 2
