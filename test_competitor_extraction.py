
import asyncio
from types import SimpleNamespace

import workflow


def competitor_sources():
    return [
        {
            "title": "Acme vs Northstar Analytics",
            "url": "https://example.com/comparison",
            "content": (
                "Acme and Northstar Analytics both sell "
                "customer analytics dashboards to small businesses."
            ),
            "published_at": None,
        },
        {
            "title": "Acme and Harbor Cloud partnership",
            "url": "https://example.com/partnership",
            "content": (
                "Acme partnered with Harbor Cloud "
                "for infrastructure services."
            ),
            "published_at": None,
        },
    ]


def company_sources():
    return [
        {
            "title": "Acme product launch",
            "url": "https://example.com/product",
            "content": "Acme released a new product.",
            "published_at": None,
        }
    ]


def test_competitor_maps_dedicated_citations():
    raw = (
        "COMPETITOR: Northstar Analytics | "
        "Both sell customer analytics dashboards "
        "to small businesses. [C1]"
    )

    result = workflow.parse_cited_competitors(
        raw,
        competitor_sources(),
    )

    assert len(result) == 1
    assert result[0]["name"] == "Northstar Analytics"

    assert "customer analytics" in (
        result[0]["overlap"]
    )

    assert result[0]["sources"][0]["url"] == (
        "https://example.com/comparison"
    )


def test_invalid_competitor_citations_are_rejected():
    raw = (
        "COMPETITOR: Northstar Analytics | "
        "Competing dashboards. [C99]\n"
        "COMPETITOR: Northstar Analytics | "
        "Competing dashboards. [S1]\n"
        "COMPETITOR: Northstar Analytics | "
        "Competing dashboards."
    )

    result = workflow.parse_cited_competitors(
        raw,
        competitor_sources(),
    )

    assert result == []


def test_competitor_name_must_appear_in_cited_evidence():
    raw = (
        "COMPETITOR: Fabricated Labs | "
        "Competing analytics platforms. [C1]"
    )

    result = workflow.parse_cited_competitors(
        raw,
        competitor_sources(),
    )

    assert result == []


def test_duplicate_competitors_are_removed():
    raw = (
        "COMPETITOR: Northstar Analytics | "
        "Competing analytics dashboards. [C1]\n"
        "COMPETITOR: northstar analytics | "
        "Competing reporting software. [C1]"
    )

    result = workflow.parse_cited_competitors(
        raw,
        competitor_sources(),
    )

    assert len(result) == 1


def test_existing_analysis_call_extracts_competitors(
    monkeypatch,
):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1

            prompt = messages[0].content

            # Both evidence sets must reach GPT-4o.
            assert "Acme released a new product" in prompt
            assert "Northstar Analytics" in prompt
            assert "C1" in prompt

            return SimpleNamespace(
                content=(
                    "SUMMARY: Acme released a product. [S1]\n"
                    "- [Product & Innovation] "
                    "Acme released a product. [S1]\n"
                    "SENTIMENT [S1]: Neutral\n"
                    "COMPETITOR: Northstar Analytics | "
                    "Both sell analytics dashboards "
                    "to small businesses. [C1]"
                )
            )

    fake_llm = FakeLLM()

    monkeypatch.setattr(
        workflow,
        "llm_analysis",
        fake_llm,
    )

    result = asyncio.run(
        workflow.analysis_node({
            "company_name": "Acme",
            "sources": company_sources(),
            "competitor_sources": competitor_sources(),
        })
    )

    assert fake_llm.calls == 1

    # Existing capabilities remain intact.
    assert len(result["insight_items"]) == 1
    assert result["executive_summary"] is not None
    assert (
        result["sentiment_distribution"]["neutral"]
        == 1
    )

    # New competitor extraction.
    competitors = result["competitors"]

    assert len(competitors) == 1
    assert competitors[0]["name"] == (
        "Northstar Analytics"
    )
