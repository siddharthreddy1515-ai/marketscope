
import asyncio
from types import SimpleNamespace

import workflow


def sample_sources():
    return [
        {
            "title": "Acme product announcement",
            "url": "https://example.com/product",
            "content": "Acme announced a successful product launch.",
            "published_at": None,
        },
        {
            "title": "Acme service disruption",
            "url": "https://example.com/outage",
            "content": "Acme experienced a service outage.",
            "published_at": None,
        },
        {
            "title": "Unrelated announcement",
            "url": "https://example.com/other",
            "content": "Another company announced a new office.",
            "published_at": None,
        },
    ]


def test_source_sentiment_maps_valid_source_ids():
    raw = (
        "SENTIMENT [S1]: Positive\n"
        "SENTIMENT [S2]: Negative\n"
        "SENTIMENT [S3]: Unclear"
    )

    result = workflow.parse_source_sentiments(
        raw,
        sample_sources(),
    )

    assert [item["label"] for item in result] == [
        "Positive",
        "Negative",
        "Unclear",
    ]

    assert [item["source_id"] for item in result] == [
        "S1",
        "S2",
        "S3",
    ]

    assert result[0]["url"] == (
        "https://example.com/product"
    )


def test_invalid_ids_and_labels_are_rejected():
    raw = (
        "SENTIMENT [S99]: Positive\n"
        "SENTIMENT [S1]: Excited\n"
        "SENTIMENT [S2]: Negative"
    )

    result = workflow.parse_source_sentiments(
        raw,
        sample_sources(),
    )

    assert len(result) == 1
    assert result[0]["source_id"] == "S2"
    assert result[0]["label"] == "Negative"


def test_distribution_excludes_unclear_and_reports_coverage():
    raw = (
        "SENTIMENT [S1]: Positive\n"
        "SENTIMENT [S2]: Negative\n"
        "SENTIMENT [S3]: Unclear"
    )

    classifications = workflow.parse_source_sentiments(
        raw,
        sample_sources(),
    )

    distribution = workflow.calculate_sentiment_distribution(
        classifications,
        sample_sources(),
    )

    assert distribution["positive"] == 1
    assert distribution["negative"] == 1
    assert distribution["neutral"] == 0

    assert distribution["assessed"] == 2
    assert distribution["unassessed"] == 1
    assert distribution["total_sources"] == 3


def test_analysis_still_uses_one_llm_call(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1

            return SimpleNamespace(
                content=(
                    "SUMMARY: Acme had a product announcement "
                    "and a service disruption. [S1] [S2]\n"
                    "- [Product & Innovation] "
                    "Acme launched a product. [S1]\n"
                    "- [Other] "
                    "Acme experienced an outage. [S2]\n"
                    "SENTIMENT [S1]: Positive\n"
                    "SENTIMENT [S2]: Negative\n"
                    "SENTIMENT [S3]: Unclear"
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
            "sources": sample_sources(),
        })
    )

    assert fake_llm.calls == 1

    # Existing functionality must still work.
    assert len(result["insight_items"]) == 2
    assert result["executive_summary"] is not None

    # New functionality.
    assert len(result["source_sentiments"]) == 3

    distribution = result["sentiment_distribution"]

    assert distribution["assessed"] == 2
    assert distribution["positive"] == 1
    assert distribution["negative"] == 1
