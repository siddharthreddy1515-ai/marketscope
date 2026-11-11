
import asyncio
from types import SimpleNamespace

import main
import workflow


def sample_sources():
    return [
        {
            "title": "Acme Series E Announcement",
            "url": "https://example.com/funding",
            "content": (
                "Acme raised a Series E funding round "
                "led by an investment firm."
            ),
            "published_at": None,
        }
    ]


def test_funding_category_is_preserved():
    raw = (
        "- [Funding & Investment] "
        "Acme raised a Series E funding round. [S1]"
    )

    items = workflow.parse_cited_insights(
        raw,
        sample_sources(),
    )

    assert len(items) == 1

    assert items[0]["category"] == (
        "Funding & Investment"
    )


def test_api_schema_preserves_funding_category():
    item = main.InsightItemResponse(
        text="Acme raised a Series E.",
        category="Funding & Investment",
        sources=[],
    )

    assert item.category == "Funding & Investment"


def test_analysis_prompt_distinguishes_funding_from_partnerships(
    monkeypatch,
):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1

            prompt = messages[0].content

            assert "Funding & Investment" in prompt

            assert (
                "A funding round is not automatically "
                "a partnership or acquisition"
            ) in prompt

            assert (
                "Do not claim future growth"
            ) in prompt

            return SimpleNamespace(
                content=(
                    "SUMMARY: Acme announced a "
                    "Series E funding round. [S1]\n"
                    "- [Funding & Investment] "
                    "Acme raised a Series E. [S1]\n"
                    "SENTIMENT [S1]: Neutral"
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
            "sources": sample_sources(),
            "competitor_sources": [],
        })
    )

    assert fake.calls == 1

    assert result["insight_items"][0]["category"] == (
        "Funding & Investment"
    )

    assert result["executive_summary"] is not None
