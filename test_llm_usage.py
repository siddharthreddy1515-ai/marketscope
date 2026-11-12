
import asyncio
from types import SimpleNamespace

import workflow


def sample_sources():
    return [
        {
            "title": "Acme Product Launch",
            "url": "https://example.com/acme-launch",
            "content": (
                "Acme launched a new analytics product."
            ),
            "published_at": None,
        }
    ]


def run_analysis_with_usage(
    monkeypatch,
    usage_metadata,
):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1

            return SimpleNamespace(
                content=(
                    "SUMMARY: Acme launched a new "
                    "analytics product. [S1]\n"
                    "- [Product & Innovation] "
                    "Acme launched an analytics "
                    "product. [S1]\n"
                    "SENTIMENT [S1]: Neutral"
                ),
                usage_metadata=usage_metadata,
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

    return result


def test_analysis_records_llm_token_usage(
    monkeypatch,
):
    result = run_analysis_with_usage(
        monkeypatch,
        {
            "input_tokens": 1200,
            "output_tokens": 160,
            "total_tokens": 1360,
        },
    )

    usage = result["llm_usage"]

    assert usage["input_tokens"] == 1200
    assert usage["output_tokens"] == 160
    assert usage["total_tokens"] == 1360


def test_missing_usage_is_not_reported_as_zero(
    monkeypatch,
):
    result = run_analysis_with_usage(
        monkeypatch,
        None,
    )

    assert result["llm_usage"] is None
