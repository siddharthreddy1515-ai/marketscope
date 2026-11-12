
import asyncio

import workflow


def run_fake_workflow(monkeypatch, graph_result):
    class FakeGraph:
        async def ainvoke(self, state, config=None):
            assert state["company_name"] == "Acme"
            return graph_result

    monkeypatch.setattr(
        workflow,
        "app_graph",
        FakeGraph(),
    )

    return asyncio.run(
        workflow.run_workflow("Acme")
    )


def sample_source():
    return {
        "title": "Acme product announcement",
        "url": "https://example.com/product",
        "content": "Acme launched a new product.",
        "published_at": None,
    }


def test_completed_workflow_preserves_llm_usage(
    monkeypatch,
):
    result = run_fake_workflow(
        monkeypatch,
        {
            "sources": [sample_source()],
            "report": "# Acme Report",
            "research_attempts": 1,
            "llm_usage": {
                "input_tokens": 1200,
                "output_tokens": 160,
                "total_tokens": 1360,
            },
        },
    )

    assert result["status"] == "completed"

    assert result["llm_usage"] == {
        "input_tokens": 1200,
        "output_tokens": 160,
        "total_tokens": 1360,
    }


def test_missing_usage_remains_unknown(
    monkeypatch,
):
    result = run_fake_workflow(
        monkeypatch,
        {
            "sources": [sample_source()],
            "report": "# Acme Report",
            "research_attempts": 1,
        },
    )

    assert result["status"] == "completed"
    assert result["llm_usage"] is None


def test_failed_research_has_no_llm_usage(
    monkeypatch,
):
    result = run_fake_workflow(
        monkeypatch,
        {
            "sources": [],
            "report": None,
            "research_attempts": 3,
        },
    )

    assert result["status"] == "failed"
    assert result["llm_usage"] is None
