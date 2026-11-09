
import asyncio

from langchain_core.callbacks import BaseCallbackHandler

import workflow


def test_research_exhaustion_reports_failure(monkeypatch, tmp_path):
    # All three research attempts will read this empty fixture.
    fixture = tmp_path / "empty.json"
    fixture.write_text('{"results": []}', encoding="utf-8")

    monkeypatch.setenv("MOCK_TAVILY", "true")
    monkeypatch.setattr(
        workflow,
        "mock_path",
        lambda company: str(fixture),
    )

    # Prevent external tracing in this test.
    monkeypatch.setattr(
        workflow,
        "langfuse_handler",
        BaseCallbackHandler(),
    )

    # Ensure the LLM is never called when research fails.
    class ForbiddenLLM:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError(
                "Analysis must not run after failed research"
            )

    monkeypatch.setattr(workflow, "llm_analysis", ForbiddenLLM())

    # Test the real compiled LangGraph through run_workflow.
    result = asyncio.run(workflow.run_workflow("Acme"))

    assert result["status"] == "failed"
    assert result["research_attempts"] == 3
    assert result["report"] is None
