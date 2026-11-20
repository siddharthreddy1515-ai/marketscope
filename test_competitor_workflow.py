
import asyncio
import json
from types import SimpleNamespace

import workflow


def setup_mock_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_TAVILY", "true")
    monkeypatch.chdir(tmp_path)

    mock_dir = tmp_path / "mocks"
    mock_dir.mkdir()

    return mock_dir


def write_company_fixture(mock_dir, results):
    (mock_dir / "Acme.json").write_text(
        json.dumps({"results": results}),
        encoding="utf-8",
    )


def company_sources():
    return [
        {
            "title": f"Acme article {i}",
            "url": f"https://example.com/company-{i}",
            "content": (
                f"Acme announced a product development {i}."
            ),
        }
        for i in range(1, 4)
    ]


def install_fake_llm(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1

            return SimpleNamespace(
                content=(
                    "SUMMARY: Acme announced a product "
                    "development. [S1]\n"
                    "- [Product & Innovation] "
                    "Acme announced a product "
                    "development. [S1]\n"
                    "SENTIMENT [S1]: Neutral\n"
                    "SENTIMENT [S2]: Neutral\n"
                    "SENTIMENT [S3]: Neutral"
                )
            )

    fake = FakeLLM()
    monkeypatch.setattr(workflow, "llm_analysis", fake)

    return fake


def test_successful_research_runs_competitor_stage(
    monkeypatch,
    tmp_path,
):
    mock_dir = setup_mock_environment(
        monkeypatch, tmp_path
    )

    write_company_fixture(
        mock_dir,
        company_sources(),
    )

    (mock_dir / "Acme_competitors.json").write_text(
        json.dumps({
            "results": [
                {
                    "title": "Acme rival comparison",
                    "url": "https://example.com/competition",
                    "content": (
                        "A comparison of Acme and "
                        "other analytics products."
                    ),
                }
            ]
        }),
        encoding="utf-8",
    )

    fake_llm = install_fake_llm(monkeypatch)

    result = asyncio.run(
        workflow.run_workflow("Acme")
    )

    assert result["status"] == "completed"

    assert result["competitor_research_attempts"] == 1

    assert len(result["competitor_sources"]) == 1

    assert result["competitor_sources"][0]["url"] == (
        "https://example.com/competition"
    )

    # The existing model call is still made once.
    assert fake_llm.calls == 1

    # Ordinary company sources remain available.
    assert len(result["sources"]) == 3


def test_competitor_research_failure_does_not_block_report(
    monkeypatch,
    tmp_path,
):
    mock_dir = setup_mock_environment(
        monkeypatch, tmp_path
    )

    write_company_fixture(
        mock_dir,
        company_sources(),
    )

    # Deliberately do not create
    # Acme_competitors.json.

    fake_llm = install_fake_llm(monkeypatch)

    result = asyncio.run(
        workflow.run_workflow("Acme")
    )

    assert result["status"] == "completed"
    assert result["report"]

    assert result["competitor_research_attempts"] == 1
    assert result["competitor_sources"] == []

    assert result["competitor_research_warnings"]
    assert fake_llm.calls == 1


def test_failed_company_research_skips_competitor_stage(
    monkeypatch,
    tmp_path,
):
    mock_dir = setup_mock_environment(
        monkeypatch, tmp_path
    )

    write_company_fixture(mock_dir, [])

    fake_llm = install_fake_llm(monkeypatch)

    result = asyncio.run(
        workflow.run_workflow("Acme")
    )

    assert result["status"] == "failed"

    # No competitor search should occur when
    # company research has no usable evidence.
    assert result["competitor_research_attempts"] == 0
    assert result["competitor_sources"] == []

    assert fake_llm.calls == 0
