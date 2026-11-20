"""Offline synthetic regression tests for the evidence presentation contract."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
import workflow
from analysis_contract import build_evidence_registry


def packet():
    return [{"title": f"Acme launch {i}", "url": f"https://example.com/acme-{i}",
             "content": f"Acme launched product {i} for business users.", "published_at": None}
            for i in range(1, 4)]


def model_payload(sources, status="complete"):
    registry = build_evidence_registry(sources)
    selected = next(iter(registry))
    insight = {"event": "Product launch", "text": sources[0]["content"],
               "category": "Product & Innovation", "evidence_ids": [selected]}
    result = {"schema_version": 2, "status": "complete", "insights": [insight],
              "context_items": [], "sentiments": [], "competitors": []}
    if status == "partial":
        result["insights"].append({**insight, "text": "Unsupported other development.", "evidence_ids": ["S99-0000000000000000-0-12"]})
    if status == "insufficient_evidence":
        result.update(status="insufficient_evidence", insights=[])
    return result


@pytest.mark.parametrize("expected", ["complete", "partial", "insufficient_evidence", "malformed"])
def test_real_analysis_status_and_api_contract(monkeypatch, expected):
    monkeypatch.setenv("ENABLE_LIVE_ANALYSIS", "true")
    sources = packet()
    payload = {"unexpected": "malformed"} if expected == "malformed" else model_payload(sources, expected)
    class FakeModel:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            assert "EVIDENCE-ID AND CONTEXT CONTRACT" in messages[0].content
            return SimpleNamespace(content=json.dumps(payload), response_metadata={"finish_reason": "stop"})
    model = FakeModel()
    monkeypatch.setattr(workflow, "llm_analysis", model)
    result = asyncio.run(workflow.analysis_node({"company_name": "Acme", "sources": sources, "competitor_sources": []}))
    assert result["analysis_status"] == expected
    assert model.calls == 1
    comparison_source = {"title": "Comparison", "url": "https://example.com/comparison",
                         "content": "Alternative products reviewed.", "published_at": None}
    async def fake_workflow(*args, **kwargs):
        return {**result, "status": "failed" if expected == "malformed" else "completed",
                "error_code": "ANALYSIS_MALFORMED" if expected == "malformed" else None,
                "report": "Unverified API draft", "sources": sources,
                "competitor_sources": [comparison_source],
                "research_warnings": ["Coverage may be incomplete."]}
    monkeypatch.setattr(main, "run_workflow", fake_workflow)
    response = TestClient(main.app).post("/api/v1/analyze", json={"company_name": "Acme"})
    assert response.status_code == (502 if expected == "malformed" else 200)
    if expected == "malformed":
        payload = response.json()
        assert payload["detail"] == "The analysis response could not be read. Please try again."
        assert payload["research"]["sources"] == sources
        assert payload["research"]["comparison_sources"] == [comparison_source]
        assert "insight_items" not in payload["research"]
        assert "report" not in payload["research"]
    else:
        data = response.json()
        assert data["analysis_status"] == expected
        assert data["analysis_format"] == "evidence_ids_v2"
        assert data["comparison_sources"] == [comparison_source]
        assert data["research_warnings"] == ["Coverage may be incomplete."]
        if expected != "insufficient_evidence":
            evidence = data["insight_items"][0]["evidence"][0]
            assert evidence == result["insight_items"][0]["evidence"][0]
            assert evidence["quote"] == sources[0]["content"]


def test_real_graph_retains_evidence_without_extra_model_calls(monkeypatch, tmp_path):
    sources = packet()
    fixture = tmp_path / "sources.json"
    fixture.write_text(json.dumps({"results": sources}), encoding="utf-8")
    monkeypatch.setenv("MOCK_TAVILY", "true")
    monkeypatch.setattr(workflow, "mock_path", lambda company: str(fixture))
    monkeypatch.setattr(workflow, "competitor_mock_path", lambda company: str(tmp_path / "absent.json"))
    class FakeModel:
        calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            return SimpleNamespace(content=json.dumps(model_payload(sources)), response_metadata={"finish_reason": "stop"})
    model = FakeModel()
    monkeypatch.setattr(workflow, "llm_analysis", model)
    result = asyncio.run(workflow.run_workflow("Acme"))
    assert result["status"] == "completed"
    assert result["analysis_format"] == "evidence_ids_v2"
    assert result["insight_items"][0]["evidence"][0]["quote"] == sources[0]["content"]
    assert result["research_attempts"] <= workflow.MAX_RESEARCH_CALLS
    assert model.calls == 1


@pytest.mark.parametrize("model_result, expected_status", [
    ("missing_overview", "insufficient_evidence"),
    ("overview_only", "insufficient_evidence"),
    ("bad_development", "partial"),
])
def test_company_basics_survive_without_a_development(monkeypatch, tmp_path, model_result, expected_status):
    monkeypatch.setenv("ENABLE_LIVE_ANALYSIS", "true")
    monkeypatch.setenv("MOCK_TAVILY", "true")
    sources = [
        {"title": "Acme", "url": "https://acme.example/about",
         "content": "Acme makes electric motorcycles for urban riders.", "published_at": None},
    ]
    fixture = tmp_path / "sources.json"
    fixture.write_text(json.dumps({"results": sources}), encoding="utf-8")
    monkeypatch.setattr(workflow, "mock_path", lambda company: str(fixture))
    monkeypatch.setattr(workflow, "competitor_mock_path", lambda company: str(tmp_path / "absent.json"))
    passage = next(iter(build_evidence_registry(sources)))

    class FakeModel:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            assert "what the company does, a named product or service" in messages[0].content
            assert passage in messages[0].content
            payload = {
                "schema_version": 2, "status": "insufficient_evidence", "insights": [],
                "context_items": [], "sentiments": [], "competitors": [],
            }
            if model_result != "missing_overview":
                payload["context_items"].append({"event": "Company overview", "text": sources[0]["content"],
                                                 "category": "Other", "evidence_ids": [passage]})
            if model_result == "bad_development":
                payload.update(status="complete", insights=[{
                    "event": "Unsupported launch", "text": "Acme launched a new vehicle.",
                    "evidence_ids": ["S99-0000000000000000-0-12"],
                }])
            return SimpleNamespace(content=json.dumps(payload), response_metadata={"finish_reason": "stop"})

    model = FakeModel()
    monkeypatch.setattr(workflow, "llm_analysis", model)
    response = TestClient(main.app).post("/api/v1/analyze", json={"company_name": "Acme"})
    assert response.status_code == 200
    data = response.json()
    assert data["analysis_status"] == expected_status
    assert data["insight_items"] == []
    assert len(data["context_items"]) == 1
    evidence = data["context_items"][0]["evidence"][0]
    assert evidence["quote"] == sources[0]["content"]
    assert evidence["url"] == sources[0]["url"]
    assert model.calls == 1
