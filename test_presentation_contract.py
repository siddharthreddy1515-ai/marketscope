"""Offline synthetic regression tests for the evidence presentation contract."""
import asyncio
import json
from types import SimpleNamespace

import pytest
import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError

import main
import workflow
from analysis_contract import build_evidence_registry, temporal_context


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


def test_analysis_prioritizes_document_excerpts_without_renumbering_sources(monkeypatch):
    direct = "Acme and a grocery chain began drone delivery in Dallas. The service is available to local shoppers."
    repeat = " ".join(direct.split()).replace(" Dallas.", "  Dallas.")
    sources = [
        {"title": "Acme launches drone delivery", "url": "https://www.google.com/url?q=search-preview",
         "content": direct, "published_at": "2026-11-17"},
        {"title": "Acme expansion", "url": "https://news.example/acme",
         "content": repeat, "published_at": "2026-11-17"},
        {"title": "Acme partner announcement", "url": "https://www.google.com/url?q=other-preview",
         "content": "Acme and an independent restaurant announced a delivery partnership",
         "published_at": "2026-11-12"},
        *packet()[:1],
    ]
    registry = build_evidence_registry(sources)
    ids = {row["source_id"]: key for key, row in registry.items()}
    payload = model_payload(sources)
    payload["insights"][0].update(text=direct, evidence_ids=[ids["S2"]])
    payload["insights"].append({"event": "Repeated launch", "text": direct, "category": "Other",
                                "evidence_ids": [ids["S1"]]})

    class FakeModel:
        async def ainvoke(self, messages):
            prompt = messages[0].content
            assert ids["S1"] not in prompt
            assert prompt.index(ids["S4"]) < prompt.index(ids["S2"]) < prompt.index(ids["S3"])
            assert "SOURCE DEPTH: short search preview" in prompt
            return SimpleNamespace(content=json.dumps(payload), response_metadata={"finish_reason": "stop"})

    monkeypatch.setattr(workflow, "llm_analysis", FakeModel())
    expected_sources = [dict(source) for source in sources]
    result = asyncio.run(workflow.analysis_node({"company_name": "Acme", "sources": sources}))
    assert result["analysis_status"] == "partial"
    assert sources == expected_sources
    assert result["insight_items"][0]["evidence"] == [registry[ids["S2"]]]
    assert len(result["insight_items"]) == 1


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


def test_provider_timeout_keeps_research_without_retrying(monkeypatch, tmp_path):
    sources = packet()
    fixture = tmp_path / "sources.json"
    fixture.write_text(json.dumps({"results": sources}), encoding="utf-8")
    monkeypatch.setenv("ENABLE_LIVE_ANALYSIS", "true")
    monkeypatch.setenv("MOCK_TAVILY", "true")
    monkeypatch.setattr(workflow, "mock_path", lambda company: str(fixture))
    monkeypatch.setattr(workflow, "competitor_mock_path", lambda company: str(tmp_path / "absent.json"))

    class TimedOutModel:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            request = httpx.Request("POST", "https://example.test/chat/completions")
            raise RuntimeError("SDK adapter error") from APITimeoutError(request=request)

    model = TimedOutModel()
    monkeypatch.setattr(workflow, "llm_analysis", model)
    response = TestClient(main.app).post("/api/v1/analyze", json={"company_name": "Acme"})
    assert response.status_code == 504
    payload = response.json()
    assert payload["detail"] == "The analysis provider timed out."
    assert payload["research"]["sources"] == sources
    assert payload["research"]["analysis_warnings"] == ["The analysis provider timed out."]
    assert "report" not in payload["research"]
    assert model.calls == 1


@pytest.mark.parametrize("title, url", [
    ("Terms of Service | Acme", "https://example.com/terms"),
    ("Acme", "https://tavily.com/knowledge/company/example"),
])
def test_reference_page_excerpt_is_kept_as_background(monkeypatch, title, url):
    sources = [{"title": title, "url": url,
                "content": "Acme provides delivery services to customers.", "published_at": None}]
    payload = model_payload(sources)

    class FakeModel:
        async def ainvoke(self, messages):
            return SimpleNamespace(content=json.dumps(payload), response_metadata={"finish_reason": "stop"})

    monkeypatch.setattr(workflow, "llm_analysis", FakeModel())
    result = asyncio.run(workflow.analysis_node({"company_name": "Acme", "sources": sources}))
    assert result["analysis_status"] == "insufficient_evidence"
    assert result["insight_items"] == []
    assert any(item["evidence"][0]["quote"] == sources[0]["content"] for item in result["context_items"])


@pytest.mark.parametrize("prefix, description", [
    ("Description: ", "Acme India is a payroll and HR systems provider."),
    ("", "Business areas: Payroll services, tax filing and time tracking."),
])
def test_regional_profile_supplies_basic_company_overview(prefix, description):
    source = {"title": "Acme India", "url": "https://tavily.com/knowledge/company/example",
              "content": prefix + description}
    overview = workflow.company_overview_from_sources("Acme", [source], build_evidence_registry([source]))
    assert overview["text"] == description
    assert overview["evidence"][0]["quote"] == source["content"]


def test_offering_uses_exact_official_product_description_before_profile():
    product_sentence = "Run payroll with Acme Vista, a payroll software platform designed for employers."
    sources = [
        {"title": "Acme India", "url": "https://tavily.com/knowledge/company/acme",
         "content": "Products: Payroll Processing, HR Administration, Time Tracking, Reporting"},
        {"title": "Acme Expands Worldwide", "url": "https://acme.example/press/acme-payroll-solutions",
         "content": "# Acme Expands Availability of Payroll Software in New Markets"},
        {"title": "Acme products", "url": "https://acme.example/what-we-offer/products.aspx",
         "content": f"BOOK A DEMO [...] {product_sentence} Contact our sales team."},
    ]
    registry = build_evidence_registry(sources)
    official = workflow.company_offering_from_sources("Acme", "acme.example", registry, [])
    assert official["text"] == product_sentence
    assert official["text"] in official["evidence"][0]["quote"]
    assert official["sources"][0]["url"] == sources[2]["url"]

    profile = workflow.company_offering_from_sources("Acme", None, registry, [])
    assert profile["text"] == "Products: Payroll Processing, HR Administration, Time Tracking"
    assert profile["text"] in profile["evidence"][0]["quote"]


@pytest.mark.parametrize("quotes, expected", [
    (["Chennai, India, 7 January, 2026: Acme, a payroll provider, has announced "
      "enhancements to its regional payroll services."], "historical"),
    (["Chennai, India, September 7, 2026: Acme announced enhancements to its regional payroll services."],
     "timing_uncertain"),
    (["7 January, 2026: Acme announced a new office. The report describes enhancements to its regional payroll services."],
     "timing_uncertain"),
    (["7 January, 2026: Acme plans to announce enhancements to its regional payroll services."], "timing_uncertain"),
    (["31 February, 2026: Acme announced enhancements to its regional payroll services."], "timing_uncertain"),
    (["7 January, 2026: Acme announced enhancements to its regional payroll services.",
      "September 7, 2026: Acme announced enhancements to its regional payroll services."], "timing_uncertain"),
])
def test_announcement_dateline_must_date_the_same_action(quotes, expected):
    result = temporal_context(
        "Acme announced enhancements to its regional payroll services.",
        [{"quote": quote} for quote in quotes],
        "Acme",
        {"start": "2026-08-22", "end": "2026-11-20"},
    )
    assert result["temporal_status"] == expected


@pytest.mark.parametrize("statement, quote, expected", [
    ("In 2023, a consortium of investors invested €49 million in Acme.",
     "Acme makes modular handsets. In 2023, a consortium of investors invested €49 million in Acme.",
     "historical"),
    ("In 2026, a consortium of investors invested €49 million in Acme.",
     "In 2026, a consortium of investors invested €49 million in Acme.",
     "timing_uncertain"),
    ("In 2023, a consortium of investors invested €49 million in Acme.",
     "In 2023, a consortium of investors announced plans to invest €49 million in Acme.",
     "timing_uncertain"),
    ("The fifth generation of the Acme Ring was released in June 2026, selling 5 million units.",
     "Acme is known for the Acme Ring, a wearable device. The fifth generation of the device was released in June 2026.",
     "historical"),
    ("The fifth generation of the Acme Ring was released in August 2026.",
     "Acme is known for the Acme Ring. The fifth generation of the device was released in August 2026.",
     "timing_uncertain"),
    ("The fifth generation of the Acme Ring was released in June 2026.",
     "Acme is known for the Acme Watch. The fifth generation of the device was released in June 2026.",
     "timing_uncertain"),
    ("Acme Ring 5 was released in June 2026.",
     "The Acme Ring 5 was released in June 2026.",
     "historical"),
])
def test_historical_date_requires_an_older_period_and_same_action(statement, quote, expected):
    result = temporal_context(statement, [{"quote": quote}], "Acme",
                              {"start": "2026-08-22", "end": "2026-11-20"})
    assert result["temporal_status"] == expected


def test_existing_integration_is_context_and_new_launch_is_retained(monkeypatch):
    capability = "Through a collaboration with Carrier, Acme integrates tracking data into its app."
    launch = "Acme launched pharmacy delivery on November 17, 2026."
    history = "In 2023, investors invested €49 million in Acme."
    overview = "Acme provides delivery services."
    offering = "The Acme app offers parcel tracking."
    claims = [capability, launch, history, overview, offering]
    sources = [{**row, "content": claim} for row, claim in zip(
        [*packet(), {"title": "About", "url": "https://example.com/about", "published_at": None},
         {"title": "Products", "url": "https://example.com/products", "published_at": None}], claims)]
    ids = list(build_evidence_registry(sources))
    labels = ["Carrier integration", "Pharmacy delivery", "Investment", "Company overview", "Products and services"]
    rows = [{"event": event, "text": claim, "category": "Other", "evidence_ids": [evidence_id]}
            for event, claim, evidence_id in zip(labels, claims, ids)]
    payload = dict(model_payload(sources), insights=rows[:3], context_items=rows[3:])

    class FakeModel:
        async def ainvoke(self, messages):
            return SimpleNamespace(content=json.dumps(payload), response_metadata={"finish_reason": "stop"})

    monkeypatch.setattr(workflow, "llm_analysis", FakeModel())
    result = asyncio.run(workflow.analysis_node({"company_name": "Acme", "sources": sources,
                                               "analysis_window": {"start": "2026-08-22", "end": "2026-11-20"}}))
    assert result["analysis_status"] == "complete"
    assert [item["text"] for item in result["insight_items"]] == [launch]
    context = {item["text"]: item for item in result["context_items"]}
    assert set(context) == {capability, history, overview, offering}
    assert context[history]["temporal_status"] == "historical"
    assert context[capability]["evidence"][0]["evidence_id"] == ids[0]


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
