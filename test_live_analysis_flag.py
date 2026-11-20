"""Exercise the public analysis gate without provider access."""

import builtins
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main


@pytest.mark.parametrize("value", [None, "", "false", "1", "yes", "on", "true,false"])
def test_disabled_analysis_never_runs_workflow(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ENABLE_LIVE_ANALYSIS", raising=False)
    else:
        monkeypatch.setenv("ENABLE_LIVE_ANALYSIS", value)
    workflow = AsyncMock(side_effect=AssertionError("Disabled analysis started the workflow"))
    monkeypatch.setattr(main, "run_workflow", workflow)

    client = TestClient(main.app)
    response = client.post("/api/v1/analyze", json={"company_name": "Acme"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Analysis is temporarily unavailable."}
    workflow.assert_not_called()


def test_disabled_app_starts_and_serves_dashboard_without_provider_imports(monkeypatch):
    for name in ("ENABLE_LIVE_ANALYSIS", "OPENAI_API_KEY", "TAVILY_API_KEY",
                 "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    original_import = builtins.__import__
    provider_modules = {"workflow", "openai", "tavily", "langchain_openai", "langchain_tavily", "langfuse"}

    def import_without_providers(name, *args, **kwargs):
        if name.partition(".")[0] in provider_modules:
            pytest.fail(f"Provider module imported in disabled mode: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_providers)
    spec = importlib.util.spec_from_file_location("disabled_app_probe", main.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    with TestClient(module.app) as client:
        assert client.get("/health").json() == {"status": "ok", "version": "0.3.0"}
        assert "source-brief-2026-11-19" in client.get("/").text
        # The kill switch takes precedence even when the request body is malformed.
        response = client.post("/api/v1/analyze", content="{", headers={"Content-Type": "application/json"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Analysis is temporarily unavailable."}
        assert set(client.get("/openapi.json").json()["paths"]) == {"/", "/health", "/api/v1/analyze"}


@pytest.mark.parametrize("value", ["true", " TRUE "])
def test_enabled_analysis_uses_existing_workflow_contract(monkeypatch, value):
    monkeypatch.setenv("ENABLE_LIVE_ANALYSIS", value)
    workflow = AsyncMock(return_value={
        "status": "completed",
        "analysis_status": "complete",
        "report": "# Saved report",
        "research_attempts": 1,
    })
    monkeypatch.setitem(sys.modules, "workflow", SimpleNamespace(run_workflow=workflow))

    response = TestClient(main.app).post(
        "/api/v1/analyze",
        json={"company_name": "Acme", "official_domain": "https://www.example.com/products"},
    )

    assert response.status_code == 200
    assert response.json()["report"] == "# Saved report"
    workflow.assert_awaited_once_with("Acme", official_domain="example.com")
