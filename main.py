
import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, field_validator

from workflow_contract import SIGNAL_CATEGORIES, normalize_company_domain


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MarketScope Multi-Agent",
    version="0.3.0",
)


def require_live_analysis_enabled():
    if os.getenv("ENABLE_LIVE_ANALYSIS", "false").strip().lower() != "true":
        raise HTTPException(
            status_code=503,
            detail="Analysis is temporarily unavailable.",
        )


class LiveAnalysisRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def guarded_handler(request: Request):
            require_live_analysis_enabled()
            return await handler(request)

        return guarded_handler


analysis_router = APIRouter(route_class=LiveAnalysisRoute)


async def run_workflow(company_name: str, official_domain: str | None = None) -> dict:
    # Provider clients initialize only after the request passes the server-side gate.
    from workflow import run_workflow as execute

    return await execute(company_name, official_domain=official_domain)


# --------------------------------------------------
# Sentiment Response Models
# --------------------------------------------------

class SourceSentimentResponse(BaseModel):
    source_id: str
    label: str
    title: str
    url: str


class SentimentDistributionResponse(BaseModel):
    positive: int
    neutral: int
    negative: int
    assessed: int
    unassessed: int
    total_sources: int


# --------------------------------------------------
# Request Model
# --------------------------------------------------

class AnalyzeRequest(BaseModel):
    company_name: str
    official_domain: str | None = None

    @field_validator("official_domain")
    @classmethod
    def validate_official_domain(
        cls,
        value: str | None,
    ) -> str | None:
        return normalize_company_domain(value)

    @field_validator("company_name")
    @classmethod
    def validate_company_name(
        cls,
        v: str,
    ) -> str:
        v = v.strip()

        if not v:
            raise ValueError(
                "Company name cannot be empty."
            )

        if len(v) > 50:
            raise ValueError(
                "Company name must be under 50 characters."
            )

        if not re.match(
            r"^[a-zA-Z0-9\s\-\.]+$",
            v,
        ):
            raise ValueError(
                "Company name contains invalid characters."
            )

        return v


# --------------------------------------------------
# Research Response Models
# --------------------------------------------------

class SourceResponse(BaseModel):
    title: str
    url: str
    content: str
    published_at: str | None = None


class InsightSourceResponse(BaseModel):
    title: str
    url: str


class EvidenceQuoteResponse(BaseModel):
    source_id: str
    quote: str
    title: str
    url: str
    evidence_id: str | None = None


class CompetitorItemResponse(BaseModel):
    name: str
    overlap: str
    sources: list[InsightSourceResponse]
    evidence: list[EvidenceQuoteResponse] = Field(default_factory=list)


class ExecutiveSummaryResponse(BaseModel):
    text: str
    sources: list[InsightSourceResponse]


class InsightItemResponse(BaseModel):
    text: str
    sources: list[InsightSourceResponse]
    event: str | None = None
    evidence: list[EvidenceQuoteResponse] = Field(default_factory=list)
    temporal_status: Literal["historical", "timing_uncertain"] = "timing_uncertain"
    temporal_note: str | None = None
    category: str = "Other"

    @field_validator("category")
    @classmethod
    def normalize_category(
        cls,
        value: str,
    ) -> str:
        return (
            value
            if value in SIGNAL_CATEGORIES
            else "Other"
        )


class ResearchStatsResponse(BaseModel):
    returned: int
    accepted: int
    discarded: int


class ResearchAttemptResponse(BaseModel):
    attempt: int
    lane: Literal["first", "business", "official"]
    status: Literal["ok", "failed", "invalid_response"]
    returned: int
    new_sources: int


class ResearchCoverageResponse(BaseModel):
    status: Literal["empty", "background_only", "development_candidates", "mixed", "uncertain"]
    background_sources: int
    development_sources: int
    uncertain_sources: int
    limitations: list[str]


class AnalysisWindowResponse(BaseModel):
    start: str
    end: str


# --------------------------------------------------
# Main API Response
# --------------------------------------------------

class AnalyzeResponse(BaseModel):
    company_name: str
    official_domain: str | None = None
    generated_at: str | None = None
    status: str
    analysis_status: Literal["complete", "partial", "insufficient_evidence"]
    analysis_format: Literal["evidence_ids_v2", "quoted_v1", "legacy_lines"] | None = None
    analysis_warnings: list[str] = Field(default_factory=list)
    analysis_window: AnalysisWindowResponse | None = None
    context_items: list[InsightItemResponse] = Field(default_factory=list)
    context_warnings: list[str] = Field(default_factory=list)
    report: str
    research_attempts: int
    research_history: list[ResearchAttemptResponse] = Field(default_factory=list)
    research_coverage: ResearchCoverageResponse | None = None
    competitor_research_warnings: list[str] = Field(default_factory=list)
    sources: list[SourceResponse]
    comparison_sources: list[SourceResponse] = Field(default_factory=list)
    source_stats: ResearchStatsResponse
    research_warnings: list[str]
    insight_items: list[InsightItemResponse]

    executive_summary: (
        ExecutiveSummaryResponse | None
    ) = None

    source_sentiments: list[
        SourceSentimentResponse
    ] = Field(
        default_factory=list
    )

    sentiment_distribution: (
        SentimentDistributionResponse | None
    ) = None

    competitors: list[
        CompetitorItemResponse
    ] = Field(
        default_factory=list
    )


# --------------------------------------------------
# Company Analysis Endpoint
# --------------------------------------------------

@analysis_router.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    responses={503: {"description": "Analysis is temporarily unavailable."}},
)
async def analyze_endpoint(
    request: AnalyzeRequest,
):
    try:
        # Use website-assisted research when
        # the user provides an official domain.
        if request.official_domain is not None:
            workflow_call = run_workflow(
                request.company_name,
                official_domain=request.official_domain,
            )

        # Preserve existing name-only behavior.
        else:
            workflow_call = run_workflow(
                request.company_name
            )

        # Apply the timeout to both paths.
        result = await asyncio.wait_for(
            workflow_call,
            timeout=60.0,
        )

    except asyncio.TimeoutError:
        logging.error(
            "Analysis timed out for %s",
            request.company_name,
        )

        raise HTTPException(
            status_code=504,
            detail=(
                "The request timed out. "
                "Please try again."
            ),
        )

    except Exception:
        logging.exception(
            "Analysis failed for %s",
            request.company_name,
        )

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred.",
        )

    # --------------------------------------------------
    # Validate workflow result
    # --------------------------------------------------

    analysis_status = result.get("analysis_status")
    if result.get("error_code") == "ANALYSIS_MALFORMED" or (
        result.get("status") == "completed"
        and analysis_status not in ("complete", "partial", "insufficient_evidence")
    ):
        logging.warning("Malformed analysis response for %s", request.company_name)
        detail = "The analysis response could not be read. Please try again."
        sources = result.get("sources") or []
        if sources:
            return JSONResponse(status_code=502, content={
                "detail": detail,
                "research": {
                    "company_name": request.company_name,
                    "official_domain": request.official_domain,
                    "research_attempts": result.get("research_attempts", 0),
                    "research_coverage": result.get("research_coverage"),
                    "research_warnings": result.get("research_warnings", []),
                    "sources": [SourceResponse.model_validate(source).model_dump() for source in sources],
                    "comparison_sources": [
                        SourceResponse.model_validate(source).model_dump()
                        for source in result.get("competitor_sources", [])
                    ],
                },
            })
        raise HTTPException(
            status_code=502,
            detail=detail,
        )

    if (
        result.get("status") != "completed"
        or not isinstance(
            result.get("report"),
            str,
        )
    ):
        logging.warning(
            "Research could not produce a report for %s",
            request.company_name,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to generate a report from "
                "the available research."
            ),
        )

    # --------------------------------------------------
    # Return structured report
    # --------------------------------------------------

    return AnalyzeResponse(
        company_name=request.company_name,
        official_domain=request.official_domain,
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=result["status"],
        analysis_status=analysis_status,
        analysis_format=result.get("analysis_format"),
        analysis_warnings=result.get("analysis_warnings", []),
        analysis_window=result.get("analysis_window"),
        context_items=result.get("context_items", []),
        context_warnings=result.get("context_warnings", []),

        executive_summary=result.get(
            "executive_summary"
        ),

        report=result["report"],

        research_attempts=result.get(
            "research_attempts",
            0,
        ),
        research_history=result.get("research_history", []),
        research_coverage=result.get("research_coverage"),
        competitor_research_warnings=result.get("competitor_research_warnings", []),

        sources=result.get(
            "sources",
            [],
        ),
        comparison_sources=result.get("competitor_sources", []),

        source_stats=result.get(
            "source_stats",
            {
                "returned": 0,
                "accepted": 0,
                "discarded": 0,
            },
        ),

        research_warnings=result.get(
            "research_warnings",
            [],
        ),

        insight_items=result.get(
            "insight_items",
            [],
        ),

        source_sentiments=result.get(
            "source_sentiments",
            [],
        ),

        sentiment_distribution=result.get(
            "sentiment_distribution",
        ),

        competitors=result.get(
            "competitors",
            [],
        ),
    )


app.include_router(analysis_router)


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
    }


# --------------------------------------------------
# Dashboard Endpoint
# --------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def serve_gui():
    html_path = os.path.join(
        os.path.dirname(__file__),
        "index.html",
    )

    if os.path.exists(html_path):
        with open(
            html_path,
            "r",
            encoding="utf-8",
        ) as f:
            return f.read()

    return (
        "<h1>MarketScope API is running. "
        "GUI file not found.</h1>"
    )
