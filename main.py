
import asyncio
import logging
import re
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator

from workflow import run_workflow

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="MarketScope Multi-Agent",
    version="0.2.0",
)


class AnalyzeRequest(BaseModel):
    company_name: str

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v: str) -> str:
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


class SourceResponse(BaseModel):
    title: str
    url: str
    content: str
    published_at: str | None = None


class InsightSourceResponse(BaseModel):
    title: str
    url: str


class InsightItemResponse(BaseModel):
    text: str
    sources: list[InsightSourceResponse]
    category: str = "Other"

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        allowed = {
            "Product & Innovation",
            "M&A / Partnerships",
            "Market Expansion",
            "Hiring & Talent",
            "Other",
        }

        return (
            value
            if value in allowed
            else "Other"
        )


class ResearchStatsResponse(BaseModel):
    returned: int
    accepted: int
    discarded: int


class AnalyzeResponse(BaseModel):
    company_name: str
    status: str
    report: str
    research_attempts: int
    sources: list[SourceResponse]
    source_stats: ResearchStatsResponse
    research_warnings: list[str]
    insight_items: list[InsightItemResponse]


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
)
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        result = await asyncio.wait_for(
            run_workflow(request.company_name),
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

    return AnalyzeResponse(
        company_name=request.company_name,
        status=result["status"],
        report=result["report"],
        research_attempts=result.get(
            "research_attempts",
            0,
        ),
        sources=result.get(
            "sources",
            [],
        ),
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
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
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
