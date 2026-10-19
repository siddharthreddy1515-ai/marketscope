# main.py
import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from workflow import run_workflow

# Setup basic logging for server errors
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MarketScope Multi-Agent", version="0.2.0")

class AnalyzeRequest(BaseModel):
    company_name: str

    @field_validator('company_name')
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        """Sanitizes input to prevent prompt injection and abuse."""
        v = v.strip()
        if not v:
            raise ValueError("Company name cannot be empty.")
        if len(v) > 50:
            raise ValueError("Company name must be under 50 characters.")
        # Whitelist: Letters, numbers, spaces, hyphens, dots
        if not re.match(r"^[a-zA-Z0-9\s\-\.]+$", v):
            raise ValueError("Company name contains invalid characters.")
        return v

class AnalyzeResponse(BaseModel):
    company_name: str
    report: str

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        # Enforce a 60-second timeout on the entire multi-agent workflow
        report = await asyncio.wait_for(run_workflow(request.company_name), timeout=60.0)
        return AnalyzeResponse(company_name=request.company_name, report=report)
        
    except asyncio.TimeoutError:
        logging.error(f"Timeout generating report for {request.company_name}")
        raise HTTPException(status_code=504, detail="The request timed out. Please try again.")
        
    except Exception as e:
        # Log the full error internally (server-side)
        logging.error(f"Error generating report for {request.company_name}: {str(e)}", exc_info=True)
        # Return a safe, generic message to the client (no leaked API keys/stack traces)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

@app.get("/health")
async def health():
    return {"status": "ok"}