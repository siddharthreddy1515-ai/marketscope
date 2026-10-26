# main.py
import asyncio
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from workflow import run_workflow
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="MarketScope Multi-Agent", version="0.2.0")

class AnalyzeRequest(BaseModel):
    company_name: str

    @field_validator('company_name')
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Company name cannot be empty.")
        if len(v) > 50:
            raise ValueError("Company name must be under 50 characters.")
        if not re.match(r"^[a-zA-Z0-9\s\-\.]+$", v):
            raise ValueError("Company name contains invalid characters.")
        return v

class AnalyzeResponse(BaseModel):
    company_name: str
    report: str

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        # run_workflow now returns a dict
        result = await asyncio.wait_for(run_workflow(request.company_name), timeout=60.0)
        
        # Extract the report string
        return AnalyzeResponse(
            company_name=request.company_name, 
            report=result["report"]
        )
        
    except asyncio.TimeoutError:
        logging.error(f"Timeout generating report for {request.company_name}")
        raise HTTPException(status_code=504, detail="The request timed out. Please try again.")
    except Exception as e:
        logging.error(f"Error generating report for {request.company_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

@app.get("/health")
async def health():
    return {"status": "ok"}

# Add this at the bottom of main.py
@app.get("/", response_class=HTMLResponse)
async def serve_gui():
    # We will put the HTML in the same folder for simplicity
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>MarketScope API is running. GUI file not found.</h1>"