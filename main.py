# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflow import run_workflow  # <-- Changed import

app = FastAPI(title="MarketScope Multi-Agent", version="0.2.0")

class AnalyzeRequest(BaseModel):
    company_name: str

class AnalyzeResponse(BaseModel):
    company_name: str
    report: str

# New endpoint specified in Phase 2: /api/v1/analyze
@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    try:
        report = await run_workflow(request.company_name)
        return AnalyzeResponse(
            company_name=request.company_name,
            report=report,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}