# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import research_company

app = FastAPI(title="MarketScope PoC", version="0.1.0")


class ResearchRequest(BaseModel):
    company_name: str


class ResearchResponse(BaseModel):
    company_name: str
    summary: str


@app.post("/api/v1/research", response_model=ResearchResponse)
async def research_endpoint(request: ResearchRequest):
    try:
        summary = await research_company(request.company_name)
        return ResearchResponse(
            company_name=request.company_name,
            summary=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
