# build_baseline.py
import os
import asyncio
import pandas as pd
from companies import COMPANIES
from workflow import run_workflow

async def main():
    os.environ["MOCK_TAVILY"] = "true"
    results = []
    
    print("Building baseline from mocks...")
    for company in COMPANIES:
        print(f"Generating baseline for {company}...")
        result = await run_workflow(company)
        results.append({
            "company": company,
            "report_text": result["report"]
        })
        
    df = pd.DataFrame(results)
    df.to_csv("baseline_reports.csv", index=False)
    print(f"✅ Saved {len(df)} baseline reports to baseline_reports.csv")

if __name__ == "__main__":
    asyncio.run(main())