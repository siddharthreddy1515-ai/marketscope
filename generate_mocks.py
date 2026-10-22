# generate_mocks.py
import os
import json
import asyncio
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from companies import COMPANIES
from workflow import build_query, mock_path

load_dotenv()
search_tool = TavilySearch(max_results=5, topic="news")

async def fetch_and_save(company: str):
    print(f"Fetching live data for {company}...")
    results = await search_tool.ainvoke({"query": build_query(company)})
    
    os.makedirs("mocks", exist_ok=True)
    filename = mock_path(company)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {filename}")

async def main():
    await asyncio.gather(*(fetch_and_save(c) for c in COMPANIES))

if __name__ == "__main__":
    asyncio.run(main())