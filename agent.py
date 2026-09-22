# agent.py
import os
from dotenv import load_dotenv

# Load .env BEFORE importing anything that reads env vars
load_dotenv()

from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langfuse.langchain import CallbackHandler

# Initialize Langfuse callback handler for tracing
langfuse_handler = CallbackHandler()

# Define the web search tool
search_tool = TavilySearch(
    max_results=5,
    topic="news",
    include_answer=False,
    include_raw_content=False,
)

# Create the research agent
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_tool],
    system_prompt=(
        "You are a competitive intelligence research agent. "
        "Given a company name, search for recent news and developments. "
        "Then produce a concise 3-bullet summary of the most important competitive insights. "
        "Each bullet should be a single sentence covering a distinct topic "
        "(e.g., product launches, funding, partnerships, market moves)."
    ),
)


async def research_company(company_name: str) -> str:
    """Run the research agent for a given company name."""
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"Research recent news about {company_name}."}]},
        config={"callbacks": [langfuse_handler]},
    )
    # Extract the final assistant message
    return result["messages"][-1].content
