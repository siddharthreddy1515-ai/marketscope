
import json
import os
from typing import TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END


class SourceRecord(TypedDict):
    title: str
    url: str
    content: str
    published_at: str | None


class ResearchStats(TypedDict):
    returned: int
    accepted: int
    discarded: int


class AgentState(TypedDict):
    company_name: str
    search_results: str
    insights: str
    report: str
    retry_count: int
    sources: list[SourceRecord]
    source_stats: ResearchStats
    research_warnings: list[str]


# --- Single source of truth for research configuration ---
def build_query(company: str) -> str:
    return (
        f"{company} strategy partnerships acquisitions funding "
        "product launches market share"
    )


def mock_path(company: str) -> str:
    return f"mocks/{company.replace(' ', '_')}.json"


langfuse_handler = CallbackHandler()

# Use GPT-4o only for analysis; the report node is deterministic Python.
llm_analysis = ChatOpenAI(model="gpt-4o", temperature=0, seed=42)
search_tool = TavilySearch(max_results=5, topic="news")


async def research_node(state: AgentState):
    company = state["company_name"]
    print(f"--- RESEARCHING {company} ---")

    if os.getenv("MOCK_TAVILY") == "true":
        path = mock_path(company)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing mock fixture: {path}")

        with open(path, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        try:
            results = await search_tool.ainvoke(
                {"query": build_query(company)}
            )
        except Exception as exc:
            print(f"Search failed: {exc}")
            # Existing graph routing will retry on a positive retry_count.
            return {"retry_count": state.get("retry_count", 0) + 1}

    # We require an article list, not just a provider-generated answer.
    raw_items = results.get("results") if isinstance(results, dict) else None

    if not isinstance(raw_items, list):
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "search_results": "",
            "sources": [],
            "source_stats": {
                "returned": 0,
                "accepted": 0,
                "discarded": 0,
            },
            "research_warnings": [
                "Search response has no valid results list."
            ],
        }

    sources: list[SourceRecord] = []

    # Validate each returned item independently. Keep valid ones even
    # when the batch also contains invalid items.
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        url = item.get("url")
        content = item.get("content")

        if not isinstance(url, str) or not isinstance(content, str):
            continue

        try:
            parsed = urlparse(url.strip())
        except ValueError:
            continue

        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or not content.strip()
        ):
            continue

        published_at = item.get("published_date") or item.get("published_at")
        title = item.get("title")

        sources.append(
            {
                "title": title if isinstance(title, str) else "",
                "url": url.strip(),
                "content": content.strip(),
                "published_at": (
                    published_at if isinstance(published_at, str) else None
                ),
            }
        )

    total = len(raw_items)
    accepted = len(sources)
    discarded = total - accepted

    stats: ResearchStats = {
        "returned": total,
        "accepted": accepted,
        "discarded": discarded,
    }

    warnings: list[str] = []
    if discarded > 0:
        warnings.append(
            f"{discarded} of {total} returned source records "
            "were discarded as unusable."
        )

    # A search with no usable source must not be counted as success.
    if not sources:
        warnings.append("No usable source evidence found.")
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "search_results": "",
            "sources": [],
            "source_stats": stats,
            "research_warnings": warnings,
        }

    # Preserve the existing text interface for the analysis node.
    search_text = "\n\n".join(source["content"] for source in sources)

    return {
        "search_results": search_text,
        "sources": sources,
        "source_stats": stats,
        "research_warnings": warnings,
        "retry_count": 0,
    }


async def analysis_node(state: AgentState):
    print("--- ANALYSING RESULTS ---")
    prompt = f"""You are a competitive intelligence analyst. Your output will be fact-checked
line by line against the source text below, so accuracy matters more than polish.

TASK
Extract the 3 most strategically important insights about {state['company_name']}.

GROUNDING RULES
1. Use ONLY information stated in the SOURCE TEXT. No outside knowledge.
2. Copy dates, years, prices, and numbers exactly as written. Never convert, round, or recompute them.
3. If the source gives a month but no year, write exactly that. Never add a year.
4. If no date is given, give none. Do not write "recently" or "this year".
5. Do not infer the current year, the timing of events, or causes the text does not state.
6. If the source supports fewer than 3 solid insights, return fewer. Never pad.
7. ZERO TOLERANCE: If a specific dollar amount, funding round, loan, or financial figure is not explicitly written in the SOURCE TEXT, you are forbidden from including it.

RELEVANCE RULES
8. Keep only insights that change how a competitor should view {state['company_name']}: strategic direction, partnerships, acquisitions, funding, major launches, market-share moves.
9. Exclude routine pricing, individual product SKUs, API/developer release notes, admin/tooling updates, routine financial line items (e.g., other income, unrealized gains/losses, interest, taxes), and legal settlements unless they represent a strategic shift.
10. Do not characterize or editorialize beyond the source wording (e.g., do not add why something happened or who it affects if the text does not say so).
11. State each fact once. Do not repeat the same item in different words.
12. FOCUS ON THE TARGET: The subject of every bullet point MUST be {state['company_name']}. If the source text discusses a third-party company, you may mention them ONLY as context for what {state['company_name']} is doing.

FORMAT
- One bullet per insight, 1-2 sentences each.
- Plain factual tone, no hype or speculation.

SOURCE TEXT
\"\"\"
{state['search_results']}
\"\"\"
"""
    response = await llm_analysis.ainvoke([HumanMessage(content=prompt)])
    return {"insights": response.content}


async def report_node(state: AgentState):
    print("--- GENERATING REPORT ---")
    # Deterministic formatting: no additional LLM call.
    # This does not verify the factual accuracy of upstream AI insights.
    return {
        "report": (
            f"# {state['company_name']} Insights Report\n\n{state['insights']}"
        )
    }


def check_research(state: AgentState):
    retry_count = state.get("retry_count", 0)
    if 0 < retry_count < 3:
        return "retry"
    if retry_count >= 3:
        return "fail"
    return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("report", report_node)

workflow.add_edge(START, "research")
workflow.add_conditional_edges(
    "research",
    check_research,
    {"continue": "analysis", "retry": "research", "fail": END},
)
workflow.add_edge("analysis", "report")
workflow.add_edge("report", END)

app_graph = workflow.compile()


async def run_workflow(company_name: str) -> dict:
    initial_state = {"company_name": company_name, "retry_count": 0}
    result = await app_graph.ainvoke(
        initial_state,
        config={"callbacks": [langfuse_handler]},
    )
    return {
        "report": result.get("report", "Failed to generate report."),
        "search_results": result.get("search_results", ""),
        "insights": result.get("insights", ""),  # Used by existing tests.
    }
