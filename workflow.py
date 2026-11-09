
import json
import os
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END
from urllib.parse import (
    urlparse,
    urlsplit,
    urlunsplit,
    parse_qsl,
    urlencode,
)


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
    research_attempts: int
    enrichment_attempted: bool
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


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


def canonical_source_url(url: str) -> str:
    """
    Produce a comparison key by removing known
    tracking parameters.

    Does not change the original source URL.
    """
    parts = urlsplit(url.strip())

    query_params = parse_qsl(
        parts.query,
        keep_blank_values=True,
    )

    filtered_params = [
        (key, value)
        for key, value in query_params
        if key.lower() not in TRACKING_PARAMS
    ]

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(filtered_params),
        parts.fragment,
    ))

langfuse_handler = CallbackHandler()

# Use GPT-4o only for analysis; the report node is deterministic Python.
llm_analysis = ChatOpenAI(model="gpt-4o", temperature=0, seed=42)
search_tool = TavilySearch(max_results=5, topic="news")
expanded_search_tool = TavilySearch(max_results=10, topic="news")
SOURCE_TARGET = 3
MAX_RESEARCH_CALLS = 3


async def research_node(state: AgentState):
    company = state["company_name"]
    print(f"--- RESEARCHING {company} ---")

    # Reuse the sources LangGraph preserved from earlier research attempts.
    sources: list[SourceRecord] = list(state.get("sources", []))
    seen_url_keys = {
        canonical_source_url(source["url"]) for source in sources
    }

    attempt = state.get("research_attempts", 0) + 1
    # Enrichment is allowed once, only if useful evidence already exists.
    is_enrichment = bool(sources) and not state.get(
        "enrichment_attempted", False
    )
    enrichment_attempted = (
        state.get("enrichment_attempted", False) or is_enrichment
    )

    previous_stats = state.get("source_stats") or {}
    previous_returned = max(previous_stats.get("returned", 0), len(sources))
    warnings = list(state.get("research_warnings", []))

    # These fields are always returned, even when the provider fails.
    common = {
        "research_attempts": attempt,
        "enrichment_attempted": enrichment_attempted,
    }

    if os.getenv("MOCK_TAVILY") == "true":
        path = mock_path(company)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing mock fixture: {path}")
        with open(path, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        tool = expanded_search_tool if is_enrichment else search_tool
        try:
            results = await tool.ainvoke({"query": build_query(company)})
        except Exception as exc:
            print(f"Search failed: {exc}")
            if sources:
                warnings.append(
                    "Additional search failed; preserving earlier sources."
                )
                warnings.append(
                    f"Only {len(sources)} unique usable source(s) collected "
                    f"(target: {SOURCE_TARGET})."
                )
            else:
                warnings.append("Search request failed.")
            return {
                **common,
                "retry_count": (
                    0 if sources else state.get("retry_count", 0) + 1
                ),
                "sources": sources,
                "search_results": "\n\n".join(
                    source["content"] for source in sources
                ),
                "source_stats": {
                    "returned": previous_returned,
                    "accepted": len(sources),
                    "discarded": previous_returned - len(sources),
                },
                "research_warnings": warnings,
            }

    raw_items = results.get("results") if isinstance(results, dict) else None
    if not isinstance(raw_items, list):
        warnings.append("Search response has no valid results list.")
        return {
            **common,
            "retry_count": (
                0 if sources else state.get("retry_count", 0) + 1
            ),
            "sources": sources,
            "search_results": "\n\n".join(
                source["content"] for source in sources
            ),
            "source_stats": {
                "returned": previous_returned,
                "accepted": len(sources),
                "discarded": previous_returned - len(sources),
            },
            "research_warnings": warnings,
        }

    duplicate_count = 0
    new_count = 0

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

        url_key = canonical_source_url(url.strip())
        if url_key in seen_url_keys:
            duplicate_count += 1
            continue

        seen_url_keys.add(url_key)
        published_at = item.get("published_date") or item.get("published_at")
        title = item.get("title")
        sources.append({
            "title": title if isinstance(title, str) else "",
            "url": url.strip(),
            "content": content.strip(),
            "published_at": (
                published_at if isinstance(published_at, str) else None
            ),
        })
        new_count += 1

    batch_returned = len(raw_items)
    cumulative_returned = previous_returned + batch_returned
    batch_discarded = batch_returned - new_count
    unusable_count = batch_discarded - duplicate_count

    if unusable_count > 0:
        warnings.append(
            f"{unusable_count} of {batch_returned} returned source "
            "records were unusable."
        )
    if duplicate_count > 0:
        warnings.append(
            f"{duplicate_count} duplicate source URL(s) were removed."
        )
    if not sources:
        warnings.append("No usable source evidence found.")
    elif len(sources) < SOURCE_TARGET and (is_enrichment or attempt >= MAX_RESEARCH_CALLS):
        warnings.append(
            f"Only {len(sources)} unique usable source(s) collected "
            f"(target: {SOURCE_TARGET})."
        )

    return {
        **common,
        "retry_count": (
            0 if sources else state.get("retry_count", 0) + 1
        ),
        "search_results": "\n\n".join(
            source["content"] for source in sources
        ),
        "sources": sources,
        "source_stats": {
            "returned": cumulative_returned,
            "accepted": len(sources),
            "discarded": cumulative_returned - len(sources),
        },
        "research_warnings": warnings,
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
    sources = state.get("sources", [])
    attempts = state.get("research_attempts", state.get("retry_count", 0))

    if len(sources) >= SOURCE_TARGET:
        return "continue"
    if attempts >= MAX_RESEARCH_CALLS:
        return "continue" if sources else "fail"
    if sources:
        # Do not make a second enrichment request, even below target.
        return "continue" if state.get("enrichment_attempted", False) else "retry"
    # Without any accepted evidence, retry until the global call cap.
    return "retry"


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
    initial_state = {
        "company_name": company_name,
        "retry_count": 0,
        "research_attempts": 0,
        "enrichment_attempted": False,
        "sources": [],
        "source_stats": {"returned": 0, "accepted": 0, "discarded": 0},
        "research_warnings": [],
    }

    result = await app_graph.ainvoke(
        initial_state,
        config={"callbacks": [langfuse_handler]},
    )

    attempts = result.get("research_attempts", 0)

    # Research exhausted its call budget without usable sources.
    if not result.get("sources"):
        return {
            "status": "failed",
            "error_code": "RESEARCH_EXHAUSTED",
            "research_attempts": attempts,
            "report": None,
            "search_results": "",
            "insights": "",
            "sources": [],
            "source_stats": result.get("source_stats", {}),
            "research_warnings": result.get(
                "research_warnings", []
            ),
        }

    # A successful workflow must produce a report.
    report = result.get("report")

    if not report:
        raise RuntimeError(
            "Workflow ended without generating a report."
        )

    return {
        "status": "completed",
        "error_code": None,
        "research_attempts": attempts,
        "report": report,
        "search_results": result.get("search_results", ""),
        "insights": result.get("insights", ""),
        "sources": result.get("sources", []),
        "source_stats": result.get("source_stats", {}),
        "research_warnings": result.get(
            "research_warnings", []
        ),
    }
