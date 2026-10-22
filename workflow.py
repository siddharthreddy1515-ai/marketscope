# workflow.py
import os
import json
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    company_name: str
    search_results: str
    insights: str
    report: str
    retry_count: int

# --- Single Source of Truth ---
def build_query(company: str) -> str:
    return f"{company} strategy partnerships acquisitions funding product launches market share"

def mock_path(company: str) -> str:
    return f"mocks/{company.replace(' ', '_')}.json"
# ------------------------------

langfuse_handler = CallbackHandler()

# Seed added for slight determinism. llm_report is deleted.
llm_analysis = ChatOpenAI(model="gpt-4o", temperature=0, seed=42)
search_tool = TavilySearch(max_results=5, topic="news")

async def research_node(state: AgentState):
    company = state["company_name"]
    print(f"--- RESEARCHING {company} ---")
    
    if os.getenv("MOCK_TAVILY") == "true":
        path = mock_path(company)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing mock fixture: {path}")
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        try:
            results = await search_tool.ainvoke({"query": build_query(company)})
        except Exception as e:
            print(f"Search failed: {e}")
            return {"retry_count": state.get("retry_count", 0) + 1}
            
    if not results:
        return {"retry_count": state.get("retry_count", 0) + 1}
        
    clean_text_parts = []
    if isinstance(results, dict):
        if results.get("answer"):
            clean_text_parts.append(results["answer"])
        if results.get("results"):
            for item in results["results"]:
                clean_text_parts.append(item.get("content", ""))
    else:
        clean_text_parts.append(str(results))
        
    search_text = "\n\n".join(filter(None, clean_text_parts))
    return {"search_results": search_text, "retry_count": 0}

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
    # PURE CODE FORMATTING. No LLM. No extra cost. No hallucination risk.
    return {"report": f"# {state['company_name']} Insights Report\n\n{state['insights']}"}

def check_research(state: AgentState):
    if state.get("retry_count", 0) > 0 and state.get("retry_count", 0) < 3:
        return "retry"
    elif state.get("retry_count", 0) >= 3:
        return "fail" 
    else:
        return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("report", report_node)

workflow.add_edge(START, "research")
workflow.add_conditional_edges("research", check_research, {"continue": "analysis", "retry": "research", "fail": END})
workflow.add_edge("analysis", "report")
workflow.add_edge("report", END)

app_graph = workflow.compile()

async def run_workflow(company_name: str) -> dict:
    initial_state = {"company_name": company_name, "retry_count": 0}
    result = await app_graph.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})
    return {
        "report": result.get("report", "Failed to generate report."),
        "search_results": result.get("search_results", ""),
        "insights": result.get("insights", "") # ADDED FOR TEST PRINTING
    }