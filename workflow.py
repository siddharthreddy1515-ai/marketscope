# workflow.py
import os
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

langfuse_handler = CallbackHandler()
# Temperature is already set to 0 here
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
search_tool = TavilySearch(max_results=5, topic="news")

async def research_node(state: AgentState):
    print(f"--- RESEARCHING {state['company_name']} ---")
    query = f"{state['company_name']} recent news, product updates, or pricing changes"
    
    try:
        results = await search_tool.ainvoke({"query": query})
        search_text = str(results) 
        
        if not results:
            return {"retry_count": state.get("retry_count", 0) + 1}
            
        return {"search_results": search_text, "retry_count": 0}
    except Exception as e:
        print(f"Search failed: {e}")
        return {"retry_count": state.get("retry_count", 0) + 1}

async def analysis_node(state: AgentState):
    print("--- ANALYSING RESULTS ---")
    prompt = f"""You are a competitive intelligence analyst. Your output will be fact-checked
line by line against the source text below, so accuracy matters more than polish.

TASK
Extract the 3 most strategically important insights about {state['company_name']}
(product launches, funding, partnerships, market moves, leadership or pricing pivots).

GROUNDING RULES
1. Use ONLY information stated in the SOURCE TEXT. No outside knowledge.
2. Copy dates, years, prices, and numbers exactly as written. Never convert,
   round, or recompute them.
3. If the source gives a month but no year (e.g. "in December"), write exactly
   that: "in December". Never add a year.
4. If no date is given, give none. Do not write "recently" or "this year" either.
5. Do not infer the current year, the timing of events, or causes the text
   does not state.
6. Skip minor details (individual model numbers, small price changes) unless
   the text presents them as a strategic shift.
7. If the source supports fewer than 3 solid insights, return fewer. Never pad.

FORMAT
- One bullet per insight, 1-2 sentences each.
- Plain factual tone, no hype or speculation.

SOURCE TEXT
\"\"\"
{state['search_results']}
\"\"\"
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"insights": response.content}

async def report_node(state: AgentState):
    print("--- GENERATING REPORT ---")
    prompt = f"""Format the insights below into a short markdown report for {state['company_name']}.
Use ONLY the facts in the insights. Do not add, infer, or change any dates, numbers, or claims.

INSIGHTS
{state['insights']}
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"report": response.content}

def check_research(state: AgentState):
    if state.get("retry_count", 0) > 0 and state.get("retry_count", 0) < 3:
        print("--- RETRYING RESEARCH ---")
        return "retry"
    elif state.get("retry_count", 0) >= 3:
        print("--- MAX RETRIES REACHED ---")
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

# Changed to return a dictionary with both the report and the search results
async def run_workflow(company_name: str) -> dict:
    initial_state = {"company_name": company_name, "retry_count": 0}
    result = await app_graph.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})
    return {
        "report": result.get("report", "Failed to generate report."),
        "search_results": result.get("search_results", "")
    }