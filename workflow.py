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

# 1. DEFINE THE STATE
class AgentState(TypedDict):
    company_name: str
    search_results: str
    insights: str
    report: str
    retry_count: int

langfuse_handler = CallbackHandler()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearch(max_results=5, topic="news")

# 2. DEFINE THE NODES (Now fully Async)
# We changed `def` to `async def` and `.invoke()` to `.ainvoke()`

async def research_node(state: AgentState):
    print(f"--- RESEARCHING {state['company_name']} ---")
    query = f"{state['company_name']} recent news, product updates, or pricing changes"
    
    try:
        # Using ainvoke for non-blocking execution
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
    prompt = f"""
    You are a competitive intelligence analyst. 
    Based on the following raw search results about {state['company_name']}, 
    extract the 3 most important STRATEGIC competitive insights (e.g., product launches, funding, market moves, partnerships).
    Do NOT include minor product pricing or specific model numbers unless it is a major strategic pivot.
    Be strictly accurate with dates and years found in the text; do not hallucinate years.
    Keep each insight to 1-2 sentences.
    
    Raw Results:
    {state['search_results']}
    """
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"insights": response.content}

async def report_node(state: AgentState):
    print("--- GENERATING REPORT ---")
    prompt = f"""
    You are a professional report writer. 
    Format the following insights into a clean, professional Markdown report.
    Include a main header with the company name ({state['company_name']}).
    Use bullet points for the insights.
    
    Insights:
    {state['insights']}
    """
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"report": response.content}

# 3. ROUTING
def check_research(state: AgentState):
    if state.get("retry_count", 0) > 0 and state.get("retry_count", 0) < 3:
        print("--- RETRYING RESEARCH ---")
        return "retry"
    elif state.get("retry_count", 0) >= 3:
        print("--- MAX RETRIES REACHED ---")
        return "fail" 
    else:
        return "continue"

# 4. BUILD GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("report", report_node)

workflow.add_edge(START, "research")
workflow.add_conditional_edges("research", check_research, {"continue": "analysis", "retry": "research", "fail": END})
workflow.add_edge("analysis", "report")
workflow.add_edge("report", END)

app_graph = workflow.compile()

# Helper function
async def run_workflow(company_name: str) -> str:
    initial_state = {"company_name": company_name, "retry_count": 0}
    result = await app_graph.ainvoke(initial_state, config={"callbacks": [langfuse_handler]})
    return result.get("report", "Failed to generate report.")