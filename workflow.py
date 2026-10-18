# workflow.py
import os
from typing import TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END

# 1. DEFINE THE STATE
# This is the shared memory passed between agents.
class AgentState(TypedDict):
    company_name: str
    search_results: str
    insights: str
    report: str
    retry_count: int

# Initialize shared tools and models
langfuse_handler = CallbackHandler()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearch(max_results=5, topic="news")

# 2. DEFINE THE NODES (The Agents)

def research_node(state: AgentState):
    """Fetches recent news using Tavily."""
    print(f"--- RESEARCHING {state['company_name']} ---")
    query = f"{state['company_name']} recent news, product updates, or pricing changes"
    
    try:
        results = search_tool.invoke({"query": query})
        # Convert the raw results to a string for the next agent
        search_text = str(results) 
        
        # Basic retry logic: if results are empty, trigger a retry
        if not results:
            return {"retry_count": state.get("retry_count", 0) + 1}
            
        return {"search_results": search_text, "retry_count": 0}
    except Exception as e:
        print(f"Search failed: {e}")
        return {"retry_count": state.get("retry_count", 0) + 1}

def analysis_node(state: AgentState):
    """Synthesises raw search results into structured insights."""
    print("--- ANALYSING RESULTS ---")
    prompt = f"""
    You are a competitive intelligence analyst. 
    Based on the following raw search results about {state['company_name']}, 
    extract the 3 most important competitive insights (e.g., product launches, funding, market moves).
    Keep each insight to 1-2 sentences.
    
    Raw Results:
    {state['search_results']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"insights": response.content}

def report_node(state: AgentState):
    """Formats insights into a clean Markdown output."""
    print("--- GENERATING REPORT ---")
    prompt = f"""
    You are a professional report writer. 
    Format the following insights into a clean, professional Markdown report.
    Include a main header with the company name ({state['company_name']}).
    Use bullet points for the insights.
    
    Insights:
    {state['insights']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"report": response.content}

# 3. DEFINE THE CONDITIONAL EDGE (Routing / Retry logic)
def check_research(state: AgentState):
    """Routes to 'analysis' if search succeeded, or 'research' if it needs a retry."""
    if state.get("retry_count", 0) > 0 and state.get("retry_count", 0) < 3:
        print("--- RETRYING RESEARCH ---")
        return "retry"
    elif state.get("retry_count", 0) >= 3:
        print("--- MAX RETRIES REACHED ---")
        return "fail" # In a real app, route to an error handling node
    else:
        return "continue"

# 4. BUILD THE LANGGRAPH STATE MACHINE
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("research", research_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("report", report_node)

# Add edges
workflow.add_edge(START, "research")

# Conditional edge after research (handles retries)
workflow.add_conditional_edges(
    "research",
    check_research,
    {
        "continue": "analysis",
        "retry": "research",
        "fail": END # For the PoC, we just end if it fails 3 times
    }
)

# Standard edges for the rest of the flow
workflow.add_edge("analysis", "report")
workflow.add_edge("report", END)

# Compile the graph
app_graph = workflow.compile()

# Helper function for FastAPI to call
async def run_workflow(company_name: str) -> str:
    """Executes the LangGraph workflow."""
    initial_state = {"company_name": company_name, "retry_count": 0}
    
    # Run the graph with Langfuse tracing
    result = await app_graph.ainvoke(
        initial_state, 
        config={"callbacks": [langfuse_handler]}
    )
    
    return result.get("report", "Failed to generate report.")