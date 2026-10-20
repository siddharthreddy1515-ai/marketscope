# test_agent.py
import os
import asyncio
import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from workflow import run_workflow

load_dotenv()

COMPANIES = [
    "Tesla", "Nvidia", "Microsoft", "Apple", "Google", 
    "Amazon", "Meta", "OpenAI", "Anthropic", "Rivian", 
    "Lucid Motors", "Ford", "General Motors", "Toyota", 
    "Intel", "AMD", "Qualcomm", "Salesforce", "Adobe", "IBM"
]

async def fetch_company_data(company: str):
    print(f"--- Generating report for {company} ---")
    # We no longer call search_tool here. run_workflow returns both.
    result = await run_workflow(company)
    return company, (str(result["search_results"]), result["report"])

async def generate_all_reports():
    results = await asyncio.gather(*(fetch_company_data(c) for c in COMPANIES))
    return dict(results)

@pytest.fixture(scope="session")
def agent_data():
    print("\nStarting concurrent agent generation for all companies...")
    return asyncio.run(generate_all_reports())

relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")
faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o")

@pytest.mark.parametrize("company", COMPANIES)
def test_marketscope_agent(agent_data, company):
    search_results, agent_report = agent_data[company]
    
    test_case = LLMTestCase(
        input=f"Provide a competitive intelligence report for {company}",
        actual_output=agent_report,
        retrieval_context=[search_results]
    )
    
    assert_test(test_case, [relevancy, faithfulness])