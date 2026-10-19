# test_agent.py
import os
import pytest
import asyncio
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from workflow import run_workflow, search_tool

load_dotenv()

# 1. Define the Golden Dataset (20+ Test Cases)
# In a real company, this would be a JSON/CSV file with expected outputs.
# For our live-agent, we define the companies we want to test.
COMPANIES = [
    "Tesla", "Nvidia", "Microsoft", "Apple", "Google", 
    "Amazon", "Meta", "OpenAI", "Anthropic", "Rivian", 
    "Lucid Motors", "Ford", "General Motors", "Toyota", 
    "Intel", "AMD", "Qualcomm", "Salesforce", "Adobe", "IBM"
]

# 2. Initialize Metrics (Shared across tests)
relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")
faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o")

# 3. Write the Pytest function
# pytest will automatically run this 20 times (once per company)
@pytest.mark.parametrize("company", COMPANIES)
def test_marketscope_agent(company):
    print(f"\n--- Testing {company} ---")
    
    # Generate the context and the agent output synchronously for the test
    query = f"{company} recent news, product updates, or pricing changes"
    
    # We use asyncio.run because pytest functions are sync by default
    search_results = str(asyncio.run(search_tool.ainvoke({"query": query})))
    agent_report = asyncio.run(run_workflow(company))
    
    # Create the test case
    test_case = LLMTestCase(
        input=f"Provide a competitive intelligence report for {company}",
        actual_output=agent_report,
        retrieval_context=[search_results]
    )
    
    # Assert that it passes BOTH metrics
    assert_test(test_case, [relevancy, faithfulness])