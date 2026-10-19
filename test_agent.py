# test_agent.py
import os
import asyncio
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from workflow import run_workflow, search_tool

load_dotenv()

async def main():
    print("Generating report for evaluation...")
    company = "Tesla"
    
    # 1. Capture the raw search context
    query = f"{company} recent news, product updates, or pricing changes"
    search_results = str(search_tool.invoke({"query": query}))
    
    # 2. Run the multi-agent workflow
    agent_report = await run_workflow(company)
    
    # 3. Package it into a DeepEval Test Case
    test_case = LLMTestCase(
        input=f"Provide a competitive intelligence report for {company}",
        actual_output=agent_report,
        retrieval_context=[search_results]
    )

    # 4. Define the Metrics (Threshold 0.7 = 70%)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")
    faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o")

    # 5. Run the evaluation
    print("Grading the report with DeepEval (LLM-as-a-Judge)...")
    evaluate(
        test_cases=[test_case],
        metrics=[relevancy, faithfulness]
    )

if __name__ == "__main__":
    asyncio.run(main())