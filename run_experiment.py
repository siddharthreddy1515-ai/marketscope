# run_experiment.py
import os
import time
import asyncio
import mlflow
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from workflow import run_workflow

# Tell the script to write to the exact same database the server is reading from
mlflow.set_tracking_uri("sqlite:///mlflow.db")

load_dotenv()

# We will test just a few companies to save cost while we verify MLflow works
COMPANIES = ["Tesla", "Microsoft", "Meta"]

async def evaluate_and_track(company: str):
    print(f"\n--- Running Experiment for {company} ---")
    
    # 1. Run the Agent and measure latency
    start_time = time.time()
    result = await run_workflow(company)
    latency = time.time() - start_time
    
    report = result["report"]
    search_results = str(result["search_results"])
    
    # 2. Run the DeepEval Judge
    test_case = LLMTestCase(
        input=f"Provide a competitive intelligence report for {company}",
        actual_output=report,
        retrieval_context=[search_results]
    )
    
    relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")
    faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o")
    
    await asyncio.gather(relevancy.a_measure(test_case), faithfulness.a_measure(test_case))
    
    # 3. Log everything to MLflow
    with mlflow.start_run(run_name=f"{company}_eval"):
        # Log Parameters
        mlflow.log_param("company", company)
        mlflow.log_param("agent_model", "gpt-4o")
        mlflow.log_param("judge_model", "gpt-4o")
        
        # Log Metrics
        mlflow.log_metric("latency_seconds", round(latency, 2))
        mlflow.log_metric("relevancy_score", relevancy.score)
        mlflow.log_metric("faithfulness_score", faithfulness.score)
        
        # Log Artifacts (The actual report)
        with open("temp_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact("temp_report.md")
        os.remove("temp_report.md") # Clean up
        
    print(f"✅ Logged {company} to MLflow.")

async def main():
    # Set the MLflow experiment name
    mlflow.set_experiment("MarketScope_Evaluation")
    
    for company in COMPANIES:
        await evaluate_and_track(company)

if __name__ == "__main__":
    asyncio.run(main())