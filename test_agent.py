# test_agent.py
import os
import asyncio
import pytest
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from companies import COMPANIES as ALL_COMPANIES
from workflow import run_workflow

load_dotenv()

# Only run all 20 if FULL_EVAL is true. Otherwise, run the "Danger 6".
if os.getenv("FULL_EVAL") == "true":
    COMPANIES = ALL_COMPANIES
else:
    COMPANIES = ["Tesla", "Microsoft", "Google", "Meta", "Rivian", "Intel"]

THRESHOLD = 0.7
MIN_PASS_RATE = 0.80   # Allows 1 flake out of 6
MIN_MEAN = 0.85        # Overall average must be high
JUDGE = "gpt-4o"

async def score_company(company, sem):
    async with sem:
        out = {"company": company, "error": None}
        try:
            r = await run_workflow(company)
            tc = LLMTestCase(
                input=f"Provide a competitive intelligence report for {company}",
                actual_output=r["report"],
                retrieval_context=[str(r["search_results"])],
            )
            rel = AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE)
            fai = FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE)
            
            # Run judge concurrently
            await asyncio.gather(rel.a_measure(tc), fai.a_measure(tc))
            
            out.update(
                relevancy=rel.score, 
                faithfulness=fai.score,
                reasons={"relevancy": rel.reason, "faithfulness": fai.reason},
                insights=r.get("insights", "")
            )
        except Exception as e:
            out["error"] = repr(e)[:200]
        return out

@pytest.fixture(scope="session")
def scored():
    async def main():
        sem = asyncio.Semaphore(5) # Limit concurrency
        return await asyncio.gather(*(score_company(c, sem) for c in COMPANIES))
    return asyncio.run(main())

def test_no_infra_errors(scored):
    errs = [(r["company"], r["error"]) for r in scored if r["error"]]
    assert not errs, f"Infra/judge errors, not quality failures: {errs}"

@pytest.mark.parametrize("metric", ["relevancy", "faithfulness"])
def test_suite_quality(scored, metric):
    ok = [r for r in scored if not r["error"]]
    scores = [r[metric] for r in ok]
    mean = sum(scores) / len(scores)
    rate = sum(s >= THRESHOLD for s in scores) / len(scores)
    
    # Print details for any failures so we can debug them
    for r in ok:
        if r[metric] < THRESHOLD:
            print(f"\n[{metric}] {r['company']} = {r[metric]:.2f}\n{r['reasons'][metric]}\nINSIGHTS:\n{r['insights']}")
            
    print(f"\n{metric}: mean={mean:.2f} pass_rate={rate:.0%}")
    assert rate >= MIN_PASS_RATE and mean >= MIN_MEAN