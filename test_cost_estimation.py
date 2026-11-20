
from cost_tracking import estimate_report_cost


def test_estimates_completed_report_cost():
    result = {
        "status": "completed",
        "research_attempts": 2,
        "competitor_research_attempts": 1,
        "llm_usage": {
            "input_tokens": 1200,
            "output_tokens": 160,
            "total_tokens": 1360,
        },
    }

    cost = estimate_report_cost(result)

    assert cost["tavily_estimated_usd"] == 0.024
    assert cost["llm_estimated_usd"] == 0.0046
    assert cost["total_estimated_usd"] == 0.0286


def test_missing_usage_means_unknown_total():
    result = {
        "status": "completed",
        "research_attempts": 1,
        "competitor_research_attempts": 1,
        "llm_usage": None,
    }

    cost = estimate_report_cost(result)

    assert cost["tavily_estimated_usd"] == 0.016
    assert cost["llm_estimated_usd"] is None
    assert cost["total_estimated_usd"] is None


def test_zero_requests_and_zero_tokens():
    result = {
        "research_attempts": 0,
        "competitor_research_attempts": 0,
        "llm_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }

    cost = estimate_report_cost(result)

    assert cost["total_estimated_usd"] == 0.0
