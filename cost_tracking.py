
from decimal import Decimal, ROUND_HALF_UP


# Pricing assumptions in USD.
# Recheck provider pricing before production use.
# GPT-4o standard, uncached API pricing.
GPT4O_INPUT_PER_MILLION = Decimal("2.50")
GPT4O_OUTPUT_PER_MILLION = Decimal("10.00")

# Tavily pay-as-you-go, one basic-search credit
# per request.
TAVILY_COST_PER_CREDIT = Decimal("0.008")

ONE_MILLION = Decimal("1000000")
PRECISION = Decimal("0.000001")


def _valid_count(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _usd(value):
    return float(
        value.quantize(
            PRECISION,
            rounding=ROUND_HALF_UP,
        )
    )


def estimate_report_cost(result: dict) -> dict:
    """
    Estimate USD cost from existing workflow telemetry.

    Assumptions:
    - GPT-4o standard, uncached token rates
    - Tavily basic search, one credit per request
    - Tavily pay-as-you-go credit pricing

    This is an estimate, not provider billing data.
    """

    company_calls = result.get("research_attempts")
    competitor_calls = result.get(
        "competitor_research_attempts"
    )

    tavily_estimated_usd = None

    if (
        _valid_count(company_calls)
        and _valid_count(competitor_calls)
    ):
        total_calls = company_calls + competitor_calls

        tavily_estimated_usd = (
            Decimal(total_calls)
            * TAVILY_COST_PER_CREDIT
        )

    usage = result.get("llm_usage")

    llm_estimated_usd = None

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        if (
            _valid_count(input_tokens)
            and _valid_count(output_tokens)
        ):
            input_cost = (
                Decimal(input_tokens)
                * GPT4O_INPUT_PER_MILLION
                / ONE_MILLION
            )

            output_cost = (
                Decimal(output_tokens)
                * GPT4O_OUTPUT_PER_MILLION
                / ONE_MILLION
            )

            llm_estimated_usd = (
                input_cost + output_cost
            )

    total_estimated_usd = None

    if (
        tavily_estimated_usd is not None
        and llm_estimated_usd is not None
    ):
        total_estimated_usd = (
            tavily_estimated_usd
            + llm_estimated_usd
        )

    return {
        "tavily_estimated_usd": (
            _usd(tavily_estimated_usd)
            if tavily_estimated_usd is not None
            else None
        ),
        "llm_estimated_usd": (
            _usd(llm_estimated_usd)
            if llm_estimated_usd is not None
            else None
        ),
        "total_estimated_usd": (
            _usd(total_estimated_usd)
            if total_estimated_usd is not None
            else None
        ),
    }
