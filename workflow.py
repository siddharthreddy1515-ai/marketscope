
import json
import os
import re
from typing import TypedDict
from urllib.parse import (
    urlparse,
    urlsplit,
    urlunsplit,
    parse_qsl,
    urlencode,
)

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END

load_dotenv()


class SourceRecord(TypedDict):
    title: str
    url: str
    content: str
    published_at: str | None


class InsightSource(TypedDict):
    title: str
    url: str


class InsightItem(TypedDict):
    text: str
    category: str
    sources: list[InsightSource]


class ExecutiveSummary(TypedDict):
    text: str
    sources: list[InsightSource]


class SourceSentiment(TypedDict):
    source_id: str
    label: str
    title: str
    url: str


class SentimentDistribution(TypedDict):
    positive: int
    neutral: int
    negative: int
    assessed: int
    unassessed: int
    total_sources: int


class ResearchStats(TypedDict):
    returned: int
    accepted: int
    discarded: int


class AgentState(TypedDict):
    company_name: str
    search_results: str
    insights: str
    insight_items: list[InsightItem]
    report: str
    retry_count: int
    research_attempts: int
    enrichment_attempted: bool
    sources: list[SourceRecord]
    source_stats: ResearchStats
    research_warnings: list[str]
    executive_summary: ExecutiveSummary | None
    source_sentiments: list[SourceSentiment]
    sentiment_distribution: SentimentDistribution


def build_query(company: str) -> str:
    return (
        f"{company} strategy partnerships acquisitions funding "
        "product launches market share"
    )


def mock_path(company: str) -> str:
    return f"mocks/{company.replace(' ', '_')}.json"


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


def canonical_source_url(url: str) -> str:
    """
    Produce a comparison key by removing known
    tracking parameters.

    Does not change the original source URL.
    """
    parts = urlsplit(url.strip())

    query_params = parse_qsl(
        parts.query,
        keep_blank_values=True,
    )

    filtered_params = [
        (key, value)
        for key, value in query_params
        if key.lower() not in TRACKING_PARAMS
    ]

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(filtered_params),
        parts.fragment,
    ))


langfuse_handler = CallbackHandler()

# GPT-4o is used only for analysis.
# Report formatting remains deterministic Python.
llm_analysis = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    seed=42,
)

search_tool = TavilySearch(
    max_results=5,
    topic="news",
)

expanded_search_tool = TavilySearch(
    max_results=10,
    topic="news",
)

SOURCE_TARGET = 3
MAX_RESEARCH_CALLS = 3

SIGNAL_CATEGORIES = (
    "Product & Innovation",
    "M&A / Partnerships",
    "Market Expansion",
    "Hiring & Talent",
    "Other",
)


async def research_node(state: AgentState):
    company = state["company_name"]
    print(f"--- RESEARCHING {company} ---")

    # Preserve sources from previous research attempts.
    sources: list[SourceRecord] = list(
        state.get("sources", [])
    )

    seen_url_keys = {
        canonical_source_url(source["url"])
        for source in sources
    }

    attempt = state.get("research_attempts", 0) + 1

    # Only one enrichment search is allowed.
    is_enrichment = (
        bool(sources)
        and not state.get("enrichment_attempted", False)
    )

    enrichment_attempted = (
        state.get("enrichment_attempted", False)
        or is_enrichment
    )

    previous_stats = state.get("source_stats") or {}

    previous_returned = max(
        previous_stats.get("returned", 0),
        len(sources),
    )

    warnings = list(
        state.get("research_warnings", [])
    )

    common = {
        "research_attempts": attempt,
        "enrichment_attempted": enrichment_attempted,
    }

    if os.getenv("MOCK_TAVILY") == "true":
        path = mock_path(company)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing mock fixture: {path}"
            )

        with open(path, "r", encoding="utf-8") as file:
            results = json.load(file)

    else:
        tool = (
            expanded_search_tool
            if is_enrichment
            else search_tool
        )

        try:
            results = await tool.ainvoke({
                "query": build_query(company)
            })

        except Exception as exc:
            print(f"Search failed: {exc}")

            if sources:
                warnings.append(
                    "Additional search failed; "
                    "preserving earlier sources."
                )

                warnings.append(
                    f"Only {len(sources)} unique usable "
                    f"source(s) collected "
                    f"(target: {SOURCE_TARGET})."
                )
            else:
                warnings.append(
                    "Search request failed."
                )

            return {
                **common,
                "retry_count": (
                    0
                    if sources
                    else state.get("retry_count", 0) + 1
                ),
                "sources": sources,
                "search_results": "\n\n".join(
                    source["content"]
                    for source in sources
                ),
                "source_stats": {
                    "returned": previous_returned,
                    "accepted": len(sources),
                    "discarded": (
                        previous_returned - len(sources)
                    ),
                },
                "research_warnings": warnings,
            }

    raw_items = (
        results.get("results")
        if isinstance(results, dict)
        else None
    )

    if not isinstance(raw_items, list):
        warnings.append(
            "Search response has no valid results list."
        )

        return {
            **common,
            "retry_count": (
                0
                if sources
                else state.get("retry_count", 0) + 1
            ),
            "sources": sources,
            "search_results": "\n\n".join(
                source["content"]
                for source in sources
            ),
            "source_stats": {
                "returned": previous_returned,
                "accepted": len(sources),
                "discarded": (
                    previous_returned - len(sources)
                ),
            },
            "research_warnings": warnings,
        }

    duplicate_count = 0
    new_count = 0

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        url = item.get("url")
        content = item.get("content")

        if (
            not isinstance(url, str)
            or not isinstance(content, str)
        ):
            continue

        try:
            parsed = urlparse(url.strip())
        except ValueError:
            continue

        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or not content.strip()
        ):
            continue

        url_key = canonical_source_url(
            url.strip()
        )

        if url_key in seen_url_keys:
            duplicate_count += 1
            continue

        seen_url_keys.add(url_key)

        published_at = (
            item.get("published_date")
            or item.get("published_at")
        )

        title = item.get("title")

        sources.append({
            "title": (
                title
                if isinstance(title, str)
                else ""
            ),
            "url": url.strip(),
            "content": content.strip(),
            "published_at": (
                published_at
                if isinstance(published_at, str)
                else None
            ),
        })

        new_count += 1

    batch_returned = len(raw_items)

    cumulative_returned = (
        previous_returned + batch_returned
    )

    batch_discarded = (
        batch_returned - new_count
    )

    unusable_count = (
        batch_discarded - duplicate_count
    )

    if unusable_count > 0:
        warnings.append(
            f"{unusable_count} of {batch_returned} "
            "returned source records were unusable."
        )

    if duplicate_count > 0:
        warnings.append(
            f"{duplicate_count} duplicate source "
            "URL(s) were removed."
        )

    if not sources:
        warnings.append(
            "No usable source evidence found."
        )

    elif (
        len(sources) < SOURCE_TARGET
        and (
            is_enrichment
            or attempt >= MAX_RESEARCH_CALLS
        )
    ):
        warnings.append(
            f"Only {len(sources)} unique usable "
            f"source(s) collected "
            f"(target: {SOURCE_TARGET})."
        )

    return {
        **common,
        "retry_count": (
            0
            if sources
            else state.get("retry_count", 0) + 1
        ),
        "search_results": "\n\n".join(
            source["content"]
            for source in sources
        ),
        "sources": sources,
        "source_stats": {
            "returned": cumulative_returned,
            "accepted": len(sources),
            "discarded": (
                cumulative_returned - len(sources)
            ),
        },
        "research_warnings": warnings,
    }


def parse_cited_insights(
    text: str,
    sources: list[SourceRecord],
) -> list[InsightItem]:
    """
    Map model-selected [S1], [S2], etc. to
    retrieved sources.

    Also extract the insight's signal category.

    Invalid or missing citations are excluded.
    This checks source identity, not factual
    correctness of the insight.
    """

    by_id = {
        f"S{i}": source
        for i, source in enumerate(
            sources,
            start=1,
        )
    }

    result: list[InsightItem] = []

    for line in text.splitlines():
        bullet = re.match(
            r"^\s*[-*]\s+(.+)$",
            line,
        )

        if not bullet:
            continue

        raw = bullet.group(1).strip()

        category = "Other"

        category_tag = re.match(
            r"^\[([^\]]+)\]\s+(.+)$",
            raw,
        )

        if (
            category_tag
            and not re.fullmatch(
                r"S\d+",
                category_tag.group(1),
            )
        ):
            proposed = (
                category_tag.group(1).strip()
            )

            category = (
                proposed
                if proposed in SIGNAL_CATEGORIES
                else "Other"
            )

            raw = (
                category_tag.group(2).strip()
            )

        cited = re.findall(
            r"\[(S\d+)\]",
            raw,
        )

        if (
            not cited
            or any(
                source_id not in by_id
                for source_id in cited
            )
        ):
            continue

        clean_text = re.sub(
            r"\s*\[S\d+\]",
            "",
            raw,
        ).strip()

        if not clean_text:
            continue

        unique_ids = list(
            dict.fromkeys(cited)
        )

        linked_sources: list[InsightSource] = [
            {
                "title": by_id[source_id]["title"],
                "url": by_id[source_id]["url"],
            }
            for source_id in unique_ids
        ]

        result.append({
            "text": clean_text,
            "category": category,
            "sources": linked_sources,
        })

        if len(result) == 3:
            break

    return result


def parse_cited_summary(
    text: str,
    sources: list[SourceRecord],
) -> ExecutiveSummary | None:
    """
    Extract a source-backed executive summary.

    The LLM provides citation IDs such as [S1].
    Python maps those IDs to retrieved URLs.

    Returns None if the summary has missing,
    invalid, or unsupported source IDs.

    This validates citation references, not
    whether every claim is factually supported.
    """

    by_id = {
        f"S{i}": source
        for i, source in enumerate(sources, start=1)
    }

    for line in text.splitlines():
        match = re.match(
            r"^\s*SUMMARY:\s*(.+?)\s*$",
            line,
            re.IGNORECASE,
        )

        if not match:
            continue

        raw_summary = match.group(1).strip()

        cited_ids = re.findall(
            r"\[(S\d+)\]",
            raw_summary,
        )

        # Summary must have valid source references.
        if not cited_ids:
            return None

        if any(
            source_id not in by_id
            for source_id in cited_ids
        ):
            return None

        clean_text = re.sub(
            r"\s*\[S\d+\]",
            "",
            raw_summary,
        ).strip()

        if not clean_text:
            return None

        unique_ids = list(dict.fromkeys(cited_ids))

        linked_sources: list[InsightSource] = [
            {
                "title": by_id[source_id]["title"],
                "url": by_id[source_id]["url"],
            }
            for source_id in unique_ids
        ]

        return {
            "text": clean_text,
            "sources": linked_sources,
        }

    return None


def parse_source_sentiments(
    text: str,
    sources: list[SourceRecord],
) -> list[SourceSentiment]:
    """
    Parse sentiment labels generated by GPT-4o.

    Expected format:
        SENTIMENT [S1]: Positive
        SENTIMENT [S2]: Neutral

    Rejects:
    - Unknown source IDs
    - Invalid labels
    - Conflicting duplicate classifications

    Classification accuracy is not independently
    verified by this function.
    """

    by_id = {
        f"S{i}": source
        for i, source in enumerate(
            sources,
            start=1,
        )
    }

    valid_labels = {
        "Positive",
        "Neutral",
        "Negative",
        "Unclear",
    }

    assignments: dict[str, str] = {}
    conflicts: set[str] = set()

    for line in text.splitlines():
        match = re.fullmatch(
            r"\s*SENTIMENT\s+\[(S\d+)\]:\s*"
            r"(Positive|Neutral|Negative|Unclear)\s*",
            line,
        )

        if not match:
            continue

        source_id, label = match.groups()

        if source_id not in by_id:
            continue

        if label not in valid_labels:
            continue

        if source_id in assignments:
            if assignments[source_id] != label:
                conflicts.add(source_id)
            continue

        assignments[source_id] = label

    result: list[SourceSentiment] = []

    # Preserve the original research source order.
    for source_id, source in by_id.items():
        if source_id in conflicts:
            continue

        label = assignments.get(source_id)

        if label is None:
            continue

        result.append({
            "source_id": source_id,
            "label": label,
            "title": source["title"],
            "url": source["url"],
        })

    return result


def calculate_sentiment_distribution(
    classifications: list[SourceSentiment],
    sources: list[SourceRecord],
) -> SentimentDistribution:
    """
    Calculate sentiment distribution from
    validated source classifications.

    'Unclear' and missing classifications
    count as unassessed.

    Percentages will be calculated by the UI
    using only assessed sources.
    """

    valid_source_ids = {
        f"S{i}"
        for i in range(1, len(sources) + 1)
    }

    labels_by_id: dict[str, str] = {}

    for item in classifications:
        source_id = item.get("source_id")
        label = item.get("label")

        if source_id not in valid_source_ids:
            continue

        if label not in {
            "Positive",
            "Neutral",
            "Negative",
            "Unclear",
        }:
            continue

        if source_id in labels_by_id:
            continue

        labels_by_id[source_id] = label

    positive = sum(
        label == "Positive"
        for label in labels_by_id.values()
    )

    neutral = sum(
        label == "Neutral"
        for label in labels_by_id.values()
    )

    negative = sum(
        label == "Negative"
        for label in labels_by_id.values()
    )

    assessed = positive + neutral + negative

    total_sources = len(sources)

    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "assessed": assessed,
        "unassessed": total_sources - assessed,
        "total_sources": total_sources,
    }


async def analysis_node(state: AgentState):
    print("--- ANALYSING RESULTS ---")

    numbered_sources = "\n\n".join(
        f"[S{i}] TITLE: {source['title']}\n"
        f"URL: {source['url']}\n"
        f"PUBLISHED: "
        f"{source['published_at'] or 'unknown'}\n"
        f"CONTENT: {source['content']}"
        for i, source in enumerate(
            state["sources"],
            start=1,
        )
    )

    prompt = f"""You are a careful competitive intelligence analyst.

Extract up to 3 strategically important, distinct insights about {state['company_name']}.

GROUNDING RULES
1. Use ONLY the numbered source texts below. Do not use outside knowledge.
2. Copy dates, years, financial amounts, and other numbers exactly as given.
3. If no date is supported, do not imply recency.
4. Do not infer causes, market impact, or business outcomes that are not stated.
5. The subject of each insight must be {state['company_name']}.
6. Prioritize strategic partnerships, acquisitions, major product launches, and major strategic shifts.
7. Exclude routine administrative, developer, and financial line items unless truly strategic.
8. Return fewer than three insights if the evidence supports fewer.
9. For EACH insight, cite the numbered source IDs that actually support its claims.
10. Never invent a source ID. Never cite a source that doesn't support the statement.
11. Choose EXACTLY ONE category for each insight from this fixed list:
    Product & Innovation; M&A / Partnerships; Market Expansion; Hiring & Talent; Other.
    Use Other when no category fits clearly. Category is about the event, not company sentiment.

EXECUTIVE SUMMARY RULES
1. Write a concise overall synthesis of the available strategic evidence.
2. Summarize the common direction across findings instead of repeating the first insight.
3. Use only claims supported by the numbered sources.
4. Do not make unsupported claims about market leadership, competitive advantage, or business impact.
5. Cite the source IDs supporting the summary.
6. If the available evidence cannot support a meaningful synthesis, write SUMMARY: None.

SOURCE SENTIMENT RULES
1. Classify every numbered research source separately.
2. Assess how the source portrays the requested company, not whether the article sounds exciting.
3. Use Positive for clearly favorable developments.
4. Use Negative for clearly unfavorable developments.
5. Use Neutral for balanced or factual coverage with no clear favorable or unfavorable portrayal.
6. Use Unclear when the source is unrelated to the requested company, lacks sufficient information, or cannot be classified reliably.
7. Do not assume that a product launch or partnership is automatically positive.
8. Use the exact numbered source ID for every classification.
9. Never invent source IDs or sentiment labels.

OUTPUT FORMAT — EXACTLY:

SUMMARY: A concise supported strategic synthesis. [S1] [S2]

- [M&A / Partnerships] A supported insight. [S1]
- [Product & Innovation] Another supported insight. [S2]

SENTIMENT [S1]: Positive
SENTIMENT [S2]: Neutral

The example labels are illustrative, not predetermined.
Generate exactly one SENTIMENT line for every actual
numbered research source provided, using its correct ID.

Keep the summary and each insight on one line.
Keep each sentiment classification on one line.
Use only the permitted categories and sentiment labels.
Do not add other headings, explanations, or bare links.

NUMBERED SOURCE TEXTS
{numbered_sources}
"""

    # One GPT-4o call for both insight generation
    # and classification.
    response = await llm_analysis.ainvoke([
        HumanMessage(content=prompt)
    ])

    raw_insights = (
        response.content
        if isinstance(response.content, str)
        else ""
    )



    mapped = parse_cited_insights(
        raw_insights,
        state["sources"],
    )

    summary = parse_cited_summary(
        raw_insights,
        state["sources"],
    )

    source_sentiments = parse_source_sentiments(
        raw_insights,
        state["sources"],
    )

    sentiment_distribution = calculate_sentiment_distribution(
        source_sentiments,
        state["sources"],
    )

    return {
        "insights": raw_insights,
        "insight_items": mapped,
        "executive_summary": summary,
        "source_sentiments": source_sentiments,
        "sentiment_distribution": sentiment_distribution,
    }




async def report_node(state: AgentState):
    print("--- GENERATING REPORT ---")

    items = state.get(
        "insight_items",
        [],
    )

    bullets = (
        "\n".join(
            f"- {item['text']}"
            for item in items
        )
        if items
        else (
            "No source-backed strategic insights "
            "were identified."
        )
    )

    return {
        "report": (
            f"# {state['company_name']} "
            f"Insights Report\n\n{bullets}"
        )
    }


def check_research(state: AgentState):
    sources = state.get(
        "sources",
        [],
    )

    attempts = state.get(
        "research_attempts",
        state.get("retry_count", 0),
    )

    if len(sources) >= SOURCE_TARGET:
        return "continue"

    if attempts >= MAX_RESEARCH_CALLS:
        return (
            "continue"
            if sources
            else "fail"
        )

    if sources:
        return (
            "continue"
            if state.get(
                "enrichment_attempted",
                False,
            )
            else "retry"
        )

    return "retry"


workflow = StateGraph(AgentState)

workflow.add_node(
    "research",
    research_node,
)

workflow.add_node(
    "analysis",
    analysis_node,
)

workflow.add_node(
    "report",
    report_node,
)

workflow.add_edge(
    START,
    "research",
)

workflow.add_conditional_edges(
    "research",
    check_research,
    {
        "continue": "analysis",
        "retry": "research",
        "fail": END,
    },
)

workflow.add_edge(
    "analysis",
    "report",
)

workflow.add_edge(
    "report",
    END,
)

app_graph = workflow.compile()


async def run_workflow(company_name: str) -> dict:
    initial_state = {
        "company_name": company_name,
        "retry_count": 0,
        "research_attempts": 0,
        "enrichment_attempted": False,
        "sources": [],
        "source_stats": {
            "returned": 0,
            "accepted": 0,
            "discarded": 0,
        },
        "research_warnings": [],
    }

    result = await app_graph.ainvoke(
        initial_state,
        config={
            "callbacks": [langfuse_handler]
        },
    )

    attempts = result.get(
        "research_attempts",
        0,
    )

    if not result.get("sources"):
        return {
            "status": "failed",
            "error_code": "RESEARCH_EXHAUSTED",
            "research_attempts": attempts,
            "report": None,
            "search_results": "",
            "insights": "",
            "insight_items": [],
            "sources": [],
            "executive_summary": None,
            "source_stats": result.get(
                "source_stats",
                {},
            ),
            "research_warnings": result.get(
                "research_warnings",
                [],
            ),
            "source_sentiments": [],
            "sentiment_distribution": {
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "assessed": 0,
                "unassessed": 0,
                "total_sources": 0,
            },
        }

    report = result.get("report")

    if not report:
        raise RuntimeError(
            "Workflow ended without generating "
            "a report."
        )

    return {
        "status": "completed",
        "error_code": None,
        "research_attempts": attempts,
        "report": report,
        "search_results": result.get(
            "search_results",
            "",
        ),
        "insights": result.get(
            "insights",
            "",
        ),
        "executive_summary": result.get(
            "executive_summary",
        ),
        "insight_items": result.get(
            "insight_items",
            [],
        ),
        "sources": result.get(
            "sources",
            [],
        ),
        "source_stats": result.get(
            "source_stats",
            {},
        ),
        "research_warnings": result.get(
            "research_warnings",
            [],
        ),
    }
