
import workflow


def source(title, url, content):
    return {
        "title": title,
        "url": url,
        "content": content,
        "published_at": None,
    }


def test_normalize_official_domain():
    assert workflow.normalize_company_domain(
        "https://www.linear.app/changelog"
    ) == "linear.app"

    assert workflow.normalize_company_domain(
        "linear.app"
    ) == "linear.app"


def test_research_query_uses_company_website():
    query = workflow.build_query(
        "Linear",
        official_domain="linear.app",
    )

    assert "Linear" in query
    assert "linear.app" in query


def test_existing_name_only_query_still_works():
    query = workflow.build_query("Linear")

    assert "Linear" in query
    assert "product" in query.lower()


def test_official_product_update_is_relevant():
    item = source(
        "Linear introduces new issue tracking features",
        "https://linear.app/changelog/example",
        (
            "Linear introduced new capabilities "
            "for software development teams."
        ),
    )

    result = workflow.classify_company_evidence(
        item,
        company_name="Linear",
        official_domain="linear.app",
    )

    assert result == "relevant"


def test_independent_source_with_identity_evidence():
    item = source(
        "Linear announces product developments",
        "https://example.com/linear-news",
        (
            "Linear (linear.app), the software "
            "issue tracking platform, announced "
            "new product capabilities."
        ),
    )

    result = workflow.classify_company_evidence(
        item,
        company_name="Linear",
        official_domain="linear.app",
    )

    assert result == "relevant"


def test_unrelated_research_is_rejected():
    unrelated = [
        source(
            "Nanobiotechnology Market Report",
            "https://example.com/nanotech",
            "Analysis of nanobiotechnology markets.",
        ),
        source(
            "Apple and TSMC semiconductor partnership",
            "https://example.com/semiconductors",
            "Apple and TSMC discuss chip production.",
        ),
    ]

    for item in unrelated:
        result = workflow.classify_company_evidence(
            item,
            company_name="Linear",
            official_domain="linear.app",
        )

        assert result == "irrelevant"


def test_ambiguous_company_reference_is_uncertain():
    item = source(
        "Linear announces new funding",
        "https://example.com/funding",
        (
            "Linear announced a funding event. "
            "The article provides no identifiable "
            "company website or product context."
        ),
    )

    result = workflow.classify_company_evidence(
        item,
        company_name="Linear",
        official_domain="linear.app",
    )

    # The name alone does not conclusively
    # establish which Linear this refers to.
    assert result == "uncertain"
