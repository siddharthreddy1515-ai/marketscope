
import asyncio
from urllib.parse import urlparse

from workflow import research_node


def test_research_preserves_source_metadata(monkeypatch):
    # Use saved research data, not live Tavily calls.
    monkeypatch.setenv("MOCK_TAVILY", "true")

    result = asyncio.run(
        research_node({
            "company_name": "Tesla",
            "retry_count": 0,
        })
    )

    sources = result["sources"]

    # The research must return at least one source.
    assert len(sources) > 0

    # Each source needs a title, valid-looking URL,
    # and its original content.
    for source in sources:
        assert source["title"].strip()
        assert isinstance(source["content"], str)

        parsed_url = urlparse(source["url"])
        assert parsed_url.scheme in ("http", "https")
        assert parsed_url.netloc

        if source["content"].strip():
            assert source["content"] in result["search_results"]

    assert result["retry_count"] == 0
    assert result["search_results"].strip()
