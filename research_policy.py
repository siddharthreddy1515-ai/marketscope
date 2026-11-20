"""Bounded research routing based on observable gaps in collected evidence."""

import re
from urllib.parse import urlsplit


PROFILE_TITLE = re.compile(
    r"\b(?:company profile|company overview|IPO timeline)\b|^how many people use\b", re.I,
)
REFERENCE_TITLE = re.compile(
    r"^(?:terms of (?:service|use)|terms and conditions|privacy policy)(?:\s*[|:–-]|\s*$)", re.I,
)
REFERENCE_PATH = re.compile(
    r"/(?:terms(?:[-_]of[-_](?:service|use)|[-_]and[-_]conditions)?|privacy(?:[-_]policy)?|legal|cookies)"
    r"(?:/|\.(?:html?|aspx?)$|$)", re.I,
)
RESTATEMENT = re.compile(r"\b(?:restat\w*|revis\w*|correct\w*)\b", re.I)
BACKGROUND_TITLE = re.compile(
    r"\b(?:growth playbook|GTM playbooks?|product launch strategies|"
    r"how .{1,80} (?:grew|grows)|ARR(?=:)|financials overview|growth strategy)\b", re.I,
)


def is_profile(source, company_name):
    return bool(PROFILE_TITLE.search(source["title"])) or (
        source["title"].strip().casefold() == company_name.strip().casefold()
        and bool(re.search(r"(?m)^\s*(?:-\s*)?Website:", source["content"][:1200]))
    )


def is_reference_page(source, company_name):
    """Identify static reference pages that are useful as background."""
    page = urlsplit(source["url"])
    return bool(
        is_profile(source, company_name)
        or REFERENCE_TITLE.search(source["title"].strip())
        or REFERENCE_PATH.search(page.path)
        or (page.hostname == "tavily.com" and page.path.startswith("/knowledge/company/"))
    )


def has_official_detail(sources, company_name, domain):
    for source in sources:
        page = urlsplit(source["url"])
        host = page.hostname or ""
        if host != domain and not host.endswith("." + domain):
            continue
        path = page.path.strip("/").lower()
        if (path and not re.match(r"^(?:help|support|login|sign-in|contact)(?:/|$)", path)
                and source_evidence_role(source, company_name) != "development_candidate"
                and not is_reference_page(source, company_name)):
            return True
    return False


def earnings_period(source):
    """Identify explicit earnings periods only, without grouping source records."""
    title = source["title"].strip()
    text = source["content"][:700] if re.fullmatch(
        r"(?:document|ex-99\.1|exhibit 99\.1)", title, re.I,
    ) else title
    if re.search(r"\b(?:preview|to announce|what to expect|upcoming|to report)\b", title, re.I):
        return None
    if not re.search(r"\b(?:earnings|financial results|quarterly results|revenue|earnings call)\b", text, re.I):
        return None
    quarter = re.search(r"\bq([1-4])\s*['’\s-]?(20\d{2}|\d{2})\b", text, re.I)
    ordinal = re.search(r"\b(first|second|third|fourth)[ -]quarter\s+(20\d{2}|\d{2})\b", text, re.I)
    if quarter:
        number, year = int(quarter[1]), int(quarter[2])
    elif ordinal:
        number = ("first", "second", "third", "fourth").index(ordinal[1].lower()) + 1
        year = int(ordinal[2])
    else:
        return None
    return (year if year >= 2000 else 2000 + year, number)


def source_evidence_role(source, company_name):
    """Triage source purpose, without asserting event dates or materiality."""
    title = source.get("title", "")
    content = source.get("content", "")
    path = urlsplit(source.get("url", "")).path
    if "/templates/" in path or path.endswith("/changelog/timeline") or is_reference_page(source, company_name):
        return "background"
    actor_action = re.compile(
        rf"(?<!\w){re.escape(company_name)}(?:['’]s)?\s+"
        r"(?:has\s+|just\s+)*(?:announc\w*|launch\w*|releas\w*|introduc\w*|"
        r"appoint\w*|names?|complet\w*|acquir\w*|raised|doubled)\b", re.I,
    )
    if actor_action.search(title) or actor_action.match(content.strip()) or earnings_period(source):
        return "development_candidate"
    if BACKGROUND_TITLE.search(title):
        return "background"
    if ("/changelog" in path or re.search(r"/releases/\d{4}-\d{2}-\d{2}", path)) and re.search(
        r"\b(?:can now|now available|introducing|we.re adding|this update|announc\w*|now respond)\b",
        content, re.I,
    ):
        return "development_candidate"
    return "uncertain"


def is_short_search_preview(source):
    page = urlsplit(source["url"])
    return (page.hostname in ("google.com", "www.google.com") and page.path == "/url"
            and len(source["content"].strip()) <= 200)


def distinct_preview_title(news_sources, company_name):
    """Find a lead worth a follow-up when previews dominate a successful news batch."""
    previews = [source for source in news_sources if is_short_search_preview(source)]
    if len(previews) < 2 or 2 * len(previews) <= len(news_sources):
        return None

    ignored = set(re.findall(r"\w+", company_name.casefold()))

    def topics(title):
        return {word for word in re.findall(r"\w+", title.casefold()) if len(word) >= 4} - ignored

    document_topics = [topics(source["title"]) for source in news_sources if not is_short_search_preview(source)]
    name = re.compile(r"(?<!\w)" + re.escape(company_name) + r"(?!\w)", re.I)
    for preview in previews:
        title, body = preview["title"], preview["content"]
        if not name.search(title) or not name.search(body):
            continue
        terms = topics(title)
        if len(terms) < 3 or any(
            len(terms & other) >= max(3, min(len(terms), len(other)) / 2)
            for other in document_topics
        ):
            continue
        # Bound the lead and keep provider-specific query syntax out of its title.
        return " ".join(re.findall(r"[\w.-]+", title[:160])[:22])
    return None


def ranked_company_sources(sources, company_name):
    """Choose the order of model inputs while retaining their original source IDs."""
    role_order = {"development_candidate": 0, "uncertain": 1, "background": 2}

    def priority(pair):
        _, source = pair
        return (is_short_search_preview(source), role_order[source_evidence_role(source, company_name)])

    ranked = sorted(enumerate(sources, start=1), key=priority)
    seen = set()
    selected = []
    for index, source in ranked:
        body = " ".join(source["content"].split())
        if body and body in seen:
            continue
        seen.add(body)
        selected.append((index, source))
    return selected


def assess_research_evidence(sources, company_name):
    roles = [source_evidence_role(source, company_name) for source in sources]
    background = roles.count("background")
    developments = roles.count("development_candidate")
    uncertain = roles.count("uncertain")
    if not sources:
        status = "empty"
    elif background == len(sources):
        status = "background_only"
    elif developments:
        status = "mixed" if background or uncertain else "development_candidates"
    else:
        status = "uncertain"
    limitations = []
    if status == "background_only":
        limitations.append(
            "The collected sources provide background information. "
            "They do not establish an overview of recent developments. Important events may be missing."
        )
    elif status in ("empty", "uncertain"):
        limitations.append("The collected evidence does not establish a reliable overview of recent developments.")
    if developments:
        limitations.append(
            "Development candidates are unverified source claims. Event dates, materiality and completeness require review."
        )
    return {
        "status": status, "background_sources": background,
        "development_sources": developments, "uncertain_sources": uncertain,
        "limitations": limitations,
    }


def research_lane(state, baseline_route, *, max_calls):
    """Select stop, business or official using a per-search outcome record."""
    if state.get("research_attempts", 0) >= max_calls:
        return "stop"
    if baseline_route == "retry":
        return "business"
    sources = state.get("sources", [])
    if not sources or not state.get("official_domain") or state.get("official_attempted"):
        return "stop"
    substantive = [s for s in sources if source_evidence_role(s, state["company_name"]) != "background"]
    profiles_dominate = (len(sources) - len(substantive)) * 2 > len(sources)
    if state.get("enrichment_attempted"):
        successful_business = (
            state.get("last_search_lane") == "business"
            and state.get("last_search_ok") is True
        )
        return "official" if profiles_dominate and successful_business else "stop"
    if profiles_dominate:
        return "official"
    if len(substantive) > 1 and not any(
        RESTATEMENT.search(f"{s['title']}\n{s['content'][:700]}") for s in substantive
    ):
        periods = [earnings_period(s) for s in substantive]
        if periods[0] and all(period == periods[0] for period in periods):
            return "official"
    return "stop"


def official_search_parameters(company, domain, start_date):
    return {
        "query": f'"{company}" changelog releases product features announcements updates',
        "include_domains": [domain],
        "start_date": start_date,
    }
