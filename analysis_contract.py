"""Validate structured analysis and keep evidence linked to immutable source IDs.

Quote matching checks provenance only. It does not prove entailment, materiality,
event dates or independent factual accuracy.
"""

import json
import re
import calendar
import hashlib
from datetime import date, datetime, timedelta, timezone


MAX_CONTEXT_ITEMS = 5


def normalized(value):
    return " ".join(value.split())


def checked_evidence(value, sources, prefix, *, excerpt_limit=None):
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        return None
    result = []
    seen = set()
    for citation in value:
        if not isinstance(citation, dict):
            return None
        source_id, quote = citation.get("source_id"), citation.get("quote")
        if not isinstance(source_id, str) or len(source_id) > 16 or not re.fullmatch(rf"{prefix}[1-9]\d*", source_id):
            return None
        index = int(source_id[1:]) - 1
        if index >= len(sources) or not isinstance(quote, str) or not 1 <= len(quote) <= 1600:
            return None
        source = sources[index]
        content = source["content"] if excerpt_limit is None else source["content"][:excerpt_limit]
        if not normalized(quote) or not re.search(
            r"(?<!\w)" + re.escape(normalized(quote)) + r"(?!\w)", normalized(content),
        ):
            return None
        key = (source_id, normalized(quote))
        if key not in seen:
            result.append({"source_id": source_id, "quote": quote, "title": source["title"], "url": source["url"]})
            seen.add(key)
    return result


def one_line(value, limit=1600):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit and "\n" not in value and "\r" not in value


def build_evidence_registry(company_sources, competitor_sources=()):
    """Index exact body spans. IDs bind namespace, position and source contents.

    Titles and publication metadata are never selectable evidence. A long line
    is split only at whitespace; no punctuation is repaired or synthesized.
    """
    registry = {}
    for prefix, sources in (("S", company_sources), ("C", competitor_sources)):
        for index, source in enumerate(sources, 1):
            content = source.get("content", "")
            if not isinstance(content, str):
                continue
            source_id = f"{prefix}{index}"
            binding = hashlib.sha256(json.dumps([source_id, source.get("title"), source.get("url"), content],
                ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
            shown = content[:1600] if prefix == "C" else content
            for line in re.finditer(r"[^\r\n]+", shown):
                start = line.start() + len(line[0]) - len(line[0].lstrip())
                end = line.end() - (len(line[0]) - len(line[0].rstrip()))
                while start < end:
                    stop = min(start + 1200, end)
                    if stop < end:
                        breaks = list(re.finditer(r"\s+", content[start:stop]))
                        if not breaks:
                            following = re.search(r"\s+", content[stop:end])
                            start = stop + following.end() if following else end
                            continue
                        stop = start + breaks[-1].start()
                    quote = content[start:stop].rstrip()
                    # Do not offer a partial last word clipped by the C excerpt.
                    if stop == len(shown) < len(content) and quote and re.match(r"\w", content[stop:stop + 1]):
                        quote = re.sub(r"\w+$", "", quote).rstrip()
                    if quote and quote not in ("o", "-", "*", "•", "●") and re.search(r"\w", quote):
                        checked = checked_evidence([{"source_id": source_id, "quote": quote}], sources, prefix,
                                                   excerpt_limit=1600 if prefix == "C" else None)
                        if checked:
                            evidence_id = f"{source_id}-{binding}-{start}-{start + len(quote)}"
                            registry[evidence_id] = {**checked[0], "evidence_id": evidence_id}
                    start = stop
                    while start < end and content[start].isspace():
                        start += 1
    return registry


def resolve_evidence_ids(value, registry, prefix):
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        return None
    result, seen = [], set()
    for evidence_id in value:
        if not isinstance(evidence_id, str) or len(evidence_id) > 80:
            return None
        evidence = registry.get(evidence_id)
        if evidence is None or not evidence["source_id"].startswith(prefix):
            return None
        if evidence_id not in seen:
            result.append(dict(evidence))
            seen.add(evidence_id)
    return result


def bounded_evidence_view(headers, registry, max_characters):
    """Fit whole canonical spans, round-robin across sources in supplied order.

    Original source records and IDs do not change. This is input budgeting, not
    a ranking or semantic support decision. Metadata itself must fit the cap.
    """
    grouped = {source_id: [] for source_id, _ in headers}
    for evidence in registry.values():
        if evidence["source_id"] in grouped:
            grouped[evidence["source_id"]].append(evidence)
    selected = {source_id: [] for source_id, _ in headers}

    def body(rows):
        return "\n".join(f"[{row['evidence_id']}] {row['quote']}" for row in rows) or "No selectable body evidence is available."

    def render():
        return "\n\n".join(header + body(selected[source_id]) for source_id, header in headers)

    current = render()
    if len(current) > max_characters:
        raise ValueError("Evidence metadata exceeds the analysis input budget before any model call.")
    remaining = max_characters - len(current)
    positions = {source_id: 0 for source_id, _ in headers}
    while True:
        progressed = False
        for source_id, _ in headers:
            position = positions[source_id]
            rows = grouped[source_id]
            if position >= len(rows):
                continue
            candidate = rows[position]
            extra = len(body([*selected[source_id], candidate])) - len(body(selected[source_id]))
            if extra > remaining:
                positions[source_id] = len(rows)
                continue
            selected[source_id].append(candidate)
            positions[source_id] += 1
            remaining -= extra
            progressed = True
        if not progressed:
            break
    return render(), {row["evidence_id"] for rows in selected.values() for row in rows}


def default_analysis_window(today=None):
    as_of = today if today is not None else datetime.now(timezone.utc).date()
    return {"start": (as_of - timedelta(days=90)).isoformat(), "end": as_of.isoformat()}


def validated_analysis_window(value):
    if not isinstance(value, dict):
        raise ValueError("Analysis window must contain ISO start/end dates.")
    start, end = date.fromisoformat(value["start"]), date.fromisoformat(value["end"])
    if start > end:
        raise ValueError("Analysis window starts after its end.")
    return {"start": start.isoformat(), "end": end.isoformat()}


def dated_announcement_from_evidence(text, evidence, company_name, months):
    """Return a date when a dateline and its announcement match the same claim."""
    subject = re.escape(company_name)
    claim = re.match(rf"^{subject} (?:has )?announced (?P<detail>.+)", normalized(text), re.I)
    if not claim:
        return None
    detail = re.findall(r"\w+", claim["detail"].casefold())[:6]
    if len(detail) < 6:
        return None
    month = "(?:" + "|".join(months) + ")"
    dateline = re.compile(
        rf"^(?:[^\d,.:!?]{{1,48}}, ){{0,2}}"
        rf"(?P<date>(?:\d{{1,2}} {month}|{month} \d{{1,2}}),? 20\d{{2}}): "
        rf"{subject}(?:, [^,.:!?]{{1,130}},)? (?:has )?(?:today )?announced (?P<detail>[^.!?]+)",
        re.I,
    )
    month_numbers = {name.casefold(): number for number, name in enumerate(months, 1)}
    dates = set()
    for item in evidence:
        match = dateline.match(normalized(item["quote"]))
        if not match or re.findall(r"\w+", match["detail"].casefold())[:6] != detail:
            continue
        first, second, year = match["date"].replace(",", "").split()
        day, month_name = (first, second) if first.isdigit() else (second, first)
        try:
            dates.add(date(int(year), month_numbers[month_name.casefold()], int(day)))
        except ValueError:
            return None
    return next(iter(dates)) if len(dates) == 1 else None


def period_end_from_evidence(text, quotes, company_name, months):
    """Recognize completed actions when the selected passage supplies the same period."""
    company = re.escape(company_name)
    investment = re.fullmatch(rf"In (20\d{{2}}), .+ invested .+ in {company}\.", text, re.I)
    if investment and any(text.casefold() in quote for quote in quotes):
        return date(int(investment[1]), 12, 31)

    month_names = "|".join(months)
    release = re.match(
        rf"^(?P<generation>The \w+ generation of )?(?:the )?(?P<product>{company}\b[\w -]{{0,60}})"
        rf"(?P<action> was (?:launched|released|introduced) in (?P<month>{month_names}) (?P<year>20\d{{2}}))(?=[,.]|$)",
        text, re.I,
    )
    if not release:
        return None
    if not any(release[0].casefold() in quote for quote in quotes):
        # A device reference must follow a sentence naming the specific product.
        if not release["generation"]:
            return None
        device_reference = (rf"\bknown for (?:the )?{re.escape(release['product'])}\b[^.!?]{{0,420}}\.\s+"
                            + re.escape(release["generation"] + "the device" + release["action"])
                            + r"(?=[,.]|$)")
        if not any(re.search(device_reference, quote, re.I) for quote in quotes):
            return None
    month = next(i for i, name in enumerate(months, 1) if name.casefold() == release["month"].casefold())
    year = int(release["year"])
    return date(year, month, calendar.monthrange(year, month)[1])


def is_current_integration(text, evidence, company_name):
    """A cited capability in the present tense belongs with company background."""
    if not company_name:
        return False
    text = normalized(text)
    pattern = rf"(?:Through a collaboration with [^,.!?]+, )?{re.escape(company_name)} integrates [^.!?]+ into [^.!?]+\."
    return bool(re.fullmatch(pattern, text, re.I)
                and not re.search(r"\b(?:now|new|first|starting|today|20\d{2})\b", text, re.I)
                and any(text.casefold() in normalized(item["quote"]).casefold() for item in evidence))


def temporal_context(text, evidence, company_name, analysis_window):
    """Narrow old-action detection, not a general event-date or truth verifier.

    Financial periods, metadata, future effective dates and bare date mentions
    do not establish when an event occurred. Ambiguous cases stay uncertain.
    """
    uncertain = {"temporal_status": "timing_uncertain", "temporal_note": "Event timing is not established by a validated dated-action pattern. Publication dates may differ from event dates."}
    if not company_name or analysis_window is None:
        return uncertain
    window = validated_analysis_window(analysis_window)
    months = "January February March April May June July August September October November December".split()
    if re.search(r"[.!?]\s+\S|\b(?:today|now|currently|yesterday|this week|this month|this year)\b", text, re.I):
        return uncertain
    quotes = [normalized(e["quote"]).casefold() for e in evidence]
    period_end = period_end_from_evidence(normalized(text), quotes, company_name, months)
    if period_end and period_end < date.fromisoformat(window["start"]):
        return {"temporal_status": "historical",
                "temporal_note": f"The selected passage dates this action to a period before {window['start']}."}
    if not re.search(r"\b20\d{2}\b", text):
        announced = dated_announcement_from_evidence(text, evidence, company_name, months)
        if announced and announced < date.fromisoformat(window["start"]):
            return {"temporal_status": "historical",
                    "temporal_note": f"The selected passage dates this announcement to {announced.isoformat()}, "
                                     f"before the report window started on {window['start']}."}
        return uncertain
    dated_mentions = re.findall(r"\b(?:" + "|".join(months) + r")\s+(?:\d{1,2},?\s+)?20\d{2}\b", text, re.I)
    if len(dated_mentions) != 1:
        return uncertain
    pattern = re.compile(r"^In\s+(" + "|".join(months) + r")\s+(20\d{2}),\s+" + re.escape(company_name)
        + r"\s+(?:has\s+)?(?:completed|announced|launched|introduced|acquired|raised|appointed)\b", re.I)
    matches = list(pattern.finditer(text))
    if not matches:
        return uncertain
    for match in matches:
        if not any(normalized(match[0]).casefold() in quote for quote in quotes):
            return uncertain
        month = next(i for i, name in enumerate(months, 1) if name.casefold() == match[1].casefold())
        year = int(match[2])
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        if last_day >= date.fromisoformat(window["start"]):
            return uncertain
    return {"temporal_status": "historical", "temporal_note": f"The claim and selected body evidence explicitly date the company's action before {window['start']}. Kept as historical context, not a recent development."}


def decode_analysis(text, company_sources, competitor_sources, categories, *, company_name=None, analysis_window=None,
                    allowed_evidence_ids=None, background_source_ids=()):
    """Return None for the legacy line protocol, otherwise a checked envelope."""
    if not text.lstrip().startswith(("{", "[", "\x60")):
        return None
    invalid = {"text": "", "rejected": {"insight"}, "insight_evidence": {},
               "competitor_evidence": {}, "competitor_warnings": [], "structured": True,
               "context_items": [], "context_warnings": []}
    try:
        envelope = json.loads(text)
    except (TypeError, ValueError):
        return invalid
    if not isinstance(envelope, dict) or type(envelope.get("schema_version")) is not int or envelope["schema_version"] not in (1, 2):
        return invalid
    version = envelope["schema_version"]
    registry = build_evidence_registry(company_sources, competitor_sources) if version == 2 else {}

    def evidence_for(row, sources, prefix):
        if version == 2:
            if allowed_evidence_ids is not None:
                ids = row.get("evidence_ids")
                if not isinstance(ids, list) or any(not isinstance(key, str) or key not in allowed_evidence_ids for key in ids):
                    return None
            return None if "evidence" in row else resolve_evidence_ids(row.get("evidence_ids"), registry, prefix)
        return checked_evidence(row.get("evidence"), sources, prefix, excerpt_limit=1600 if prefix == "C" else None)
    status = envelope.get("status")
    insights = envelope.get("insights")
    if status not in ("complete", "insufficient_evidence") or not isinstance(insights, list):
        return invalid
    if (status == "insufficient_evidence" and insights) or (status == "complete" and not insights):
        return invalid
    lines = ["SUMMARY: None"]
    rejected = set()
    metadata = {}
    seen_text = set()
    context_items, context_warnings, seen_context = [], [], set()

    def add_context(row, evidence, timing, *, limit=MAX_CONTEXT_ITEMS):
        key = normalized(row["text"]).casefold()
        if key in seen_context or key in seen_text:
            return
        if len(context_items) >= limit:
            context_warnings.append("Additional background context was excluded at the bounded item limit.")
            return
        seen_context.add(key)
        category = row.get("category")
        context_items.append({"text": row["text"].strip(), "event": row["event"].strip(),
            "category": category if isinstance(category, str) and category in categories else "Other",
            "evidence": evidence, "sources": list({e["url"]: {"title": e["title"], "url": e["url"]} for e in evidence}.values()), **timing})
    for row in insights:
        if not isinstance(row, dict):
            rejected.add("insight")
            continue
        text, event = row.get("text"), row.get("event")
        evidence = evidence_for(row, company_sources, "S")
        if not one_line(text) or not one_line(event, 160) or evidence is None:
            rejected.add("insight")
            continue
        # IDs belong in the evidence array, never inside the claim text.
        if re.search(r"\[[A-Z]\d+", text):
            rejected.add("insight")
            continue
        timing = temporal_context(text, evidence, company_name, analysis_window) if version == 2 else {}
        if timing.get("temporal_status") == "historical":
            add_context(row, evidence, timing, limit=3)
            context_warnings.append("Historical observations were moved out of selected developments and retained as context.")
            continue
        if version == 2 and is_current_integration(text, evidence, company_name):
            add_context(row, evidence, {**timing, "temporal_note": "The selected passage describes an existing integration."}, limit=3)
            continue
        if version == 2 and all(item["source_id"] in background_source_ids for item in evidence):
            add_context(row, evidence, timing, limit=3)
            context_warnings.append("Observations supported only by reference pages are shown as background.")
            continue
        key = normalized(text).casefold()
        if key in seen_text:
            continue
        seen_text.add(key)
        if len(metadata) == 3:
            rejected.add("insight")
            continue
        category = row.get("category")
        if not isinstance(category, str) or category not in categories:
            category = "Other"
        text = text.strip()
        ids = list(dict.fromkeys(item["source_id"] for item in evidence))
        lines.append(f"- [{category}] {text} " + " ".join(f"[{item}]" for item in ids))
        metadata[text] = {"event": event.strip(), "evidence": evidence, **timing}
    if version == 2:
        supplied_context = envelope.get("context_items", [])
        if not isinstance(supplied_context, list):
            supplied_context = []
            context_warnings.append("Generated background context had an invalid structure.")
        for row in supplied_context:
            if not isinstance(row, dict):
                context_warnings.append("A background observation was excluded because its evidence could not be validated.")
                continue
            evidence = evidence_for(row, company_sources, "S")
            if (not one_line(row.get("text")) or not one_line(row.get("event"), 160) or evidence is None
                or re.search(r"\[[A-Z]\d+", row["text"])):
                context_warnings.append("A background observation was excluded because its evidence could not be validated.")
                continue
            add_context(row, evidence, temporal_context(row["text"], evidence, company_name, analysis_window))
    sentiments = envelope.get("sentiments", [])
    if isinstance(sentiments, list):
        for row in sentiments:
            if (isinstance(row, dict) and isinstance(row.get("source_id"), str)
                and re.fullmatch(r"S[1-9]\d*", row["source_id"])
                and row.get("label") in ("Positive", "Neutral", "Negative", "Unclear")):
                lines.append(f"SENTIMENT [{row['source_id']}]: {row['label']}")
    competitors = envelope.get("competitors", [])
    competitor_metadata = {}
    competitor_warnings = []
    if not isinstance(competitors, list):
        competitors = []
        competitor_warnings.append("Generated competitor records had an invalid structure.")
    for row in competitors:
        if not isinstance(row, dict):
            competitor_warnings.append("A generated competitor was excluded because its quoted evidence could not be checked.")
            continue
        name, overlap = row.get("name"), row.get("overlap")
        evidence = evidence_for(row, competitor_sources, "C")
        if not one_line(name, 120) or "|" in name or not one_line(overlap) or evidence is None:
            competitor_warnings.append("A generated competitor was excluded because its quoted evidence could not be checked.")
            continue
        if re.search(r"\[[A-Z]\d+", overlap):
            competitor_warnings.append("A generated competitor was excluded because its quoted evidence could not be checked.")
            continue
        name = normalized(name)
        if name.casefold() in competitor_metadata:
            continue
        ids = list(dict.fromkeys(item["source_id"] for item in evidence))
        lines.append(f"COMPETITOR: {name.strip()} | {overlap.strip()} " + " ".join(f"[{item}]" for item in ids))
        competitor_metadata[name.strip().casefold()] = evidence
    return {"text": "\n".join(lines), "rejected": rejected, "insight_evidence": metadata,
            "competitor_evidence": competitor_metadata, "competitor_warnings": list(dict.fromkeys(competitor_warnings)),
            "structured": True, "schema_version": version, "context_items": context_items,
            "context_warnings": list(dict.fromkeys(context_warnings))}
