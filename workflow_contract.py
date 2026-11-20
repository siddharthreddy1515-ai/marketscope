"""Provider-free inputs and finding categories shared by the API and workflow."""

import re
from urllib.parse import urlsplit


SIGNAL_CATEGORIES = (
    "Product & Innovation",
    "Funding & Investment",
    "M&A / Partnerships",
    "Market Expansion",
    "Hiring & Talent",
    "Other",
)


def normalize_company_domain(
    website: str | None,
) -> str | None:
    """
    Normalize a supplied website to a hostname.

    https://www.linear.app/changelog -> linear.app
    linear.app -> linear.app
    """
    if website is None or not website.strip():
        return None

    value = website.strip()

    if "://" not in value:
        value = "https://" + value

    parsed = urlsplit(value)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Invalid company website")

    domain = parsed.hostname.lower().rstrip(".")

    if domain.startswith("www."):
        domain = domain[4:]

    labels = domain.split(".")

    if (
        len(labels) < 2
        or any(
            not re.fullmatch(r"[a-z0-9-]+", label)
            or label.startswith("-")
            or label.endswith("-")
            for label in labels
        )
    ):
        raise ValueError("Invalid company domain")

    return domain
