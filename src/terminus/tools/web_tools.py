"""Web research tools for the /ask agent, backed by the official Firecrawl SDK.

Verified against firecrawl-py 4.44.0:

* ``Firecrawl(api_key=..., api_url=..., timeout=..., max_retries=..., ...)``
* ``client.search(query, sources=[...], limit=..., timeout=...) -> SearchData``
  with ``SearchData.web`` holding ``SearchResultWeb(url, title, description,
  position, category)``
* ``client.scrape(url, formats=["markdown"], only_main_content=True,
  timeout=<ms>) -> Document`` with ``Document.markdown`` and
  ``Document.metadata`` (``.title``, ``.source_url``, ``.status_code``)
* failures raise subclasses of ``firecrawl.v2.utils.error_handler.FirecrawlError``
  (``UnauthorizedError``, ``RateLimitError``, ``RequestTimeoutError``, ...)

Search intentionally does NOT pass ``scrape_options``: Firecrawl documents the
two-step search-then-scrape pattern for when you want to select a result, it
costs 1 extra credit per result, and it would return a full page body for every
hit. Use ``web_search`` to discover a URL, then ``web_fetch`` to read one.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse

from langchain.tools import tool

from terminus.env import load_project_env
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

_API_KEY_ENV = "FIRECRAWL_API_KEY"

# Firecrawl bills 2 credits per 10 search results and 1 credit per scraped page,
# so a small limit keeps both cost and context bounded.
_SEARCH_LIMIT = 5
_MAX_DESCRIPTION_CHARS = 300
_MAX_FETCH_CHARS = 20_000

# Firecrawl per-call timeouts are in MILLISECONDS; the client-level timeout below
# is in seconds and bounds the HTTP request itself.
_SEARCH_TIMEOUT_MS = 30_000
_FETCH_TIMEOUT_MS = 45_000
_CLIENT_TIMEOUT_SECONDS = 60

# The configured key plus the documented Firecrawl key shape, so neither can ever
# reach the model, the logs, or a tool result.
_KEY_SHAPE = re.compile(r"fc-[A-Za-z0-9_\-]{8,}")

_client = None


def _redact(text: object) -> str:
    """Collapse a message to one short line with any credential removed."""
    message = " ".join(str(text or "").split())
    key = os.environ.get(_API_KEY_ENV)
    if key:
        message = message.replace(key, "[REDACTED]")
    message = _KEY_SHAPE.sub("[REDACTED]", message)
    return message[:300] or "no detail available"


def _reason(exc: BaseException) -> str:
    """Short, redacted, actionable reason for a failed web call."""
    detail = _redact(str(exc))
    hints = {
        "UnauthorizedError": "authentication failed, check FIRECRAWL_API_KEY",
        "PaymentRequiredError": "the Firecrawl account needs payment or credits",
        "RateLimitError": "rate limited by Firecrawl, wait before retrying",
        "RequestTimeoutError": "Firecrawl timed out handling the request",
        "WebsiteNotSupportedError": "Firecrawl cannot access this site",
        "BadRequestError": "Firecrawl rejected the request as invalid",
        "InternalServerError": "Firecrawl reported an internal error",
        "ProviderTermsRequiredError": "the target site requires provider terms",
    }
    hint = hints.get(type(exc).__name__)
    if not hint:
        return detail
    if not detail or detail.lower() == type(exc).__name__.lower():
        return hint
    return f"{hint} ({detail})"


def _get_client():
    """Return a process-wide Firecrawl client, creating it on first use.

    Mirrors the existing caching style in llm/factory.py: one client per process
    so connection pooling is reused instead of rebuilt per tool call.
    """
    global _client
    if _client is not None:
        return _client

    load_project_env(Path.cwd())
    api_key = os.environ.get(_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{_API_KEY_ENV} is not set; add it to the project .env file"
        )

    try:
        from firecrawl import Firecrawl
    except ImportError as exc:
        raise RuntimeError(
            "the firecrawl-py package is not installed in this environment"
        ) from exc

    _client = Firecrawl(api_key=api_key, timeout=_CLIENT_TIMEOUT_SECONDS)
    logger.info("Firecrawl client ready (key from %s)", _API_KEY_ENV)
    return _client


def _field(item: object, name: str, default: str = "") -> str:
    """Read a field from either a pydantic model or a plain dict."""
    if item is None:
        return default
    value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
    return default if value is None else str(value)


def _result_field(item: object, name: str) -> str:
    """Read a search-result field, falling back to scrape Document metadata.

    ``SearchData.web`` is typed ``Union[SearchResultWeb, Document]``; with plain
    search it is always SearchResultWeb, but a Document carries the same
    information under ``metadata``.
    """
    value = _field(item, name)
    if value:
        return value
    metadata = getattr(item, "metadata", None)
    if metadata is not None:
        if name == "url":
            return _field(metadata, "source_url")
        if name == "title":
            return _field(metadata, "title") or _field(metadata, "og_title")
        if name == "description":
            return _field(metadata, "description")
    return ""


@tool
def web_search(query: str) -> str:
    """
    Search the public web and return a short list of results with their titles,
    URLs and descriptions. Use this when you need external or up-to-date
    information, or when you need to discover which pages to read. This only
    lists candidates - call 'web_fetch' with a URL to read the actual page.
    """
    if not query or not query.strip():
        return "No search query provided"
    query = query.strip()

    try:
        client = _get_client()
    except Exception as exc:
        return f"Web search failed: {_reason(exc)}"

    try:
        results = client.search(
            query,
            sources=["web"],
            limit=_SEARCH_LIMIT,
            timeout=_SEARCH_TIMEOUT_MS,
        )
    except Exception as exc:
        logger.warning("Firecrawl search failed: %s", type(exc).__name__)
        return f"Web search failed: {_reason(exc)}"

    items = list(getattr(results, "web", None) or [])
    if not items:
        return (
            f"No web results found for {query!r}. Try different wording, or use "
            "web_fetch if you already know a URL."
        )

    lines = [f"Web results for {query!r} ({len(items)} of at most {_SEARCH_LIMIT}):", ""]
    for position, item in enumerate(items[:_SEARCH_LIMIT], 1):
        url = _result_field(item, "url") or "(no url returned)"
        title = _result_field(item, "title") or "(no title)"
        description = _result_field(item, "description")
        lines.append(f"{position}. {title}")
        lines.append(f"   URL: {url}")
        if description:
            lines.append(f"   {description[:_MAX_DESCRIPTION_CHARS]}")
        lines.append("")
    lines.append("Call web_fetch(url) to read the full content of any result above.")
    return "\n".join(lines)


@tool
def web_fetch(url: str) -> str:
    """
    Fetch one specific web URL and return its content as Markdown, with the
    source URL and page title. Use this when you already know the URL - for
    example one returned by 'web_search'. Only that single page is fetched; no
    links are followed. Very long pages are truncated and say so.
    """
    if not url or not url.strip():
        return "No URL provided"
    url = url.strip()

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"Web fetch failed: {url!r} is not a valid http or https URL"

    try:
        client = _get_client()
    except Exception as exc:
        return f"Web fetch failed: {_reason(exc)}"

    try:
        document = client.scrape(
            url,
            formats=["markdown"],
            only_main_content=True,
            timeout=_FETCH_TIMEOUT_MS,
        )
    except Exception as exc:
        logger.warning("Firecrawl scrape failed: %s", type(exc).__name__)
        return f"Web fetch failed: {_reason(exc)}"

    metadata = getattr(document, "metadata", None)
    source = _field(metadata, "source_url") or url
    title = _field(metadata, "title")
    status = _field(metadata, "status_code")
    markdown = _field(document, "markdown")

    if not markdown.strip():
        detail = f" (page status {status})" if status else ""
        return f"Web fetch failed: no readable content returned for {source}{detail}"

    header = [f"Source: {source}"]
    if title:
        header.append(f"Title: {title}")
    if status:
        header.append(f"Page status: {status}")
    header.append("")

    body = markdown
    notice = ""
    if len(body) > _MAX_FETCH_CHARS:
        body = body[:_MAX_FETCH_CHARS]
        notice = (
            f"\n\n[TRUNCATED] Stopped at {_MAX_FETCH_CHARS} of {len(markdown)} "
            f"characters. The full page is still at {source}."
        )
    return "\n".join(header) + body + notice
