"""Focused tests for the Firecrawl-backed web tools.

All network access is mocked: no test in this file contacts Firecrawl.
"""

import pytest

from terminus.tools import web_tools
from terminus.tools.web_tools import web_fetch, web_search


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeWebResult:
    def __init__(self, url, title, description):
        self.url = url
        self.title = title
        self.description = description
        self.position = 1


class FakeSearchData:
    def __init__(self, web):
        self.web = web


class FakeMetadata:
    def __init__(self, title="", source_url="", status_code=200):
        self.title = title
        self.source_url = source_url
        self.status_code = status_code


class FakeDocument:
    def __init__(self, markdown, metadata=None):
        self.markdown = markdown
        self.metadata = metadata or FakeMetadata()


class FakeClient:
    def __init__(self, search_result=None, document=None, search_exc=None, scrape_exc=None):
        self.search_result = search_result
        self.document = document
        self.search_exc = search_exc
        self.scrape_exc = scrape_exc
        self.search_calls = []
        self.scrape_calls = []

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        if self.search_exc:
            raise self.search_exc
        return self.search_result

    def scrape(self, url, **kwargs):
        self.scrape_calls.append((url, kwargs))
        if self.scrape_exc:
            raise self.scrape_exc
        return self.document


@pytest.fixture
def fake_client(monkeypatch):
    def _install(client):
        monkeypatch.setattr(web_tools, "_client", client)
        return client
    return _install


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(web_tools, "_client", None)


# ---------------------------------------------------------------------------
# configuration / key handling
# ---------------------------------------------------------------------------

def test_missing_key_produces_clean_tool_error(no_key, monkeypatch):
    monkeypatch.setattr(web_tools, "load_project_env", lambda *a, **k: None)
    out = web_search.invoke({"query": "python"})
    assert out.startswith("Web search failed:")
    assert "FIRECRAWL_API_KEY" in out
    assert "Traceback" not in out


def test_missing_key_produces_clean_fetch_error(no_key, monkeypatch):
    monkeypatch.setattr(web_tools, "load_project_env", lambda *a, **k: None)
    out = web_fetch.invoke({"url": "https://example.com"})
    assert out.startswith("Web fetch failed:")
    assert "FIRECRAWL_API_KEY" in out


def test_key_is_read_from_environment(monkeypatch):
    """_get_client must build the SDK client from FIRECRAWL_API_KEY."""
    monkeypatch.setattr(web_tools, "_client", None)
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-unit-test-key-123456")
    seen = {}

    class FakeFirecrawl:
        def __init__(self, api_key=None, **kwargs):
            seen["api_key"] = api_key
            seen["kwargs"] = kwargs

    import firecrawl
    monkeypatch.setattr(firecrawl, "Firecrawl", FakeFirecrawl)
    client = web_tools._get_client()
    assert isinstance(client, FakeFirecrawl)
    assert seen["api_key"] == "fc-unit-test-key-123456"
    # client is cached for the process
    assert web_tools._get_client() is client


def test_client_is_cached_not_rebuilt(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-unit-test-key-123456")
    monkeypatch.setattr(web_tools, "_client", None)
    builds = []

    class FakeFirecrawl:
        def __init__(self, api_key=None, **kwargs):
            builds.append(1)

    import firecrawl
    monkeypatch.setattr(firecrawl, "Firecrawl", FakeFirecrawl)
    web_tools._get_client()
    web_tools._get_client()
    web_tools._get_client()
    assert len(builds) == 1


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def test_search_passes_query_and_bounds(fake_client):
    client = fake_client(FakeClient(search_result=FakeSearchData([
        FakeWebResult("https://a.example", "A", "desc a"),
    ])))
    out = web_search.invoke({"query": "  python docs  "})
    query, kwargs = client.search_calls[0]
    assert query == "python docs"
    assert kwargs["limit"] == web_tools._SEARCH_LIMIT
    assert kwargs["sources"] == ["web"]
    # deliberately no scrapeOptions: search lists candidates only
    assert "scrape_options" not in kwargs
    assert "A" in out


def test_search_formats_title_url_description(fake_client):
    fake_client(FakeClient(search_result=FakeSearchData([
        FakeWebResult("https://docs.example/x", "Official Docs", "The real documentation."),
        FakeWebResult("https://blog.example/y", "Blog Post", None),
    ])))
    out = web_search.invoke({"query": "docs"})
    assert "1. Official Docs" in out
    assert "URL: https://docs.example/x" in out
    assert "The real documentation." in out
    assert "2. Blog Post" in out
    assert "URL: https://blog.example/y" in out
    # second result had no description -> no empty description line
    assert "web_fetch(url)" in out


def test_search_output_is_bounded(fake_client):
    fake_client(FakeClient(search_result=FakeSearchData([
        FakeWebResult(f"https://e{i}.example", f"T{i}", "d" * 5000)
        for i in range(12)
    ])))
    out = web_search.invoke({"query": "many"})
    assert len(out) < 12 * 5200          # far below 12 full descriptions
    assert out.count("URL:") == web_tools._SEARCH_LIMIT
    assert "5000" not in out.replace("d" * 5000, "")


def test_search_empty_results(fake_client):
    fake_client(FakeClient(search_result=FakeSearchData([])))
    out = web_search.invoke({"query": "nothing here"})
    assert out.startswith("No web results found")
    assert "web_fetch" in out


def test_search_none_web_attribute(fake_client):
    fake_client(FakeClient(search_result=FakeSearchData(None)))
    out = web_search.invoke({"query": "nothing"})
    assert out.startswith("No web results found")


def test_search_rejects_empty_query(fake_client):
    client = fake_client(FakeClient())
    assert web_search.invoke({"query": "   "}) == "No search query provided"
    assert client.search_calls == []


def test_search_handles_api_failure(fake_client, monkeypatch):
    from firecrawl.v2.utils.error_handler import RateLimitError, UnauthorizedError
    fake_client(FakeClient(search_exc=RateLimitError("429 too many requests")))
    out = web_search.invoke({"query": "x"})
    assert out.startswith("Web search failed:")
    assert "rate limited" in out

    fake_client(FakeClient(search_exc=UnauthorizedError("bad key")))
    out = web_search.invoke({"query": "x"})
    assert "authentication failed" in out
    assert "FIRECRAWL_API_KEY" in out


def test_search_handles_generic_network_failure(fake_client):
    fake_client(FakeClient(search_exc=OSError("connection reset by peer")))
    out = web_search.invoke({"query": "x"})
    assert out.startswith("Web search failed:")
    assert "connection reset" in out
    assert "Traceback" not in out


def test_search_never_leaks_the_key(fake_client, monkeypatch):
    from firecrawl.v2.utils.error_handler import UnauthorizedError
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-supersecretvalue9999")
    fake_client(FakeClient(search_exc=UnauthorizedError(
        "rejected key fc-supersecretvalue9999 at https://api.firecrawl.dev")))
    out = web_search.invoke({"query": "x"})
    assert "fc-supersecretvalue9999" not in out
    assert "[REDACTED]" in out


def test_search_redacts_any_fc_shaped_token(fake_client):
    from firecrawl.v2.utils.error_handler import BadRequestError
    fake_client(FakeClient(search_exc=BadRequestError("bad token fc-abcdef1234567890xyz")))
    out = web_search.invoke({"query": "x"})
    assert "fc-abcdef1234567890xyz" not in out
    assert "[REDACTED]" in out


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

def test_fetch_passes_url_and_markdown_only(fake_client):
    client = fake_client(FakeClient(document=FakeDocument(
        "# Title\n\nbody", FakeMetadata("T", "https://x.example", 200))))
    out = web_fetch.invoke({"url": " https://x.example/page "})
    url, kwargs = client.scrape_calls[0]
    assert url == "https://x.example/page"
    assert kwargs["formats"] == ["markdown"]
    assert kwargs["only_main_content"] is True
    assert "actions" not in kwargs       # no browser interaction
    assert "body" in out


def test_fetch_returns_markdown_with_source_metadata(fake_client):
    fake_client(FakeClient(document=FakeDocument(
        "# Heading", FakeMetadata("Page Title", "https://src.example/a", 200))))
    out = web_fetch.invoke({"url": "https://src.example/a"})
    assert "Source: https://src.example/a" in out
    assert "Title: Page Title" in out
    assert "Page status: 200" in out
    assert "# Heading" in out


def test_fetch_preserves_original_url_in_source(fake_client):
    fake_client(FakeClient(document=FakeDocument(
        "text", FakeMetadata("", "", 200))))
    out = web_fetch.invoke({"url": "https://original.example/keep-me"})
    assert "Source: https://original.example/keep-me" in out


def test_fetch_rejects_invalid_urls(fake_client):
    client = fake_client(FakeClient())
    for bad in ("not-a-url", "ftp://example.com", "file:///etc/passwd", "https://"):
        out = web_fetch.invoke({"url": bad})
        assert out.startswith("Web fetch failed:"), bad
        assert "not a valid" in out, bad
    assert client.scrape_calls == []


def test_fetch_rejects_empty_url(fake_client):
    client = fake_client(FakeClient())
    assert web_fetch.invoke({"url": "  "}) == "No URL provided"
    assert client.scrape_calls == []


def test_fetch_truncates_large_content(fake_client):
    huge = "A" * (web_tools._MAX_FETCH_CHARS + 5_000)
    fake_client(FakeClient(document=FakeDocument(huge, FakeMetadata("T", "https://x", 200))))
    out = web_fetch.invoke({"url": "https://x"})
    assert "[TRUNCATED]" in out
    assert f"of {len(huge)} characters" in out
    # url must not be dropped by truncation
    assert "https://x" in out
    assert len(out) < len(huge)


def test_fetch_does_not_truncate_small_content(fake_client):
    fake_client(FakeClient(document=FakeDocument("small", FakeMetadata("T", "https://x", 200))))
    out = web_fetch.invoke({"url": "https://x"})
    assert "[TRUNCATED]" not in out


def test_fetch_handles_empty_result(fake_client):
    fake_client(FakeClient(document=FakeDocument("   ", FakeMetadata("T", "https://x", 404))))
    out = web_fetch.invoke({"url": "https://x"})
    assert out.startswith("Web fetch failed:")
    assert "no readable content" in out
    assert "404" in out


def test_fetch_handles_missing_markdown_field(fake_client):
    fake_client(FakeClient(document=FakeDocument(None, FakeMetadata("T", "https://x", 200))))
    out = web_fetch.invoke({"url": "https://x"})
    assert out.startswith("Web fetch failed:")


def test_fetch_handles_api_failure(fake_client):
    from firecrawl.v2.utils.error_handler import WebsiteNotSupportedError
    fake_client(FakeClient(scrape_exc=WebsiteNotSupportedError("blocked")))
    out = web_fetch.invoke({"url": "https://x"})
    assert out.startswith("Web fetch failed:")
    assert "cannot access" in out


def test_fetch_handles_network_failure(fake_client):
    fake_client(FakeClient(scrape_exc=TimeoutError("read timed out")))
    out = web_fetch.invoke({"url": "https://x"})
    assert out.startswith("Web fetch failed:")
    assert "timed out" in out


def test_fetch_never_leaks_the_key(fake_client, monkeypatch):
    from firecrawl.v2.utils.error_handler import UnauthorizedError
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-topsupersecret1234")
    fake_client(FakeClient(scrape_exc=UnauthorizedError(
        "key fc-topsupersecret1234 rejected")))
    out = web_fetch.invoke({"url": "https://x"})
    assert "fc-topsupersecret1234" not in out
    assert "[REDACTED]" in out


# ---------------------------------------------------------------------------
# /ask wiring
# ---------------------------------------------------------------------------

def test_web_tools_are_in_the_ask_tool_list():
    import terminus.agent.factory as factory
    names = {tool.name for tool in factory.ASK_TOOLS}
    assert "web_search" in names
    assert "web_fetch" in names
    # still no destructive tools; run_command is present but policy-gated
    for forbidden in ("delete_file", "append_file", "run_in_directory"):
        assert forbidden not in names


def test_web_tools_are_not_exposed_to_the_plan_executor():
    """The /plan executor keeps its own tool list; web tools stay on /ask only."""
    import terminus.tasks.executor as executor
    source = open(executor.__file__, encoding="utf-8").read()
    assert "web_search" not in source
    assert "web_fetch" not in source
