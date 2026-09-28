"""Tests for the LLM-client timeout / retryable-failure classification layer."""
import asyncio

import httpx

from terminus.tasks.errors import is_retryable_error


def test_httpx_readtimeout_is_retryable():
    assert is_retryable_error(httpx.ReadTimeout("the read operation timed out"))


def test_httpx_connecttimeout_is_retryable():
    assert is_retryable_error(httpx.ConnectTimeout("connect timed out"))


def test_builtin_timeout_error_is_retryable():
    assert is_retryable_error(TimeoutError("task agent stream timed out"))


def test_asyncio_timeout_error_is_retryable():
    assert is_retryable_error(asyncio.TimeoutError("timed out"))


def test_connectionerror_is_not_classified_retryable():
    # ConnectionError is deliberately NOT in the retryable set: it is too broad
    # and commonly wraps deterministic local failures.
    assert not is_retryable_error(ConnectionError("boom"))


def test_logical_failures_are_not_retryable():
    assert not is_retryable_error(ValueError("Judge rejected output for task A"))
    assert not is_retryable_error(ValueError("Agent returned an empty response"))
    assert not is_retryable_error(RuntimeError("agent exc"))


def _httpx_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.test/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"Client error '{status}'", request=request, response=response
    )


def test_http_5xx_status_is_retryable():
    assert is_retryable_error(_httpx_status_error(500))
    assert is_retryable_error(_httpx_status_error(502))
    assert is_retryable_error(_httpx_status_error(503))


def test_http_429_rate_limit_is_retryable():
    assert is_retryable_error(_httpx_status_error(429))


def test_http_408_and_412_are_retryable():
    assert is_retryable_error(_httpx_status_error(408))
    assert is_retryable_error(_httpx_status_error(412))


def test_http_4xx_client_errors_are_not_retryable():
    assert not is_retryable_error(_httpx_status_error(400))
    assert not is_retryable_error(_httpx_status_error(401))
    assert not is_retryable_error(_httpx_status_error(403))
    assert not is_retryable_error(_httpx_status_error(404))
    assert not is_retryable_error(_httpx_status_error(422))