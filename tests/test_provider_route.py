"""Provider-route behaviour when a rate limit will not clear on its own.

The properties under test:

    A rate limit that resets imminently is still retried, a quota that does not
    is not retried at all, and neither the category nor the task-level
    retryable flag changes - only whether this particular route is worth
    another attempt.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from terminus.llm.fallback import FallbackChatModel
from terminus.tasks.errors import QUOTA_HORIZON_SECONDS, classify_failure

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------------
# helpers: the exact shapes providers send
# ---------------------------------------------------------------------------


def _status_error(
    status: int, headers: dict[str, str] | None = None
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", OPENROUTER_URL)
    response = httpx.Response(status, request=request, headers=headers or {})
    error = httpx.HTTPStatusError(
        f"Client error '{status}'", request=request, response=response
    )
    error.response = response
    return error


def _openrouter_daily_quota(reset_epoch_ms: int | None = None) -> httpx.HTTPStatusError:
    """The real 429 from OpenRouter when the free daily allowance is gone."""
    headers = {"x-ratelimit-limit": "50", "x-ratelimit-remaining": "0"}
    if reset_epoch_ms is not None:
        headers["x-ratelimit-reset"] = str(reset_epoch_ms)
    error = _status_error(429, headers)
    error.response._content = (
        b'{"error":{"message":"Rate limit exceeded: free-models-per-day. '
        b'Add 10 credits to unlock 1000 free model requests per day","code":429,'
        b'"metadata":{"limit_source":"openrouter_free_tier_daily"}}}'
    )
    return error


class _AlwaysFails:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def invoke(self, *_a, **_k):
        self.calls += 1
        raise self.exc

    async def ainvoke(self, *_a, **_k):
        return self.invoke()


class _Succeeds:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, *_a, **_k):
        self.calls += 1
        return "fallback answered"

    async def ainvoke(self, *_a, **_k):
        return self.invoke()


def _route(primary_exc: BaseException, fallback_ok: bool = True):
    primary = _AlwaysFails(primary_exc)
    fallback = _Succeeds() if fallback_ok else _AlwaysFails(primary_exc)
    model = FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_provider="openrouter",
        primary_model="poolside/laguna-s-2.1:free",
        fallback_provider="google_genai",
        fallback_model="gemini-3.5-flash-lite",
        primary_attempts=2,
        backoff_seconds=5.0,
    )
    return model, primary, fallback


@pytest.fixture(autouse=True)
def _forget_exhausted_providers():
    """Provider exhaustion is deliberate process state.

    It must not leak between tests: a route marked spent in one test would be
    skipped in the next, and every "is it retried?" assertion would silently
    read as zero attempts.
    """
    from terminus.llm import fallback

    fallback.clear_provider_exhaustion()
    yield
    fallback.clear_provider_exhaustion()


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_real_daily_quota_is_marked_exhausted():
    failure = classify_failure(_openrouter_daily_quota(1790640000000), provider="openrouter")
    assert failure.category == "rate_limit"
    assert failure.exhausted is True


def test_quota_exhausted_does_not_change_category_or_retryable():
    """The task-level retry policy keeps seeing an ordinary retryable rate limit."""
    plain = classify_failure(_status_error(429), provider="openrouter")
    quota = classify_failure(_openrouter_daily_quota(1790640000000), provider="openrouter")
    assert quota.category == plain.category == "rate_limit"
    assert quota.retryable is plain.retryable is True


def test_plain_429_is_not_exhausted():
    """A bare throttled 429 may clear in seconds, so it is still retried."""
    assert classify_failure(_status_error(429)).exhausted is False


def test_a_near_reset_is_not_exhausted():
    """A limit resetting inside the horizon is throttling, not a spent quota.

    Deliberately a plain body: the OpenRouter payload always names a *daily*
    allowance, which is a spent quota whatever the header says. The horizon
    signal has to be provable on its own.
    """
    import time as _time

    soon = int((_time.time() + 5) * 1000)
    assert classify_failure(
        _status_error(429, {"x-ratelimit-reset": str(soon)})
    ).exhausted is False


def test_daily_wording_wins_over_a_near_reset():
    """A body that names a daily allowance is a quota, reset time aside."""
    import time as _time

    soon = int((_time.time() + 5) * 1000)
    assert classify_failure(_openrouter_daily_quota(soon)).exhausted is True


def test_reset_in_seconds_is_understood():
    """Some providers send epoch seconds where OpenRouter sends milliseconds."""
    import time as _time

    later = int(_time.time() + 4000)
    assert classify_failure(_openrouter_daily_quota(later)).exhausted is True


def test_quota_detected_from_body_text_without_headers():
    error = _status_error(429)
    error.response._content = b'{"error":{"message":"Insufficient credits for this request"}}'
    assert classify_failure(error).exhausted is True


@pytest.mark.parametrize("status", [500, 502, 503, 408])
def test_server_errors_are_never_exhausted(status: int):
    """A 5xx says nothing about how long the provider stays down."""
    assert classify_failure(_status_error(status)).exhausted is False


def test_timeouts_are_never_exhausted():
    assert classify_failure(httpx.ReadTimeout("timed out")).exhausted is False


def test_authentication_failure_is_not_exhausted_and_stays_fatal():
    failure = classify_failure(_status_error(401))
    assert failure.category == "authentication"
    assert failure.retryable is False
    assert failure.exhausted is False


def test_invalid_model_is_not_exhausted():
    failure = classify_failure(
        RuntimeError("Error code: 404 - model not found")
    )
    assert failure.category == "invalid_model"
    assert failure.exhausted is False


def test_horizon_is_a_sane_duration():
    assert 0 < QUOTA_HORIZON_SECONDS <= 300


# ---------------------------------------------------------------------------
# routing: the quota must not be retried
# ---------------------------------------------------------------------------


def test_exhausted_route_is_not_retried_and_falls_back_immediately():
    model, primary, fallback = _route(_openrouter_daily_quota(1790640000000))
    started = time.time()
    result = model.invoke([{"role": "user", "content": "hi"}])
    elapsed = time.time() - started
    assert result.content == "fallback answered"
    assert primary.calls == 1, "a spent quota must not be retried"
    assert fallback.calls == 1
    assert elapsed < 1.0, f"the backoff sleep should not be paid, took {elapsed:.1f}s"


def test_plain_429_is_still_retried_before_falling_back():
    model, primary, fallback = _route(_status_error(429))
    model.invoke([{"role": "user", "content": "hi"}])
    assert primary.calls == 2
    assert fallback.calls == 1


def test_503_is_still_retried_before_falling_back():
    model, primary, fallback = _route(_status_error(503))
    model.invoke([{"role": "user", "content": "hi"}])
    assert primary.calls == 2
    assert fallback.calls == 1


def test_exhausted_route_on_the_async_path():
    model, primary, fallback = _route(_openrouter_daily_quota(1790640000000))
    started = time.time()
    result = asyncio.run(
        model.ainvoke([{"role": "user", "content": "hi"}])
    )
    assert time.time() - started < 1.0
    assert primary.calls == 1
    assert result.content == "fallback answered"


def test_async_plain_429_is_still_retried():
    model, primary, _ = _route(_status_error(429))
    asyncio.run(model.ainvoke([{"role": "user", "content": "hi"}]))
    assert primary.calls == 2


# ---------------------------------------------------------------------------
# termination
# ---------------------------------------------------------------------------


def test_chain_terminates_with_a_clear_error_when_everything_fails():
    """No hang: the whole route must end in a named, inspectable error."""
    model, primary, fallback = _route(_openrouter_daily_quota(1790640000000), fallback_ok=False)
    started = time.time()
    with pytest.raises(Exception) as caught:
        model.invoke([{"role": "user", "content": "hi"}])
    assert time.time() - started < 2.0
    assert type(caught.value).__name__ == "ProviderCallError"
    assert "openrouter" in str(caught.value) or "google_genai" in str(caught.value)
    assert primary.calls == 1
    assert fallback.calls == 1


def test_non_retryable_primary_still_raises_without_trying_the_fallback():
    """An auth failure is fatal; it must not be papered over by a fallback."""
    model, primary, fallback = _route(_status_error(401))
    with pytest.raises(Exception):
        model.invoke([{"role": "user", "content": "hi"}])
    assert primary.calls == 1
    assert fallback.calls == 0
