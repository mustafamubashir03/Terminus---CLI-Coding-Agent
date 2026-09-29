"""The Groq provider, and provider-level exhaustion in the router.

The properties under test:

    Groq is built through the existing OpenAI-compatible abstraction with the
    documented base URL and its own credential; unsupported reasoning levels are
    never sent; a provider whose quota is spent is skipped outright rather than
    retried, and becomes eligible again when the horizon it published passes;
    and a nested multi-route chain does not skip healthy intermediate providers.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from terminus.llm import fallback as fb
from terminus.llm.fallback import FallbackChatModel
from terminus.tasks.errors import QUOTA_HORIZON_SECONDS, classify_failure

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _status_error(status: int, headers: dict[str, str] | None = None, body: bytes = b"") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {}, content=body)
    error = httpx.HTTPStatusError(f"Client error '{status}'", request=request, response=response)
    error.response = response
    return error


def _openrouter_daily_quota() -> httpx.HTTPStatusError:
    """The recorded OpenRouter free-tier daily exhaustion, 6h out."""
    import time as _t

    reset_ms = int((_t.time() + 6 * 3600) * 1000)
    return _status_error(
        429,
        {
            "x-ratelimit-limit": "50",
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(reset_ms),
        },
        body=b'{"error":{"message":"Rate limit exceeded: free-models-per-day. '
        b'Add 10 credits to unlock 1000 free model requests per day","code":429,'
        b'"metadata":{"limit_source":"openrouter_free_tier_daily"}}}',
    )


def _groq_rpd_exhausted() -> httpx.HTTPStatusError:
    """Groq style: Go-duration reset headers, long-window daily limit."""
    return _status_error(
        429,
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "6h0m0s",
            "retry-after": "3600",
        },
        body=b'{"error":{"message":"Rate limit reached for requests per day"}}',
    )


class _Fails:
    def __init__(self, exc, label="primary") -> None:
        self.exc, self.calls, self.label = exc, 0, label

    def invoke(self, *_a, **_k):
        self.calls += 1
        raise self.exc

    async def ainvoke(self, *_a, **_k):
        return self.invoke()


class _Works:
    def __init__(self, label="fallback", answer="answered") -> None:
        self.calls, self.label, self.answer = 0, label, answer

    def invoke(self, *_a, **_k):
        self.calls += 1
        return self.answer

    async def ainvoke(self, *_a, **_k):
        return self.invoke()


def _route(primary, fallback, *, primary_provider="openrouter", primary_model="m",
           attempts=2, backoff=5.0):
    return FallbackChatModel(
        primary=primary,
        fallback=fallback,
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider="groq",
        fallback_model="openai/gpt-oss-120b",
        primary_attempts=attempts,
        backoff_seconds=backoff,
    )


@pytest.fixture(autouse=True)
def _clean_exhaustion():
    """The exhaustion registry is process state; never let it leak between tests."""
    fb.clear_provider_exhaustion()
    yield
    fb.clear_provider_exhaustion()


# ---------------------------------------------------------------------------
# Groq provider construction
# ---------------------------------------------------------------------------


def test_groq_is_an_openai_compatible_provider():
    from terminus.llm.factory import _api_kind, _provider_endpoint

    assert _provider_endpoint("groq") == "https://api.groq.com/openai/v1/chat/completions"
    assert _api_kind("groq") == "chat_completions"


def test_groq_base_url_is_the_documented_one():
    from terminus.llm.factory import GROQ_BASE_URL

    assert GROQ_BASE_URL == "https://api.groq.com/openai/v1"


def test_groq_client_is_built_against_groq_not_openrouter(monkeypatch):
    """The credential and the base URL must both be Groq's."""
    from langchain_openai import ChatOpenAI

    from terminus.llm import factory

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    client = factory._build_model_direct(
        "openai/gpt-oss-120b", "groq", factory.get_llm_config()
    )
    assert isinstance(client, ChatOpenAI)
    assert str(client.openai_api_base) == "https://api.groq.com/openai/v1"
    assert client.model_name == "openai/gpt-oss-120b"
    assert "openrouter" not in str(client.openai_api_base)


def test_groq_requires_its_own_credential(monkeypatch):
    from terminus.llm import factory

    # _build_model_direct reloads the project .env, which would put the real
    # credential back. Neutralise that so the guard is actually exercised.
    monkeypatch.setattr(factory, "_load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        factory._build_model_direct(
            "openai/gpt-oss-120b", "groq", factory.get_llm_config()
        )


def test_groq_uses_chat_completions_not_the_responses_api(monkeypatch):
    from terminus.llm import factory

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    client = factory._build_model_direct(
        "openai/gpt-oss-120b", "groq", factory.get_llm_config()
    )
    assert client.use_responses_api is False


# ---------------------------------------------------------------------------
# reasoning mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,effort,expected",
    [
        ("openai/gpt-oss-120b", "low", "low"),
        ("openai/gpt-oss-120b", "high", "high"),
        ("openai/gpt-oss-20b", "medium", "medium"),
        # Qwen documents more levels than the GPT-OSS models accept.
        ("qwen/qwen3.8-27b", "max", "max"),
        # Not documented for GPT-OSS: must be dropped, never sent.
        ("openai/gpt-oss-120b", "max", None),
        ("openai/gpt-oss-120b", "xhigh", None),
        # Unknown model: nothing is guessed.
        ("some/unknown-model", "low", None),
    ],
)
def test_reasoning_effort_is_mapped_per_model(model, effort, expected):
    from terminus.llm.factory import _groq_reasoning_effort

    cfg = {"reasoning": {"effort": effort, "exclude": False}}
    assert _groq_reasoning_effort(cfg, model) == expected


def test_no_reasoning_setting_sends_nothing():
    from terminus.llm.factory import _groq_reasoning_effort

    assert _groq_reasoning_effort({}, "openai/gpt-oss-120b") is None
    assert _groq_reasoning_effort({"reasoning": None}, "openai/gpt-oss-120b") is None
    assert _groq_reasoning_effort({"reasoning": {"effort": ""}}, "openai/gpt-oss-120b") is None


def test_supported_effort_reaches_the_wire(monkeypatch):
    from terminus.llm import factory

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    cfg = dict(factory.get_llm_config())
    cfg["reasoning"] = {"effort": "low", "exclude": False}
    client = factory._build_model_direct("openai/gpt-oss-120b", "groq", cfg)
    assert client.extra_body.get("reasoning_effort") == "low"
    # The OpenRouter-shaped payload must not be forwarded to Groq.
    assert "reasoning" not in (client.extra_body or {})


def test_unsupported_effort_is_not_forwarded(monkeypatch):
    from terminus.llm import factory

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    cfg = dict(factory.get_llm_config())
    cfg["reasoning"] = {"effort": "max", "exclude": False}
    client = factory._build_model_direct("openai/gpt-oss-120b", "groq", cfg)
    assert "reasoning_effort" not in (client.extra_body or {})


# ---------------------------------------------------------------------------
# reset parsing (Groq's Go-style durations and OpenRouter's epoch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,kind,value",
    [
        ("1m26.4s", "duration", 86.4),
        ("810ms", "duration", 0.81),
        ("6h0m0s", "duration", 21600.0),
        ("1m30s", "duration", 90.0),
        ("30", "duration", 30.0),
        ("3600", "duration", 3600.0),
        ("1790640000000", "epoch", 1790640000.0),
        ("", None, None),
        (None, None, None),
    ],
)
def test_reset_values_are_parsed_in_every_documented_shape(raw, kind, value):
    from terminus.tasks.errors import _parse_reset_value

    parsed = _parse_reset_value(raw)
    if kind is None:
        assert parsed is None
    else:
        assert parsed == (kind, pytest.approx(value))


def test_groq_daily_exhaustion_is_detected_with_its_horizon():
    """Groq sends both retry-after (1h) and reset-requests (6h); the binding
    constraint is the longest one, so the shorter hint must not win."""
    failure = classify_failure(_groq_rpd_exhausted(), provider="groq")
    assert failure.category == "rate_limit"
    assert failure.exhausted is True
    assert failure.reset_in_seconds == pytest.approx(6 * 3600, abs=1.0)


def test_openrouter_daily_exhaustion_reports_its_horizon():
    failure = classify_failure(_openrouter_daily_quota(), provider="openrouter")
    assert failure.exhausted is True
    assert failure.reset_in_seconds == pytest.approx(6 * 3600, abs=60)


def test_a_plain_429_is_not_exhaustion():
    failure = classify_failure(_status_error(429), provider="groq")
    assert failure.exhausted is False
    assert failure.retryable is True


def test_5xx_is_not_exhaustion():
    assert classify_failure(_status_error(503)).exhausted is False
    assert classify_failure(_status_error(500)).exhausted is False


def test_timeout_is_not_exhaustion():
    assert classify_failure(httpx.ReadTimeout("timed out")).exhausted is False


def test_auth_failure_is_not_exhaustion():
    failure = classify_failure(_status_error(401), provider="groq")
    assert failure.category == "authentication"
    assert failure.exhausted is False
    assert failure.retryable is False


def test_exhaustion_preserves_category_and_retryable():
    """Task-level retry semantics must be untouched by route-level skipping."""
    plain = classify_failure(_status_error(429))
    spent = classify_failure(_groq_rpd_exhausted())
    assert spent.category == plain.category == "rate_limit"
    assert spent.retryable is plain.retryable is True


# ---------------------------------------------------------------------------
# provider-level skipping
# ---------------------------------------------------------------------------


def test_an_exhausted_provider_is_skipped_without_any_request():
    model = _route(_Fails(_groq_rpd_exhausted()), _Works(answer="from groq"))
    assert model.invoke([{"role": "user", "content": "hi"}]).content == "from groq"
    # First call learns it is spent...
    assert model.primary.calls == 1

    # ...and the very next call must not touch it at all.
    before = model.primary.calls
    assert model.invoke([{"role": "user", "content": "hi"}]).content == "from groq"
    assert model.primary.calls == before, "an exhausted provider was called again"


def test_first_call_spends_no_backoff_on_exhaustion():
    model = _route(_Fails(_groq_rpd_exhausted()), _Works(answer="from groq"))
    started = time.time()
    model.invoke([{"role": "user", "content": "hi"}])
    assert time.time() - started < 1.0


def test_a_transient_429_still_retries():
    model = _route(_Fails(_status_error(429)), _Works(answer="from groq"))
    model.invoke([{"role": "user", "content": "hi"}])
    assert model.primary.calls == 2, "a plain 429 must still be retried"


def test_a_5xx_still_retries():
    model = _route(_Fails(_status_error(503)), _Works(answer="from groq"))
    model.invoke([{"role": "user", "content": "hi"}])
    assert model.primary.calls == 2


def test_exhaustion_is_recorded_against_the_provider_not_the_model():
    model = _route(_Fails(_groq_rpd_exhausted()), _Works(), primary_provider="groq")
    model.invoke([{"role": "user", "content": "hi"}])
    assert "groq" in fb.exhaustion_snapshot()


def test_a_provider_becomes_eligible_again_after_its_horizon(monkeypatch):
    """A spent provider is retried once the horizon it published has passed."""
    import time as _time

    now = [1_000_000.0]
    monkeypatch.setattr(_time, "time", lambda: now[0])

    model = _route(_Fails(_groq_rpd_exhausted()), _Works(answer="from groq"),
                   primary_provider="groq")
    model.invoke([{"role": "user", "content": "hi"}])
    assert fb.provider_exhausted_until("groq") is not None

    # Jump past the recorded deadline: the provider is eligible again.
    now[0] += 6 * 3600 + 1
    assert fb.provider_exhausted_until("groq") is None
    before = model.primary.calls
    model.invoke([{"role": "user", "content": "hi"}])
    assert model.primary.calls == before + 1, "an elapsed horizon must be re-tested"


def test_an_unreported_horizon_still_becomes_eligible(monkeypatch):
    """A provider that says nothing about when it recovers must not stay
    disabled forever."""
    import time as _time

    now = [1_000_000.0]
    monkeypatch.setattr(_time, "time", lambda: now[0])
    fb.mark_provider_exhausted("cohere", None)
    assert fb.provider_exhausted_until("cohere") is not None
    now[0] += fb._UNKNOWN_HORIZON_SECONDS + 1
    assert fb.provider_exhausted_until("cohere") is None


def test_exhaustion_is_case_insensitive():
    fb.mark_provider_exhausted("Groq", 600)
    assert fb.provider_exhausted_until("groq") is not None
    assert fb.provider_exhausted_until("GROQ") is not None


def test_clearing_exhaustion():
    fb.mark_provider_exhausted("groq", 600)
    fb.clear_provider_exhaustion("groq")
    assert fb.provider_exhausted_until("groq") is None


def test_marking_without_a_provider_is_a_no_op():
    assert fb.mark_provider_exhausted(None, 60) is None
    assert fb.provider_exhausted_until(None) is None


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


def test_async_path_skips_an_exhausted_provider():
    async def run():
        model = _route(_Fails(_groq_rpd_exhausted()), _Works(answer="from groq"))
        first = await model.ainvoke([{"role": "user", "content": "hi"}])
        before = model.primary.calls
        second = await model.ainvoke([{"role": "user", "content": "hi"}])
        return first, second, before, model.primary.calls

    first, second, before, after = asyncio.run(run())
    assert first.content == second.content == "from groq"
    assert after == before, "async path called an exhausted provider again"


def test_async_path_still_retries_a_transient_failure():
    async def run():
        model = _route(_Fails(_status_error(429)), _Works(answer="from groq"))
        await model.ainvoke([{"role": "user", "content": "hi"}])
        return model.primary.calls

    assert asyncio.run(run()) == 2


def test_no_blocking_sleep_in_the_async_failure_path():
    """A sync time.sleep inside _agenerate would stall the whole event loop."""

    async def run():
        model = _route(_Fails(_status_error(429)), _Works(answer="ok"))
        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(tick())
        started = time.time()
        await model.ainvoke([{"role": "user", "content": "hi"}])
        elapsed = time.time() - started
        task.cancel()
        return ticks, elapsed

    ticks, elapsed = asyncio.run(run())
    # 5s of backoff happened, yet the loop kept running: the sleep is awaited,
    # not blocking. The bound is loose because asyncio timers fire marginally
    # early; the load-bearing assertion is the tick count.
    assert elapsed >= 4.5
    assert ticks > 50, "the event loop was blocked during the async backoff"


# ---------------------------------------------------------------------------
# nested multi-route chains
# ---------------------------------------------------------------------------


def test_a_nested_chain_does_not_skip_its_healthy_intermediate_route():
    """The outer route must not jump past a working middle provider.

    ``_with_fallback`` nests routes, and every level reports the *original*
    provider. Skipping the outer level because that provider is spent would
    also skip the intermediate fallback that still works.
    """
    inner_fallback = _Works(label="gemini", answer="from gemini")
    inner = _route(_Fails(_openrouter_daily_quota()), inner_fallback, primary_provider="openrouter")
    outer_fallback = _Works(label="cohere", answer="from cohere")
    outer = FallbackChatModel(
        primary=inner,
        fallback=outer_fallback,
        primary_provider="openrouter",
        primary_model="m",
        fallback_provider="cohere",
        fallback_model="c",
        primary_attempts=2,
        backoff_seconds=0.0,
    )

    result = outer.invoke([{"role": "user", "content": "hi"}])
    assert result.content == "from gemini", "a healthy intermediate route was skipped"
    assert inner_fallback.calls == 1
    assert outer_fallback.calls == 0


def test_the_chain_reaches_the_last_fallback_when_all_routes_are_spent():
    inner = _route(_Fails(_openrouter_daily_quota()), _Fails(_status_error(503), label="gemini"))
    last = _Works(label="groq", answer="from groq")
    outer = FallbackChatModel(
        primary=inner,
        fallback=last,
        primary_provider="openrouter",
        primary_model="m",
        fallback_provider="groq",
        fallback_model="openai/gpt-oss-120b",
        primary_attempts=1,
        backoff_seconds=0.0,
    )
    assert outer.invoke([{"role": "user", "content": "hi"}]).content == "from groq"


def test_shipped_fallback_order_ends_at_groq():
    """The configured chain must reach a non-OpenRouter provider at the end."""
    from terminus.config import CONFIG

    fallbacks = CONFIG.get("llm", {}).get("fallbacks") or []
    assert fallbacks, "no fallbacks configured"
    last = fallbacks[-1]
    assert last["provider"] == "groq"
    assert last["model"] == "openai/gpt-oss-120b"
    # Every fallback after the primary must be a different provider: retrying
    # inside one account's quota cannot help.
    assert all(f["provider"] != "openrouter" for f in fallbacks)


def test_horizon_is_a_sane_duration():
    assert 0 < QUOTA_HORIZON_SECONDS <= 300
