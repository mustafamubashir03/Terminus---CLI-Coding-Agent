import json
from typing import ClassVar

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_openai import ChatOpenAI
from typer.testing import CliRunner

from terminus.cli_app import app
from terminus.llm._openrouter_model import OpenRouterChatModel
from terminus.llm.fallback import FallbackChatModel
from terminus.tasks.errors import FailureInfo, ProviderCallError, classify_failure
from terminus.tasks.executor import _verdict_from_text


class _RetryableFailureModel:
    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        raise TimeoutError("provider timeout")

    async def ainvoke(self, *args, **kwargs):
        raise TimeoutError("provider timeout")


class _PermanentFailureModel:
    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, *args, **kwargs):
        raise ValueError("invalid model identifier")

    async def ainvoke(self, *args, **kwargs):
        raise ValueError("invalid model identifier")


def test_openrouter_wire_contract_uses_chat_completions():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "poolside/laguna-s-2.1:free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = ChatOpenAI(
        model="poolside/laguna-s-2.1:free",
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        use_responses_api=False,
        streaming=False,
        max_retries=0,
        extra_body={
            "reasoning": {"effort": "low", "exclude": False},
            "include_reasoning": True,
        },
        http_client=client,
    )
    model.invoke("hello")
    path, payload = seen[0]
    assert path == "/api/v1/chat/completions"
    assert payload["model"] == "poolside/laguna-s-2.1:free"
    assert payload["stream"] is False
    assert payload["reasoning"] == {"effort": "low", "exclude": False}
    assert payload["include_reasoning"] is True


def test_openrouter_preserves_reasoning_details_on_continuation():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "poolside/laguna-s-2.1:free",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_details": [{"type": "reasoning.text", "text": "r"}],
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenRouterChatModel(
        model="poolside/laguna-s-2.1:free",
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        use_responses_api=False,
        max_retries=0,
        http_client=client,
    )
    first = model.invoke("hello")
    model.invoke([HumanMessage(content="hello"), first])
    assert first.response_metadata["reasoning_details"] == [
        {"type": "reasoning.text", "text": "r"}
    ]
    assert seen[1]["messages"][1]["reasoning_details"] == [
        {"type": "reasoning.text", "text": "r"}
    ]
    client.close()


def test_retryable_primary_switches_to_fallback():
    model = FallbackChatModel(
        primary=_RetryableFailureModel(),
        fallback=FakeListChatModel(responses=["fallback result"]),
        primary_provider="openrouter",
        primary_model="poolside/laguna-s-2.1:free",
        fallback_provider="google_genai",
        fallback_model="gemini-3.5-flash-lite",
        primary_attempts=1,
    )
    result = model.invoke("hello")
    assert result.content == "fallback result"
    assert result.response_metadata["terminus_provider"] == "google_genai"
    assert result.response_metadata["terminus_model"] == "gemini-3.5-flash-lite"


def test_nested_fallback_reaches_secondary_provider():
    inner = FallbackChatModel(
        primary=_RetryableFailureModel(),
        fallback=_RetryableFailureModel(),
        primary_provider="openrouter",
        primary_model="poolside/laguna-s-2.1:free",
        fallback_provider="google_genai",
        fallback_model="gemini-3.5-flash-lite",
        primary_attempts=1,
    )
    outer = FallbackChatModel(
        primary=inner,
        fallback=FakeListChatModel(responses=["cohere result"]),
        primary_provider="openrouter",
        primary_model="poolside/laguna-s-2.1:free",
        fallback_provider="cohere",
        fallback_model="command-r-plus-08-2024",
        primary_attempts=1,
    )
    result = outer.invoke("hello")
    assert result.content == "cohere result"
    assert result.response_metadata["terminus_provider"] == "cohere"


def test_permanent_primary_failure_does_not_fallback():
    model = FallbackChatModel(
        primary=_PermanentFailureModel(),
        fallback=FakeListChatModel(responses=["must not run"]),
        primary_provider="openrouter",
        primary_model="bad-model",
        fallback_provider="google_genai",
        fallback_model="gemini-3.5-flash-lite",
        primary_attempts=2,
    )
    with pytest.raises(ValueError, match="invalid model"):
        model.invoke("hello")


def test_failure_records_preserve_category_and_chain():
    failure = classify_failure(
        ProviderCallError(
            "google_genai",
            "gemini-3.5-flash-lite",
            classify_failure(TimeoutError("upstream timeout")),
            attempts=2,
        )
    )
    assert failure.category == "timeout"
    assert failure.retryable is True
    assert failure.provider == "google_genai"
    assert failure.model == "gemini-3.5-flash-lite"
    assert failure.chain[0] == "ProviderCallError"


def test_configuration_diagnostics_report_effective_models():
    from terminus.llm.factory import get_provider_diagnostics

    diagnostics = get_provider_diagnostics()
    assert diagnostics["provider"] == "openrouter"
    assert diagnostics["models"]["planner"] == "poolside/laguna-s-2.1:free"
    assert diagnostics["models"]["executor"] == "poolside/laguna-s-2.1:free"
    assert diagnostics["models"]["judge"] == "poolside/laguna-s-2.1:free"
    assert diagnostics["endpoint"].endswith("/chat/completions")
    assert diagnostics["fallback"]["provider"] == "google_genai"
    assert diagnostics["fallbacks"][1]["provider"] == "cohere"


def test_text_verdict_parser_accepts_passed_and_rejects_negation():
    assert _verdict_from_text("Verdict: passed\nReason: criteria met").passed is True
    assert _verdict_from_text("The result is not passed.").passed is False
    assert _verdict_from_text("No verdict was returned") is None



# --- streaming --------------------------------------------------------------
#
# FallbackChatModel had no _stream/_astream, so BaseChatModel.stream fell back to
# _generate and emitted the whole answer as one chunk. `streaming: true` bought
# nothing and the user saw the response appear in a single block at the end.

class _Streamer(BaseChatModel):
    """A model that honours the streaming contract, in pieces."""

    PARTS: ClassVar[tuple[str, ...]] = ("Hello", " streamed", " world")

    @property
    def _llm_type(self) -> str:
        return "test-streamer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="".join(self.PARTS)))
            ]
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for piece in self.PARTS:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for piece in self.PARTS:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


def _route():
    from terminus.llm.fallback import FallbackChatModel, clear_reported_routes

    clear_reported_routes()
    return FallbackChatModel(
        primary=_Streamer(), fallback=_Streamer(),
        primary_provider="openrouter", primary_model="m1",
        fallback_provider="groq", fallback_model="m2",
        primary_attempts=1, backoff_seconds=0,
    )


def test_wrapper_streams_in_pieces_rather_than_one_block():
    chunks = [c for c in _route().stream("hi") if c.content]
    assert len(chunks) == len(_Streamer.PARTS), (
        "the wrapper answered in one block; per-token streaming is broken"
    )
    assert "".join(c.content for c in chunks) == "".join(_Streamer.PARTS)


def test_wrapper_streams_asynchronously_too():
    import asyncio

    async def collect():
        return [c async for c in _route().astream("hi") if c.content]

    chunks = asyncio.run(collect())
    assert len(chunks) == len(_Streamer.PARTS)
    assert "".join(c.content for c in chunks) == "".join(_Streamer.PARTS)


def test_streamed_chunks_carry_the_route():
    chunks = list(_route().stream("hi"))
    meta = chunks[0].response_metadata
    assert meta.get("terminus_provider") == "openrouter"
    assert meta.get("terminus_model") == "m1"


def test_invoke_is_unaffected_by_streaming():
    result = _route().invoke("hi")
    assert result.content == "".join(_Streamer.PARTS)


def test_a_mid_stream_failure_does_not_switch_provider():
    """Half an answer is already out; splicing in another provider is nonsense."""

    class MidStreamBreak(_Streamer):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="par"))
            raise RuntimeError("connection dropped")

    from terminus.llm.fallback import FallbackChatModel, clear_reported_routes

    clear_reported_routes()
    route = FallbackChatModel(
        primary=MidStreamBreak(responses=[]), fallback=_Streamer(responses=[]),
        primary_provider="openrouter", primary_model="m1",
        fallback_provider="groq", fallback_model="m2",
        primary_attempts=1, backoff_seconds=0,
    )
    emitted = []
    with pytest.raises(Exception):
        for chunk in route.stream("hi"):
            emitted.append(chunk)
    assert len(emitted) == 1, "should yield what it had, then propagate the error"


# --- routing logs are not user-facing noise ---------------------------------

def test_routing_decisions_are_reported_once_not_once_per_layer(caplog):
    """Three nested fallback layers used to print the same line three times."""
    import logging

    from terminus.llm.fallback import clear_reported_routes

    clear_reported_routes()
    # One route for the whole loop: _route() resets the dedup set, so building a
    # new one per iteration would defeat the thing under test.
    route = _route()
    with caplog.at_level(logging.INFO, logger="terminus.llm.fallback"):
        for _ in range(3):
            route._report_route_once(
                "Answering from %s/%s instead of %s/%s", "groq", "m2", "openrouter", "m1"
            )
    matching = [r for r in caplog.records if "Answering from" in r.getMessage()]
    assert len(matching) == 1, f"announced {len(matching)} times, expected once"


def test_repeat_attempts_are_debug_not_warning(caplog):
    """A route attempt failing is routing working, not a problem for the user."""
    import logging

    from terminus.llm.fallback import clear_reported_routes

    clear_reported_routes()
    with caplog.at_level(logging.DEBUG, logger="terminus.llm.fallback"):
        _route()._log_primary_failure(
            FailureInfo(message="429", category="rate_limit", status_code=429,
                        retryable=True), 1
        )
    records = [r for r in caplog.records if "route attempt failed" in r.getMessage()]
    assert records
    assert all(r.levelno == logging.DEBUG for r in records), (
        "per-attempt routing detail is back at WARNING and will flood the terminal"
    )


# --- dev mode ---------------------------------------------------------------

def test_dev_flag_is_advertised():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--dev" in result.output


def test_dev_flag_on_agent_is_advertised():
    result = CliRunner().invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "--dev" in result.output


def test_dev_turns_on_debug_logging(monkeypatch):
    import logging

    import terminus.observability.logging as logging_module

    monkeypatch.setattr(logging_module, "configure_tracing", lambda: False)
    try:
        from typer.testing import CliRunner

        CliRunner().invoke(app, ["agent", "--dev", "-p", "x", "--provider", "groq"])
        assert logging.getLogger("terminus").level == logging.DEBUG
    finally:
        logging_module.set_log_level("WARNING")
