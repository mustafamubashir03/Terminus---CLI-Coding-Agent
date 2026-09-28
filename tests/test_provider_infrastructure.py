import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from terminus.llm.fallback import FallbackChatModel
from terminus.llm.factory import _OpenRouterChatModel
from terminus.tasks.errors import ProviderCallError, classify_failure
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
    model = _OpenRouterChatModel(
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

