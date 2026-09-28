from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from terminus.cache import cache_llm_client, get_cached_llm_client, llm_cache_key
from terminus.config import CONFIG, CONFIG_SOURCE, CONFIG_SOURCE_KIND
from terminus.env import load_project_env
from terminus.llm.fallback import FallbackChatModel
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def _install_threaded_aiohttp_resolver() -> None:
    import aiohttp
    from aiohttp.resolver import ThreadedResolver

    if getattr(aiohttp.TCPConnector, "_terminus_threaded_resolver", False):
        return
    original_init = aiohttp.TCPConnector.__init__

    def init_with_threaded_resolver(self, *args, **kwargs):
        if kwargs.get("resolver") is None:
            kwargs["resolver"] = ThreadedResolver()
        original_init(self, *args, **kwargs)

    aiohttp.TCPConnector.__init__ = init_with_threaded_resolver
    aiohttp.TCPConnector._terminus_threaded_resolver = True


_install_threaded_aiohttp_resolver()

_embedder_cache: dict = {}
_current_model_label = ""
_current_provider_label = ""
_active_llm_clients: list[Any] = []


class _OpenRouterChatModel(ChatOpenAI):
    @staticmethod
    def _reasoning_details(response: Any, index: int) -> Any:
        if isinstance(response, dict):
            choices = response.get("choices") or []
        else:
            choices = getattr(response, "choices", None) or []
        if index >= len(choices):
            return None
        choice = choices[index]
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        if isinstance(message, dict):
            return message.get("reasoning_details")
        return getattr(message, "reasoning_details", None)

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        result = super()._create_chat_result(response, generation_info)
        for index, generation in enumerate(result.generations):
            details = self._reasoning_details(response, index)
            if details is None:
                continue
            metadata = dict(getattr(generation.message, "response_metadata", None) or {})
            metadata["reasoning_details"] = details
            generation.message.response_metadata = metadata
            additional = dict(getattr(generation.message, "additional_kwargs", None) or {})
            additional["reasoning_details"] = details
            generation.message.additional_kwargs = additional
        return result

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        wire_messages = payload.get("messages") or []
        for index, message in enumerate(messages):
            if index >= len(wire_messages) or getattr(message, "type", "") != "ai":
                continue
            details = (getattr(message, "response_metadata", None) or {}).get(
                "reasoning_details"
            ) or (getattr(message, "additional_kwargs", None) or {}).get(
                "reasoning_details"
            )
            if details is not None:
                wire_messages[index]["reasoning_details"] = details
        return payload


def _load_dotenv() -> None:
    load_project_env(Path.cwd())


def _set_current_labels(model: str, provider: str) -> None:
    global _current_model_label, _current_provider_label
    _current_model_label = model
    _current_provider_label = provider


def get_current_model_label() -> str:
    return _current_model_label


def get_current_provider_label() -> str:
    return _current_provider_label


def _track(client: Any) -> None:
    if client is not None and not any(item is client for item in _active_llm_clients):
        _active_llm_clients.append(client)


async def _close_object(value: Any) -> None:
    if value is None:
        return
    for name in ("root_async_client", "async_client", "client", "primary", "fallback"):
        try:
            child = getattr(value, name)
        except Exception:
            continue
        if child is value:
            continue
        await _close_object(child)
    for name in ("aclose", "close"):
        close = getattr(value, name, None)
        if not callable(close):
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
            return
        except Exception as exc:
            logger.debug("LLM client close skipped: %s", type(exc).__name__)
            return


async def aclose_llm_clients() -> None:
    clients = list(_active_llm_clients)
    _active_llm_clients.clear()
    for client in clients:
        await _close_object(client)


def get_llm_config() -> dict[str, Any]:
    llm = CONFIG.get("llm", {})
    return {
        "provider": llm.get("provider", "openrouter"),
        "timeout": llm.get("request_timeout_seconds", 120),
        "max_retries": llm.get("max_retries", 0),
        "reasoning": llm.get("reasoning"),
        "include_reasoning": llm.get("include_reasoning", True),
        "streaming": llm.get("streaming", False),
        "route_max_attempts": llm.get("route_max_attempts", 2),
        "route_backoff_seconds": llm.get("route_backoff_seconds", 5),
    }


def _provider_endpoint(provider: str) -> str:
    return {
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "google_genai": "https://generativelanguage.googleapis.com/v1beta",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "openai": "https://api.openai.com/v1/chat/completions",
    }.get(provider.lower(), "provider-defined")


def _api_kind(provider: str) -> str:
    if provider.lower() == "openrouter":
        return "chat_completions"
    if provider.lower() in {"google_genai", "google"}:
        return "google_generate_content"
    return "provider-defined"


def _build_model_direct(model: str, provider: str, cfg: dict[str, Any]):
    _load_dotenv()
    provider_lower = provider.lower()

    if provider_lower == "fireworks":
        from langchain_fireworks import ChatFireworks

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is not set")
        client = ChatFireworks(
            model=f"accounts/fireworks/models/{model}",
            temperature=0,
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
        )
        _track(client)
        return client

    if provider_lower == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        client = ChatOpenAI(
            model=model,
            api_key=api_key,
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
        )
        _track(client)
        return client

    if provider_lower == "cerebras":
        from langchain_cerebras import ChatCerebras

        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY is not set")
        client = ChatCerebras(
            model=model,
            api_key=api_key,
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
        )
        _track(client)
        return client

    if provider_lower == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        client = ChatAnthropic(
            model=model,
            api_key=api_key,
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
        )
        _track(client)
        return client

    if provider_lower == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        extra_body: dict[str, Any] = {}
        if cfg.get("reasoning"):
            extra_body["reasoning"] = cfg["reasoning"]
        if cfg.get("include_reasoning", True):
            extra_body["include_reasoning"] = True
        client = _OpenRouterChatModel(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=cfg["timeout"],
            max_retries=cfg["max_retries"],
            temperature=0,
            streaming=cfg.get("streaming", False),
            use_responses_api=False,
            extra_body=extra_body or None,
        )
        _track(client)
        return client

    if provider_lower in {"google_genai", "google"}:
        from google import genai
        from google.genai import types
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")
        google_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        client = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            client=google_client,
            vertexai=False,
            request_timeout=cfg["timeout"],
            streaming=cfg.get("streaming", False),
        )
        _track(client)
        return client

    if provider_lower == "cohere":
        from langchain_cohere import ChatCohere

        class _CohereChatModel(ChatCohere):
            def bind_tools(self, tools, **kwargs):
                if kwargs.get("tool_choice") in ("any", "auto"):
                    kwargs["tool_choice"] = "REQUIRED"
                return super().bind_tools(tools, **kwargs)

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is not set")
        client = _CohereChatModel(
            model=model,
            api_key=api_key,
            timeout_seconds=cfg["timeout"],
            max_retries=cfg["max_retries"],
        )
        _track(client)
        return client

    raise ValueError(f"Unknown LLM provider: {provider}")


def _fallback_specs() -> list[tuple[str, str]]:
    llm = CONFIG.get("llm", {})
    raw = llm.get("fallbacks")
    if not isinstance(raw, list):
        raw = [llm["fallback"]] if llm.get("fallback") else []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider")
        model = item.get("model")
        if provider and model:
            pair = (str(provider), str(model))
            if pair not in result:
                result.append(pair)
    return result


def _cache_key(model: str, provider: str) -> str:
    fallbacks = _fallback_specs()
    suffix = "".join(f"|fallback={p}/{m}" for p, m in fallbacks)
    return f"{llm_cache_key(model, provider)}{suffix}"


def _with_fallback(
    primary: Any,
    provider: str,
    model: str,
    cfg: dict[str, Any],
) -> Any:
    routed = primary
    for fallback_provider, fallback_model in _fallback_specs():
        if fallback_provider == provider and fallback_model == model:
            continue
        try:
            alternate = _build_model_direct(fallback_model, fallback_provider, cfg)
        except Exception as exc:
            logger.warning(
                "Provider fallback disabled: provider=%s model=%s error=%s",
                fallback_provider,
                fallback_model,
                type(exc).__name__,
            )
            continue
        routed = FallbackChatModel(
            primary=routed,
            fallback=alternate,
            primary_provider=provider,
            primary_model=model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            primary_attempts=max(1, int(cfg.get("route_max_attempts", 2))),
            backoff_seconds=max(0.0, float(cfg.get("route_backoff_seconds", 5))),
        )
        _track(routed)
    return routed


def get_llm():
    llm = CONFIG.get("llm", {})
    return get_chat_model(
        llm.get("model", "poolside/laguna-s-2.1:free"), llm.get("provider")
    )


def get_chat_model(model: str, model_provider: str | None = None):
    from langchain.chat_models import init_chat_model

    cfg = get_llm_config()
    provider = model_provider or cfg["provider"]
    key = _cache_key(model, provider)
    cached = get_cached_llm_client(key)
    if cached is not None:
        _set_current_labels(model, provider)
        return cached

    _set_current_labels(model, provider)
    if provider.lower() in {"cohere", "openrouter", "google_genai", "google"}:
        client = _build_model_direct(model, provider, cfg)
    else:
        try:
            client = init_chat_model(
                model,
                model_provider=provider,
                timeout=cfg["timeout"],
                max_retries=cfg["max_retries"],
            )
        except (ValueError, ImportError) as exc:
            if "Unsupported provider" not in str(exc) and "requires the" not in str(exc):
                raise
            client = _build_model_direct(model, provider, cfg)
    _track(client)
    routed = _with_fallback(client, provider, model, cfg)
    return cache_llm_client(key, routed)


def _role_model(role: str) -> str:
    llm = CONFIG.get("llm", {})
    if role == "planner":
        return llm.get("planner_model", llm.get("model", ""))
    if role == "judge":
        return llm.get("judge_model", llm.get("model", ""))
    return llm.get("model", "")


def get_provider_diagnostics(role: str = "executor") -> dict[str, Any]:
    _load_dotenv()
    llm = CONFIG.get("llm", {})
    provider = str(llm.get("provider", ""))
    fallbacks = _fallback_specs()
    key_names = (
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "FIREWORKS_API_KEY",
        "CEREBRAS_API_KEY",
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "QDRANT_API_KEY",
        "CLUSTER_ENDPOINT",
    )
    return {
        "config_source": str(CONFIG_SOURCE) if CONFIG_SOURCE else None,
        "config_source_kind": CONFIG_SOURCE_KIND,
        "runtime_source": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "provider": provider,
        "role": role,
        "requested_model": _role_model(role),
        "models": {
            "ask": _role_model("ask"),
            "planner": _role_model("planner"),
            "executor": _role_model("executor"),
            "judge": _role_model("judge"),
        },
        "endpoint": _provider_endpoint(provider),
        "api_kind": _api_kind(provider),
        "streaming": bool(get_llm_config()["streaming"]),
        "fallback": (
            {"provider": fallbacks[0][0], "model": fallbacks[0][1]}
            if fallbacks
            else None
        ),
        "fallbacks": [
            {"provider": p, "model": m, "endpoint": _provider_endpoint(p)}
            for p, m in fallbacks
        ],
        "fallback_endpoint": _provider_endpoint(fallbacks[0][0]) if fallbacks else None,
        "vector_store": {
            "provider": CONFIG.get("vector_store", {}).get("provider"),
            "mode": CONFIG.get("rag", {}).get("mode"),
            "fallback": CONFIG.get("_runtime", {}).get("indexer_fallback"),
        },
        "retry_policy": {
            "sdk_max_retries": get_llm_config()["max_retries"],
            "route_max_attempts": get_llm_config()["route_max_attempts"],
            "route_backoff_seconds": get_llm_config()["route_backoff_seconds"],
        },
        "credential_presence": {name: bool(os.environ.get(name)) for name in key_names},
    }


def format_provider_diagnostics() -> str:
    data = get_provider_diagnostics()
    lines = [
        f"Provider: {data['provider']}",
        f"Requested model ({data['role']}): {data['requested_model']}",
        f"Models: {data['models']}",
        f"Endpoint: {data['endpoint']}",
        f"API kind: {data['api_kind']}",
        f"Streaming: {data['streaming']}",
        f"Configuration source: {data['config_source']} ({data['config_source_kind']})",
        f"Runtime source: {data['runtime_source']}",
        f"Fallback: {data['fallback']}",
        f"Fallback chain: {data['fallbacks']}",
        f"Fallback endpoint: {data['fallback_endpoint']}",
        f"Vector store: {data['vector_store']}",
        f"Retry policy: {data['retry_policy']}",
        f"Credential presence: {data['credential_presence']}",
    ]
    return "\n".join(lines)


def get_embedder():
    global _embedder_cache
    provider = CONFIG["embeddings"]["provider"]
    model = CONFIG["embeddings"]["model"]
    cache_key = f"{provider}:{model}"
    if cache_key in _embedder_cache:
        return _embedder_cache[cache_key]
    logger.info("Using Embedder Provider: %s, Model: %s", provider, model)
    if provider.lower() == "cerebras":
        from langchain_cerebras import CerebrasEmbeddings

        embedder = CerebrasEmbeddings(model=model)
    elif provider.lower() == "openai":
        from langchain_openai import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(model=model)
    elif provider.lower() == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(model_name=model)
    else:
        raise ValueError(f"Embedder Provider not found: {provider}")
    _embedder_cache[cache_key] = embedder
    return embedder
