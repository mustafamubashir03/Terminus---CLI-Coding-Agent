from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any

from terminus.cache import cache_llm_client, get_cached_llm_client, llm_cache_key
from terminus.config import CONFIG, CONFIG_SOURCE, CONFIG_SOURCE_KIND
from terminus.env import load_project_env
from terminus.llm import providers
from terminus.observability.logging import get_logger

# ``terminus.llm.fallback`` is imported inside ``_with_fallback`` rather than
# here. It pulls in langchain_core, which costs seconds, and nothing in this
# module needs it until a client is actually built - so importing it eagerly
# charged every process, including the ones that only read configuration to
# report diagnostics.

logger = get_logger(__name__)


def _install_threaded_aiohttp_resolver() -> None:
    """Use aiohttp's threaded DNS resolver on platforms where the default blocks.

    Windows' default resolver blocks the event loop on every DNS lookup, which
    stalls a concurrent agent turn behind a name lookup. The patch swaps in
    ``ThreadedResolver`` for any connector built after it runs.

    It is applied inside the builder rather than at import time: importing
    ``aiohttp`` costs a measurable fraction of a second, and only the Google
    client uses it. Patching ``TCPConnector.__init__`` affects connectors
    *created later*, so applying it just before the client is built is both
    sufficient and far cheaper.
    """
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


_embedder_cache: dict = {}
_current_model_label = ""
_current_provider_label = ""
_active_llm_clients: list[Any] = []


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


async def _aclose(obj: Any) -> None:
    """Close one object if it has a close hook, sync or async.

    Provider SDKs disagree on the name - ``aclose`` on an async client, ``close``
    on a sync one - so both are tried, and the first callable wins.
    """
    for name in ("aclose", "close"):
        close = getattr(obj, name, None)
        if not callable(close):
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            # Shutdown is best-effort: one client that will not close must not
            # strand the rest. The exception is not actionable to the user, so
            # it stays at debug level.
            logger.debug("LLM client close skipped: %s: %s", type(exc).__name__, exc)
        return


#: Attributes on a LangChain chat client that hold an HTTP transport we opened.
#: Named explicitly rather than discovered: a client we did not build is not our
#: object to walk, and guessing at attribute names is how the previous version
#: ended up recursing into arbitrary internals. ``FallbackChatModel`` needs no
#: entry - it wraps only clients that are tracked in their own right, so it is
#: never in the registry.
_TRANSPORT_ATTRS = ("root_async_client", "async_client", "client")


async def _close_llm_client(client: Any) -> None:
    """Close a provider client and the transport underneath it."""
    for attr in _TRANSPORT_ATTRS:
        transport = getattr(client, attr, None)
        if transport is not None and transport is not client:
            await _aclose(transport)
    await _aclose(client)


async def aclose_llm_clients() -> None:
    """Close every LLM client this process built.

    Called on shutdown. Clients are tracked individually at construction, so
    this walks a flat list rather than trying to rediscover a graph.
    """
    clients = list(_active_llm_clients)
    _active_llm_clients.clear()
    for client in clients:
        await _close_llm_client(client)


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
    """The provider's request URL, for diagnostics only.

    Read from the provider table so it cannot disagree with what the builder
    actually points a client at.
    """
    spec = providers.get(provider)
    return spec.endpoint if spec else "provider-defined"


def _api_kind(provider: str) -> str:
    spec = providers.get(provider)
    return spec.api_kind if spec else "provider-defined"


# Groq publishes reasoning levels per model, and a value a model does not
# accept is a hard 400 rather than something it quietly ignores. Discovered
# live from GET https://api.groq.com/openai/v1/models on 2026-09-28; only the
# levels documented for each model are ever sent, and an unknown model sends
# nothing rather than guessing.
GROQ_REASONING_EFFORTS: dict[str, frozenset[str]] = {
    "openai/gpt-oss-20b": frozenset({"low", "medium", "high"}),
    "openai/gpt-oss-120b": frozenset({"low", "medium", "high"}),
    "qwen/qwen3.8-27b": frozenset(
        {"none", "default", "minimal", "low", "medium", "high", "xhigh", "max"}
    ),
}

GROQ_BASE_URL = providers.get("groq").resolved_base_url()
"""Groq's documented OpenAI-compatible base URL.

Read from the provider table, so the URL a client is built against and the URL
reported in diagnostics cannot drift apart.

Groq is OpenAI-compatible, so the whole provider is a ``ChatOpenAI`` pointed at
this base URL - no separate SDK and no separate client type. Chat Completions is
used rather than Groq's Responses API because Chat Completions is what the rest
of Terminus is built on (tool calling, streaming, async), and Groq documents
Responses as beta.
"""


def _groq_reasoning_effort(cfg: dict[str, Any], model: str) -> str | None:
    """Map Terminus's configured reasoning effort onto a Groq level.

    Terminus already carries an OpenRouter-shaped ``reasoning.effort``. Groq
    wants a bare ``reasoning_effort`` string, and only some levels, so this
    returns None whenever the value is not documented for the selected model -
    the caller then sends no reasoning parameter at all rather than a rejected
    one.
    """
    reasoning = cfg.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    effort = str(reasoning.get("effort") or "").strip().lower()
    if not effort:
        return None
    supported = GROQ_REASONING_EFFORTS.get(model)
    if supported is None or effort not in supported:
        logger.warning(
            "Not sending reasoning_effort=%s to Groq model %s: not a documented "
            "level for that model. Supported: %s",
            effort,
            model,
            ", ".join(sorted(supported)) if supported else "unknown model",
        )
        return None
    return effort


def _require_key(provider: str, spec: providers.Provider) -> str:
    """The first of *provider*'s credential variables that is set.

    Raises naming every accepted variable, because "the key is missing" is only
    actionable if the user is told which key to set. A provider that does not
    require one still uses a key when it is set - a local server ignores it, and
    sending what the user configured keeps their setup working unchanged.
    """
    import os

    for name in spec.env_keys:
        value = os.environ.get(name)
        if value:
            return value
    if not spec.key_required:
        return "not-needed"
    accepted = " or ".join(spec.env_keys) if spec.env_keys else spec.name
    raise ValueError(f"{accepted} is not set")


def _build_openai_compatible(model: str, spec: providers.Provider, cfg: dict[str, Any]):
    """A ChatOpenAI pointed at an OpenAI-compatible endpoint.

    One construction path for every provider that speaks OpenAI Chat Completions:
    the vendor-specific part is only the body fields each one wants.

    The ``langchain_openai`` import is deferred into this function on purpose:
    it costs seconds, and a Cohere or Google deployment should not pay for an SDK
    it never constructs.
    """
    from langchain_openai import ChatOpenAI

    api_key = _require_key(spec.name, spec)
    extra_body: dict[str, Any] = {}
    if spec.name == "openrouter":
        if cfg.get("reasoning"):
            extra_body["reasoning"] = cfg["reasoning"]
        if cfg.get("include_reasoning", True):
            extra_body["include_reasoning"] = True
    elif spec.name == "groq":
        effort = _groq_reasoning_effort(cfg, model)
        if effort:
            extra_body["reasoning_effort"] = effort

    if spec.name == "openrouter":
        from terminus.llm._openrouter_model import OpenRouterChatModel

        cls = OpenRouterChatModel
    else:
        cls = ChatOpenAI
    return cls(
        model=model,
        api_key=api_key,
        base_url=spec.resolved_base_url() or None,
        timeout=cfg["timeout"],
        max_retries=cfg["max_retries"],
        temperature=0,
        streaming=cfg.get("streaming", False),
        use_responses_api=False,
        extra_body=extra_body or None,
    )


def _build_google(model: str, spec: providers.Provider, cfg: dict[str, Any]):
    from google import genai
    from google.genai import types
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Applied before the client is built, because patching TCPConnector.__init__
    # only affects connectors created afterwards. This is the only route that
    # pulls in aiohttp, so it is also the only place the patch is needed.
    _install_threaded_aiohttp_resolver()

    api_key = _require_key(spec.name, spec)
    google_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        client=google_client,
        vertexai=False,
        request_timeout=cfg["timeout"],
        streaming=cfg.get("streaming", False),
    )


def _build_cohere(model: str, spec: providers.Provider, cfg: dict[str, Any]):
    from terminus.llm._cohere_model import CohereChatModel

    return CohereChatModel(
        model=model,
        api_key=_require_key(spec.name, spec),
        timeout_seconds=cfg["timeout"],
        max_retries=cfg["max_retries"],
    )


def _build_langchain(model: str, spec: providers.Provider, cfg: dict[str, Any]):
    """The providers with a first-class LangChain integration, but no quirks.

    These four differ from each other only in SDK name and key, so the builder
    takes both from the table instead of carrying a near-identical branch each.
    """
    api_key = _require_key(spec.name, spec)
    module_name, class_name = {
        "fireworks": ("langchain_fireworks", "ChatFireworks"),
        "cerebras": ("langchain_cerebras", "ChatCerebras"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "openai": ("langchain_openai", "ChatOpenAI"),
    }[spec.name]
    cls = getattr(importlib.import_module(module_name), class_name)
    if spec.name == "fireworks":
        # Fireworks namespaces its models under the owning account.
        model = f"accounts/fireworks/models/{model}"
    return cls(
        model=model,
        api_key=api_key,
        timeout=cfg["timeout"],
        max_retries=cfg["max_retries"],
    )


#: name -> builder. A provider missing from here is constructed by
#: ``init_chat_model`` instead, which is why this is not the whole list.
_BUILDERS = {
    "openrouter": _build_openai_compatible,
    "groq": _build_openai_compatible,
    "ollama": _build_openai_compatible,
    "google_genai": _build_google,
    "google": _build_google,
    "cohere": _build_cohere,
    "fireworks": _build_langchain,
    "cerebras": _build_langchain,
    "anthropic": _build_langchain,
    "openai": _build_langchain,
}


def _build_model_direct(model: str, provider: str, cfg: dict[str, Any]):
    """Build a client for *provider* without consulting LangChain's factory.

    Used for the providers whose wire format or credentials need handling of our
    own, and as the fallback when ``init_chat_model`` does not recognise a
    provider name.
    """
    _load_dotenv()
    spec = providers.get(provider)
    if spec is None:
        raise ValueError(f"Unknown LLM provider: {provider}")
    builder = _BUILDERS.get(spec.name)
    if builder is None:
        raise ValueError(f"Unknown LLM provider: {provider}")
    client = builder(model, spec, cfg)
    _track(client)
    return client


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
    from terminus.llm.fallback import FallbackChatModel

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
        # The wrapper is deliberately not tracked. It holds only the clients
        # built above, and each of those was tracked when it was constructed, so
        # tracking the wrapper as well would mean closing the same transports
        # twice and would force shutdown to rediscover its own structure.
    return routed


def get_llm():
    llm = CONFIG.get("llm", {})
    return get_chat_model(
        llm.get("model", "poolside/laguna-s-2.1:free"), llm.get("provider")
    )


def get_chat_model(model: str, model_provider: str | None = None):
    """The client for *model*, routing through the fallback chain when there is one.

    A provider Terminus declares in its own table is built directly. Anything else
    goes through LangChain's factory, which is how Anthropic, OpenAI and the rest
    are supported without Terminus knowing their SDKs.
    """
    from langchain.chat_models import init_chat_model

    cfg = get_llm_config()
    provider = model_provider or cfg["provider"]
    key = _cache_key(model, provider)
    cached = get_cached_llm_client(key)
    if cached is not None:
        _set_current_labels(model, provider)
        return cached

    _set_current_labels(model, provider)
    if provider.lower() in providers.BUILT_LOCALLY:
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
            if _BUILDERS.get(provider.lower()) is None:
                raise
            logger.info(
                "init_chat_model could not build %s (%s); building it directly",
                provider, exc,
            )
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
    import time as _time

    from terminus.llm.fallback import provider_exhausted_until

    llm = CONFIG.get("llm", {})
    provider = str(llm.get("provider", ""))
    fallbacks = _fallback_specs()
    cfg = get_llm_config()

    def _status(name: str) -> str:
        deadline = provider_exhausted_until(name)
        if deadline is None:
            return "available"
        return f"exhausted, usable again in {max(0.0, deadline - _time.time()):.0f}s"

    def _route(name: str) -> dict[str, Any]:
        spec = providers.get(name)
        entry: dict[str, Any] = {
            "provider": name,
            "endpoint": _provider_endpoint(name),
            "api_kind": _api_kind(name),
            # Presence only. The secret itself is never placed in diagnostics.
            "api_key_configured": (
                any(os.environ.get(key) for key in spec.env_keys)
                if spec and spec.env_keys
                else None
            ),
            "status": _status(name),
        }
        if name.lower() == "groq":
            entry["base_url"] = spec.resolved_base_url()
            entry["available_models"] = sorted(GROQ_REASONING_EFFORTS)
        return entry

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
        "streaming": bool(cfg["streaming"]),
        # `fallback` and `fallback_endpoint` are the first entry of `fallbacks`,
        # kept as a convenience for callers that only care where Terminus goes
        # when the primary route is unavailable.
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
        "primary_route": _route(provider),
        "fallback_routes": [_route(p) for p, _m in fallbacks],
        "exhausted_providers": {
            name: _status(name)
            for name in sorted({provider, *(p for p, _ in fallbacks)})
            if provider_exhausted_until(name) is not None
        },
        "vector_store": {
            "provider": CONFIG.get("vector_store", {}).get("provider"),
            "mode": CONFIG.get("rag", {}).get("mode"),
            "fallback": CONFIG.get("_runtime", {}).get("indexer_fallback"),
        },
        "retry_policy": {
            "sdk_max_retries": cfg["max_retries"],
            "route_max_attempts": cfg["route_max_attempts"],
            "route_backoff_seconds": cfg["route_backoff_seconds"],
        },
        "credential_presence": {
            name: bool(os.environ.get(name))
            for name in providers.credential_env_names()
        },
    }


def format_provider_diagnostics() -> str:
    data = get_provider_diagnostics()
    return "\n".join(f"{key.replace('_', ' ')}: {value}" for key, value in data.items())


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
