"""Process-level caches for prompts and LLM clients, plus retry-with-backoff.

Two caches exist because two things are genuinely expensive to rebuild:
  * the composed /ask system prompt (it re-reads the project listing and
    TERMINUS.md), and
  * LLM client objects (connection pooling).

Keys must include the identity of whatever the value describes - today, the
project root. A bare constant key would hand Project A's listing to an agent
now working in Project B.

Provider construction and fallback belong to ``terminus.llm``; the retry helper
here is only for callers that own their own retry policy.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from terminus.observability.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt string cache
#
# Keys must include the identity of whatever the prompt describes (today: the
# project root), so two workspaces in one process never share a snapshot.
# ---------------------------------------------------------------------------
_prompt_cache: dict[str, str] = {}


def cache_prompt(key: str, value: str) -> str:
    """Store *value* under *key* and return it."""
    _prompt_cache[key] = value
    return value


def get_cached_prompt(key: str) -> str | None:
    """Return a previously cached prompt string, or None."""
    return _prompt_cache.get(key)


# ---------------------------------------------------------------------------
# LLM client cache  (reuses client objects for the same model+provider)
# ---------------------------------------------------------------------------
_llm_cache: dict[str, Any] = {}


def cache_llm_client(key: str, client: Any) -> Any:
    """Store an LLM client under *key* and return it."""
    _llm_cache[key] = client
    return client


def get_cached_llm_client(key: str) -> Any | None:
    """Return a previously cached LLM client, or None."""
    return _llm_cache.get(key)


def llm_cache_key(model: str, provider: str) -> str:
    """Deterministic cache key for a model+provider pair."""
    return f"{provider}/{model}"


async def retry_async(
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 4.0,
    **kwargs: Any,
) -> Any:
    from terminus.tasks.errors import classify_failure

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            failure = classify_failure(exc)
            if not failure.retryable or attempt == max_retries - 1:
                raise
            delay = failure.retry_after or base_delay * (2 ** attempt)
            logger.warning(
                "LLM call attempt %s/%s failed category=%s status=%s; retrying in %.0fs",
                attempt + 1,
                max_retries,
                failure.category,
                failure.status_code,
                delay,
            )
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry_async exhausted without an exception")
