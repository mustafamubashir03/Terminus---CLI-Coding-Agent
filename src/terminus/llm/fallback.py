from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from terminus.observability.logging import get_logger
from terminus.tasks.errors import FailureInfo, ProviderCallError, classify_failure

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider-level exhaustion
#
# A rate limit and a spent account are different things. A throttle clears in
# seconds, so retrying is right. A daily quota, a monthly cap or an empty credit
# balance does not clear on a backoff, so retrying the same provider is wasted
# work repeated on every model call in an agent loop.
#
# The registry is keyed by provider and stores an absolute deadline, so it holds
# no vendor-specific knowledge: a provider that publishes a reset horizon is
# skipped until that horizon passes, then becomes eligible again on its own. A
# provider that reports no horizon is still skipped for this process, because we
# have no evidence it will recover, and the next process re-tests it.
#
# Process-scoped on purpose, matching the existing module-level caches here.
# ---------------------------------------------------------------------------

_UNKNOWN_HORIZON_SECONDS = 900.0
"""How long to stay away from a provider that says it is spent but not for how long.

Long enough to cover the rest of a working session, short enough that a
transient mis-report does not disable a provider for the day. The point is not
this number; it is that the registry prefers a real reported horizon whenever
the provider gives one.
"""

_exhausted_providers: dict[str, float] = {}


def provider_exhausted_until(provider: str | None) -> float | None:
    """Absolute epoch second this provider is unusable until, if it is.

    Returns None when the provider is eligible, which is also the answer once a
    recorded deadline has passed - the entry is dropped on read so it cannot go
    stale and keep a recovered provider disabled.
    """
    if not provider:
        return None
    deadline = _exhausted_providers.get(provider.lower())
    if deadline is None:
        return None
    import time

    if deadline <= time.time():
        _exhausted_providers.pop(provider.lower(), None)
        return None
    return deadline


def mark_provider_exhausted(
    provider: str | None, reset_in_seconds: float | None = None
) -> float | None:
    """Record that a provider is spent, and until when.

    ``reset_in_seconds`` is whatever the provider reported. When it reported
    nothing, a bounded default is used so the entry cannot disable the provider
    indefinitely. Returns the deadline for logging and tests.
    """
    if not provider:
        return None
    import time

    seconds = (
        reset_in_seconds
        if reset_in_seconds is not None and reset_in_seconds > 0
        else _UNKNOWN_HORIZON_SECONDS
    )
    deadline = time.time() + seconds
    _exhausted_providers[provider.lower()] = deadline
    return deadline


def clear_provider_exhaustion(provider: str | None = None) -> None:
    """Forget exhaustion state. All providers when *provider* is None."""
    if provider is None:
        _exhausted_providers.clear()
    else:
        _exhausted_providers.pop(provider.lower(), None)


def exhaustion_snapshot() -> dict[str, float]:
    """Current provider deadlines, for diagnostics and tests."""
    return dict(_exhausted_providers)


class FallbackChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: Any
    fallback: Any
    primary_provider: str = ""
    primary_model: str = ""
    fallback_provider: str = ""
    fallback_model: str = ""
    primary_attempts: int = 2
    backoff_seconds: float = 5.0

    @property
    def _llm_type(self) -> str:
        return "terminus-provider-route"

    @property
    def model_name(self) -> str:
        return self.primary_model

    @property
    def model(self) -> str:
        return self.primary_model

    def _result(self, response: Any, provider: str, model: str) -> ChatResult:
        if isinstance(response, ChatResult):
            result = response
        elif isinstance(response, AIMessage):
            result = ChatResult(generations=[ChatGeneration(message=response)])
        elif isinstance(response, dict):
            result = ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=response))]
            )
        else:
            result = ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=str(response)))]
            )
        for generation in result.generations:
            message = generation.message
            metadata = dict(getattr(message, "response_metadata", None) or {})
            metadata.setdefault("terminus_provider", provider)
            metadata.setdefault("terminus_model", model)
            message.response_metadata = metadata
        return result

    def _kwargs(
        self,
        stop: list[str] | None,
        run_manager: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        call_kwargs = dict(kwargs)
        if stop is not None:
            call_kwargs["stop"] = stop
        return call_kwargs

    def _fallback_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        result = dict(kwargs)
        choice = result.get("tool_choice")
        if self.primary_provider.lower() == "openrouter" and choice in {
            "any",
            "required",
        }:
            result["tool_choice"] = "auto"
        return result

    def _bind(
        self,
        tools: list[Any],
        primary_kwargs: dict[str, Any],
        fallback_kwargs: dict[str, Any],
    ) -> "FallbackChatModel":
        primary = self.primary.bind_tools(tools, **primary_kwargs)
        fallback = self.fallback.bind_tools(tools, **fallback_kwargs)
        return FallbackChatModel(
            primary=primary,
            fallback=fallback,
            primary_provider=self.primary_provider,
            primary_model=self.primary_model,
            fallback_provider=self.fallback_provider,
            fallback_model=self.fallback_model,
            primary_attempts=self.primary_attempts,
            backoff_seconds=self.backoff_seconds,
        )

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "FallbackChatModel":
        primary_kwargs = self._fallback_kwargs(kwargs)
        fallback_kwargs = dict(kwargs)
        if fallback_kwargs.get("tool_choice") in {"any", "required"}:
            fallback_kwargs["tool_choice"] = "auto"
        return self._bind(tools, primary_kwargs, fallback_kwargs)

    def _should_fallback(self, exc: BaseException) -> FailureInfo:
        return classify_failure(
            exc,
            provider=self.primary_provider,
            model=self.primary_model,
        )

    def _log_primary_failure(self, failure: FailureInfo, attempt: int) -> None:
        logger.warning(
            "Provider route attempt failed: provider=%s model=%s category=%s "
            "status=%s retryable=%s exhausted=%s attempt=%s",
            self.primary_provider,
            self.primary_model,
            failure.category,
            failure.status_code,
            failure.retryable,
            failure.exhausted,
            attempt,
        )

    def _skip_exhausted_primary(self) -> bool:
        """True if this route's primary is known-spent, so it must not be called.

        Only consulted when the primary is a real model client. When it is
        another FallbackChatModel (the nested multi-route chain), that inner
        model already performs this check for its own provider, and skipping the
        whole chain here would also skip the intermediate fallbacks that are
        still perfectly healthy.
        """
        if isinstance(self.primary, FallbackChatModel):
            return False
        deadline = provider_exhausted_until(self.primary_provider)
        if deadline is None:
            return False
        import time

        logger.warning(
            "Skipping exhausted provider: provider=%s model=%s usable_again_in=%.0fs",
            self.primary_provider,
            self.primary_model,
            max(0.0, deadline - time.time()),
        )
        return True

    def _note_exhausted(self, failure: FailureInfo) -> None:
        deadline = mark_provider_exhausted(
            self.primary_provider, failure.reset_in_seconds
        )
        import time

        if deadline is not None:
            logger.warning(
                "Provider exhausted: provider=%s model=%s usable_again_in=%.0fs",
                self.primary_provider,
                self.primary_model,
                max(0.0, deadline - time.time()),
            )

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        import time

        call_kwargs = self._kwargs(stop, run_manager, kwargs)
        primary_exhausted = self._skip_exhausted_primary()
        for attempt in range(1, self.primary_attempts + 1):
            if primary_exhausted:
                break
            try:
                return self._result(
                    self.primary.invoke(messages, **call_kwargs),
                    self.primary_provider,
                    self.primary_model,
                )
            except Exception as exc:
                failure = self._should_fallback(exc)
                self._log_primary_failure(failure, attempt)
                if failure.exhausted:
                    # The quota does not return within this call. Retrying it
                    # would spend a backoff sleep and another doomed request,
                    # on every model call, to arrive at the same answer.
                    self._note_exhausted(failure)
                    break
                if not failure.retryable:
                    raise
                if attempt < self.primary_attempts:
                    time.sleep(failure.retry_after or self.backoff_seconds * attempt)
        logger.warning(
            "Provider fallback selected: from=%s/%s to=%s/%s",
            self.primary_provider,
            self.primary_model,
            self.fallback_provider,
            self.fallback_model,
        )
        try:
            return self._result(
                self.fallback.invoke(messages, **call_kwargs),
                self.fallback_provider,
                self.fallback_model,
            )
        except Exception as exc:
            failure = classify_failure(
                exc,
                provider=self.fallback_provider,
                model=self.fallback_model,
            )
            raise ProviderCallError(
                self.fallback_provider,
                self.fallback_model,
                failure,
                attempts=self.primary_attempts + 1,
            ) from exc

    async def _agenerate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        call_kwargs = self._kwargs(stop, run_manager, kwargs)
        primary_exhausted = self._skip_exhausted_primary()
        for attempt in range(1, self.primary_attempts + 1):
            if primary_exhausted:
                break
            try:
                response = await self.primary.ainvoke(messages, **call_kwargs)
                return self._result(
                    response,
                    self.primary_provider,
                    self.primary_model,
                )
            except Exception as exc:
                failure = self._should_fallback(exc)
                self._log_primary_failure(failure, attempt)
                if failure.exhausted:
                    self._note_exhausted(failure)
                    break
                if not failure.retryable:
                    raise
                if attempt < self.primary_attempts:
                    await asyncio.sleep(
                        failure.retry_after or self.backoff_seconds * attempt
                    )
        logger.warning(
            "Provider fallback selected: from=%s/%s to=%s/%s",
            self.primary_provider,
            self.primary_model,
            self.fallback_provider,
            self.fallback_model,
        )
        try:
            response = await self.fallback.ainvoke(messages, **call_kwargs)
            return self._result(
                response,
                self.fallback_provider,
                self.fallback_model,
            )
        except Exception as exc:
            failure = classify_failure(
                exc,
                provider=self.fallback_provider,
                model=self.fallback_model,
            )
            raise ProviderCallError(
                self.fallback_provider,
                self.fallback_model,
                failure,
                attempts=self.primary_attempts + 1,
            ) from exc
