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
            "status=%s retryable=%s attempt=%s",
            self.primary_provider,
            self.primary_model,
            failure.category,
            failure.status_code,
            failure.retryable,
            attempt,
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
        for attempt in range(1, self.primary_attempts + 1):
            try:
                return self._result(
                    self.primary.invoke(messages, **call_kwargs),
                    self.primary_provider,
                    self.primary_model,
                )
            except Exception as exc:
                failure = self._should_fallback(exc)
                self._log_primary_failure(failure, attempt)
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
        for attempt in range(1, self.primary_attempts + 1):
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
