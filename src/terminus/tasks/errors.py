from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailureInfo:
    category: str
    retryable: bool
    status_code: int | None = None
    error_type: str | None = None
    provider_code: str | None = None
    upstream_provider: str | None = None
    retry_after: float | None = None
    message: str = ""
    provider: str | None = None
    model: str | None = None
    chain: tuple[str, ...] = ()


class JudgeRejectionError(ValueError):
    pass


class ProviderCallError(RuntimeError):
    def __init__(
        self,
        provider: str,
        model: str,
        failure: FailureInfo,
        attempts: int = 1,
    ) -> None:
        self.provider = provider
        self.model = model
        self.failure = failure
        self.attempts = attempts
        super().__init__(format_failure(failure, provider, model, attempts))


_NON_RETRYABLE_PREFIXES = ("[non-retryable] ", "[retryable] ")
_RETRYABLE_HTTP_STATUSES = frozenset((408, 425, 429, *range(500, 600)))
_RETRYABLE_NAMES = (
    "timeout",
    "ratelimit",
    "serviceunavailable",
    "badgateway",
    "internalserver",
    "connectionclosed",
    "serverdisconnected",
    "apierror",
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+\S+"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def _redact(value: Any) -> str:
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return re.sub(r"\s+", " ", text).strip()[:400]


def _exception_chain(exc: BaseException) -> list[BaseException]:
    result: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append(current)
        current = current.__cause__ or current.__context__
    return result


def _dict_value(data: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    for value in data.values():
        found = _dict_value(value, keys)
        if found is not None:
            return found
    return None


def _status_from_value(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("status_code", "status", "http_status", "code"):
            found = _dict_value(value, (key,))
            if isinstance(found, int) and 100 <= found <= 599:
                return found
            if isinstance(found, str) and found.isdigit() and 100 <= int(found) <= 599:
                return int(found)
    return None


def _http_status_from(exc: BaseException) -> int | None:
    for current in _exception_chain(exc):
        response = getattr(current, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
            if isinstance(status, int):
                return status
        for attr in ("status_code", "http_status", "status", "code"):
            status = getattr(current, attr, None)
            if isinstance(status, int) and 100 <= status <= 599:
                return status
        body = getattr(current, "body", None)
        status = _status_from_value(body)
        if status is not None:
            return status
        for arg in getattr(current, "args", ()):
            status = _status_from_value(arg)
            if status is not None:
                return status
        message = str(current)
        match = re.search(
            r"(?:error\s+code|http|status|code)\s*[:= ]\s*(\d{3})\b",
            message,
            re.I,
        )
        if match:
            return int(match.group(1))
    return None


def _metadata(exc: BaseException) -> dict[str, Any]:
    values: list[Any] = []
    for current in _exception_chain(exc):
        values.extend(
            [getattr(current, "body", None), getattr(current, "error", None)]
        )
        values.extend(getattr(current, "args", ()))
    error_type = None
    provider_code = None
    upstream_provider = None
    for value in values:
        error_type = error_type or _dict_value(value, ("error_type", "type"))
        provider_code = provider_code or _dict_value(value, ("provider_code", "code"))
        upstream_provider = upstream_provider or _dict_value(
            value, ("provider_name", "upstream_provider")
        )
    return {
        "error_type": str(error_type) if error_type is not None else None,
        "provider_code": str(provider_code) if provider_code is not None else None,
        "upstream_provider": str(upstream_provider)
        if upstream_provider is not None
        else None,
    }


def _retry_after(exc: BaseException) -> float | None:
    for current in _exception_chain(exc):
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass
    for current in _exception_chain(exc):
        match = re.search(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*s", str(current), re.I)
        if match:
            return float(match.group(1))
    return None


def _category(exc: BaseException, status: int | None, message: str) -> str:
    name = type(exc).__name__.lower()
    lower = message.lower()
    if isinstance(exc, JudgeRejectionError):
        return "verification_failure"
    if isinstance(exc, ConnectionError) and type(exc).__module__ == "builtins":
        return "unknown"
    if status == 401 or status == 403 or "authentication" in lower or "unauthorized" in lower or "api key" in lower and "invalid" in lower:
        return "authentication"
    if "invalid model" in lower or "model not found" in lower or "no longer available" in lower:
        return "invalid_model"
    if status == 404 and ("model" in lower or "not found" in lower):
        return "invalid_model"
    if status in (400, 404, 409, 422) or (status == 412 and "precondition" in lower):
        return "invalid_request"
    if status == 429 or "rate limit" in lower or "rate-limit" in lower or "temporarily rate" in lower:
        return "rate_limit"
    if status is not None and (status in _RETRYABLE_HTTP_STATUSES or status == 412):
        return "temporary_provider"
    if "timeout" in name or "timed out" in lower or isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    if any(token in name for token in ("connect", "network", "disconnected")):
        return "network"
    if "tool" in lower and ("call" in lower or "schema" in lower or "argument" in lower):
        return "tool_call_failure"
    if (
        isinstance(exc, TypeError)
        or "nonetype" in lower
        or "json" in lower
        or "parse" in lower
    ):
        return "response_parsing"
    if any(token in name for token in _RETRYABLE_NAMES):
        return "temporary_provider"
    return "unknown"


def _instructive_message(chain: list[BaseException]) -> str:
    """The most useful message in an exception chain.

    Scans from the innermost cause outwards and takes the first exception that
    actually says something, so:

      * a wrapper such as ``RuntimeError("worker failed") from
        ValueError("provider rejected the model id")`` reports the cause, which
        is the part that tells you what to fix;
      * a cancelled task, whose innermost frame is a textless
        ``CancelledError``, falls through to the TimeoutError underneath instead
        of persisting an empty error and losing its retryable classification.

    Falls back to the outermost message when nothing in the chain has text.
    """
    for item in reversed(chain):
        text = _redact(item)
        if text.strip():
            return text
    return _redact(chain[0]) if chain else ""


def classify_failure(
    exc: BaseException,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> FailureInfo:
    if isinstance(exc, ProviderCallError):
        failure = exc.failure
        return FailureInfo(
            category=failure.category,
            retryable=failure.retryable,
            status_code=failure.status_code,
            error_type=failure.error_type,
            provider_code=failure.provider_code,
            upstream_provider=failure.upstream_provider,
            retry_after=failure.retry_after,
            message=failure.message,
            provider=provider or exc.provider,
            model=model or exc.model,
            chain=(type(exc).__name__,) + failure.chain,
        )
    chain = _exception_chain(exc)
    status = _http_status_from(exc)
    message = _instructive_message(chain)
    category = _category(exc, status, message)
    metadata = _metadata(exc)
    retryable = category in {
        "rate_limit",
        "temporary_provider",
        "timeout",
        "network",
        "response_parsing",
        "verification_failure",
    }
    return FailureInfo(
        category=category,
        retryable=retryable,
        status_code=status,
        retry_after=_retry_after(exc),
        message=message,
        provider=provider,
        model=model,
        chain=tuple(type(item).__name__ for item in chain),
        **metadata,
    )


def is_retryable_error(exc: BaseException) -> bool:
    return classify_failure(exc).retryable


def format_failure(
    failure: FailureInfo,
    provider: str | None = None,
    model: str | None = None,
    attempts: int = 1,
) -> str:
    status = str(failure.status_code) if failure.status_code is not None else "n/a"
    upstream = failure.upstream_provider or "unknown"
    return (
        f"provider={provider or failure.provider or 'unknown'} "
        f"model={model or failure.model or 'unknown'} "
        f"category={failure.category} status={status} "
        f"retryable={str(failure.retryable).lower()} attempts={attempts} "
        f"upstream={upstream} chain={'>'.join(failure.chain) or type(failure).__name__}: "
        f"{failure.message}"
    )
