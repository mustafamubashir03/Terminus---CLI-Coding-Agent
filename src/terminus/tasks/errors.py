from __future__ import annotations

import asyncio
import re
import time
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
    exhausted: bool = False
    """This route is spent, not merely slow.

    Distinct from ``retryable``. A daily quota or an exhausted credit balance
    is still a rate limit, so the task-level retry policy keeps treating it as
    retryable and unchanged. What it must not do is retry the *same provider*:
    the quota does not come back in a few seconds, so the provider route has to
    give up on it immediately and move to the next one instead of burning a
    backoff sleep and another doomed request per call.
    """
    reset_in_seconds: float | None = None
    """How long until the provider says this limit lifts, when it says so.

    Provider-neutral: read from whatever the provider publishes (an epoch
    instant, a relative Retry-After, a reset counter), never from a hardcoded
    number for any one vendor. A provider that reports a horizon is skipped
    until it passes and becomes eligible again afterwards.
    """


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


_RETRY_IN_TEXT = re.compile(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*s", re.I)


def _retry_after_from_text(exc: BaseException) -> float | None:
    """A ``retry in 30s`` hint written into the message instead of a header."""
    for current in _exception_chain(exc):
        match = _RETRY_IN_TEXT.search(str(current))
        if match:
            return float(match.group(1))
    return None


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
    return _retry_after_from_text(exc)


QUOTA_HORIZON_SECONDS = 60.0
"""Beyond this, a rate limit is a quota rather than throttling.

A provider that will not serve you again for minutes or hours is not going to
serve you again after a five-second backoff. This is the line above which
retrying the same route is wasted work, and it is a duration, not a status
code, so it applies to any provider that reports one.
"""

# Words that mean the limit is over a long window or prepaid credits are gone,
# as opposed to "slow down".
_QUOTA_TEXT = re.compile(
    r"free[-_ ]?models?[-_ ]?per[-_ ]?(?:day|month)"
    r"|per[-_ ]?day|per[-_ ]?month|daily\s+(?:limit|quota)"
    r"|monthly\s+(?:limit|quota)"
    r"|insufficient[_ ]?(?:credits|quota|balance)"
    r"|exceeded\s+your\s+current\s+(?:credits|quota|budget)"
    r"|out\s+of\s+credits"
    r"|add\s+\d+\s*credits",
    re.I,
)


# Header names providers use to say when a limit lifts. All of these are
# consulted; none of them is tied to one vendor's naming.
_RESET_HEADERS = (
    "retry-after",
    "x-ratelimit-reset",
    "x-rate-limit-reset",
    "ratelimit-reset",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

_DURATION_UNITS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?", re.I)


def _parse_reset_value(raw: Any) -> tuple[str, float] | None:
    """Interpret a reset header value.

    Providers publish this three different ways, so all three are accepted:

    * an absolute epoch instant, in seconds or milliseconds;
    * a relative duration, either a bare number of seconds or a Go-style
      compound duration such as ``6h0m0s`` or ``1m30s``;
    * a reset counter, which some providers report as a bare epoch value.

    Returns ``("epoch" | "duration", seconds)`` or None.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        # Above ~1e11 the value is milliseconds, not seconds.
        if number > 1e11:
            return "epoch", number / 1000.0
        # A small bare number is a relative delay; a large one is an instant.
        if number > 1e9:
            return "epoch", number
        return "duration", max(0.0, number)
    total = 0.0
    matched = False
    for value, unit in _DURATION_PART.findall(text):
        if not value:
            continue
        matched = True
        total += float(value) * _DURATION_UNITS[(unit or "s").lower()]
    return ("duration", total) if matched else None


def _reset_delay_seconds(exc: BaseException) -> float | None:
    """Seconds until the provider says this limit resets, if it says so at all.

    Prefers the largest horizon any header reports: if one counter says 2s and
    another says 6h, the binding constraint is the 6h, and treating the limit
    as short-lived would send us back to a provider that cannot serve us.
    """
    now = time.time()
    best: float | None = None
    for current in _exception_chain(exc):
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            continue
        for name in _RESET_HEADERS:
            parsed = _parse_reset_value(headers.get(name))
            if parsed is None:
                continue
            kind, value = parsed
            delay = (value - now) if kind == "epoch" else value
            if delay <= 0:
                continue
            best = delay if best is None else max(best, delay)
    if best is not None:
        return best
    return _retry_after_from_text(exc)


def _body_texts(exc: BaseException) -> list[str]:
    """Every scrap of body-ish text attached to an exception chain.

    SDKs disagree about where the payload lives - some hang it off the
    exception, some only on the response object - so all of them are read, and
    a body that cannot be decoded is simply absent rather than fatal.
    """
    texts: list[str] = []
    for current in _exception_chain(exc):
        for attribute in ("body", "error"):
            value = getattr(current, attribute, None)
            if value is not None:
                texts.append(_redact(value))
        response = getattr(current, "response", None)
        if response is not None:
            try:
                texts.append(_redact(getattr(response, "text", "")))
            except Exception:
                # An unreadable, streamed or already-closed body carries no
                # extra signal; the headers and the status have spoken already.
                pass
    return texts


def _is_quota_exhausted(
    exc: BaseException, status: int | None, reset_in_seconds: float | None = None
) -> bool:
    """True when retrying this same provider cannot possibly help.

    Two independent signals, either of which is sufficient:

    * the provider advertises a reset further out than :data:`QUOTA_HORIZON_SECONDS`;
    * the body names a long-window limit or exhausted credits.
    """
    if reset_in_seconds is not None and reset_in_seconds > QUOTA_HORIZON_SECONDS:
        return True
    for text in _body_texts(exc):
        if text and _QUOTA_TEXT.search(text):
            return True
    for current in _exception_chain(exc):
        if _QUOTA_TEXT.search(str(current)):
            return True
    return False


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
            exhausted=failure.exhausted,
            # Carried over explicitly. Copying the field list by hand once lost
            # the reset horizon, which left a re-wrapped quota failure marked
            # ``exhausted`` with no idea when the route comes back.
            reset_in_seconds=failure.reset_in_seconds,
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
    # Only a rate limit can be a spent quota. A 5xx or a timeout says nothing
    # about how long the provider will stay unavailable, so it keeps retrying.
    reset_in = _reset_delay_seconds(exc)
    exhausted = category == "rate_limit" and _is_quota_exhausted(exc, status, reset_in)
    return FailureInfo(
        category=category,
        retryable=retryable,
        status_code=status,
        retry_after=_retry_after(exc),
        message=message,
        provider=provider,
        model=model,
        chain=tuple(type(item).__name__ for item in chain),
        exhausted=exhausted,
        reset_in_seconds=reset_in,
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
