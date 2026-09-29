"""Token usage + prompt-caching telemetry.

Every LLM call in the process is routed through langchain callbacks, so a
single ``BaseCallbackHandler`` hunched over ``on_llm_end`` can record
input/output token counts and, crucially, ``cached_tokens`` reported by
providers (OpenAI prompt caching, OpenRouter cached reasoning, etc.).

The accumulator lets us report, per session / plan run:
- how many tokens were billed vs. served from the prompt cache (savings %)
- latency and per-role breakdown implied by the delta

Costs are estimated with standard OpenAI pricing so "did caching actually
save money" is answered in concrete numbers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from terminus.observability.logging import get_logger
from terminus.tasks.errors import classify_failure

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    model: str = ""
    provider: str = ""
    kind: str = ""  # "ask" | "plan" | "executor" | "judge" | "planner" | "summarize" | "other"
    started: float = 0.0
    finished: float = 0.0
    error_category: str = ""
    status_code: int | None = None
    retryable: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    latency_seconds: float = 0.0

    @property
    def billed_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_tokens)


@dataclass
class UsageSummary:
    records: list[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord) -> None:
        self.records.append(rec)

    @property
    def total_input(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_cached(self) -> int:
        return sum(r.cached_tokens for r in self.records)

    @property
    def total_billed_input(self) -> int:
        return sum(r.billed_input_tokens for r in self.records)

    @property
    def savings_percent(self) -> float:
        total_in = self.total_input
        if total_in <= 0:
            return 0.0
        return (self.total_cached / total_in) * 100.0

    def by_kind(self, kind: str) -> list[CallRecord]:
        return [r for r in self.records if r.kind == kind]

    def to_table(self) -> str:
        lines = [
            "Call kind        | calls | input tok | cached | billed | output tok | est cost | cache %",
            "-" * 100,
        ]
        for kind in ("ask", "planner", "executor", "judge", "summarize", "other"):
            recs = self.by_kind(kind)
            if not recs:
                continue
            inp = sum(r.input_tokens for r in recs)
            cached = sum(r.cached_tokens for r in recs)
            billed = sum(r.billed_input_tokens for r in recs)
            out = sum(r.output_tokens for r in recs)
            pct = (cached / inp * 100.0) if inp else 0.0
            cost = estimate_cost(billed, out)
            lines.append(
                f"{kind:<16} | {len(recs):>3} | {inp:>9} | {cached:>6} | "
                f"{billed:>6} | {out:>10} | ${cost:>8.4f} | {pct:>5.1f}%"
            )
        total_in = self.total_input
        total_cached = self.total_cached
        total_billed = self.total_billed_input
        total_out = self.total_output
        total_pct = (total_cached / total_in * 100.0) if total_in else 0.0
        total_cost = estimate_cost(total_billed, total_out)
        lines.append("-" * 100)
        lines.append(
            f"{'TOTAL':<16} | {len(self.records):>3} | {total_in:>9} | {total_cached:>6} | "
            f"{total_billed:>6} | {total_out:>10} | ${total_cost:>8.4f} | {total_pct:>5.1f}%"
        )
        routes = sorted(
            {
                (r.provider or "unknown", r.model or "unknown")
                for r in self.records
            }
        )
        if routes:
            lines.append("Routes: " + ", ".join(f"{p}/{m}" for p, m in routes))
        return "\n".join(lines)


# Rough USD/M tokens (input / output), per model family.  Used for *estimation*
# only — swap in your actual billed prices when they differ.
_INPUT_PRICE_PER_M = {
    "gpt-4o": 2.50,
    "gpt-4o-mini": 0.15,
    "gpt-4.1": 2.00,
    "gpt-4.1-mini": 0.40,
    "gpt-5": 1.25,
    "gpt-5-mini": 0.25,
    "claude": 3.00,
    "command-r-plus": 3.00,
    "command-a": 2.50,
    "command": 2.00,
    "deepseek": 0.27,
    "default": 0.15,
}
_OUTPUT_PRICE_PER_M = {
    "gpt-4o": 10.00,
    "gpt-4o-mini": 0.60,
    "gpt-4.1": 8.00,
    "gpt-4.1-mini": 1.60,
    "gpt-5": 10.00,
    "gpt-5-mini": 2.00,
    "claude": 15.00,
    "command-r-plus": 15.00,
    "command-a": 12.50,
    "command": 10.00,
    "deepseek": 1.10,
    "default": 0.60,
}


def estimate_cost(billed_input_tokens: int, output_tokens: int, model: str = "") -> float:
    model_l = (model or "").lower()
    in_price = _INPUT_PRICE_PER_M.get("default", 0)
    out_price = _OUTPUT_PRICE_PER_M.get("default", 0)
    for key, price in _INPUT_PRICE_PER_M.items():
        if key != "default" and key in model_l:
            in_price = price
            break
    for key, price in _OUTPUT_PRICE_PER_M.items():
        if key != "default" and key in model_l:
            out_price = price
            break
    return (billed_input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


# ---------------------------------------------------------------------------
# Callback handler
# ---------------------------------------------------------------------------


class UsageCallbackHandler(BaseCallbackHandler):
    """Record token usage + prompt-cache hits from every completed LLM run."""

    def __init__(self, kind: str = "other"):
        self.kind = kind
        self.records: list[CallRecord] = []
        self._started: float | None = None

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._started = time.perf_counter()

    def _extract_usage(self, response: Any) -> tuple[int, int, int, str]:
        """Return (input_tokens, output_tokens, cached_tokens, model)."""
        model = ""
        inp = out = cached = 0

        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            model = llm_output.get("model_name", "") or model
            tu = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if isinstance(tu, dict):
                inp = int(tu.get("prompt_tokens", 0) or 0)
                out = int(tu.get("completion_tokens", 0) or 0)
                details = tu.get("prompt_tokens_details") or {}
                if isinstance(details, dict):
                    cached = int(details.get("cached_tokens", 0) or 0)

        # langchain-core >= 1.x puts usage on each generation's metadata too.
        if (not inp and not out) or not model:
            try:
                gens = response.generations or []
                if gens and gens[0]:
                    gen = gens[0][0]
                    meta = getattr(gen, "generation_info", None) or {}
                    if isinstance(meta, dict):
                        if not model:
                            model = meta.get("model_name", "")
                        um = meta.get("usage_metadata")
                        if isinstance(um, dict):
                            inp = int(um.get("input_tokens", 0) or 0)
                            out = int(um.get("output_tokens", 0) or 0)
                            idd = um.get("input_token_details", {}) or {}
                            cached = int(idd.get("cache_read", 0) or idd.get("cached_tokens", 0) or 0)
                    # Some providers (Cohere) attach usage on the message
                    # (AIMessage.usage_metadata) rather than in generation_info.
                    if not inp and not out:
                        msg = getattr(gen, "message", None)
                        um = getattr(msg, "usage_metadata", None) or {}
                        if isinstance(um, dict):
                            inp = int(um.get("input_tokens", 0) or 0)
                            out = int(um.get("output_tokens", 0) or 0)
                            idd = um.get("input_token_details", {}) or {}
                            cached = int(idd.get("cache_read", 0) or idd.get("cached_tokens", 0) or 0)
            except Exception:
                pass

        if not model:
            try:
                model = getattr(response, "model_name", "") or getattr(response, "model", "")
            except Exception:
                pass
        if not model:
            try:
                from terminus.llm.factory import get_current_model_label
                model = get_current_model_label()
            except Exception:
                pass
        return inp, out, cached, model

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        inp, out, cached, model = self._extract_usage(response)
        provider = ""
        metadata: dict[str, Any] = {}
        try:
            generations = response.generations or []
            if generations and generations[0]:
                message = getattr(generations[0][0], "message", None)
                metadata = dict(getattr(message, "response_metadata", None) or {})
        except Exception:
            metadata = {}
        provider = str(
            metadata.get("terminus_provider")
            or metadata.get("model_provider")
            or ""
        )
        model = str(
            metadata.get("terminus_model")
            or metadata.get("model_name")
            or model
            or ""
        )
        if not provider:
            try:
                from terminus.llm.factory import get_current_provider_label

                provider = get_current_provider_label()
            except Exception:
                provider = ""
        if provider == "openai":
            try:
                from terminus.llm.factory import get_current_provider_label

                if get_current_provider_label() == "openrouter":
                    provider = "openrouter"
            except Exception:
                pass
        finished = time.perf_counter()
        rec = CallRecord(
            model=model,
            provider=provider,
            kind=self.kind,
            started=self._started or 0.0,
            finished=finished,
            latency_seconds=max(0.0, finished - self._started)
            if self._started is not None
            else 0.0,
            input_tokens=inp,
            output_tokens=out,
            cached_tokens=cached,
        )
        self.records.append(rec)
        self._started = None

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        failure = classify_failure(error)
        try:
            from terminus.llm.factory import get_current_model_label, get_current_provider_label

            model = get_current_model_label()
            provider = get_current_provider_label()
        except Exception:
            model = ""
            provider = ""
        finished = time.perf_counter()
        self.records.append(
            CallRecord(
                model=model,
                provider=provider,
                kind=self.kind,
                started=self._started or 0.0,
                finished=finished,
                latency_seconds=max(0.0, finished - self._started)
                if self._started is not None
                else 0.0,
                error_category=failure.category,
                status_code=failure.status_code,
                retryable=failure.retryable,
            )
        )
        self._started = None

    def flush(self) -> list[CallRecord]:
        return self.records


_global_summary = UsageSummary()
_summary_lock = threading.Lock()

# Delegation lifecycle events, kept beside the model-usage summary so there is
# one place to look at what a run cost. Bounded: one entry per child, trimmed on
# insert, because a long session must not grow this without limit.
CHILD_EVENT_LIMIT = 200
_child_events: list[dict] = []


def record_child_event(
    *,
    parent_agent_id: str | None = None,
    child_agent_id: str = "",
    role: str = "",
    status: str = "",
    skills: list | None = None,
    tools: list | None = None,
    duration_seconds: float = 0.0,
    failure_category: str | None = None,
) -> None:
    """Record one delegated-child lifecycle event.

    Metadata only. No prompt, no task text and no child output: a subagent's
    answer routinely contains source code, and that does not belong in telemetry.

    This is an event log, not a second observability stack - it shares the
    module and the lock discipline of the usage summary beside it.
    """
    with _summary_lock:
        _child_events.append({
            "parent_agent_id": parent_agent_id or "",
            "child_agent_id": child_agent_id,
            "role": role,
            "status": status,
            "skills": list(skills or []),
            "tools": list(tools or []),
            "duration_seconds": round(float(duration_seconds), 3),
            "failure_category": failure_category,
        })
        if len(_child_events) > CHILD_EVENT_LIMIT:
            del _child_events[: len(_child_events) - CHILD_EVENT_LIMIT]


def get_child_events() -> list[dict]:
    """The recorded delegation events, oldest first."""
    with _summary_lock:
        return list(_child_events)


def clear_child_events() -> None:
    _summary_lock.acquire()
    try:
        _child_events.clear()
    finally:
        _summary_lock.release()


def record(_records: list[CallRecord], kind: str | None = None) -> None:
    """Absorb a handler's records into the global summary."""
    with _summary_lock:
        for r in _records:
            if kind:
                r.kind = kind
            _global_summary.add(r)


def get_summary() -> UsageSummary:
    return _global_summary
