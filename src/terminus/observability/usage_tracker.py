"""Token usage + prompt-caching telemetry, and tool-call observation.

Every LLM call in the process is routed through langchain callbacks, so a
single ``BaseCallbackHandler`` hunched over ``on_llm_end`` can record
input/output token counts and, crucially, ``cached_tokens`` reported by
providers (OpenAI prompt caching, OpenRouter cached reasoning, etc.).

The accumulator lets us report, per session / plan run:
- how many tokens were billed vs. served from the prompt cache (savings %)
- latency and per-role breakdown implied by the delta

Costs are estimated with standard OpenAI pricing so "did caching actually
save money" is answered in concrete numbers.

The same callbacks are how tool execution is observed. ``ToolCallbackHandler``
implements the framework's ``on_tool_start`` / ``on_tool_end`` / ``on_tool_error``
hooks rather than inventing an interception point, so it sees every tool call
LangGraph's ``ToolNode`` makes with no second dispatch path and nothing to keep
in step with the graph. LangChain hands each callback its own ``run_id`` and the
``parent_run_id`` it was invoked under, so a tool execution is already correlatable
with the model call that asked for it; both are recorded.

Nothing here executes a tool, resolves a tool name, or decides what a tool
returned. That is ``ToolNode``'s job and it stays there.
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


@dataclass
class CallRecord:
    model: str = ""
    provider: str = ""
    kind: str = ""  # "ask" | "plan" | "executor" | "judge" | "planner" | "summarize" | "other"
    run_id: str = ""
    """The framework run id of this model call, so a turn can be traced."""
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


#: How much of a tool's arguments a record keeps. Enough to say which file was
#: touched; not enough to persist a file's whole contents into telemetry.
_MAX_RECORDED_ARGS_CHARS = 400


@dataclass
class ToolRecord:
    """One tool call: what was asked, what happened, and how long it took.

    ``status`` distinguishes the three outcomes the model and the harness must
    be able to tell apart, using LangChain's own vocabulary rather than a
    Terminus-specific one:

    * ``success`` - the tool ran and returned a result. Note that a tool which
      *reports* a refusal as its result (``"Refused: ..."``) is still
      ``success``: it ran, and its result was that string.
    * ``error``   - the call was rejected or failed. LangGraph's ``ToolNode``
      turns a raised exception into a ``ToolMessage`` with
      ``status="error"``; this is that case, including a workspace violation.
    * ``cancelled`` - the run stopped before the tool finished.
    """

    name: str = ""
    tool_call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    """One of "success" | "error" | "cancelled"."""
    started: float = 0.0
    finished: float = 0.0
    duration_seconds: float = 0.0
    result: str = ""
    error: str = ""
    run_id: str = ""
    parent_run_id: str = ""
    mutated_paths: list[str] = field(default_factory=list)
    """Workspace-relative paths this call *named*, if it was a mutating one.

    Derived from the tool's own name and arguments rather than tracked
    separately, because a tool that mutates the filesystem already says which
    path it touched - and a second bookkeeping path could only ever disagree
    with it.

    Naming a path is not the same as changing it: a refused or deferred mutation
    names one and touches none. :meth:`ToolCallbackHandler.files_changed` is the
    answer to "what actually changed"; this is the answer to "what was aimed at".
    """

    @property
    def performed(self) -> bool:
        """Did this call change the workspace, as opposed to naming a path?

        False for a rejected call, and for a mutating call the permission guard
        declined. Both are decided by the guard that produced the wording, not
        re-guessed here.
        """
        from terminus.coordination import was_performed

        return self.ok and was_performed(self.result)

    @property
    def ok(self) -> bool:
        return self.status == "success"


#: Tools that change the workspace, and the argument naming the path they change.
#: Deliberately a plain table next to the record, not a registry concern: the
#: registry knows which tools exist, this knows which of them write.
#:
#: ``git_commit`` and ``git_branch`` are absent even though both change repository
#: state, because this table answers a narrower question: which *files* did the
#: call change. A commit records the tree as it already is and a branch creates a
#: ref; neither rewrites a file, so claiming a path for them would report work
#: that did not happen. ``git_checkout`` does rewrite the working tree, so it is
#: here. The harness's own mutated/observed signal lives in
#: ``agent.orchestrator`` and covers all three.
_MUTATING_TOOL_PATH_ARGS: dict[str, str] = {
    "write_file": "file_path",
    "edit_file": "file_path",
    "append_file": "file_path",
    "delete_file": "file_path",
    "run_command": "working_directory",
    "run_shell_command": None,
    "run_command_in_directory": "directory",
}

#: Mutating tools whose damage is the workspace itself rather than one argument
#: naming a path. Listed separately so the path table above keeps meaning "the
#: argument that names what changed".
_WHOLE_WORKSPACE_TOOLS = frozenset({"run_shell_command", "run_command_in_directory", "git_checkout"})



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


class UsageCallbackHandler(BaseCallbackHandler):
    """Record token usage + prompt-cache hits from every completed LLM run."""

    def __init__(self, kind: str = "other", run_id: str = ""):
        self.kind = kind
        self.run_id = run_id
        """Optional caller-supplied id for the whole execution.

        LangChain already assigns every run its own id and parents tool runs
        under the model run that requested them, so correlation works without
        this. It exists so one *execution* - one /ask turn, one worker attempt,
        one child - can be named and found in the logs, rather than only a
        chain of anonymous ids.
        """
        self.records: list[CallRecord] = []
        self._started: float | None = None
        self._last_run_id: str = ""

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._started = time.perf_counter()
        self._last_run_id = str(kwargs.get("run_id") or "")
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
            run_id=str(kwargs.get("run_id") or self._last_run_id or ""),
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
                run_id=str(kwargs.get("run_id") or self._last_run_id or ""),
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


def _bounded_args(args: Any) -> dict[str, Any]:
    """Tool arguments, truncated so telemetry cannot become a content store.

    A file path, a pattern and a command are what matter. A 40 KB blob of
    generated source passed as ``content`` is not, and must not end up in a
    process-lifetime log.
    """
    if not isinstance(args, dict):
        return {}
    bounded: dict[str, Any] = {}
    for key, value in args.items():
        text = value if isinstance(value, str) else str(value)
        if len(text) > _MAX_RECORDED_ARGS_CHARS:
            text = f"{text[:_MAX_RECORDED_ARGS_CHARS]}...[truncated]"
        bounded[str(key)] = text
    return bounded


def _parse_inputs(input_str: str) -> Any:
    """Recover arguments from the string form, when ``inputs`` was not supplied.

    ``on_tool_start`` is given both; this is only reached if a provider or
    wrapper collapses them, and a string is still better than losing the record.
    """
    import json

    try:
        return json.loads(input_str)
    except (TypeError, ValueError):
        return {}


def mutated_paths(name: str, args: Any) -> list[str]:
    """Workspace paths a tool call *aims at*, relative to the workspace.

    A pure function of the tool's name and arguments, so the answer is the same
    whether it is asked before or after the call ran, and a caller that wants to
    know what a call touched does not need to have been listening. Whether the
    call actually changed them is a separate question - see
    :attr:`ToolRecord.performed`.
    """
    if not isinstance(args, dict):
        return []
    if name in _MUTATING_TOOL_PATH_ARGS:
        raw = args.get("working_directory") if name == "run_command" else args.get(
            _MUTATING_TOOL_PATH_ARGS[name] or ""
        )
        if raw and str(raw).strip():
            from terminus.workspace import relative_to_workspace

            return [relative_to_workspace(str(raw))]
        return []
    if name in _WHOLE_WORKSPACE_TOOLS:
        from terminus.workspace import project_root, relative_to_workspace

        raw = args.get("directory")
        return [relative_to_workspace(str(raw)) if raw else relative_to_workspace(
            project_root()
        )]
    return []


class ToolCallbackHandler(BaseCallbackHandler):
    """Observe every tool call LangGraph executes, through framework callbacks.

    This is observation and nothing else. It resolves no tool name, executes no
    tool and rewrites no result: ``ToolNode`` already does all three, and a second
    path to any of them is a second thing that can be wrong.

    Instances are per-execution. Records live on the handler, so the caller that
    created it - an /ask turn, a worker attempt, a child agent - reads its own
    tool activity directly and correlates it with its own usage records through
    the shared ``kind``.
    """

    def __init__(self, kind: str = "other") -> None:
        self.kind = kind
        self.records: list[ToolRecord] = []
        self._pending: dict[str, ToolRecord] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        # LangChain puts the tool's name in `serialized` and the parsed arguments
        # in `inputs`; `name` arrives as None and `input` is not passed at all.
        run_id = str(kwargs.get("run_id") or "")
        name = str((serialized or {}).get("name") or kwargs.get("name") or "")
        args = kwargs.get("inputs")
        if not isinstance(args, dict):
            args = _parse_inputs(input_str)
        rec = ToolRecord(
            name=name,
            tool_call_id=str(kwargs.get("tool_call_id") or ""),
            args=_bounded_args(args),
            started=time.perf_counter(),
            run_id=run_id,
            parent_run_id=str(kwargs.get("parent_run_id") or ""),
        )
        rec.mutated_paths = mutated_paths(rec.name, rec.args)
        self._pending[run_id] = rec

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        rec = self._take(kwargs)
        if rec is None:
            return
        rec.finished = time.perf_counter()
        rec.duration_seconds = max(0.0, rec.finished - rec.started)
        rec.result = _bounded_result(output)
        # A tool that raised and had handle_tool_error set does not reach
        # on_tool_error: BaseTool.run turns it into the tool's result with
        # status="error" and reports a normal end. Reading the status is how one
        # observable channel covers both ways a tool can fail.
        if getattr(output, "status", None) == "error":
            rec.status = "error"
            rec.error = rec.result or "tool call failed"
        else:
            rec.status = "success"
        self.records.append(rec)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        rec = self._take({**kwargs, "run_id": run_id})
        if rec is None:
            return
        rec.finished = time.perf_counter()
        rec.duration_seconds = max(0.0, rec.finished - rec.started)
        rec.status = "error"
        failure = classify_failure(error)
        rec.error = f"{type(error).__name__}: {error}"[:_MAX_RECORDED_ARGS_CHARS]
        if failure.category:
            rec.error = f"{rec.error} [{failure.category}]"
        self.records.append(rec)

    def _take(self, kwargs: dict[str, Any]) -> ToolRecord | None:
        rec = self._pending.pop(str(kwargs.get("run_id") or ""), None)
        if rec is None and self._pending:
            # A callback that reached us without the id we keyed on still belongs
            # to the call in flight; dropping it would lose the only record of a
            # tool that failed.
            rec = self._pending.pop(next(iter(self._pending)))
        return rec

    def flush(self) -> list[ToolRecord]:
        return self.records

    def files_changed(self) -> list[str]:
        """Workspace paths a mutating call actually changed, in first-seen order.

        Excludes anything the permission guard declined. A refused write names a
        path and changes nothing, and a harness that reported it as changed would
        go looking for a verification that cannot exist.
        """
        seen: list[str] = []
        for rec in self.records:
            if not rec.performed:
                continue
            for path in rec.mutated_paths:
                if path not in seen:
                    seen.append(path)
        return seen

    def failures(self) -> list[ToolRecord]:
        return [r for r in self.records if not r.ok]


def _bounded_result(output: Any) -> str:
    """A tool's result, as text and truncated to the same bound as its arguments.

    ``ToolMessage`` and ``Command`` outputs are read through ``content`` rather
    than ``repr``, because what the model actually read is the thing worth
    recording.
    """
    content = getattr(output, "content", None)
    if content is None and hasattr(output, "messages"):
        messages = getattr(output, "messages", None) or []
        content = getattr(messages[-1], "content", "") if messages else ""
    if content is None:
        content = output if isinstance(output, str) else str(output)
    text = content if isinstance(content, str) else str(content)
    if len(text) > _MAX_RECORDED_ARGS_CHARS:
        return f"{text[:_MAX_RECORDED_ARGS_CHARS]}...[truncated]"
    return text


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
