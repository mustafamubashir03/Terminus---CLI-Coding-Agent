"""Running one agent turn: run the loop, decide how it ended, return the answer.

`handle_query` is the whole conversational path. It builds a policy, hands it to
`build_agent`, streams the graph inside an execution scope, and turns what came
back into a `TurnResult`.

Three things worth knowing before changing it.

**It owns terminal output.** The answer is streamed to stdout as the model
produces it, so callers must not also print the returned string.

**The stream is consumed twice over.** `messages` yields per-token deltas for
live output; `values` yields graph state so the final answer can be located. The
first is presentation, the second is the source of truth - which is why a
non-streaming provider still produces an answer, just all at once.

**Message history is not assembled here.** The checkpointer replays the thread on
the next call for the same id, which is what carries the conversation between
turns.

The loop itself belongs to LangGraph. This module decides when to stop caring
about it: what counts as an answer, what counts as a failure, and what happens
when the model wants to claim success it has not observed.
"""

import sys
from dataclasses import dataclass
from enum import Enum

from terminus.agent.factory import ask_policy, build_agent
from terminus.agent.observation import observation_pending
from terminus.execution import ask_context, execution_scope
from terminus.llm.text import message_text
from terminus.observability.logging import get_logger
from terminus.permissions import redact_secrets
from terminus.observability.usage_tracker import (
    ToolCallbackHandler,
    UsageCallbackHandler,
    record,
)

logger = get_logger(__name__)

_MODEL_NODE = "model"

_LIMIT_NOTICE = "Model call limits exceeded:"
"""Prefix of the notice ``ModelCallLimitMiddleware`` appends when it ends a run.

With ``exit_behavior="end"`` the framework writes this as the turn's last AI
message rather than raising, so it arrives looking exactly like an answer. Left
alone it would be reported as a completed turn whose answer is a budget warning.
Matched on its prefix because that string is what the user is shown, and it is
the only signal the middleware gives - there is no state key to read instead.
"""

# Tools whose result means the agent has looked at the effect of its own work,
# and the tools that mean something changed, are defined once in
# ``agent.observation`` and enforced by the middleware inside the graph. This
# module only reports what the graph already decided.


class Outcome(str, Enum):
    """How a turn ended."""

    DONE = "done"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TurnResult:
    """What one turn produced, and how it ended."""

    outcome: Outcome
    text: str = ""
    unverified: bool = False
    """True when the agent claimed completion after changing files without
    reading the result back or running anything."""
    files_changed: tuple[str, ...] = ()
    """Workspace-relative paths the turn's successful tool calls reported writing.

    Observed, not asserted: it is derived from the tool calls LangGraph actually
    executed, so it cannot claim a change that was refused or that never landed.
    Workspace-relative because that is how the model and the user both refer to
    these paths.
    """
    tool_failures: tuple[str, ...] = ()
    """``name: error`` for tool calls that raised, including workspace refusals.

    A tool that *returns* a refusal is not here - it ran, and its result was the
    refusal. This is the list of calls the runtime rejected.
    """

    def __str__(self) -> str:
        return self.text


def _extract_text(msg) -> str:
    return message_text(msg).strip()


def _stream_delta(msg) -> str:
    """Return the visible text one streamed chunk contributes.

    Deliberately does NOT strip: every chunk is a fragment of a single message,
    so trimming would eat the spaces between words. Non-text blocks (reasoning,
    tool-call payloads) are dropped so only the answer reaches the terminal.
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _write(text: str) -> None:
    """Write to stdout immediately so tokens appear while the model generates."""
    sys.stdout.write(text)
    sys.stdout.flush()


def _budget_message(model_calls: int) -> str:
    return (
        f"Model call limit reached after {model_calls} calls, with no final "
        "answer. Here is what was established so far, stated as observations rather "
        "than conclusions."
    )


async def run_turn(
    question: str,
    thread_id: str | None = None,
    *,
    interactive: bool = True,
    policy=None,
    kind: str = "ask",
    permission=None,
) -> TurnResult:
    """Run one agent turn and return what it produced.

    ``policy`` defaults to the /ask policy. ``permission`` defaults to the /ask
    permission policy. Both exist so this is the single execution path for every
    kind of agent rather than one per surface.
    """
    from terminus.agent.factory import ask_permission_policy

    agent = await build_agent(policy or ask_policy())
    context = permission or ask_context(ask_permission_policy(interactive))
    agent_config = {"configurable": {"thread_id": thread_id}}
    handler = UsageCallbackHandler(kind=kind)
    # Tool observation, on the same config as usage. The framework's on_tool_*
    # hooks fire for every call ToolNode makes, so this sees the run without a
    # second execution path - and it sees failures, which reading AIMessage
    # .tool_calls afterwards cannot.
    tools_seen = ToolCallbackHandler(kind=kind)
    model_calls = 0

    best: str | None = None
    history: list = []
    printed = False
    model_call: str | None = None
    model_call_text = ""
    step_unverified = False
    limit_notice = ""

    def finish(outcome: Outcome, text: str) -> TurnResult:
        """Wrap up one turn: publish usage, publish tool activity, shape the result."""
        record(handler.records, kind)
        return TurnResult(
            outcome,
            text=text,
            unverified=step_unverified,
            files_changed=tuple(tools_seen.files_changed()),
            tool_failures=tuple(
                f"{r.name}: {r.error}" for r in tools_seen.failures()
            ),
        )

    def flush_turn() -> None:
        nonlocal printed
        if model_call_text:
            _write("\n")
            printed = True

    try:
        with execution_scope(context):
            async for mode, step in agent.astream(
                {"messages": [{"role": "user", "content": question}]},
                config={**agent_config, "callbacks": [handler, tools_seen]},
                stream_mode=["values", "messages"],
            ):
                if mode == "messages":
                    chunk, meta = step
                    meta = meta or {}
                    if meta.get("langgraph_node") != _MODEL_NODE:
                        continue
                    # checkpoint_ns is unique per model call, so a change means
                    # the previous call ended. This keeps multi-turn tool loops
                    # and a provider fallback retry from printing as one run-on,
                    # and it is also where a *call* is counted - a streamed chunk
                    # is not a call, so the budget message must not count them.
                    call_id = meta.get("checkpoint_ns")
                    if call_id != model_call:
                        flush_turn()
                        if model_call is not None:
                            model_calls += 1
                        model_call = call_id
                        model_call_text = ""
                    delta = _stream_delta(chunk)
                    if delta:
                        _write(delta)
                        model_call_text += delta
                    continue

                history = step.get("messages") or []

                for msg in reversed(history):
                    if type(msg).__name__ != "AIMessage":
                        continue
                    if getattr(msg, "tool_calls", None):
                        # Text alongside tool calls is the model narrating what it
                        # is about to do, not an answer. Treating it as one would
                        # report a turn that ran out of budget mid-tool-call as a
                        # completed answer.
                        continue
                    content = _extract_text(msg)
                    if content.startswith(_LIMIT_NOTICE):
                        limit_notice = content
                        continue
                    if content:
                        best = content
                        break

                # Read from the graph state the middleware writes, so this reports
                # the rule that was actually enforced rather than a second copy of
                # it. Sampled on every state update and read once at the end,
                # because it describes how the turn *finished*: a model that
                # claimed completion, was sent back, and then read its own work has
                # verified it, even though the claim arrived a step earlier.
                step_unverified = observation_pending(step)
    except (KeyboardInterrupt, SystemExit):
        flush_turn()
        record(handler.records, kind)
        raise
    except Exception as exc:
        flush_turn()
        if best:
            # Partial output is better than none, and the traceback belongs in
            # the log: a user who sees a stack trace concludes the agent broke.
            logger.warning(
                "Agent stream failed after partial output; returning it: %s: %s",
                type(exc).__name__,
                exc,
            )
            return finish(Outcome.FAILED, best)
        logger.error("Agent turn failed: %s: %s", type(exc).__name__, exc)
        # The exception text reaches a human, and an exception raised anywhere
        # below this can have a provider's URL, a request header or an API key in
        # its message. It goes through the same redactor the shell tool uses rather
        # than through a second, differently-behaving scanner.
        return finish(Outcome.FAILED, f"Query failed: {redact_secrets(str(exc))}")

    flush_turn()

    if limit_notice:
        # The framework's own words for it, kept rather than replaced.
        return finish(Outcome.BUDGET_EXHAUSTED, limit_notice)

    if best:
        if not printed:
            _write(best + "\n")
        return finish(Outcome.DONE, best)

    # No final text. Either the budget ended the graph mid-loop or the model
    # produced nothing usable. Both are reported as what they are rather than
    # dressed up as a completed answer.
    if history:
        return finish(Outcome.BUDGET_EXHAUSTED, _budget_message(model_calls))
    return finish(Outcome.FAILED, "The agent produced no answer.")


def outcome_note(result: TurnResult) -> str:
    """One honest line about how a turn ended, or ``""`` when it simply worked.

    A turn that ended badly and a turn that ended well both used to reach the user
    as the same block of text, because :func:`handle_query` returns only the
    answer. A caller that cares can print this; a caller that does not is
    unaffected. An unverified answer is called out even when the outcome is DONE,
    because "it says it passed" and "it passed" are different claims and the
    model is the one that made the first one.
    """
    if result.outcome is not Outcome.DONE:
        return f"[{result.outcome.value}]"
    if result.unverified:
        return (
            "[unverified: the agent changed files and did not read the result "
            "back before answering]"
        )
    return ""


async def handle_query(question: str, thread_id: str | None = None,
                       interactive: bool = True) -> str:
    """Run one /ask turn and return its answer text.

    Owns the permission policy for the turn: the whole stream runs inside an
    execution scope, so the tools see /ask's authority and nothing else.

    Returns the text only. A caller that needs to know *how* the turn ended -
    failed, cancelled, out of budget, or answered without verifying its own work -
    should call :func:`run_turn` and read the :class:`TurnResult`, optionally
    rendering :func:`outcome_note`.
    """
    logger.info("Handling query: %s", question)
    result = await run_turn(question, thread_id, interactive=interactive)
    return str(result)
