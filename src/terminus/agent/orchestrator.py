import sys

from terminus.agent.factory import ask_permission_policy, build_agent
from terminus.execution import ask_context, execution_scope
from terminus.llm.factory import get_llm
from terminus.llm.text import message_text
from terminus.observability.logging import get_logger
from terminus.observability.usage_tracker import UsageCallbackHandler, record

logger = get_logger(__name__)

_AI_MSG_NAME = "AIMessage"
_MODEL_NODE = "model"


def _extract_text(msg) -> str:
    return message_text(msg).strip()


def _stream_delta(msg) -> str:
    """Return the visible text one streamed chunk contributes.

    Deliberately does NOT strip: every chunk is a fragment of a single message,
    so trimming would eat the spaces between words.  Non-text blocks (reasoning,
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


async def handle_query(question: str, thread_id: str | None = None,
                       interactive: bool = True) -> str:
    """Handle a single /ask query and return the final answer.

    The agent graph is consumed with two stream modes at once:

    * ``messages`` ΓÇö per-token deltas from the model node.  These are written to
      stdout as they arrive, so the user sees the answer being produced.  Tool
      output and middleware-internal calls (e.g. the summarizer's own model call)
      are filtered out.
    * ``values``  ΓÇö the full graph state after each superstep, used exactly as
      before to locate the last real answer.

    ``messages`` is a superset trigger rather than a replacement: a provider that
    does not stream simply yields one chunk per model call, and the ``values``
    path still produces the answer, which is then printed in full.  This function
    owns all terminal output, so callers must not print the returned string.

    Message history is not assembled here.  The checkpointer wired in
    ``agent/factory.build_agent`` stores the thread's messages and replays them
    on the next call for the same ``thread_id``, which is what carries the
    conversation between questions.

    This function owns the permission policy for the turn: it wraps the whole
    stream in an execution scope, so the tools see /ask's authority and nothing
    else. A /plan worker running elsewhere in the process cannot change it.
    """
    logger.info(f"Handling query: {question}")
    agent = await build_agent()
    context = ask_context(ask_permission_policy(interactive))
    agent_config = {"configurable": {"thread_id": thread_id}}
    best: str | None = None
    history: list = []
    handler = UsageCallbackHandler(kind="ask")

    printed = False
    model_call: str | None = None
    model_call_text = ""

    def _flush_turn() -> None:
        """Close off the current model call's output, if it produced any."""
        nonlocal printed
        if model_call_text:
            _write("\n")
            printed = True

    try:
        with execution_scope(context):
            async for mode, step in agent.astream(
                {"messages": [{"role": "user", "content": question}]},
                config={**agent_config, "callbacks": [handler]},
                stream_mode=["values", "messages"],
            ):
                if mode == "messages":
                    chunk, meta = step
                    meta = meta or {}
                    if meta.get("langgraph_node") != _MODEL_NODE:
                        continue
                    # checkpoint_ns is unique per model call, so a change means the
                    # previous call ended.  This keeps multi-turn tool loops and a
                    # provider fallback retry from being printed as one run-on.
                    call_id = meta.get("checkpoint_ns")
                    if call_id != model_call:
                        _flush_turn()
                        model_call = call_id
                        model_call_text = ""
                    delta = _stream_delta(chunk)
                    if delta:
                        _write(delta)
                        model_call_text += delta
                    continue

                msgs = step.get("messages") or []
                history = msgs
                for msg in reversed(msgs):
                    if type(msg).__name__ == _AI_MSG_NAME:
                        content = _extract_text(msg)
                        if content:
                            best = content
                            break
    except Exception as exc:
        _flush_turn()
        if best:
            logger.warning(
                "Agent stream failed after partial output; returning it",
                exc_info=True,
            )
            record(handler.records, "ask")
            return best
        raise exc

    _flush_turn()
    record(handler.records, "ask")

    if best:
        if not printed:
            # Nothing was streamed (non-streaming provider): show it now.
            _write(best + "\n")
        return best

    # No final text answer emitted (tool loop without a closing message).
    # Force an answer from the context the agent already gathered.
    if len(history) > 1:
        try:
            llm = get_llm()
            forced = await llm.ainvoke([
                {
                    "role": "system",
                    "content": (
                        "You are a code assistant. Based strictly on the tool "
                        "results in the conversation below, answer the user's "
                        "original question with a final, concise text answer. "
                        "Do NOT call any tools."
                    ),
                },
                *history[-14:],
                {
                    "role": "user",
                    "content": "Now produce your final text answer to the original question.",
                },
            ])
            content = _extract_text(forced)
            if content:
                logger.info("Agent ended without text; forced final answer from context")
                _write(content + "\n")
                return content
        except Exception:
            logger.warning(
                "Forced final-answer fallback failed",
                exc_info=True,
            )

    raise ValueError("Agent produced no answer")
