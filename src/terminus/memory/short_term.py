"""Conversational memory: the checkpoint store, and when to summarise it.

Two related jobs, both about the length of a conversation rather than its
content:

* **Checkpointing.** One ``AsyncSqliteSaver`` is created on first use and shared
  by every agent running on the same event loop. Sharing matters: it is what lets a
  /ask turn and a /plan worker each resume their own thread, and what makes
  ``close_checkpointer`` a single shutdown step rather than one per agent.
* **Summarisation.** Once a thread passes a token threshold, older messages are
  replaced with a summary so a long conversation stays affordable. This is a
  middleware, so LangGraph applies it as part of the run rather than Terminus
  having to trim history itself.

The aiosqlite connection runs its own worker thread. It is non-blocking only
while the process is alive, so ``close_checkpointer`` must be awaited on shutdown
or the thread is left dangling.

Why the saver is keyed on the event loop
----------------------------------------
``AsyncSqliteSaver`` captures the loop it was constructed on and its async methods
use an ``asyncio.Lock`` without re-checking it:

    self.lock = asyncio.Lock()
    self.loop = asyncio.get_running_loop()

It is therefore only valid on the loop that created it. Handing one back to a
different loop raises ``RuntimeError: ... is bound to a different event loop`` the
moment two operations contend for that lock - which is what a checkpointed graph
run does as soon as it has concurrent writes. The CLI has one loop for the life
of the process and never noticed; a test suite, or anything that calls
``asyncio.run`` more than once, gets a hard failure.

So the cached saver is reused while the running loop is the same one, and
replaced when it is not. Reconnecting costs one file open, and the checkpointer
path is not hot; correctness is worth more than avoiding it.
"""

import asyncio
import aiosqlite
from terminus.llm.factory import get_llm
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pathlib import Path
from terminus.observability.logging import get_logger
from terminus.config import CONFIG

logger = get_logger(__name__)

_checkpointer = None
_db_conn = None
_checkpointer_loop = None
"""The loop ``_checkpointer`` was built for, or None. See the module docstring."""


async def get_checkpointer() -> AsyncSqliteSaver:
    """The shared checkpoint saver, for the loop this call is running on.

    The same object is returned for every call on one event loop, so a /ask turn,
    a child agent and a /plan worker still share one connection and one set of
    tables. A call on a different loop gets its own, because the previous one is
    unusable there.
    """
    global _checkpointer, _db_conn, _checkpointer_loop

    loop = asyncio.get_running_loop()
    if _checkpointer is not None and _checkpointer_loop is loop:
        return _checkpointer

    if _checkpointer is not None:
        await _discard_previous()

    db_path = Path(CONFIG["memory"]["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _db_conn = await aiosqlite.connect(str(db_path), check_same_thread=False)
    _checkpointer = AsyncSqliteSaver(_db_conn)
    _checkpointer_loop = loop
    return _checkpointer


async def _discard_previous() -> None:
    """Drop a saver built for a different loop, closing it if that is possible.

    Closing is best effort. The connection belongs to a loop that is no longer
    running, so the close may not be awaitable from here, and its worker thread is
    a daemon that will go with the process. Failing to close is worth a log line,
    not an exception: the caller asked for a checkpointer, and it is the stale one
    that is already unusable.
    """
    global _checkpointer, _db_conn, _checkpointer_loop

    conn, _checkpointer, _db_conn, _checkpointer_loop = _db_conn, None, None, None
    if conn is None:
        return
    try:
        await conn.close()
    except Exception as e:
        logger.debug(
            "Could not close the memory db from its original loop (%s: %s); "
            "leaving it to the interpreter", type(e).__name__, e,
        )


async def close_checkpointer() -> None:
    """Close the shared memory database connection on shutdown.

    The aiosqlite worker thread is non-blocking only while the process runs;
    a leaked open connection keeps it alive and blocks the interpreter from
    exiting. This drains queued writes and stops the worker so the CLI can
    terminate cleanly.
    """
    await _discard_previous()


def get_summarization_middleware()->SummarizationMiddleware:
    return SummarizationMiddleware(
        model=get_llm(),
        trigger=("tokens", CONFIG["memory"]["summarize_at_tokens"]),
        keep=("messages", CONFIG["memory"]["max_messages"]),
    )
