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


async def get_checkpointer() -> AsyncSqliteSaver:
    global _checkpointer, _db_conn

    if _checkpointer is None:
        db_path = Path(CONFIG["memory"]["db_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _db_conn = await aiosqlite.connect(
            str(db_path),
            check_same_thread=False
        )

        _checkpointer = AsyncSqliteSaver(_db_conn)

    return _checkpointer


async def close_checkpointer() -> None:
    """Close the shared memory database connection on shutdown.

    The aiosqlite worker thread is non-blocking only while the process runs;
    a leaked open connection keeps it alive and blocks the interpreter from
    exiting. This drains queued writes and stops the worker so the CLI can
    terminate cleanly.
    """
    global _checkpointer, _db_conn
    conn, _checkpointer, _db_conn = _db_conn, None, None
    if conn is not None:
        try:
            await conn.close()
        except Exception as e:
            logger.debug(f"Error closing memory db: {type(e).__name__}: {e}")


def get_summarization_middleware()->SummarizationMiddleware:
    return SummarizationMiddleware(
        model=get_llm(),
        trigger=("tokens", CONFIG["memory"]["summarize_at_tokens"]),
        keep=("messages", CONFIG["memory"]["max_messages"]),
    )
