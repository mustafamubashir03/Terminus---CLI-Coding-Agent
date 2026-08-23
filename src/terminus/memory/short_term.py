from terminus.llm.factory import get_llm
from langchain.agents.middleware import SummarizationMiddleware
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path

from terminus.observability.logging import get_logger
from terminus.config import CONFIG

logger = get_logger(__name__)

_checkpointer = None


def get_checkpointer() -> SqliteSaver:
    global _checkpointer

    if _checkpointer is None:
        db_path = Path(CONFIG["memory"]["db_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False
        )

        _checkpointer = SqliteSaver(conn)

    return _checkpointer


def get_session_history(thread_id: str) -> list[dict]:
    checkpointer = get_checkpointer()
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = checkpointer.get(config)
    if not checkpoint:
        logger.info(f"No checkpoint found for session: {thread_id}")
        return []
    messages = checkpoint["channel_values"].get("messages",[])
    return [
        {
            "role": "user" if m["type"] == "human" else "assistant",
            "content": m.content
        } for m in messages  
    ]


def get_summarization_middleware()->SummarizationMiddleware:
    return SummarizationMiddleware(
        model=get_llm(),
        trigger=("tokens", CONFIG["memory"]["summarize_at_tokens"]),
        keep=("messages", CONFIG["memory"]["max_messages"]),
    )
    
    
    
    
