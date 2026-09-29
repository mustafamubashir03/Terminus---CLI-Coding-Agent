"""Which conversation the CLI is currently in.

The current session is a single UUID in a small file next to the checkpoint
database, not a row in a table and not an argument threaded through every call.
The reason is that LangGraph's checkpointer keys threads by id, and something has
to remember which id this terminal is talking to between invocations - including
across a restart, which is what makes a conversation resumable.

It is deliberately *only* the current session. Terminus has no catalogue of past
sessions, and ``terminus sessions list`` reports that plainly rather than
implying a history it does not keep.

The state file lives under the memory database path, so it inherits the same
project isolation: two projects in two directories cannot see each other's
current session.
"""

import uuid
from pathlib import Path


from terminus.observability.logging import get_logger
from terminus.config import CONFIG

logger = get_logger(__name__)

def _session_file()->Path:
    return Path(CONFIG["memory"]["db_path"]).parent / "current_session"

def get_current_session()->str:
    session_file = _session_file()
    if session_file.exists():
        session_id = session_file.read_text().strip()
        logger.debug(f"Current session ID: {session_id}")
        return session_id
    return new_session()

def new_session()->str:
    session_id = str(uuid.uuid4())
    session_file = _session_file()
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(session_id)
    logger.info(f"Created new session: {session_id}")
    return session_id

def switch_session(session_id:str)->None:
    """Switch to a different session"""
    session_file = _session_file()
    # Must mirror new_session(): without this, switching raises FileNotFoundError
    # whenever the state directory is absent, which takes the whole CLI down
    # (the /switch command is not inside a try/except).
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(session_id)
    logger.info(f"Switched to session: {session_id}")
