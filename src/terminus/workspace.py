"""Project identity: the one thing everything else keys on.

Terminus has no explicit "project" object. A project *is* the directory the
process was started in, and all of its state hangs off `<root>/.terminus/`
because every configured path in ``config.py`` is relative:

    memory.db_path  -> .terminus/memory/terminus.db   (checkpoints, session id)
    tasks.db_path    -> .terminus/tasks/tasks.db       (projects, tasks)
    chromadb.persist -> .terminus/chromadb/            (semantic index)

That is what keeps two projects from sharing conversational or task state, and
it is why isolation works today. It is also why the resolved root must be used
explicitly anywhere state is *cached*: a relative path is re-interpreted
against whatever cwd happens to be at call time, so a process-wide cache keyed
on the configured string can hand Project A's data to Project B.

This module deliberately does no path *containment* (that is deferred, and is
documented in tools/filesystem_tools.py). It only names the workspace.
"""

from __future__ import annotations

from pathlib import Path

from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def project_root() -> Path:
    """The resolved directory that defines the current project.

    Falls back to the unresolved cwd if resolution fails (a deleted or
    permission-denied directory should not stop the CLI).
    """
    try:
        return Path.cwd().resolve()
    except OSError as exc:  # pragma: no cover - platform dependent
        logger.warning("Could not resolve cwd (%s); using it unresolved", exc)
        return Path.cwd()


def project_key() -> str:
    """A stable string identity for the current project, safe for cache keys."""
    return str(project_root())


PROJECT_METADATA_KEY = "project"
"""Qdrant payload key that scopes an indexed chunk to one project.

The collection is shared by every project, so this field is the only thing
stopping project A's code from being retrieved while working in project B. The
writer (the indexers) and the reader (the retriever) must agree on the name, so
it is defined here next to the identity it carries rather than in either of
them.
"""

