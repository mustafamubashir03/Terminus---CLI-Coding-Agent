"""The workspace: what directory Terminus is allowed to act in, and what a
model-supplied path is allowed to mean.

Two responsibilities, deliberately in one module because they are two halves of
the same question:

**Identity.** A workspace is a resolved directory. By default it is the
directory the process was started in, which is also what keeps two projects from
sharing conversational or task state: every configured path in ``config.py`` is
relative, so it all hangs off ``<workspace>/.terminus/``. ``TERMINUS_WORKSPACE``
overrides that, which is how a second process, a script, or a test attaches to a
workspace the process did not start in. It is read on every call rather than at
import, so it is ordinary configuration and not hidden process state.

**Containment.** ``resolve_in_workspace`` is the only sanctioned way to turn a
model-supplied path into a real one. It resolves against the workspace root and
then *proves* the result is inside it. Resolution happens before the check, so
``..`` traversal, absolute paths, Windows drive letters, mixed separators, and
symlinks or junctions that point out of the tree are all rejected by the same
comparison rather than by pattern-matching each spelling.

The invariant this module exists to make true:

    An agent operates inside a workspace.
    A workspace may be shared by multiple agents and sessions.
    Every filesystem path is resolved relative to that workspace.
    Host paths outside the workspace cannot be reached through filesystem tools.

It is enforced here in code, not described in a prompt. A prompt can tell a
model where it is; only this function can stop it leaving.

Not here, on purpose:

* **Permissions.** Whether an operation is *allowed* is
  :mod:`terminus.permissions`; whether it can happen *right now* is
  :mod:`terminus.coordination`. Containment is neither of those - it is a
  statement about which paths exist to name at all.
* **Filesystem operations.** This module resolves and validates; it does not
  read, write or delete. ``tools/filesystem_tools.py`` owns the operations and
  calls in here for every path it touches.
"""

from __future__ import annotations

import os
from pathlib import Path

from terminus.observability.logging import get_logger

logger = get_logger(__name__)

WORKSPACE_ENV_VAR = "TERMINUS_WORKSPACE"
"""Attach this process to a workspace other than the one it started in.

Configuration, read at call time. Two processes pointed at the same directory
therefore share one durable filesystem state, which is what makes a workspace
usable as an artifact layer across sessions - and it is also the single value a
test sets to say "these paths are inside the workspace".
"""


class WorkspaceViolation(ValueError):
    """Raised when a path cannot be resolved to something inside the workspace.

    A ``ValueError`` because it is a bad argument, not a runtime fault. Callers
    that are talking to a model turn it into an observable failure (see
    ``tools/filesystem_tools``); callers that are talking to a user let it
    propagate.
    """


def _resolve_dir(path) -> Path:
    """Resolve *path* as far as the platform allows.

    A deleted or permission-denied directory falls back to the unresolved path
    rather than raising: workspace identity should never be the thing that stops
    the CLI from starting.
    """
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve()
    except OSError as exc:  # pragma: no cover - platform dependent
        logger.warning("Could not resolve %s (%s); using it unresolved", candidate, exc)
        return candidate


def project_root() -> Path:
    """The resolved directory that defines the current workspace.

    ``TERMINUS_WORKSPACE`` when set, otherwise the working directory. Falls back
    to the unresolved cwd if resolution fails.
    """
    configured = os.getenv(WORKSPACE_ENV_VAR, "").strip()
    if configured:
        return _resolve_dir(configured)
    return _resolve_dir(Path.cwd())


def project_key() -> str:
    """A stable string identity for the current workspace, safe for cache keys."""
    return str(project_root())


def is_within(root: Path, target: Path) -> bool:
    """Is *target* the same as, or inside, *root*?

    Comparison is done with ``Path.relative_to``, so it inherits the platform's
    own rules: case-insensitive on Windows, case-sensitive on POSIX, and
    ``root`` itself counts as inside ``root`` because ``list_directory(".")`` and
    ``grep(path=".")`` both have to work.
    """
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_in_workspace(path, *, workspace=None) -> Path:
    """Resolve a model-supplied *path* inside the workspace, or refuse it.

    Returns an absolute, fully-resolved ``Path``. Relative paths are taken as
    workspace-relative, which is what the prompt tells the model they mean.

    Raises :class:`WorkspaceViolation` for an empty path, an embedded NUL, a path
    the platform cannot resolve, or a path that lands outside the workspace.
    Ordering matters: the path is *resolved first* and checked second, which is
    what makes ``..`` traversal, absolute paths, drive letters, backslash
    separators, and symlinks or junctions pointing out of the tree all fail the
    same single comparison instead of needing one guard per spelling.

    The root itself is resolved before use for the same reason - otherwise a
    workspace reached through a symlink would reject every path under it.
    """
    root = project_root() if workspace is None else _resolve_dir(workspace)

    raw = str(path if path is not None else "").strip()
    if not raw:
        raise WorkspaceViolation(
            "No path provided. Filesystem paths must name a file or directory."
        )
    if "\0" in raw:
        raise WorkspaceViolation("Path contains an invalid null character.")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkspaceViolation(f"Cannot resolve path {raw!r}: {exc}") from exc

    if not is_within(root, resolved):
        raise WorkspaceViolation(
            f"{raw!r} resolves to {resolved}, which is outside the workspace "
            f"{root}. Filesystem tools can only reach paths inside the "
            "workspace; use a workspace-relative path instead."
        )
    return resolved


def relative_to_workspace(path, *, workspace=None) -> str:
    """Render *path* the way the model should have written it.

    Used only for messages and reports. A path outside the workspace comes back
    absolute, because pretending it was relative would hide the very thing the
    caller is complaining about.
    """
    root = project_root() if workspace is None else _resolve_dir(workspace)
    try:
        return Path(path).resolve().relative_to(root).as_posix() or "."
    except (OSError, RuntimeError, ValueError):
        return str(path)


PROJECT_METADATA_KEY = "project"
"""Qdrant payload key that scopes an indexed chunk to one project.

The collection is shared by every project, so this field is the only thing
stopping project A's code from being retrieved while working in project B. The
writer (the indexers) and the reader (the retriever) must agree on the name, so
it is defined here next to the identity it carries rather than in either of
them.
"""
