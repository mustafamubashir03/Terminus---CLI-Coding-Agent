"""Startup environment context for the conversational agent.

This is a one-shot snapshot taken the first time an agent system prompt is built
in a Terminus process. It tells the model where it is and what the project root
contains, without spending a tool call on discovery.

Deliberately NOT authoritative: the project can change after the snapshot is
taken, so the filesystem tools remain the source of truth and the snapshot says
so explicitly.
"""

from __future__ import annotations

import platform
from pathlib import Path

from terminus.observability.logging import get_logger

logger = get_logger(__name__)

# The listing is an orientation aid, not a full inventory, so it is capped. The
# actual contents of a project-instructions file are never capped.
_MAX_LISTED_ENTRIES = 200

PROJECT_INSTRUCTIONS_FILENAME = "TERMINUS.md"


def _directory_listing(directory: Path) -> str:
    """Return a sorted, depth-1 listing of *directory*, directories first."""
    try:
        entries = sorted(
            directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except OSError as exc:
        return f"(unavailable: {type(exc).__name__})"

    lines = []
    for entry in entries[:_MAX_LISTED_ENTRIES]:
        try:
            is_dir = entry.is_dir()
        except OSError:
            is_dir = False
        lines.append(f"{entry.name}/" if is_dir else entry.name)
    if len(entries) > _MAX_LISTED_ENTRIES:
        lines.append(f"... and {len(entries) - _MAX_LISTED_ENTRIES} more entries")
    return "\n".join(lines) if lines else "(empty)"


def _project_instructions(cwd: Path) -> str:
    """Return the contents of a TERMINUS.md sitting directly in *cwd*.

    Only that exact path is checked - the search is never recursive, so a
    nested or vendored TERMINUS.md cannot hijack the project instructions.
    Returns "" when the file is absent, empty, or unreadable, so that no empty
    or broken section is emitted.
    """
    path = cwd / PROJECT_INSTRUCTIONS_FILENAME
    if not path.is_file():
        return ""
    try:
        # errors="replace" so one malformed byte cannot stop the CLI starting.
        contents = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return ""
    if not contents:
        return ""
    return (
        "## Project instructions\n\n"
        f"Source: {PROJECT_INSTRUCTIONS_FILENAME} at the project root.\n\n"
        f"{contents}"
    )


def build_startup_context(cwd: Path | None = None) -> str:
    """Build the environment block for the agent system prompt."""
    directory = Path(cwd) if cwd is not None else Path.cwd()
    try:
        resolved = directory.resolve()
    except OSError:
        resolved = directory

    environment = "\n".join(
        [
            "## Environment",
            "",
            "Snapshot taken at startup. The filesystem tools are authoritative;",
            "if the project has changed since, verify with a tool instead of trusting this.",
            "",
            f"Working directory: {resolved}",
            f"Operating system: {platform.system()}",
            "",
            f"Contents of {resolved.name or resolved}:",
            _directory_listing(resolved),
        ]
    )

    instructions = _project_instructions(resolved)
    return f"{environment}\n\n{instructions}" if instructions else environment
