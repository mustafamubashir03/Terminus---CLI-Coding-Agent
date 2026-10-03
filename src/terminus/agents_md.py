"""AGENTS.md: what this workspace has taught a previous session.

One Markdown file at the workspace root, holding knowledge that should outlive a
conversation. It is written by whoever learns something - usually the model, with
``edit_file`` like any other file - and it is read back at the start of every
session so a later run starts where the last one stopped.

This is deliberately the whole mechanism. There is no database, no embedding, no
extraction step and no background process: the filesystem persists it, the harness
loads it, and the model maintains it. Anything that made this file durable would
also make it stale, and there is already a database for state that genuinely needs
one (``.terminus/``).

Where the content goes, and where it does not, is in :data:`AGENT_MD_TEMPLATE`:
conventions, decisions, gotchas, active tasks. Not transient reasoning, not command
output, not anything rediscoverable by reading the repository.

    Workspace
        └── AGENTS.md
                │
                ▼
          load_agents_md()
                │
                ▼
          _build_system_prompt()      (agent/factory.py, tasks/executor.py)
                │
                ▼
              MODEL
"""

from __future__ import annotations

from pathlib import Path

from terminus.workspace import project_root, resolve_in_workspace

AGENTS_MD_FILENAME = "AGENTS.md"

AGENT_MD_TEMPLATE = """# Agent Memory

This file is the agent's durable memory across sessions.
The harness loads it at the start of every session.
Update it whenever you learn something useful.

## Conventions

## Decisions

## Gotchas

## Active Tasks
"""

_NOTICE = """\
This file is the agent's durable memory across sessions. The harness loads it at
the start of every session, and the agent maintains it with the ordinary file
tools.

Worth keeping: project naming conventions, architecture decisions, dependency
constraints, commands that must run in a particular order, repository quirks, and
work a future session should know is unfinished.

Not worth keeping: reasoning you have already acted on, tool calls, command
output, logs, chat transcripts, generated files, and anything you could rediscover
by reading the repository. Keep it short enough that a future session reads it."""


def agents_md_path() -> Path:
    """The resolved path of this workspace's AGENTS.md.

    There is deliberately no module-level absolute path. The workspace is resolved
    per call - it can be changed with ``TERMINUS_WORKSPACE``, and tests point it
    elsewhere - so a constant computed at import would name one workspace for the
    life of the process. This function is the single place that says where the
    file lives.

    Routed through :func:`terminus.workspace.resolve_in_workspace`, so it is
    located by the same containment rule as every other path a tool can name.
    """
    return resolve_in_workspace(AGENTS_MD_FILENAME, workspace=project_root())


def load_agents_md() -> str:
    """This workspace's durable memory, created from the template when absent.

    Idempotent and never destructive: an existing file is read and returned
    untouched however many times this is called, so no session can lose what an
    earlier one wrote.

    Re-read on every call rather than cached. A session is only as good as what it
    started with, and a process-wide cache would pin one session's memory for the
    life of the process - exactly the staleness this file exists to prevent.
    Callers already compose their context per turn, so the cost is one small file
    read.

    Raises ``OSError`` when the file exists but cannot be read, or cannot be
    created. That is not the same as it being absent, and quietly returning the
    template would look to the model like its memory had been wiped.
    """
    path = agents_md_path()

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise OSError(f"Cannot read {path}: {exc}") from exc

    return _create(path)


def _create(path: Path) -> str:
    """Write the template to *path* without ever clobbering an existing file.

    Exclusive creation is the whole concurrency story. Two sessions starting at
    once both find the file missing; exactly one creates it; the loser reads back
    the winner's copy. A check-then-write would let the second session overwrite
    notes the first had only just started, which is the one race this file cannot
    afford. No second lock is involved: the file itself is the arbiter.
    """
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(AGENT_MD_TEMPLATE)
    except FileExistsError:
        # Another session created it between the read and the write; its content
        # is the authoritative one.
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Cannot create {path}: {exc}") from exc
    return AGENT_MD_TEMPLATE


def agents_md_section(content: str) -> str:
    """Wrap AGENTS.md for the model, or return "" when there is nothing to say.

    A section of its own rather than an addition to the base prompt, because what
    is in it changes between sessions: it is learned knowledge about this project,
    not policy the harness stands behind.
    """
    if not content or not content.strip():
        return ""
    return f"## Agent memory (AGENTS.md)\n\n{_NOTICE}\n\n{content.strip()}"
