"""Which tools exist, and what a name resolves to.

That is the whole job. LangChain already owns the tool object, its JSON schema,
tool-call parsing, routing and execution (``create_agent`` builds a ``ToolNode``
for all of it), so nothing here wraps a tool in a second Terminus type, builds a
schema, or calls anything.

What LangChain does not do is the part that is Terminus's: knowing *which* tools
this project ships, and turning a tool *name* - how roles, configs, prompts and
the CLI all refer to tools - back into the tool object.

This module exists because that answer was previously written down in three
unrelated places: ``ASK_TOOLS`` for /ask, per-task-type lists in
``terminus.tasks.executor``, and name tuples in ``terminus.agents.roles``. They
drifted: /plan's shell tools register under different names from /ask's, and
nothing noticed. One catalogue makes that drift impossible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from terminus.skills.skill_tools import load_skill
from terminus.tools.codebase_tool import search_codebase
from terminus.tools.git_tools import (
    git_branch,
    git_checkout,
    git_commit,
    git_diff,
    git_log,
    git_status,
)
from terminus.tools.filesystem_tools import (
    append_file,
    delete_file,
    edit_file,
    file_exists,
    grep,
    list_directory,
    read_file,
    write_file,
)
from terminus.tools.project_status_tool import project_status
from terminus.tools.shell_tools import run_command
from terminus.tools.spawn_agent_tool import spawn_agent
from terminus.tools.terminal_tools import run_command as run_plan_shell
from terminus.tools.terminal_tools import run_in_directory
from terminus.tools.web_tools import web_fetch, web_search

__all__ = [
    "CATALOGUE",
    "ASK_TOOL_NAMES",
    "GIT_READ_TOOL_NAMES",
    "PLAN_FILESYSTEM_TOOL_NAMES",
    "PLAN_GIT_TOOL_NAMES",
    "PLAN_TOOL_NAMES",
    "ask_tools",
    "catalogue",
    "missing_tool_names",
    "plan_tools_for",
    "resolve",
    "tool_names",
]

#: Every tool Terminus ships, under the name the model calls it. /ask and /plan
#: have separate shell tools with different timeouts and report formats; they are
#: registered under their own names rather than aliased to one another.
CATALOGUE: tuple[Any, ...] = (
    read_file,
    write_file,
    append_file,
    edit_file,
    delete_file,
    list_directory,
    file_exists,
    grep,
    run_command,
    run_plan_shell,
    run_in_directory,
    git_status,
    git_diff,
    git_commit,
    git_log,
    git_checkout,
    git_branch,
    search_codebase,
    project_status,
    web_search,
    web_fetch,
    load_skill,
    spawn_agent,
)

ASK_TOOL_NAMES: tuple[str, ...] = (
    "search_codebase",
    "grep",
    "list_directory",
    "read_file",
    "file_exists",
    "write_file",
    "edit_file",
    "web_search",
    "web_fetch",
    "run_command",
    "git_status",
    "git_diff",
    "git_commit",
    "git_log",
    "git_checkout",
    "git_branch",
    "load_skill",
    "project_status",
    "spawn_agent",
)
"""The /ask toolset, as names.

``delete_file`` and ``append_file`` are withheld: /ask may edit and create, but not
destroy. ``run_command`` is present and policy-gated at call time by
:mod:`terminus.permissions`.

Stated as names because that is what a role, a config file or the CLI can
express, and :func:`resolve` is the only way a name becomes a tool.
"""

GIT_READ_TOOL_NAMES: tuple[str, ...] = ("git_status", "git_diff", "git_log")
"""The git tools that only observe.

``git_commit``, ``git_checkout`` and ``git_branch`` can move the working tree
and are granted to no role: a child sharing the parent's tree could otherwise
switch the branch its parent is on, or record unfinished work as a checkpoint.
"""

PLAN_FILESYSTEM_TOOL_NAMES: tuple[str, ...] = (
    "list_directory",
    "read_file",
    "write_file",
    "delete_file",
    "file_exists",
    "append_file",
)
"""The filesystem tools a /plan worker gets, on top of ``search_codebase``.

/plan workers are told to create specific deliverable paths, so they get the
destructive filesystem tools /ask withholds. A worker has no approver, so DELETE
is still refused there.
"""

PLAN_GIT_TOOL_NAMES: tuple[str, ...] = GIT_READ_TOOL_NAMES
"""The git tools a /plan worker gets: readers only.

A worker has no approver, so anything at WRITE or DESTRUCTIVE is refused there
anyway. Offering a tool the model can see and never use just wastes its calls
discovering that, so the writers are withheld from every task type.
"""

PLAN_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "design": (*PLAN_FILESYSTEM_TOOL_NAMES, *PLAN_GIT_TOOL_NAMES, "load_skill"),
    "implement": (
        *PLAN_FILESYSTEM_TOOL_NAMES,
        *PLAN_GIT_TOOL_NAMES,
        "run_shell_command",
        "run_command_in_directory",
        "load_skill",
    ),
    "test": (
        *PLAN_FILESYSTEM_TOOL_NAMES,
        *PLAN_GIT_TOOL_NAMES,
        "run_shell_command",
        "run_command_in_directory",
    ),
    "review": (*PLAN_FILESYSTEM_TOOL_NAMES, *PLAN_GIT_TOOL_NAMES, "load_skill"),
    "integrate": (
        *PLAN_FILESYSTEM_TOOL_NAMES,
        *PLAN_GIT_TOOL_NAMES,
        "run_shell_command",
        "run_command_in_directory",
    ),
    "configure": (
        *PLAN_FILESYSTEM_TOOL_NAMES,
        *PLAN_GIT_TOOL_NAMES,
        "run_shell_command",
        "run_command_in_directory",
    ),
}
"""Tool names per /plan task type, before MCP tools are appended.

An unknown task type falls back to ``search_codebase`` alone.
"""

_BY_NAME: dict[str, Any] = {tool.name: tool for tool in CATALOGUE}

assert len(_BY_NAME) == len(CATALOGUE), "duplicate tool name in CATALOGUE"


def tool_names() -> tuple[str, ...]:
    """Every tool name Terminus ships, sorted."""
    return tuple(sorted(_BY_NAME))


def catalogue() -> dict[str, Any]:
    """A copy of the name -> tool mapping.

    A copy, so a caller cannot register or drop a tool by mutating what it got
    back. Registration happens in ``CATALOGUE`` at import time and nowhere else.
    """
    return dict(_BY_NAME)


def missing_tool_names(names: Iterable[str]) -> tuple[str, ...]:
    """Which of *names* the catalogue does not have, in the order given.

    Returned rather than raised: narrowing a toolset is normal for a role or a
    task type to do, and a missing tool must not be able to take down an agent
    whose remaining tools are fine. The CLI prints it; a child agent logs it.
    """
    return tuple(name for name in names if name not in _BY_NAME)


def resolve(names: Iterable[str], *, extra: Sequence[Any] = ()) -> tuple[Any, ...]:
    """Turn tool *names* into the LangChain tools they refer to.

    Deduplicated, in the order *names* was given, so the order a toolset is
    written in is the order the model sees it in. *extra* carries tools with no
    catalogue entry - MCP tools, whose names are only known once a server has
    connected - and is appended after the named ones.

    Unknown names are skipped rather than raising; see
    :func:`missing_tool_names`.
    """
    resolved: list[Any] = []
    seen: set[int] = set()

    def add(tool: Any) -> None:
        if id(tool) not in seen:
            seen.add(id(tool))
            resolved.append(tool)

    for name in names:
        tool = _BY_NAME.get(name)
        if tool is not None:
            add(tool)
    for tool in extra:
        add(tool)
    return tuple(resolved)


def ask_tools() -> tuple[Any, ...]:
    """The /ask toolset, resolved."""
    return resolve(ASK_TOOL_NAMES)


def plan_tools_for(task_type: str, *, extra: Sequence[Any] = ()) -> tuple[Any, ...]:
    """The toolset for one /plan task type, resolved, plus any *extra* tools."""
    names = PLAN_TOOL_NAMES.get((task_type or "").strip().lower())
    if names is None:
        return resolve(("search_codebase",), extra=extra)
    return resolve(names, extra=extra)
