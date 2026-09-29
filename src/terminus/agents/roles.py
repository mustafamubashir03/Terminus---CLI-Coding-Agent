"""Child agent roles.

A role is a small, immutable configuration - a tool allowance, a write capability,
default skills and a budget. Deliberately data, not classes: a role is a
description of what an agent is *allowed* to do, and there is no per-role
behaviour to override. If a role ever needs its own logic, that logic belongs in
the child runtime, not in a subclass hierarchy.

The important property is that a role can only ever *narrow* capability. A role
grants a named subset of tools; the runtime intersects that with the tools the
parent actually has, so a role can never widen what the caller may do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tool names, matching terminus.tools. A role lists names rather than objects so
# the set is inspectable, testable, and survives a tool module being refactored.
READ_TOOLS = ("read_file", "grep", "search_codebase", "list_directory", "file_exists")
WRITE_TOOLS = ("write_file", "edit_file")
SHELL_TOOLS = ("run_command",)
SKILL_TOOLS = ("load_skill",)
CONTEXT_TOOLS = ("project_status",)


@dataclass(frozen=True)
class AgentRole:
    """What one kind of child agent may do."""

    name: str
    description: str
    tools: tuple[str, ...]
    write: bool = False
    shell: bool = False
    skills: tuple[str, ...] = ()
    max_tool_calls: int = 20
    max_model_calls: int = 12
    timeout_seconds: int = 300
    destructive: bool = False
    """Whether this role may be granted destructive operations.

    False for every built-in role. A child agent is not an approval path: nothing
    here should be able to escalate to destructive on its own.
    """
    tags: tuple[str, ...] = field(default_factory=tuple)

    def tool_names(self) -> set[str]:
        return set(self.tools)

    def policy_description(self) -> str:
        parts = [
            f"tools: {', '.join(sorted(self.tools)) or 'none'}",
            f"write: {'yes' if self.write else 'no'}",
            f"shell: {'yes' if self.shell else 'no'}",
            f"destructive: {'yes' if self.destructive else 'no'}",
            f"budget: {self.max_model_calls} model calls / {self.max_tool_calls} tool calls",
        ]
        return "; ".join(parts)


ROLES: dict[str, AgentRole] = {}


def _register(role: AgentRole) -> AgentRole:
    ROLES[role.name] = role
    return role


# A researcher reads and reports. It cannot write, cannot run commands, and
# therefore cannot disturb the working tree no matter what it concludes.
_register(AgentRole(
    name="researcher",
    description="Investigates and reports. Read-only: cannot modify files or run commands.",
    tools=tuple(READ_TOOLS) + CONTEXT_TOOLS,
    write=False,
    shell=False,
    max_tool_calls=25,
    max_model_calls=14,
    tags=("read-only",),
))

# An implementer writes, and therefore takes a write scope so two of them cannot
# edit the same files at once.
_register(AgentRole(
    name="implementer",
    description="Writes code. Can edit files and run commands within a claimed write scope.",
    tools=tuple(READ_TOOLS) + WRITE_TOOLS + SHELL_TOOLS + CONTEXT_TOOLS,
    write=True,
    shell=True,
    max_tool_calls=40,
    max_model_calls=20,
    tags=("writes",),
))

# A tester runs things but does not ship fixes: it may run commands and read
# output, and it may write only into a scratch location it is told about.
_register(AgentRole(
    name="tester",
    description="Runs tests and verifies behaviour. Read and execute, but does not implement fixes.",
    tools=tuple(READ_TOOLS) + SHELL_TOOLS + CONTEXT_TOOLS,
    write=False,
    shell=True,
    max_tool_calls=30,
    max_model_calls=16,
    tags=("read-only", "executes"),
))

_register(AgentRole(
    name="reviewer",
    description="Reviews existing code and reports findings. Strictly read-only.",
    tools=tuple(READ_TOOLS) + CONTEXT_TOOLS,
    write=False,
    shell=False,
    max_tool_calls=25,
    max_model_calls=14,
    tags=("read-only",),
))

_register(AgentRole(
    name="debugger",
    description="Diagnoses a failure and reports a root cause. Read-only unless explicitly given a write scope.",
    tools=tuple(READ_TOOLS) + SHELL_TOOLS + CONTEXT_TOOLS,
    write=False,
    shell=True,
    max_tool_calls=30,
    max_model_calls=18,
    tags=("read-only", "executes"),
))


def get_role(name: str) -> AgentRole | None:
    return ROLES.get((name or "").strip().lower())


def role_names() -> list[str]:
    return sorted(ROLES)


def describe_roles() -> str:
    """The catalogue the parent sees when deciding whether to delegate."""
    lines = ["Available child agent roles:"]
    for role in (ROLES[name] for name in role_names()):
        lines.append(f"- {role.name}: {role.description}")
        lines.append(f"    {role.policy_description()}")
    return "\n".join(lines)
