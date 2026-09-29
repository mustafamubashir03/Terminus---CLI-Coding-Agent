"""Bounded child agent execution.

Skills provide knowledge; agents provide bounded execution. They are separate
abstractions on purpose: a skill is text that changes what an agent knows, and an
agent is a set of permissions, tools, a budget and a deadline. Keeping them apart
is what allows a skill to be shared by the main agent and any child, and a child
to exist with no skill at all.
"""

from terminus.agents.roles import AgentRole, ROLES, describe_roles, get_role, role_names
from terminus.agents.spawn import (
    MAX_CHILD_AGENTS,
    MAX_CHILD_DEPTH,
    MAX_PARALLEL_AGENTS,
    AgentResult,
    AgentSpawner,
    AgentStatus,
    ChildAgent,
    ChildSpec,
    WriteScopeError,
    build_child_context,
    child_policy,
    recent_delegations,
    spawn_agent,
    spawn_agents,
    write_scope_holder,
)

__all__ = [
    "AgentRole",
    "AgentResult",
    "AgentSpawner",
    "AgentStatus",
    "ChildAgent",
    "ChildSpec",
    "MAX_CHILD_AGENTS",
    "MAX_CHILD_DEPTH",
    "MAX_PARALLEL_AGENTS",
    "ROLES",
    "WriteScopeError",
    "build_child_context",
    "child_policy",
    "describe_roles",
    "get_role",
    "recent_delegations",
    "role_names",
    "spawn_agent",
    "spawn_agents",
    "write_scope_holder",
]
