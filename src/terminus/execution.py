"""What execution am I in?

Every unit of work that can change a workspace - one /ask turn, one /plan task
worker, and later a spawned child agent - is an *execution*. This module is the
single place that answers "which project, which task, and under whose
permissions", and the single place that activates those answers for the
duration of a block.

Why it exists
-------------
The permission policy used to be a module global set as a side effect of
building an agent. With one agent at a time that happened to work, and it was
demonstrably unsafe as soon as two existed: a second worker overwrote the first
worker's policy before its tools ran, and a worker's policy leaked into the
next /ask. Ownership now belongs to the execution, via a context variable that
asyncio copies per Task.

What it deliberately does not do
--------------------------------
It does not change directory. The workspace is an *identity to validate*, not a
process-wide setting to mutate; relative tool paths and the shell's working
directory already resolve to the project the process was started in. chdir is
process-global and would be exactly the kind of shared mutable state this
module exists to eliminate.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from terminus.permissions import PermissionPolicy, permission_scope
from terminus.workspace import project_root

__all__ = [
    "ASK",
    "TASK",
    "ExecutionContext",
    "WorkspaceMismatch",
    "current_execution",
    "execution_scope",
    "require_workspace",
]


class WorkspaceMismatch(RuntimeError):
    """Raised when an execution is asked to act on the wrong project."""


ASK = "ask"
TASK = "task"
MAX_CHILDREN_PER_PARENT = 8
"""Total children one parent execution may delegate, across every call.

Per execution, not per ``spawn_agent`` call and not per conversation. Without
this, a model could call ``spawn_agent`` repeatedly and multiply the budget by
the number of calls, which is how a convenience primitive becomes an unbounded
cost. Eight is enough for a genuinely multi-area investigation and small enough
that a delegation habit is immediately visible in telemetry.

This is the one place the number is declared. ``terminus.agents.spawn`` re-exports
it as ``MAX_CHILD_AGENTS`` for callers that think in terms of the spawner; nothing
declares it twice, so the spawner's ceiling and the budget's ceiling cannot
disagree.
"""

CHILD = "child"
"""A delegated subagent spawned by an existing execution.

Its own context, with its own policy, rather than the parent's. The parent's
authority is not inherited implicitly: a child is granted what its role allows,
which is a subset of what the parent could do, and never more.
"""


@dataclass
class ExecutionBudget:
    """A per-execution allowance, spent as work is delegated.

    Scoped to one parent execution and owned by it: the object is created with
    the context and dies with it. There is no global or cross-turn store, so a
    busy conversation cannot spend a later, unrelated turn's allowance, and
    nothing has to be cleaned up when the execution ends.

    Mutable on purpose - unlike :class:`ExecutionContext`, which stays frozen.
    The context says *who* is executing; this says *how much is left*.
    """

    max_children: int
    _used: int = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_children - self._used)

    def claim(self, count: int = 1) -> int:
        """Reserve up to *count* children; return how many were actually granted.

        Partial grants are the point. Asking for five with three left grants
        three and returns three, so the caller can run what it got and report
        the rest as blocked rather than either silently dropping them or
        silently exceeding the budget.
        """
        wanted = max(0, int(count))
        granted = min(wanted, self.remaining)
        self._used += granted
        return granted


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable identity and authority for one unit of work.

    ``workspace`` is the project directory this execution is allowed to act on.
    ``policy`` is the authority for mutations. Neither is ever passed to a tool
    by the model: the tool layer reads the *active* context instead, so a task
    description or a tool call cannot widen it.
    """

    workspace: Path
    kind: str = ASK
    project_id: str | None = None
    task_id: str | None = None
    policy: PermissionPolicy = PermissionPolicy()
    parent_agent_id: str | None = None
    """Set on a child execution, naming the agent that delegated to it.

    Kept here rather than only in the agent layer because this is the object
    the permission and tool layer already reads, so a child is identifiable at
    the point where authority is actually enforced.
    """

    budget: "ExecutionBudget | None" = None
    """This execution's remaining delegation allowance, or None if it delegates.

    A parent carries one; a child does not, because children may not delegate.
    """

    @property
    def label(self) -> str:
        """Short identity for logs and error messages."""
        if self.parent_agent_id:
            return f"{self.kind} of {self.parent_agent_id}"
        if self.task_id:
            return f"task {self.task_id}"
        return self.kind


_current: ContextVar[ExecutionContext | None] = ContextVar(
    "terminus_execution", default=None
)


def current_execution() -> ExecutionContext | None:
    """The execution in force, or None outside any scope."""
    return _current.get()


@contextmanager
def execution_scope(context: ExecutionContext):
    """Make *context* the active execution for the duration of the block.

    The permission policy is activated through the same scope, so an execution's
    authority and its identity can never disagree. Both are restored on exit,
    including when the body raises.
    """
    token = _current.set(context)
    try:
        with permission_scope(context.policy):
            yield context
    finally:
        _current.reset(token)


def require_workspace(workspace) -> Path:
    """Validate that *workspace* is the project this process is working on.

    A task planned for Project A must never execute against Project B, however
    it was invoked. Returns the resolved workspace, or raises
    :class:`WorkspaceMismatch` with both paths named.
    """
    try:
        wanted = Path(workspace).expanduser().resolve()
    except OSError as exc:
        raise WorkspaceMismatch(
            f"Cannot resolve workspace {workspace!r}: {exc}"
        ) from exc

    current = project_root()
    if os.path.normcase(str(wanted)) != os.path.normcase(str(current)):
        raise WorkspaceMismatch(
            f"This work belongs to {wanted}, but Terminus is running in {current}. "
            f"Refusing to execute it here; run Terminus from {wanted} instead."
        )
    return wanted


def ask_context(
    policy: PermissionPolicy, max_children: int = MAX_CHILDREN_PER_PARENT
) -> ExecutionContext:
    """The execution for one /ask turn.

    Carries a fresh delegation budget. A new turn gets a new budget, so a long
    conversation does not permanently exhaust the allowance - unrelated later
    work is not charged for earlier delegation.
    """
    return ExecutionContext(
        workspace=project_root(), kind=ASK, policy=policy,
        budget=ExecutionBudget(max_children=max_children),
    )


def task_context(
    task_id: str,
    project_id: str | None,
    workspace,
    policy: PermissionPolicy,
) -> ExecutionContext:
    """The execution for one task worker, validated against this process.

    Raises :class:`WorkspaceMismatch` rather than constructing an execution that
    would act on the wrong project.
    """
    resolved = require_workspace(workspace)
    return ExecutionContext(
        workspace=resolved,
        kind=TASK,
        project_id=project_id,
        task_id=task_id,
        policy=policy,
        budget=ExecutionBudget(max_children=MAX_CHILDREN_PER_PARENT),
    )
