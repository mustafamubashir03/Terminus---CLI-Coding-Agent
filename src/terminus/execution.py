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

    @property
    def label(self) -> str:
        """Short identity for logs and error messages."""
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


def ask_context(policy: PermissionPolicy) -> ExecutionContext:
    """The execution for one /ask turn."""
    return ExecutionContext(workspace=project_root(), kind=ASK, policy=policy)


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
    )
