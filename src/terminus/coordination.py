"""One writer per project at a time.

Invariant
---------
For a given project/workspace, at most one task may be performing a
workspace-mutating operation at a time. Read-only work is never blocked.

This is a *coordination* concern, deliberately separate from the four things
that are easy to confuse with it:

    task claiming         TaskStore.claim_task - which task may start
    execution identity    ExecutionContext - who is running
    permission policy     may this operation be done at all
    workspace validation  is this the right project
    write coordination    may it happen right now, alongside other tasks

A task may hold WRITE permission and still have to wait its turn. Authorisation
answers "is this allowed?", coordination answers "is now?".

Scope
-----
The lock is **process-local**, because the architecture is single-orchestrator per
project: ``TaskStore.recover_interrupted_tasks`` documents that assumption and
``claim_task`` is atomic so a second process cannot double-execute a task. What
this module guarantees is therefore "one writer per project per Terminus
process". Two Terminus processes pointed at the same directory are NOT covered -
see the note in ``project_write_guard``.

Why a threading lock, not asyncio
---------------------------------
The mutation tools are synchronous (``write_file``, ``run_command`` ...), so they
run on LangChain's thread executor and cannot await. A ``threading.Lock`` is
therefore the honest primitive. It is always acquired with a timeout, so it can
never block forever - and a timeout never breaks the invariant, it just makes
the loser decline rather than wait.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from terminus.permissions import (
    REFUSAL_PREFIX,
    Operation,
    PermissionDecision,
    PermissionLevel,
    authorize_operation,
)

logger = logging.getLogger(__name__)

WRITE_WAIT_SECONDS = 30.0
"""How long a mutation waits for the project writer lock before giving up.

Waiting is bounded rather than indefinite: a lost or wedged holder can never
freeze a run, and the loser reports back instead of hanging. The invariant is
unaffected - declining is not overlapping.
"""

DEFERRED_PREFIX = "Deferred:"
"""The two openings a denial can have, and the whole vocabulary of one.

``REFUSAL_PREFIX`` is imported from :mod:`terminus.permissions`, which is where a
permission refusal is worded; the pair is declared together here.

Permission refusal is produced by :meth:`PermissionDecision.refusal_message`;
contention is produced by :meth:`MutationGrant.deferred`. Both are returned to
the model as an ordinary tool result, which is right - the model has to read it
and choose - and both are *results*, not attempts that changed something.

:func:`was_performed` exists so that anything downstream which needs to tell a
real mutation from a refused one reads these two constants instead of
re-recognising the wording. That is the whole reason they are named.
"""


def was_performed(result: str) -> bool:
    """Did a mutating tool call actually change anything?

    True for every ordinary result. False when the result is a permission refusal
    or a contention deferral - the two cases in which the guard declined, the tool
    body never ran, and whatever path it named is untouched.

    Lives beside the guard because the guard is what produces both strings, and
    every tool reaches both through it.
    """
    text = (result or "").lstrip()
    return not text.startswith((REFUSAL_PREFIX, DEFERRED_PREFIX))


class _LockRegistry:
    """One lock per workspace, created on first use.

    Holds only synchronisation primitives. It carries no authority, no policy and
    no data, so it is not execution state - the permission decision and the
    execution identity stay in their own modules.
    """

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def for_workspace(self, workspace: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(workspace)
            if lock is None:
                lock = threading.Lock()
                self._locks[workspace] = lock
            return lock

    def held(self, workspace: str) -> bool:
        return self.for_workspace(workspace).locked()

    def tracked(self) -> int:
        with self._guard:
            return len(self._locks)


_registry = _LockRegistry()


def is_writing(workspace: str) -> bool:
    """Is some execution currently mutating this workspace?"""
    return _registry.held(str(workspace))


def _current_workspace() -> str:
    """The workspace being mutated: the active execution's, else this project.

    Real work always runs inside an ``ExecutionContext``; the fallback keeps
    direct tool use (tests, and any future caller without a scope) writing to the
    same project the process is in, so it is still coordinated rather than
    silently exempt.
    """
    from terminus.execution import current_execution
    from terminus.workspace import project_root

    running = current_execution()
    if running is not None:
        return str(running.workspace)
    return str(project_root())


@dataclass(frozen=True)
class MutationGrant:
    """Outcome of asking to mutate the workspace.

    ``refused`` is a model-visible refusal string, or None to proceed.
    ``decision`` is the permission decision that was applied, so callers can
    report the level without authorising twice.

    ``wanted_lock`` separates "this operation did not need the project lock" (a
    read, or a refusal) from "it needed the lock and could not get it". Without
    that distinction a plain read would be reported as a busy project.
    """

    decision: PermissionDecision
    locked: bool
    wanted_lock: bool = False

    @property
    def refused(self) -> str | None:
        if not self.decision.allowed:
            return self.decision.refusal_message()
        return None

    @property
    def deferred(self) -> str | None:
        """Set when permission was granted but the project was busy."""
        if self.decision.allowed and self.wanted_lock and not self.locked:
            return (
                f"{DEFERRED_PREFIX} another task is currently writing this project, "
                "so this operation was not performed. Nothing was changed. Wait for "
                "that task to finish, then try again."
            )
        return None


@contextmanager
def project_write_guard(
    operation: Operation,
    target: str = "",
    command: str | None = None,
    context: str | None = None,
):
    """Authorise *operation*, and hold the project writer lock while it runs.

    Yields a ``MutationGrant``; check ``grant.refused`` and proceed only if it is
    None:

        with project_write_guard(Operation.WRITE, target=path) as grant:
            if grant.refused:
                return grant.refused
            ...perform the mutation...

    The grant carries the decision so a caller can log the level that was applied
    without having to authorise a second time (which would re-run human approval).

    A READ_ONLY operation is authorised but never takes the lock, so read-only
    work in one task never delays another. A refused operation never takes the
    lock either, so a permission failure cannot hold up the project.

    The lock is released in a ``finally``, so a tool that raises, times out, or is
    interrupted cannot leave the project permanently locked.

    Scope note: the lock is process-local, so it serialises the tasks of *one*
    Terminus process. ``TaskStore.recover_interrupted_tasks`` already documents
    that a project is expected to have a single orchestrator; that assumption is
    preserved here rather than silently extended. Two Terminus processes pointed
    at the same directory remain uncoordinated by design, not by oversight.

    *context* defaults to the active execution's label, so a refusal names the task
    or agent that hit it rather than looking like a global rule. Tools call this
    directly; there is no second wrapper to forget to use.
    """
    if context is None:
        from terminus.execution import current_execution

        running = current_execution()
        context = running.label if running else None

    decision: PermissionDecision = authorize_operation(
        operation, target, command, context
    )
    if not decision.allowed:
        yield MutationGrant(decision, locked=False, wanted_lock=False)
        return
    if decision.level is PermissionLevel.READ_ONLY:
        yield MutationGrant(decision, locked=False, wanted_lock=False)
        return

    workspace = _current_workspace()
    lock = _registry.for_workspace(workspace)
    if not lock.acquire(timeout=WRITE_WAIT_SECONDS):
        logger.warning("Workspace busy, declining mutation: %s", workspace)
        yield MutationGrant(decision, locked=False, wanted_lock=True)
        return
    try:
        yield MutationGrant(decision, locked=True, wanted_lock=True)
    finally:
        lock.release()


def tracked_workspaces() -> int:
    """How many workspaces have a lock. Test/diagnostic helper."""
    return _registry.tracked()
