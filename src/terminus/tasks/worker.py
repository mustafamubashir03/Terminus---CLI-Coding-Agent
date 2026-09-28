"""Executing one task: the reusable primitive behind /plan, and future spawns.

This is the only supported way to run a task. It is deliberately independent of
how a task came to exist - the CLI's ``/plan`` loop uses it today, and a
``SpawnAgentTool`` would call the same function with the same arguments.

The split of responsibility:

    tasks/task_store.py   persistence: projects, tasks, claims, transitions
    tasks/planner.py      what to do (a plan)
    tasks/orchestrator.py scheduling: which task runs next
    tasks/worker.py       THIS MODULE: run one task, return one result
    tasks/executor.py     the machinery: agent construction, streaming, judging

A worker is not a conversation. It gets no ``/ask`` checkpoint, no chat history
and no second message store: its state is the task row and its result. That
keeps a task's durable footprint small and means a retry starts clean.
"""

from __future__ import annotations

from dataclasses import dataclass

from terminus.config import CONFIG
from terminus.execution import ExecutionContext, execution_scope, task_context
from terminus.observability.logging import get_logger
from terminus.permissions import PermissionLevel, PermissionPolicy
from terminus.tasks.errors import FailureInfo, classify_failure, format_failure

logger = get_logger(__name__)


def worker_permission_policy() -> PermissionPolicy:
    """The policy a task worker runs under.

    WRITE is allowed because executing an approved plan is the entire purpose of
    a worker, and the user approved the plan before any task ran. DESTRUCTIVE is
    denied with no approver: a worker has nobody to ask, so it must not be able
    to do it at all.

    This is returned, not installed. The caller passes it to
    :func:`execute_task`, which activates it for that execution only, so one
    worker can never change another's permissions or /ask's.
    """
    return PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    )


@dataclass(frozen=True)
class TaskResult:
    """The outcome of one attempt at one task.

    ``success`` distinguishes a completed task from a failed one without the
    caller having to interpret an exception. ``output`` is the worker's own
    summary; ``error`` is a formatted, already-classified failure. ``attempt`` is
    1-based and reflects the task's retry budget, so a caller does not have to
    re-read the row to know which attempt this was.
    """

    success: bool
    output: str = ""
    error: str | None = None
    retryable: bool = False
    category: str = ""
    attempt: int = 1
    failure: FailureInfo | None = None

    @classmethod
    def ok(cls, output: str, attempt: int = 1) -> "TaskResult":
        return cls(success=True, output=output, attempt=attempt)

    @classmethod
    def failed(cls, failure: FailureInfo, attempt: int = 1,
               provider: str | None = None, model: str | None = None) -> "TaskResult":
        return cls(
            success=False,
            error=format_failure(failure, provider, model),
            retryable=failure.retryable,
            category=failure.category,
            attempt=attempt,
            failure=failure,
        )


async def execute_task(
    task: dict,
    *,
    workspace,
    attempt: int,
    dep_outputs: list[dict] | None = None,
    feedback: str = "",
    policy: PermissionPolicy | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> TaskResult:
    """Run one task to completion in *workspace* and return its result.

    ``attempt`` is required and is supplied by the caller from the store
    (``claim_task`` returns it). The worker deliberately does not derive or keep
    an attempt number of its own: the persistent task row is the single record
    of how many times a task has been tried, and a second rule living here could
    silently disagree with it.

    Validates that the task's workspace is this project, activates that
    execution's permissions for the duration, runs a fresh worker agent, and has
    the judge verify the output against the task's acceptance criteria.

    ``provider``/``model`` default to the active runtime configuration - the same
    routing and fallbacks /ask uses.

    Every expected failure comes back as ``TaskResult(success=False, ...)`` so a
    scheduler can decide between retrying and giving up. System-level signals
    (cancellation, KeyboardInterrupt) still propagate.

    A task planned for one project cannot be run against another: a mismatched
    workspace raises :class:`~terminus.execution.WorkspaceMismatch` rather than
    quietly editing the wrong tree.
    """
    from terminus.tasks.executor import _run_worker_agent, judge_task

    provider = provider or CONFIG["llm"]["provider"]
    model = model or CONFIG["llm"]["model"]

    context: ExecutionContext = task_context(
        task_id=task.get("id", "?"),
        project_id=task.get("project_id"),
        workspace=workspace,
        policy=policy or worker_permission_policy(),
    )

    try:
        with execution_scope(context):
            output, deliverable_contents = await _run_worker_agent(
                task, dep_outputs or [], feedback=feedback,
                provider=provider, model=model,
            )
    except Exception as exc:
        failure = classify_failure(exc, provider=provider, model=model)
        logger.warning(
            "Task %s attempt %s failed (%s): %s",
            context.label, attempt, failure.category, failure.message,
        )
        return TaskResult.failed(failure, attempt, provider, model)

    try:
        with execution_scope(context):
            verdict = await judge_task(task, output, deliverable_contents)
    except Exception as exc:
        failure = classify_failure(exc, provider=provider, model=model)
        logger.warning(
            "Task %s attempt %s failed verification (%s): %s",
            context.label, attempt, failure.category, failure.message,
        )
        return TaskResult.failed(failure, attempt, provider, model)

    if not verdict.passed:
        failure = FailureInfo(
            message=f"judge rejected the output: {verdict.reason}",
            retryable=True,
            category="judge_rejected",
        )
        return TaskResult.failed(failure, attempt, provider, model)

    return TaskResult.ok(output, attempt)
