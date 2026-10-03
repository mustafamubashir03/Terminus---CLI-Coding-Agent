import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from rich.console import Console

from terminus.config import CONFIG
from terminus.agent.factory import human_is_present
from terminus.observability.logging import get_logger
from terminus.ownership import (
    OwnerInfo,
    OwnershipConflict,
    current_ownership,
    lock_dir_for,
)
from terminus.tasks.approval import present_plan_for_approval
from terminus.tasks.errors import classify_failure, format_failure
from terminus.tasks.planner import create_plan, validate_plan
from terminus.tasks.task_store import (
    MAX_RECOVERY_CYCLES,
    ProjectStatus,
    TaskStatus,
    TaskStore,
)
from terminus.tasks.worker import execute_task
from terminus.workspace import project_root

logger = get_logger(__name__)
console = Console()


def acquire_project_ownership(store: TaskStore, project_id: str) -> OwnerInfo:
    """Take sole ownership of *project_id*, or raise :class:`OwnershipConflict`.

    Ref-counted, so the nesting in ``_run_orchestration`` and the command flow
    can each take and give back without either releasing early.
    """
    return current_ownership().acquire(lock_dir_for(store.db_path), project_id)


def release_project_ownership(project_id: str) -> None:
    """Give back one level of ownership this process holds."""
    current_ownership().release(project_id)


def holds_project_ownership(project_id: str) -> bool:
    """Is this process currently the owner of *project_id*?"""
    return current_ownership().holds(project_id)


@contextmanager
def project_ownership(store: TaskStore, project_id: str) -> Iterator[None]:
    """Hold sole ownership of *project_id* for the duration of the block.

    Yields normally, or raises :class:`OwnershipConflict` before yielding if
    another live Terminus process already owns the project. Release is guaranteed
    even if the body raises, so a failure inside the block cannot leave the
    project permanently unownable.

    Ownership is ref-counted, so nesting this is safe: the inner exit gives back
    one level rather than dropping the lock the outer one still needs.
    """
    acquire_project_ownership(store, project_id)
    try:
        yield
    finally:
        release_project_ownership(project_id)


def recover_project(store: TaskStore, project_id: str) -> dict[str, int]:
    """Re-open work that a previous run left in flight, and reset hard failures.

    Two distinct operations, reported separately because the user needs to know
    which happened: *interrupted* tasks were mid-flight when the process died,
    and *retried* tasks had exhausted their automatic budget. The second is
    subject to the project's recovery-cycle cap; see
    :meth:`TaskStore.reset_failed_tasks_for_recovery`.

    A function rather than a manager object: it holds no state of its own beyond
    the store it is handed, and the only invariant that matters here - own the
    project before calling it - is enforced by :func:`project_ownership` at the
    call site, where it can be seen.
    """
    return {
        "interrupted": store.recover_interrupted_tasks(project_id),
        "retried": store.reset_failed_tasks_for_recovery(project_id),
    }


MAX_CONCURRENT_TASKS = 4
"""Hard ceiling on concurrently executing tasks.

A config file is untrusted input as far as this process is concerned, so the
value is clamped rather than trusted. 4 keeps a run inside a normal provider
concurrency window while leaving enough headroom to be useful.
"""


def clamp_max_concurrent(value: object) -> int:
    """Coerce a configured ``tasks.max_concurrent`` into a safe integer.

    Default 1 (serial) when absent or unintelligible, capped at
    MAX_CONCURRENT_TASKS, and never below 1.
    """
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return max(1, min(parsed, MAX_CONCURRENT_TASKS))


def configured_max_concurrent() -> int:
    """The opt-in concurrency for this run, from the existing tasks config.

    Serial unless a project explicitly sets ``tasks.max_concurrent`` above 1.
    """
    return clamp_max_concurrent(
        CONFIG.get("tasks", {}).get("max_concurrent", 1)
    )


class TaskOrchestrator:
    """
    Main execution loop: finds tasks whose dependencies are met,
    claims them atomically, and dispatches them to subtask agents.

    Serial by default (max_concurrent=1) for cost control and determinism.
    """

    def __init__(self, store: TaskStore, max_concurrent: int = 1) -> None:
        self.store = store
        self.max_concurrent = clamp_max_concurrent(max_concurrent)

    async def run(self, project_id: str) -> None:
        """Loop until all tasks are completed, failed, or blocked."""
        self.store.update_project_status(project_id, ProjectStatus.IN_PROGRESS.value)
        console.print(f"[bold yellow]Starting orchestration for project {project_id}[/bold yellow]")

        while True:
            progress = self.store.get_progress(project_id)
            pending = progress.get("pending", 0)
            in_progress = progress.get("in_progress", 0)
            completed = progress.get("completed", 0)
            failed = progress.get("failed", 0)
            total = sum(progress.values())

            console.print(
                f"[bold blue]Pending: {pending}/{total} pending, "
                f"{in_progress}/{total} in progress, {completed}/{total} completed, "
                f"{failed}/{total} failed[/bold blue]"
            )

            # No task can ever become ready: nothing pending or in-progress.
            if pending == 0 and in_progress == 0:
                _print_final_summary(progress)
                self._explain_blocked_state(project_id)
                break

            ready = self.store.get_ready_tasks(project_id)

            if not ready:
                # If in-progress tasks exist, they may unblock pending tasks when
                # they complete - wait instead of terminating prematurely.
                if in_progress > 0:
                    console.print("[yellow]Waiting for in-progress tasks to complete...[/yellow]")
                    await asyncio.sleep(5)
                    continue
                # No ready tasks and nothing in progress: either everything is
                # blocked by failed tasks or the graph is otherwise stuck.
                self._print_final_summary(progress)
                self._explain_blocked_state(project_id)
                break

            batch = ready[: self.max_concurrent]
            # return_exceptions=True, and _execute contains its own failures.
            # Without both, one unexpected error propagated out of gather,
            # skipped finalize_project_status below, and left the project row
            # stuck at in_progress - where `/plan continue` would then resume
            # nothing, forever.
            results = await asyncio.gather(
                *[self._execute(task) for task in batch],
                return_exceptions=True,
            )
            for task, outcome in zip(batch, results, strict=True):
                if isinstance(outcome, BaseException):
                    self._record_unexpected_failure(task, outcome)

            # Brief pause between task batches to avoid bursting free-tier
            # rate limits (OpenRouter 429s queue requests beyond the timeout).
            if ready:
                await asyncio.sleep(3)

        self.store.finalize_project_status(project_id)

    def _record_unexpected_failure(self, task: dict, exc: BaseException) -> None:
        """Fail a task whose execution raised rather than returned.

        An exception here is an infrastructure or data fault - a corrupted row,
        a database error - not the agent's answer, so it is logged and recorded
        as a non-retryable task failure. Retrying a fault that is not the model's
        would burn the whole retry budget on the same broken input.
        """
        task_id = task.get("id", "?")
        detail = f"[failed] unexpected {type(exc).__name__}: {exc}"
        logger.error("Task %s raised out of execution: %s: %s",
                     task_id, type(exc).__name__, exc, exc_info=True)
        console.print(f"[bold red]Task {task_id} could not be run: {exc}[/bold red]")
        try:
            self.store.fail_task(task["project_id"], task_id, detail, force=True)
        except Exception as store_exc:  # pragma: no cover - the store is also broken
            logger.error("Could not record the failure of task %s: %s", task_id, store_exc)

    def _explain_blocked_state(self, project_id: str) -> None:
        blocked = self.store.get_blocked_by_failed(project_id)
        if not blocked:
            return
        console.print(
            "[bold red]Some tasks are blocked by permanently failed dependencies:[/bold red]"
        )
        for item in blocked:
            task = item["task"]
            deps = ", ".join(item["blocked_by"])
            console.print(f"  - {task['id']} blocked by: {deps}")

        tasks = self.store.get_all_tasks(project_id)
        for task in tasks:
            if task["status"] == TaskStatus.FAILED.value:
                console.print(
                    f"  - {task['id']} permanently failed: {task.get('error', 'unknown error')}"
                )

    async def _execute(self, task: dict) -> None:
        """Run one task and record the outcome: completed, or failed/retry."""
        console.print(
            f"[bold yellow]Executing task {task.get('id')}: {task.get('description')}[/bold yellow]"
        )
        # The store owns the attempt number: it derives it in the same statement
        # that claims the task, so the scheduler and the worker cannot disagree.
        attempt = self.store.claim_task(task["project_id"], task["id"])
        if not attempt:
            console.print(
                f"[bold red]Task {task['id']} already claimed by another process. Skipping.[/bold red]"
            )
            return

        dep_ids = json.loads(task.get("depends_on", "[]"))
        dep_outputs = self.store.get_dep_results(task["project_id"], dep_ids)

        # The project records the workspace it was planned for. A task must run
        # there, and never here if the two disagree.
        workspace = self.store.get_project_workspace(task["project_id"])
        result = await execute_task(
            task,
            workspace=workspace or project_root(),
            attempt=attempt,
            dep_outputs=dep_outputs,
            feedback=task.get("error") or "",
        )

        if result.success:
            self.store.complete_task(task["project_id"], task["id"], result.output)
            console.print(f"[bold green]Task {task['id']} completed[/bold green]")
            return

        label = "[retryable]" if result.retryable else "[non-retryable]"
        logger.error("Failed to execute task %s (attempt %s) %s: %s",
                     task["id"], attempt, label, result.error)
        console.print(
            f"[bold red]Failed to execute task {task['id']} (attempt {attempt}) "
            f"{label}: {result.error}[/bold red]"
        )
        status = self.store.fail_task(
            task["project_id"], task["id"], f"{label} {result.error}",
            force=not result.retryable,
        )
        if status == TaskStatus.PENDING.value:
            console.print(
                f"[yellow]Task {task['id']} will be retried automatically[/yellow]"
            )
            if result.category == "rate_limit":
                delay = CONFIG.get("tasks", {}).get("rate_limit_backoff_seconds", 30)
                if delay > 0:
                    console.print(
                        f"[yellow]Waiting {delay}s before retrying provider rate limit[/yellow]"
                    )
                    await asyncio.sleep(delay)


async def _run_orchestration(store: TaskStore, project_id: str) -> None:
    orchestrator = TaskOrchestrator(
        store, max_concurrent=configured_max_concurrent()
    )
    await orchestrator.run(project_id)

    console.print("[bold green]Re-indexing generated files[/bold green]")
    # Indexing is sync and may connect to external services (e.g. Qdrant), so it is
    # bounded: a hung indexer must not block the CLI forever, and must never corrupt
    # project/task state (it runs outside any task DB transaction).
    index_timeout = CONFIG.get("tasks", {}).get("index_timeout_seconds", 300)
    try:
        from terminus.context.indexers.reindexer import incremental_reindex

        _vs, result = await asyncio.wait_for(
            asyncio.to_thread(incremental_reindex, str(Path.cwd())),
            timeout=index_timeout,
        )
        if result.files_added or result.files_modified or result.files_deleted:
            console.print(
                f"[bold green]Index updated: +{result.files_added} new, "
                f"~{result.files_modified} modified, "
                f"-{result.files_deleted} deleted "
                f"({result.chunks_added} chunks) in {result.elapsed_seconds:.1f}s[/bold green]"
            )
        else:
            console.print("[bold green]Index already up to date[/bold green]")
    except asyncio.TimeoutError:
        logger.error("Re-indexing timed out after %ss", index_timeout)
        console.print("[bold red]Re-indexing timed out (project state is unaffected)[/bold red]")
    except Exception as e:
        logger.error("Failed to re-index: %s: %s", type(e).__name__, e, exc_info=True)
        console.print("[bold red]Failed to re-index[/bold red]")


MAX_PLAN_ATTEMPTS = 3
"""How many times plan generation is retried before giving up.

A provider failure here is usually transient (a rate limit, a dropped
connection), so retrying is worth it. Three is enough to ride out a throttle and
small enough that a systematically unplannable goal fails in a reasonable time.
"""


async def _plan_with_approval(goal: str) -> object | None:
    """Plan *goal* and get it approved. Returns the plan, or None if there is none.

    A rejection asks what to change and re-plans with the answer, so the feedback
    string stays local to this function.

    Re-planning needs a person. Without a terminal there is nobody to ask, so a
    rejected plan stops rather than blocking forever on a pipe or in CI.
    """
    feedback = ""
    provider = CONFIG.get("llm", {}).get("provider")
    for attempt in range(MAX_PLAN_ATTEMPTS):
        try:
            raw_plan = await asyncio.to_thread(create_plan, goal, feedback)
            validate_plan(raw_plan)
            approved = present_plan_for_approval(raw_plan)
            if approved is not None:
                return approved
            if not human_is_present():
                console.print(
                    "[bold red]Plan rejected and no interactive terminal is "
                    "available to ask what to change it. Not creating a project.[/bold red]"
                )
                return None
            feedback = input("What should be changed or added in the plan? : \n").strip()
            console.print("\n Re-planning with your feedback\n", style="cyan")
        except Exception as exc:
            failure = classify_failure(exc, provider=provider)
            detail = format_failure(failure, provider=provider)
            logger.error("Plan generation failed: %s", detail)
            if not failure.retryable:
                # A missing key, a rejected key, an unknown model: the same
                # request will fail the same way. It was logging
                # "retryable=false" and then retrying anyway, which turned one
                # clear error into three identical ones and buried the cause.
                console.print(f"[bold red]Plan generation failed: {detail}[/bold red]")
                if "is not set" in detail or "API key" in detail:
                    console.print(
                        "[yellow]No API key for this provider. Add one with "
                        "[cyan]terminus setup[/cyan], then try again.[/yellow]"
                    )
                return None
            if attempt < MAX_PLAN_ATTEMPTS - 1:
                console.print(
                    f"[yellow]Plan failed, retrying ({attempt + 2}/{MAX_PLAN_ATTEMPTS})...[/yellow]"
                )
            else:
                console.print(f"[bold red]Plan generation failed: {detail}[/bold red]")
    return None


async def handle_plan_command(goal: str) -> None:
    """The /plan flow. Two entry shapes, and nothing else:

    ``/plan continue``  resume the newest project that still has recoverable work
    ``/plan <goal>``    always plan something new

    Both branches end the same way - take ownership, run, release - and that is
    the point of the ``project_ownership`` scope. Recovery resets tasks that were
    in flight when a previous process died, which is only safe if no live process
    is working on them, so ownership is taken *before* any state is read for
    mutation rather than after.

    Neither branch raises on a lost race: a conflict is reported and the command
    returns, because "another Terminus is already doing this" is an ordinary
    situation the user can act on, not a crash.
    """
    db_path = CONFIG.get("tasks", {}).get("db_path", ".terminus/tasks/tasks.db")
    store = TaskStore(db_path)

    if goal.strip().lower() == "continue":
        project_id = store.get_resumable_project()
        if not project_id:
            console.print(
                "[bold yellow]No resumable project found. "
                "Use /plan <goal> to start a new project.[/bold yellow]"
            )
            return

        # A project records the directory it was planned for. Executing it in a
        # different directory would run its workers against the wrong tree while
        # the task state claimed otherwise, so refuse rather than guess.
        if not store.workspace_matches(project_id):
            planned_for = store.get_project_workspace(project_id)
            console.print(
                f"[bold red]This project belongs to a different workspace.[/bold red]\n"
                f"  planned for : {planned_for}\n"
                f"  current dir : {project_root()}\n"
                f"Run Terminus from {planned_for} to continue it, so its tasks "
                f"edit the files they were planned for."
            )
            return

        console.print(f"[yellow]Resuming project {project_id}[/yellow]")

        # Ownership is taken BEFORE any recovery, and before any task state is
        # read for mutation. recover_interrupted_tasks() resets in_progress tasks
        # on the assumption that nobody else is working on them; taking the lock
        # first is the only thing that makes that assumption true. Doing it the
        # other way round would let this process reset a live orchestrator's
        # running task before discovering it was not alone.
        try:
            with project_ownership(store, project_id):
                _resume_existing(store, project_id)
                await _run_orchestration(store, project_id)
        except OwnershipConflict as conflict:
            console.print(f"[bold red]{conflict}[/bold red]")
        return

    console.print("[bold yellow]Planning new project...[/bold yellow]")
    approved_plan = await _plan_with_approval(goal)
    if approved_plan is None:
        console.print("[red]No approved plan; not creating a project.[/red]")
        return

    project_id = store.create_project(goal, approved_plan)
    console.print(f"[bold green]Project created: {project_id}[/bold green]")

    # A brand-new project cannot collide with an existing one, but ownership is
    # still taken here so the run is protected for its whole lifetime rather than
    # only from the next `/plan continue`. A conflict is still honoured: it would
    # mean someone else is already orchestrating this project id.
    try:
        with project_ownership(store, project_id):
            await _run_orchestration(store, project_id)
    except OwnershipConflict as conflict:
        console.print(f"[bold red]{conflict}[/bold red]")


def _resume_existing(store: TaskStore, project_id: str) -> None:
    """Report and re-open whatever a previous run left behind."""
    stats = recover_project(store, project_id)
    if stats["interrupted"]:
        console.print(
            f"[yellow]Recovered {stats['interrupted']} interrupted task(s)[/yellow]"
        )
    if stats["retried"]:
        console.print(
            f"[yellow]Reset {stats['retried']} permanently failed task(s) "
            f"for manual recovery (recovery cycle "
            f"{store.get_recovery_cycles(project_id)}/{MAX_RECOVERY_CYCLES})[/yellow]"
        )
    elif store.get_recovery_cycles(project_id) >= MAX_RECOVERY_CYCLES:
        # The cap is the reason nothing was reset; say so rather than leaving
        # the user to wonder why their failed task did not run.
        console.print(
            f"[bold red]This project has already used all {MAX_RECOVERY_CYCLES} "
            f"recovery cycles, so its failed tasks will not be retried again. "
            f"Fix the underlying problem, or start a new plan.[/bold red]"
        )
    if not stats["interrupted"] and not stats["retried"]:
        console.print("[yellow]No interrupted or failed tasks to recover.[/yellow]")


def _print_final_summary(progress: dict[str, int]) -> None:
    completed = progress.get("completed", 0)
    pending = progress.get("pending", 0)
    failed = progress.get("failed", 0)
    in_progress = progress.get("in_progress", 0)
    total = completed + pending + failed + in_progress
    console.print("\n[bold green]Final Summary:[/bold green]")
    console.print(f"[bold green]Completed: {completed}/{total}[/bold green]")
    if pending:
        console.print(f"[bold yellow]Pending: {pending}/{total}[/bold yellow]")
    if in_progress:
        console.print(f"[bold yellow]In progress: {in_progress}/{total}[/bold yellow]")
    if failed:
        console.print(f"[bold red]Failed: {failed}/{total}[/bold red]")
