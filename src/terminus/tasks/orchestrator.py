import asyncio
import json
from pathlib import Path
from rich.console import Console

from terminus.config import CONFIG
from terminus.context.indexers.factory import get_or_create_indexer
from terminus.observability.logging import get_logger
from terminus.tasks.approval import present_plan_for_approval
from terminus.tasks.planner import create_plan, validate_plan
from terminus.tasks.task_store import ProjectStatus, TaskStatus, TaskStore
from terminus.tasks.executor import run_subtask_agent

logger = get_logger(__name__)
console = Console()


class RecoveryManager:
    """Handle resuming interrupted or failed projects."""

    def __init__(self, store: TaskStore):
        self.store = store

    def recover(self, project_id: str) -> dict[str, int]:
        interrupted = self.store.recover_interrupted_tasks(project_id)
        retried = self.store.reset_retryable_failed_tasks(project_id)
        return {"interrupted": interrupted, "retried": retried}


class TaskOrchestrator:
    """
    Main execution loop: finds tasks whose dependencies are met,
    claims them atomically, and dispatches them to subtask agents.

    Serial by default (max_concurrent=1) for cost control and determinism.
    """

    def __init__(self, store: TaskStore, max_concurrent: int = 1) -> None:
        self.store = store
        self.max_concurrent = max_concurrent

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

            if pending == 0 and in_progress == 0:
                _print_final_summary(progress)
                self._explain_blocked_state(project_id)
                break

            ready = self.store.get_ready_tasks(project_id)

            if not ready:
                if in_progress > 0:
                    console.print("[yellow]Waiting for in-progress tasks to complete...[/yellow]")
                else:
                    self._explain_blocked_state(project_id)
                    break
                await asyncio.sleep(5)
                continue

            batch = ready[: self.max_concurrent]
            await asyncio.gather(*[self._execute(task) for task in batch])

        self.store.finalize_project_status(project_id)

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

        tasks = self.store._get_all_tasks(project_id)
        for task in tasks:
            if task["status"] == TaskStatus.FAILED.value:
                console.print(
                    f"  - {task['id']} permanently failed: {task.get('error', 'unknown error')}"
                )

    async def _execute(self, task: dict) -> None:
        """Execute a single task: claim, execute, update DB."""
        console.print(
            f"[bold yellow]Executing task {task.get('id')}: {task.get('description')}[/bold yellow]"
        )
        if not self.store.claim_task(task["project_id"], task["id"]):
            console.print(
                f"[bold red]Task {task['id']} already claimed by another process. Skipping.[/bold red]"
            )
            return

        try:
            dep_ids = json.loads(task.get("depends_on", "[]"))
            dep_outputs = self.store.get_dep_results(task["project_id"], dep_ids)
            result = await run_subtask_agent(task, dep_outputs)

            self.store.complete_task(task["project_id"], task["id"], result)
            console.print(f"[bold green]Task {task['id']} completed[/bold green]")
        except Exception as e:
            logger.error(f"Failed to execute task {task['id']}: {e}")
            console.print(f"[bold red]Failed to execute task {task['id']}: {e}[/bold red]")
            status = self.store.fail_task(task["project_id"], task["id"], str(e))
            if status == TaskStatus.PENDING.value:
                console.print(
                    f"[yellow]Task {task['id']} will be retried automatically[/yellow]"
                )


async def _run_orchestration(store: TaskStore, project_id: str) -> None:
    orchestrator = TaskOrchestrator(store, max_concurrent=1)
    await orchestrator.run(project_id)

    console.print("[bold green]Re-indexing generated files[/bold green]")
    try:
        get_or_create_indexer(str(Path.cwd()))
        console.print("[bold green]Index updated[/bold green]")
    except Exception as e:
        logger.error(f"Failed to re-index: {e}")
        console.print("[bold red]Failed to re-index[/bold red]")


async def handle_plan_command(goal: str) -> None:
    """
    Full /plan flow - entry point called by cli.py.

    /plan continue  -> resume the newest resumable project
    /plan <goal>    -> always create a new project
    """
    db_path = CONFIG.get("tasks", {}).get("db_path", ".terminus/tasks/tasks.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    store = TaskStore(db_path)

    if goal.strip().lower() == "continue":
        project_id = store.get_resumable_project()
        if not project_id:
            console.print(
                "[bold yellow]No resumable project found. "
                "Use /plan <goal> to start a new project.[/bold yellow]"
            )
            return

        console.print(f"[yellow]Resuming project {project_id}[/yellow]")
        recovery = RecoveryManager(store)
        stats = recovery.recover(project_id)
        if stats["interrupted"]:
            console.print(
                f"[yellow]Recovered {stats['interrupted']} interrupted task(s)[/yellow]"
            )
        if stats["retried"]:
            console.print(
                f"[yellow]Reset {stats['retried']} retryable failed task(s)[/yellow]"
            )

        await _run_orchestration(store, project_id)
        return

    console.print("[bold yellow]Planning new project...[/bold yellow]")
    extra_content = ""
    approved_plan = None
    while approved_plan is None:
        raw_plan = create_plan(goal, extra_content)
        validate_plan(raw_plan)
        approved_plan = present_plan_for_approval(raw_plan)
        if approved_plan is None:
            extra_content = input("What should be changed or added in the plan? : \n").strip()
            console.print("\n Re-planning with your feedback\n", style="cyan")

    project_id = store.create_project(goal, approved_plan)
    console.print(f"[bold green]Project created: {project_id}[/bold green]")
    await _run_orchestration(store, project_id)


def _print_final_summary(progress: dict[str, int]) -> None:
    completed = progress.get("completed", 0)
    pending = progress.get("pending", 0)
    failed = progress.get("failed", 0)
    total = completed + pending + failed
    console.print("\n[bold green]Final Summary:[/bold green]")
    console.print(f"[bold green]Completed: {completed}/{total}[/bold green]")
    console.print(f"[bold yellow]Pending: {pending}/{total}[/bold yellow]")
    console.print(f"[bold red]Failed: {failed}/{total}[/bold red]")
