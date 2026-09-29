"""Human approval of a generated plan, before any task runs.

This is the one mandatory stop in the /plan flow. A plan becomes a set of tasks
that will edit files, and the model that wrote it is the same model that will
report on it, so the user gets to see the task list and say no before anything is
created.

Returning ``None`` means rejected. That is a first-class outcome, not an error:
the caller re-plans with feedback, and if there is no terminal to ask for
feedback, it stops rather than proceeding unapproved.

Requires a real terminal. ``human_is_present`` gates it, because a blocking
``input()`` on a pipe or in CI would hang forever rather than decline.
"""

from terminus.agent.factory import human_is_present
from terminus.tasks.planner import ExecutionPlan
from rich.console import Console
from rich.table import Table

console = Console()



def present_plan_for_approval(plan : ExecutionPlan)->ExecutionPlan | None:
    """
    Render the plan as a Rich table and prompt the user to :
    [A] Approve - returns the plan as-is
    [M] Modify - edit a task description in-place and re-render
    [R] Reject - returns None so the caller re-plans with feedback

    Loop continues until the user approves or rejects.

    Without an interactive terminal there is nobody to ask, and ``input()`` would
    block forever on a pipe or in CI. In that case the plan is not approved:
    returning None means rejected, which is the safe default - a plan must never
    start executing because a prompt was answered by an empty pipe.
    """
    if not human_is_present():
        console.print(
            "[bold red]No interactive terminal available to approve the plan.[/bold red]\n"
            f"  {len(plan.tasks)} task(s) were planned but will NOT be executed.\n"
            "Run Terminus interactively to review and approve a plan."
        )
        return None

    while True:
        _render_plan(plan)
        choice = input("[A] Approve | [M] Modify | [R] Reject: ").strip().upper()
        if choice == "A":
            return plan
        if choice == "M":
            task_id = input("Enter task ID to modify: ").strip()
            task = next((
                t
                for t in plan.tasks
                if t.id == task_id
            ),None)
            if not task:
                console.print(f"[red]Task {task_id} not found[/red]")
                continue
            console.print(f"Current description:\n{task.description}")
            new_task_description = input("Enter new task description: ").strip()
            if new_task_description:
                task.description = new_task_description
            console.print("Task updated.")
        elif choice == "R":
            return None
        else:
            console.print("[yellow]Invalid choice. Please try again.[/yellow]")


def _render_plan(plan : ExecutionPlan):
    """ Render the plan as a Rich table """
    console.print(f"[bold cyan]Plan: {plan.project_name}")
    console.print(f"[cyan]Goal: {plan.goal_summary}")
    console.print(f"[cyan]Tech Stack: {plan.tech_stack}")
    console.print(f"[cyan]Risks: {plan.risks}")
    console.print(f"[cyan]Assumptions: {plan.assumptions}")
    table = Table(title="Execution Plan")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Description")
    table.add_column("Type", justify="right")

    for task in plan.tasks:
        table.add_row(
            task.id,
            task.description,
            task.task_type.value
        )
    console.print(table)
