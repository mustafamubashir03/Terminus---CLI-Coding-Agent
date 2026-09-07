from terminus.tasks.planner import ExecutionPlan
from rich.console import Console
from rich.table import Table

console = Console()

def _render_plan(plan : ExecutionPlan):
    """ Render the plan as a Rich table """
    table = Table(title="Execution Plan")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Description")
    table.add_column("Status", justify="right")
    
    for task in plan.tasks:
        table.add_row(
            task.id,
            task.description,
            task.status
        )
    console.print(table)


def present_plan_for_approval(plan : ExecutionPlan)->ExecutionPlan | None:
    """ 
    Render the plan as a Rich table and prompt the user to :
    [A] Approve - returns the plan as-is
    [M] Modify - edit a task description in-place and re-render
    [R] Reject - returns None so the caller re-plans with feedback

    Loop continues until the user approves or rejects.
    """
    while True:
        _render_plan(plan)
        choice = input("[A] Approve | [M] Modify | [R] Reject: ").strip().upper()
        if choice == "A":
            return plan
        elif choice == "M":
            # TODO: implement modify
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
            
            pass
        elif choice == "R":
            return None
        else:
            console.print("[yellow]Invalid choice. Please try again.[/yellow]")
            
        