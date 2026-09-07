from __future__ import annotations
from terminus.tasks.planner import create_plan
import asyncio
import json

from rich.console import Console

from pathlib import Path

from terminus.config import CONFIG
from terminus.context.indexers.factory import get_or_create_indexer
from terminus.observability.logging import get_logger


logger = get_logger(__name__)
console = Console()

class TaskOrchestrator:
    """
    Main execution loop: finds tasks whose dependencies are met,
    claims them atomically, and dispatches them to subtask agents.

    Serial by default (max_concurrent=1) for cost control and determinism.
     """

    def __init__(self, store: SQLiteTaskStore, max_concurrent:int = 1) -> None:
        self.store = store
        self.max_concurrent = max_concurrent

    async def run(self, project_id :str)-> None:


async def handle_plan_command(goal:str)->None:
    """
    Full /plan flow - entry point called by main.py.
    1. Check DB for an existing approved project -> resume + recover if found
    2. Otherwise: plan -> human approval loop -> persist -> execute.
    
        """
        db_path = CONFIG.get("tasks",{}).get("db_path",".terminus/tasks/tasks.db")
        store = SQLiteTaskStore(db_path)
        recover = RecoveryManager(store)

        project_id = store.get_latest_approved_project_id()

        if project_id:
            # resume + recover
            logger.info(f"Resuming project {project_id}")
            console.print(f"[yellow]Resuming project {project_id}[/yellow]")
            recovered = recover.recover(project_id)

            if recovered:
                logger.info(f"Resumed project {project_id}")
                console.print(f"[yellow]Resumed project {project_id}[/yellow]")

        else:
            console.print("[bold yellow]No active project. Planning new project...[/bold yellow]")
            extra_content = ""
            approved_plan = None
            while approved_plan is None:
                raw_plan = create_plan(goal, extra_content)
                approved_plan = present_plan_for_approval(raw_plan)
                if approved_plan is None:
                    extra_content = input("What should be changed or added in the plan? : \n").strip()
                    console.print("\n Re-planning with your feedback\n",style="cyan")

            project_id = store.create_project(goal, approved_plan)
            console.print(f"[bold green]Project created: {project_id}[/bold green]")

            orchestrator = TaskOrchestrator(store, max_concurrent=1)
            await orchestrator.run(project_id)



            console.print("[bold green]Re-indexing generated files[/bold green]")
            try:
                get_or_create_indexer()(str(Path.cwd()))
                console.print("[bold green]Index updated[/bold green]")
            except Exception as e:
                logger.error(f"Failed to re-index: {e}")
                console.print("[bold red]Failed to re-index[/bold red]")

            # recover.claim_and_dispatch(plan_task_id)

    

         
             
