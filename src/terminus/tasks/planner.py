from terminus.tasks.task_store import TaskType
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from terminus.config import CONFIG
from terminus.observability.logging import get_logger


logger = get_logger(__name__)



class PlannedTask(BaseModel):
    id: str
    title: str
    description: str
    task_type: TaskType
    depends_on: list[str]
    estimated_minutes: int
    output_files: list[str]
    acceptance_criteria: list[str]


class ExecutionPlan(BaseModel):
    project_name: str
    goal_summary: str
    tech_stack : list[str]
    total_estimated_hours: float
    tasks: list[PlannedTask]
    risks: list[str]
    assumptions: list[str]
    

SYSTEM_PROMPT="""

You are an AI Software Architect.

I will give you a **Goal**. Your job is to design a **complete, executable development plan** for a real-world software project, following these strict rules:

Rules:
- min 5 to max 8 tasks
- Task IDs must be stable snake_case strings (e.g. "task__001")
- depends_on must reference valid task ids in the same plan or be empty array.
- Ordering must form a valide DAG (no cycles): architecture -> schema -> config -> core -> tests -> integrations
- output files must list every file the task will write to disk
- acceptance_criteria must be concreate and verifiable (3-5 items per task)
- task_type must be one of : design, implement, test, review, integrate, configure
"""


def create_plan(goal: str, extra_content: str = "") -> ExecutionPlan:
    """Call the LLM planner and return a structured ExecutionPlan."""

    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"]["planner_model"]

    llm = init_chat_model(model, model_provider=provider)

    structured_llm = llm.with_structured_output(ExecutionPlan)

    user_message = f"Goal: {goal}"

    if extra_content:
        user_message += f"\nExtra Context: {extra_content}"

    plan: ExecutionPlan = structured_llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    return plan


def validate_plan(plan: ExecutionPlan) -> None:
    """Reject structurally invalid plans before they reach the executor."""
    if not plan.tasks:
        raise ValueError("Plan must contain at least one task")

    task_ids = [task.id for task in plan.tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Plan contains duplicate task IDs")

    id_set = set(task_ids)
    graph: dict[str, list[str]] = {task_id: [] for task_id in id_set}

    for task in plan.tasks:
        if task.id in task.depends_on:
            raise ValueError(f"Task {task.id} cannot depend on itself")

        for dep_id in task.depends_on:
            if dep_id not in id_set:
                raise ValueError(
                    f"Task {task.id} depends on unknown task {dep_id}"
                )
            graph[task.id].append(dep_id)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"Plan contains a dependency cycle involving {node}")
        if node in visited:
            return
        visiting.add(node)
        for dep in graph[node]:
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for task_id in id_set:
        visit(task_id)