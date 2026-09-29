from terminus.tasks.errors import classify_failure, format_failure
from terminus.tasks.task_store import TaskType
from pydantic import BaseModel
from terminus.llm.factory import get_chat_model
from terminus.llm.text import message_text
from terminus.config import CONFIG
from terminus.observability.logging import get_logger
from terminus.observability.usage_tracker import UsageCallbackHandler, record
from terminus.context.environment import build_startup_context
from terminus.project_context import project_prompt_section
from terminus.workspace import project_root


logger = get_logger(__name__)

MAX_PLANNER_CONTEXT_CHARS = 4_000
"""Ceiling on the context block appended to the planner's goal.

The planner is a single-shot call with no tools, so this snapshot is the only
thing it can know about the machine it is planning for. It has to be bounded: the
goal is the message that matters, and context that crowds it out makes plans
worse, not better.
"""


def _planner_context() -> str:
    """Bounded snapshot of the workspace and any project already here.

    Two cheap, read-only sources, no model calls and no network:

    * the environment/repo snapshot, so plans are grounded in the real tree
      instead of being invented from the goal text alone;
    * the project read model, so re-planning an existing project does not lose
      what it already built or what already failed.

    Every part is optional. A missing snapshot degrades the plan's grounding; it
    must never stop a plan from being produced.
    """
    parts: list[str] = []
    try:
        snapshot = build_startup_context(project_root()).strip()
        if snapshot:
            parts.append(snapshot)
    except Exception as exc:
        logger.debug("Planner startup context unavailable: %s", exc)
    try:
        existing = project_prompt_section().strip()
        if existing:
            parts.append(existing)
    except Exception as exc:
        logger.debug("Planner project context unavailable: %s", exc)

    context = "\n\n".join(parts)
    if len(context) > MAX_PLANNER_CONTEXT_CHARS:
        context = context[: MAX_PLANNER_CONTEXT_CHARS - 20].rstrip() + "\n... [truncated]"
    return context



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

Sizing the work. Terminus can delegate a self-contained task to a bounded
subagent (research, review, testing, or implementation inside an explicit write
scope). Delegation costs a separate agent run, so it pays only when the work is
genuinely separable:

  delegate  - an investigation before implementation, an independent review of
              risky work, a verification pass, or a self-contained component
              built in parallel with other work
  do not    - a typo, a one-line fix, anything small, or work that needs the whole
              picture to be understood first

Do not add a task merely to create something to delegate to. A plan made of 12
small delegated tasks is worse than 4 well-scoped ones executed directly.

Response format: Return a single STRICT JSON object matching this schema exactly. No markdown fences, no prose, no explanation outside the JSON:

{
  "project_name": "string",
  "goal_summary": "string",
  "tech_stack": ["string"],
  "total_estimated_hours": 0.0,
  "tasks": [
    {
      "id": "task__001",
      "title": "string",
      "description": "string",
      "task_type": "design",
      "depends_on": [],
      "estimated_minutes": 30,
      "output_files": ["path/to/file.ext"],
      "acceptance_criteria": ["string"]
    }
  ],
  "risks": ["string"],
  "assumptions": ["string"]
}
"""


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction.

    Free-tier models frequently wrap JSON in markdown fences or stray prose
    even when asked for strict JSON.  Try strict parse first, then strip
    fences, then extract the first `{...}` balanced block.
    """
    import json

    t = text.strip()
    try:
        parsed = json.loads(t)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strip ```json ... ```  / ``` ... ``` fences.
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].strip().lstrip("`").strip() in ("json", ""):
            t = "\n".join(lines[1:])
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        try:
            parsed = json.loads(t)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: extract a balanced { ... } block (first { to last }).
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON object found in planner output: {text[:200]}")


def create_plan(goal: str, extra_content: str = "") -> ExecutionPlan:
    """Call the LLM planner once and return a structured ExecutionPlan.

    Uses plain invoke plus JSON-extraction instead of with_structured_output,
    because free-tier providers frequently return markdown-wrapped or
    prose-wrapped JSON that the strict structured-output parser rejects.

    There is no retry loop here on purpose: the provider route
    (llm.FallbackChatModel) already retries transient failures, and
    handle_plan_command re-plans up to three times when the user asks for
    changes. A third layer would multiply the wait on an already slow path.
    """
    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"]["planner_model"]
    llm = get_chat_model(model, model_provider=provider)

    user_message = f"Goal: {goal}"
    if extra_content:
        user_message += f"\nExtra Context: {extra_content}"
    context = _planner_context()
    if context:
        user_message += (
            f"\n\n## Context about this workspace and project\n"
            f"(snapshot; the filesystem is authoritative)\n\n{context}"
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    handler = UsageCallbackHandler(kind="planner")
    try:
        raw = llm.invoke(messages, config={"callbacks": [handler]})
        plan = ExecutionPlan.model_validate(_extract_json(message_text(raw)))
    except Exception as exc:
        record(handler.records, "planner")
        failure = classify_failure(exc, provider=provider, model=model)
        logger.warning("Planner call failed: %s", format_failure(failure, provider, model))
        raise

    record(handler.records, "planner")
    if not plan.tasks:
        raise ValueError("Planner returned an empty plan")
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
