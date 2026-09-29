"""A bounded read model of the project state Terminus already persists.

This module is deliberately **not** a source of truth and **not** memory. It
owns nothing and writes nothing. Every fact it reports is read live from
something that is already authoritative:

    plan, tasks, results, attempts  ->  TaskStore
    skills                          ->  SkillRegistry
    indexed code                    ->  the retriever / indexer
    conversation replay             ->  LangGraph's checkpointer (not read here)
    authority / workspace           ->  ExecutionContext (not read here)

Why it exists
-------------
Terminus has two capable halves that do not know about each other. ``/ask`` has
conversation memory and a code index but cannot see that a plan exists, what it
achieved, or why a task failed. ``/plan`` has durable task state that is
write-only in practice: results are persisted and then read by exactly one
consumer (a sibling task that declares ``depends_on``), so neither the agent nor
the user can see them. This module is the join.

Design rules
------------
* **Bounded, always.** Every field has an explicit cap. A read model that can
  emit an entire task database is a database dump wearing a prompt.
* **Cheap.** No model calls, no vector queries, no network. Reading the project
  must never cost more than the prompt section it produces.
* **Project-scoped.** The task database is already per-workspace, and a project
  whose recorded workspace does not match this one is refused rather than
  silently associated.
* **Degrades to nothing.** No database, no plan, no skills - all fine. Every
  consumer must work with an empty :class:`ProjectFacts`.

It is also not conversation memory: replaying a thread is LangGraph's job, and
this module never reads a checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terminus.workspace import project_root

# ---------------------------------------------------------------------------
# Explicit bounds
#
# Chosen so a ProjectContext section stays a small fraction of a normal system
# prompt. The task-result cap mirrors the existing persisted-result bound
# (task_store.MAX_PERSISTED_RESULT_CHARS = 8_000): we are re-sending to the model
# something that is already bounded on disk, and truncating it again is cheaper
# than letting a chatty task dominate the context window.
# ---------------------------------------------------------------------------

MAX_TASKS_SHOWN = 8
"""Recent tasks included in a section or tool response."""

MAX_RESULT_CHARS = 1200
"""Characters kept per task result / error. Head+tail, with a marker."""

MAX_ERROR_CHARS = 400
"""Characters kept per failure message."""

MAX_LIST_ITEMS = 5
"""Items kept for tech_stack, risks and assumptions."""

MAX_GOAL_CHARS = 500
"""Characters kept from the plan goal summary."""

MAX_SKILLS_SHOWN = 20
"""Skills named in a section or tool response."""

MAX_TOTAL_CHARS = 6_000
"""Hard ceiling on any rendered ProjectContext string.

Enforced by :func:`render`, so no combination of many tasks, long results and a
large skills catalogue can overflow the budget.
"""

_TRUNCATION_NOTE = "\n... [truncated]"


def _clip(text: str, limit: int) -> str:
    """Keep the head and tail of *text*, which is where the useful signal is.

    The marker is paid for out of the budget first, so both ends survive
    whenever the limit can accommodate them; only a limit too small to hold the
    marker degrades to a plain head.
    """
    if text is None:
        return ""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    budget = limit - len(_TRUNCATION_NOTE)
    if budget <= 1:
        return text[:limit].rstrip() + _TRUNCATION_NOTE
    head = budget * 2 // 3
    tail = budget - head
    return (
        text[:head].rstrip()
        + _TRUNCATION_NOTE
        + text[-tail:].lstrip()
    )


def _clip_list(items: Any, limit: int) -> list[str]:
    if not items:
        return []
    if isinstance(items, str):
        items = [items]
    return [_clip(str(i), 120) for i in list(items)[:limit] if str(i).strip()]


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskFact:
    """One task, as far as an agent or a user needs to see it."""

    id: str
    status: str
    attempts: int = 0
    title: str = ""
    result: str = ""
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


@dataclass(frozen=True)
class ProjectFacts:
    """Bounded, read-only view of one project.

    ``project_id`` is None when this workspace has no project, which is the
    normal state for a repository that has never been planned. Every consumer
    must handle that.
    """

    workspace: str
    project_id: str | None = None
    project_name: str = ""
    goal: str = ""
    tech_stack: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    recent: list[TaskFact] = field(default_factory=list)
    failures: list[TaskFact] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    index_status: str = ""

    @property
    def has_project(self) -> bool:
        return self.project_id is not None

    @property
    def completed(self) -> int:
        return int(self.counts.get("completed", 0))

    @property
    def failed(self) -> int:
        return int(self.counts.get("failed", 0))

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# ---------------------------------------------------------------------------
# Project resolution
# ---------------------------------------------------------------------------


def resolve_project_id(store: Any) -> str | None:
    """The project this workspace is working on, or None.

    A resumable project wins over merely-recent one, because that is the one the
    user could continue. Both candidates must have been planned for *this*
    workspace: a project recorded against a different directory is refused rather
    than silently associated, which is the same guard ``/plan continue`` applies
    before it will execute a task.
    """
    for candidate in (store.get_resumable_project(), store.get_latest_project()):
        if not candidate:
            continue
        try:
            if store.workspace_matches(candidate, project_root()):
                return candidate
        except Exception:
            continue
    return None


def plan_fields(plan_json: str | None) -> dict[str, Any]:
    """Pull the plan-level fields out of ``projects.plan_json``.

    Tolerates a missing, empty, truncated or non-JSON value: this is a read
    model and must never be the reason a run fails.
    """
    if not plan_json:
        return {}
    try:
        data = json.loads(plan_json)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_project_facts(
    store: Any,
    project_id: str | None = None,
    *,
    include_skills: bool = True,
) -> ProjectFacts:
    """Read bounded project facts.

    Never raises: a missing database, a missing project, or a malformed row all
    yield empty facts, because every caller treats this as optional context.
    """
    workspace = str(project_root())
    facts = ProjectFacts(workspace=workspace)

    try:
        if project_id is None:
            project_id = resolve_project_id(store)
    except Exception:
        project_id = None
    if not project_id:
        return facts
    facts = ProjectFacts(workspace=workspace, project_id=project_id)

    try:
        project = _get_project(store, project_id)
    except Exception:
        project = None
    if project:
        facts = _replace(facts, project_name=_clip(project.get("name") or "", 120))
        plan = plan_fields(project.get("plan_json"))
        facts = _replace(
            facts,
            goal=_clip(plan.get("goal_summary") or project.get("goal") or "", MAX_GOAL_CHARS),
            tech_stack=_clip_list(plan.get("tech_stack"), MAX_LIST_ITEMS),
            risks=_clip_list(plan.get("risks"), MAX_LIST_ITEMS),
            assumptions=_clip_list(plan.get("assumptions"), MAX_LIST_ITEMS),
        )

    tasks: list[dict] = []
    try:
        tasks = list(store.get_all_tasks(project_id))
    except Exception:
        tasks = []
    if tasks:
        facts = _replace(facts, counts=_count_statuses(tasks))
        recent = _recent_facts(tasks, MAX_TASKS_SHOWN)
        facts = _replace(
            facts,
            recent=recent,
            failures=[t for t in recent if t.status == "failed" and (t.error or t.result)][:3],
        )

    if include_skills:
        facts = _replace(facts, skills=_available_skills())
    return facts


def _get_project(store: Any, project_id: str) -> dict | None:
    getter = getattr(store, "get_project", None)
    if callable(getter):
        return getter(project_id)
    return None


def _replace(facts: ProjectFacts, **changes: Any) -> ProjectFacts:
    from dataclasses import replace

    return replace(facts, **changes)


def _count_statuses(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _recent_facts(tasks: list[dict], limit: int) -> list[TaskFact]:
    """The most interesting tasks: successes and failures first, then the rest.

    Ordering by execution order rather than by rowid keeps this stable between
    calls, which matters because it is rendered into a prompt.
    """

    def sort_key(task: dict) -> tuple[int, int, str]:
        try:
            order = int(task.get("execution_order") or 0)
        except (TypeError, ValueError):
            order = 0
        # Completed and failed first: they carry results worth reading.
        rank = 0 if str(task.get("status")) in ("completed", "failed") else 1
        return (rank, order, str(task.get("id")))

    ordered = sorted(tasks, key=sort_key)
    # Most recent work last, so truncate from the front.
    chosen = ordered[-limit:] if len(ordered) > limit else ordered
    return [_to_fact(task) for task in chosen]


def _to_fact(task: dict) -> TaskFact:
    try:
        attempts = int(task.get("total_attempts") or 0)
    except (TypeError, ValueError):
        attempts = 0
    return TaskFact(
        id=str(task.get("id") or "?"),
        status=str(task.get("status") or "unknown"),
        attempts=attempts,
        title=_clip(str(task.get("title") or ""), 120),
        result=_clip(str(task.get("result") or ""), MAX_RESULT_CHARS),
        error=_clip(str(task.get("error") or ""), MAX_ERROR_CHARS),
    )


def _available_skills() -> list[str]:
    """Skill names for the current project, bounded.

    Imported lazily and defensively: skills are optional, and a missing skills
    directory must not stop a project section from rendering.
    """
    try:
        from terminus.skills.skill_tools import _get_registry

        return sorted(_get_registry().skills)[:MAX_SKILLS_SHOWN]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(facts: ProjectFacts, *, include_results: bool = True) -> str:
    """Render facts as the bounded markdown block the agents receive.

    Returns "" when there is no project, so a repository that has never been
    planned contributes nothing at all to the prompt.
    """
    if not facts.has_project:
        return ""
    lines: list[str] = ["## Project context", ""]
    lines.append(f"- Workspace: {facts.workspace}")
    if facts.project_name:
        lines.append(f"- Project: {facts.project_name} ({facts.project_id})")
    if facts.goal:
        lines.append(f"- Goal: {facts.goal}")
    if facts.tech_stack:
        lines.append(f"- Tech stack: {', '.join(facts.tech_stack)}")
    if facts.risks:
        lines.append(f"- Known risks: {'; '.join(facts.risks)}")
    if facts.assumptions:
        lines.append(f"- Assumptions: {'; '.join(facts.assumptions)}")
    if facts.counts:
        summary = ", ".join(f"{n} {k}" for k, n in sorted(facts.counts.items()))
        lines.append(f"- Task progress: {summary}")
    if facts.recent:
        lines.append("")
        lines.append("Recent tasks:")
        lines.extend(_render_task_lines(facts.recent, include_results))
    if facts.skills:
        lines.append(f"- Skills available: {', '.join(facts.skills)}")
    if facts.index_status:
        lines.append(f"- Code index: {facts.index_status}")

    text = "\n".join(lines).rstrip()
    if len(text) > MAX_TOTAL_CHARS:
        # Keep the head (identity, goal, progress) and drop the tail of detail
        # rather than returning a truncated sentence.
        text = text[: MAX_TOTAL_CHARS - len(_TRUNCATION_NOTE)].rstrip() + _TRUNCATION_NOTE
    return text


def _render_task_lines(tasks: list[TaskFact], include_results: bool) -> list[str]:
    lines: list[str] = []
    for task in tasks:
        attempts = f", {task.attempts} attempt(s)" if task.attempts else ""
        lines.append(f"  - {task.id} [{task.status}{attempts}]")
        if not include_results:
            continue
        if task.result:
            lines.append(f"    result: {_indent(task.result)}")
        elif task.error:
            lines.append(f"    error: {_indent(task.error)}")
    return lines


def _indent(text: str) -> str:
    return text.replace("\n", "\n    ")


def open_readable_store(db_path: str) -> Any | None:
    """Open an existing task database, or None if there is not one yet.

    ``TaskStore.__init__`` creates the directory and the database file. That is
    right for a store you are going to write to, and wrong here: building the
    /ask prompt must not leave behind a ``.terminus/tasks/tasks.db`` in a
    repository that has never run /plan. Reading something that does not exist
    yields no project.
    """
    if not Path(db_path).is_file():
        return None
    from terminus.tasks.task_store import TaskStore

    return TaskStore(db_path)


def task_db_path() -> str:
    from terminus.config import CONFIG

    return CONFIG.get("tasks", {}).get("db_path", ".terminus/tasks/tasks.db")


def project_prompt_section(store: Any = None, project_id: str | None = None) -> str:
    """The prompt block, ready to append to a system prompt.

    Opens its own store when one is not supplied, so callers that have no store
    (the /ask agent factory) need not know that TaskStore exists.
    """
    try:
        if store is None:
            store = open_readable_store(task_db_path())
            if store is None:
                return ""
        return render(collect_project_facts(store, project_id))
    except Exception:
        return ""
