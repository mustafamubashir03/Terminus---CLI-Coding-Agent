"""Read-only project status, exposed to /ask as a tool.

The system prompt already carries a bounded project summary, but a prompt
snapshot is frozen at the moment the agent was built and cannot answer "what is
the current state of task 7". This tool lets the agent ask again on demand.

It is read-only and project-scoped: it inspects a project for this workspace and
refuses to read one recorded against a different directory. It never mutates
anything, and it reports absence plainly rather than inventing a project.
"""

from __future__ import annotations

from langchain_core.tools import tool

from terminus.project_context import (
    collect_project_facts,
    open_readable_store,
    render,
    resolve_project_id,
    task_db_path,
)


@tool
def project_status() -> str:
    """Get the current state of the project being worked on in this workspace.

    Use this when you need to know what has already been planned and achieved
    here, rather than what the code looks like right now: the project's goal,
    tech stack, task progress, recent task results and failures, and the skills
    that are available. The answer is refreshed on every call.

    Returns a short summary, or a plain statement that this workspace has no
    project yet.
    """
    try:
        store = open_readable_store(task_db_path())
    except Exception as exc:
        return f"Could not open the project database: {exc}"
    if store is None:
        return "No project has been created in this workspace yet. Run /plan to create one."

    try:
        project_id = resolve_project_id(store)
        if not project_id:
            return (
                "No project has been created in this workspace yet. "
                "Run /plan to create one."
            )
        facts = collect_project_facts(store, project_id)
    except Exception as exc:
        return f"Could not read project status: {exc}"

    body = render(facts)
    if not body:
        return f"Project {project_id} has no readable state yet."
    return body
