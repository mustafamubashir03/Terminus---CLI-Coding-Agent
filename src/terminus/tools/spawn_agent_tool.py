"""The ``spawn_agent`` tool.

Deliberately narrow. A child agent is worth spawning when the work is separable
and a focused context would do better than more of the same context - a survey
before implementing, an independent review of work already done, a verification
pass on a claim. It is not worth spawning for a typo, and this tool's description
says so, because a parent that delegates everything spends model budget to learn
nothing.

Read-only roles may be asked for several at once. Write roles are one at a time
and must name a write scope, so two agents cannot edit the same files in
parallel. That asymmetry is the whole safety argument, so it is enforced in the
runtime rather than left to the model to respect.

The tool is async on purpose: a child runs several model calls, and a sync tool
would block the event loop for the whole delegation and stall the parent's own
stream.
"""

from __future__ import annotations

from langchain_core.tools import tool

from terminus.agents.roles import describe_roles
from terminus.agents.spawn import AgentSpawner, ChildSpec
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

ROLES_HELP = (
    "\n\nRoles:\n"
    "  researcher  - read-only investigation; cannot write or run commands\n"
    "  reviewer    - strict read-only review of existing code\n"
    "  debugger    - read + execute; diagnoses a failure and reports a cause\n"
    "  tester      - read + execute; runs tests, does not implement fixes\n"
    "  implementer - can write and run commands; must declare a write_scope"
)


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


@tool
async def spawn_agent(
    task: str,
    role: str = "researcher",
    skills: str = "",
    tools: str = "",
    context: str = "",
    write_scope: str = "",
) -> str:
    """Delegate one bounded, independent task to a specialist subagent.

    Use this when the work is separable and a focused agent would do better than
    more of the parent's own context: investigating an unfamiliar area, reviewing
    code already written, independently verifying a claim, or implementing a
    self-contained piece while you do something else. Do NOT use it for a typo, a
    one-line change, or anything that depends on the conversation you are already
    having - a subagent cannot see this conversation and will start cold.

    The subagent receives your task, the current project state, and any skills
    you name. It does not receive your transcript and cannot ask follow-ups.

    Args:
        task: What the subagent must determine or produce. Be specific and
            self-contained; it cannot ask questions.
        role: One of researcher, reviewer, debugger, tester, implementer.
        skills: Comma-separated skill names, e.g. "react-best-practices". Named
            skills are always used; left blank, matching is automatic.
        tools: Comma-separated extra tool names. It can never exceed what the
            role allows.
        context: Extra context to hand over. Keep it small and factual.
        write_scope: Required for the implementer role. The file or directory the
            subagent owns while it runs. A second agent on the same scope is
            refused rather than allowed to collide.

    Returns:
        The subagent's report: status, summary, findings, files changed, tests
        run, and the reason if it did not finish. A completed subagent is not
        proof the work is correct - verify before reporting it as done.
    """
    spec = ChildSpec(
        task=task,
        role=role,
        context=context,
        skills=tuple(_split(skills)),
        tools=tuple(_split(tools)),
        write_scope=write_scope,
    )
    spawner = AgentSpawner(max_children=1, max_parallel=1)

    # Claim from the *parent execution's* allowance, not a fresh per-call one.
    # Each tool call gets its own spawner, so without this the budget would be
    # per invocation and the model could multiply it by calling again.
    budget = _parent_budget()
    granted = budget.claim(1) if budget else 1
    if granted == 0:
        spent = f"of {budget.max_children}" if budget else ""
        return (
            f"Not spawned: this execution has already delegated its full "
            f"allowance ({spent} children). Finish the work yourself, or wait "
            f"for a new execution."
        )

    try:
        child = spawner.create(spec)
    except ValueError as exc:
        return f"{exc}{ROLES_HELP}"
    except RuntimeError as exc:
        return (
            f"Could not spawn: {exc}. Wait for a running subagent to finish, or "
            "do the work yourself."
        )

    conflicts = _skill_conflicts(spawner, child)
    if conflicts:
        return (
            "Refused to spawn: the selected skills conflict ("
            + "; ".join(conflicts)
            + "). Name a single skill to resolve it, or do the work directly so "
            "precedence can be applied by hand."
        )

    await spawner.run(child)
    return spawner.aggregate() or child.result.to_report()


def _parent_budget():
    """The delegation allowance of the execution this call is running in.

    Reads the ambient execution context rather than keeping a registry, so the
    budget is scoped to the parent execution exactly as long as that execution
    lasts. A tool call made outside any execution - a direct library call, or a
    test - gets None and is allowed, because there is no parent to exhaust.
    """
    try:
        from terminus.execution import current_execution

        context = current_execution()
    except Exception:
        return None
    return getattr(context, "budget", None) if context is not None else None


def _skill_conflicts(spawner: AgentSpawner, child) -> list[str]:
    """Report competing skills rather than merging contradictory instructions.

    Advisory, not fatal for the ordinary case: two skills touching the same
    concern is usually correct, so only a strong overlap is surfaced.
    """
    try:
        from terminus.skills.matcher import detect_conflicts, match_skills
        from terminus.skills.skill_tools import _get_registry

        matches = match_skills(
            _get_registry(), child.spec.task, explicit=child.spec.skills
        )
        return detect_conflicts(matches)
    except Exception as exc:
        logger.debug("Skill conflict check skipped: %s", exc)
        return []


__all__ = ["spawn_agent", "describe_roles"]
