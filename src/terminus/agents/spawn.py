"""Bounded child agent execution.

A child is a fresh, short-lived agent with exactly the task it was given. It does
not inherit the parent's conversation, the parent's plan, or the parent's tools.
That isolation is the point: it keeps delegation cheap, keeps two agents from
influencing each other mid-task, and makes a child's behaviour reproducible from
its own inputs.

Everything here is bounded. A child gets a task, a budget, a tool set, a skill
set, a permission policy and a deadline, and it stops at the first of those
boundaries it reaches.

Concurrency is deliberately small and write-safe. Several read-only children may
run together because they cannot conflict. A child that can write must claim a
write scope first, and two writers on the same scope is refused rather than
serialised silently - the caller decides what to do about it.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterable, Sequence

from terminus.agents.roles import AgentRole, get_role, role_names
from terminus.execution import MAX_CHILDREN_PER_PARENT
from terminus.observability.logging import get_logger
from terminus.permissions import PermissionLevel, PermissionPolicy
from terminus.project_context import (
    MAX_TOTAL_CHARS as MAX_CONTEXT_CHARS,
    project_prompt_section,
)
from terminus.tasks.errors import FailureInfo, classify_failure

logger = get_logger(__name__)

# --- bounds ----------------------------------------------------------------

MAX_CHILD_AGENTS = MAX_CHILDREN_PER_PARENT
"""How many children one parent turn may spawn, in total.

An alias of :data:`terminus.execution.MAX_CHILDREN_PER_PARENT`, which is the
single owner of that number because :class:`~terminus.execution.ExecutionBudget`
is what actually enforces it. This name is kept because it is what the
``terminus.agents`` package exports and what callers reach for; it used to be a
second, independently declared ``8``, which meant the spawner's ceiling and the
budget's ceiling could drift apart without anything failing.
"""

MAX_PARALLEL_AGENTS = 3
"""How many children may run at the same time.

Above this, children queue. Read-only work parallelises well; write work is
serialised anyway by scope ownership.
"""

MAX_CHILD_DEPTH = 1
"""How deep delegation may nest. One level.

A child that could spawn its own children would multiply model budgets
exponentially and make the total cost of a single user request unknowable, for
no benefit that a better-scoped first child would not give. This is a real
ceiling, not a convention: it is checked, and additionally no role is granted
the spawn tool, so a child cannot ask to delegate at all.
"""

DEFAULT_CHILD_TIMEOUT_SECONDS = 300
"""A child that has not finished by now is cancelled, not waited on forever."""


class AgentStatus:
    """Lifecycle states a child can be in."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"

    TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED, TIMED_OUT, BLOCKED})


@dataclass
class AgentResult:
    """What a child produced, and what it cost.

    Mirrors the existing result conventions rather than inventing a new
    taxonomy: ``error`` is a formatted :class:`FailureInfo` in the same shape the
    rest of Terminus logs, and ``summary``/``findings`` are already bounded by
    the caller.
    """

    agent_id: str
    role: str
    status: str = AgentStatus.QUEUED
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)
    error: str = ""
    failure: FailureInfo | None = None
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    duration_seconds: float = 0.0
    write_scope: str = ""
    verification: str = "not_required"
    """What verification, if any, the child itself established.

    Deliberately not a boolean. A child that ran the tests and saw them pass has
    done something a child that only read code has not, and the parent needs to
    tell those apart. ``not_required`` covers research and review, where there is
    nothing to execute. A child's success is never ``verified`` on the parent's
    behalf: the parent owns that decision.
    """

    @property
    def ok(self) -> bool:
        return self.status == AgentStatus.COMPLETED

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "status": self.status,
            "summary": self.summary,
            "findings": list(self.findings),
            "files_changed": list(self.files_changed),
            "tests_run": list(self.tests_run),
            "error": self.error,
            "skills": list(self.skills),
            "tools": list(self.tools),
            "model": self.model,
            "provider": self.provider,
            "duration_seconds": round(self.duration_seconds, 2),
            "write_scope": self.write_scope,
            "verification": self.verification,
        }

    def to_report(self) -> str:
        """The text a parent aggregates. Bounded, and states failure plainly."""
        if not self.ok:
            return (
                f"[{self.role} {self.agent_id}] {self.status.upper()}: "
                f"{self.error or 'no detail reported'}"
            )
        lines = [f"[{self.role} {self.agent_id}] {self.summary or 'done'}"]
        for finding in self.findings:
            lines.append(f"  - {finding}")
        if self.files_changed:
            lines.append(f"  files changed: {', '.join(self.files_changed)}")
        if self.tests_run:
            lines.append(f"  tests run: {', '.join(self.tests_run)}")
        return "\n".join(lines)


# --- write ownership --------------------------------------------------------


class WriteScopeError(RuntimeError):
    """Raised when a write scope is already held by a live child."""


class _WriteScopeRegistry:
    """Process-local claim table for write scopes.

    Scheduler-level rather than file-level locking: a child declares the scope it
    intends to work in (a path, a task, a module), and two children may not hold
    the same scope at once. That is enough to stop two agents corrupting one
    file, and it is honest about what it does not do - it does not coordinate
    across machines.
    """

    def __init__(self) -> None:
        self._holders: dict[str, str] = {}

    def claim(self, scope: str, agent_id: str) -> None:
        key = (scope or "").strip()
        if not key:
            return
        holder = self._holders.get(key)
        if holder is not None and holder != agent_id:
            raise WriteScopeError(
                f"write scope {key!r} is already held by {holder}"
            )
        self._holders[key] = agent_id

    def release(self, scope: str) -> None:
        if scope:
            self._holders.pop(scope.strip(), None)

    def release_agent(self, agent_id: str) -> None:
        for scope, holder in list(self._holders.items()):
            if holder == agent_id:
                del self._holders[scope]

    def holder(self, scope: str) -> str | None:
        return self._holders.get((scope or "").strip())

    def snapshot(self) -> dict[str, str]:
        return dict(self._holders)


_write_scopes = _WriteScopeRegistry()


def write_scope_holder(scope: str) -> str | None:
    return _write_scopes.holder(scope)


def recent_delegations(limit: int = 20) -> str:
    """A compact view of delegated executions, for the CLI.

    Reads the telemetry events rather than keeping a second live registry, so
    what the user sees is exactly what was recorded. A tree of
    ``parent -> child -> status`` is enough for now; this is an inspection
    surface, not a monitoring UI.
    """
    try:
        from terminus.observability.usage_tracker import get_child_events
        events = get_child_events()
    except Exception:
        return ""
    if not events:
        return ""

    by_parent: dict[str, list[dict]] = {}
    for event in events:
        by_parent.setdefault(event.get("parent_agent_id") or "(root)", []).append(event)

    lines: list[str] = []
    for parent, children in list(by_parent.items())[-limit:]:
        lines.append(f"{parent}")
        for event in children[-limit:]:
            duration = event.get("duration_seconds", 0.0)
            detail = f"{duration:.1f}s"
            if event.get("skills"):
                detail += f", skills={','.join(event['skills'])}"
            failure = event.get("failure_category")
            suffix = f" ({failure})" if failure else ""
            lines.append(
                f"  +- {event.get('role','?')}: {event.get('status','?')}{suffix} "
                f"[{event.get('child_agent_id','?')}, {detail}]"
            )
    return "\n".join(lines)


def _reset_write_scopes() -> None:
    """Test seam. Live callers release their own scope."""
    _write_scopes._holders.clear()


# --- permission policy ------------------------------------------------------


def child_policy(role: AgentRole) -> PermissionPolicy:
    """The policy a child runs under.

    Built from the role, not from the parent, and deliberately narrow: read-only
    roles get no approver and no write level, so a read-only child cannot write
    even if something in its prompt asks it to. A role marked destructive is not
    grantable here - escalation is the parent's job, with a human in the loop.
    """
    if role.destructive:
        raise ValueError(
            f"role {role.name} is not grantable to a child agent: destructive "
            "operations require the parent execution"
        )
    if role.write:
        return PermissionPolicy(
            auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
            approver=None,
            deny_levels=(PermissionLevel.DESTRUCTIVE,),
        )
    return PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,),
        approver=None,
        deny_levels=(PermissionLevel.WRITE, PermissionLevel.DESTRUCTIVE),
    )


# --- context ----------------------------------------------------------------


def build_child_context(
    task: str,
    *,
    context: str = "",
    project_facts: str = "",
    skill_block: str = "",
    constraints: str = "",
) -> str:
    """Assemble a child's prompt context.

    Everything is capped, and the whole block is capped again at the end. The
    project context comes in already bounded from
    :mod:`terminus.project_context`; the cap here is the backstop that keeps a
    caller who passes a large extra blob from defeating that.
    """
    parts = [f"# Task\n\n{task.strip()}"]
    if context.strip():
        parts.append(f"# Provided context\n\n{context.strip()}")
    if project_facts.strip():
        parts.append(
            f"# Project state\n\n{project_facts.strip()}\n"
            "(snapshot; the filesystem is authoritative)"
        )
    if skill_block.strip():
        parts.append(f"# Skills in effect\n\n{skill_block.strip()}")
    if constraints.strip():
        parts.append(f"# Constraints\n\n{constraints.strip()}")
    text = "\n\n".join(parts)
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[: MAX_CONTEXT_CHARS - 32].rstrip() + "\n\n[context truncated]"
    return text


# --- the child --------------------------------------------------------------


@dataclass
class ChildSpec:
    """A request to run a child agent. Cheap to construct, inert until spawned."""

    task: str
    role: str = "researcher"
    context: str = ""
    skills: Sequence[str] = ()
    tools: Sequence[str] = ()
    model: str | None = None
    provider: str | None = None
    timeout: int | None = None
    write_scope: str = ""
    constraints: str = ""
    agent_id: str = ""
    parent_id: str = ""
    task_id: str = ""
    depth: int = 0
    """How many delegations deep this request sits. Bounded by MAX_CHILD_DEPTH."""


class ChildAgent:
    """One spawned child: its identity, its bounds, and its result.

    Construction is where every bound is resolved - role, tools, skills, policy,
    deadline, write scope - so that running it cannot later acquire a capability
    it did not have when it was created.
    """

    def __init__(
        self,
        spec: ChildSpec,
        *,
        available_tools: Iterable[Any] | None = None,
        project_facts: str = "",
        skill_block: str = "",
        runner: Any = None,
    ) -> None:
        self.spec = spec
        self.agent_id = spec.agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.parent_id = spec.parent_id
        self.task_id = spec.task_id
        self.status = AgentStatus.QUEUED
        self.result = AgentResult(agent_id=self.agent_id, role=spec.role)

        role = get_role(spec.role)
        if role is None:
            raise ValueError(
                f"unknown agent role {spec.role!r}; known roles: "
                f"{', '.join(role_names())}"
            )
        self.role = role
        self.policy = child_policy(role)
        self.timeout = spec.timeout or role.timeout_seconds or DEFAULT_CHILD_TIMEOUT_SECONDS

        # Requested tools are intersected with the role's, then with what the
        # parent actually has. Three successive narrowings: a child can never
        # widen its own reach by naming a tool.
        wanted = set(spec.tools) | set(role.tools)
        allowed = role.tool_names() & wanted
        if available_tools is not None:
            present = {getattr(t, "name", str(t)) for t in available_tools}
            allowed &= present
        self.tools = sorted(allowed)
        # Two different refusals, kept apart because they mean different things
        # to whoever is debugging a child that could not do what it was asked.
        self.out_of_role_tools = sorted(wanted - role.tool_names())
        self.rejected_tools = sorted((wanted & role.tool_names()) - set(self.tools))

        self.skills = list(spec.skills)
        self.skill_block = skill_block
        self.project_facts = project_facts
        self._runner = runner
        self._owns_scope = False

    # -- lifecycle ---------------------------------------------------------

    def claim_scope(self) -> None:
        if not self.role.write:
            return
        scope = (self.spec.write_scope or self.task_id or self.agent_id).strip()
        _write_scopes.claim(scope, self.agent_id)
        self._owns_scope = True
        self.result.write_scope = scope

    def release_scope(self) -> None:
        if self._owns_scope:
            _write_scopes.release(self.result.write_scope)
            self._owns_scope = False

    def prompt(self) -> str:
        head = (
            f"You are a bounded subagent. Complete the task below and report "
            f"what you found. You cannot see the main conversation, and you "
            f"cannot ask questions mid-task.\n"
            f"Role: {self.role.name} - {self.role.description}\n"
            f"Allowed tools: {', '.join(self.tools) or 'none'}\n"
            f"You may write: {'yes, only inside your write scope' if self.role.write else 'no'}\n"
        )
        return head + build_child_context(
            self.spec.task,
            context=self.spec.context,
            project_facts=self.project_facts,
            skill_block=self.skill_block,
            constraints=self.spec.constraints,
        )

    async def run(self) -> AgentResult:
        """Execute the child under its bounds and produce a result.

        The runner is injectable so orchestration can be tested without a model.
        The default runner is a real agent built through
        ``agent.factory.build_agent(tools_override=...)``, which is the hook that
        already existed for this.
        """
        if not (self.spec.task or "").strip():
            self.status = AgentStatus.BLOCKED
            self.result.status = AgentStatus.BLOCKED
            self.result.error = "refused: an empty task would produce an untraceable agent"
            return self.result

        try:
            self.claim_scope()
        except WriteScopeError as exc:
            self.status = AgentStatus.BLOCKED
            self.result.status = AgentStatus.BLOCKED
            self.result.error = f"blocked: {exc}"
            self.result.failure = classify_failure(exc)
            logger.warning("Child %s blocked: %s", self.agent_id, exc)
            return self.result

        self.status = AgentStatus.RUNNING
        self.result.status = AgentStatus.RUNNING
        started = time.time()
        runner = self._runner
        try:
            if runner is None:
                runner = _default_runner
            outcome = await asyncio.wait_for(
                runner(self), timeout=self.timeout
            )
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except asyncio.TimeoutError:
            self.status = AgentStatus.TIMED_OUT
            self.result.status = AgentStatus.TIMED_OUT
            self.result.error = f"timed out after {self.timeout}s"
            self.result.failure = classify_failure(
                TimeoutError(f"child agent exceeded {self.timeout}s")
            )
            logger.warning("Child %s timed out after %ss", self.agent_id, self.timeout)
            return self.result
        except asyncio.CancelledError:
            self.status = AgentStatus.CANCELLED
            self.result.status = AgentStatus.CANCELLED
            self.result.error = "cancelled"
            raise
        except Exception as exc:
            failure = classify_failure(exc)
            self.status = AgentStatus.FAILED
            self.result.status = AgentStatus.FAILED
            self.result.error = f"{type(exc).__name__}: {exc}"
            self.result.failure = failure
            logger.warning("Child %s failed: %s", self.agent_id, exc)
            return self.result
        finally:
            self.result.duration_seconds = time.time() - started
            self.release_scope()

        if isinstance(outcome, AgentResult):
            self.result = outcome
            outcome.agent_id = self.agent_id
            outcome.role = self.role.name
        else:
            self.result.summary = str(outcome or "")
        self.result.skills = list(self.skills)
        self.result.tools = list(self.tools)
        if self.result.status in (AgentStatus.QUEUED, AgentStatus.RUNNING):
            self.status = AgentStatus.COMPLETED
            self.result.status = AgentStatus.COMPLETED
        else:
            self.status = self.result.status
        return self.result

    def cancel(self) -> None:
        if self.status in AgentStatus.TERMINAL:
            return
        self.status = AgentStatus.CANCELLED
        self.result.status = AgentStatus.CANCELLED
        self.result.error = self.result.error or "cancelled before completion"
        self.release_scope()


def _record_child_event(child: ChildAgent, result: AgentResult) -> None:
    """Record one child lifecycle event through the existing telemetry path.

    Uses the same ``usage_tracker`` summary the rest of Terminus writes to
    rather than a second observability system, and records the fields a
    delegation is actually worth tracking: lineage, role, budget, and outcome.
    No prompt text and no child output is recorded - a child's report can
    contain source code, and telemetry is not the place for it.
    """
    try:
        from terminus.observability.usage_tracker import record_child_event

        record_child_event(
            parent_agent_id=child.parent_id,
            child_agent_id=child.agent_id,
            role=child.role.name,
            status=result.status,
            skills=result.skills or list(child.skills),
            tools=result.tools or list(child.tools),
            duration_seconds=result.duration_seconds,
            failure_category=result.failure.category if result.failure else None,
        )
    except Exception as exc:
        # Telemetry must never be able to fail a delegation.
        logger.debug("Child telemetry not recorded: %s", exc)


async def _default_runner(child: ChildAgent) -> Any:
    """Run a child through the real agent factory.

    Reuses ``build_agent(tools_override=...)`` - the hook that existed for exactly
    this - so a child is the same agent construction as /ask, with a different
    tool set, its own conversation thread, and a smaller model-call budget. No
    new agent construction path is introduced.

    Two things are deliberately different from /ask:

    * **Its own thread.** The child gets a thread id derived from its own id, so
      it starts with an empty conversation. Passing the parent's thread would
      replay the whole parent transcript into the child and quietly destroy the
      isolation the primitive exists to provide.
    * **The parent's execution scope.** The child's role policy is installed for
      the duration, so a read-only child cannot write even if its prompt asks it
      to, and it cannot affect the parent's permissions on return.
    """
    from terminus.agent.factory import build_agent
    from terminus.execution import CHILD, ExecutionContext, execution_scope
    from terminus.observability.usage_tracker import UsageCallbackHandler, record
    from terminus.workspace import project_root

    tools = _resolve_tools(child)
    agent = await build_agent(
        tools_override=tools,
        model=child.spec.model,
        provider=child.spec.provider,
        max_model_calls=child.role.max_model_calls,
    )
    handler = UsageCallbackHandler(kind=f"child_agent:{child.role.name}")
    context = ExecutionContext(
        workspace=project_root(),
        kind=CHILD,
        policy=child.policy,
        parent_agent_id=child.parent_id,
        task_id=child.task_id,
    )
    try:
        with execution_scope(context):
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": child.prompt()}]},
                config={
                    "configurable": {"thread_id": f"child-{child.agent_id}"},
                    "callbacks": [handler],
                },
            )
    finally:
        record(handler.records, kind=f"child_agent:{child.role.name}")

    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", "") or "")


def _resolve_tools(child: ChildAgent) -> list[Any]:
    """Map the child's allowed tool *names* to the real tool objects.

    Resolution happens here rather than being passed in so a child can never be
    constructed holding a tool object the parent does not have: the names come
    from the role, and the objects come from the parent's own registry.
    """
    from terminus.agent.factory import tools_by_name

    catalogue = tools_by_name()
    resolved = [catalogue[name] for name in child.tools if name in catalogue]
    if child.rejected_tools:
        logger.info(
            "Child %s asked for tools outside its role or the parent toolset, "
            "dropped: %s",
            child.agent_id,
            ", ".join(child.rejected_tools),
        )
    return resolved


# --- parent side ------------------------------------------------------------


class AgentSpawner:
    """Creates and runs children on behalf of one parent.

    Holds the per-parent accounting: how many children may exist, how many may
    run at once, and which write scopes are in flight. A fresh spawner per parent
    turn means the limits are per-delegation, not global, so one turn cannot
    exhaust the budget for the next.
    """

    def __init__(
        self,
        *,
        parent_id: str = "",
        task_id: str = "",
        max_children: int = MAX_CHILD_AGENTS,
        max_parallel: int = MAX_PARALLEL_AGENTS,
        available_tools: Iterable[Any] | None = None,
        runner: Any = None,
        project_facts: str | None = None,
    ) -> None:
        self.parent_id = parent_id or f"parent-{uuid.uuid4().hex[:8]}"
        self.task_id = task_id
        self.max_children = max(0, min(max_children, MAX_CHILD_AGENTS))
        self.max_parallel = max(1, min(max_parallel, MAX_PARALLEL_AGENTS))
        self.available_tools = available_tools
        self.runner = runner
        self.project_facts = project_facts
        self.children: list[ChildAgent] = []
        self.results: list[AgentResult] = []
        self._semaphore = asyncio.Semaphore(self.max_parallel)

    # -- construction ------------------------------------------------------

    def resolve_skills(
        self, task: str, requested: Sequence[str] = (), **signals: Any
    ) -> tuple[list[str], str]:
        """Pick skills for a child and render their bounded instructions.

        Explicitly named skills are honoured regardless of score; the rest are
        matched deterministically against the task.
        """
        from terminus.skills.matcher import match_skills, render_selection
        from terminus.skills.skill_tools import _get_registry

        try:
            registry = _get_registry()
        except Exception as exc:
            logger.debug("Skill selection unavailable for child: %s", exc)
            return [], ""
        matches = match_skills(registry, task, explicit=requested, **signals)
        block = render_selection(matches, registry)
        return [m.name for m in matches if m.selected], block

    def create(self, spec: ChildSpec) -> ChildAgent:
        """Build a child, or refuse if this parent is already at its ceiling."""
        if len(self.children) >= self.max_children:
            raise RuntimeError(
                f"child agent limit reached: this delegation may spawn at most "
                f"{self.max_children} agents"
            )
        depth = spec.depth + 1
        if depth > MAX_CHILD_DEPTH:
            raise RuntimeError(
                f"delegation depth limit reached: children may nest at most "
                f"{MAX_CHILD_DEPTH} level deep"
            )
        spec.depth = depth
        spec.parent_id = spec.parent_id or self.parent_id
        spec.task_id = spec.task_id or self.task_id
        skills, block = self.resolve_skills(spec.task, spec.skills)
        facts = self.project_facts
        if facts is None:
            # The bounded project read model, not a transcript. A child learns
            # what has been planned and achieved, never what the parent said.
            facts = project_prompt_section()
        child = ChildAgent(
            spec,
            available_tools=self.available_tools,
            runner=self.runner,
            skill_block=block,
            project_facts=facts or "",
        )
        child.skills = skills
        self.children.append(child)
        logger.info(
            "Spawned child %s role=%s skills=%s tools=%s",
            child.agent_id, child.role.name, skills, child.tools,
        )
        return child

    # -- execution ---------------------------------------------------------

    async def run(self, child: ChildAgent) -> AgentResult:
        async with self._semaphore:
            result = await child.run()
        self.results.append(result)
        _record_child_event(child, result)
        return result

    async def run_all(self, specs: Sequence[ChildSpec]) -> list[AgentResult]:
        """Run several children, bounded by the parallelism limit.

        Results come back in the order the specs were given, not the order they
        finished, so aggregation is deterministic. A child that fails does not
        stop its siblings; the failure is represented in its own result.
        """
        children: list[ChildAgent] = []
        refused: list[AgentResult] = []
        for spec in specs:
            try:
                children.append(self.create(spec))
            except (RuntimeError, ValueError) as exc:
                # At the ceiling, or an unknown role: record the refusal rather
                # than dropping it, so the caller learns one of its children
                # never ran.
                blocked = AgentResult(
                    agent_id="-", role=spec.role, status=AgentStatus.BLOCKED,
                    error=str(exc),
                )
                self.results.append(blocked)
                refused.append(blocked)
                logger.warning("Refused to spawn child: %s", exc)
        gathered = await asyncio.gather(
            *(self.run(child) for child in children), return_exceptions=True
        )
        ordered: list[AgentResult] = []
        for child, outcome in zip(children, gathered, strict=False):
            if isinstance(outcome, BaseException):
                child.cancel()
                ordered.append(child.result)
            else:
                ordered.append(outcome)
        return ordered + refused

    def cancel_all(self) -> None:
        for child in self.children:
            child.cancel()
        _write_scopes.release_agent(self.parent_id)

    # -- aggregation -------------------------------------------------------

    def aggregate(self) -> str:
        """Combine child results into one bounded report for the parent.

        This is context for the parent, not a verdict. Success here means "this
        child did not fail", never "the task is correct" - verification stays the
        parent's responsibility, which is the whole reason a child's report is
        treated as a claim rather than a conclusion.
        """
        if not self.results:
            return ""
        succeeded = [r for r in self.results if r.ok]
        failed = [r for r in self.results if not r.ok]
        header = (
            f"{len(self.results)} subagent(s) ran: {len(succeeded)} completed, "
            f"{len(failed)} did not."
        )
        header += (
            " A child that completed is not evidence that the work is correct, "
            "and one that did not is not evidence that it is wrong - verify "
            "before reporting success."
        )
        body = "\n".join(r.to_report() for r in self.results)
        text = f"{header}\n\n{body}"
        if len(text) > MAX_CONTEXT_CHARS:
            text = text[: MAX_CONTEXT_CHARS - 64].rstrip() + "\n[child reports truncated]"
        return text


async def spawn_agent(
    task: str,
    *,
    role: str = "researcher",
    context: str = "",
    skills: Sequence[str] = (),
    tools: Sequence[str] = (),
    model: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
    write_scope: str = "",
    constraints: str = "",
    parent_id: str = "",
    task_id: str = "",
    available_tools: Iterable[Any] | None = None,
    runner: Any = None,
) -> AgentResult:
    """Run one bounded child agent and return its result.

    The single-agent entry point. ``spawn_agents`` is the batch form; this exists
    so the common case - one delegation - does not need a spawner.
    """
    spawner = AgentSpawner(
        parent_id=parent_id,
        task_id=task_id,
        available_tools=available_tools,
        runner=runner,
    )
    spec = ChildSpec(
        task=task,
        role=role,
        context=context,
        skills=tuple(skills),
        tools=tuple(tools),
        model=model,
        provider=provider,
        timeout=timeout,
        write_scope=write_scope,
        constraints=constraints,
    )
    child = spawner.create(spec)
    return await spawner.run(child)


async def spawn_agents(specs: Sequence[ChildSpec], **kwargs: Any) -> list[AgentResult]:
    """Run several bounded children under one parent."""
    spawner = AgentSpawner(**kwargs)
    return await spawner.run_all(specs)
