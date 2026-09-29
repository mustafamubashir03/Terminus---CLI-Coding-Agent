"""Delegation milestone: the real execution path, depth, telemetry, and isolation.

Everything here is deterministic and offline. The real agent factory is stubbed
where it would otherwise reach a provider, but the *wiring* is asserted: that a
child goes through the existing factory, on its own thread, under its own
policy, with the model route it was asked for.
"""

from __future__ import annotations

import asyncio
import functools
import json

import pytest

from terminus.agents import (
    MAX_CHILD_AGENTS,
    MAX_CHILD_DEPTH,
    AgentResult,
    AgentSpawner,
    AgentStatus,
    ChildSpec,
    recent_delegations,
    spawn_agent,
)
from terminus.agents.roles import ROLES
from terminus.agents.spawn import _reset_write_scopes
from terminus.execution import CHILD
from terminus.observability.usage_tracker import (
    clear_child_events,
    get_child_events,
)
from terminus.permissions import PermissionLevel


def sync_async(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@pytest.fixture(autouse=True)
def _clean():
    _reset_write_scopes()
    clear_child_events()
    yield
    _reset_write_scopes()
    clear_child_events()


async def ok_runner(child):
    return f"handled {child.spec.task[:30]}"


def spawner(**kw):
    kw.setdefault("runner", ok_runner)
    kw.setdefault("project_facts", "")
    return AgentSpawner(**kw)


# ---------------------------------------------------------------------------
# the real execution path
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Stands in for the compiled LangChain agent."""

    def __init__(self, sink):
        self.sink = sink

    async def ainvoke(self, payload, config=None, **kw):
        self.sink["config"] = config
        self.sink["payload"] = payload
        return {"messages": [{"role": "assistant", "content": "child answer"}]}


@sync_async
async def test_the_real_path_builds_an_agent_on_the_existing_factory(monkeypatch):
    """No second agent construction: the child's agent comes from build_agent."""
    import terminus.agent.factory as factory

    built = {}

    async def fake_build_agent(tools_override=None, *, model=None, provider=None,
                               max_model_calls=None):
        built["tools"] = tools_override
        built["model"] = model
        built["provider"] = provider
        built["max_model_calls"] = max_model_calls
        return _FakeAgent(built)

    monkeypatch.setattr(factory, "build_agent", fake_build_agent)
    monkeypatch.setattr(
        "terminus.agents.spawn._default_runner",
        __import__("terminus.agents.spawn", fromlist=["x"])._default_runner,
    )
    # keep the real runner, only stub the factory it calls

    result = await spawn_agent("inspect the auth flow", role="reviewer",
                               runner=None)
    assert built["model"] is None, "no override requested, so none is passed"
    assert built["max_model_calls"] == ROLES["reviewer"].max_model_calls
    assert result.status == AgentStatus.COMPLETED


@sync_async
async def test_the_child_runs_on_its_own_thread_not_the_parent_s(monkeypatch):
    """Reusing the parent's thread would replay the whole parent transcript."""
    import terminus.agent.factory as factory

    sink = {}

    async def fake_build_agent(**_kw):
        return _FakeAgent(sink)

    monkeypatch.setattr(factory, "build_agent", fake_build_agent)
    sp = spawner(runner=None)
    child = sp.create(ChildSpec(task="check the parser", role="reviewer"))
    await sp.run(child)
    thread = sink["config"]["configurable"]["thread_id"]
    assert thread.startswith("child-")
    assert child.agent_id in thread
    assert sp.parent_id not in thread


@sync_async
async def test_the_child_runs_under_its_own_execution_context(monkeypatch):
    """The role's policy is installed for the child while it actually runs.

    Asserted from inside the agent's own call, because that is where authority
    is enforced: whatever the tool layer reads at that moment is the child's
    policy, and a prompt that asks for more cannot change it.
    """
    import terminus.agent.factory as factory
    from terminus.execution import current_execution

    seen = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None, **kw):
            seen["ctx"] = current_execution()
            return {"messages": [{"role": "assistant", "content": "ok"}]}

    async def fake_build_agent(**_kw):
        return _CapturingAgent()

    monkeypatch.setattr(factory, "build_agent", fake_build_agent)

    sp = spawner(runner=None)
    child = sp.create(ChildSpec(task="look around", role="researcher"))
    from terminus.agents.spawn import _default_runner

    await _default_runner(child)

    context = seen["ctx"]
    assert context is not None, "the child ran with no execution scope at all"
    assert context.kind == CHILD
    assert context.parent_agent_id == sp.parent_id
    assert context.task_id == child.task_id
    assert PermissionLevel.WRITE in context.policy.deny_levels
    assert PermissionLevel.DESTRUCTIVE in context.policy.deny_levels


@sync_async
async def test_a_write_capable_child_gets_write_permission_under_its_own_scope(monkeypatch):
    import terminus.agent.factory as factory
    from terminus.execution import current_execution

    seen = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None, **kw):
            seen["ctx"] = current_execution()
            return {"messages": [{"role": "assistant", "content": "ok"}]}

    async def fake_build_agent(**_kw):
        return _CapturingAgent()

    monkeypatch.setattr(factory, "build_agent", fake_build_agent)
    sp = spawner(runner=None)
    child = sp.create(ChildSpec(task="edit it", role="implementer",
                                write_scope="src/A.tsx"))
    from terminus.agents.spawn import _default_runner

    await _default_runner(child)
    context = seen["ctx"]
    assert PermissionLevel.WRITE in context.policy.auto_approve
    assert PermissionLevel.DESTRUCTIVE in context.policy.deny_levels


def test_a_read_only_child_context_denies_writes():
    from terminus.agents.spawn import child_policy

    policy = child_policy(ROLES["researcher"])
    assert PermissionLevel.WRITE in policy.deny_levels
    assert PermissionLevel.DESTRUCTIVE in policy.deny_levels


@sync_async
async def test_a_requested_model_goes_through_the_existing_router(monkeypatch):
    import terminus.agent.factory as factory

    built = {}

    async def fake_build_agent(tools_override=None, *, model=None, provider=None,
                               max_model_calls=None):
        built.update(model=model, provider=provider)
        return _FakeAgent({})

    monkeypatch.setattr(factory, "build_agent", fake_build_agent)
    await spawn_agent("do it", role="researcher", model="openai/gpt-oss-120b",
                      provider="groq", runner=None)
    assert built["model"] == "openai/gpt-oss-120b"
    assert built["provider"] == "groq"


# ---------------------------------------------------------------------------
# depth: no recursive delegation
# ---------------------------------------------------------------------------


@sync_async
async def test_a_grandchild_is_refused():
    sp = spawner()
    sp.create(ChildSpec(task="child"))
    with pytest.raises(RuntimeError, match="depth limit"):
        sp.create(ChildSpec(task="grandchild", depth=MAX_CHILD_DEPTH))


def test_no_role_is_granted_the_spawn_tool():
    """Recursion is structurally impossible, not merely discouraged."""
    for name, role in ROLES.items():
        assert "spawn_agent" not in role.tool_names(), f"{name} could delegate again"


def test_the_depth_ceiling_is_one():
    assert MAX_CHILD_DEPTH == 1


@sync_async
async def test_a_child_cannot_widen_its_own_tools_by_asking():
    child = spawner().create(
        ChildSpec(task="x", role="researcher", tools=["spawn_agent", "write_file"])
    )
    assert "spawn_agent" not in child.tools
    assert "write_file" not in child.tools


# ---------------------------------------------------------------------------
# result shape
# ---------------------------------------------------------------------------


def test_a_result_carries_lineage_status_and_verification():
    result = AgentResult(
        agent_id="agent-1", role="implementer", status=AgentStatus.COMPLETED,
        summary="did the thing", findings=["a"], verification="passed",
    )
    data = result.as_dict()
    for key in ("agent_id", "role", "status", "summary", "findings",
                "verification", "duration_seconds", "skills", "tools"):
        assert key in data
    assert data["verification"] == "passed"


def test_verification_defaults_to_not_required():
    assert AgentResult(agent_id="a", role="researcher").verification == "not_required"


@sync_async
async def test_a_child_never_marks_itself_verified():
    """A child's claim is a claim; the parent owns verification."""
    result = await spawn_agent("did it", runner=ok_runner)
    assert result.verification != "verified"


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------


@sync_async
async def test_a_child_run_is_recorded_in_the_existing_tracker():
    sp = spawner(parent_id="parent-x")
    await sp.run(sp.create(ChildSpec(task="investigate", role="debugger")))
    events = get_child_events()
    assert len(events) == 1
    event = events[0]
    assert event["parent_agent_id"] == "parent-x"
    assert event["role"] == "debugger"
    assert event["status"] == AgentStatus.COMPLETED
    assert event["child_agent_id"].startswith("agent-")


@sync_async
async def test_a_failed_child_records_its_failure_category():
    async def boom(child):
        raise RuntimeError("child exploded")

    sp = spawner(runner=boom)
    await sp.run(sp.create(ChildSpec(task="explode", role="researcher")))
    event = get_child_events()[0]
    assert event["status"] == AgentStatus.FAILED
    assert event["failure_category"]


@sync_async
async def test_telemetry_records_no_prompt_or_output_text():
    """A child's answer can contain source code; telemetry must not store it."""
    async def chatty(child):
        return "SECRET_SOURCE_CODE from the child"

    sp = spawner(runner=chatty)
    await sp.run(sp.create(ChildSpec(task="TASK_TEXT_SHOULD_NOT_APPEAR")))
    blob = json.dumps(get_child_events())
    assert "SECRET_SOURCE_CODE" not in blob
    assert "TASK_TEXT_SHOULD_NOT_APPEAR" not in blob


def test_telemetry_is_bounded():
    from terminus.observability.usage_tracker import (
        CHILD_EVENT_LIMIT,
        record_child_event,
    )
    for i in range(CHILD_EVENT_LIMIT + 25):
        record_child_event(child_agent_id=f"a{i}", role="researcher", status="completed")
    assert len(get_child_events()) <= CHILD_EVENT_LIMIT


# ---------------------------------------------------------------------------
# CLI visibility
# ---------------------------------------------------------------------------


@sync_async
async def test_delegations_are_visible_as_a_tree():
    sp = spawner(parent_id="parent-tree")
    await sp.run(sp.create(ChildSpec(task="research", role="researcher")))
    await sp.run(sp.create(ChildSpec(task="review", role="reviewer")))
    view = recent_delegations()
    assert "parent-tree" in view
    assert "researcher" in view and "reviewer" in view
    assert "completed" in view


def test_the_view_is_empty_before_anything_runs():
    assert recent_delegations() == ""


# ---------------------------------------------------------------------------
# isolation of context
# ---------------------------------------------------------------------------


@sync_async
async def test_a_child_receives_only_what_the_parent_passed():
    parent_conversation = "USER: what about the database? ASSISTANT: the schema is fine"
    sp = spawner()
    child = sp.create(ChildSpec(
        task="Inspect the authentication flow",
        context="src/auth/oauth.py is the entry point",
        constraints="do not modify anything",
    ))
    prompt = child.prompt()
    assert "Inspect the authentication flow" in prompt
    assert "src/auth/oauth.py is the entry point" in prompt
    assert "do not modify anything" in prompt
    assert parent_conversation not in prompt
    assert "the schema is fine" not in prompt


@sync_async
async def test_project_context_is_bounded_in_a_child():
    sp = spawner(project_facts="X" * 50_000)
    child = sp.create(ChildSpec(task="look at everything"))
    assert len(child.prompt()) < 10_000


@sync_async
async def test_a_child_gets_only_its_assigned_skills():
    sp = spawner()
    child = sp.create(ChildSpec(task="unrelated work", skills=["mcp-builder"]))
    assert "mcp-builder" in child.skills
    assert "frontend-design" not in child.skills


@sync_async
async def test_a_child_inherits_nothing_it_was_not_given():
    sp = spawner()
    child = sp.create(ChildSpec(task="x", role="implementer"))
    assert "write_file" in child.tools, "the role allows it"
    assert "web_search" not in child.tools, "not in any role, so never granted"
    assert child.parent_id == sp.parent_id
    assert child.task_id == sp.task_id


@sync_async
async def test_bounded_parallelism_and_fan_out_survive_together():
    live = 0
    peak = 0

    async def counting(child):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return "ok"

    sp = spawner(runner=counting, max_parallel=2, max_children=MAX_CHILD_AGENTS)
    results = await sp.run_all(
        [ChildSpec(task=f"read {i}", role="researcher") for i in range(5)]
    )
    assert peak <= 2
    assert len(results) == 5
    assert all(r.status == AgentStatus.COMPLETED for r in results)
