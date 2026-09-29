"""Child agents: identity, bounds, isolation, lifecycle, and write safety.

The properties under test:

    A child is uniquely identified and traceable to its parent; it sees a
    bounded task context rather than a transcript; its tools, skills,
    permissions and deadline are fixed before it runs and cannot be widened
    afterwards; failure, timeout and cancellation are distinct states; fan-out
    and parallelism are bounded; and two writers can never hold the same scope.
"""

from __future__ import annotations

import asyncio
import functools

import pytest

from terminus.agents import (
    MAX_CHILD_AGENTS,
    MAX_PARALLEL_AGENTS,
    AgentResult,
    AgentSpawner,
    AgentStatus,
    ChildSpec,
    build_child_context,
    child_policy,
    spawn_agent,
)
from terminus.agents.roles import ROLES, describe_roles, get_role, role_names
from terminus.agents.spawn import _reset_write_scopes, write_scope_holder
from terminus.permissions import PermissionLevel


@pytest.fixture(autouse=True)
def _clean_scopes():
    _reset_write_scopes()
    yield
    _reset_write_scopes()


# The project runs async tests with asyncio.run inside a sync test (see
# tests/test_concurrency.py); pytest-asyncio is not installed. This adapter keeps
# the tests readable while matching that convention.
def sync_async(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


async def ok_runner(child):
    return f"handled: {child.spec.task[:40]}"


async def boom_runner(child):
    raise RuntimeError("the child could not do the thing")


async def slow_runner(child):
    await asyncio.sleep(5)
    return "never"


def spawner(**kwargs):
    kwargs.setdefault("runner", ok_runner)
    kwargs.setdefault("project_facts", "")
    return AgentSpawner(**kwargs)


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------


def test_the_five_initial_roles_exist():
    assert role_names() == ["debugger", "implementer", "researcher", "reviewer", "tester"]


def test_read_only_roles_cannot_write():
    for name in ("researcher", "reviewer", "tester", "debugger"):
        assert ROLES[name].write is False, f"{name} must not be able to write"
        policy = child_policy(ROLES[name])
        assert PermissionLevel.WRITE in policy.deny_levels
        assert PermissionLevel.DESTRUCTIVE in policy.deny_levels


def test_the_implementer_can_write_but_never_destructively():
    role = ROLES["implementer"]
    assert role.write is True
    policy = child_policy(role)
    assert PermissionLevel.WRITE in policy.auto_approve
    assert PermissionLevel.DESTRUCTIVE in policy.deny_levels


def test_no_builtin_role_may_be_destructive():
    assert all(not role.destructive for role in ROLES.values())


def test_a_destructive_role_is_refused_at_policy_construction():
    from dataclasses import replace

    role = replace(ROLES["researcher"], destructive=True)
    with pytest.raises(ValueError, match="not grantable"):
        child_policy(role)


def test_roles_are_described_for_the_parent():
    text = describe_roles()
    for name in role_names():
        assert name in text
    assert "write: no" in text


def test_an_unknown_role_is_rejected():
    assert get_role("nonexistent") is None
    with pytest.raises(ValueError, match="unknown agent role"):
        spawner().create(ChildSpec(task="x", role="wizard"))


# ---------------------------------------------------------------------------
# identity and lineage
# ---------------------------------------------------------------------------


@sync_async
async def test_a_child_gets_a_unique_id():
    sp = spawner()
    ids = {sp.create(ChildSpec(task=f"task {i}")).agent_id for i in range(4)}
    assert len(ids) == 4
    assert all(i.startswith("agent-") for i in ids)


@sync_async
async def test_a_child_records_its_parent_and_task():
    sp = spawner(parent_id="parent-42", task_id="task-7")
    child = sp.create(ChildSpec(task="investigate"))
    assert child.parent_id == "parent-42"
    assert child.task_id == "task-7"


@sync_async
async def test_a_parent_gets_an_id_when_it_does_not_supply_one():
    sp = spawner()
    assert sp.parent_id.startswith("parent-")


# ---------------------------------------------------------------------------
# context isolation and bounds
# ---------------------------------------------------------------------------


def test_child_context_is_bounded():
    huge = build_child_context("do it", context="X" * 100_000)
    assert len(huge) < 10_000
    assert "truncated" in huge


def test_child_context_carries_only_what_it_was_given():
    text = build_child_context(
        "audit X", context="some facts", project_facts="goal: y", constraints="no writes"
    )
    for expected in ("# Task", "audit X", "some facts", "goal: y", "no writes"):
        assert expected in text


def test_a_child_prompt_does_not_contain_a_transcript():
    child = spawner().create(ChildSpec(task="check the parser"))
    prompt = child.prompt()
    assert "check the parser" in prompt
    assert "cannot see the main conversation" in prompt
    for marker in ("user:", "assistant:", "<conversation>"):
        assert marker not in prompt.lower()


def test_a_child_prompt_states_its_own_limits():
    child = spawner().create(ChildSpec(task="x", role="researcher"))
    prompt = child.prompt()
    assert "You may write: no" in prompt
    assert "Allowed tools:" in prompt


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


@sync_async
async def test_a_researcher_gets_no_write_or_shell_tools():
    child = spawner().create(ChildSpec(task="look around", role="researcher"))
    assert "write_file" not in child.tools
    assert "edit_file" not in child.tools
    assert "run_command" not in child.tools
    assert "read_file" in child.tools


@sync_async
async def test_a_requested_tool_outside_the_role_is_dropped():
    child = spawner().create(
        ChildSpec(task="x", role="researcher", tools=["write_file", "read_file"])
    )
    assert "write_file" not in child.tools
    assert "read_file" in child.tools
    assert "write_file" in child.out_of_role_tools


@sync_async
async def test_a_tool_the_parent_lacks_is_dropped():
    class OnlyRead:
        name = "read_file"

    child = spawner(available_tools=[OnlyRead()]).create(
        ChildSpec(task="x", role="implementer")
    )
    assert child.tools == ["read_file"]
    assert "write_file" in child.rejected_tools, "dropped for lacking it, not for role"


@sync_async
async def test_tools_are_fixed_before_the_child_runs():
    child = spawner().create(ChildSpec(task="x", role="implementer"))
    before = list(child.tools)
    await child.run()
    assert child.tools == before, "a running child must not gain tools"


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


@sync_async
async def test_a_child_receives_selected_skills():
    child = spawner().create(
        ChildSpec(task="Audit the React frontend for performance", role="reviewer")
    )
    assert "react-best-practices" in child.skills
    assert "react-best-practices" in child.skill_block


@sync_async
async def test_an_explicitly_named_skill_reaches_the_child():
    child = spawner().create(
        ChildSpec(task="do something unrelated", skills=["systematic-debugging"])
    )
    assert "systematic-debugging" in child.skills


@sync_async
async def test_child_skills_are_recorded_on_the_result():
    sp = spawner()
    child = sp.create(ChildSpec(task="Review the UI for accessibility", role="reviewer"))
    result = await sp.run(child)
    assert result.skills == child.skills


@sync_async
async def test_a_missing_skill_directory_does_not_break_a_child():
    class NoRegistry:
        @staticmethod
        def _get_registry():
            raise RuntimeError("no skills")

    import terminus.skills.skill_tools as st

    original = st._get_registry
    st._get_registry = NoRegistry._get_registry
    try:
        child = spawner().create(ChildSpec(task="x"))
        assert child.skills == []
    finally:
        st._get_registry = original


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


@sync_async
async def test_a_successful_child_completes():
    result = await spawn_agent("investigate the retry logic", runner=ok_runner)
    assert result.status == AgentStatus.COMPLETED
    assert result.ok
    assert "investigate" in result.summary
    assert result.duration_seconds >= 0


@sync_async
async def test_a_failing_child_is_failed_with_a_reason():
    sp = spawner(runner=boom_runner)
    result = await sp.run(sp.create(ChildSpec(task="explode")))
    assert result.status == AgentStatus.FAILED
    assert not result.ok
    assert "the child could not do the thing" in result.error
    assert result.failure is not None


@sync_async
async def test_a_slow_child_times_out():
    sp = spawner(runner=slow_runner)
    result = await sp.run(sp.create(ChildSpec(task="hang", timeout=1)))
    assert result.status == AgentStatus.TIMED_OUT
    assert "timed out" in result.error


@sync_async
async def test_a_cancelled_child_is_cancelled():
    sp = spawner(runner=slow_runner)
    child = sp.create(ChildSpec(task="hang", timeout=30))
    child.cancel()
    assert child.status == AgentStatus.CANCELLED
    assert child.result.status == AgentStatus.CANCELLED


@sync_async
async def test_cancelling_a_finished_child_does_nothing():
    sp = spawner()
    child = sp.create(ChildSpec(task="quick"))
    await sp.run(child)
    child.cancel()
    assert child.status == AgentStatus.COMPLETED


@sync_async
async def test_an_empty_task_is_refused_rather_than_run():
    result = await spawn_agent("   ", runner=ok_runner)
    assert result.status == AgentStatus.BLOCKED
    assert "untraceable" in result.error


@sync_async
async def test_every_terminal_state_is_distinct():
    assert AgentStatus.FAILED != AgentStatus.TIMED_OUT != AgentStatus.CANCELLED
    assert AgentStatus.BLOCKED not in (AgentStatus.COMPLETED, AgentStatus.FAILED)
    for name in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT", "BLOCKED"):
        assert getattr(AgentStatus, name) in AgentStatus.TERMINAL


# ---------------------------------------------------------------------------
# bounds on fan-out
# ---------------------------------------------------------------------------


@sync_async
async def test_the_child_count_is_capped():
    sp = spawner()
    for i in range(MAX_CHILD_AGENTS):
        sp.create(ChildSpec(task=f"t{i}"))
    with pytest.raises(RuntimeError, match="child agent limit"):
        sp.create(ChildSpec(task="one too many"))


@sync_async
async def test_the_cannot_be_raised_above_the_hard_ceiling():
    assert AgentSpawner(max_children=999).max_children == MAX_CHILD_AGENTS


@sync_async
async def test_parallelism_is_capped():
    assert AgentSpawner(max_parallel=999).max_parallel == MAX_PARALLEL_AGENTS
    assert AgentSpawner(max_parallel=0).max_parallel == 1


@sync_async
async def test_parallelism_is_actually_enforced():
    live = 0
    peak = 0

    async def counting_runner(child):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return "done"

    sp = spawner(runner=counting_runner, max_parallel=2)
    await sp.run_all([ChildSpec(task=f"t{i}", role="researcher") for i in range(6)])
    assert peak <= 2, f"ran {peak} children at once"


@sync_async
async def test_a_refused_spawn_is_recorded_not_swallowed():
    sp = spawner(max_children=1)
    specs = [ChildSpec(task="a"), ChildSpec(task="b"), ChildSpec(task="c")]
    results = await sp.run_all(specs)
    assert len(results) == 3
    blocked = [r for r in results if r.status == AgentStatus.BLOCKED]
    assert len(blocked) == 2
    assert all("limit" in r.error for r in blocked)


@sync_async
async def test_results_come_back_in_request_order():
    async def jitter(child):
        # Later specs finish first, so ordering cannot come from completion.
        await asyncio.sleep(0.02 if "first" in child.spec.task else 0.0)
        return child.spec.task

    sp = spawner(runner=jitter)
    results = await sp.run_all([
        ChildSpec(task="first task"), ChildSpec(task="second task"),
    ])
    assert [r.summary for r in results] == ["first task", "second task"]


@sync_async
async def test_one_failing_child_does_not_stop_the_others():
    calls = []

    async def sometimes(child):
        calls.append(child.spec.task)
        if "bad" in child.spec.task:
            raise RuntimeError("nope")
        return "ok"

    sp = spawner(runner=sometimes)
    results = await sp.run_all([
        ChildSpec(task="good one"), ChildSpec(task="bad one"), ChildSpec(task="good two"),
    ])
    assert len(calls) == 3
    assert [r.status for r in results] == [
        AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.COMPLETED,
    ]


# ---------------------------------------------------------------------------
# write ownership
# ---------------------------------------------------------------------------


@sync_async
async def test_a_write_scope_is_claimed_and_released():
    sp = spawner()
    child = sp.create(ChildSpec(task="edit", role="implementer", write_scope="src/a.py"))
    await sp.run(child)
    # Released once the child finished, so the next one may take it.
    assert write_scope_holder("src/a.py") is None


@sync_async
async def test_two_writers_cannot_share_a_scope():
    sp = spawner()
    first = sp.create(ChildSpec(task="one", role="implementer", write_scope="src/a.py"))
    first.claim_scope()
    second = sp.create(ChildSpec(task="two", role="implementer", write_scope="src/a.py"))
    result = await second.run()
    assert result.status == AgentStatus.BLOCKED
    assert "already held" in result.error


@sync_async
async def test_a_different_scope_is_fine():
    sp = spawner()
    a = sp.create(ChildSpec(task="a", role="implementer", write_scope="src/a.py"))
    b = sp.create(ChildSpec(task="b", role="implementer", write_scope="src/b.py"))
    a.claim_scope()
    result = await sp.run(b)
    assert result.status == AgentStatus.COMPLETED
    a.release_scope()


@sync_async
async def test_a_read_only_child_never_takes_a_scope():
    sp = spawner()
    child = sp.create(ChildSpec(task="look", role="researcher", write_scope="src/a.py"))
    await sp.run(child)
    assert write_scope_holder("src/a.py") is None


@sync_async
async def test_a_scope_is_released_even_when_the_child_fails():
    sp = spawner(runner=boom_runner)
    child = sp.create(ChildSpec(task="boom", role="implementer", write_scope="src/a.py"))
    await sp.run(child)
    assert write_scope_holder("src/a.py") is None


@sync_async
async def test_parallel_read_only_children_are_allowed():
    sp = spawner()
    results = await sp.run_all([
        ChildSpec(task=f"read {i}", role="researcher", write_scope=f"file{i}.py")
        for i in range(4)
    ])
    assert all(r.status == AgentStatus.COMPLETED for r in results)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@sync_async
async def test_aggregation_reports_success_and_failure():
    sp = spawner()
    await sp.run(sp.create(ChildSpec(task="fine", role="researcher")))
    text = sp.aggregate()
    assert "1 subagent" in text
    assert "verify" in text.lower(), "aggregation must not read as a verdict"


@sync_async
async def test_aggregation_states_a_failure_clearly():
    sp = spawner(runner=boom_runner)
    await sp.run(sp.create(ChildSpec(task="boom", role="researcher")))
    text = sp.aggregate()
    assert "FAILED" in text


@sync_async
async def test_aggregation_of_nothing_is_empty():
    assert spawner().aggregate() == ""


@sync_async
async def test_aggregation_is_bounded():
    sp = spawner()

    async def verbose(child):
        return "Z" * 5_000

    sp.runner = verbose
    for i in range(MAX_CHILD_AGENTS):
        await sp.run(sp.create(ChildSpec(task=f"t{i}")))
    assert len(sp.aggregate()) < 8_000


def test_a_result_serialises_for_aggregation():
    result = AgentResult(agent_id="a1", role="reviewer", status=AgentStatus.COMPLETED,
                         summary="found two issues", findings=["a", "b"],
                         skills=["x"], tools=["read_file"], duration_seconds=1.234)
    data = result.as_dict()
    assert data["agent_id"] == "a1"
    assert data["duration_seconds"] == 1.23
    assert data["findings"] == ["a", "b"]
    assert "found two issues" in result.to_report()
