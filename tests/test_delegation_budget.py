"""Cumulative delegation budget, scoped to the parent execution.

The property: a parent execution may delegate a bounded number of children in
total, across every ``spawn_agent`` call it makes. Not per call, and not per
conversation - each new execution starts fresh.

These exercise the tool the model actually calls, so the budget is claimed
through the same path production uses rather than a private back door.
"""

from __future__ import annotations

import asyncio
import functools

import pytest

from terminus.agents.spawn import _reset_write_scopes
from terminus.execution import (
    MAX_CHILDREN_PER_PARENT,
    ExecutionBudget,
    ask_context,
    execution_scope,
)
from terminus.observability.usage_tracker import clear_child_events
from terminus.permissions import PermissionLevel, PermissionPolicy


def sync_async(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _policy() -> PermissionPolicy:
    return PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    )


async def _invoke_tool(*, count: int = 1):
    """Call the real spawn_agent tool *count* times and collect the replies."""
    from terminus.tools.spawn_agent_tool import spawn_agent

    return [
        await spawn_agent.ainvoke(
            {"task": f"investigate area {i}", "role": "researcher"}
        )
        for i in range(count)
    ]


@pytest.fixture(autouse=True)
def _clean():
    _reset_write_scopes()
    clear_child_events()
    yield
    _reset_write_scopes()
    clear_child_events()


@pytest.fixture
def stubbed_child(monkeypatch):
    ran: list[str] = []

    async def fake_child_runner(child):
        ran.append(child.spec.task)
        return f"[{child.role.name}] findings"

    monkeypatch.setattr("terminus.agents.spawn._default_runner", fake_child_runner)
    return ran


# ---------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------


def test_a_budget_grants_up_to_its_limit():
    budget = ExecutionBudget(max_children=8)
    assert [budget.claim(1) for _ in range(8)] == [1] * 8
    assert budget.used == 8
    assert budget.remaining == 0
    assert budget.claim(1) == 0


def test_a_partial_grant_returns_only_what_is_left():
    """Asking for five with three left must grant three, not zero and not five."""
    budget = ExecutionBudget(max_children=3)
    assert budget.claim(3) == 3
    assert budget.claim(5) == 0
    budget = ExecutionBudget(max_children=8)
    assert budget.claim(5) == 5
    assert budget.claim(5) == 3
    assert budget.used == 8


def test_claim_is_monotonic():
    budget = ExecutionBudget(max_children=4)
    seen = [budget.claim(1) for _ in range(10)]
    assert seen[:4] == [1, 1, 1, 1]
    assert set(seen[4:]) == {0}
    assert budget.used == 4


# ---------------------------------------------------------------------------
# Test A: one call may use the whole budget
# ---------------------------------------------------------------------------


@sync_async
async def test_a_eight_children_in_one_execution_are_allowed(stubbed_child):
    context = ask_context(_policy(), max_children=8)
    with execution_scope(context):
        replies = await _invoke_tool(count=8)
    assert len(stubbed_child) == 8
    assert all("Not spawned" not in r for r in replies)
    assert context.budget.used == 8
    assert context.budget.remaining == 0


# ---------------------------------------------------------------------------
# Test B: the second call is cut off, and says so
# ---------------------------------------------------------------------------


@sync_async
async def test_b_a_second_call_is_refused_once_the_budget_is_spent(stubbed_child):
    """Five delegations, then the allowance is not per-call and runs dry.

    ``spawn_agent`` delegates one child per call, so "5 + 5" is ten sequential
    calls: the first five are allowed, the next three use what is left, and the
    last two are refused and said so.
    """
    context = ask_context(_policy(), max_children=8)
    with execution_scope(context):
        replies = list(await _invoke_tool(count=10))

    allowed = [r for r in replies if "Not spawned" not in r]
    refused = [r for r in replies if "Not spawned" in r]
    assert len(allowed) == 8
    assert len(refused) == 2
    assert len(stubbed_child) == 8
    assert context.budget.used == 8
    # The refusal names the cause rather than looking like a transient failure.
    assert "allowance" in refused[0]


# ---------------------------------------------------------------------------
# Test C: no amount of calling gets past the ceiling
# ---------------------------------------------------------------------------


@sync_async
async def test_c_repeated_calls_never_exceed_the_budget(stubbed_child):
    context = ask_context(_policy())
    with execution_scope(context):
        for _ in range(20):
            await _invoke_tool(count=3)
    assert len(stubbed_child) == MAX_CHILDREN_PER_PARENT
    assert context.budget.used == MAX_CHILDREN_PER_PARENT


# ---------------------------------------------------------------------------
# Test D: a new execution starts fresh
# ---------------------------------------------------------------------------


@sync_async
async def test_d_a_new_parent_execution_gets_a_full_budget(stubbed_child):
    first = ask_context(_policy())
    with execution_scope(first):
        await _invoke_tool(count=8)
    assert first.budget.remaining == 0

    second = ask_context(_policy())
    with execution_scope(second):
        await _invoke_tool(count=3)
    assert second.budget.remaining == MAX_CHILDREN_PER_PARENT - 3
    assert len(stubbed_child) == 11


# ---------------------------------------------------------------------------
# Test E: concurrency and total budget are different limits
# ---------------------------------------------------------------------------


def test_e_total_budget_and_parallelism_are_independent():
    from terminus.agents.spawn import MAX_PARALLEL_AGENTS

    assert MAX_CHILDREN_PER_PARENT == 8
    assert MAX_PARALLEL_AGENTS == 3
    assert MAX_CHILDREN_PER_PARENT > MAX_PARALLEL_AGENTS
    # The total budget is a count over the whole execution, not a rate limit.
    budget = ExecutionBudget(max_children=MAX_CHILDREN_PER_PARENT)
    assert budget.claim(MAX_CHILDREN_PER_PARENT) == MAX_CHILDREN_PER_PARENT


# ---------------------------------------------------------------------------
# scoping
# ---------------------------------------------------------------------------


@sync_async
async def test_the_budget_lives_on_the_execution_not_in_global_state(stubbed_child):
    """Two concurrent executions must not share one allowance."""
    a = ask_context(_policy())
    b = ask_context(_policy())
    with execution_scope(a):
        await _invoke_tool(count=8)
    with execution_scope(b):
        await _invoke_tool(count=2)
    assert a.budget.used == 8
    assert b.budget.used == 2


@sync_async
async def test_a_child_execution_carries_no_delegation_budget(stubbed_child):
    """Children may not delegate, so they must not hold an allowance."""
    from pathlib import Path

    from terminus.execution import CHILD, ExecutionContext

    child_ctx = ExecutionContext(workspace=Path.cwd(), kind=CHILD, parent_agent_id="p")
    assert child_ctx.budget is None


def test_task_executions_also_get_a_budget():
    from terminus.tasks.task_store import TaskStore  # noqa: F401
    from terminus.tasks.worker import worker_permission_policy

    from terminus.execution import task_context

    context = task_context("task-1", "proj-1", __import__("pathlib").Path.cwd(),
                           worker_permission_policy())
    assert context.budget is not None
    assert context.budget.max_children == MAX_CHILDREN_PER_PARENT
