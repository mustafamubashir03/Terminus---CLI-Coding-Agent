"""End-to-end: a frontend request selecting skills, delegating, and reporting.

Walks the whole path the architecture describes, with no live model:

    request -> skill selection -> planner task -> child reviewer -> skill
    injection -> aggregated report -> verification still required

The last step is the point of the test. A child completing is not the same as
the work being correct, and this asserts that distinction survives into the
parent's view.
"""

from __future__ import annotations

import asyncio
import functools

import pytest

from terminus.agents import (
    AgentResult,
    AgentSpawner,
    AgentStatus,
    ChildSpec,
    spawn_agents,
)
from terminus.agents.spawn import _reset_write_scopes
from terminus.skills.matcher import match_skills, render_selection
from terminus.skills.registry import (
    MAX_SKILLS_TOTAL_CHARS,
    SkillRegistry,
    builtin_skills_dir,
)
from terminus.tasks import executor as ex


def sync_async(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


@pytest.fixture(autouse=True)
def _clean():
    _reset_write_scopes()
    yield
    _reset_write_scopes()


@pytest.fixture(scope="module")
def library():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    return reg


REQUEST = "Build a polished React dashboard for our analytics product"


@sync_async
async def test_a_frontend_request_selects_the_expected_skills(library):
    selected = {m.name for m in match_skills(library, REQUEST) if m.selected}
    assert "frontend-design" in selected
    assert "react-best-practices" in selected


@sync_async
async def test_the_full_path_reaches_a_reviewer_and_aggregates(library):
    """Discovery -> selection -> child -> report, with the planner's task in the middle."""
    seen: list[dict] = {}

    async def fake_runner(child):
        seen["prompt"] = child.prompt()
        seen["role"] = child.role.name
        seen["skills"] = list(child.skills)
        seen["tools"] = list(child.tools)
        return "Reviewed the dashboard: one contrast issue, one missing focus style."

    spawner = AgentSpawner(runner=fake_runner, project_facts="Goal: ship the dashboard")
    # A planner-produced task, with the skills the planner would choose.
    task = {
        "id": "task__001",
        "task_type": "review",
        "description": "Review the React dashboard for performance and interface issues",
        "acceptance_criteria": "['no console errors', 'keyboard operable']",
    }
    child = spawner.create(ChildSpec(
        task=task["description"],
        role="reviewer",
        skills=["react-best-practices", "web-design-guidelines"],
        context="The dashboard lives in src/components/Dashboard.tsx",
    ))
    result = await spawner.run(child)
    report = spawner.aggregate()

    # 1. the child got the task, not a transcript
    assert "Review the React dashboard" in seen["prompt"]
    assert "cannot see the main conversation" in seen["prompt"]
    # 2. the named skills were honoured
    assert set(seen["skills"]) >= {"react-best-practices", "web-design-guidelines"}
    # 3. a reviewer cannot write
    assert "write_file" not in seen["tools"]
    assert seen["role"] == "reviewer"
    # 4. the result came back completed
    assert result.status == AgentStatus.COMPLETED
    assert "contrast" in result.summary
    # 5. and the parent still must verify
    assert "verify" in report.lower()


@sync_async
async def test_a_planner_task_reaches_a_worker_with_its_skills(library):
    """The same selection decision is made for a planned task, not just a question."""
    task = {
        "id": "task__001",
        "task_type": "implement",
        "description": "Build a polished React dashboard for our analytics product",
        "acceptance_criteria": "['renders charts', 'react components only']",
    }
    block = ex._worker_selected_skills(task)
    assert "frontend-design" in block
    assert "react-best-practices" in block
    assert len(block) <= MAX_SKILLS_TOTAL_CHARS


@sync_async
async def test_a_backend_task_reaches_the_worker_with_no_skills(library):
    task = {
        "id": "task__002",
        "task_type": "implement",
        "description": "Add a retry wrapper to the upstream HTTP client",
    }
    assert ex._worker_selected_skills(task) == ""


@sync_async
async def test_parallel_readers_and_one_writer_do_not_collide(library):
    async def fake_runner(child):
        return f"{child.role.name} finished {child.spec.task[:20]}"

    spawner = AgentSpawner(runner=fake_runner, project_facts="")
    results = await spawner.run_all([
        ChildSpec(task="audit performance", role="researcher"),
        ChildSpec(task="audit accessibility", role="reviewer"),
        ChildSpec(task="implement the chart", role="implementer", write_scope="src/Chart.tsx"),
        ChildSpec(task="implement the table", role="implementer", write_scope="src/Table.tsx"),
    ])
    assert all(r.status == AgentStatus.COMPLETED for r in results)


@sync_async
async def test_a_conflicting_second_writer_is_refused_not_run(library):
    async def fake_runner(child):
        return "wrote it"

    spawner = AgentSpawner(runner=fake_runner, project_facts="")
    first = spawner.create(ChildSpec(task="one", role="implementer", write_scope="src/A.tsx"))
    await spawner.run(first)
    first.claim_scope()
    second = spawner.create(ChildSpec(task="two", role="implementer", write_scope="src/A.tsx"))
    result = await spawner.run(second)
    assert result.status == AgentStatus.BLOCKED
    assert "already held" in result.error


@sync_async
async def test_spawn_agents_reports_one_failure_without_hiding_the_rest(library):
    async def fake_runner(child):
        if "bad" in child.spec.task:
            raise RuntimeError("investigation hit a wall")
        return "ok"

    results = await spawn_agents(
        [
            ChildSpec(task="good research", role="researcher"),
            ChildSpec(task="bad research", role="researcher"),
        ],
        runner=fake_runner,
        project_facts="",
    )
    statuses = [r.status for r in results]
    assert AgentStatus.COMPLETED in statuses
    assert AgentStatus.FAILED in statuses


@sync_async
async def test_a_spawner_can_be_retired_without_stranding_children(library):
    async def slow(child):
        await asyncio.sleep(5)

    spawner = AgentSpawner(runner=slow, project_facts="")
    spawner.create(ChildSpec(task="hangs", role="implementer", write_scope="src/A.tsx"))
    spawner.cancel_all()
    assert all(c.status == AgentStatus.CANCELLED for c in spawner.children)


def test_the_selection_block_never_exceeds_the_budget(library):
    for prompt in (REQUEST, "review my ui audit accessibility react next.js bundle"):
        block = render_selection(match_skills(library, prompt), library)
        assert len(block) <= MAX_SKILLS_TOTAL_CHARS


def test_the_child_result_type_is_the_shared_one(library):
    """Agents reuse the project's result shape rather than a second taxonomy."""
    result = AgentResult(agent_id="a", role="reviewer")
    for field in ("status", "summary", "findings", "error", "duration_seconds"):
        assert field in result.as_dict()
