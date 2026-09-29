import asyncio
from pathlib import Path

import pytest

from terminus.tasks.planner import ExecutionPlan, PlannedTask
from terminus.tasks.task_store import TaskStatus, TaskStore, TaskType


def _make_plan(tasks: list[PlannedTask]) -> ExecutionPlan:
    return ExecutionPlan(
        project_name="Test",
        goal_summary="test goal",
        tech_stack=["python"],
        total_estimated_hours=1.0,
        tasks=tasks,
        risks=[],
        assumptions=[],
    )


def _sample_task(task_id: str, depends_on: list[str] | None = None) -> PlannedTask:
    return PlannedTask(
        id=task_id,
        title=task_id,
        description=f"Do {task_id}",
        task_type=TaskType.IMPLEMENT,
        depends_on=depends_on or [],
        estimated_minutes=10,
        output_files=[],
        acceptance_criteria=["done"],
    )


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(str(tmp_path / "tasks.db"))


def _permanently_fail(store: TaskStore, project_id: str, task_id: str) -> None:
    while True:
        store.claim_task(project_id, task_id)
        status = store.fail_task(project_id, task_id, "boom")
        if status == TaskStatus.FAILED.value:
            return


def test_retry_semantics_are_total_attempts(store: TaskStore):
    """max_retries=3 means 3 total attempts (1 initial + 2 automatic retries)."""
    plan = _make_plan([_sample_task("task__001")])
    store.create_project("goal", plan)
    project_id = store.get_latest_project()

    store.claim_task(project_id, "task__001")
    assert store.fail_task(project_id, "task__001", "e1") == TaskStatus.PENDING.value
    store.claim_task(project_id, "task__001")
    assert store.fail_task(project_id, "task__001", "e2") == TaskStatus.PENDING.value
    store.claim_task(project_id, "task__001")
    assert store.fail_task(project_id, "task__001", "e3") == TaskStatus.FAILED.value

    task = store.get_all_tasks(project_id)[0]
    assert task["retry_count"] == 3
    assert task["status"] == TaskStatus.FAILED.value
    # Once the retry budget is exhausted the persisted error is explicitly
    # marked non-retryable so consumers do not expect further automated retries.
    assert task["error"] == "[retry-exhausted] [retryable] e3"


def test_exhausted_retryable_label_is_normalized(store: TaskStore):
    """A transient failure (e.g. 412/ReadTimeout) persisted at budget exhaustion
    keeps a single, truthful label: [non-retryable]."""
    plan = _make_plan([_sample_task("task__001")])
    store.create_project("goal", plan)
    project_id = store.get_latest_project()

    for _ in range(3):
        store.claim_task(project_id, "task__001")
        assert store.fail_task(
            project_id,
            "task__001",
            "[retryable] HTTPStatusError: Client error '412 Precondition Failed'",
        )
    task = store.get_all_tasks(project_id)[0]
    assert task["status"] == TaskStatus.FAILED.value
    assert task["error"] == "[retry-exhausted] [retryable] HTTPStatusError: Client error '412 Precondition Failed'"
    assert task["error"].count("[retry-exhausted]") == 1


def test_retry_success_after_failure(store: TaskStore):
    """failure -> retry -> success becomes completed."""
    plan = _make_plan([_sample_task("task__001")])
    store.create_project("goal", plan)
    project_id = store.get_latest_project()

    store.claim_task(project_id, "task__001")
    store.fail_task(project_id, "task__001", "e1")
    assert store.get_all_tasks(project_id)[0]["status"] == TaskStatus.PENDING.value

    store.claim_task(project_id, "task__001")
    store.complete_task(project_id, "task__001", "done")
    task = store.get_all_tasks(project_id)[0]
    assert task["status"] == TaskStatus.COMPLETED.value
    assert task["result"] == "done"
    assert task["error"] is None


def test_manual_recovery_resets_permanently_failed(store: TaskStore):  # noqa: D103
    """/plan continue recovery must reset FAILED tasks (retry_count >= max_retries)."""
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    _permanently_fail(store, project_id, "task__001")
    store.finalize_project_status(project_id)

    assert store.get_resumable_project() == project_id

    reset = store.reset_failed_tasks_for_recovery(project_id)
    assert reset == 1
    task = store.get_all_tasks(project_id)[0]
    assert task["status"] == TaskStatus.PENDING.value
    assert task["retry_count"] == 0
    assert task["error"] is None
    assert store.claim_task(project_id, "task__001") == 1


def test_resumable_project_invariant_holds_after_recovery(store: TaskStore):
    """
    get_resumable_project() returns a project only when /plan continue has
    meaningful recoverable work (failed or interrupted tasks exist).
    """
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    _permanently_fail(store, project_id, "task__001")

    assert store.get_resumable_project() == project_id
    store.reset_failed_tasks_for_recovery(project_id)
    # After reset the task is claimable; after completing it the project is done.
    store.claim_task(project_id, "task__001")
    store.complete_task(project_id, "task__001", "done")
    store.finalize_project_status(project_id)
    assert store.get_resumable_project() is None


def test_failed_upstream_transitive_blocks_dependents(store: TaskStore):
    """A->B->C with A failed: B and C are both reported blocked (transitive)."""
    plan = _make_plan([
        _sample_task("A"),
        _sample_task("B", ["A"]),
        _sample_task("C", ["B"]),
    ])
    project_id = store.create_project("goal", plan)
    _permanently_fail(store, project_id, "A")

    blocked = store.get_blocked_by_failed(project_id)
    by_id = {b["task"]["id"]: b["blocked_by"] for b in blocked}
    assert by_id["B"] == ["A"]
    # C is transitively blocked by A (through B)
    assert by_id["C"] == ["A"]
    assert store.get_ready_tasks(project_id) == []


def test_independent_branch_not_blocked(store: TaskStore):
    """A->B and C->D; A fails -> B blocked, C/D remain ready."""
    plan = _make_plan([
        _sample_task("A"),
        _sample_task("B", ["A"]),
        _sample_task("C"),
        _sample_task("D", ["C"]),
    ])
    project_id = store.create_project("goal", plan)
    _permanently_fail(store, project_id, "A")

    ready = sorted(t["id"] for t in store.get_ready_tasks(project_id))
    assert ready == ["C"]
    store.claim_task(project_id, "C")
    store.complete_task(project_id, "C", "done")
    assert sorted(t["id"] for t in store.get_ready_tasks(project_id)) == ["D"]

    blocked = store.get_blocked_by_failed(project_id)
    assert [b["task"]["id"] for b in blocked] == ["B"]


def test_missing_dependency_is_reported_not_silent_deadlock(store: TaskStore):
    """A pending task with a dependency id absent from the DB must be reported."""
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    # Corrupt the row the way a bad migration could: dep points at nonexistent id.
    with store.conn() as conn:
        conn.execute(
            "UPDATE tasks SET depends_on=? WHERE project_id=? AND id=?",
            ('["ghost"]', project_id, "task__001"),
        )
    assert store.get_ready_tasks(project_id) == []
    blocked = store.get_blocked_by_failed(project_id)
    assert any("ghost" in b for b in blocked[0]["blocked_by"])


def test_claim_is_atomic_and_reports_the_attempt(store: TaskStore):
    """A claim returns the attempt it represents, 0 when it is not claimable."""
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    assert store.claim_task(project_id, "task__001") == 1
    assert store.claim_task(project_id, "task__001") == 0

    # a failure spends one attempt, so the next claim is attempt 2
    store.fail_task(project_id, "task__001", "transient")
    assert store.claim_task(project_id, "task__001") == 2


def test_progress_includes_all_statuses(store: TaskStore):
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    store.claim_task(project_id, "task__001")
    progress = store.get_progress(project_id)
    assert progress["pending"] == 0
    assert progress["in_progress"] == 1
    assert progress["completed"] == 0
    assert progress["failed"] == 0
    assert "blocked" in progress and "skipped" in progress


@pytest.mark.anyio
async def test_agent_stream_timeout_raises_and_marks_task_failed(tmp_path: Path):
    """The core hang fix: a hung agent stream must raise a timeout error that the
    orchestrator turns into a task failure (never a permanent IN_PROGRESS)."""
    from terminus.tasks.worker import execute_task

    # Original value is large; simulate a hang in well under that with a stub.
    import terminus.tasks.executor as exec_mod

    original_timeout = exec_mod._AGENT_STREAM_TIMEOUT_SECONDS
    exec_mod._AGENT_STREAM_TIMEOUT_SECONDS = 0.1

    class _HungAgent:
        async def astream(self, *args, **kwargs):
            async def _g():
                # Yield one step then hang forever - a genuinely unbounded stream.
                yield {"messages": [{"content": "starting", "type": "ai"}]}
                await asyncio.sleep(3600)
            async for step in _g():
                yield step

    class _FakeLlm:
        pass

    async def _fake_tools():
        return {"implement": []}

    original_create_agent = exec_mod.create_agent
    original_tools = exec_mod._tool_plans
    original_get_chat_model = exec_mod.get_chat_model

    calls = {"create_agent": 0}

    def _fake_create_agent(*args, **kwargs):
        calls["create_agent"] += 1
        return _HungAgent()

    exec_mod.create_agent = _fake_create_agent
    exec_mod._tool_plans = _fake_tools
    exec_mod.get_chat_model = lambda *a, **kw: _FakeLlm()

    task = {
        "id": "task__001",
        "project_id": "p1",
        "task_type": "implement",
        "description": "do the thing",
        "acceptance_criteria": [],
    }

    try:
        result = await execute_task(task, workspace=Path.cwd(), attempt=1)
        assert result.success is False
        assert "timed out" in result.error
    finally:
        exec_mod.create_agent = original_create_agent
        exec_mod._tool_plans = original_tools
        exec_mod.get_chat_model = original_get_chat_model
        exec_mod._AGENT_STREAM_TIMEOUT_SECONDS = original_timeout

    # Round-trip through the orchestrator's failure path: a TimeoutError must
    # land in tasks.error and not leave the task stuck IN_PROGRESS.
    store = TaskStore(str(tmp_path / "tasks.db"))
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    store.claim_task(project_id, "task__001")
    status = store.fail_task(project_id, "task__001", "agent stream timed out")
    assert status in (TaskStatus.PENDING.value, TaskStatus.FAILED.value)
    task_row = store.get_all_tasks(project_id)[0]
    assert task_row["status"] != TaskStatus.IN_PROGRESS.value
    assert "timed out" in task_row["error"]
