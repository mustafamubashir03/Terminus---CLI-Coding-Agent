from pathlib import Path

import pytest

from terminus.tasks.planner import ExecutionPlan, PlannedTask, validate_plan
from terminus.tasks.task_store import ProjectStatus, TaskStatus, TaskStore, TaskType


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


def test_new_plan_does_not_auto_resume(store: TaskStore):
    plan = _make_plan([_sample_task("task__001")])
    project_a = store.create_project("goal A", plan)
    store.update_project_status(project_a, ProjectStatus.IN_PROGRESS.value)

    resumable = store.get_resumable_project()
    assert resumable == project_a

    plan_b = _make_plan([_sample_task("task__001")])
    project_b = store.create_project("goal B", plan_b)
    assert project_b != project_a
    assert store.get_resumable_project() == project_b


def test_completed_project_not_resumable(store: TaskStore):
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    store.claim_task(project_id, "task__001")
    store.complete_task(project_id, "task__001", "ok")
    store.finalize_project_status(project_id)

    assert store.get_resumable_project() is None
    assert store.get_latest_project() == project_id


def test_fail_task_retries_until_max(store: TaskStore):
    plan = _make_plan([_sample_task("task__001")])
    store.create_project("goal", plan)
    project_id = store.get_latest_project()
    store.claim_task(project_id, "task__001")

    status = store.fail_task(project_id, "task__001", "transient error")
    assert status == TaskStatus.PENDING.value

    tasks = store._get_all_tasks(store.get_latest_project())
    task = tasks[0]
    assert task["retry_count"] == 1
    assert task["status"] == TaskStatus.PENDING.value

    store.claim_task(project_id, "task__001")
    store.fail_task(project_id, "task__001", "again")
    store.claim_task(project_id, "task__001")
    status = store.fail_task(project_id, "task__001", "final")
    assert status == TaskStatus.FAILED.value

    task = store._get_all_tasks(store.get_latest_project())[0]
    assert task["retry_count"] == 3
    assert task["status"] == TaskStatus.FAILED.value


def test_dependency_blocks_until_upstream_completes(store: TaskStore):
    plan = _make_plan([
        _sample_task("task__001"),
        _sample_task("task__002", ["task__001"]),
    ])
    project_id = store.create_project("goal", plan)

    ready = store.get_ready_tasks(project_id)
    assert [t["id"] for t in ready] == ["task__001"]

    store.claim_task(project_id, "task__001")
    store.complete_task(project_id, "task__001", "done")

    ready = store.get_ready_tasks(project_id)
    assert [t["id"] for t in ready] == ["task__002"]


def test_failed_upstream_blocks_dependents(store: TaskStore):
    plan = _make_plan([
        _sample_task("task__001"),
        _sample_task("task__002", ["task__001"]),
    ])
    project_id = store.create_project("goal", plan)
    store.claim_task(project_id, "task__001")

    for _ in range(3):
        store.fail_task(project_id, "task__001", "error")
        if store._get_all_tasks(project_id)[0]["status"] == TaskStatus.PENDING.value:
            store.claim_task(project_id, "task__001")

    blocked = store.get_blocked_by_failed(project_id)
    assert len(blocked) == 1
    assert blocked[0]["task"]["id"] == "task__002"
    assert store.get_ready_tasks(project_id) == []


def test_recover_interrupted_tasks(store: TaskStore):
    plan = _make_plan([_sample_task("task__001")])
    project_id = store.create_project("goal", plan)
    store.claim_task(project_id, "task__001")

    recovered = store.recover_interrupted_tasks(project_id)
    assert recovered == 1
    task = store._get_all_tasks(project_id)[0]
    assert task["status"] == TaskStatus.PENDING.value


def test_validate_plan_rejects_cycle():
    plan = _make_plan([
        _sample_task("task__001", ["task__003"]),
        _sample_task("task__002", ["task__001"]),
        _sample_task("task__003", ["task__002"]),
    ])
    with pytest.raises(ValueError, match="cycle"):
        validate_plan(plan)


def test_validate_plan_rejects_unknown_dependency():
    plan = _make_plan([_sample_task("task__001", ["task__999"])])
    with pytest.raises(ValueError, match="unknown task"):
        validate_plan(plan)
