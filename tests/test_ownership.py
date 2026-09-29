"""Single orchestrator ownership per project.

The contested cases here are deliberately real: a second *process*, not a second
object in the same interpreter. An in-process fake would pass even if the lock
were a plain instance attribute with no cross-process meaning at all, which is
the property that actually has to hold.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from terminus.ownership import (
    OwnershipConflict,
    OwnerInfo,
    ProcessOwnership,
    ProjectOwnership,
    current_ownership,
    lock_dir_for,
)
from terminus.permissions import (
    Operation,
    PermissionLevel,
    PermissionPolicy,
    set_permission_policy,
)
from terminus.tasks.orchestrator import (
    acquire_project_ownership,
    handle_plan_command,
    holds_project_ownership,
    release_project_ownership,
)
from terminus.tasks.planner import ExecutionPlan, PlannedTask, TaskType
from terminus.tasks.task_store import TaskStore
from terminus.tasks.worker import TaskResult
from terminus.workspace import project_root

# A second Terminus process: takes ownership of a project and holds it until it
# is told to stop, either by exiting cleanly or by dying without cleanup.
HOLDER = textwrap.dedent(
    """
    import json, os, sys, time
    sys.path.insert(0, r"G:\\terminus\\src")
    from terminus.ownership import ProjectOwnership, OwnershipConflict

    lock_dir, project_id, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    lock = ProjectOwnership(lock_dir, project_id)
    try:
        info = lock.acquire()
    except OwnershipConflict as exc:
        print(json.dumps({"acquired": False, "error": str(exc)}), flush=True)
        sys.exit(3)
    print(json.dumps({"acquired": True, "pid": info.pid, "host": info.host}), flush=True)
    if mode == "crash":
        os._exit(1)          # no cleanup at all: the OS must release the lock
    if mode == "exit":
        lock.release()
        sys.exit(0)
    sys.stdin.readline()      # "hold" until told to finish
    lock.release()
    """
)


def _holder_cmd(lock_dir: Path, project_id: str, mode: str) -> list[str]:
    return [sys.executable, "-c", HOLDER, str(lock_dir), project_id, mode]


class Holder:
    """A real second process holding (or trying to hold) a project."""

    def __init__(self, lock_dir: Path, project_id: str, mode: str = "hold") -> None:
        self.proc = subprocess.Popen(
            _holder_cmd(lock_dir, project_id, mode),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def report(self) -> dict:
        line = self.proc.stdout.readline().strip()
        return json.loads(line) if line else {}

    def finish(self, timeout: int = 30) -> int:
        try:
            self.proc.stdin.write("go\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        return self.proc.wait(timeout=timeout)

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=30)


def run_capture(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "LANGSMITH_TRACING": "false", "PYTHONPATH": r"G:\terminus\src"},
    )


def plan(*deps: list[str], count: int | None = None) -> ExecutionPlan:
    deps = list(deps)
    if count is not None:
        deps = [[] for _ in range(count)]
    return ExecutionPlan(
        project_name="P",
        goal_summary="ownership",
        tech_stack=["python"],
        total_estimated_hours=1.0,
        tasks=[
            PlannedTask(
                id=f"task__{i:03d}",
                title=f"t{i}",
                description=f"task {i}",
                task_type=TaskType.IMPLEMENT,
                depends_on=list(d),
                estimated_minutes=1,
                output_files=[],
                acceptance_criteria=["done"],
            )
            for i, d in enumerate(deps, start=1)
        ],
        risks=[],
        assumptions=[],
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A real TaskStore, also wired into CONFIG.

    ``handle_plan_command`` builds its own store from ``tasks.db_path``, so the
    command-level tests would otherwise look at Terminus's real project
    database instead of this fixture's. Pointing CONFIG at the temporary path
    keeps the command flow real while leaving the repository untouched.
    """
    from terminus.config import CONFIG

    db_path = str(tmp_path / "tasks" / "tasks.db")
    monkeypatch.setitem(CONFIG.setdefault("tasks", {}), "db_path", db_path)
    s = TaskStore(db_path=db_path)
    yield s
    current_ownership().release_all()


@pytest.fixture(autouse=True)
def no_leaked_ownership():
    """No test may leave this process holding a project."""
    yield
    current_ownership().release_all()


def seed(store: TaskStore, deps: list[list[str]] | None = None, count: int | None = None) -> str:
    if count is not None:
        deps = [[] for _ in range(count)]
    return store.create_project(
        "goal", plan(*(deps or [[]])), workspace=str(project_root())
    )


# ---------------------------------------------------------------------------
# A. first owner
# ---------------------------------------------------------------------------


def test_first_owner_acquires(store):
    pid = seed(store)
    info = acquire_project_ownership(store, pid)
    assert isinstance(info, OwnerInfo)
    assert info.pid == os.getpid()
    assert holds_project_ownership(pid)
    # The record names this process, so a refusal elsewhere can say who holds it.
    record = json.loads(
        (lock_dir_for(store.db_path) / f"owner-{pid}.owner.json").read_text("utf-8")
    )
    assert record["pid"] == os.getpid()
    assert record["host"] == info.host


# ---------------------------------------------------------------------------
# B. second owner is refused and changes nothing
# ---------------------------------------------------------------------------


def test_second_process_is_refused_and_changes_no_task_state(store):
    """A live owner's project must be untouched by the refused process."""
    pid = seed(store)
    # A task mid-flight, exactly as a real orchestrator would leave it.
    store.claim_task(pid, "task__001")
    before = {r["id"]: r["status"] for r in store.get_all_tasks(pid)}
    assert before["task__001"] == "in_progress"

    lock_dir = lock_dir_for(store.db_path)
    holder = Holder(lock_dir, pid)
    assert holder.report()["acquired"] is True
    try:
        # This process now tries to continue the same project, as a second
        # Terminus would.
        with pytest.raises(OwnershipConflict) as exc:
            acquire_project_ownership(store, pid)
        assert "already being orchestrated" in str(exc.value)
        assert not holds_project_ownership(pid)

        after = {r["id"]: r["status"] for r in store.get_all_tasks(pid)}
        assert after == before, "a refused acquisition must not touch task state"
        assert store.get_recovery_cycles(pid) == 0, "recovery must not have run"
    finally:
        assert holder.finish() == 0


def test_refused_continue_does_not_recover_a_live_orchestrators_task(store, capsys):
    """`/plan continue` under a live owner must not reset its running task."""
    pid = seed(store)
    store.claim_task(pid, "task__001")

    lock_dir = lock_dir_for(store.db_path)
    holder = Holder(lock_dir, pid)
    assert holder.report()["acquired"] is True
    try:
        asyncio.run(handle_plan_command("continue"))
        out = capsys.readouterr().out
        assert "already being orchestrated" in out
        row = store.get_all_tasks(pid)[0]
        assert row["status"] == "in_progress", (
            "the refused process recovered a task that is still running"
        )
        assert store.get_recovery_cycles(pid) == 0
    finally:
        assert holder.finish() == 0


def test_refused_acquisition_reports_holder_without_leaking(tmp_path, store):
    pid = seed(store)
    lock_dir = lock_dir_for(store.db_path)
    holder = Holder(lock_dir, pid)
    report = holder.report()
    try:
        with pytest.raises(OwnershipConflict) as exc:
            acquire_project_ownership(store, pid)
        message = str(exc.value)
        assert f"pid {report['pid']}" in message
        assert report["host"] in message
        # Diagnostics only: no environment, no arguments, no file contents.
        assert "OPENROUTER" not in message
        assert str(store.db_path) not in message
    finally:
        assert holder.finish() == 0


# ---------------------------------------------------------------------------
# C. clean release
# ---------------------------------------------------------------------------


def test_release_lets_the_next_process_take_over(store):
    pid = seed(store)
    acquire_project_ownership(store, pid)
    assert holds_project_ownership(pid)
    release_project_ownership(pid)
    assert not holds_project_ownership(pid)

    lock_dir = lock_dir_for(store.db_path)
    # A fresh process, as if the first had exited.
    result = run_capture(_holder_cmd(lock_dir, pid, "exit"), Path.cwd())
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[0])["acquired"] is True


def test_release_is_idempotent(store):
    pid = seed(store)
    acquire_project_ownership(store, pid)
    release_project_ownership(pid)
    release_project_ownership(pid)  # must not raise
    assert not holds_project_ownership(pid)


# ---------------------------------------------------------------------------
# D/E. crash recovery vs. a live owner
# ---------------------------------------------------------------------------


def test_crashed_owner_is_recovered_by_the_os(store):
    """A crash with no cleanup must not leave the project locked.

    Nothing is deleted to make this pass: the kernel drops the lock when the
    process dies, which is the whole reason this design was chosen over a
    recorded PID.
    """
    pid = seed(store)
    lock_dir = lock_dir_for(store.db_path)
    crashed = Holder(lock_dir, pid, mode="crash")
    report = crashed.report()
    assert report["acquired"] is True
    crashed.proc.wait(timeout=30)
    assert crashed.proc.returncode == 1, "the crashing process should not exit 0"

    # The lock file is still on disk; the *lock* is gone.
    assert (lock_dir / f"owner-{pid}.lock").exists()

    info = acquire_project_ownership(store, pid)
    assert info.pid == os.getpid()
    release_project_ownership(pid)


def test_new_owner_can_recover_interrupted_tasks_after_a_crash(store):
    """The whole point: B acquires, and only then recovers A's interrupted task."""
    pid = seed(store)
    store.claim_task(pid, "task__001")
    store.claim_task(pid, "task__002")
    store.complete_task(pid, "task__002", "done")
    assert store.get_all_tasks(pid)[0]["status"] == "in_progress"

    lock_dir = lock_dir_for(store.db_path)
    crashed = Holder(lock_dir, pid, mode="crash")
    crashed.report()
    crashed.proc.wait(timeout=30)

    # A live owner blocks the whole continue path, recovery included.
    alive = Holder(lock_dir, pid)
    assert alive.report()["acquired"] is True
    try:
        with pytest.raises(OwnershipConflict):
            acquire_project_ownership(store, pid)
    finally:
        assert alive.finish() == 0

    # With no owner, acquiring first and then recovering works.
    acquire_project_ownership(store, pid)
    assert store.recover_interrupted_tasks(pid) == 1
    assert store.get_all_tasks(pid)[0]["status"] == "pending"
    release_project_ownership(pid)


def test_a_live_owner_is_never_mistaken_for_stale(store):
    """Holding the lock for a long time must not look like staleness."""
    pid = seed(store)
    lock_dir = lock_dir_for(store.db_path)
    holder = Holder(lock_dir, pid)
    assert holder.report()["acquired"] is True
    try:
        time.sleep(2.0)  # well past any plausible heartbeat interval
        with pytest.raises(OwnershipConflict):
            acquire_project_ownership(store, pid)
        assert not holds_project_ownership(pid)
    finally:
        assert holder.finish() == 0


def test_slow_owner_is_not_preempted_while_working(store):
    """A live owner doing slow work keeps ownership for its whole run.

    Contention must come from another process: a second acquire in the same
    process is reference-counted, not a conflict.
    """
    pid = seed(store)
    lock_dir = lock_dir_for(store.db_path)
    holder = Holder(lock_dir, pid)
    assert holder.report()["acquired"] is True
    try:
        for _ in range(3):
            time.sleep(0.4)
            with pytest.raises(OwnershipConflict):
                acquire_project_ownership(store, pid)
        assert not holds_project_ownership(pid)
    finally:
        assert holder.finish() == 0


# ---------------------------------------------------------------------------
# F. recovery ordering
# ---------------------------------------------------------------------------


def test_ownership_is_acquired_before_recovery(store, monkeypatch):
    """Ordering is the whole safety property, so it is asserted directly."""
    pid = seed(store)
    store.claim_task(pid, "task__001")
    observed: list[tuple[str, bool]] = []

    import terminus.tasks.orchestrator as orch

    real_recover = orch.recover_project

    def spy(active_store, project_id):
        observed.append(("recover", holds_project_ownership(project_id)))
        return real_recover(active_store, project_id)

    async def spy_run(active_store, project_id):
        # Deliberately does not run the real orchestration. What is under test is
        # the *ordering* - that ownership is held before recovery and for the
        # whole run - and proving that does not require dispatching a real
        # worker, which would make the test depend on a live model and take
        # minutes. The real loop is covered by
        # test_owner_can_still_run_max_concurrent_two, which stubs the worker
        # deliberately.
        observed.append(("execute", holds_project_ownership(project_id)))

    monkeypatch.setattr(orch, "recover_project", spy)
    monkeypatch.setattr(orch, "_run_orchestration", spy_run)
    monkeypatch.setattr(orch, "_print_final_summary", lambda *a, **k: None)

    asyncio.run(handle_plan_command("continue"))

    assert [name for name, _ in observed] == ["recover", "execute"]
    assert all(owned for _, owned in observed), (
        f"ownership was not held for {observed}"
    )
    # And it is given back once the run is over.
    assert not holds_project_ownership(pid)


def test_recovery_is_impossible_without_ownership(store):
    """Recovery must be unreachable while another live process owns the project.

    ``recover_project`` on its own has no guard, by design - the command layer
    owns the ordering. So the guard is asserted where it is relied upon: the
    ownership scope must refuse to yield at all while the project is contended,
    which means the recovery body can never run.

    This is asserted by calling the scope rather than by reading the module
    source and comparing string offsets, which only ever verified that the code
    still looked a particular way rather than that it still worked.
    """
    pid = seed(store)
    store.claim_task(pid, "task__001")
    lock_dir = lock_dir_for(store.db_path)
    holder = Holder(lock_dir, pid)
    assert holder.report()["acquired"] is True
    try:
        import terminus.tasks.orchestrator as orch

        with pytest.raises(OwnershipConflict):
            with orch.project_ownership(store, pid):
                orch.recover_project(store, pid)
                pytest.fail("the ownership scope must not yield while contended")

        # Nothing was reset: the task is still as it was left.
        assert store.get_all_tasks(pid)[0]["status"] == "in_progress"
    finally:
        assert holder.finish() == 0


# ---------------------------------------------------------------------------
# G. one owner still runs concurrent workers
# ---------------------------------------------------------------------------


def test_owner_can_still_run_max_concurrent_two(store, monkeypatch):
    """Ownership must not serialise the workers it exists to protect."""
    import terminus.tasks.orchestrator as orch

    pid = seed(store, count=2)
    peak = {"value": 0}
    live = {"value": 0}
    lock = threading.Lock()

    async def worker(task, **_kwargs):
        with lock:
            live["value"] += 1
            peak["value"] = max(peak["value"], live["value"])
        try:
            await asyncio.sleep(0.3)
        finally:
            with lock:
                live["value"] -= 1
        return TaskResult.ok(output=f"done {task['id']}", attempt=1)

    monkeypatch.setattr(orch, "execute_task", worker)
    monkeypatch.setattr(orch, "_print_final_summary", lambda *a, **k: None)

    # Concurrency comes from config in the real flow, so opt in that way - which
    # also covers the documented opt-in path rather than a private constructor.
    from terminus.config import CONFIG

    monkeypatch.setitem(CONFIG["tasks"], "max_concurrent", 2)
    assert orch.configured_max_concurrent() == 2

    # asyncio.sleep is deliberately NOT stubbed: it is the shared asyncio
    # module, so patching it would also collapse the worker's own sleep and make
    # overlap unmeasurable. The between-batch pause is left in place instead.
    acquire_project_ownership(store, pid)
    asyncio.run(orch._run_orchestration(store, pid))

    assert peak["value"] == 2, f"workers were serialised (peak {peak['value']})"
    assert all(r["status"] == "completed" for r in store.get_all_tasks(pid))
    release_project_ownership(pid)



# ---------------------------------------------------------------------------
# H. different projects are independent
# ---------------------------------------------------------------------------


def test_different_projects_have_independent_owners(store):
    a = seed(store)
    b = store.create_project(
        "goal 2", plan(count=1), workspace=str(project_root())
    )
    acquire_project_ownership(store, a)
    acquire_project_ownership(store, b)
    assert holds_project_ownership(a) and holds_project_ownership(b)
    # Each has its own lock file, so neither is the other's blocker.
    lock_dir = lock_dir_for(store.db_path)
    assert (lock_dir / f"owner-{a}.lock").exists()
    assert (lock_dir / f"owner-{b}.lock").exists()
    current_ownership().release_all()
    assert not holds_project_ownership(a) and not holds_project_ownership(b)


def test_two_processes_own_two_projects_simultaneously(store):
    a = seed(store)
    b = store.create_project("goal 2", plan(count=1), workspace=str(project_root()))
    lock_dir = lock_dir_for(store.db_path)
    ha = Holder(lock_dir, a)
    hb = Holder(lock_dir, b)
    assert ha.report()["acquired"] is True
    assert hb.report()["acquired"] is True, "project B was blocked by project A"
    ha.finish()
    hb.finish()


# ---------------------------------------------------------------------------
# I/J/K. lifecycle: startup failure, /plan failure, continue
# ---------------------------------------------------------------------------


def test_failure_during_orchestration_releases_ownership(store, monkeypatch):
    pid = seed(store)
    import terminus.tasks.orchestrator as orch

    async def boom(s, project_id):
        acquire_project_ownership(s, project_id)
        try:
            raise RuntimeError("orchestration exploded")
        finally:
            release_project_ownership(project_id)

    monkeypatch.setattr(orch, "_run_orchestration", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(handle_plan_command("continue"))

    assert not holds_project_ownership(pid), "a failed run must not keep ownership"
    # Another process can now take it.
    result = run_capture(
        _holder_cmd(lock_dir_for(store.db_path), pid, "exit"), Path.cwd()
    )
    assert result.returncode == 0, result.stderr


def test_task_failure_semantics_survive_ownership(store, monkeypatch):
    """A task failing permanently keeps its own failure/retry accounting."""
    from terminus.tasks.errors import FailureInfo

    import terminus.tasks.orchestrator as orch

    pid = seed(store, count=1)
    calls: list[int] = []

    async def worker(task, **_kwargs):
        calls.append(1)
        return TaskResult.failed(
            FailureInfo(category="runtime", retryable=False, message="hard failure"),
            attempt=1,
        )

    monkeypatch.setattr(orch, "execute_task", worker)
    monkeypatch.setattr(orch, "_print_final_summary", lambda *a, **k: None)

    acquire_project_ownership(store, pid)
    asyncio.run(orch._run_orchestration(store, pid))
    release_project_ownership(pid)

    row = store.get_all_tasks(pid)[0]
    assert row["status"] == "failed"
    assert row["total_attempts"] == 1
    assert "hard failure" in (row["error"] or "")


def test_continue_acquires_and_releases_across_the_whole_run(store, monkeypatch):
    import terminus.tasks.orchestrator as orch

    pid = seed(store)
    store.claim_task(pid, "task__001")
    seen: list[bool] = []

    async def worker(task, **_kwargs):
        seen.append(holds_project_ownership(pid))
        return TaskResult.ok(output="done", attempt=1)

    monkeypatch.setattr(orch, "execute_task", worker)
    monkeypatch.setattr(orch, "_print_final_summary", lambda *a, **k: None)

    assert not holds_project_ownership(pid)
    asyncio.run(handle_plan_command("continue"))
    assert seen == [True], "worker ran without ownership"
    assert not holds_project_ownership(pid), "ownership outlived the run"
    assert store.get_all_tasks(pid)[0]["status"] == "completed"


def test_second_continue_after_a_clean_run_succeeds(store, monkeypatch):
    """Ownership must not be sticky: repeated /plan continue keeps working."""
    import terminus.tasks.orchestrator as orch

    pid = seed(store)

    async def worker(task, **_kwargs):
        return TaskResult.ok(output="done", attempt=1)

    monkeypatch.setattr(orch, "execute_task", worker)
    monkeypatch.setattr(orch, "_print_final_summary", lambda *a, **k: None)

    asyncio.run(handle_plan_command("continue"))
    store.claim_task(pid, "task__001")  # pretend it was interrupted
    asyncio.run(handle_plan_command("continue"))
    assert store.get_all_tasks(pid)[0]["status"] == "completed"


# ---------------------------------------------------------------------------
# L. no permission regression
# ---------------------------------------------------------------------------


def test_ownership_does_not_widen_permissions(store, tmp_path):
    """Holding ownership must not grant any capability on its own."""
    from terminus.execution import execution_scope, task_context
    from terminus.tools.filesystem_tools import write_file

    pid = seed(store)
    acquire_project_ownership(store, pid)
    try:
        target = tmp_path / "nope.txt"
        set_permission_policy(PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,)))
        ctx = task_context(
            task_id="t", project_id=pid, workspace=project_root(),
            policy=PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,)),
        )
        with execution_scope(ctx):
            result = write_file.invoke(
                {"file_path": str(target), "content": "x"}
            )
        assert "Refused" in result
        assert not target.exists()
    finally:
        set_permission_policy(PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,)))


def test_ownership_is_not_permission_state(store):
    """Acquiring ownership must not alter the active permission policy."""
    from terminus.permissions import get_permission_policy

    policy = PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,))
    set_permission_policy(policy)
    pid = seed(store)
    acquire_project_ownership(store, pid)
    assert get_permission_policy() is policy
    release_project_ownership(pid)
    assert get_permission_policy() is policy


# ---------------------------------------------------------------------------
# registry mechanics
# ---------------------------------------------------------------------------


def test_registry_is_reentrant_and_reference_counted(tmp_path):
    reg = ProcessOwnership()
    first = reg.acquire(tmp_path, "p")
    second = reg.acquire(tmp_path, "p")
    assert first.pid == second.pid
    reg.release("p")
    # Still held: one level remains, so a competing process must be refused.
    result = run_capture(_holder_cmd(tmp_path, "p", "exit"), Path.cwd())
    assert result.returncode == 3, "a reference-counted lock was released too early"
    reg.release("p")
    result = run_capture(_holder_cmd(tmp_path, "p", "exit"), Path.cwd())
    assert result.returncode == 0, result.stderr


def test_release_all_reports_what_it_released(tmp_path):
    reg = ProcessOwnership()
    reg.acquire(tmp_path, "a")
    reg.acquire(tmp_path, "b")
    assert sorted(reg.release_all()) == ["a", "b"]
    assert reg.owned() == []


def test_i_o_fault_is_not_reported_as_contention(tmp_path, monkeypatch):
    """A broken lock file must surface, not masquerade as a busy project."""
    from terminus import ownership

    def boom(handle):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(ownership, "_lock_exclusive", boom)
    lock = ProjectOwnership(tmp_path, "p")
    with pytest.raises(OSError) as exc:
        lock.acquire()
    assert not isinstance(exc.value, OwnershipConflict)
    assert not lock.held


def test_a_damaged_record_does_not_hide_the_conflict(tmp_path):
    """A corrupt owner record must not stop the refusal."""
    pid = "p"
    lock = ProjectOwnership(tmp_path, pid)
    lock.acquire()
    try:
        (tmp_path / f"owner-{pid}.owner.json").write_text("{not json", "utf-8")
        other = ProjectOwnership(tmp_path, pid)
        with pytest.raises(OwnershipConflict) as exc:
            other.acquire()
        assert "already being orchestrated" in str(exc.value)
    finally:
        lock.release()


def test_unlock_helper_is_used_after_release(tmp_path):
    pid = "p"
    lock = ProjectOwnership(tmp_path, pid)
    lock.acquire()
    lock.release()
    again = ProjectOwnership(tmp_path, pid)
    assert again.acquire().pid == os.getpid()
    again.release()


def test_operation_destructive_still_needs_its_own_permission(tmp_path):
    """Sanity: the guard is untouched by ownership and still level-based."""
    from terminus.coordination import project_write_guard

    set_permission_policy(PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,)))
    try:
        with project_write_guard(Operation.DELETE, target="x") as grant:
            assert grant.refused is not None
            assert grant.locked is False
    finally:
        set_permission_policy(PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,)))
