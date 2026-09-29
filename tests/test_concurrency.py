"""Concurrency and project write coordination.

Two independent mechanisms are under test, and they are deliberately kept
separate:

1. ``TaskOrchestrator(max_concurrent=N)`` - how many *tasks* run at once.
2. ``project_write_guard`` - how many of those tasks may *mutate the workspace*
   at once, which is always one per project regardless of N.

The second is not redundant: two read-only tasks should overlap freely, and a
task that spends minutes talking to a model must not hold a lock merely because
it might later write a file.

Overlap is proven with ``threading.Barrier`` / ``Event`` rather than sleeps, so
a serial implementation cannot pass these tests.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import threading
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from terminus.coordination import (
    WRITE_WAIT_SECONDS,
    is_writing,
    project_write_guard,
    tracked_workspaces,
)
from terminus.execution import execution_scope, task_context
from terminus.permissions import (
    Operation,
    PermissionLevel,
    PermissionPolicy,
    get_permission_policy,
    set_permission_policy,
)
from terminus.workspace import project_root

WRITE_OK = PermissionPolicy(
    auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE)
)
READ_ONLY = PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,))


@pytest.fixture
def allow_write():
    """Permit non-destructive mutation, no approver (the worker policy)."""
    set_permission_policy(WRITE_OK)
    yield
    set_permission_policy(READ_ONLY)


def _in_task(label: str = "task", workspace=None):
    """Enter a real task ExecutionContext, exactly as a worker does.

    The policy is whatever is currently installed, so a test that narrows it with
    ``set_permission_policy`` is genuinely testing that narrower policy rather
    than one this helper silently re-widened. The lock is keyed off the same
    identity the runtime uses - not a test shortcut.
    """
    return execution_scope(
        task_context(
            task_id=label,
            project_id="p1",
            workspace=workspace or project_root(),
            policy=get_permission_policy(),
        )
    )


def _write_guard(**kwargs):
    return project_write_guard(Operation.WRITE, target="f.txt", **kwargs)


def _start(fn, *args):
    """Start *fn* in a thread with this thread's contextvars copied in.

    This mirrors how LangChain actually runs a synchronous tool: ``arun`` hands
    work to the default executor through ``copy_context().run``, so a ContextVar
    policy set on the calling side is visible inside the tool. Without this the
    tests would be exercising a context the runtime never produces.
    """
    result: dict[str, object] = {}
    # Captured here, entered inside the thread: a fresh Context cannot be
    # entered twice at once, and each _start call copies its own.
    captured = contextvars.copy_context()

    def runner():
        try:
            result["value"] = captured.run(fn, *args)
        except BaseException as exc:  # surfaced by _join below
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, result


def _join(thread, result, timeout=15):
    """Join and re-raise whatever the thread body raised."""
    thread.join(timeout)
    assert not thread.is_alive(), "worker thread did not finish"
    if "error" in result:
        raise result["error"]
    return result.get("value")


# ---------------------------------------------------------------------------
# Default: behaviour unchanged
# ---------------------------------------------------------------------------


def test_default_max_concurrent_is_one():
    from terminus.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["tasks"]["max_concurrent"] == 1


@pytest.mark.parametrize(
    "given,expected",
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (0, 1),
        (-5, 1),
        (10_000, 4),
        (None, 1),
        ("nonsense", 1),
        ({}, 1),
    ],
)
def test_clamp_keeps_serial_default_and_bounds_fanout(given, expected):
    from terminus.tasks.orchestrator import clamp_max_concurrent

    assert clamp_max_concurrent(given) == expected


def test_configured_default_is_serial():
    from terminus.tasks.orchestrator import configured_max_concurrent

    assert configured_max_concurrent() == 1


def test_orchestrator_clamps_at_construction():
    from terminus.tasks.orchestrator import MAX_CONCURRENT_TASKS, TaskOrchestrator

    assert TaskOrchestrator(None).max_concurrent == 1
    assert TaskOrchestrator(None, max_concurrent=2).max_concurrent == 2
    assert (
        TaskOrchestrator(None, max_concurrent=999).max_concurrent
        == MAX_CONCURRENT_TASKS
    )


# ---------------------------------------------------------------------------
# Same-project mutation serialises
# ---------------------------------------------------------------------------


def test_same_workspace_writers_are_serialised(allow_write):
    """A enters, B waits, A exits, B enters - never both inside at once."""
    events: list[str] = []
    a_inside = threading.Event()
    b_entered = threading.Event()
    release_a = threading.Event()

    def writer_a():
        with _in_task("A"):
            with _write_guard() as grant:
                assert grant.refused is None and grant.locked
                events.append("A enter")
                a_inside.set()
                assert release_a.wait(10)
                events.append("A exit")

    def writer_b():
        assert a_inside.wait(10), "A never entered"
        with _in_task("B"):
            with _write_guard() as grant:
                assert grant.refused is None and grant.locked
                events.append("B enter")
                b_entered.set()
                events.append("B exit")

    ta, _ = _start(writer_a)
    tb, _ = _start(writer_b)

    # B must still be queued while A holds the lock.
    assert not b_entered.wait(0.4), "B mutated while A held the write lock"
    release_a.set()
    assert b_entered.wait(10), "B never acquired the lock after A released"
    ta.join(15)
    tb.join(15)
    assert not ta.is_alive() and not tb.is_alive()
    assert events == ["A enter", "A exit", "B enter", "B exit"]


def test_different_workspaces_do_not_block_each_other(allow_write, tmp_path):
    """A writer in project A must never delay a writer in project B.

    ``require_workspace`` refuses to run work belonging to another project in
    this process, so the foreign workspace is injected at the coordination layer
    (where the lock key is derived) rather than through ``task_context``.
    """
    other = tmp_path / "other"
    other.mkdir()
    a_inside = threading.Event()
    release_a = threading.Event()
    done: dict[str, str] = {}

    def writer_a():
        with _write_guard():
            a_inside.set()
            assert release_a.wait(10)
            done["A"] = "wrote"

    def writer_b():
        assert a_inside.wait(10)
        # Pretend this execution belongs to a different project.
        with _foreign_workspace(other):
            with _write_guard() as grant:
                # Would defer if the lock were keyed globally instead of by
                # workspace.
                assert grant.locked, grant.deferred
                done["B"] = "wrote"
                release_a.set()

    ta, _ = _start(writer_a)
    tb, _ = _start(writer_b)
    ta.join(20)
    tb.join(20)
    assert not ta.is_alive() and not tb.is_alive(), f"only finished {done}"
    assert done == {"A": "wrote", "B": "wrote"}


@contextmanager
def _foreign_workspace(path):
    """Make the coordination layer resolve the current workspace to *path*."""
    with patch(
        "terminus.coordination._current_workspace", lambda: str(path.resolve())
    ):
        yield


def test_lock_is_keyed_by_workspace_not_task(allow_write):
    """Same project, different task labels: still exactly one writer."""
    order: list[str] = []
    first_inside = threading.Event()
    release = threading.Event()

    def writer_one():
        with _in_task("task-1"):
            with _write_guard() as grant:
                assert grant.locked
                order.append("one")
                first_inside.set()
                assert release.wait(10)

    def writer_two():
        assert first_inside.wait(10)
        with _in_task("task-2"):
            with _write_guard() as grant:
                assert grant.locked
                order.append("two")
                release.set()

    ta, _ = _start(writer_one)
    tb, _ = _start(writer_two)
    time.sleep(0.4)
    assert order == ["one"], "second task entered before the first released"
    release.set()
    ta.join(15)
    tb.join(15)
    assert not ta.is_alive() and not tb.is_alive()
    assert order == ["one", "two"]


# ---------------------------------------------------------------------------
# Read-only work is not serialised
# ---------------------------------------------------------------------------


def test_read_only_never_takes_the_lock(allow_write):
    with _write_guard() as held:
        assert held.locked
        with project_write_guard(Operation.READ, target="f.txt") as reader:
            assert reader.refused is None
            assert reader.locked is False
            assert reader.deferred is None


def test_read_only_reads_overlap(allow_write):
    with project_write_guard(Operation.READ, target="a") as first:
        with project_write_guard(Operation.READ, target="b") as second:
            assert first.locked is False and second.locked is False


def test_a_read_never_queues_behind_a_busy_writer(allow_write, monkeypatch):
    """A slow writer must not delay a reader that arrives mid-write."""
    monkeypatch.setattr("terminus.coordination.WRITE_WAIT_SECONDS", 0.1)
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with _write_guard():
            inside.set()
            release.wait(10)

    ta, _ = _start(holder)
    assert inside.wait(10)
    try:
        started = time.monotonic()
        with project_write_guard(Operation.READ, target="f.txt") as reader:
            elapsed = time.monotonic() - started
            assert reader.refused is None
            assert reader.locked is False
        assert elapsed < 0.2, f"a read waited {elapsed:.2f}s behind a writer"
    finally:
        release.set()
        ta.join(15)


# ---------------------------------------------------------------------------
# Permission interaction
# ---------------------------------------------------------------------------


def test_refused_mutation_never_acquires_the_lock():
    """A refusal neither takes nor retains the writer lock."""
    set_permission_policy(READ_ONLY)
    try:
        for operation in (Operation.WRITE, Operation.DELETE):
            with project_write_guard(operation, target="f.txt") as grant:
                assert grant.refused is not None
                assert grant.locked is False
                assert grant.deferred is None
        assert not is_writing(str(project_root()))
    finally:
        set_permission_policy(WRITE_OK)


def test_refusal_precedes_coordination_wait():
    """A denied operation must not block on the lock before being refused.

    Ordering matters: coordinating first would let a refused mutation stall a
    legitimate writer for the length of its own pointless wait.
    """
    set_permission_policy(READ_ONLY)
    try:
        started = time.monotonic()
        with project_write_guard(Operation.DELETE, target="f.txt") as grant:
            assert grant.refused is not None
        assert time.monotonic() - started < 0.5
        assert WRITE_WAIT_SECONDS > 0
    finally:
        set_permission_policy(WRITE_OK)


def test_refusal_does_not_block_a_later_writer(allow_write):
    """A refusal under a narrow policy must leave the lock free afterwards."""
    set_permission_policy(READ_ONLY)
    try:
        with project_write_guard(Operation.WRITE, target="f.txt") as refused:
            assert refused.refused is not None
            assert refused.locked is False
        assert not is_writing(str(project_root()))
    finally:
        set_permission_policy(WRITE_OK)

    with _in_task():
        with _write_guard() as granted:
            assert granted.refused is None
            assert granted.locked


def test_busy_writer_declines_rather_than_overlapping(allow_write, monkeypatch):
    """Out of patience, the loser is told to retry - it never proceeds anyway."""
    monkeypatch.setattr("terminus.coordination.WRITE_WAIT_SECONDS", 0.2)
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with _write_guard() as grant:
            assert grant.locked
            inside.set()
            release.wait(10)

    ta, _ = _start(holder)
    assert inside.wait(10)
    try:
        with _write_guard() as second:
            assert second.refused is None
            assert second.locked is False
            assert "Deferred" in (second.deferred or "")
    finally:
        release.set()
        ta.join(15)


# ---------------------------------------------------------------------------
# Release on failure, cancellation, retry
# ---------------------------------------------------------------------------


def test_lock_released_when_mutation_raises(allow_write):
    with pytest.raises(RuntimeError):
        with _write_guard():
            assert is_writing(str(project_root()))
            raise RuntimeError("tool blew up")

    assert not is_writing(str(project_root()))
    with _write_guard() as again:
        assert again.locked, "a failed mutation left the project locked"


def test_cancelled_waiter_does_not_wedge_the_project(allow_write, monkeypatch):
    """A waiter that gives up must leave no trace.

    The timeout path is the runtime's own escape hatch: it declines, releases
    nothing (it never held anything) and leaves the lock with its real holder.
    Once that holder finishes, the next writer is served immediately.
    """
    monkeypatch.setattr("terminus.coordination.WRITE_WAIT_SECONDS", 0.3)
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with _write_guard():
            inside.set()
            release.wait(20)

    def impatient():
        with _write_guard() as grant:
            # Timed out: told to retry, and did not mutate.
            assert grant.locked is False
            assert "Deferred" in (grant.deferred or "")

    ta, _ = _start(holder)
    assert inside.wait(10)

    tb, _ = _start(impatient)
    tb.join(15)
    assert not tb.is_alive()

    # The lock still belongs to the holder, not to the abandoned waiter.
    release.set()
    ta.join(15)
    assert not ta.is_alive()
    assert not is_writing(str(project_root()))

    with _write_guard() as afterwards:
        assert afterwards.locked, "a timed-out waiter left the project locked"


def test_retry_can_reacquire_the_lock(allow_write):
    """A retried attempt mutates exactly like any other attempt."""
    for attempt in (1, 2, 3):
        with _in_task(f"attempt-{attempt}"):
            with _write_guard() as grant:
                assert grant.locked, f"attempt {attempt} could not re-acquire"
        assert not is_writing(str(project_root()))


def test_locks_are_reused_not_leaked(allow_write):
    """One lock per workspace, created once, not one per task or mutation."""
    before = tracked_workspaces()
    for _ in range(50):
        with _write_guard():
            pass
    assert tracked_workspaces() == before
    assert WRITE_WAIT_SECONDS > 0


# ---------------------------------------------------------------------------
# Tools hold the lock across the whole mutation
# ---------------------------------------------------------------------------


def test_write_file_holds_lock_across_the_write(allow_write, monkeypatch, tmp_path):
    """A second writer cannot land inside another writer's critical section."""
    from terminus.tools import filesystem_tools as fs

    inside = threading.Event()
    release = threading.Event()
    overlaps = []
    real_replace = os.replace

    def watched_replace(src, dst):
        if inside.is_set() and not release.is_set():
            overlaps.append(dst)
        inside.set()
        release.wait(10)
        return real_replace(src, dst)

    monkeypatch.setattr(fs.os, "replace", watched_replace)
    results: dict[str, str] = {}

    def write(name: str):
        with _in_task(name):
            results[name] = fs.write_file.invoke(
                {"file_path": str(tmp_path / f"{name}.txt"), "content": "hi"}
            )

    ta, _ = _start(write, "a")
    assert inside.wait(10)
    tb, _ = _start(write, "b")
    time.sleep(0.4)
    release.set()
    ta.join(15)
    tb.join(15)

    assert not ta.is_alive() and not tb.is_alive(), f"only wrote {sorted(results)}"
    assert not overlaps, f"overlapping writes: {overlaps}"
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
    assert "written successfully" in results["a"]
    assert "written successfully" in results["b"]


def test_append_file_is_authorised(allow_write, tmp_path):
    """Regression: append was a /plan-only tool that bypassed policy entirely."""
    from terminus.tools import filesystem_tools as fs

    target = tmp_path / "log.txt"
    target.write_text("a", encoding="utf-8")

    set_permission_policy(READ_ONLY)
    with _in_task():
        assert "Refused" in fs.append_file.invoke(
            {"file_path": str(target), "content": "b"}
        )
    assert target.read_text(encoding="utf-8") == "a", "refused append wrote anyway"

    set_permission_policy(WRITE_OK)
    with _in_task():
        assert "appended successfully" in fs.append_file.invoke(
            {"file_path": str(target), "content": "b"}
        )
    assert target.read_text(encoding="utf-8") == "ab"


def test_delete_file_is_authorised_as_destructive(allow_write, tmp_path):
    """Regression: delete was unauthorised, and is DESTRUCTIVE when governed."""
    from terminus.tools import filesystem_tools as fs

    target = tmp_path / "doomed.txt"
    target.write_text("x", encoding="utf-8")

    set_permission_policy(READ_ONLY)
    with _in_task():
        assert "Refused" in fs.delete_file.invoke({"file_path": str(target)})
    assert target.exists(), "a refused delete removed the file"

    # WRITE_OK still carries no DESTRUCTIVE approval and has no approver.
    set_permission_policy(WRITE_OK)
    with _in_task():
        assert "Refused" in fs.delete_file.invoke({"file_path": str(target)})
    assert target.exists(), "write-only policy must not permit delete"

    # Destructive still needs an approver: WRITE_OK has none, so it is refused.
    set_permission_policy(
        PermissionPolicy(
            auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
            approver=lambda *_: True,
            deny_levels=(),
        )
    )
    with _in_task():
        assert "deleted successfully" in fs.delete_file.invoke(
            {"file_path": str(target)}
        )
    assert not target.exists()


def test_read_only_tools_still_work_under_a_busy_writer(allow_write, tmp_path):
    from terminus.tools import filesystem_tools as fs

    target = tmp_path / "data.txt"
    target.write_text("content", encoding="utf-8")
    inside = threading.Event()
    release = threading.Event()

    def holder():
        with _write_guard():
            inside.set()
            release.wait(10)

    ta, _ = _start(holder)
    assert inside.wait(10)
    try:
        assert "content" in fs.read_file.func(file_path=str(target))
    finally:
        release.set()
        ta.join(15)


# ---------------------------------------------------------------------------
# Orchestrator scheduling
# ---------------------------------------------------------------------------


class _RecordingStore:
    """Store stub that records overlap. Holds no SQL and no task semantics."""

    def __init__(self, batches: list[list[dict]] | None = None) -> None:
        self.ready_batches = list(batches or [])
        self.total_tasks = sum(len(batch) for batch in self.ready_batches)
        self.started: list[str] = []
        self.completed: list[str] = []
        self.in_flight = 0
        self.peak = 0
        self.barrier: threading.Barrier | None = None
        self.cycle = 0
        self.batches_handed_out = 0

    def update_project_status(self, *_a, **_k):
        pass

    def finalize_project_status(self, *_a, **_k):
        pass

    def get_progress(self, _project_id):
        done = len(self.completed)
        total = self.total_tasks
        return {
            "pending": max(0, total - done - self.in_flight),
            "in_progress": self.in_flight,
            "completed": done,
            "failed": 0,
        }

    def get_ready_tasks(self, _project_id):
        self.batches_handed_out += 1
        return self.ready_batches.pop(0) if self.ready_batches else []

    def get_project_workspace(self, _project_id):
        return None

    def get_dep_results(self, _project_id, _dep_ids):
        return {}

    def claim_task(self, _project_id, task_id):
        self.cycle += 1
        self.started.append(task_id)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        return self.cycle

    def complete_task(self, _project_id, task_id, _output):
        self.in_flight -= 1
        self.completed.append(task_id)

    def fail_task(self, *_a, **_k):
        self.in_flight -= 1
        return "FAILED"

    def get_blocked_by_failed(self, _project_id):
        return []

    def get_all_tasks(self, _project_id):
        return []


def _task(task_id: str, depends_on: list[str] | None = None) -> dict:
    return {
        "id": task_id,
        "project_id": "p1",
        "description": f"task {task_id}",
        "depends_on": json.dumps(depends_on or []),
        "error": None,
    }


async def _fake_execute(task, *, attempt=1, **_kwargs):
    """Stand-in worker.

    The body runs in a *thread* and waits on a store-level barrier, which is what
    a real tool call does (``run_in_executor``), so this exercises the same
    path. A serial scheduler cannot satisfy a barrier of 2, so it fails loudly
    rather than quietly appearing to pass.
    """
    from terminus.tasks.worker import TaskResult

    store = _fake_execute.store
    barrier = store.barrier

    def work():
        if barrier is not None:
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        return TaskResult.ok(output=f"done {task['id']}", attempt=attempt)

    return await asyncio.to_thread(work)


def test_max_concurrent_one_runs_serially(monkeypatch):
    """Default behaviour is unchanged: never two tasks in flight at once."""
    from terminus.tasks import orchestrator as orch_mod

    store = _RecordingStore([[_task("A")], [_task("B")]])
    # A barrier of 2 can never be satisfied serially, so if the scheduler ever
    # went parallel this stub would break loudly instead of silently passing.
    store.barrier = threading.Barrier(2, timeout=0.6)
    _fake_execute.store = store
    monkeypatch.setattr(orch_mod, "execute_task", _fake_execute)
    monkeypatch.setattr(orch_mod.asyncio, "sleep", _instant_sleep)

    _run(orch_mod.TaskOrchestrator(store, max_concurrent=1).run("p1"))

    assert store.peak == 1, f"expected serial execution, peak was {store.peak}"
    assert store.completed == ["A", "B"]


def test_max_concurrent_two_overlaps_independent_tasks(monkeypatch):
    """Two independent tasks genuinely run at the same time.

    The barrier only releases once both workers have arrived, so this cannot
    pass unless both are in flight simultaneously.
    """
    from terminus.tasks import orchestrator as orch_mod

    store = _RecordingStore([[_task("A"), _task("B")]])
    store.barrier = threading.Barrier(2, timeout=10)
    _fake_execute.store = store
    monkeypatch.setattr(orch_mod, "execute_task", _fake_execute)
    monkeypatch.setattr(orch_mod.asyncio, "sleep", _instant_sleep)

    _run(orch_mod.TaskOrchestrator(store, max_concurrent=2).run("p1"))

    assert store.peak == 2, f"expected overlap, peak was {store.peak}"
    assert sorted(store.completed) == ["A", "B"]


def test_dependent_task_waits_for_both_parents(monkeypatch):
    """A ┬ B ┴ C: C cannot start until A and B have COMPLETED."""
    from terminus.tasks import orchestrator as orch_mod

    store = _RecordingStore([[_task("A"), _task("B")], [_task("C", ["A", "B"])]])
    store.barrier = threading.Barrier(2, timeout=10)
    _fake_execute.store = store
    order: list[str] = []
    order_lock = threading.Lock()

    async def recording(task, **kwargs):
        with order_lock:
            order.append(f"start:{task['id']}")
        result = await _fake_execute(task, **kwargs)
        with order_lock:
            order.append(f"end:{task['id']}")
        return result

    monkeypatch.setattr(orch_mod, "execute_task", recording)
    monkeypatch.setattr(orch_mod.asyncio, "sleep", _instant_sleep)

    _run(orch_mod.TaskOrchestrator(store, max_concurrent=2).run("p1"))

    assert store.peak == 2, f"A and B did not overlap: {order}"
    # The stub only re-hands C in the second round, i.e. after the first batch
    # was fully awaited, so this asserts the real scheduler invariant.
    assert store.batches_handed_out == 2, store.batches_handed_out
    start_c = order.index("start:C")
    for parent in ("A", "B"):
        assert order.index(f"end:{parent}") < start_c, (
            f"C started before {parent} completed: {order}"
        )


async def _instant_sleep(*_a, **_k):
    return None


def test_orchestrator_cancellation_does_not_strand_the_lock(allow_write):
    """Cancelling a running task releases coordination.

    The scheduler awaits N workers with gather(); if a task is cancelled
    mid-flight, the write lock its tools were using must not stay held.
    """
    from terminus.tasks import orchestrator as orch_mod

    store = _RecordingStore([[_task("A"), _task("B")]])

    async def cancellable(task, **_kwargs):
        # The guard's finally must run even though this coroutine never returns.
        with _write_guard():
            await asyncio.sleep(3600)

    async def scenario():
        orch_mod.execute_task = cancellable
        try:
            orch = orch_mod.TaskOrchestrator(store, max_concurrent=2)
            running = asyncio.ensure_future(orch.run("p1"))
            for _ in range(100):
                await asyncio.sleep(0.02)
                if is_writing(str(project_root())):
                    break
            assert is_writing(str(project_root())), "no worker reached the lock"
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        finally:
            orch_mod.execute_task = _fake_execute

    _run(scenario())

    assert not is_writing(str(project_root())), (
        "cancelling the scheduler left the project write lock held"
    )


def _run(coro):
    return asyncio.run(coro)
