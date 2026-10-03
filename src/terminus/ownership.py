"""One active Terminus orchestrator per project.

Invariant
---------
At most one Terminus process may orchestrate a given project at a time. This is
*orchestrator ownership*, one level above worker concurrency: the owning process
may still run ``max_concurrent = 1..4`` workers, and those workers still
serialise their writes through ``coordination.project_write_guard``.

Why it exists
-------------
``TaskStore.recover_interrupted_tasks`` resets ``in_progress`` tasks to
``pending`` on the assumption that nobody else is working on them. That
assumption only holds if a single process is orchestrating the project. Before
this module two Terminus processes in the same directory could satisfy it at
once: ``/plan continue`` in process B would reset the very task process A was
actively running, and B would then start a duplicate worker for it.

How liveness is determined
--------------------------
By an **operating-system file lock**, not by a heartbeat, an age threshold, or a
recorded PID. The kernel releases an advisory file lock when the holding process
exits *however* it exits - clean shutdown, ``kill``, or a crash with no cleanup -
so "is the previous owner still alive?" is answered exactly rather than guessed.

That matters because every cheaper option is a guess:

* a recorded PID is ambiguous once the OS recycles it;
* an age threshold needs a heartbeat clock, and still cannot tell a slow worker
  from a dead one;
* an SQLite lock would tie the lock's lifetime to a transaction, which is the
  wrong lifetime - ownership spans a whole orchestration, not one statement.

The cost is that the guarantee is only as good as the local filesystem. Two
processes on the same machine are mutually excluded. Advisory locks are
unreliable on some network filesystems (NFS, SMB), so a project directory on
such a share is not covered; that limitation is stated rather than papered over.

Persistence
-----------
None is required and none is added. The lock file *is* the record: it is created
next to the task database, named for the project it guards, and holds the owner's
PID, hostname and start time purely so a refusal can say who holds it. Because
the kernel owns the truth, a stored record can never disagree with reality, so
there is no schema change and no migration to get wrong. Adding ownership columns
to ``TaskStore`` would have created a second source of truth that could disagree
with the lock, and then something would have to reconcile them.

Non-goals
---------
Not a distributed lock. It does not coordinate across machines, and it does not
protect a workspace shared by two *different* projects - that remains the
documented limit of the process-local writer lock.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "OwnershipConflict",
    "OwnerInfo",
    "ProcessOwnership",
    "ProjectOwnership",
    "current_ownership",
    "lock_dir_for",
]

# Errno values meaning "somebody else holds the lock", as opposed to a real
# filesystem fault. A genuine I/O error must never be reported as contention,
# because that would disguise a broken database as a busy project.
_CONTENTION_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, name, None)
        for name in ("EACCES", "EAGAIN", "EDEADLK", "EWOULDBLOCK")
    )
    if value is not None
)


class OwnershipConflict(RuntimeError):
    """Another live Terminus process already owns this project.

    Carries only non-secret diagnostics about the holder: PID, host, and when it
    took ownership. Nothing about the holder's environment, arguments or files
    is recorded, so a refusal cannot leak anything sensitive.
    """

    def __init__(self, project_id: str, holder: "OwnerInfo | None") -> None:
        self.project_id = project_id
        self.holder = holder
        if holder is None:
            detail = "held by another Terminus process"
        else:
            age = max(0, int(time.time() - holder.since))
            detail = f"held by Terminus pid {holder.pid} on {holder.host} for {age}s"
        super().__init__(
            f"Project {project_id} is already being orchestrated: {detail}. "
            f"Nothing was changed. Let that process finish, or stop it, before "
            f"continuing this project from here."
        )


@dataclass(frozen=True)
class OwnerInfo:
    """Who holds a project, for diagnostics only.

    Never consulted to decide whether the owner is still alive - the file lock
    answers that - so it cannot become the reason two owners coexist.
    """

    project_id: str
    pid: int
    host: str
    since: float


# The lock and the record live in two files on purpose.
#
# On Windows a byte-range lock is enforced by the kernel against *reads* as well
# as writes, so a single file cannot both be locked and have its contents read by
# the process that needs to explain the refusal. Storing the owner record beside
# the lock rather than inside it keeps the record readable at all times, and
# keeps the locked file to one meaningless byte with no layout to get wrong.
_LOCK_SUFFIX = ".lock"
_RECORD_SUFFIX = ".owner.json"


if os.name == "nt":
    import msvcrt

    def _lock_exclusive(handle) -> None:
        """Take a non-blocking exclusive lock on the file. Raises if held."""
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

else:  # pragma: no cover - this host is Windows
    import fcntl

    def _lock_exclusive(handle) -> None:
        """Take a non-blocking exclusive advisory lock. Raises if held."""
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class ProjectOwnership:
    """Exclusive ownership of one project, backed by a lock file.

    The handle stays open while ownership is held; the OS releases the lock when
    the handle closes or the process dies, whichever happens first.
    """

    def __init__(self, lock_dir: Path, project_id: str) -> None:
        stem = f"owner-{project_id}"
        self.lock_path = Path(lock_dir) / f"{stem}{_LOCK_SUFFIX}"
        self.record_path = Path(lock_dir) / f"{stem}{_RECORD_SUFFIX}"
        self.project_id = project_id
        self._handle = None
        self._owner: OwnerInfo | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    @property
    def info(self) -> OwnerInfo:
        if self._owner is None:  # pragma: no cover - callers check `held`
            raise RuntimeError(f"project {self.project_id} is not owned")
        return self._owner

    def acquire(self) -> OwnerInfo:
        """Take ownership, or raise :class:`OwnershipConflict`.

        Order matters: the lock is taken *before* the record is written, so a
        process that loses the race is reading a file that nobody has locked.
        """
        if self.held:
            return self.info

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # "a+b" creates the file and never truncates a lock file another process
        # still owns.
        handle = open(self.lock_path, "a+b")
        try:
            _lock_exclusive(handle)
        except OSError as exc:
            holder = _read_owner(self.record_path)
            _close_quietly(handle)
            if exc.errno in _CONTENTION_ERRNOS:
                raise OwnershipConflict(self.project_id, holder) from None
            raise

        owner = OwnerInfo(
            project_id=self.project_id,
            pid=os.getpid(),
            host=socket.gethostname(),
            since=time.time(),
        )
        try:
            _write_owner(self.record_path, owner)
        except OSError:
            # Never hold a lock whose owner nobody can see; fail closed.
            _close_quietly(handle)
            raise
        self._handle = handle
        self._owner = owner
        logger.info("Acquired orchestrator ownership of project %s", self.project_id)
        return owner

    def release(self) -> None:
        """Give ownership up. Safe to call when not held.

        Closing the handle is what actually releases the lock, so this never
        claims a release that did not happen. The record is left behind on
        purpose: it is harmless, it is not the lock, and deleting it would race
        with a process that is mid-acquisition.
        """
        handle, self._handle = self._handle, None
        self._owner = None
        if handle is None:
            return
        try:
            handle.close()
        except OSError as exc:  # pragma: no cover - close() rarely fails
            logger.error(
                "Failed to release ownership of project %s: %s", self.project_id, exc
            )
            raise
        logger.info("Released orchestrator ownership of project %s", self.project_id)


def _write_owner(record_path: Path, info: OwnerInfo) -> None:
    """Write the owner record. Diagnostics only; never decides liveness.

    Written atomically via a sibling temp file so a reader can never observe a
    half-written record and report a truncated PID.
    """
    payload = json.dumps(asdict(info)).encode("utf-8")
    temp = record_path.with_name(record_path.name + ".tmp")
    with open(temp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, record_path)


def _close_quietly(handle) -> None:
    try:
        handle.close()
    except OSError:  # pragma: no cover - close() rarely fails
        pass


def _read_owner(record_path: Path) -> OwnerInfo | None:
    """Best-effort read of the recorded owner; a damaged file is not fatal."""
    try:
        raw = record_path.read_bytes().strip()
        if not raw:
            return None
        data = json.loads(raw.decode("utf-8"))
        return OwnerInfo(
            project_id=str(data.get("project_id", "")),
            pid=int(data.get("pid", 0)),
            host=str(data.get("host", "")),
            since=float(data.get("since", 0.0)),
        )
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None


class ProcessOwnership:
    """The projects *this* process is orchestrating.

    Reference-counted, so nested orchestration paths in one process cannot
    deadlock against each other or release early. Holds only bookkeeping about
    locks this process took itself: it grants no authority it does not hold, and
    it is not permission state.
    """

    def __init__(self) -> None:
        self._locks: dict[str, ProjectOwnership] = {}
        self._depth: dict[str, int] = {}

    def acquire(self, lock_dir: Path, project_id: str) -> OwnerInfo:
        """Own *project_id*, or raise :class:`OwnershipConflict`.

        Taking the same project twice in one process succeeds and counts, so a
        matching release is required before the lock is actually dropped.
        """
        held = self._locks.get(project_id)
        if held is not None:
            self._depth[project_id] += 1
            return held.info

        lock = ProjectOwnership(Path(lock_dir), project_id)
        info = lock.acquire()
        self._locks[project_id] = lock
        self._depth[project_id] = 1
        return info

    def release(self, project_id: str) -> None:
        """Release one level of ownership. No-op if this process never held it."""
        if project_id not in self._locks:
            return
        if self._depth[project_id] > 1:
            self._depth[project_id] -= 1
            return
        lock = self._locks.pop(project_id)
        self._depth.pop(project_id, None)
        lock.release()

    def release_all(self) -> list[str]:
        """Release everything this process owns. Used on shutdown.

        A failure to release one project must not strand the others, so each is
        attempted and any error is reported rather than raised.
        """
        released: list[str] = []
        for project_id in list(self._locks):
            lock = self._locks.pop(project_id)
            self._depth.pop(project_id, None)
            try:
                lock.release()
                released.append(project_id)
            except OSError as exc:  # pragma: no cover - close() rarely fails
                logger.error(
                    "Could not release ownership of project %s: %s", project_id, exc
                )
        return released

    def holds(self, project_id: str) -> bool:
        return project_id in self._locks

    def owned(self) -> list[str]:
        return list(self._locks)


_current = ProcessOwnership()


def current_ownership() -> ProcessOwnership:
    """The process-wide ownership registry, driven by the CLI lifecycle."""
    return _current


def lock_dir_for(db_path) -> Path:
    """Where a project's ownership lock lives.

    Derived from the task database path so ownership is scoped to the same
    directory as the task state it guards, and travels with the project rather
    than with any one process.
    """
    return Path(db_path).parent / "owners"
