"""SQLite persistence for projects and the tasks inside them.

This is the single owner of task state. The scheduler claims work here, the
worker reports outcomes here, and the CLI reads status from here, so there is one
place where a task's status, attempt counts and result text are decided rather
than one per caller.

Two invariants worth knowing before changing anything here:

* **Claiming is atomic.** ``claim_task`` performs the pending -> in_progress
  transition and increments ``total_attempts`` in a single statement, so the
  attempt number the scheduler reports and the attempt the worker is running can
  never disagree, and two orchestrators cannot both start the same task.
* **Two attempt counters, on purpose.** ``retry_count`` is spent attempts in the
  current recovery cycle and is what the automatic retry budget is measured
  against. ``total_attempts`` counts every attempt ever started and is never
  reset, so a task's real history survives any number of ``/plan continue``
  cycles. Conflating them would make the automatic budget either unbounded or
  permanently exhausted.

The schema is created on construction and migrated in place, so an existing
``.terminus/tasks/tasks.db`` from an older Terminus keeps working.
"""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

from terminus.observability.logging import get_logger
from terminus.tasks.errors import _NON_RETRYABLE_PREFIXES

logger = get_logger(__name__)

class TaskStatus(str, Enum):
    """All valid states of a task.
    PENDING: task is waiting to be started (also the retry state).
    IN_PROGRESS: task is currently being worked on.
    COMPLETED: task has been completed.
    FAILED: task has permanently failed (retry budget exhausted in this run).
    BLOCKED: (reserved) task is blocked by another task and cannot be started.
    SKIPPED: (reserved) task has been skipped and will not be worked on.

    Transitions:
      PENDING -> IN_PROGRESS  (claim_task)
      IN_PROGRESS -> COMPLETED (complete_task)
      IN_PROGRESS -> PENDING   (fail_task, when retry budget remains; recover_interrupted_tasks)
      IN_PROGRESS -> FAILED    (fail_task, when retry budget exhausted)
      FAILED -> PENDING        (reset_failed_tasks_for_recovery, i.e. manual /plan continue)
      PENDING -> BLOCKED       (never persisted; blocking is computed dynamically)
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class TaskType(str,Enum):
    DESIGN = "design"
    IMPLEMENT = "implement"
    TEST = "test"
    REVIEW = "review"
    INTEGRATE = "integrate"
    CONFIGURE = "configure"


class ProjectStatus(str, Enum):
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


def _same_workspace(a: str, b: str) -> bool:
    """Compare two workspace paths, tolerating symlinks and case on Windows."""
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except OSError:
        return os.path.normcase(a) == os.path.normcase(b)


MAX_PERSISTED_RESULT_CHARS = 8_000
"""Ceiling on what one task row may store in ``result`` or ``error``.

A worker's answer is model output, so its length is not something the runtime
controls. 8k matches the other text budgets in Terminus (a shell stream, one
deliverable file) and is far more than a task summary needs, while keeping a
chatty or looping worker from growing the task database without limit.

Downstream tasks read these strings through get_dep_results, so bounding at the
persistence boundary bounds dependency inputs too.
"""

MAX_RECOVERY_CYCLES = 3
"""How many times ``/plan continue`` may re-open a project's failed tasks.

Each cycle grants a fresh per-cycle attempt budget, so without a cap the total
attempts for one task would be unbounded. Three cycles is deliberately explicit
and small: it is enough to recover from a genuine environment problem (a bad
credential, a missing dependency) without letting a task that cannot succeed
retry forever. A module constant rather than configuration, matching the other
runtime bounds in Terminus.
"""


def bounded_result(text: str) -> str:
    """Bound *text* to MAX_PERSISTED_RESULT_CHARS, keeping head and tail.

    Truncation is explicit and never changes meaning: the marker states how much
    was dropped. The head keeps the summary of what was done, the tail keeps the
    closing remarks and quoted file contents, which is where a worker's most
    useful detail usually sits.

    This is the single owner of task-result truncation. It must never turn a
    successful task into a failed one, so it is a pure string operation and is
    applied by the store, not by the worker.
    """
    if text is None:
        return ""
    text = str(text)
    if len(text) <= MAX_PERSISTED_RESULT_CHARS:
        return text
    head = (MAX_PERSISTED_RESULT_CHARS * 2) // 3
    tail = MAX_PERSISTED_RESULT_CHARS - head
    dropped = len(text) - MAX_PERSISTED_RESULT_CHARS
    return (
        f"{text[:head]}\n"
        f"... [truncated: {dropped} of {len(text)} characters omitted; "
        f"showing the first {head} and last {tail}]\n"
        f"{text[-tail:]}"
    )

def _strip_retry_prefix(error: str) -> str:
    """Remove the caller's retryability tag from a stored error message.

    ``_run_orchestration`` prefixes the error with ``[retryable]`` or
    ``[non-retryable]`` so the log line reads clearly. That prefix is redundant in
    the stored column, which records the outcome in its own ``[failed]`` /
    ``[retry-exhausted]`` tag, so keeping both left the column reading
    ``[failed] [non-retryable] ...``.
    """
    text = error or ""
    for prefix in _NON_RETRYABLE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


class TaskStore:
    def __init__(self, db_path:str="tasks.db"):
        self.db_path = db_path
        # The store owns its own location: a caller that only wants to *read*
        # state (cli.show_task_status) must not have to create the directory
        # first, or it crashes with "unable to open database file" on a
        # project that has never run /plan.
        parent = Path(db_path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def conn(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self):
        with self.conn() as connection:
            connection.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    goal TEXT,
                    plan_json TEXT,
                    status TEXT,
                    created_at REAL DEFAULT (unixepoch()),
                    workspace TEXT,
                    recovery_cycles INTEGER DEFAULT 0
                )
            ''')
            connection.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    task_type TEXT,
                    status TEXT DEFAULT 'pending',
                    depends_on TEXT DEFAULT '[]',
                    output_files TEXT DEFAULT '[]',
                    acceptance_criteria TEXT DEFAULT '[]',
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    total_attempts INTEGER DEFAULT 0,
                    execution_order INTEGER DEFAULT 0,
                    started_at REAL,
                    created_at REAL DEFAULT (unixepoch()),
                    completed_at REAL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                )
            ''')
            self._migrate_task_primary_key(connection)
            self._migrate_project_workspace(connection)
            self._add_column_if_missing(connection, "projects", "recovery_cycles",
                                        "INTEGER DEFAULT 0")
            self._add_column_if_missing(connection, "tasks", "total_attempts",
                                        "INTEGER DEFAULT 0")

    def _add_column_if_missing(self, connection: sqlite3.Connection, table: str,
                               column: str, definition: str) -> None:
        """Add one column to an existing database, if it is not already there."""
        existing = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column in existing:
            return
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        logger.info("Added %s.%s to existing task database", table, column)

    def _migrate_project_workspace(self, connection: sqlite3.Connection) -> None:
        """Add projects.workspace to databases created before workspace identity.

        A project records the directory it was planned for. Without it, a project
        whose .terminus directory has been copied or moved is still reported as
        resumable from anywhere, and its workers then edit the wrong tree.
        Existing rows are left NULL, which is treated as "unknown" and allowed
        once so pre-existing projects keep working.
        """
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(projects)")
        }
        if "workspace" in columns:
            return
        connection.execute("ALTER TABLE projects ADD COLUMN workspace TEXT")
        logger.info("Added projects.workspace column to existing task database")

    def _migrate_task_primary_key(self, connection: sqlite3.Connection) -> None:
        """Migrate legacy single-column task PK schema to composite (project_id, id)."""
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone()
        if not row or "PRIMARY KEY (project_id, id)" in row[0]:
            return

        connection.executescript(
            """
            ALTER TABLE tasks RENAME TO tasks_legacy;
            CREATE TABLE tasks (
                id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                title TEXT,
                description TEXT,
                task_type TEXT,
                status TEXT DEFAULT 'pending',
                depends_on TEXT DEFAULT '[]',
                output_files TEXT DEFAULT '[]',
                acceptance_criteria TEXT DEFAULT '[]',
                result TEXT,
                error TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                execution_order INTEGER DEFAULT 0,
                started_at REAL,
                created_at REAL DEFAULT (unixepoch()),
                completed_at REAL,
                PRIMARY KEY (project_id, id),
                FOREIGN KEY(project_id) REFERENCES projects(id)
            );
            INSERT INTO tasks SELECT * FROM tasks_legacy;
            DROP TABLE tasks_legacy;
            """
        )

    def get_ready_tasks(self, project_id: str) -> list:
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, title, description, task_type,
                    depends_on, output_files, acceptance_criteria, execution_order
                FROM tasks
                WHERE project_id=? AND status='pending'
                ORDER BY execution_order
                """,
                (project_id,)
            ).fetchall()

        tasks = [
            dict(row)
            for row in rows
        ]

        return [
            task
            for task in tasks
            if self._all_deep_done(
                project_id,
                json.loads(task.get("depends_on", "[]"))
            )
        ]

    def _all_deep_done(self, project_id:str, dep_ids: list[str])->bool:
        """
        Return true if all dependencies are completed or skipped.
        """
        if not dep_ids:
            return True
        with self.conn() as conn:
            rows = conn.execute(
                "SELECT status FROM tasks WHERE project_id=? AND id IN ({})".format(",".join(["?"]*len(dep_ids))),
                (project_id, *dep_ids)
            ).fetchall()
            return len(rows) == len(dep_ids) and all(
                row["status"] in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
                for row in rows
            )

    def get_all_tasks(self, project_id: str) -> list[dict]:
        """Every task in *project_id*, with its full state.

        Named publicly because the orchestrator reports on failed tasks to the
        user and was reaching past the store's interface into ``get_all_tasks``
        to do it. One accessor, one owner of the query.
        """
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, title, description, task_type,
                status, depends_on, output_files, acceptance_criteria,
                result, error, retry_count, max_retries, total_attempts,
                execution_order, started_at, created_at, completed_at
                FROM tasks
                WHERE project_id=?
                """,
                (project_id,)
            ).fetchall()

        return [dict(row) for row in rows]

    def create_project(self, goal:str, plan, workspace: str | None = None)->str:
        """
        Persist an approved ExecutionPlan as a project + task rows.
        Returns the new project_id (UUID)

        ``workspace`` records the project directory this plan belongs to, so a
        later run can refuse to execute it somewhere else. Defaults to the
        current project root.
        """
        if workspace is None:
            from terminus.workspace import project_root
            workspace = str(project_root())
        project_id = str(uuid.uuid4())
        created_at = time.time()
        with self.conn() as conn:
            conn.execute(
                "INSERT INTO projects(id, name, goal, plan_json, status, created_at, workspace) VALUES(?,?,?,?,?,?,?)",
                (
                    project_id,
                    plan.project_name,
                    goal,
                    plan.model_dump_json(),
                    ProjectStatus.APPROVED.value,
                    created_at,
                    workspace,
                ),
            )
            _placeholder_terms = {"", "placeholder", "dummy", "none", "null", "tbd", "todo"}
            skipped_placeholders = 0
            for i, pt in enumerate(plan.tasks):
                title_lower = (pt.title or "").strip().lower()
                desc_lower = (pt.description or "").strip().lower()
                criteria = [c.strip().lower() for c in pt.acceptance_criteria or []]
                is_placeholder = (
                    title_lower in _placeholder_terms
                    and desc_lower in _placeholder_terms
                ) or (
                    title_lower in ("placeholder", "dummy")
                    and criteria == ["placeholder"]
                )
                if is_placeholder:
                    conn.execute(
                        """INSERT INTO tasks(id, project_id, title, description, task_type, depends_on, output_files, acceptance_criteria, execution_order, status, result, completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            pt.id, project_id, pt.title, pt.description,
                            pt.task_type.value, json.dumps(pt.depends_on),
                            json.dumps(pt.output_files), json.dumps(pt.acceptance_criteria),
                            i, TaskStatus.COMPLETED.value,
                            "skipped: placeholder task from the plan", time.time(),
                        ),
                    )
                    skipped_placeholders += 1
                    logger.info("Skipping placeholder task %s '%s'", pt.id, pt.title)
                    continue
                conn.execute(
                    """INSERT INTO tasks(id, project_id, title, description, task_type, depends_on, output_files, acceptance_criteria, execution_order) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (pt.id, project_id, pt.title, pt.description, pt.task_type.value, json.dumps(pt.depends_on), json.dumps(pt.output_files), json.dumps(pt.acceptance_criteria), i)
                )
            extra = {"project_id": project_id, "task_count": len(plan.tasks)}
            if skipped_placeholders:
                extra["skipped_placeholders"] = skipped_placeholders
        logger.info(
            "Successfully created project and tasks",
            extra=extra,
        )
        return project_id

    def get_resumable_project(self) -> str | None:
        """Return the newest project that still has recoverable work."""
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT p.id
                FROM projects p
                WHERE p.status IN (?, ?, ?, ?)
                  AND EXISTS (
                      SELECT 1 FROM tasks t
                      WHERE t.project_id = p.id
                        AND t.status IN (?, ?, ?)
                  )
                ORDER BY p.created_at DESC, p.rowid DESC
                LIMIT 1
                """,
                (
                    ProjectStatus.APPROVED.value,
                    ProjectStatus.IN_PROGRESS.value,
                    ProjectStatus.FAILED.value,
                    ProjectStatus.PAUSED.value,
                    TaskStatus.PENDING.value,
                    TaskStatus.IN_PROGRESS.value,
                    TaskStatus.FAILED.value,
                ),
            ).fetchone()
            if row:
                return row[0]
        return None

    def get_latest_project(self) -> str | None:
        """Return the most recently created project for status display."""
        with self.conn() as conn:
            row = conn.execute(
                "SELECT id FROM projects ORDER BY created_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
            if row:
                return row[0]
        return None

    def get_project(self, project_id: str) -> dict | None:
        """Return one project row, or None.

        Read-only accessor for the project-context read model. ``plan_json`` is
        returned raw and unparsed: TaskStore stores what the planner produced and
        does not get a say in how a caller chooses to read it.
        """
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT id, name, goal, plan_json, status, created_at, workspace,
                       recovery_cycles
                FROM projects WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_project_workspace(self, project_id: str) -> str | None:
        """The project directory a project was created for, or None if unknown."""
        with self.conn() as conn:
            row = conn.execute(
                "SELECT workspace FROM projects WHERE id=?", (project_id,)
            ).fetchone()
        return (row["workspace"] if row else None) or None

    def workspace_matches(self, project_id: str, workspace: str | None = None) -> bool:
        """Is this project safe to run in *workspace* (defaults to the current one)?

        True when the recorded workspace matches, or when nothing was recorded
        (a project created before workspace identity existed, or one that has
        never been given a directory). False means the project belongs to a
        different directory and must not be executed here.
        """
        recorded = self.get_project_workspace(project_id)
        if not recorded:
            return True
        if workspace is None:
            from terminus.workspace import project_root
            workspace = str(project_root())
        return _same_workspace(recorded, workspace)

    def update_project_status(self, project_id: str, status: str) -> None:
        with self.conn() as conn:
            conn.execute(
                "UPDATE projects SET status=? WHERE id=?",
                (status, project_id),
            )

    def finalize_project_status(self, project_id: str) -> None:
        progress = self.get_progress(project_id)
        failed = progress.get("failed", 0)
        pending = progress.get("pending", 0)
        in_progress = progress.get("in_progress", 0)
        completed = progress.get("completed", 0)

        if in_progress > 0:
            return
        if pending == 0 and failed == 0 and completed > 0:
            self.update_project_status(project_id, ProjectStatus.COMPLETED.value)
        elif failed > 0:
            self.update_project_status(project_id, ProjectStatus.FAILED.value)
        elif pending > 0:
            self.update_project_status(project_id, ProjectStatus.PAUSED.value)

    def recover_interrupted_tasks(self, project_id: str) -> int:
        """
        Reset in-progress tasks to pending on resume.

        ASSUMPTION: Terminus is intentionally single-process/single-orchestrator for
        a given project. A task that is still IN_PROGRESS when an orchestration run
        begins can therefore only belong to a previous crashed/interrupted process
        (Ctrl+C, terminal crash, Python exception, machine restart), so resetting it
        to PENDING is safe. Two Terminus processes must not claim the same pending
        task for the same project; `claim_task()` remains atomic so a second process
        can never double-execute a task.
        """
        with self.conn() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status=?, started_at=NULL
                WHERE project_id=? AND status=?
                """,
                (TaskStatus.PENDING.value, project_id, TaskStatus.IN_PROGRESS.value),
            )
            return cur.rowcount

    def reset_failed_tasks_for_recovery(self, project_id: str) -> int:
        """
        Manual recovery for `/plan continue`.

        Each call starts a new *recovery cycle*: the per-cycle attempt counter is
        reset so the task gets a fresh automatic budget, and the project's cycle
        counter goes up. Without a cap, an operator (or a script) could invoke
        /plan continue forever and a permanently failing task would retry
        without limit - which is the loop this method exists to prevent.

        So the number of cycles is bounded by MAX_RECOVERY_CYCLES. A project that
        has used them is not reset again; its failure stands and the user is told
        why. The cumulative ``total_attempts`` on each task is never reset, so
        the real history of a task is always visible.

        Returns the number of tasks reset (0 when the cap has been reached).
        """
        with self.conn() as conn:
            row = conn.execute(
                "SELECT recovery_cycles FROM projects WHERE id=?", (project_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Project {project_id} not found")
            cycles = row["recovery_cycles"] or 0
            if cycles >= MAX_RECOVERY_CYCLES:
                logger.warning(
                    "Project %s has used all %d recovery cycles; not resetting",
                    project_id, MAX_RECOVERY_CYCLES,
                )
                return 0

            conn.execute(
                "UPDATE projects SET recovery_cycles=? WHERE id=?",
                (cycles + 1, project_id),
            )
            cur = conn.execute(
                """
                UPDATE tasks
                SET status=?, retry_count=0, error=NULL, started_at=NULL
                WHERE project_id=?
                  AND status=?
                """,
                (TaskStatus.PENDING.value, project_id, TaskStatus.FAILED.value),
            )
            logger.info(
                "Recovery cycle %s/%s for project %s: reset %s failed task(s)",
                cycles + 1, MAX_RECOVERY_CYCLES, project_id, cur.rowcount,
            )
            return cur.rowcount

    def get_recovery_cycles(self, project_id: str) -> int:
        """How many `/plan continue` recovery cycles this project has used."""
        with self.conn() as conn:
            row = conn.execute(
                "SELECT recovery_cycles FROM projects WHERE id=?", (project_id,)
            ).fetchone()
        return (row["recovery_cycles"] or 0) if row else 0

    def get_blocked_by_failed(self, project_id: str) -> list[dict]:
        """Return pending tasks blocked (directly or transitively) by failed dependencies.

        A task X is reported as blocked when any failed task F is reachable from X
        through its dependency graph (X -> ... -> F). This gives the user-facing
        recovery explanation the full chain, e.g. A -> B -> C with A failed reports
        both B (blocked by A) and C (blocked by A).
        """
        tasks = self.get_all_tasks(project_id)
        status_by_id = {t["id"]: t["status"] for t in tasks}
        dep_by_id: dict[str, list[str]] = {}
        for t in tasks:
            try:
                dep_by_id[t["id"]] = json.loads(t.get("depends_on", "[]"))
            except (TypeError, ValueError):
                dep_by_id[t["id"]] = []

        failed_ids = {t["id"] for t in tasks if t["status"] == TaskStatus.FAILED.value}

        def _transitively_blocks(task_id: str) -> set[str]:
            """Return set of failed dependency ids reachable from task_id."""
            blocking: set[str] = set()
            seen: set[str] = set()
            stack = list(dep_by_id.get(task_id, []))
            while stack:
                dep_id = stack.pop()
                if dep_id in seen:
                    continue
                seen.add(dep_id)
                if dep_id not in status_by_id:
                    # Missing dependency ids must not silently deadlock; report them
                    # as blocking so the user can fix the plan.
                    blocking.add(f"missing:{dep_id}")
                    continue
                if dep_id in failed_ids:
                    blocking.add(dep_id)
                stack.extend(dep_by_id.get(dep_id, []))
            return blocking

        blocked = []
        for task in tasks:
            if task["status"] != TaskStatus.PENDING.value:
                continue
            blocking = sorted(_transitively_blocks(task["id"]))
            if blocking:
                blocked.append({"task": task, "blocked_by": blocking})
        return blocked

    def claim_task(self, project_id: str, task_id: str) -> int:
        """
        Atomically claim a pending task and report which attempt this is.

        Returns the 1-based attempt number of the claim that was won, or 0 when
        the task was not claimable (already in progress, completed or failed).
        The attempt number is derived here, in the same statement that performs
        the transition, so the scheduler and the worker can never disagree about
        which attempt is running.

        ``max_retries`` is the TOTAL attempts allowed in one recovery cycle, and
        ``retry_count`` is how many have already been spent in this cycle, so a
        claim always represents attempt ``retry_count + 1``.

        ``total_attempts`` is incremented here, when the attempt *starts*, so it
        counts every attempt whether it went on to succeed or fail. It is never
        reset, which is what makes it a usable record of a task's real history
        across /plan continue cycles.
        """
        with self.conn() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status=?, started_at=unixepoch(),
                    total_attempts=COALESCE(total_attempts, 0) + 1
                WHERE project_id=? AND id=? AND status=?
                """,
                (
                    TaskStatus.IN_PROGRESS.value,
                    project_id,
                    task_id,
                    TaskStatus.PENDING.value,
                ),
            )
            if cur.rowcount != 1:
                return 0
            row = conn.execute(
                "SELECT retry_count FROM tasks WHERE project_id=? AND id=?",
                (project_id, task_id),
            ).fetchone()
            return (row["retry_count"] or 0) + 1

    def next_attempt(self, project_id: str, task_id: str) -> int:
        """The attempt a claim of this task would represent, without claiming."""
        with self.conn() as conn:
            row = conn.execute(
                "SELECT retry_count FROM tasks WHERE project_id=? AND id=?",
                (project_id, task_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"Task {task_id} not found in project {project_id}")
        return (row["retry_count"] or 0) + 1

    def complete_task(self, project_id: str, task_id: str, result: str) -> None:
        """Mark a task as completed and store a bounded result.

        A task result is the worker's summary, not its conversation. The store
        bounds what it persists so a chatty worker cannot grow the task database
        without limit, and so downstream tasks read a bounded dependency string.
        """
        bounded = bounded_result(result)
        with self.conn() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status=?, result=?, error=NULL, completed_at=unixepoch()
                WHERE project_id=? AND id=?
                """,
                (TaskStatus.COMPLETED.value, bounded, project_id, task_id),
            )
        logger.info(
            "Completed task",
            extra={"task_id": task_id, "result_chars": len(bounded),
                   "truncated": len(bounded) != len(str(result or ""))},
        )

    def fail_task(self, project_id: str, task_id: str, error: str, *, force: bool = False) -> str:
        """
        Record a task failure.

        Retry semantics (explicit): `max_retries` is the TOTAL number of attempts.
        A task fails on attempt N, retry_count becomes N. While retry_count < max_retries
        the task returns to PENDING (will be retried within this run); once
        retry_count >= max_retries it becomes permanently FAILED for this run.

        When force=True (used for non-retryable/deterministic errors), the retry
        budget is exhausted immediately and the task goes straight to FAILED.

        Returns the resulting status: 'pending' (will retry) or 'failed'.

        Two counters, deliberately:
          retry_count     attempts in the CURRENT recovery cycle that have FAILED.
                          Reset by reset_failed_tasks_for_recovery, and the only
                          thing the automatic budget is measured against.
          total_attempts  every attempt ever started, successful or not.
                          Incremented by claim_task and never reset, so a task's
                          real history stays visible however often a human runs
                          /plan continue.
        """
        with self.conn() as conn:
            row = conn.execute(
                "SELECT retry_count, max_retries, total_attempts "
                "FROM tasks WHERE project_id=? AND id=?",
                (project_id, task_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Task {task_id} not found in project {project_id}")

            new_retry_count = row["retry_count"] + 1
            max_retries = row["max_retries"]
            total_attempts = row["total_attempts"] or 0

            if not force and new_retry_count < max_retries:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status=?, error=?, retry_count=?, started_at=NULL
                    WHERE project_id=? AND id=?
                    """,
                    (
                        TaskStatus.PENDING.value,
                        bounded_result(error),
                        new_retry_count,
                        project_id,
                        task_id,
                    ),
                )
                logger.warning(
                    "Task %s failed, retry %s/%s (attempt %s overall): %s",
                    task_id, new_retry_count, max_retries, total_attempts, error[:200],
                )
                return TaskStatus.PENDING.value

            # The stored error is re-tagged rather than the raw text kept, so the
            # row itself says whether the failure was retried to exhaustion or
            # refused outright. The prefix the caller attached is stripped first:
            # it is already encoded in the tag, and keeping both made the column
            # read "[failed] [non-retryable] ...".
            persist_error = _strip_retry_prefix(error)
            persist_error = (
                f"[retry-exhausted] [retryable] {persist_error}"
                if not force
                else f"[failed] {persist_error}"
            )
            conn.execute(
                """
                UPDATE tasks
                SET status=?, error=?, retry_count=?
                WHERE project_id=? AND id=?
                """,
                (TaskStatus.FAILED.value, bounded_result(persist_error),
                 new_retry_count, project_id, task_id),
            )
            logger.error(
                "Task %s permanently failed after %s attempts in this cycle "
                "(%s overall): %s",
                task_id, new_retry_count, total_attempts, persist_error[:200],
            )
            return TaskStatus.FAILED.value

    def get_progress(self, project_id:str)->dict[str,int]:
        """
        Return a dict with counts for every task status of a project.
        """
        with self.conn() as conn:
            rows = conn.execute(
                    """ SELECT status, COUNT(*) as count FROM tasks WHERE project_id=? GROUP BY status """,
                    (project_id,)
                ).fetchall()
            progress = {
                TaskStatus.PENDING.value: 0,
                TaskStatus.IN_PROGRESS.value: 0,
                TaskStatus.COMPLETED.value: 0,
                TaskStatus.FAILED.value: 0,
                TaskStatus.BLOCKED.value: 0,
                TaskStatus.SKIPPED.value: 0,
            }
            for row in rows:
                if row["status"] in progress:
                    progress[row["status"]] = row["count"]
            return progress

    def get_dep_results(self, project_id: str, dep_ids: list[str]) -> list[dict[str, str]]:
        """Return completed dependency results scoped to a project."""
        if not dep_ids:
            return []
        with self.conn() as conn:
            placeholders = ",".join(["?"] * len(dep_ids))
            rows = conn.execute(
                f"""
                SELECT id, result FROM tasks
                WHERE project_id=? AND id IN ({placeholders}) AND status=?
                """,
                (project_id, *dep_ids, TaskStatus.COMPLETED.value),
            ).fetchall()
            return [dict(row) for row in rows]
