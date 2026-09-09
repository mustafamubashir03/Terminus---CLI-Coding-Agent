import time
from dataclasses import dataclass, field
from enum import Enum
from terminus.observability.logging import get_logger
import json
import uuid
import sqlite3
from contextlib import contextmanager

logger = get_logger(__name__)

class TaskStatus(str, Enum):
    """All valid states of a task. 
    PENDING: task is waiting to be started.
    IN_PROGRESS: task is currently being worked on.
    COMPLETED: task has been completed.
    FAILED: task has failed and cannot be retried.
    BLOCKED: task is blocked by another task and cannot be started.
    SKIPPED: task has been skipped and will not be worked on.
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

@dataclass
class Task:
    """ 
    Single unit of work inside a project
    """
    id: str
    project_id: str
    title: str
    description: str
    task_type: str
    status: str = TaskStatus.PENDING
    depends_on: str="[]"
    output_files: str = "[]"
    result: str| None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    execution_order: int = 0
    started_at: float | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class TaskStore:
    def __init__(self, db_path:str="tasks.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def conn(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
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
                    created_at REAL DEFAULT (unixepoch())
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
                    execution_order INTEGER DEFAULT 0,
                    started_at REAL,
                    created_at REAL DEFAULT (unixepoch()),
                    completed_at REAL,
                    PRIMARY KEY (project_id, id),
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                )
            ''')
            self._migrate_task_primary_key(connection)

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

    def _get_all_tasks(self, project_id: str) -> list:
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, title, description, task_type,
                status, depends_on, output_files, acceptance_criteria,
                result, error, retry_count, max_retries,
                execution_order, started_at, created_at, completed_at
                FROM tasks
                WHERE project_id=?
                """,
                (project_id,)
            ).fetchall()

        return [dict(row) for row in rows]

    def create_project(self, goal:str, plan)->str:
        """
        Persist an approved ExecutionPlan as a project + task rows.
        Returns the new project_id (UUID)
         """
        project_id = str(uuid.uuid4())
        created_at = time.time()
        with self.conn() as conn:
            conn.execute(
                "INSERT INTO projects(id, name, goal, plan_json, status, created_at) VALUES(?,?,?,?,?,?)",
                (
                    project_id,
                    plan.project_name,
                    goal,
                    plan.model_dump_json(),
                    ProjectStatus.APPROVED.value,
                    created_at,
                ),
            )
            for i, pt in enumerate(plan.tasks):
                conn.execute(
                    """INSERT INTO tasks(id, project_id, title, description, task_type, depends_on, output_files, acceptance_criteria, execution_order) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (pt.id, project_id, pt.title, pt.description, pt.task_type.value, json.dumps(pt.depends_on), json.dumps(pt.output_files), json.dumps(pt.acceptance_criteria), i)
                )
        logger.info(
            "Successfully created project and tasks",
            extra={"project_id": project_id, "task_count": len(plan.tasks)},
        )
        return project_id

    def get_latest_approved_project(self) -> str | None:
        """Return the most recently approved project_id, or None if none exists."""
        with self.conn() as conn:
            row = conn.execute(
                "SELECT id FROM projects WHERE status=? ORDER BY created_at DESC LIMIT 1",
                (ProjectStatus.APPROVED.value,),
            ).fetchone()
            if row:
                return row[0]
        return None

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
        """Reset in-progress tasks to pending on resume (single-process assumption)."""
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

    def reset_retryable_failed_tasks(self, project_id: str) -> int:
        """Reset permanently-failed tasks that still have retries left."""
        with self.conn() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status=?, started_at=NULL
                WHERE project_id=?
                  AND status=?
                  AND retry_count < max_retries
                """,
                (TaskStatus.PENDING.value, project_id, TaskStatus.FAILED.value),
            )
            return cur.rowcount

    def get_blocked_by_failed(self, project_id: str) -> list[dict]:
        """Return pending tasks blocked by permanently failed dependencies."""
        tasks = self._get_all_tasks(project_id)
        by_id = {t["id"]: t for t in tasks}
        failed_ids = {t["id"] for t in tasks if t["status"] == TaskStatus.FAILED.value}
        blocked = []
        for task in tasks:
            if task["status"] != TaskStatus.PENDING.value:
                continue
            dep_ids = json.loads(task.get("depends_on", "[]"))
            blocking = [dep_id for dep_id in dep_ids if dep_id in failed_ids]
            if blocking:
                blocked.append({"task": task, "blocked_by": blocking})
        return blocked

    def claim_task(self, project_id: str, task_id: str) -> bool:
        """
        Atomically claim a task if it's pending.
        Returns True if claimed, False if already in-progress or completed/failed.
        """
        with self.conn() as conn:
            cur = conn.execute(
                """
                UPDATE tasks
                SET status=?, started_at=unixepoch()
                WHERE project_id=? AND id=? AND status=?
                """,
                (
                    TaskStatus.IN_PROGRESS.value,
                    project_id,
                    task_id,
                    TaskStatus.PENDING.value,
                ),
            )
            return cur.rowcount == 1

    def complete_task(self, project_id: str, task_id: str, result: str) -> None:
        """Mark a task as completed and store the result."""
        with self.conn() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status=?, result=?, completed_at=unixepoch()
                WHERE project_id=? AND id=?
                """,
                (TaskStatus.COMPLETED.value, result, project_id, task_id),
            )
        logger.info(
            "Completed task",
            extra={"task_id": task_id, "result_chars": len(result)},
        )

    def fail_task(self, project_id: str, task_id: str, error: str) -> str:
        """
        Record a task failure. Retries while retry_count < max_retries.
        Returns the resulting status: 'pending' (will retry) or 'failed'.
        """
        with self.conn() as conn:
            row = conn.execute(
                "SELECT retry_count, max_retries FROM tasks WHERE project_id=? AND id=?",
                (project_id, task_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Task {task_id} not found in project {project_id}")

            new_retry_count = row["retry_count"] + 1
            max_retries = row["max_retries"]

            if new_retry_count < max_retries:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status=?, error=?, retry_count=?, started_at=NULL
                    WHERE project_id=? AND id=?
                    """,
                    (
                        TaskStatus.PENDING.value,
                        error,
                        new_retry_count,
                        project_id,
                        task_id,
                    ),
                )
                logger.warning(
                    f"Task {task_id} failed, retry {new_retry_count}/{max_retries}: {error[:200]}"
                )
                return TaskStatus.PENDING.value

            conn.execute(
                """
                UPDATE tasks
                SET status=?, error=?, retry_count=?
                WHERE project_id=? AND id=?
                """,
                (TaskStatus.FAILED.value, error, new_retry_count, project_id, task_id),
            )
            logger.error(
                f"Task {task_id} permanently failed after {new_retry_count} attempts: {error[:200]}"
            )
            return TaskStatus.FAILED.value

    def get_progress(self, project_id:str)->dict[str,int]:
        """
        Return a dict with counts for pending, in-progress, completed, and failed tasks for a project.
        """
        with self.conn() as conn:
            rows = conn.execute(
                    """ SELECT status, COUNT(*) as count FROM tasks WHERE project_id=? GROUP BY status """,
                    (project_id,)
                ).fetchall()
            progress = {"pending":0,"in_progress":0,"completed":0,"failed":0}
            for row in rows:
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