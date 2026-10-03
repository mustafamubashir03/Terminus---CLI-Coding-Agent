"""Shell execution for the /ask agent.

Deliberately separate from ``tools/terminal_tools.py``, which belongs to the
/plan executor and is left untouched.

The model supplies only ``command`` and an optional ``working_directory``. It
cannot supply a permission level, an approval, or a timeout: those come from
``terminus.permissions`` and from module constants below. Permission is always
resolved by the runtime before a subprocess is created.

The working directory is the other half of the model's reach. ``run_command`` is
the tool that can touch anything the OS lets it, so its ``working_directory`` is
resolved through ``terminus.workspace`` like every filesystem path and defaults
to the workspace root rather than the process cwd.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from langchain_core.tools import ToolException

from terminus.coordination import project_write_guard
from terminus.observability.logging import get_logger
from terminus.permissions import (
    Operation,
    PermissionPolicy,
    get_permission_policy,
    redact_secrets,
    sanitized_env,
    set_permission_policy,
)
from terminus.tools import refusing_tool
from terminus.workspace import (
    WorkspaceViolation,
    project_root,
    resolve_in_workspace,
)

logger = get_logger(__name__)

# Runtime-owned limits. Not model-controllable.
_COMMAND_TIMEOUT_SECONDS = 120
_MAX_STREAM_CHARS = 8_000
_MAX_TOTAL_CHARS = 20_000

# What the timeout does and does not do.
#
# It bounds how long the *agent* waits, which is the property that matters here:
# a command that hangs produces an observation the model can reason about instead
# of a turn that never ends.
#
# Getting that bound to actually hold took killing the process tree, not just the
# process. With `shell=True` the process Python starts is the shell, and the
# command is its child holding the same stdout/stderr pipes. `subprocess.run`
# kills the shell and then drains the pipes - which cannot reach EOF until every
# descendant exits. Measured on Windows: a 2 second timeout returned after 29
# seconds, because that was when `ping` finished on its own. A command that
# backgrounds something long-lived made the tool hang for as long as that thing
# lived, which is the one outcome a timeout exists to prevent.
#
# So the tree is killed on the way out, and the agent's wait is bounded. What
# this is *not*: containment. A child that deliberately detaches - `start /b` on
# Windows, `nohup` or a double fork on POSIX - is re-parented away from the tree,
# and `taskkill /T` and `killpg` do not reach it. Measured on Windows: a
# `start /b` grandchild still wrote its file after the tool had returned. Closing
# that needs a job object with KILL_ON_JOB_CLOSE on Windows, which is not a
# timeout constant's worth of work and is not attempted here.
_KILL_GRACE_SECONDS = 3
"""How long to wait for the pipes to close after the tree has been killed.

Short, because at this point the command is already dead; the grace is only for
the pipes to drain what was written before it stopped. Exceeding it costs
partial output, never an unbounded wait.
"""

# The permission policy is process-wide and lives in terminus.permissions, so
# write_file / edit_file / run_command all consult the same one.
# set_permission_policy / get_permission_policy are re-exported here because
# callers already import them from this module.

__all__ = [
    "run_command",
    "set_permission_policy",
    "get_permission_policy",
    "PermissionPolicy",
]


def _bounded(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _describe_exit(returncode: int) -> str:
    if returncode == 0:
        return "exit code 0 (success)"
    if returncode < 0:
        return f"terminated by signal {-returncode}"
    return f"exit code {returncode} (failure)"


@refusing_tool
def run_command(command: str, working_directory: str | None = None) -> str:
    """
    Run a shell command in the project and return its exit code, stdout and
    stderr. Use it to run tests, linters, type checks, builds, or to inspect
    generated output and version-control state.

    Read-only commands (ls, git status, grep, pytest --collect-only, version
    probes) run directly. Anything that modifies state needs runtime approval,
    and destructive commands are refused unless a human explicitly allows this
    one command. You cannot grant yourself permission.

    'working_directory' is workspace-relative and defaults to the workspace root.
    It cannot point outside the workspace: a refusal is raised as a tool error,
    not returned as output.

    Output is truncated; a '[truncated]' marker is added. A non-zero exit code
    or a timeout is reported as a normal result, not a failure of the tool.
    """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()

    cwd = _working_directory(working_directory)
    if not cwd.is_dir():
        return f"Cannot run command: working directory does not exist: {cwd}"

    # Authorise, then hold the project writer lock for a mutating command.
    # Read-only commands never take the lock, so two tasks may read at once.
    with project_write_guard(
        Operation.EXECUTE, command=command, context=str(cwd)
    ) as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked + f"\n  working directory: {cwd}"
        return _execute(command, cwd, grant.decision)


def _working_directory(working_directory: str | None) -> Path:
    """The directory a command runs in: the workspace root, or somewhere inside it.

    The workspace root rather than the process cwd, because the workspace is the
    identity that is actually validated everywhere else; a bare ``Path.cwd()``
    here would silently reintroduce "wherever the process happens to be".
    """
    try:
        return (
            resolve_in_workspace(working_directory)
            if working_directory
            else project_root()
        )
    except WorkspaceViolation as exc:
        raise ToolException(
            f"{exc} run_command cannot execute outside the workspace."
        ) from exc


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill *process* and the commands it started.

    Best effort by design: a process that is already gone, or a platform whose
    kill mechanism is unavailable, must not turn a timeout into a different
    error. The caller's own ``kill()`` runs afterwards regardless.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=_KILL_GRACE_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, AttributeError):
            pass
    try:
        process.kill()
    except OSError:
        pass


def _spawn(command: str, cwd: Path) -> subprocess.Popen:
    """Start *command* through a shell, in its own process group.

    The group is what makes a timeout stop the whole command rather than the
    shell that was asked to run it.
    """
    popen_kwargs: dict = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd),
        env=sanitized_env(),
        **popen_kwargs,
    )


def _execute(command: str, cwd: Path, decision) -> str:
    """Run an authorised command. Called with the project writer lock held
    when the command can mutate the workspace."""

    try:
        process = _spawn(command, cwd)
    except (OSError, ValueError) as exc:
        return f"Cannot run command: {type(exc).__name__}: {exc}"

    try:
        stdout, stderr = process.communicate(timeout=_COMMAND_TIMEOUT_SECONDS)
        returncode = process.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            # Something outside the tree still holds a pipe open. Waiting for it
            # would reinstate exactly the unbounded wait this code exists to
            # prevent, so report the timeout and drop the partial output rather
            # than pretend to have captured it.
            stdout, stderr = "", ""
            for pipe in (process.stdout, process.stderr):
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        returncode = None
    except (OSError, ValueError) as exc:
        return f"Cannot run command: {type(exc).__name__}: {exc}"

    if timed_out:
        partial, cut = _bounded((stdout or "") + (stderr or ""), _MAX_TOTAL_CHARS)
        note = "\n[truncated]" if cut else ""
        body = f"\npartial output:\n{partial}{note}" if partial else ""
        return (
            f"Command timed out after {_COMMAND_TIMEOUT_SECONDS}s and was terminated."
            f"\n  command: {command}\n  working directory: {cwd}{body}"
        )

    stdout, out_cut = _bounded(stdout or "", _MAX_STREAM_CHARS)
    stderr, err_cut = _bounded(stderr or "", _MAX_STREAM_CHARS)
    stdout = redact_secrets(stdout)
    stderr = redact_secrets(stderr)

    parts = [
        f"$ {command}",
        f"working directory: {cwd}",
        f"result: {_describe_exit(returncode)}",
    ]
    if out_cut:
        parts.append(f"stdout: [truncated at {_MAX_STREAM_CHARS} chars]")
    if stdout.strip():
        parts.append("--- stdout ---")
        parts.append(stdout.rstrip())
    else:
        parts.append("--- stdout --- (empty)")
    if err_cut:
        parts.append(f"stderr: [truncated at {_MAX_STREAM_CHARS} chars]")
    if stderr.strip():
        parts.append("--- stderr ---")
        parts.append(stderr.rstrip())
    else:
        parts.append("--- stderr --- (empty)")

    total = len(stdout) + len(stderr)
    if total > _MAX_TOTAL_CHARS:
        parts.append(f"[total output truncated at {_MAX_TOTAL_CHARS} characters]")

    logger.info("ran command (level=%s, rc=%s)", decision.level.value, returncode)
    return "\n".join(parts)
