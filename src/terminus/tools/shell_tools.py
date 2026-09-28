"""Shell execution for the /ask agent.

Deliberately separate from ``tools/terminal_tools.py``, which belongs to the
/plan executor and is left untouched.

The model supplies only ``command`` and an optional ``working_directory``. It
cannot supply a permission level, an approval, or a timeout: those come from
``terminus.permissions`` and from module constants below. Permission is always
resolved by the runtime before a subprocess is created.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from langchain.tools import tool

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

logger = get_logger(__name__)

# Runtime-owned limits. Not model-controllable.
_COMMAND_TIMEOUT_SECONDS = 120
_MAX_STREAM_CHARS = 8_000
_MAX_TOTAL_CHARS = 20_000

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


@tool
def run_command(command: str, working_directory: str | None = None) -> str:
    """
    Run a shell command in the project and return its exit code, stdout and
    stderr. Use it to run tests, linters, type checks, builds, or to inspect
    generated output and version-control state.

    Read-only commands (ls, git status, grep, pytest --collect-only, version
    probes) run directly. Anything that modifies state needs runtime approval,
    and destructive commands are refused unless a human explicitly allows this
    one command. You cannot grant yourself permission.

    Output is truncated; a '[truncated]' marker is added. A non-zero exit code
    or a timeout is reported as a normal result, not a failure of the tool.
    """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()

    cwd = Path(working_directory).expanduser() if working_directory else Path.cwd()
    try:
        cwd = cwd.resolve()
    except OSError:
        pass
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


def _execute(command: str, cwd: Path, decision) -> str:
    """Run an authorised command. Called with the project writer lock held
    when the command can mutate the workspace."""

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT_SECONDS,
            cwd=str(cwd),
            env=sanitized_env(),
        )
    except subprocess.TimeoutExpired as exc:
        partial = ""
        for stream in (exc.stdout, exc.stderr):
            if stream:
                partial += stream if isinstance(stream, str) else stream.decode("utf-8", "replace")
        partial, cut = _bounded(partial, _MAX_TOTAL_CHARS)
        note = "\n[truncated]" if cut else ""
        return (
            f"Command timed out after {_COMMAND_TIMEOUT_SECONDS}s and was terminated.\n"
            f"  command: {command}\n"
            f"  working directory: {cwd}\n"
            f"partial output:\n{partial}{note}"
        )
    except (OSError, ValueError) as exc:
        return f"Cannot run command: {type(exc).__name__}: {exc}"

    stdout, out_cut = _bounded(completed.stdout or "", _MAX_STREAM_CHARS)
    stderr, err_cut = _bounded(completed.stderr or "", _MAX_STREAM_CHARS)
    stdout = redact_secrets(stdout)
    stderr = redact_secrets(stderr)

    parts = [
        f"$ {command}",
        f"working directory: {cwd}",
        f"result: {_describe_exit(completed.returncode)}",
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

    logger.info("ran command (level=%s, rc=%s)", decision.level.value, completed.returncode)
    return "\n".join(parts)
