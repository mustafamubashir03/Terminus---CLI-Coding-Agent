"""Shell execution for /plan workers.

This module is NOT the /ask shell (``tools/shell_tools.py``) and does not
replace it. The two differ deliberately:

  * /ask      120s timeout, rich structured report, ``working_directory`` arg
  * /plan      30s timeout, compact plain-text report, ``directory`` arg, and a
               per-task-type tool list

What they must NOT differ on is *authorisation*. Previously this module gated
commands with a substring denylist of eleven exact strings, which is trivially
bypassed (``rm  -rf /``, ``Remove-Item -Recurse``, ``git reset --hard``, any
``curl | bash`` URL not on the list) and did not consult the permission policy at
all. A /plan worker could therefore run anything the runtime would have refused
for /ask.

Authorisation now goes through the same execution-scoped boundary: the worker's
own ``ExecutionContext`` supplies the policy, so a /plan worker is allowed
exactly what its policy allows and nothing more. A worker has no approver, so
DESTRUCTIVE is refused. Output is redacted and bounded like /ask's, so a worker's
transcript cannot leak credentials or blow up its context.
"""

import subprocess
from contextlib import contextmanager

from langchain.tools import tool

from terminus.coordination import project_write_guard
from terminus.execution import current_execution
from terminus.permissions import Operation, redact_secrets

_TIMEOUT_SECONDS = 30
_MAX_STREAM_CHARS = 8_000
"""Matches the /ask per-stream bound and the persisted task-result bound."""


@contextmanager
def _write_guard(command: str, directory: str | None):
    """Authorise a shell command and hold the project writer lock while it runs.

    Same contract as ``terminus.coordination.project_write_guard``, wrapped here
    so both /plan shell tools label their refusal with the directory. Read-only
    commands are authorised but never take the lock.
    """
    running = current_execution()
    context = f"{running.label} in {directory}" if running else None
    with project_write_guard(
        Operation.EXECUTE, command=command, context=context
    ) as grant:
        yield grant


def _bounded(text: str) -> str:
    if len(text) <= _MAX_STREAM_CHARS:
        return text
    return f"{text[:_MAX_STREAM_CHARS]}\n... [truncated at {_MAX_STREAM_CHARS} chars]"


def _format_result(result: subprocess.CompletedProcess) -> str:
    parts = []
    if result.stdout:
        parts.append(redact_secrets(_bounded(result.stdout.rstrip())))
    if result.stderr:
        parts.append(f"ERROR:\n{redact_secrets(_bounded(result.stderr.rstrip()))}")
    if result.returncode != 0:
        parts.append(f"Exit code {result.returncode}")
    return "\n".join(parts) if parts else "No output"


@tool("run_shell_command")
def run_command(command: str) -> str:
    """ Run a shell command in the project. Times out after 30 seconds.

    Authorized by the current execution's permission policy: read-only commands
    run, state-changing ones depend on the policy, and destructive ones are
    refused. Refusals are returned as text, not raised.
    """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()

    with _write_guard(command, None) as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked
        return _execute(command, None)


def _execute(command: str, directory: str | None) -> str:
    """Run an authorised command. Called with the project writer lock held when
    the command can mutate the workspace."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=directory,
        )
        return _format_result(result)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {_TIMEOUT_SECONDS} seconds: {command}"
    except Exception as e:
        return f"Error running command: {str(e)}"


@tool("run_command_in_directory")
def run_in_directory(command: str, directory: str = None) -> str:
    """ Run a shell command inside a specific directory. Times out after 30 seconds.

    Authorized by the current execution's permission policy, exactly like
    'run_shell_command'.
    """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()

    with _write_guard(command, directory) as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked
        return _execute(command, directory)
