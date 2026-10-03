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

The directory argument gets the same treatment. ``run_command_in_directory``
used to hand ``directory`` straight to ``subprocess.run(cwd=...)`` with no check
at all, which made it the one tool in the project that could run a command
anywhere on the host. It now resolves through ``terminus.workspace`` like every
other model-supplied path, and a directory outside the workspace is refused.
"""

import subprocess

from langchain_core.tools import ToolException

from terminus.coordination import project_write_guard
from terminus.permissions import Operation, redact_secrets, sanitized_env
from terminus.sandbox import SandboxError, current_sandbox
from terminus.tools import refusing_tool
from terminus.workspace import (
    WorkspaceViolation,
    project_root,
    resolve_in_workspace,
)

_TIMEOUT_SECONDS = 30
_MAX_STREAM_CHARS = 8_000
"""Matches the /ask per-stream bound and the persisted task-result bound."""


def _bounded(text: str) -> str:
    if len(text) <= _MAX_STREAM_CHARS:
        return text
    return f"{text[:_MAX_STREAM_CHARS]}\n... [truncated at {_MAX_STREAM_CHARS} chars]"


def _exit_code(result) -> int:
    """The exit status of either backend's result object.

    ``subprocess.CompletedProcess`` spells this ``returncode`` and
    :class:`~terminus.sandbox.ExecutionResult` spells it ``exit_code``. Reading
    one name off both meant the host path raised ``AttributeError`` on every
    command - a silent break of the only backend that was there before.
    """
    for attribute in ("exit_code", "returncode"):
        value = getattr(result, attribute, None)
        if value is not None:
            return int(value)
    raise AttributeError(
        f"{type(result).__name__} has neither exit_code nor returncode"
    )


def _format_result(result) -> str:
    """The compact report a worker reads.

    Accepts a ``subprocess.CompletedProcess`` or a
    :class:`~terminus.sandbox.ExecutionResult`; both carry stdout, stderr and an
    exit status, so one formatter serves both execution backends.
    """
    parts = []
    if result.stdout:
        parts.append(redact_secrets(_bounded(result.stdout.rstrip())))
    if result.stderr:
        parts.append(f"ERROR:\n{redact_secrets(_bounded(result.stderr.rstrip()))}")
    exit_code = _exit_code(result)
    if exit_code != 0:
        parts.append(f"Exit code {exit_code}")
    return "\n".join(parts) if parts else "No output"


@refusing_tool(name="run_shell_command")
def run_command(command: str) -> str:
    """ Run a shell command in the project. Times out after 30 seconds.

    Authorized by the current execution's permission policy: read-only commands
    run, state-changing ones depend on the policy, and destructive ones are
    refused. Refusals are returned as text, not raised.
    """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()

    with project_write_guard(Operation.EXECUTE, command=command) as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked
        # The workspace root, explicitly. Passing None would be read as "the host
        # process's current directory" by the host executor, which is whatever
        # directory Terminus happened to be launched from - not this project. The
        # container path resolves None to /workspace, so leaving it implicit made
        # the two execution backends disagree about where "here" is.
        return _execute(command, project_root())


def _execute(command: str, directory: str | None) -> str:
    """Run an authorised command. Called with the project writer lock held when
    the command can mutate the workspace.

    The permission decision, the lock, this module's own 30 second timeout and
    the report format below are unchanged. Only where the command runs has moved:
    with a Sandbox installed for the execution it runs in that container, and
    without one it runs as a host process.
    """
    sandbox = current_sandbox()
    if sandbox is not None:
        try:
            result = sandbox.execute(
                command, timeout=_TIMEOUT_SECONDS, cwd=directory
            )
        except SandboxError as exc:
            return f"Error running command: {exc}"
        return _format_result(result)
    return _execute_on_host(command, directory)


def _execute_on_host(command: str, directory: str | None) -> str:
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
            env=sanitized_env(),
        )
        return _format_result(result)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {_TIMEOUT_SECONDS} seconds: {command}"
    except Exception as e:
        return f"Error running command: {str(e)}"


@refusing_tool(name="run_command_in_directory")
def run_in_directory(command: str, directory: str = None) -> str:
    """ Run a shell command inside a workspace directory. Times out after 30 seconds.

    Authorized by the current execution's permission policy, exactly like
    'run_shell_command'. 'directory' is workspace-relative, defaults to the
    workspace root, and cannot point outside the workspace.
    """
    if not command or not command.strip():
        return "No command provided"
    command = command.strip()

    try:
        target = (
            resolve_in_workspace(directory) if directory else project_root()
        )
    except WorkspaceViolation as exc:
        raise ToolException(
            f"{exc} run_command_in_directory cannot execute outside the workspace."
        ) from exc

    if not target.is_dir():
        return f"Cannot run command: directory does not exist: {target}"

    with project_write_guard(Operation.EXECUTE, command=command) as grant:
        blocked = grant.refused or grant.deferred
        if blocked:
            return blocked
        return _execute(command, str(target))
