"""The execution boundary for model-issued shell commands.

A Sandbox is one long-lived container that a session runs its commands in. The
project is bind-mounted into it, so the container edits the same files the host
does; what changes is where the *process* runs.

    Workspace  = persistent project state
    Sandbox    = disposable execution environment
    Bash tools = policy and tool interface
    Docker     = this boundary's implementation

What lives where, and why it matters:

* :mod:`terminus.permissions` decides whether a command may run. Unchanged, and
  deliberately not reimplemented here. A Sandbox never classifies anything.
* The shell tools own the tool schema, the approval flow, the writer lock, the
  per-tool timeout and how output is truncated, redacted and presented.
* This module owns the container and nothing else: create it, keep it, run a
  command in it, throw it away.

No host fallback. If Docker is unavailable or a command fails in the container,
that is reported as a failure. Silently running the command on the host instead
would defeat the entire boundary while looking like it worked.
"""

from terminus.sandbox.docker_sandbox import (
    CONTAINER_NAME_PREFIX,
    CONTAINER_STARTUP_TIMEOUT_SECONDS,
    INTERNAL_SHELL,
    OWNERSHIP_LABEL_KEY,
    OWNERSHIP_LABEL_VALUE,
    SANDBOX_IMAGE,
    SANDBOX_WORKSPACE_PATH,
    ExecutionResult,
    Sandbox,
    SandboxError,
    SandboxTimeout,
    SandboxUnavailable,
    current_sandbox,
    sandbox_disabled,
    sandbox_enabled,
    sandbox_scope,
    set_sandbox,
)

__all__ = [
    "CONTAINER_NAME_PREFIX",
    "CONTAINER_STARTUP_TIMEOUT_SECONDS",
    "INTERNAL_SHELL",
    "OWNERSHIP_LABEL_KEY",
    "OWNERSHIP_LABEL_VALUE",
    "SANDBOX_IMAGE",
    "SANDBOX_WORKSPACE_PATH",
    "ExecutionResult",
    "Sandbox",
    "SandboxError",
    "SandboxTimeout",
    "SandboxUnavailable",
    "current_sandbox",
    "sandbox_disabled",
    "sandbox_enabled",
    "sandbox_scope",
    "set_sandbox",
]
