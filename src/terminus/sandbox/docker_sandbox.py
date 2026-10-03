"""One Docker container, used as the execution environment for a session.

A command runs where a Docker container can see it, in a container Terminus owns
and can throw away. The workspace is bind-mounted, so the container reads and
writes the real project files; only the process moves.

Design notes worth knowing before changing anything:

* **One container per Sandbox, not per command.** A session's commands share
  state that a per-command container would lose: a build in ``target/``, a
  ``pip install``, a ``git`` index warmed by an earlier call. The lifecycle is
  therefore ``start()`` / ``execute()`` ... / ``stop()``.
* **No host fallback, ever.** A Docker failure raises. Running the command on the
  host instead would preserve the behaviour this boundary exists to remove, and
  would do it invisibly.
* **The timeout is the caller's.** The two shell tools have different contracts
  (120s and 30s) and neither is the Sandbox's to change.
* **This module does not classify commands.** Permission lives in
  :mod:`terminus.permissions`. Everything arriving here has already been approved.

Fixed constants live here rather than in a shared module because this project has
none: the values are invariants of this boundary, not configuration, and inventing
a constants module for six values would be a larger change than the feature.

One environment caveat, found by testing rather than by reading. On Windows with
Docker Desktop, a workspace directory whose ACL excludes Docker's file-sharing
account causes every file the container creates to come back with a Windows ACL
that denies the logged-in user: the command succeeds, the file is on disk, and
the host then gets "access denied" reading it. Python's ``tempfile.mkdtemp`` is
one way to produce such a directory; a normal git checkout is not. It looks like a
broken sandbox while the container log shows a clean write, so it is worth
recognising before suspecting the image or the mount.
"""

from __future__ import annotations

import contextvars
import shlex
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from terminus.observability.logging import get_logger
from terminus.workspace import project_root

logger = get_logger(__name__)

SANDBOX_IMAGE = "terminus-sandbox:0.1.0"
"""The image every sandbox runs. Built from ``sandbox/Dockerfile``.

An explicit tag rather than ``latest``, so a container is always started from an
image whose contents are known. With ``latest`` a rebuild silently changes what
every later session executes, and a bug that reproduces once cannot be pinned to
the toolbelt that caused it.

The tag is written here and nowhere else. The Dockerfile and its comments refer
to it rather than restating it, so renaming the image is a one-line change that
cannot leave the build instructions pointing at a tag that no longer exists.
"""

SANDBOX_WORKSPACE_PATH = "/workspace"
"""Where the host workspace appears inside the container.

Fixed because it is baked into the bind mount and into every container path the
tools reason about. Host paths and container paths are different namespaces; see
:meth:`Sandbox.container_path`.
"""

CONTAINER_NAME_PREFIX = "terminus-sandbox-"
"""Readable prefix on every container. For humans, and nothing else.

Ownership is decided by :data:`OWNERSHIP_LABEL_KEY`, never by this prefix: a
prefix can be forged by anything that can create a container, and a container
whose name merely starts with ours is not necessarily ours.
"""

OWNERSHIP_LABEL_KEY = "terminus.managed"
OWNERSHIP_LABEL_VALUE = "sandbox"
"""The label that marks a container as Terminus's, and the only thing orphan
cleanup is allowed to select on."""

CONTAINER_STARTUP_TIMEOUT_SECONDS = 60
"""How long ``start()`` waits for the container to report running.

Bounded because an unbounded wait is the failure mode a timeout exists to
prevent: a wedged daemon would otherwise hang the session before the model has
been asked anything.
"""

INTERNAL_SHELL = "/bin/sh"
"""Commands run through a real shell, so pipes, ``&&``, ``||``, ``;``,
substitution and redirection mean what they mean to the permission classifier
that just approved them."""

_WORKSPACE_ENVIRONMENT = {
    "HOME": "/root",
    "TERM": "xterm",
    "LANG": "C.UTF-8",
}
"""The entire environment handed to a container.

Deliberately not the host's. ``terminus.permissions.sanitized_env`` decides what
a *host* subprocess may see, and it still does for the tools' own use; a
container gets a fixed minimal set and never the developer's shell.
"""


class SandboxError(RuntimeError):
    """The sandbox could not do what was asked."""


class SandboxUnavailable(SandboxError):
    """Docker itself is unusable: no daemon, no client, no such image."""


class SandboxTimeout(SandboxError):
    """A command exceeded the timeout its caller asked for."""


class ExecutionResult(NamedTuple):
    """What running a command produced.

    A tuple so ``exit_code, stdout, stderr = sandbox.execute(...)`` reads the way
    the contract says, with names so a caller can take one field by name.
    """

    exit_code: int
    stdout: str
    stderr: str


class Sandbox:
    """One container, started once, used for every command in the session."""

    def __init__(
        self,
        *,
        client: Any = None,
        image: str = SANDBOX_IMAGE,
        workspace: Path | None = None,
    ) -> None:
        self.image = image
        # Resolved through the existing workspace abstraction, never re-derived.
        # Resolved to an absolute path because this value is handed to Docker as
        # a volume source: a relative one is interpreted against the daemon's
        # working directory, not the client's, and the container then mounts
        # something that is not this workspace. A relative path also defeats
        # container_path's containment check, which compares resolved paths.
        self.workspace = Path(workspace).resolve() if workspace else project_root()
        self.sandbox_id = uuid.uuid4()
        self.container_name = f"{CONTAINER_NAME_PREFIX}{self.sandbox_id}"
        self._client = client
        self._container: Any = None

    # -- identity ---------------------------------------------------------

    @property
    def labels(self) -> dict[str, str]:
        """Labels applied to the container. Ownership is one of them."""
        return {
            OWNERSHIP_LABEL_KEY: OWNERSHIP_LABEL_VALUE,
            f"{OWNERSHIP_LABEL_KEY}.id": str(self.sandbox_id),
        }

    @property
    def container_id(self) -> str | None:
        return getattr(self._container, "id", None)

    @property
    def started(self) -> bool:
        return self._container is not None

    # -- docker client ----------------------------------------------------

    @property
    def client(self) -> Any:
        """The Docker client, built on first use.

        Deferred so importing this module never touches Docker, which keeps the
        test suite and a non-Docker install working.
        """
        if self._client is None:
            try:
                import docker
            except ImportError as exc:
                raise SandboxUnavailable(
                    "The docker package is not installed. Install the project's "
                    "dependencies, or set llm.sandbox_enabled = false to run "
                    "commands on the host."
                ) from exc
            try:
                self._client = docker.from_env()
            except Exception as exc:
                raise SandboxUnavailable(
                    f"Cannot reach the Docker daemon: {exc}"
                ) from exc
        return self._client

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Clean up stale sandboxes, then create and start this container.

        Idempotent in the sense that matters: a Sandbox that is already running is
        left alone rather than being silently replaced, because a caller that
        started it expects its state - a build directory, an installed package -
        to still be there.
        """
        if self.started:
            return

        self._cleanup_orphans()

        try:
            container = self.client.containers.run(
                self.image,
                name=self.container_name,
                command=["sleep", "infinity"],
                detach=True,
                tty=True,
                working_dir=SANDBOX_WORKSPACE_PATH,
                volumes={
                    str(self.workspace): {
                        "bind": SANDBOX_WORKSPACE_PATH,
                        "mode": "rw",
                    }
                },
                labels=self.labels,
                environment=dict(_WORKSPACE_ENVIRONMENT),
            )
        except Exception as exc:
            raise SandboxUnavailable(
                f"Could not start a sandbox from image {self.image!r}: {exc}. "
                f"Build it from the repository root with: "
                f"docker build -f src/terminus/sandbox/Dockerfile -t {self.image} ."
            ) from exc

        self._container = container
        self._wait_for_running()

    def stop(self) -> None:
        """Remove the container. Never raises."""
        container, self._container = self._container, None
        if container is None:
            return
        try:
            container.remove(force=True)
        except Exception as exc:
            logger.warning("Could not remove sandbox %s: %s", self.container_name, exc)
            return
        logger.info("Sandbox %s stopped", self.container_name)

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def _wait_for_running(self) -> None:
        """Block until the container reports running, or fail with the reason.

        Polls Docker's own state rather than sleeping a fixed interval, so a
        container that dies during startup is reported as a crash with its exit
        code rather than as a timeout.
        """
        deadline = time.monotonic() + CONTAINER_STARTUP_TIMEOUT_SECONDS
        last_state = "unknown"
        while True:
            try:
                self._container.reload()
                state = (self._container.status or "").lower()
            except Exception as exc:
                raise SandboxUnavailable(
                    f"Lost contact with sandbox {self.container_name}: {exc}"
                ) from exc

            last_state = state
            if state == "running":
                self._verify_workspace_visible()
                return
            if state in {"exited", "dead"}:
                raise SandboxUnavailable(
                    f"Sandbox {self.container_name} exited during startup "
                    f"(status={state}, exit_code={self._exit_code()}). "
                    f"The image {self.image!r} may not support "
                    f"{INTERNAL_SHELL!r} or the sleep command."
                )
            if time.monotonic() >= deadline:
                raise SandboxTimeout(
                    f"Sandbox {self.container_name} was still {last_state!r} after "
                    f"{CONTAINER_STARTUP_TIMEOUT_SECONDS}s."
                )
            time.sleep(0.2)

    def _exit_code(self) -> int | None:
        try:
            return self._container.attrs["State"].get("ExitCode")
        except Exception:
            return None

    def _verify_workspace_visible(self) -> None:
        """Confirm the bind mount maps the host workspace onto /workspace.

        A container can be perfectly healthy, ``/workspace`` can exist, and every
        command can still run against the wrong tree: the image creates
        ``/workspace`` at build time, so a mount that silently did not happen
        leaves a perfectly good empty directory sitting where the workspace
        should be. The earlier check here asked the container whether
        ``/workspace`` was a directory, which that build-time directory answers
        yes to - it passed against a container that had the host workspace
        nowhere in it.

        So this proves the mapping instead of its shape: write a token file on
        the host, read that exact path in the container, and require the token
        back. A missing mount, a mount of the wrong directory, or a mount that
        landed somewhere else all fail this. The token file is removed
        afterwards, including on failure, so a workspace is never left holding
        the proof of a check.
        """
        token = f"terminus-sandbox-probe-{uuid.uuid4().hex}"
        host_probe = self.workspace / f".{token}"
        try:
            host_probe.write_text(token, encoding="utf-8")
        except OSError as exc:
            raise SandboxUnavailable(
                f"Could not write a probe file into the workspace "
                f"{self.workspace}: {exc}"
            ) from exc
        container_probe = self.container_path(host_probe)
        try:
            result = self.client.containers.get(self.container_name).exec_run(
                [INTERNAL_SHELL, "-c", f"cat {shlex.quote(container_probe)}"],
                workdir=SANDBOX_WORKSPACE_PATH,
                demux=True,
            )
        except Exception as exc:
            raise SandboxUnavailable(
                f"Could not verify the workspace mount for {self.container_name}: {exc}"
            ) from exc
        finally:
            try:
                host_probe.unlink()
            except OSError:
                pass
        if token not in _decode(result.output[0]):
            raise SandboxUnavailable(
                f"Sandbox {self.container_name} is running but "
                f"{SANDBOX_WORKSPACE_PATH} is not the host workspace "
                f"{self.workspace}. Nothing will be executed against the wrong "
                f"tree; check the bind mount."
            )

    def _cleanup_orphans(self) -> int:
        """Remove Terminus sandbox containers left behind by a crashed process.

        Selects on the ownership label alone. A container that is still *running*
        is left alone even though it is labelled, because it most likely belongs to
        another live Terminus session; only containers that have stopped are
        orphans by definition. Anything not carrying the label is invisible here,
        so a user's own containers cannot be touched.
        """
        try:
            candidates = self.client.containers.list(
                all=True,
                filters={"label": f"{OWNERSHIP_LABEL_KEY}={OWNERSHIP_LABEL_VALUE}"},
            )
        except Exception as exc:
            logger.warning("Could not list sandbox containers for cleanup: %s", exc)
            return 0

        removed = 0
        for container in candidates:
            if getattr(container, "name", "") == self.container_name:
                continue
            try:
                if (container.status or "").lower() == "running":
                    continue
                container.remove(force=True)
                removed += 1
                logger.info("Removed orphaned sandbox %s", container.name)
            except Exception as exc:
                logger.warning(
                    "Could not remove orphaned sandbox %s: %s", container.name, exc
                )
        return removed

    # -- execution --------------------------------------------------------

    def container_path(self, host_path: Path | str) -> str:
        """Map a host path inside the workspace to its container path.

        Host and container paths are different namespaces. Passing a Windows host
        path straight through as a container working directory would silently mean
        a directory that does not exist, or worse, some unrelated one.
        """
        resolved = Path(host_path).resolve()
        try:
            relative = resolved.relative_to(self.workspace.resolve())
        except ValueError as exc:
            raise SandboxError(
                f"{resolved} is not inside the workspace {self.workspace}, so it "
                f"has no path inside the sandbox. The workspace is mounted at "
                f"{SANDBOX_WORKSPACE_PATH}."
            ) from exc
        # PurePosixPath, not Path: on Windows ``Path("/workspace") / "sub"``
        # yields ``\\workspace\\sub``, which is not a path inside the container.
        # The container's filesystem is POSIX whatever the host's is.
        return str(PurePosixPath(SANDBOX_WORKSPACE_PATH) / PurePosixPath(relative.as_posix()))

    def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: Path | str | None = None,
    ) -> ExecutionResult:
        """Run *command* in this container's shell and return what it produced.

        ``timeout`` is the caller's contract, not this module's: the /ask shell
        allows 120s and a /plan worker 30s, and neither is the Sandbox's to
        change. ``cwd`` is a *host* path inside the workspace and is mapped into
        the container.
        """
        if self._container is None:
            raise SandboxError(
                f"Sandbox {self.container_name} is not running. Call start() "
                f"before execute()."
            )
        if not command or not command.strip():
            raise SandboxError("No command provided")

        workdir = self.container_path(cwd) if cwd else SANDBOX_WORKSPACE_PATH

        # A shell, because the permission classifier approved shell syntax, not
        # an argv list. Running it through exec_run rather than `sh -c '...'` as a
        # single string means Docker does the argv quoting and the command text
        # reaches the shell exactly as written.
        argv = [INTERNAL_SHELL, "-c", command]

        def run() -> tuple[int, Any]:
            return self._container.exec_run(
                argv, workdir=workdir, demux=True, stdout=True, stderr=True
            )

        exit_code, output = self._await(run, command, timeout)
        stdout, stderr = output if isinstance(output, tuple) else (output, None)
        return ExecutionResult(
            exit_code=int(exit_code),
            stdout=_decode(stdout),
            stderr=_decode(stderr),
        )

    def _await(self, run, command: str, timeout: float | None) -> tuple[int, Any]:
        """Run *run*, bounded by *timeout* seconds if one was given.

        ``Container.exec_run`` has no timeout parameter, so the bound is applied
        here. What that can and cannot do is worth stating plainly: the caller
        stops waiting, and the error is raised - but Docker offers no way to kill
        an exec that is already running, so the process inside the container may
        carry on until it finishes on its own. It cannot reach the host beyond the
        mounted workspace, and ``stop()`` removes the container regardless.
        """
        if timeout is None:
            try:
                return run()
            except SandboxError:
                raise
            except Exception as exc:
                raise SandboxError(
                    f"Sandbox execution failed for: {command}: {exc}"
                ) from exc

        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["result"] = run()
            except BaseException as exc:  # surfaced on the calling thread
                box["error"] = exc

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            raise SandboxTimeout(
                f"Command exceeded {timeout:g}s and was abandoned: {command}"
            )
        if "error" in box:
            raise SandboxError(
                f"Sandbox execution failed for: {command}: {box['error']}"
            ) from box["error"]
        return box["result"]

    @staticmethod
    def _shell_quote(s: str) -> str:
        """Quote *s* so a shell reads it as one literal argument.

        Delegated to :mod:`shlex`, which is the quoting rules rather than a
        hand-rolled subset of them. Only for composing a shell command line -
        never for deciding what may run.
        """
        return shlex.quote(s)


def _decode(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


# ---------------------------------------------------------------------------
# Which sandbox the current execution uses
# ---------------------------------------------------------------------------

_current: contextvars.ContextVar[Sandbox | None] = contextvars.ContextVar(
    "terminus_sandbox", default=None
)
"""The Sandbox for the execution in force, or None when there is none.

A ContextVar for the same reason :mod:`terminus.permissions` uses one: an
execution is not a process. The CLI's session, a /plan task worker and a spawned
child each run as their own asyncio Task, and asyncio copies the context per Task,
so a sandbox installed for one cannot be observed or changed by a sibling. Tools
read it rather than constructing one, which is what keeps a container per session
instead of a container per command.
"""


def current_sandbox() -> Sandbox | None:
    """The Sandbox for the current execution, or None."""
    return _current.get()


def set_sandbox(sandbox: Sandbox | None):
    """Install *sandbox* for the current execution; returns the reset token."""
    return _current.set(sandbox)


@contextmanager
def sandbox_scope(sandbox: Sandbox | None):
    """Run a block with *sandbox* installed, then restore the previous one."""
    token = _current.set(sandbox)
    try:
        yield sandbox
    finally:
        _current.reset(token)


def sandbox_enabled() -> bool:
    """Whether commands should run in a Sandbox.

    Read from configuration so an operator can turn the boundary off
    deliberately. There is no automatic fallback: if this is on and Docker cannot
    be reached, commands fail rather than quietly running on the host.
    """
    from terminus.config import CONFIG

    sandbox = CONFIG.get("sandbox") or {}
    return bool(sandbox.get("enabled", True))


def sandbox_disabled() -> bool:
    return not sandbox_enabled()
