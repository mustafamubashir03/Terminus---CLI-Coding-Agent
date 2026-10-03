"""The Sandbox boundary, with Docker faked.

These are about the boundary's own contract - identity, container
configuration, lifecycle, the structured result, orphan selection, and the
absence of a host fallback. They do not need a daemon, so the suite never
requires Docker.

The real thing is exercised separately in ``test_sandbox_docker.py``, which skips
itself when Docker is absent.
"""

from __future__ import annotations

import inspect
import os

import pytest

from terminus.sandbox import (
    CONTAINER_NAME_PREFIX,
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
)
from terminus.sandbox import docker_sandbox as module


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeExecResult(tuple):
    """What ``exec_run`` returns: a pair, but with the fields the SDK exposes."""

    def __new__(cls, exit_code, output):
        return super().__new__(cls, (exit_code, output))

    @property
    def exit_code(self):
        return self[0]

    @property
    def output(self):
        return self[1]


class FakeContainer:
    """A container answering both the startup probe and later commands.

    The two are told apart by the command text, because one fake serving both
    from a single canned result either fails the workspace check or makes every
    later command return the probe's answer.
    """

    def __init__(
        self,
        *,
        status="running",
        exec_result=(0, (b"out", b"err")),
        probe_result=(0, (b"ok", b"")),
        attrs=None,
    ):
        self.id = "c0ffee"
        self.name = "sandbox"
        self.status = status
        self.attrs = attrs or {"State": {"ExitCode": 0}}
        self.execs: list[dict] = []
        self.removed = False
        self._exec_result = exec_result
        self._probe_result = probe_result
        self.labels: dict[str, str] = {}
        # Where the fake believes /workspace points. None models a container whose
        # mount never happened: the probe then finds an empty directory, which is
        # the failure the real check exists to catch.
        self.workspace = None

    def reload(self):
        return None

    def exec_run(self, cmd, workdir=None, demux=False, stdout=True, stderr=True):
        self.execs.append({"cmd": cmd, "workdir": workdir})
        joined = " ".join(str(part) for part in cmd)
        if "cat " in joined:
            return FakeExecResult(0, (self._read_mounted(joined).encode(), b""))
        result = self._probe_result if "probe" in joined else self._exec_result
        if isinstance(result, Exception):
            raise result
        return FakeExecResult(*result)

    def _read_mounted(self, joined: str) -> str:
        """Serve ``cat /workspace/<name>`` from the fake's idea of the host tree."""
        if self.workspace is None:
            return ""
        target = joined.split("cat ", 1)[1].strip().strip("'\"")
        name = target.rsplit("/", 1)[-1]
        candidate = self.workspace / name
        return candidate.read_text(encoding="utf-8") if candidate.exists() else ""

    def remove(self, force=False):
        self.removed = True


class FakeContainerCollection:
    def __init__(self, container=None, listed=None, run_error=None, workspace=None):
        self.container = container
        self.listed = listed or []
        self.filters = None
        self.workspace = workspace
        self.run_error = run_error
        self.run_kwargs: dict | None = None

    def run(self, image, **kwargs):
        if self.run_error is not None:
            raise self.run_error
        self.run_kwargs = dict(kwargs, image=image)
        if self.container is None:
            self.container = FakeContainer()
        self.container.name = kwargs.get("name", self.container.name)
        self.container.labels = dict(kwargs.get("labels") or {})
        self.container.workspace = self.workspace
        return self.container

    def get(self, name):
        return self.container

    def list(self, all=False, filters=None):
        """Filter by label the way the daemon does, so the test means something.

        A fake that returned everything regardless of ``filters`` would make
        "cleanup selects on the label" untestable - it would pass for the wrong
        reason.
        """
        self.filters = filters
        wanted = (filters or {}).get("label")
        if not wanted:
            return list(self.listed)
        key, _, value = wanted.partition("=")
        return [c for c in self.listed if c.labels.get(key) == value]


class FakeClient:
    def __init__(self, containers=None, list_error=None):
        self.containers = containers or FakeContainerCollection()
        self.list_error = list_error

    def _list(self, all=False, filters=None):
        if self.list_error is not None:
            raise self.list_error
        return self.containers.list(all=all, filters=filters)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    return tmp_path


def make_sandbox(tmp_path, container=None, listed=None, **kwargs):
    kwargs.setdefault("workspace", tmp_path)
    collection = FakeContainerCollection(container=container, listed=listed, **kwargs)
    return Sandbox(client=FakeClient(collection), workspace=tmp_path), collection


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_is_a_uuid_and_a_unique_container_name(tmp_path):
    first = Sandbox(client=FakeClient(), workspace=tmp_path)
    second = Sandbox(client=FakeClient(), workspace=tmp_path)
    assert first.sandbox_id != second.sandbox_id
    assert first.container_name.startswith(CONTAINER_NAME_PREFIX)
    assert first.container_name != second.container_name


def test_ownership_is_a_label_not_a_name_prefix(tmp_path):
    sandbox = Sandbox(client=FakeClient(), workspace=tmp_path)
    assert sandbox.labels[OWNERSHIP_LABEL_KEY] == OWNERSHIP_LABEL_VALUE
    assert sandbox.labels[f"{OWNERSHIP_LABEL_KEY}.id"] == str(sandbox.sandbox_id)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_container_is_configured_for_a_long_lived_session(tmp_path):
    sandbox, collection = make_sandbox(tmp_path)
    sandbox.start()
    kwargs = collection.run_kwargs
    assert kwargs["image"] == SANDBOX_IMAGE
    assert kwargs["name"] == sandbox.container_name
    assert kwargs["detach"] is True
    assert kwargs["tty"] is True
    assert kwargs["working_dir"] == SANDBOX_WORKSPACE_PATH
    assert kwargs["command"] == ["sleep", "infinity"]


def test_workspace_is_bind_mounted_and_nothing_else(tmp_path):
    sandbox, collection = make_sandbox(tmp_path)
    sandbox.start()
    volumes = collection.run_kwargs["volumes"]
    assert volumes == {
        str(tmp_path): {"bind": SANDBOX_WORKSPACE_PATH, "mode": "rw"}
    }
    # No home directory, no docker socket, no credentials.
    for mount in (str(volumes), str(collection.run_kwargs.get("environment"))):
        assert ".ssh" not in mount
        assert "docker.sock" not in mount
        assert "Terminus" not in mount


def test_container_gets_a_fixed_environment_not_the_host_s(tmp_path):
    sandbox, collection = make_sandbox(tmp_path)
    sandbox.start()
    environment = collection.run_kwargs["environment"]
    assert set(environment) == {"HOME", "TERM", "LANG"}
    assert "OPENROUTER_API_KEY" not in environment


def test_the_workspace_comes_from_the_workspace_abstraction(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    sandbox = Sandbox(client=FakeClient())
    assert sandbox.workspace == tmp_path


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_reports_running_after_a_container_that_exits_cleanly(tmp_path):
    sandbox, _ = make_sandbox(tmp_path)
    sandbox.start()
    assert sandbox.started is True
    assert sandbox.container_id == "c0ffee"


def test_start_does_not_replace_a_container_that_is_already_running(tmp_path):
    container = FakeContainer()
    sandbox, collection = make_sandbox(tmp_path, container=container)
    sandbox.start()
    sandbox.start()
    assert collection.run_kwargs is not None
    assert sandbox.container_id == "c0ffee"


def test_a_container_that_exits_during_startup_is_reported(tmp_path):
    container = FakeContainer(status="exited", attrs={"State": {"ExitCode": 127}})
    sandbox, _ = make_sandbox(tmp_path, container=container)
    with pytest.raises(SandboxUnavailable) as exc:
        sandbox.start()
    assert "exited during startup" in str(exc.value)
    assert "127" in str(exc.value)


def test_a_docker_error_during_start_is_surfaced(tmp_path):
    sandbox, _ = make_sandbox(tmp_path, run_error=RuntimeError("no such image"))
    with pytest.raises(SandboxUnavailable) as exc:
        sandbox.start()
    assert "no such image" in str(exc.value)


def test_a_container_that_never_starts_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CONTAINER_STARTUP_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    sandbox, _ = make_sandbox(tmp_path, container=FakeContainer(status="created"))
    with pytest.raises(SandboxTimeout) as exc:
        sandbox.start()
    assert "still 'created'" in str(exc.value)


def test_stop_removes_the_container_and_is_repeatable(tmp_path):
    container = FakeContainer()
    sandbox, _ = make_sandbox(tmp_path, container=container)
    sandbox.start()
    sandbox.stop()
    assert container.removed is True
    assert sandbox.started is False
    sandbox.stop()  # must not raise


def test_the_workspace_mount_is_verified_at_startup(tmp_path):
    """A healthy container whose /workspace is not the host tree is refused.

    The container here is running and answering every command; what it does not
    have is the host workspace. ``workspace=None`` is how the fake models a mount
    that never happened, so ``/workspace`` is an empty directory - which exists,
    and is a directory, and is the reason a check that only asks "is /workspace a
    directory?" passes against the one container it must never pass.
    """
    sandbox, _ = make_sandbox(tmp_path, workspace=None)
    with pytest.raises(SandboxUnavailable) as exc:
        sandbox.start()
    assert "not the host workspace" in str(exc.value)
    assert sandbox.started is False
    assert not list(tmp_path.glob(".terminus-sandbox-probe-*")), (
        "the probe file is the workspace's own file; it must not be left behind"
    )


def test_the_mount_check_reads_the_token_back_rather_than_asking_for_a_directory(
    tmp_path,
):
    """The check has to be able to fail, which "is /workspace a dir?" cannot.

    Pins the specific mistake: the image creates /workspace at build time, so a
    directory test is answered by the image whether or not a bind mount exists.
    """
    sandbox, collection = make_sandbox(tmp_path)
    sandbox.start()
    probe = [
        call["cmd"]
        for call in collection.container.execs
        if "cat " in " ".join(str(part) for part in call["cmd"])
    ]
    assert probe, "startup must read a host-written token back out of /workspace"
    assert "test -d" not in " ".join(str(p) for p in probe[0])


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _started(tmp_path, exec_result=(0, (b"out", b"err"))):
    container = FakeContainer(exec_result=exec_result)
    sandbox, _ = make_sandbox(tmp_path, container=container)
    sandbox.start()
    return sandbox, container


def test_the_command_runs_in_the_container_through_a_shell(tmp_path):
    sandbox, container = _started(tmp_path)
    sandbox.execute("pytest -q")
    assert container.execs[-1]["cmd"][:2] == [INTERNAL_SHELL, "-c"]
    assert container.execs[-1]["cmd"][2] == "pytest -q"


def test_exit_code_stdout_and_stderr_are_all_returned(tmp_path):
    sandbox, _ = _started(tmp_path, (3, (b"o", b"e")))
    result = sandbox.execute("false")
    assert isinstance(result, ExecutionResult)
    assert (result.exit_code, result.stdout, result.stderr) == (3, "o", "e")


def test_the_result_unpacks_as_a_three_tuple(tmp_path):
    sandbox, _ = _started(tmp_path)
    exit_code, stdout, stderr = sandbox.execute("echo hi")
    assert exit_code == 0
    assert stdout == "out"
    assert stderr == "err"


def test_bytes_and_none_are_decoded(tmp_path):
    sandbox, _ = _started(tmp_path, (0, (b"x", None)))
    result = sandbox.execute("echo x")
    assert result.stdout == "x"
    assert result.stderr == ""


def test_a_working_directory_is_mapped_into_the_container(tmp_path):
    (tmp_path / "sub").mkdir()
    sandbox, container = _started(tmp_path)
    sandbox.execute("pwd", cwd=tmp_path / "sub")
    assert container.execs[-1]["workdir"] == f"{SANDBOX_WORKSPACE_PATH}/sub"


def test_a_working_directory_outside_the_workspace_is_refused(tmp_path):
    sandbox, _ = _started(tmp_path)
    with pytest.raises(SandboxError) as exc:
        sandbox.execute("pwd", cwd=tmp_path.parent / "elsewhere")
    assert "not inside the workspace" in str(exc.value)


def test_several_commands_share_one_container(tmp_path):
    sandbox, container = _started(tmp_path)
    for command in ("one", "two", "three"):
        sandbox.execute(command)
    assert sandbox.container_id == "c0ffee"
    assert container.removed is False
    assert [e["cmd"][2] for e in container.execs[-3:]] == ["one", "two", "three"]


def test_a_timeout_is_the_callers_not_the_sandboxs(tmp_path):
    import threading

    sandbox, container = _started(tmp_path)

    def slow(*_a, **_k):
        threading.Event().wait(5)

    container.exec_run = slow
    with pytest.raises(SandboxTimeout) as exc:
        sandbox.execute("sleep 100", timeout=0.2)
    assert "0.2s" in str(exc.value)


def test_without_a_timeout_the_caller_waits(tmp_path):
    sandbox, _ = _started(tmp_path)
    assert sandbox.execute("quick").exit_code == 0


def test_a_docker_exec_failure_is_surfaced_not_swallowed(tmp_path):
    """A failure inside the container is reported, never quietly retried."""
    container = FakeContainer()
    sandbox, _ = make_sandbox(tmp_path, container=container)
    sandbox.start()

    def boom(*_a, **_k):
        raise RuntimeError("daemon gone")

    container.exec_run = boom
    with pytest.raises(SandboxError) as exc:
        sandbox.execute("ls")
    assert "daemon gone" in str(exc.value)


def test_execute_before_start_is_an_error(tmp_path):
    sandbox = Sandbox(client=FakeClient(), workspace=tmp_path)
    with pytest.raises(SandboxError) as exc:
        sandbox.execute("ls")
    assert "not running" in str(exc.value)


def test_execute_rejects_an_empty_command(tmp_path):
    sandbox, _ = _started(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.execute("   ")


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------


def test_cleanup_removes_a_stopped_terminus_container(tmp_path):
    orphan = FakeContainer()
    orphan.name = "terminus-sandbox-old"
    orphan.status = "exited"
    orphan.labels = {OWNERSHIP_LABEL_KEY: OWNERSHIP_LABEL_VALUE}
    sandbox, collection = make_sandbox(tmp_path, listed=[orphan])
    assert sandbox._cleanup_orphans() == 1
    assert orphan.removed is True
    assert collection.filters == {
        "label": f"{OWNERSHIP_LABEL_KEY}={OWNERSHIP_LABEL_VALUE}"
    }


def test_cleanup_preserves_an_unrelated_container(tmp_path):
    """Selection is on the ownership label, so a user's container is invisible."""
    mine = FakeContainer()
    mine.name = "my-database"
    mine.status = "exited"
    mine.labels = {"com.example.someone-else": "true"}
    sandbox, _ = make_sandbox(tmp_path, listed=[mine])
    assert sandbox._cleanup_orphans() == 0
    assert mine.removed is False


def test_cleanup_leaves_a_running_container_that_may_own_another_session(tmp_path):
    theirs = FakeContainer()
    theirs.name = "terminus-sandbox-theirs"
    theirs.status = "running"
    theirs.labels = {OWNERSHIP_LABEL_KEY: OWNERSHIP_LABEL_VALUE}
    sandbox, _ = make_sandbox(tmp_path, listed=[theirs])
    assert sandbox._cleanup_orphans() == 0
    assert theirs.removed is False


def test_cleanup_never_removes_its_own_container(tmp_path):
    mine = FakeContainer()
    mine.status = "exited"
    mine.labels = {OWNERSHIP_LABEL_KEY: OWNERSHIP_LABEL_VALUE}
    sandbox, _ = make_sandbox(tmp_path, listed=[mine])
    mine.name = sandbox.container_name
    assert sandbox._cleanup_orphans() == 0
    assert mine.removed is False


def test_cleanup_with_no_containers_is_a_no_op(tmp_path):
    sandbox, _ = make_sandbox(tmp_path, listed=[])
    assert sandbox._cleanup_orphans() == 0


def test_a_docker_error_during_cleanup_does_not_stop_startup(tmp_path):
    client = FakeClient(list_error=RuntimeError("api down"))
    sandbox = Sandbox(client=client, workspace=tmp_path)
    assert sandbox._cleanup_orphans() == 0


# ---------------------------------------------------------------------------
# Shell quoting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [
    "plain",
    "with space",
    "semi;colon && and || or",
    "$(whoami)",
    "`backticks`",
    "quote'single",
    'quote"double',
    "new\nline",
    "",
])
def test_shell_quote_round_trips_through_a_real_shell(value):
    """Not a hand-rolled subset: ask sh what the quoting actually produced."""
    import subprocess

    quoted = Sandbox._shell_quote(value)
    if os.name == "nt":
        pytest.skip("no POSIX shell available to verify quoting")
    result = subprocess.run(
        ["/bin/sh", "-c", f"printf %s {quoted}"],
        capture_output=True, text=True, check=False,
    )
    assert result.stdout == value


def test_shell_quote_neutralises_injection():
    """The text survives intact, inside quotes - which is the whole point.

    shlex.quote does not remove the semicolon; it makes the shell read the whole
    thing as one argument. Asserting the character is gone would be asserting a
    weaker and wrong guarantee.
    """
    import shlex

    hostile = "x; rm -rf /"
    quoted = Sandbox._shell_quote(hostile)
    assert quoted.startswith("'") and quoted.endswith("'")
    assert shlex.split(quoted) == [hostile]


# ---------------------------------------------------------------------------
# The invariant that matters
# ---------------------------------------------------------------------------


def test_a_sandbox_failure_never_falls_back_to_the_host():
    """No host process can be reached from the sandbox module.

    Checked against the parsed module rather than its text, so the prose is free
    to explain that subprocess is deliberately absent.
    """
    import ast

    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("subprocess", "pty"):
        assert forbidden not in imported, (
            f"the sandbox module imports {forbidden}: a Docker failure has to "
            "surface, not re-run the command on the host"
        )


def test_the_shell_tools_choose_the_sandbox_and_do_not_build_one(tmp_path):
    """One container per session: the tools read the current one."""
    import inspect

    from terminus.tools import shell_tools, terminal_tools

    for module_under_test in (shell_tools, terminal_tools):
        source = inspect.getsource(module_under_test)
        assert "current_sandbox()" in source, (
            f"{module_under_test.__name__} does not read the session sandbox"
        )
        assert "Sandbox(" not in source, (
            f"{module_under_test.__name__} constructs a Sandbox; that would make a "
            "container per command"
        )


def test_the_sandbox_is_installed_per_execution_not_globally(tmp_path):
    import asyncio

    from terminus.sandbox import current_sandbox, sandbox_scope, set_sandbox

    async def scenario():
        set_sandbox(None)
        assert current_sandbox() is None
        outer = Sandbox(client=FakeClient(), workspace=tmp_path)
        with sandbox_scope(outer):
            assert current_sandbox() is outer

            async def sibling():
                return current_sandbox()

            # A child task copies the context, so it inherits rather than shares
            # mutable global state.
            assert await asyncio.create_task(sibling()) is outer
        assert current_sandbox() is None

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# The boundary itself: no silent fall back to the host
# ---------------------------------------------------------------------------


class ExplodingSandbox:
    """A container that fails the way a broken Docker daemon does."""

    sandbox_id = "broken"
    container_name = "terminus-sandbox-broken"

    def __init__(self, message="daemon gone"):
        self.message = message
        self.commands: list[str] = []

    def execute(self, command, timeout=None, cwd=None):
        self.commands.append(command)
        raise SandboxUnavailable(self.message)


def _allow_read_only():
    """The real decision object, so the tests exercise the real formatting path."""
    from terminus.permissions import PermissionDecision, PermissionLevel

    return PermissionDecision(
        allowed=True,
        level=PermissionLevel.READ_ONLY,
        reason="test",
        requires_approval=False,
    )


def _host_is_unreachable(monkeypatch, module_under_test):
    """Make the host path fail the test if anything reaches for it."""
    def forbidden(*_a, **_k):
        raise AssertionError(
            f"{module_under_test.__name__} fell back to the host"
        )

    monkeypatch.setattr(module_under_test.subprocess, "run", forbidden)
    monkeypatch.setattr(module_under_test.subprocess, "Popen", forbidden)


def test_a_failed_sandbox_is_never_re_run_on_the_host(tmp_path, monkeypatch):
    """The central guarantee, on both tools, with the host path booby-trapped."""
    from terminus.sandbox import sandbox_scope
    from terminus.tools import shell_tools, terminal_tools

    sandbox = ExplodingSandbox()
    for module_under_test in (shell_tools, terminal_tools):
        _host_is_unreachable(monkeypatch, module_under_test)
        with sandbox_scope(sandbox):
            if module_under_test is shell_tools:
                out = module_under_test._execute("echo hi", tmp_path, _allow_read_only())
            else:
                out = module_under_test._execute("echo hi", tmp_path)
        assert "daemon gone" in out, f"{module_under_test.__name__} hid the failure"
        assert "timed out" not in out.lower(), (
            f"{module_under_test.__name__} reported a Docker failure as a timeout, "
            "which is a claim about a command that never ran"
        )


def test_a_failed_sandbox_does_not_leak_a_command_to_the_host(tmp_path, monkeypatch):
    """No host process is even started, as distinct from starting one and failing."""
    from terminus.sandbox import sandbox_scope
    from terminus.tools import shell_tools

    started = []
    monkeypatch.setattr(
        shell_tools.subprocess, "Popen", lambda *a, **k: started.append(a)
    )
    with sandbox_scope(ExplodingSandbox()):
        shell_tools._execute("echo hi", tmp_path, _allow_read_only())
    assert started == [], "the host process was started before the sandbox failed"


def test_session_startup_refuses_to_run_unsandboxed(monkeypatch):
    """An enabled boundary that will not start must end the session.

    The alternative - log a warning, leave the sandbox unset, and carry on -
    produces a session that believes it is sandboxed while every command runs on
    the host. That is the failure this test exists to prevent, so it is asserted
    at the point where the choice is made.
    """
    import asyncio

    from terminus import cli
    from terminus.sandbox import SandboxError

    class Unstartable:
        container_name = "terminus-sandbox-unstartable"

        def start(self):
            raise SandboxUnavailable("no docker daemon")

    monkeypatch.setattr(cli, "sandbox_enabled", lambda: True, raising=False)
    monkeypatch.setattr("terminus.sandbox.sandbox_enabled", lambda: True)
    monkeypatch.setattr("terminus.sandbox.Sandbox", lambda **_k: Unstartable())

    with pytest.raises(SandboxError):
        asyncio.run(cli.start_sandbox())


def test_session_startup_is_a_choice_when_the_boundary_is_off(monkeypatch):
    """Disabling the boundary is how you run without Docker; it is not a failure."""
    import asyncio

    from terminus import cli

    monkeypatch.setattr("terminus.sandbox.sandbox_enabled", lambda: False)
    assert asyncio.run(cli.start_sandbox()) is None


def test_a_host_path_failure_is_still_reported_as_a_command_problem(
    tmp_path, monkeypatch
):
    """The boundary being off must not have quietly broken the pre-existing path."""
    import subprocess as real_subprocess

    from terminus.tools import terminal_tools

    monkeypatch.setattr(
        terminal_tools.subprocess,
        "run",
        lambda *a, **k: real_subprocess.CompletedProcess(
            args=a[0] if a else "", returncode=0, stdout="Python 3.12.0", stderr=""
        ),
    )
    out = terminal_tools._execute("python --version", None)
    assert "Python 3.12.0" in out


def test_the_sandbox_reaches_a_child_task_the_way_a_plan_worker_runs(tmp_path):
    """The /plan workers run in child tasks; they must inherit the container.

    A ContextVar that was only ever set on the REPL's own task would leave every
    worker running on the host while the session reported a sandbox - the same
    silent downgrade this boundary is meant to remove, one task boundary away.
    """
    import asyncio

    from terminus.sandbox import current_sandbox, sandbox_scope

    async def scenario():
        sandbox = Sandbox(client=FakeClient(), workspace=tmp_path)
        with sandbox_scope(sandbox):

            async def worker():
                return current_sandbox()

            found = await asyncio.gather(worker(), worker(), worker())
        assert found == [sandbox, sandbox, sandbox]

    asyncio.run(scenario())


def test_the_configured_image_is_the_one_the_code_pins():
    """config.yaml's default and the module constant must not drift.

    start_sandbox reads CONFIG, not the constant, so a stale default quietly
    overrides the pinned tag and every session runs whatever `latest` happens to
    be - the exact thing pinning the tag was for.
    """
    from terminus.config import CONFIG
    from terminus.sandbox import SANDBOX_IMAGE

    assert CONFIG["sandbox"]["image"] == SANDBOX_IMAGE
    assert CONFIG["sandbox"]["image"] != "terminus-sandbox:latest", (
        "latest changes under a session without anyone rebuilding anything"
    )
