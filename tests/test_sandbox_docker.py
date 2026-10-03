"""Integration tests against a real Docker daemon.

Skipped rather than failed when Docker is unavailable, so the suite is honest on
a machine without it instead of pretending the boundary is verified. Nothing here
is mocked: the point is the thing the unit tests cannot reach - whether a command
really runs in a container, whether the workspace is really shared, and whether a
file written in there is really readable on the host afterwards.

The image is required to exist. Building it takes minutes and is not a test's
job, so these tests skip with the build command rather than building silently.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from terminus.sandbox import (
    SANDBOX_IMAGE,
    SANDBOX_WORKSPACE_PATH,
    Sandbox,
    SandboxError,
    SandboxUnavailable,
)


def _docker_is_usable() -> tuple[bool, str]:
    try:
        import docker
    except ImportError:
        return False, "the docker SDK is not installed"
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # daemon down, no Desktop, no socket
        return False, f"no reachable Docker daemon ({type(exc).__name__}: {exc})"
    return True, ""


_AVAILABLE, _REASON = _docker_is_usable()
IMAGE = SANDBOX_IMAGE  # the one tag, not a second copy of it

pytestmark = [
    pytest.mark.skipif(not _AVAILABLE, reason=_REASON or "Docker unavailable"),
    pytest.mark.skipif(
        not shutil.which("docker"), reason="docker CLI needed to check the image"
    ),
]


def _image_exists() -> bool:
    import docker

    try:
        docker.from_env().images.get(IMAGE)
    except Exception:
        return False
    return True


pytestmark.append(
    pytest.mark.skipif(
        not _image_exists(),
        reason=(
            f"{IMAGE} is not built. Build it from the repository root with: "
            f"docker build -f src/terminus/sandbox/Dockerfile -t {IMAGE} ."
        ),
    )
)


@pytest.fixture
def workspace():
    """A workspace inside the repository, removed however the test ends.

    Deliberately not tmp_path. On Windows ``tempfile.mkdtemp`` applies mode
    0700, which becomes an owner-only ACL, and Docker's file sharing cannot write
    through it - so a sandbox test run against a temp directory fails on a host
    restriction that has nothing to do with the sandbox. The repository is where
    these containers are meant to work.

    The cleanup lives here rather than in the ``sandbox`` fixture because several
    tests use the workspace without one. When it lived there, a test that built
    its own container left its files behind in the checkout.
    """
    probe = Path.cwd() / ".sandbox-integration"
    shutil.rmtree(probe, ignore_errors=True)
    probe.mkdir(parents=True, exist_ok=True)
    try:
        yield probe
    finally:
        shutil.rmtree(probe, ignore_errors=True)


@pytest.fixture
def sandbox(workspace: Path):
    box = Sandbox(image=IMAGE, workspace=workspace)
    box.start()
    try:
        yield box
    finally:
        box.stop()


def test_the_container_is_running_and_is_not_root(sandbox: Sandbox):
    """The image runs unprivileged, which is half of why it is safe to bind-mount."""
    assert sandbox.started is True
    whoami = sandbox.execute("whoami").stdout.strip()
    assert whoami != "root", f"the sandbox is running as {whoami}"
    uid = sandbox.execute("id -u").stdout.strip()
    assert uid.isdigit() and int(uid) != 0


def test_a_command_really_runs_in_the_container(sandbox: Sandbox):
    """A marker file in the container's root filesystem proves where it ran."""
    result = sandbox.execute("touch /tmp/inside-the-container && ls /tmp/inside-the-container")
    assert result.exit_code == 0
    assert "inside-the-container" in result.stdout


def test_the_workspace_is_really_shared(sandbox: Sandbox, workspace: Path):
    """Written on the host, read in the container: the mount is bidirectional."""
    (workspace / "from-host.txt").write_text("written on the host\n", encoding="utf-8")
    result = sandbox.execute(f"cat {SANDBOX_WORKSPACE_PATH}/from-host.txt")
    assert result.exit_code == 0
    assert "written on the host" in result.stdout


def test_a_file_written_in_the_container_is_readable_on_the_host(
    sandbox: Sandbox, workspace: Path
):
    """The round trip that a bind mount exists for, and the one Windows breaks.

    A container whose writes the host cannot read would leave an agent editing
    files Terminus then cannot see - the failure is silent from the container's
    side, which is what makes it worth asserting from the host's side.
    """
    result = sandbox.execute(
        f"printf 'written in the container\\n' > {SANDBOX_WORKSPACE_PATH}/from-container.txt"
    )
    assert result.exit_code == 0
    assert (workspace / "from-container.txt").read_text(
        encoding="utf-8"
    ) == "written in the container\n"


def test_a_file_created_by_the_container_can_be_appended_to_by_the_host(
    sandbox: Sandbox, workspace: Path
):
    """Permissions must work in both directions, not just host-to-container."""
    sandbox.execute(f"printf 'first\\n' > {SANDBOX_WORKSPACE_PATH}/shared.txt")
    (workspace / "shared.txt").open("a", encoding="utf-8").write("second\n")
    result = sandbox.execute(f"cat {SANDBOX_WORKSPACE_PATH}/shared.txt")
    assert result.stdout.split() == ["first", "second"]


def test_a_directory_created_in_the_container_is_usable_from_the_host(
    sandbox: Sandbox, workspace: Path
):
    result = sandbox.execute(
        f"mkdir -p {SANDBOX_WORKSPACE_PATH}/nested/deep "
        f"&& printf 'deep\\n' > {SANDBOX_WORKSPACE_PATH}/nested/deep/file.txt"
    )
    assert result.exit_code == 0
    assert (workspace / "nested" / "deep" / "file.txt").exists()


def test_the_workspace_path_maps_and_a_path_outside_it_is_refused(
    sandbox: Sandbox, workspace: Path
):
    """The model's reach stops at the workspace, as it does on the host."""
    # Its own directory. Relying on another test to have created this makes the
    # test depend on execution order, and it fails confusingly with a Docker
    # chdir error rather than a missing-directory one.
    (workspace / "nested").mkdir(exist_ok=True)

    inside = sandbox.execute("pwd", cwd=workspace / "nested")
    assert inside.exit_code == 0
    assert inside.stdout.strip() == f"{SANDBOX_WORKSPACE_PATH}/nested"

    with pytest.raises(SandboxError) as exc:
        sandbox.execute("pwd", cwd=workspace.parent)
    assert "not inside the workspace" in str(exc.value)


def test_stdout_stderr_and_exit_codes_come_back_separately(sandbox: Sandbox):
    result = sandbox.execute(
        "printf 'to stdout\\n'; printf 'to stderr\\n' >&2; exit 3"
    )
    assert result.exit_code == 3
    assert "to stdout" in result.stdout
    assert "to stderr" in result.stderr
    assert "to stderr" not in result.stdout


def test_a_failing_command_reports_its_own_output_not_a_sandbox_failure(
    sandbox: Sandbox,
):
    result = sandbox.execute("grep -q nothing-here /etc/hostname || echo absent")
    assert result.exit_code == 0
    assert "absent" in result.stdout


def test_the_promised_tools_are_actually_in_the_image(sandbox: Sandbox):
    """If a tool is advertised to an agent it had better be there."""
    probes = {
        "bash": "bash --version",
        "git": "git --version",
        "python": "python --version",
        "node": "node --version",
        "npm": "npm --version",
        "rg": "rg --version",
        "jq": "jq --version",
        "curl": "curl --version",
        "grep": "grep --version",
        "pytest": "python -m pytest --version",
    }
    missing = [
        name
        for name, command in probes.items()
        if sandbox.execute(command).exit_code != 0
    ]
    assert not missing, f"missing from {IMAGE}: {', '.join(missing)}"


def test_pytest_runs_a_projects_tests_in_the_sandbox(workspace: Path):
    """The reason the image ships pytest: an agent can run tests where it works.

    Scoped to what is actually true. The image carries a toolchain, not this
    repository's dependencies - installing langgraph and the rest into every
    disposable container is the opposite of the image's design - so running
    *this* project's suite in the bare image fails importing conftest. What
    matters is that a project checkout on the mount can be tested: pytest is
    there, it reads the config from the mount, and its output and exit code come
    back. An agent installs the dependencies the project pins, in the container,
    per session.
    """
    (workspace / "test_sample.py").write_text(
        "def test_arithmetic():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )
    box = Sandbox(image=IMAGE, workspace=workspace)
    box.start()
    try:
        passing = box.execute(
            f"python -m pytest {SANDBOX_WORKSPACE_PATH}/test_sample.py -q --no-header",
            timeout=300,
        )
        assert passing.exit_code == 0, passing.stdout[-1500:] + passing.stderr[-800:]

        (workspace / "test_failing.py").write_text(
            "def test_broken():\n    assert False, 'deliberate'\n", encoding="utf-8"
        )
        failing = box.execute(
            f"python -m pytest {SANDBOX_WORKSPACE_PATH}/test_failing.py -q --no-header",
            timeout=300,
        )
    finally:
        box.stop()

    assert failing.exit_code != 0, "a failing suite must not report success"
    assert "deliberate" in failing.stdout, (
        f"the failure reason never reached the host: {failing.stdout[-1500:]}"
    )


def test_an_agent_can_install_what_the_project_needs(workspace: Path):
    """Dependencies are the agent's to install, per session, in the container.

    The image is not network-isolated on purpose - that is a policy question for a
    later layer - so this checks the thing a session depends on: a package can be
    installed and then imported, without touching the host.
    """
    box = Sandbox(image=IMAGE, workspace=workspace)
    box.start()
    try:
        install = box.execute("pip install --quiet requests", timeout=300)
        assert install.exit_code == 0, install.stderr[-1500:]
        imported = box.execute("python -c \"import requests; print(requests.__version__)\"")
    finally:
        box.stop()
    assert imported.exit_code == 0, imported.stderr[-800:]
    assert imported.stdout.strip(), "the installed package did not report a version"
    assert not (workspace / "site-packages").exists(), (
        "the install leaked out of the container into the host workspace"
    )


def test_one_container_serves_every_command(sandbox: Sandbox, workspace: Path):
    """Session state survives between commands, which is the point of one container.

    A file written by one command has to still be there for the next one; that
    only holds if they share a container.
    """
    (workspace / "kept.txt").write_text("first\n", encoding="utf-8")
    assert sandbox.execute("printf 'second\\n' >> kept.txt").exit_code == 0
    result = sandbox.execute("cat kept.txt")
    assert result.stdout.split() == ["first", "second"]


def test_a_timeout_is_reported_and_does_not_hang_the_session(sandbox: Sandbox):
    from terminus.sandbox import SandboxTimeout

    with pytest.raises(SandboxTimeout):
        sandbox.execute("sleep 30", timeout=1)


def test_the_container_is_removed_by_stop(sandbox: Sandbox):
    name = sandbox.container_name
    import docker

    client = docker.from_env()
    assert client.containers.get(name) is not None
    sandbox.stop()
    assert sandbox.started is False
    with pytest.raises(docker.errors.NotFound):
        client.containers.get(name)


def test_a_host_path_is_not_mounted(sandbox: Sandbox):
    """Nothing outside the workspace crosses the boundary.

    The home directory is the one that matters: it holds SSH keys, tokens and
    cloud credentials, and a bind-mounted home makes all of them readable by any
    command the model writes.
    """
    for path in ("/root", "/home/agent/.ssh", "/var/run/docker.sock"):
        result = sandbox.execute(f"test -e {path} && echo present || echo absent")
        if result.stdout.strip() == "present" and path == "/var/run/docker.sock":
            pytest.fail("the Docker socket is mounted into the sandbox")
    result = sandbox.execute("ls /root/.ssh 2>/dev/null | wc -l")
    assert result.stdout.strip() == "0", "host credentials are visible in the sandbox"


def test_a_second_start_does_not_replace_a_running_container(sandbox: Sandbox):
    """Replacing it would throw away everything the session had built."""
    sandbox.execute("printf 'state\\n' > /tmp/session-state.txt")
    sandbox.start()
    result = sandbox.execute("cat /tmp/session-state.txt")
    assert result.stdout.strip() == "state"


def test_a_sandbox_over_a_missing_image_is_reported_clearly(workspace: Path):
    box = Sandbox(image="terminus-sandbox:definitely-not-built", workspace=workspace)
    with pytest.raises(SandboxUnavailable) as exc:
        box.start()
    assert "Could not start a sandbox from image" in str(exc.value)
