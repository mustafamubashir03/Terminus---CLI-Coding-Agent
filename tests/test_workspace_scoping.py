"""Workspace scoping: the invariant, and every way a path can try to break it.

    An agent operates inside a workspace.
    A workspace may be shared by multiple agents and sessions.
    Every filesystem path is resolved relative to that workspace.
    Host paths outside the workspace cannot be reached through filesystem tools.

The boundary is enforced in :mod:`terminus.workspace`. These tests drive it
through the tools, because a rule nothing exercises is a rule nobody should rely
on, and through a real agent in the observability suite next door.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest

from terminus.permissions import (
    PermissionLevel,
    PermissionPolicy,
    get_permission_policy,
    set_permission_policy,
)
from terminus.tools import registry
from terminus.tools.filesystem_tools import (
    append_file,
    delete_file,
    edit_file,
    file_exists,
    grep,
    list_directory,
    read_file,
    write_file,
)
from terminus.tools.shell_tools import run_command
from terminus.tools.terminal_tools import run_command as run_plan_shell
from terminus.tools.terminal_tools import run_in_directory
from terminus.workspace import (
    WORKSPACE_ENV_VAR,
    WorkspaceViolation,
    is_within,
    project_root,
    relative_to_workspace,
    resolve_in_workspace,
)

PERMISSIVE = PermissionPolicy(
    auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
    approver=None,
    deny_levels=(),
)


@pytest.fixture(autouse=True)
def writable(tmp_path, monkeypatch):
    """The workspace is tmp_path, and every permission level is pre-authorised.

    Both matter: authorisation has to be irrelevant to the assertions below, so
    that a refusal can only have come from containment.
    """
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    previous = get_permission_policy()
    set_permission_policy(PERMISSIVE)
    yield tmp_path
    set_permission_policy(previous)


_outside_counter = itertools.count()


def outside_dir(tmp_path: Path) -> Path:
    """A directory beside the workspace, unique per call.

    tmp_path.parent is shared by every test in a run, so a fixed sibling name
    would let one test's file become another test's evidence.
    """
    sibling = tmp_path.parent / f"not-the-workspace-{next(_outside_counter)}"
    sibling.mkdir(parents=True, exist_ok=True)
    return sibling


# ---------------------------------------------------------------------------
# the read / write / list / delete surface
# ---------------------------------------------------------------------------

def test_write_read_round_trip_inside_the_workspace(tmp_path):
    assert "written successfully: a.txt" in write_file.invoke(
        {"file_path": "a.txt", "content": "hello"}
    )
    assert read_file.invoke({"file_path": "a.txt"}) == "hello"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello"


def test_write_file_creates_parent_directories(tmp_path):
    """mkdir is part of write_file: the model creates a tree by writing into it."""
    out = write_file.invoke({"file_path": "x/y/z/deep.txt", "content": "d"})
    assert "written successfully: x/y/z/deep.txt" in out
    assert (tmp_path / "x" / "y" / "z" / "deep.txt").is_file()
    assert "x" in list_directory.invoke({"directory": "x/y/z"})


def test_edit_appends_and_delete_inside_the_workspace(tmp_path):
    (tmp_path / "f.txt").write_text("one\ntwo\n", encoding="utf-8")
    assert "Replaced 1 occurrence" in edit_file.invoke(
        {"file_path": "f.txt", "old_text": "two", "new_text": "TWO"}
    )
    assert "appended successfully" in append_file.invoke(
        {"file_path": "f.txt", "content": "three\n"}
    )
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE,
                      PermissionLevel.DESTRUCTIVE),
        approver=None,
        deny_levels=(),
    ))
    assert "deleted successfully: f.txt" in delete_file.invoke({"file_path": "f.txt"})
    assert not (tmp_path / "f.txt").exists()


def test_paths_are_reported_workspace_relative(tmp_path):
    """The model wrote a relative path, so results name a relative path."""
    out = write_file.invoke({"file_path": "sub/rel.txt", "content": "x"})
    assert out.endswith("sub/rel.txt")
    assert str(tmp_path) not in out
    assert file_exists.invoke({"file_path": "sub/rel.txt"}) == (
        "File exists: sub/rel.txt"
    )


def test_an_absolute_path_inside_the_workspace_still_works(tmp_path):
    out = write_file.invoke({"file_path": str(tmp_path / "abs.txt"), "content": "x"})
    assert "written successfully: abs.txt" in out


def test_nonexistent_paths_are_reported_not_crashed(tmp_path):
    assert "not found" in read_file.invoke({"file_path": "no/such/file.py"})
    assert "Directory not found" in list_directory.invoke({"directory": "nope"})
    assert "does not exist" in file_exists.invoke({"file_path": "nope.py"})
    assert "Path not found" in grep.invoke({"pattern": "x", "path": "nope"})


def test_a_directory_is_not_a_file(tmp_path):
    (tmp_path / "adir").mkdir()
    (tmp_path / "plain.txt").write_text("x", encoding="utf-8")
    # reading a directory is refused by the OS on Windows and EISDIR elsewhere;
    # either way it is reported, never mistaken for file contents
    assert "adir" in read_file.invoke({"file_path": "adir"})
    assert "not a directory" in list_directory.invoke({"directory": "plain.txt"})
    assert "is a directory" in edit_file.invoke(
        {"file_path": "adir", "old_text": "a", "new_text": "b"}
    )


def test_a_blank_path_is_a_result_not_a_refusal(tmp_path):
    """A malformed call the model can fix immediately comes back as text."""
    assert write_file.invoke({"file_path": "   ", "content": "x"}) == (
        "No file path provided"
    )
    assert list_directory.invoke({"directory": ""}) == "No directory provided"


def test_workspace_root_itself_is_reachable(tmp_path):
    """``list_directory(".")`` is how the model looks around; it must work."""
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    assert "marker.txt" in list_directory.invoke({"directory": "."})
    assert "marker.txt" in grep.invoke({"pattern": "x", "path": "."})


def test_large_and_binary_files_are_handled(tmp_path):
    from terminus.tools.filesystem_tools import _MAX_FILE_SIZE_BYTES

    big = tmp_path / "big.bin"
    big.write_bytes(b"\x00" * (_MAX_FILE_SIZE_BYTES + 1024))
    assert "too large" in read_file.invoke({"file_path": "big.bin"})

    small = tmp_path / "latin.txt"
    small.write_bytes(b"value = 'caf\xe9'\n")
    text = read_file.invoke({"file_path": "latin.txt"})
    assert "not valid UTF-8" in text
    assert "value" in text


def test_write_file_leaves_no_temp_file_behind(tmp_path):
    write_file.invoke({"file_path": "atomic.txt", "content": "x"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["atomic.txt"]


# ---------------------------------------------------------------------------
# the boundary: everything below must be refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "../escape.txt",
    "sub/../../escape.txt",
    "./sub/../../../escape.txt",
    "..\\..\\escape.txt",
    "sub\\..\\..\\escape.txt",
])
def test_traversal_is_refused(tmp_path, raw):
    out = write_file.invoke({"file_path": raw, "content": "x"})
    assert "outside the workspace" in out
    assert not (tmp_path.parent / "escape.txt").exists()


def test_an_absolute_path_outside_the_workspace_is_refused(tmp_path):
    outside = outside_dir(tmp_path)
    out = write_file.invoke({"file_path": str(outside / "abs.txt"), "content": "x"})
    assert "outside the workspace" in out
    assert list(outside.iterdir()) == []


def test_a_drive_rooted_path_is_refused(tmp_path):
    out = write_file.invoke({"file_path": "C:/Windows/System32/drivers/etc/hosts",
                             "content": "x"})
    assert "outside the workspace" in out


def test_a_unc_path_is_refused(tmp_path):
    out = write_file.invoke({"file_path": r"\\server\share\file.txt", "content": "x"})
    assert "outside the workspace" in out


def test_a_host_rooted_path_is_refused(tmp_path):
    out = write_file.invoke({"file_path": "/etc/hosts", "content": "x"})
    assert "outside the workspace" in out


def test_every_read_only_tool_is_contained_too(tmp_path):
    """Containment is not only about writes."""
    outside = outside_dir(tmp_path)
    (outside / "secret.txt").write_text("classified", encoding="utf-8")
    for out in (
        read_file.invoke({"file_path": str(outside / "secret.txt")}),
        file_exists.invoke({"file_path": str(outside / "secret.txt")}),
        list_directory.invoke({"directory": str(outside)}),
        grep.invoke({"pattern": "classified", "path": str(outside)}),
    ):
        assert "outside the workspace" in out, out
    # a workspace-wide search cannot see it either
    assert "No matches for" in grep.invoke({"pattern": "classified"})


def test_every_mutating_tool_is_contained(tmp_path):
    outside = outside_dir(tmp_path)
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE,
                      PermissionLevel.DESTRUCTIVE),
        approver=None,
        deny_levels=(),
    ))
    cases = [
        (write_file, {"file_path": str(outside / "w.txt"), "content": "x"}),
        (append_file, {"file_path": str(outside / "a.txt"), "content": "x"}),
        (edit_file, {"file_path": str(outside / "e.txt"),
                     "old_text": "x", "new_text": "y"}),
        (delete_file, {"file_path": str(outside / "d.txt")}),
    ]
    for tool, args in cases:
        out = tool.invoke(args)
        assert "outside the workspace" in out, (tool.name, out)
    assert sorted(p.name for p in outside.iterdir()) == []


def test_a_null_byte_is_refused(tmp_path):
    out = write_file.invoke({"file_path": "bad\x00name.txt", "content": "x"})
    assert "null character" in out
    assert not (tmp_path / "bad\x00name.txt").exists()


def _link_outward(link: Path, target: Path) -> bool:
    """Point *link* at *target* however the host allows, or report that it cannot.

    A Windows symlink needs a privilege an ordinary test run does not have, but a
    directory *junction* does not - and it is the case that matters, because a
    junction is a first-class way for a path inside the workspace to leave it.
    """
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name != "nt" or not target.is_dir():
        return False
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def test_a_link_pointing_out_of_the_workspace_is_refused(tmp_path):
    """Resolution happens before the containment check, so this is caught.

    A symlink is the clean case, but a directory junction is the one that works
    on an ordinary Windows host, and it is the same attack: a path that looks
    like it is inside the workspace and is not.
    """
    outside = outside_dir(tmp_path)
    (outside / "target.txt").write_text("host file", encoding="utf-8")
    link = tmp_path / "link"
    if not _link_outward(link, outside):
        pytest.skip("cannot create a symlink or junction on this host")

    assert "outside the workspace" in read_file.invoke({"file_path": "link/target.txt"})
    assert "outside the workspace" in write_file.invoke(
        {"file_path": "link/target.txt", "content": "overwritten"}
    )
    assert "outside the workspace" in list_directory.invoke({"directory": "link"})
    assert (outside / "target.txt").read_text(encoding="utf-8") == "host file"


def test_a_contained_link_is_allowed(tmp_path):
    """Containment is not "no links": a link that stays inside is fine."""
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "f.txt").write_text("inside", encoding="utf-8")
    link = tmp_path / "alias"
    if not _link_outward(link, tmp_path / "real"):
        pytest.skip("cannot create a symlink or junction on this host")
    assert read_file.invoke({"file_path": "alias/f.txt"}) == "inside"


# ---------------------------------------------------------------------------
# the shell tools, which are the widest reach
# ---------------------------------------------------------------------------

def test_run_command_defaults_to_the_workspace_root(tmp_path):
    out = run_command.invoke({
        "command": f'"{sys.executable}" -c "import os;print(os.getcwd())"',
    })
    assert str(tmp_path.resolve()).lower() in out.lower()


def test_run_command_accepts_a_directory_inside_the_workspace(tmp_path):
    (tmp_path / "inner").mkdir()
    out = run_command.invoke({
        "command": f'"{sys.executable}" -c "import os;print(os.getcwd())"',
        "working_directory": "inner",
    })
    assert str((tmp_path / "inner").resolve()).lower() in out.lower()


def test_run_command_refuses_a_directory_outside_the_workspace(tmp_path):
    outside = outside_dir(tmp_path)
    out = run_command.invoke({
        "command": f'"{sys.executable}" --version',
        "working_directory": str(outside),
    })
    assert "outside the workspace" in out


def test_run_command_reports_a_missing_directory_as_a_result(tmp_path):
    out = run_command.invoke({
        "command": f'"{sys.executable}" --version',
        "working_directory": "no/such/dir",
    })
    assert "working directory does not exist" in out


def test_plan_shell_directory_is_contained(tmp_path):
    outside = outside_dir(tmp_path)
    out = run_in_directory.invoke({
        "command": f'"{sys.executable}" --version',
        "directory": str(outside),
    })
    assert "outside the workspace" in out
    # and an ordinary /plan shell call, with no directory, still runs
    assert "Python" in run_plan_shell.invoke({"command": f'"{sys.executable}" --version'})


# ---------------------------------------------------------------------------
# the module itself
# ---------------------------------------------------------------------------

def test_resolve_returns_an_absolute_resolved_path(tmp_path):
    resolved = resolve_in_workspace("sub/../f.txt")
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "f.txt").resolve()


def test_resolve_handles_a_nonexistent_target(tmp_path):
    assert resolve_in_workspace("never/created.txt") == (tmp_path / "never" / "created.txt")


def test_resolve_raises_workspace_violation_for_the_escape_cases(tmp_path):
    for raw in ("../x", str(tmp_path.parent / "x"), "C:/x", "", "   "):
        with pytest.raises(WorkspaceViolation):
            resolve_in_workspace(raw)


def test_is_within_counts_the_root_as_inside(tmp_path):
    assert is_within(tmp_path, tmp_path)
    assert is_within(tmp_path, tmp_path / "a" / "b")
    assert not is_within(tmp_path, tmp_path.parent)


def test_relative_to_workspace_renders_relative_or_absolute(tmp_path):
    assert relative_to_workspace(tmp_path / "a" / "b.txt") == "a/b.txt"
    assert relative_to_workspace(tmp_path) == "."
    outside = tmp_path.parent / "elsewhere.txt"
    assert relative_to_workspace(outside) == str(outside)


def test_workspace_is_the_cwd_when_nothing_is_configured(tmp_path, monkeypatch):
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert project_root() == tmp_path.resolve()


def test_two_processes_share_one_workspace(tmp_path):
    """Sharing is real, not claimed: a separate process sees the same files."""
    write_file.invoke({"file_path": "handoff/plan.md", "content": "step 1\n"})
    child = subprocess.run(
        [sys.executable, "-c",
         "import os,pathlib;"
         "print(pathlib.Path('handoff/plan.md').read_text(encoding='utf-8').strip())"],
        cwd=str(tmp_path),
        env={**os.environ, WORKSPACE_ENV_VAR: str(tmp_path)},
        capture_output=True, text=True, timeout=60,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "step 1"


def test_a_second_workspace_is_isolated(tmp_path, monkeypatch):
    other = tmp_path.parent / "other-workspace"
    other.mkdir(exist_ok=True)
    write_file.invoke({"file_path": "only-here.txt", "content": "x"})

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(other))
    assert project_root() == other.resolve()
    assert "not found" in read_file.invoke({"file_path": "only-here.txt"})
    assert "outside the workspace" in write_file.invoke(
        {"file_path": str(tmp_path / "only-here.txt"), "content": "y"}
    )


def test_offloading_notes_survive_as_ordinary_files(tmp_path):
    """The offloading story needs nothing but files: no extra memory system."""
    write_file.invoke({"file_path": ".terminus/notes/findings.md",
                       "content": "# findings\n- keep\n"})
    assert (tmp_path / ".terminus" / "notes" / "findings.md").is_file()
    # and it is readable again on the next turn, by the same mechanism
    assert "keep" in read_file.invoke({"file_path": ".terminus/notes/findings.md"})


# ---------------------------------------------------------------------------
# every tool that takes a path is covered by the fixture above
# ---------------------------------------------------------------------------

PATH_TOOLS = [
    read_file, write_file, append_file, edit_file, delete_file,
    list_directory, file_exists, grep, run_command, run_in_directory,
]

PATH_ARGUMENTS = {"file_path", "directory", "path", "working_directory", "cwd"}


def test_every_catalogue_tool_taking_a_path_is_contained(tmp_path):
    """No path-taking tool escaped the change that introduced containment.

    A tool is in scope if it declares a path-shaped argument, so a new tool that
    takes one cannot be added without being brought under the boundary too.
    """
    import inspect

    catalogue = registry.catalogue()
    assert {t.name for t in PATH_TOOLS} <= set(catalogue), (
        "the list of path-taking tools here is out of date"
    )
    for tool in catalogue.values():
        if not (PATH_ARGUMENTS & set(getattr(tool, "args", {}))):
            continue
        source = inspect.getsource(tool.func)
        assert ("resolve_in_workspace" in source
                or "_resolve(" in source
                or "_working_directory" in source
                or "project_root" in source), (
            f"{tool.name} takes a model-supplied path but does not resolve it "
            "inside the workspace"
        )


def test_a_tool_with_no_path_argument_needs_no_path_check():
    """run_shell_command runs in the workspace and takes no directory at all."""
    from terminus.tools.terminal_tools import run_command as run_plan_shell

    assert set(run_plan_shell.args) == {"command"}


def test_git_tools_cannot_be_pointed_outside_the_workspace():
    """Git acts on the workspace root, so containment is structural.

    None of the git tools accepts a path, a directory or a repository argument.
    That is the whole containment guarantee for them: they cannot be aimed at
    another tree, because there is no argument to aim them with.
    """
    names = {n for n in registry.tool_names() if n.startswith("git_")}
    if not names:
        pytest.skip("no git tools registered")
    catalogue = registry.catalogue()
    for name in sorted(names):
        assert not ({"directory", "cwd", "path", "repo", "repository"}
                    & set(catalogue[name].args)), (
            f"{name} accepts a path argument, so it can be pointed outside the "
            "workspace and needs the same resolution as every other path tool"
        )
