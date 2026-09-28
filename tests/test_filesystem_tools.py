"""Focused tests for the filesystem tools exposed to the /ask agent.

Mutation tests all operate inside pytest's tmp_path so no real project file is
touched. They install a policy that allows writes explicitly, so they do not
depend on which policy another test file happened to leave installed.
"""

from pathlib import Path

import pytest

from terminus.permissions import (
    PermissionLevel,
    PermissionPolicy,
    get_permission_policy,
    set_permission_policy,
)
from terminus.tools.filesystem_tools import edit_file, write_file


@pytest.fixture(autouse=True)
def writable_project():
    previous = get_permission_policy()
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    ))
    yield
    set_permission_policy(previous)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def test_write_file_creates_new_file(tmp_path: Path):
    target = tmp_path / "created.py"
    result = write_file.invoke(
        {"file_path": str(target), "content": "print('hi')\n"}
    )
    assert "File written successfully" in result
    assert target.read_text(encoding="utf-8") == "print('hi')\n"


def test_write_file_overwrites_existing_file(tmp_path: Path):
    target = tmp_path / "existing.txt"
    target.write_text("old content that is long enough to see replaced", encoding="utf-8")
    result = write_file.invoke({"file_path": str(target), "content": "new"})
    assert "File written successfully" in result
    assert target.read_text(encoding="utf-8") == "new"


def test_write_file_creates_parent_directories(tmp_path: Path):
    target = tmp_path / "deep" / "nested" / "file.txt"
    result = write_file.invoke({"file_path": str(target), "content": "body"})
    assert "File written successfully" in result
    assert target.read_text(encoding="utf-8") == "body"


def test_write_file_rejects_empty_path(tmp_path: Path):
    assert write_file.invoke({"file_path": "  ", "content": "x"}) == "No file path provided"


def test_write_file_rejects_empty_content(tmp_path: Path):
    target = tmp_path / "unchanged.txt"
    assert write_file.invoke({"file_path": str(target), "content": ""}) == "No content provided"
    assert not target.exists()


def test_write_file_errors_on_unusable_path(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    result = write_file.invoke(
        {"file_path": str(blocker / "child.txt"), "content": "data"}
    )
    assert result.startswith("Could not write")
    # the existing file must be untouched
    assert blocker.read_text(encoding="utf-8") == "i am a file, not a directory"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------

def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_edit_file_replaces_exact_single_match(tmp_path: Path):
    target = _write(tmp_path / "a.py", "alpha\nBETA\ngamma\n")
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "BETA", "new_text": "delta"}
    )
    assert "Replaced 1 occurrence" in result
    assert "line 2" in result
    assert target.read_text(encoding="utf-8") == "alpha\ndelta\ngamma\n"


def test_edit_file_preserves_everything_around_the_match(tmp_path: Path):
    original = "# header\nvalue = 1\n# footer\n"
    target = _write(tmp_path / "b.py", original)
    edit_file.invoke(
        {"file_path": str(target), "old_text": "value = 1", "new_text": "value = 2"}
    )
    assert target.read_text(encoding="utf-8") == "# header\nvalue = 2\n# footer\n"


def test_edit_file_missing_old_text_makes_no_change(tmp_path: Path):
    original = "alpha\nbeta\n"
    target = _write(tmp_path / "c.py", original)
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "GAMMA", "new_text": "x"}
    )
    assert "old_text not found" in result
    assert result.startswith("No change made:")
    assert target.read_text(encoding="utf-8") == original


def test_edit_file_duplicate_old_text_makes_no_change(tmp_path: Path):
    original = "dup\nmiddle\ndup\n"
    target = _write(tmp_path / "d.py", original)
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "dup", "new_text": "unique"}
    )
    assert "found 2 times" in result
    assert result.startswith("No change made:")
    assert target.read_text(encoding="utf-8") == original


def test_edit_file_ambiguity_requires_more_context(tmp_path: Path):
    original = "value = 1\nother = 0\nvalue = 1\n"
    target = _write(tmp_path / "e.py", original)
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "value = 1", "new_text": "value = 9"}
    )
    assert "found 2 times" in result
    result = edit_file.invoke(
        {
            "file_path": str(target),
            "old_text": "other = 0\nvalue = 1",
            "new_text": "other = 0\nvalue = 9",
        }
    )
    assert "Replaced 1 occurrence" in result
    assert target.read_text(encoding="utf-8") == "value = 1\nother = 0\nvalue = 9\n"


def test_edit_file_empty_new_text_deletes_the_match(tmp_path: Path):
    target = _write(tmp_path / "f.py", "keep\nREMOVE_ME\nkeep2\n")
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "REMOVE_ME\n", "new_text": ""}
    )
    assert "Replaced 1 occurrence" in result
    assert target.read_text(encoding="utf-8") == "keep\nkeep2\n"


def test_edit_file_missing_target_file_is_not_created(tmp_path: Path):
    target = tmp_path / "never_created.py"
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "anything", "new_text": "something"}
    )
    assert "file not found" in result
    assert not target.exists()


def test_edit_file_does_not_create_parent_directories(tmp_path: Path):
    target = tmp_path / "missing_dir" / "file.py"
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "a", "new_text": "b"}
    )
    assert "file not found" in result
    assert not (tmp_path / "missing_dir").exists()


def test_edit_file_rejects_directory_path(tmp_path: Path):
    result = edit_file.invoke(
        {"file_path": str(tmp_path), "old_text": "a", "new_text": "b"}
    )
    assert "is a directory" in result


def test_edit_file_rejects_empty_old_text(tmp_path: Path):
    original = "body\n"
    target = _write(tmp_path / "g.py", original)
    result = edit_file.invoke(
        {"file_path": str(target), "old_text": "", "new_text": "x"}
    )
    assert "old_text must not be empty" in result
    assert target.read_text(encoding="utf-8") == original


def test_edit_file_rejects_empty_path(tmp_path: Path):
    assert edit_file.invoke(
        {"file_path": "  ", "old_text": "a", "new_text": "b"}
    ) == "No file path provided"


def test_edit_file_leaves_no_temp_file_behind(tmp_path: Path):
    target = _write(tmp_path / "h.py", "a\nb\n")
    edit_file.invoke({"file_path": str(target), "old_text": "b", "new_text": "c"})
    assert target.read_text(encoding="utf-8") == "a\nc\n"
    assert [p.name for p in tmp_path.iterdir()] == ["h.py"]


def test_edit_file_multiline_replacement_is_exact(tmp_path: Path):
    original = "def f():\n    return 1\n"
    target = _write(tmp_path / "i.py", original)
    edit_file.invoke(
        {
            "file_path": str(target),
            "old_text": "def f():\n    return 1",
            "new_text": "def f():\n    return 2",
        }
    )
    assert target.read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_edit_file_and_write_file_interoperate(tmp_path: Path):
    target = tmp_path / "j.py"
    write_file.invoke({"file_path": str(target), "content": "one\ntwo\n"})
    edit_file.invoke({"file_path": str(target), "old_text": "two", "new_text": "three"})
    assert target.read_text(encoding="utf-8") == "one\nthree\n"


# ---------------------------------------------------------------------------
# tool wiring expected by /ask
# ---------------------------------------------------------------------------

def test_expected_ask_tools_are_importable_and_named():
    from terminus.tools.codebase_tool import search_codebase
    from terminus.skills.skill_tools import load_skill
    from terminus.tools.filesystem_tools import (
        file_exists,
        grep,
        list_directory,
        read_file,
    )

    names = {
        search_codebase.name,
        grep.name,
        list_directory.name,
        read_file.name,
        file_exists.name,
        write_file.name,
        edit_file.name,
        load_skill.name,
    }
    assert names == {
        "search_codebase",
        "grep",
        "list_directory",
        "read_file",
        "file_exists",
        "write_file",
        "edit_file",
        "load_skill",
    }


def test_destructive_tools_are_not_exposed_to_ask_agent():
    """delete_file / append_file / run_in_directory must stay off /ask.

    'run_command' IS now on /ask (shell execution was added deliberately), but
    it is policy-gated at call time rather than unrestricted.
    """
    import terminus.agent.factory as factory

    names = {tool.name for tool in factory.ASK_TOOLS}
    for forbidden in ("delete_file", "append_file", "run_in_directory"):
        assert forbidden not in names, f"{forbidden} must not be in the /ask tool list"
    assert "run_command" in names


def test_ask_run_command_is_policy_gated():
    """/ask shell access must go through the runtime permission policy."""
    import terminus.agent.factory as factory
    from terminus.tools.shell_tools import run_command

    assert run_command in factory.ASK_TOOLS
    # no permission/approval/force argument exists for the model to set
    assert set(run_command.args) == {"command", "working_directory"}
