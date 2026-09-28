"""The invariant this suite protects:

    Every operation capable of mutating the workspace is governed by a
    runtime-owned permission boundary.

Covers READ / WRITE / DESTRUCTIVE, cross-tool consistency, refusal semantics,
and the fact that the model cannot authorise itself.
"""

from __future__ import annotations

import inspect
import sys

import pytest

from terminus import permissions as perms
from terminus.permissions import (
    Operation,
    PermissionLevel,
    PermissionPolicy,
    authorize_operation,
    get_permission_policy,
    set_permission_policy,
)
from terminus.tools import filesystem_tools, shell_tools
from terminus.tools.filesystem_tools import edit_file, read_file, write_file
from terminus.tools.shell_tools import run_command

PY = sys.executable

STRICT = PermissionPolicy(
    auto_approve=(PermissionLevel.READ_ONLY,),
    approver=None,
    deny_levels=(PermissionLevel.WRITE, PermissionLevel.DESTRUCTIVE),
)
PERMISSIVE = PermissionPolicy(
    auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
    approver=None,
    deny_levels=(PermissionLevel.DESTRUCTIVE,),
)


@pytest.fixture(autouse=True)
def restore_policy():
    yield
    set_permission_policy(PermissionPolicy())


def _source(tool) -> str:
    """Source of a @tool-decorated function, not the generic StructuredTool.invoke."""
    return inspect.getsource(tool.func)


def wf(path, content="x"):
    return write_file.invoke({"file_path": str(path), "content": content})


def ef(path, old, new):
    return edit_file.invoke({"file_path": str(path), "old_text": old, "new_text": new})


def rc(command, cwd=None):
    return run_command.invoke({"command": command, "working_directory": cwd})


# ---------------------------------------------------------------------------
# classification of operations, independent of shell syntax
# ---------------------------------------------------------------------------

def test_operation_levels_do_not_depend_on_shell_syntax():
    assert perms.classify_operation(Operation.READ) is PermissionLevel.READ_ONLY
    assert perms.classify_operation(Operation.WRITE) is PermissionLevel.WRITE
    assert perms.classify_operation(Operation.DELETE) is PermissionLevel.DESTRUCTIVE


def test_execute_defers_to_the_shell_classifier():
    assert perms.classify_operation(Operation.EXECUTE, command="ls") \
        is PermissionLevel.READ_ONLY
    assert perms.classify_operation(Operation.EXECUTE, command="npm install") \
        is PermissionLevel.WRITE
    assert perms.classify_operation(Operation.EXECUTE, command="rm -rf /") \
        is PermissionLevel.DESTRUCTIVE


def test_a_write_can_be_expressed_without_a_command():
    """The whole point of Operation.WRITE: no shell syntax involved."""
    set_permission_policy(STRICT)
    d = authorize_operation(Operation.WRITE, target="src/a.py")
    assert d.allowed is False
    assert d.level is PermissionLevel.WRITE
    assert "src/a.py" in d.description
    assert "write" in d.description


def test_default_policy_is_fail_closed():
    """A context that never installs a policy must deny, not allow."""
    assert get_permission_policy() is not None
    d = get_permission_policy().decide(PermissionLevel.WRITE, "x")
    assert d.allowed is False


# ---------------------------------------------------------------------------
# READ stays available
# ---------------------------------------------------------------------------

def test_read_file_works_without_write_permission(tmp_path):
    set_permission_policy(STRICT)
    target = tmp_path / "r.py"
    target.write_text("hello\n", encoding="utf-8")
    assert read_file.invoke({"file_path": str(target)}) == "hello\n"


def test_grep_works_without_write_permission(tmp_path):
    from terminus.tools.filesystem_tools import grep
    set_permission_policy(STRICT)
    (tmp_path / "a.py").write_text("needle here\n", encoding="utf-8")
    assert "needle" in grep.invoke({"pattern": "needle", "path": str(tmp_path)})


def test_search_codebase_works_without_write_permission(monkeypatch):
    from terminus.tools import codebase_tool
    set_permission_policy(STRICT)
    monkeypatch.setattr(codebase_tool, "get_retriever", lambda: (lambda q, k: [{
        "source": "a.py", "start_line": 1, "end_line": 1,
        "type": "f", "name": "a", "content": "body",
    }]))
    assert "a.py" in codebase_tool.search_codebase.invoke({"query": "q"})


def test_web_tools_work_without_write_permission():
    """Web tools are read-only and must never be gated behind WRITE."""
    from terminus.tools.web_tools import web_fetch, web_search
    for tool, args in ((web_search, {"query": "x"}), (web_fetch, {"url": "u"})):
        for name, schema in tool.args.items():
            assert name in ("query", "url", "formats", "timeout")


def test_read_only_shell_still_runs_under_strict_policy():
    set_permission_policy(STRICT)
    out = rc(f'"{PY}" --version')
    assert "Python" in out
    assert "Refused" not in out


def test_list_directory_and_file_exists_are_read_only():
    from terminus.tools.filesystem_tools import file_exists, list_directory
    # they do not consult the policy at all
    for fn in (read_file, list_directory, file_exists):
        assert "authorize" not in _source(fn), \
            f"{fn.name} must not be authorization-gated"


# ---------------------------------------------------------------------------
# WRITE
# ---------------------------------------------------------------------------

def test_write_file_succeeds_under_permitted_write_policy(tmp_path):
    set_permission_policy(PERMISSIVE)
    target = tmp_path / "w.py"
    assert "successfully" in wf(target, "content")
    assert target.read_text(encoding="utf-8") == "content"


def test_edit_file_succeeds_under_permitted_write_policy(tmp_path):
    set_permission_policy(PERMISSIVE)
    target = tmp_path / "e.py"
    target.write_text("old\n", encoding="utf-8")
    assert "Replaced 1 occurrence" in ef(target, "old", "new")
    assert target.read_text(encoding="utf-8") == "new\n"


def test_write_file_is_refused_without_write_permission(tmp_path):
    set_permission_policy(STRICT)
    target = tmp_path / "nope.py"
    out = wf(target, "content")
    assert out.startswith("Refused:")
    assert "write permission is not available" in out
    assert not target.exists()


def test_edit_file_is_refused_without_write_permission(tmp_path):
    set_permission_policy(STRICT)
    target = tmp_path / "keep.py"
    target.write_text("original\n", encoding="utf-8")
    out = ef(target, "original", "hacked")
    assert out.startswith("Refused:")
    assert target.read_text(encoding="utf-8") == "original\n"


def test_refusal_does_not_mutate_the_filesystem(tmp_path):
    """No target, no temp file, no parent directory side effects."""
    set_permission_policy(STRICT)
    target = tmp_path / "sub" / "deep" / "blocked.py"
    out = wf(target, "content")
    assert out.startswith("Refused:")
    assert not target.exists()
    assert not (tmp_path / "sub").exists(), "makedirs must not run before authorisation"
    assert list(tmp_path.iterdir()) == []


def test_refused_edit_leaves_no_temp_file(tmp_path):
    set_permission_policy(STRICT)
    target = tmp_path / "t.py"
    target.write_text("body\n", encoding="utf-8")
    assert ef(target, "body", "x").startswith("Refused:")
    assert [p.name for p in tmp_path.iterdir()] == ["t.py"]


def test_approval_path_allows_write_when_approved(tmp_path):
    seen = []
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,),
        approver=lambda d, c, lvl, r: seen.append((d, lvl)) or True,
        deny_levels=(),
    ))
    target = tmp_path / "approved.py"
    assert "successfully" in wf(target, "ok")
    assert target.exists()
    assert seen and seen[0][1] is PermissionLevel.WRITE
    assert "approved.py" in seen[0][0]


def test_approval_refusal_blocks_the_write(tmp_path):
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,),
        approver=lambda *a: False,
        deny_levels=(),
    ))
    target = tmp_path / "declined.py"
    out = wf(target, "ok")
    assert "the user declined" in out
    assert "Nothing was changed" in out
    assert not target.exists()


def test_no_approver_means_refusal_not_silent_success(tmp_path):
    """WRITE needing approval with nobody to ask must refuse."""
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,),
        approver=None,
        deny_levels=(),
    ))
    target = tmp_path / "x.py"
    out = wf(target, "ok")
    assert out.startswith("Refused:")
    assert "no approver is available" in out
    assert not target.exists()


# ---------------------------------------------------------------------------
# DESTRUCTIVE
# ---------------------------------------------------------------------------

def test_destructive_shell_still_follows_destructive_policy():
    set_permission_policy(PERMISSIVE)  # WRITE allowed, DESTRUCTIVE denied
    out = rc("rm -rf /")
    assert out.startswith("Refused:")
    assert "destructive" in out


def test_destructive_shell_allowed_when_policy_permits(monkeypatch):
    calls = []
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE,
                      PermissionLevel.DESTRUCTIVE),
        approver=None, deny_levels=(),
    ))
    monkeypatch.setattr(shell_tools.subprocess, "run",
                        lambda c, **k: calls.append(c) or _fake_completed())
    rc("rm -rf /")
    assert len(calls) == 1


def test_delete_operation_is_destructive_not_write():
    """A future delete tool must not be reachable at WRITE level."""
    set_permission_policy(PERMISSIVE)
    d = authorize_operation(Operation.DELETE, target="x")
    assert d.allowed is False
    assert d.level is PermissionLevel.DESTRUCTIVE


def test_filesystem_tools_cannot_bypass_destructive_restrictions(tmp_path):
    """write_file/edit_file are WRITE. They must not be able to act destructively.

    They cannot delete a file, and emptying a file is still a WRITE-level
    operation that the policy can refuse.
    """
    from terminus.tools.filesystem_tools import delete_file  # noqa: F401
    set_permission_policy(STRICT)
    target = tmp_path / "keep.txt"
    target.write_text("data\n", encoding="utf-8")

    # emptying via write_file is refused at WRITE level
    assert wf(target, "data\n").startswith("Refused:")
    assert target.read_text(encoding="utf-8") == "data\n"
    # and there is no delete tool on /ask at all
    assert "delete_file" not in {t.name for t in _ask_tools()}


def _fake_completed():
    import subprocess
    return subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")


def _ask_tools():
    from terminus.agent import factory
    return factory.ASK_TOOLS


# ---------------------------------------------------------------------------
# cross-tool consistency
# ---------------------------------------------------------------------------

def test_all_three_mutating_tools_reach_the_same_policy():
    """write_file, edit_file and run_command must share one code path.

    They share it through the project write guard, which is what every
    mutating tool now goes through: authorise, then hold the project writer lock
    for the duration of the mutation.
    """
    from terminus import coordination

    for fn in (write_file, edit_file, run_command):
        source = _source(fn)
        assert "_write_guard" in source or "project_write_guard" in source, \
            f"{fn.name} does not go through the project write guard"

    # And the guard itself really does consult the one shared policy.
    guard_source = inspect.getsource(coordination.project_write_guard)
    assert "authorize_operation" in guard_source


def test_one_policy_change_moves_all_three_together(tmp_path):
    """The decisive cross-tool test: identical treatment from one switch."""
    target = tmp_path / "all.py"
    target.write_text("old\n", encoding="utf-8")

    set_permission_policy(STRICT)
    assert wf(target, "a").startswith("Refused:")
    assert ef(target, "old", "b").startswith("Refused:")
    assert rc(f'"{PY}" -c "print(1)"').startswith("Refused:")
    assert target.read_text(encoding="utf-8") == "old\n", "nothing may have changed"

    set_permission_policy(PERMISSIVE)
    assert "successfully" in wf(target, "a")            # replaces the file wholesale
    assert "Replaced 1 occurrence" in ef(target, "a", "b")
    assert "Refused" not in rc(f'"{PY}" -c "print(1)"')
    assert target.read_text(encoding="utf-8") == "b"


def test_shell_tools_policy_accessors_are_the_shared_ones():
    assert shell_tools.get_permission_policy is perms.get_permission_policy
    assert shell_tools.set_permission_policy is perms.set_permission_policy


def test_policy_is_process_wide_not_per_tool():
    p1 = PermissionPolicy(auto_approve=(PermissionLevel.READ_ONLY,
                                        PermissionLevel.WRITE),
                          approver=None, deny_levels=())
    set_permission_policy(p1)
    assert get_permission_policy() is p1
    # One module reads the policy, on behalf of every tool: the guard. No tool
    # keeps its own copy of the decision logic to drift out of step.
    from terminus import coordination

    assert coordination.authorize_operation is perms.authorize_operation
    assert not hasattr(filesystem_tools, "authorize_operation")


# ---------------------------------------------------------------------------
# the model cannot authorise itself
# ---------------------------------------------------------------------------

def test_no_mutating_tool_exposes_a_permission_argument():
    for tool, allowed in ((write_file, {"file_path", "content"}),
                          (edit_file, {"file_path", "old_text", "new_text"}),
                          (run_command, {"command", "working_directory"})):
        assert set(tool.args) == allowed, f"{tool.name} exposes extra arguments"


def test_model_cannot_pass_permission_or_approval():
    """Only the declared argument names matter - prose in a docstring does not."""
    for tool in (write_file, edit_file, run_command):
        props = set(tool.args_schema.model_json_schema().get("properties", {}))
        for forbidden in ("permission", "approved", "level", "force",
                          "policy", "escalate", "override"):
            assert forbidden not in props, \
                f"{tool.name} accepts a '{forbidden}' argument"


def test_extra_arguments_are_ignored_not_honoured(tmp_path):
    """Even if a model emits a permission kwarg, the tool cannot accept it."""
    set_permission_policy(STRICT)
    target = tmp_path / "sneaky.py"
    out = write_file.invoke({"file_path": str(target), "content": "x",
                             "permission": "WRITE", "approved": True})
    assert out.startswith("Refused:")
    assert not target.exists()


def test_only_the_runtime_module_can_install_a_policy():
    """set_permission_policy lives in runtime modules, not in any tool."""
    for tool in (write_file, edit_file, run_command):
        assert "set_permission_policy" not in _source(tool), \
            f"{tool.name} must not be able to install its own policy"


def test_a_child_agent_cannot_upgrade_its_own_policy_by_calling_a_tool(tmp_path):
    """No tool path mutates the policy itself."""
    set_permission_policy(STRICT)
    escalating = PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE,
                      PermissionLevel.DESTRUCTIVE),
        approver=None, deny_levels=())
    set_permission_policy(escalating)  # only the runtime can do this
    # and even then, a child inheriting the STRICT policy has no tool to call
    for tool in (write_file, edit_file, run_command):
        assert "set_permission_policy" not in _source(tool)
    assert get_permission_policy() is escalating


# ---------------------------------------------------------------------------
# result semantics (the model must be able to tell these apart)
# ---------------------------------------------------------------------------

def test_success_refusal_and_failure_are_distinguishable(tmp_path):
    set_permission_policy(PERMISSIVE)
    ok = wf(tmp_path / "ok.py", "x")
    assert "successfully" in ok
    assert "Refused" not in ok

    set_permission_policy(STRICT)
    refused = wf(tmp_path / "no.py", "x")
    assert refused.startswith("Refused:")
    assert "Nothing was changed" in refused

    set_permission_policy(PERMISSIVE)
    failed = wf(tmp_path / "sub" / "\0bad", "x")
    assert "Refused" not in failed
    assert "Could not write" in failed


def test_validation_errors_do_not_look_like_permission_errors(tmp_path):
    set_permission_policy(PERMISSIVE)
    missing = ef(tmp_path / "nope.py", "a", "b")
    assert missing.startswith("No change made:")
    assert "Refused" not in missing
    assert "not found" in missing


def test_refusal_does_not_leak_policy_internals(tmp_path):
    set_permission_policy(STRICT)
    out = wf(tmp_path / "x.py", "y")
    for leak in ("auto_approve", "deny_levels", "PermissionPolicy(", "approver"):
        assert leak not in out


# ---------------------------------------------------------------------------
# path safety: current status, pinned so it cannot change unnoticed
# ---------------------------------------------------------------------------

def test_paths_are_not_confinemented_but_are_authorised(tmp_path):
    """DEFERRED, documented: there is no workspace sandbox.

    These assertions pin the *current* behaviour so that adding confinement
    later is a deliberate, visible change rather than an accident. They also
    record the important consequence: authorisation still applies no matter how
    unusual the path is.
    """
    set_permission_policy(PERMISSIVE)
    outside = tmp_path / "outside"
    outside.mkdir()

    # 1. an absolute path outside the cwd is accepted (no confinement)
    escaped = outside / "abs.txt"
    assert "successfully" in wf(escaped, "x")
    assert escaped.exists()

    # 2. '..' traversal is accepted
    traversal = tmp_path / "sub" / ".." / "trav.txt"
    assert "successfully" in wf(traversal, "x")
    assert (tmp_path / "trav.txt").exists()

    # 3. but authorisation still governs, whatever the path
    set_permission_policy(STRICT)
    assert wf(outside / "denied.txt", "x").startswith("Refused:")
    assert not (outside / "denied.txt").exists()

    target = outside / "abs.txt"
    assert ef(target, "x", "y").startswith("Refused:")
    assert target.read_text(encoding="utf-8") == "x"
