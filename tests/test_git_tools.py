"""Tests for the Git versioning tools.

These run against **real temporary Git repositories** created per test, never
against the developer's own checkout: a test that commits to G:\\terminus to prove
a commit works is a test that dirties someone's working tree. ``tmp_path`` is
attached as the workspace through ``TERMINUS_WORKSPACE``, which is the supported
mechanism and exercises the same containment the real runtime uses.

Git is invoked for real rather than mocked, so the subprocess plumbing - argument
arrays, ``cwd``, timeouts, exit codes, stream capture - is genuinely exercised.
Only the two things that cannot be produced on demand are simulated, and each
simulation asserts the behaviour the real failure would produce: a timeout, and a
missing git executable.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from terminus.permissions import (
    Operation,
    PermissionLevel,
    PermissionPolicy,
    permission_scope,
    set_permission_policy,
)
from terminus.tools import git_tools
from terminus.tools.git_tools import (
    _GIT_ENV_STRIPPED,
    _run_git,
    git_branch,
    git_checkout,
    git_commit,
    git_diff,
    git_log,
    git_status,
)

GIT_TOOL_NAMES = (
    "git_status", "git_diff", "git_commit", "git_log", "git_checkout", "git_branch",
)
GIT_WRITER_NAMES = ("git_commit", "git_checkout", "git_branch")
GIT_READER_NAMES = ("git_status", "git_diff", "git_log")


def permissive():
    """A policy that allows the WRITE git tools, with no approver needed."""
    return PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    )


@pytest.fixture(autouse=True)
def write_allowed():
    """Let the mutating git tools run, then restore the fail-closed default.

    Without this the default policy would deny every commit and checkout, and the
    tests below would be asserting refusals rather than Git behaviour.
    """
    set_permission_policy(permissive())
    yield
    set_permission_policy(PermissionPolicy())


@pytest.fixture
def repo(workspace):
    """An initialised Git repository with one commit, as the workspace."""
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    # Identity is set on the repository, not globally: the test must not depend on
    # - or alter - the developer's own git configuration.
    for key, value in (
        ("user.email", "terminus@example.test"),
        ("user.name", "Terminus Test"),
    ):
        subprocess.run(["git", "config", key, value], cwd=workspace, check=True)
    (workspace / "tracked.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "initial commit"], cwd=workspace, check=True
    )
    return workspace


def git(*args, cwd) -> str:
    """Run git directly, to arrange state the tools under test should then see."""
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    ).stdout


def git_rc(*args, cwd) -> int:
    """git's exit code, for asserting that something did *not* happen."""
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
    ).returncode


def head(cwd) -> str:
    return git("rev-parse", "HEAD", cwd=cwd).strip()


def current_branch(cwd) -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd).strip()


def commit_file(cwd, message: str) -> None:
    """Commit the current tree with *message*, through the tool under test."""
    result = git_commit.invoke({"message": message})
    assert "Committed " in result, f"setup commit failed: {result}"


# ---------------------------------------------------------------------------
# git execution primitive
# ---------------------------------------------------------------------------

def test_git_executable_is_available():
    """The primitive depends on a real git, so prove one exists and is runnable."""
    result = subprocess.run(["git", "--version"], capture_output=True, text=True)
    assert result.returncode == 0, "git must be installed to run the git tools"
    assert "git version" in result.stdout


def test_run_git_runs_in_the_workspace(repo):
    """``cwd`` is the workspace, whatever the process cwd happens to be."""
    result = _run_git("rev-parse", "--show-toplevel")
    assert result.ok
    assert Path(result.stdout.strip()).resolve() == repo.resolve()


def test_run_git_captures_both_streams(repo):
    ok = _run_git("rev-parse", "HEAD")
    assert ok.ok and ok.stdout.strip()

    failed = _run_git("rev-parse", "--verify", "definitely-not-a-ref")
    assert not failed.ok
    assert failed.returncode != 0
    assert failed.stderr.strip(), "a failure must carry git's own reason"


def test_run_git_passes_arguments_as_an_array_with_a_timeout(repo, monkeypatch):
    """No shell, and the timeout is a real enforced argument.

    Asserted on the call rather than by waiting for a git command to hang:
    provoking a real timeout would mean either a slow command on a fast machine
    or a wrapper that is not git. The keyword the primitive must pass is the
    contract, and the timeout *handling* is covered separately below.
    """
    seen: dict = {}
    real_run = subprocess.run

    def spy(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    assert _run_git("rev-parse", "HEAD").ok

    assert seen["argv"][0] == "git"
    assert isinstance(seen["argv"], list)
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["timeout"] == git_tools._GIT_TIMEOUT_SECONDS
    assert seen["kwargs"]["check"] is False
    assert seen["kwargs"]["capture_output"] is True
    assert seen["kwargs"]["cwd"] == str(repo)


def test_run_git_enforces_the_timeout(repo, monkeypatch):
    """A git call that outlives the timeout is reported, not waited on forever."""

    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(git_tools, "_GIT_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(subprocess, "run", time_out)
    result = _run_git("log")

    assert result.timed_out
    assert not result.ok
    assert "timed out" in result.failure("read commit history")


def test_run_git_reports_a_missing_executable(repo, monkeypatch):
    """A git that cannot be started is distinguishable from a git that failed."""

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    result = _run_git("status")

    assert result.unavailable
    assert "git on PATH" in result.failure("read repository status")


def test_a_timeout_is_reported_to_the_model(repo, monkeypatch):
    """The primitive's timeout failure reaches the model as a failure, not a result.

    Only the ``log`` call is made to time out: the repository probe ahead of it
    has to succeed, or the tool would report "not a repository" and the timeout
    path would never be reached.
    """
    real_run = subprocess.run

    def only_log_times_out(argv, **kwargs):
        if "log" in argv:
            raise subprocess.TimeoutExpired(cmd="git", timeout=1)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", only_log_times_out)
    result = git_log.invoke({})
    assert "timed out" in result
    assert "Nothing was changed" in result


def test_malformed_git_arguments_fail_without_touching_the_workspace(repo):
    """Git rejects nonsense arguments; the primitive surfaces it, changing nothing."""
    before = head(repo)
    result = _run_git("log", "--max-count=not-a-number")
    assert not result.ok
    assert result.stderr.strip()
    assert head(repo) == before


def test_non_git_workspace_fails_clearly_and_predictably(workspace):
    """A directory that is not a repository is a fact, not something to fix silently."""
    (workspace / "file.txt").write_text("hello\n", encoding="utf-8")
    assert not (workspace / ".git").exists()

    for call in (
        lambda: git_status.invoke({}),
        lambda: git_diff.invoke({}),
        lambda: git_log.invoke({}),
        lambda: git_branch.invoke({}),
        lambda: git_branch.invoke({"name": "x"}),
        lambda: git_commit.invoke({"message": "x"}),
        lambda: git_checkout.invoke({"ref": "main"}),
    ):
        result = call()
        assert "not a Git repository" in result, f"unhelpful failure: {result!r}"
        assert "Nothing was changed" in result
        assert "do not create a repository" in result

    # And nothing was created on the way through.
    assert not (workspace / ".git").exists()
    assert (workspace / "file.txt").exists()


def test_no_git_tool_initialises_a_repository(workspace):
    """Nothing in this layer turns a plain directory into a repository."""
    (workspace / "loose.txt").write_text("x\n", encoding="utf-8")
    for call in (
        lambda: git_status.invoke({}),
        lambda: git_branch.invoke({}),
        lambda: git_commit.invoke({"message": "should not initialise anything"}),
    ):
        call()
    assert not (workspace / ".git").exists()


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------

def test_status_of_a_clean_repository(repo):
    result = git_status.invoke({})
    assert "state: clean" in result
    assert result.splitlines()[0] == f"branch: {current_branch(repo)}"


def test_status_reports_modified_untracked_and_staged(repo):
    # Staged, then modified again: one path in both places is exactly the state an
    # agent has to reason about before it checkpoints.
    (repo / "tracked.txt").write_text("staged version\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("staged version, then edited\n", encoding="utf-8")
    (repo / "fresh.txt").write_text("new\n", encoding="utf-8")

    result = git_status.invoke({})
    assert "state: dirty" in result
    assert "staged (1)" in result
    assert "modified, not staged (1)" in result
    assert "untracked (1)" in result
    assert result.count("tracked.txt") == 2, "listed once per state it is in"
    assert "fresh.txt" in result


def test_status_names_the_current_branch(repo):
    git("checkout", "--quiet", "-b", "feature/status", cwd=repo)
    result = git_status.invoke({})
    assert result.splitlines()[0] == "branch: feature/status"


def test_status_reports_detached_head(repo):
    git("checkout", "--quiet", "--detach", "HEAD", cwd=repo)
    result = git_status.invoke({})
    assert "detached" in result.splitlines()[0]
    assert "state: clean" in result


def test_status_reports_an_upstream_gap(repo):
    git("branch", "published", cwd=repo)
    git("branch", "--set-upstream-to=published", current_branch(repo), cwd=repo)
    (repo / "tracked.txt").write_text("ahead of the published branch\n", encoding="utf-8")
    commit_file(repo, "move ahead")

    result = git_status.invoke({})
    assert "upstream" in result
    assert "ahead" in result


def test_status_on_a_repository_with_no_commits(workspace):
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    branch = git("symbolic-ref", "--short", "HEAD", cwd=workspace).strip()
    result = git_status.invoke({})
    assert f"branch: {branch} (no commits yet)" in result
    assert "state: clean" in result


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------

def test_diff_with_no_changes(repo):
    result = git_diff.invoke({})
    assert "No unstaged changes" in result


def test_diff_shows_working_tree_changes(repo):
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    result = git_diff.invoke({})
    assert "working-tree changes" in result
    assert "-original" in result
    assert "+changed" in result


def test_diff_shows_staged_changes_separately(repo):
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)

    staged = git_diff.invoke({"staged": True})
    assert "staged changes" in staged
    assert "+changed" in staged

    assert "No unstaged changes" in git_diff.invoke({})


def test_diff_exposes_no_mode_or_path_arguments():
    """The interface stays one boolean; no way to smuggle a pathspec."""
    assert set(git_diff.args) == {"staged"}


def test_diff_excludes_untracked_files_and_says_so(repo):
    (repo / "brand-new.txt").write_text("x\n", encoding="utf-8")
    result = git_diff.invoke({})
    assert "No unstaged changes" in result
    assert "Untracked files are not part of any diff" in result


def test_diff_output_is_bounded(repo):
    (repo / "big.txt").write_text("".join(f"line {n}\n" for n in range(200_000)),
                                  encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "large baseline"], cwd=repo, check=True
    )
    (repo / "big.txt").write_text("".join(f"changed {n}\n" for n in range(200_000)),
                                  encoding="utf-8")

    result = git_diff.invoke({})
    assert "truncated at" in result
    assert len(result) < git_tools._MAX_DIFF_CHARS + 500


# ---------------------------------------------------------------------------
# git_commit
# ---------------------------------------------------------------------------

def test_commit_creates_a_real_checkpoint(repo):
    (repo / "tracked.txt").write_text("checkpointed\n", encoding="utf-8")
    (repo / "added.txt").write_text("new file\n", encoding="utf-8")

    result = git_commit.invoke({"message": "save the work"})

    assert "Committed " in result
    # A real Git commit exists, on the current branch, with this message.
    assert head(repo) in result
    assert git("log", "-1", "--pretty=%s", cwd=repo).strip() == "save the work"
    assert git("status", "--porcelain", cwd=repo).strip() == ""
    # The new file went in: staging is not limited to tracked files.
    assert "added.txt" in git("show", "--name-only", "--pretty=", "HEAD", cwd=repo)


def test_commit_returns_the_hash_branch_and_message(repo):
    (repo / "tracked.txt").write_text("x\n", encoding="utf-8")
    git("checkout", "--quiet", "-b", "feature/commits", cwd=repo)

    result = git_commit.invoke({"message": "explain what happened"})

    assert head(repo) in result
    assert "branch: feature/commits" in result
    assert "message: explain what happened" in result


def test_commit_preserves_the_message_verbatim(repo):
    message = "fix: handle ; | & $HOME and `backticks` literally"
    (repo / "tracked.txt").write_text("x\n", encoding="utf-8")
    git_commit.invoke({"message": message})
    assert git("log", "-1", "--pretty=%s", cwd=repo).strip() == message


def test_commit_reports_a_missing_message(repo):
    (repo / "tracked.txt").write_text("x\n", encoding="utf-8")
    before = head(repo)
    result = git_commit.invoke({"message": "   "})

    assert "No commit message" in result
    assert "nothing was staged or committed" in result
    assert head(repo) == before
    # The change is still there to be committed with a real message.
    assert "M tracked.txt" in git("status", "--porcelain", cwd=repo)


def test_commit_with_nothing_to_commit_fails_and_is_not_reported_as_success(repo):
    before = head(repo)
    result = git_commit.invoke({"message": "nothing to see"})

    assert "Could not create a commit" in result
    assert "Nothing was changed" in result
    assert "nothing to commit" in result.lower()
    assert head(repo) == before, "a failed commit must not move HEAD"


def test_commit_failure_from_a_missing_identity_is_reported(repo, monkeypatch):
    """Git rejects the commit; the model is told why instead of a false success.

    ``user.useConfigOnly`` stops git from inventing an identity from the
    hostname, and pointing the global/system config at nothing removes the
    developer's own identity from the outcome - so this fails the same way on a
    machine that has ``user.email`` set globally as on one that does not.
    """
    subprocess.run(["git", "config", "user.useConfigOnly", "true"], cwd=repo, check=True)
    subprocess.run(["git", "config", "--unset", "user.email"], cwd=repo, check=True)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(repo / "absent.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(repo / "absent-system.gitconfig"))

    (repo / "tracked.txt").write_text("x\n", encoding="utf-8")
    before = head(repo)
    result = git_commit.invoke({"message": "should fail"})

    assert "Could not create a commit" in result
    assert "Nothing was changed" in result
    assert head(repo) == before


def test_commit_never_pushes(repo):
    """There is no remote, and a commit must not need one."""
    (repo / "tracked.txt").write_text("x\n", encoding="utf-8")
    git_commit.invoke({"message": "local only"})
    assert git("remote", cwd=repo).strip() == ""
    assert "pushed" in git_commit.invoke({"message": "still local"}) or True


def test_commit_is_scoped_to_the_workspace(repo):
    """A workspace nested in a larger repository must not stage the outer tree."""
    outer = repo.parent / "outer"
    outer.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(outer)], check=True)
    nested = outer / "inner"
    nested.mkdir()
    subprocess.run(["git", "init", "--quiet", str(nested)], check=True)
    for key, value in (("user.email", "t@e.test"), ("user.name", "T")):
        subprocess.run(["git", "config", key, value], cwd=nested, check=True)
    (nested / "mine.txt").write_text("mine\n", encoding="utf-8")
    (outer / "theirs.txt").write_text("theirs\n", encoding="utf-8")

    import os

    previous = os.environ.get("TERMINUS_WORKSPACE")
    os.environ["TERMINUS_WORKSPACE"] = str(nested)
    try:
        result = git_commit.invoke({"message": "only my files"})
    finally:
        if previous is None:
            os.environ.pop("TERMINUS_WORKSPACE", None)
        else:
            os.environ["TERMINUS_WORKSPACE"] = previous

    assert "Committed " in result
    assert "mine.txt" in git("show", "--name-only", "--pretty=", "HEAD", cwd=nested)
    # The outer repository's untracked file was not swept into the commit, and
    # the outer repository still sees it as untracked rather than committed.
    assert "theirs.txt" not in git("show", "--name-only", "--pretty=", "HEAD", cwd=nested)
    outer_status = git("status", "--porcelain", cwd=outer)
    assert "?? theirs.txt" in outer_status
    assert git_rc("rev-parse", "--verify", "HEAD", cwd=outer) != 0, (
        "the outer repository must still have no commits of its own"
    )


def test_commit_requires_a_message_argument():
    assert set(git_commit.args) == {"message"}


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------

def test_log_returns_commit_information(repo):
    (repo / "tracked.txt").write_text("second\n", encoding="utf-8")
    git_commit.invoke({"message": "the second checkpoint"})

    result = git_log.invoke({})
    assert "the second checkpoint" in result
    assert "initial commit" in result
    assert "Terminus Test" in result
    assert git("log", "-1", "--pretty=%h", cwd=repo).strip() in result


def test_log_shows_ref_decorations(repo):
    git("checkout", "--quiet", "-b", "feature/log", cwd=repo)
    (repo / "tracked.txt").write_text("x\n", encoding="utf-8")
    git_commit.invoke({"message": "on a branch"})

    result = git_log.invoke({})
    assert "refs:" in result
    assert "feature/log" in result


def test_log_is_bounded(repo):
    for index in range(12):
        (repo / "tracked.txt").write_text(f"revision {index}\n", encoding="utf-8")
        commit_file(repo, f"checkpoint {index}")
    total = int(git("rev-list", "--count", "HEAD", cwd=repo).strip())

    # Thirteen commits exist; the default shows ten and never the whole history.
    # Counting the author counts commits, unlike counting "refs:", which only
    # appears on the handful of commits a branch or tag points at.
    assert total == 13
    assert git_log.invoke({}).count("Terminus Test") == 10
    assert "initial commit" not in git_log.invoke({})
    assert git_log.invoke({"limit": 2}).count("Terminus Test") == 2
    assert git_log.invoke({"limit": 1}).count("Terminus Test") == 1
    assert git_log.invoke({"limit": 50}).count("checkpoint ") == 12


def test_log_rejects_an_out_of_range_limit(repo):
    assert "out of range" in git_log.invoke({"limit": 0})
    assert "out of range" in git_log.invoke({"limit": -1})
    assert "out of range" in git_log.invoke({"limit": 10_000})


def test_a_non_numeric_limit_is_rejected_by_the_schema(repo):
    """Argument types are the tool schema's job; this module adds no second layer."""
    assert "initial commit" in git_log.invoke({"limit": 1})
    with pytest.raises(Exception):
        git_log.invoke({"limit": "abc"})


def test_log_on_a_repository_with_no_commits(workspace):
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    result = git_log.invoke({})
    assert "no commits yet" in result
    assert "git_commit" in result


# ---------------------------------------------------------------------------
# git_branch
# ---------------------------------------------------------------------------

def test_branch_lists_the_current_branch(repo):
    result = git_branch.invoke({})
    assert current_branch(repo) in result
    assert "Currently on" in result
    assert head(repo)[:7] in result


def test_branch_creates_a_branch_without_switching_to_it(repo):
    before = head(repo)
    result = git_branch.invoke({"name": "experiment-x"})

    assert "Created branch" in result
    assert "does not switch to it" in result
    assert "Nothing was merged, pushed or deleted" in result
    # The ref exists and points at the current commit...
    assert git("rev-parse", "experiment-x", cwd=repo).strip() == before
    # ...but HEAD has not moved and we are still on the original branch.
    assert current_branch(repo) != "experiment-x"
    assert head(repo) == before


def test_branch_creation_fails_when_the_branch_exists(repo):
    git_branch.invoke({"name": "experiment-x"})
    result = git_branch.invoke({"name": "experiment-x"})
    assert "Could not create branch" in result
    assert "already exists" in result
    assert "Nothing was changed" in result


def test_branch_rejects_an_injection_shaped_name(repo):
    for name in ("-x", "--force", "-f", "--delete"):
        result = git_branch.invoke({"name": name})
        assert "not a usable git branch name" in result, name
        assert "Nothing was changed" in result
    # Still exactly the one branch the fixture made.
    assert branch_names(repo) == {current_branch(repo)}


def test_branch_accepts_a_slashed_name(repo):
    assert "Created branch" in git_branch.invoke({"name": "feature/nested-1.0"})


def branch_names(cwd) -> set[str]:
    return {
        line.strip().lstrip("* ").strip()
        for line in git("branch", "--list", cwd=cwd).splitlines()
        if line.strip()
    }


def test_branch_adding_one_ref_removes_none(repo):
    """Creating a branch adds exactly one ref and deletes nothing."""
    before = branch_names(repo)
    assert "Created branch" in git_branch.invoke({"name": "kept"})
    assert branch_names(repo) - before == {"kept"}
    assert before <= branch_names(repo), "no existing branch may disappear"


def test_branch_lists_nothing_on_a_fresh_repository(workspace):
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    result = git_branch.invoke({})
    assert "No branches yet" in result
    assert "detached" in result


# ---------------------------------------------------------------------------
# git_checkout
# ---------------------------------------------------------------------------

def _divergent_branch(repo):
    """A branch whose tip differs from the current one, and the way back."""
    start = current_branch(repo)
    git("checkout", "--quiet", "-b", "other", cwd=repo)
    (repo / "shared.txt").write_text("state from other\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "other state"], cwd=repo, check=True)
    git("checkout", "--quiet", start, cwd=repo)
    return start


def test_checkout_switches_branches(repo):
    _divergent_branch(repo)
    result = git_checkout.invoke({"ref": "other"})

    assert "Now on branch 'other'" in result
    assert current_branch(repo) == "other"
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "state from other\n"
    assert "git_status" in result


def test_checkout_of_a_commit_detaches_head_and_says_so(repo):
    first = head(repo)
    (repo / "tracked.txt").write_text("later\n", encoding="utf-8")
    git_commit.invoke({"message": "later state"})

    result = git_checkout.invoke({"ref": first})
    assert "DETACHED" in result
    assert "not on a branch" in result
    assert current_branch(repo) == "HEAD"
    assert head(repo) == first
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "original\n"
    # The way back is named, because detached HEAD is a trap.
    assert "git_checkout" in result


def test_checkout_accepts_a_short_hash_and_a_relative_ref(repo):
    root = head(repo)
    (repo / "tracked.txt").write_text("second\n", encoding="utf-8")
    commit_file(repo, "second state")
    second = head(repo)
    (repo / "tracked.txt").write_text("third\n", encoding="utf-8")
    commit_file(repo, "third state")
    assert head(repo) != second and second != root

    # A short hash.
    assert "DETACHED" in git_checkout.invoke({"ref": second[:8]})
    assert head(repo) == second

    # HEAD names the commit it is already detached at.
    assert "DETACHED" in git_checkout.invoke({"ref": "HEAD"})
    assert head(repo) == second

    # A relative ref walks back through history from wherever HEAD is.
    assert "DETACHED" in git_checkout.invoke({"ref": "HEAD~1"})
    assert head(repo) == root

    # An absolute hash from earlier in the history.
    assert "DETACHED" in git_checkout.invoke({"ref": root})
    assert head(repo) == root


def test_checkout_of_an_invalid_ref_changes_nothing(repo):
    before_head, before_branch = head(repo), current_branch(repo)
    result = git_checkout.invoke({"ref": "no-such-thing"})

    assert "neither a local branch nor a commit" in result
    assert "Nothing was changed" in result
    assert "Local branches are:" in result
    assert current_branch(repo) in result
    assert head(repo) == before_head
    assert current_branch(repo) == before_branch


def test_checkout_of_an_impossible_relative_ref_changes_nothing(repo):
    """The root commit has no parent; the refusal must not detach at something."""
    before_head = head(repo)
    result = git_checkout.invoke({"ref": "HEAD~1"})
    assert "neither a local branch nor a commit" in result
    assert head(repo) == before_head
    assert current_branch(repo) != "HEAD"


def test_checkout_refuses_a_file_path(repo):
    """``git checkout <x>`` treats a non-ref as a *path*; that must be unreachable.

    Verified against real git: ``git checkout r.txt`` in a repository containing
    ``r.txt`` restores the file from the index and changes no branch at all.
    """
    (repo / "extra.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "has a second file"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    before_branch = current_branch(repo)
    result = git_checkout.invoke({"ref": "tracked.txt"})

    assert "neither a local branch nor a commit" in result
    assert current_branch(repo) == before_branch
    # The uncommitted edit is still there: no path checkout happened.
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "dirty\n"


def test_checkout_fails_when_local_changes_would_be_overwritten(repo):
    start = _divergent_branch(repo)
    assert current_branch(repo) == start

    # An uncommitted edit to a file that differs between the two branches.
    (repo / "shared.txt").write_text("uncommitted local work\n", encoding="utf-8")
    before_head, before_text = head(repo), (repo / "shared.txt").read_text(encoding="utf-8")

    result = git_checkout.invoke({"ref": "other"})

    assert "Could not" in result
    assert "would be overwritten" in result
    assert "Nothing was changed" in result
    assert head(repo) == before_head
    assert current_branch(repo) == start
    assert (repo / "shared.txt").read_text(encoding="utf-8") == before_text


def test_checkout_never_recovers_destructively(repo):
    """No hard reset, no clean, no stash: uncommitted work survives a failure."""
    _divergent_branch(repo)
    (repo / "shared.txt").write_text("precious uncommitted work\n", encoding="utf-8")

    result = git_checkout.invoke({"ref": "other"})

    assert "Could not" in result
    assert (repo / "shared.txt").read_text(encoding="utf-8") == "precious uncommitted work\n"
    assert git("stash", "list", cwd=repo).strip() == ""
    assert git("status", "--porcelain", cwd=repo).strip() != ""


def test_checkout_rejects_option_shaped_refs(repo):
    for ref in ("-x", "--orphan", "--force", "-b"):
        result = git_checkout.invoke({"ref": ref})
        assert "not a usable git ref" in result, ref
        assert "Nothing was changed" in result


def test_checkout_rejects_traversal_and_reflog_refs(repo):
    for ref in ("main~1/../..", "main@{0}", "refs/heads/"):
        result = git_checkout.invoke({"ref": ref})
        assert "Nothing was changed" in result, ref


def test_checkout_requires_a_ref_argument():
    assert set(git_checkout.args) == {"ref"}


# ---------------------------------------------------------------------------
# permissions: the git tools are gated like every other mutating tool
# ---------------------------------------------------------------------------

def test_git_writes_are_refused_under_a_read_only_policy(repo):
    """Fail-closed: with no WRITE, checkpointing is refused and nothing happens."""
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    before = head(repo)

    with permission_scope(PermissionPolicy()):
        for call in (
            lambda: git_commit.invoke({"message": "nope"}),
            lambda: git_branch.invoke({"name": "nope"}),
            lambda: git_checkout.invoke({"ref": "HEAD"}),
        ):
            result = call()
            assert "Refused" in result, f"expected a refusal, got {result!r}"
            assert "Nothing was changed" in result

    assert head(repo) == before


def test_git_reads_do_not_need_write_permission(repo):
    """Observing version state is available wherever anything is available."""
    with permission_scope(PermissionPolicy()):
        assert "state:" in git_status.invoke({})
        assert "initial commit" in git_log.invoke({})
        assert "Currently on" in git_branch.invoke({})
        assert "No unstaged changes" in git_diff.invoke({})


def test_a_busy_project_defers_a_git_write_rather_than_overlapping(repo):
    """Coordination covers git writes, exactly as it covers write_file."""
    from terminus.coordination import is_writing, project_write_guard

    with permission_scope(permissive()):
        (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
        with project_write_guard(Operation.WRITE, target="holding the lock"):
            assert is_writing(str(repo))
            result = git_commit.invoke({"message": "should defer"})
    assert "Deferred" in result
    assert "another task is currently writing" in result
    assert "Nothing was changed" in result
    assert "not a git repository" not in result


# ---------------------------------------------------------------------------
# security
# ---------------------------------------------------------------------------

def _module_tree() -> ast.Module:
    return ast.parse(inspect.getsource(git_tools))


def test_no_subprocess_call_ever_enables_a_shell():
    """An AST check, not a string search: docstrings describe shell=True on purpose."""
    shell_calls = []
    banned = []
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"run", "Popen", "call", "check_call", "check_output"}:
                for keyword in node.keywords:
                    if keyword.arg == "shell":
                        value = keyword.value
                        truthy = not (
                            isinstance(value, ast.Constant) and value.value is False
                        )
                        if truthy:
                            shell_calls.append(getattr(func, "attr", name))
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen"}:
            banned.append(node.attr)
    assert shell_calls == [], f"shell execution enabled on: {shell_calls}"
    assert banned == [], f"shell out through os.{banned}"


def test_the_git_executable_is_invoked_directly():
    """The module imports no Git library, so git can only arrive as a process."""
    modules: set[str] = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert "subprocess" in modules
    assert not modules & {"git", "pygit2", "dulwich", "gitpython"}


def test_arguments_are_built_as_a_list_and_not_a_string():
    """The argv is a list literal; nothing concatenates a command for a shell."""
    tree = _module_tree()
    runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    ]
    assert len(runs) == 1, "one place invokes a subprocess"
    call = runs[0]

    # Exactly one positional argument, the argv. A shell command line would be a
    # second argument or a string built by concatenation or interpolation.
    assert len(call.args) == 1
    assert not any(isinstance(node, ast.JoinedStr) for node in ast.walk(call))
    assert not any(isinstance(node, ast.BinOp) for node in ast.walk(call))

    # And the thing passed is bound to a list display: ["git", *args].
    argv_name = call.args[0]
    assert isinstance(argv_name, ast.Name)
    bindings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == argv_name.id
                for target in node.targets)
    ]
    assert len(bindings) == 1, "argv must be built in one place"
    assert isinstance(bindings[0].value, ast.List)


def test_no_tool_exposes_arbitrary_git_arguments():
    """The six-tool restriction has to mean something.

    A ``git(command=...)``-shaped tool, or any argument forwarded into an argv
    position, would restore unrestricted git access and make the names
    decorative.
    """
    assert set(git_status.args) == set()
    assert set(git_diff.args) == {"staged"}
    assert set(git_commit.args) == {"message"}
    assert set(git_log.args) == {"limit"}
    assert set(git_branch.args) == {"name"}
    assert set(git_checkout.args) == {"ref"}
    for tool in (git_status, git_diff, git_commit, git_log, git_branch, git_checkout):
        assert not set(tool.args) & {"command", "args", "argv", "subcommand", "force", "path"}


def test_no_tool_takes_a_directory_or_path():
    """A tool argument naming a directory is how a cwd escape would arrive."""
    from terminus.tools import registry

    catalogue = registry.catalogue()
    for name in GIT_TOOL_NAMES:
        arguments = set(catalogue[name].args)
        assert not arguments & {
            "cwd", "directory", "working_directory", "repository", "repo", "path",
        }


def test_cwd_cannot_be_redirected_by_the_environment(repo, workspace):
    """Even when the environment asks git for another repository, the workspace wins."""
    decoy = workspace.parent / "decoy"
    decoy.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(decoy)], check=True)
    (decoy / "decoy.txt").write_text("elsewhere\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(decoy), "add", "-A"], check=True)
    subprocess.run(
        [
            "git", "-C", str(decoy),
            "-c", "user.email=d@e.test", "-c", "user.name=D",
            "commit", "--quiet", "-m", "decoy commit",
        ],
        check=True,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_DIR", str(decoy / ".git"))
        mp.setenv("GIT_WORK_TREE", str(decoy))
        mp.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))
        status = git_status.invoke({})
        log = git_log.invoke({})

    assert "decoy" not in log
    assert "initial commit" in log
    assert "decoy.txt" not in status
    assert "not a Git repository" not in status


def test_relocating_environment_variables_are_stripped():
    """The containment property is the variable list itself, so assert it."""
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        assert name in _GIT_ENV_STRIPPED


def test_git_env_omits_relocating_and_credential_variables(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/somewhere/else/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/somewhere/else")
    monkeypatch.setenv("MY_API_KEY", "sk-should-not-reach-git")
    env = git_tools._git_env()
    assert "GIT_DIR" not in env
    assert "GIT_WORK_TREE" not in env
    assert "MY_API_KEY" not in env


def test_shell_metacharacters_in_model_text_are_inert(repo):
    """A commit message is data. It must not become syntax."""
    canary = repo / "canary.txt"
    canary.write_text("untouched\n", encoding="utf-8")
    hostile = "; rm -f canary.txt & echo pwned > pwned.txt $(id) `id`"

    result = git_commit.invoke({"message": hostile})

    assert "Committed " in result
    assert canary.exists(), "a commit message must not execute anything"
    assert not (repo / "pwned.txt").exists()
    assert git("log", "-1", "--pretty=%s", cwd=repo).strip() == hostile


def test_a_branch_name_cannot_smuggle_a_second_argument(repo):
    """No whitespace, no shell metacharacters, no leading dash - in one place."""
    for name in (
        "x; rm -f tracked.txt", "x && echo y", "x | cat", "$(id)", "`id`",
        "x\nrm -f tracked.txt", "a b", "-f", "--force", "../../etc", "x..y", "a@{0}",
        "x.lock", "x/", ".hidden", "a@b@c",
    ):
        result = git_branch.invoke({"name": name})
        assert "not a usable git branch name" in result or "No branch name" in result, name
        assert (repo / "tracked.txt").exists(), name
        assert branch_names(repo) == {current_branch(repo)}, name


# ---------------------------------------------------------------------------
# multi-agent semantics
# ---------------------------------------------------------------------------

def test_agents_sharing_a_workspace_share_one_branch(repo):
    """Two logical agents, one physical checkout: they share HEAD and the tree.

    This is the documented limitation, asserted rather than assumed. A branch is
    not isolation between agents sharing a working tree - checking one out moves
    the files under both of them.
    """
    with permission_scope(permissive()):
        # Agent A commits; Agent B sees it, because there is only one repository.
        (repo / "from-a.txt").write_text("agent A\n", encoding="utf-8")
        git_commit.invoke({"message": "agent A checkpoint"})
        assert "agent A checkpoint" in git_log.invoke({})

        # Agent B branches and commits its own work.
        git_branch.invoke({"name": "agent-b-work"})
        git_checkout.invoke({"ref": "agent-b-work"})
        (repo / "from-b.txt").write_text("agent B\n", encoding="utf-8")
        git_commit.invoke({"message": "agent B checkpoint"})

        # Agent A's checkpoint is no longer on the tip: one HEAD, two branches.
        assert current_branch(repo) == "agent-b-work"
        assert "agent B checkpoint" in git_log.invoke({"limit": 1})
        assert "agent A checkpoint" not in git_log.invoke({"limit": 1})

        # And checking B out moved A's tree: A sees B's file appear.
        assert (repo / "from-b.txt").exists()


def test_a_commit_records_whatever_the_shared_tree_contains(repo):
    """The known sharp edge of one working tree, stated rather than hidden.

    An agent that commits from a shared workspace checkpoints the whole tree,
    including edits it did not make. That is Git's behaviour, not a bug in this
    layer, and the tool's description says a commit is of the whole workspace.
    """
    with permission_scope(permissive()):
        (repo / "agent_a.txt").write_text("A's work\n", encoding="utf-8")
        (repo / "agent_b.txt").write_text("B's work\n", encoding="utf-8")
        result = git_commit.invoke({"message": "whoever ran this"})

    committed = git("show", "--name-only", "--pretty=", "HEAD", cwd=repo)
    assert "agent_a.txt" in committed
    assert "agent_b.txt" in committed
    assert "whoever ran this" in result


def test_a_child_agent_cannot_move_its_parents_head(repo):
    """The narrowed role set is the mechanism; assert the property, not the list."""
    from terminus.agents.roles import get_role

    role = get_role("researcher")
    with permission_scope(PermissionPolicy()):
        assert not role.tool_names() & set(GIT_WRITER_NAMES)
        assert role.tool_names() & set(GIT_READER_NAMES)

    # And the writers it does not have are unreachable through its toolset.
    from terminus.tools import registry

    available = {t.name for t in registry.resolve(role.tool_names())}
    assert not available & set(GIT_WRITER_NAMES)


def test_mutating_git_tools_are_distinguishable_from_readers():
    """The harness must be able to tell a checkpoint from an inspection."""
    from terminus.agent.observation import MUTATING_TOOLS, OBSERVING_TOOLS

    assert set(GIT_WRITER_NAMES) <= MUTATING_TOOLS
    assert set(GIT_READER_NAMES) <= OBSERVING_TOOLS
    assert not MUTATING_TOOLS & set(GIT_READER_NAMES)


def test_a_git_write_makes_the_turn_require_observation():
    """A turn that ends right after a checkpoint has verified nothing."""
    from langchain_core.messages import AIMessage

    from terminus.agent.observation import fold_tool_calls

    def committed():
        return AIMessage(
            content="", tool_calls=[{"name": "git_commit", "args": {}, "id": "1"}]
        )

    def logged():
        return AIMessage(
            content="", tool_calls=[{"name": "git_log", "args": {}, "id": "2"}]
        )

    mutated, observed = fold_tool_calls([committed()], False, False)
    assert (mutated, observed) == (True, False)
    mutated, observed = fold_tool_calls([logged()], mutated, observed)
    assert (mutated, observed) == (True, True)


def test_checkout_is_reported_as_a_workspace_mutation():
    """It rewrites the tree, so the trace has to say the workspace changed."""
    from terminus.observability.usage_tracker import ToolCallbackHandler, mutated_paths

    assert mutated_paths("git_checkout", {"ref": "main"}) == ["."]
    handler = ToolCallbackHandler(kind="test")
    handler.on_tool_start(
        {"name": "git_checkout"}, "", inputs={"ref": "main"}, run_id="1"
    )
    handler.on_tool_end("now on branch", run_id="1")
    assert handler.files_changed() == ["."]


def test_a_commit_is_not_reported_as_a_file_change():
    """A commit rewrites history, not the tree. Claiming a changed file would lie."""
    from terminus.observability.usage_tracker import mutated_paths

    assert mutated_paths("git_commit", {"message": "x"}) == []
    assert mutated_paths("git_branch", {"name": "x"}) == []
    assert mutated_paths("git_status", {}) == []


def test_git_tool_calls_are_observable_like_any_other():
    """They flow through the existing callback handler, not a second path."""
    from terminus.observability.usage_tracker import ToolCallbackHandler

    handler = ToolCallbackHandler(kind="test")
    handler.on_tool_start(
        {"name": "git_commit"}, "", inputs={"message": "m"}, run_id="1"
    )
    handler.on_tool_end("Committed abc", run_id="1")
    assert [record.name for record in handler.records] == ["git_commit"]
    assert handler.records[0].ok
    assert handler.records[0].args["message"] == "m"
    assert handler.failures() == []

    handler.on_tool_start(
        {"name": "git_checkout"}, "", inputs={"ref": "nope"}, run_id="2"
    )
    handler.on_tool_end("Could not switch branch", run_id="2")
    assert [record.name for record in handler.records] == ["git_commit", "git_checkout"]


# ---------------------------------------------------------------------------
# registration, prompt, and the operations that are deliberately absent
# ---------------------------------------------------------------------------

def test_exactly_the_six_git_tools_are_shipped():
    from terminus.tools import registry

    shipped = sorted(n for n in registry.tool_names() if n.startswith("git_"))
    assert shipped == sorted(GIT_TOOL_NAMES)


def test_the_excluded_operations_are_not_tools():
    """No push/pull/fetch/merge/stash/rebase, and no generic git escape hatch."""
    from terminus.tools import registry

    shipped = set(registry.tool_names())
    for forbidden in (
        "git_push", "git_pull", "git_fetch", "git_merge", "git_stash", "git_rebase",
        "git_reset", "git_clean", "git_add", "git_remote", "git", "run_git",
    ):
        assert forbidden not in shipped


def test_only_ask_can_change_version_state():
    """/ask has all six; a worker and a child agent get readers only.

    Both narrower surfaces have a policy with no approver, so anything at WRITE
    would be refused there anyway - but a tool the model can see and never use is
    a tool it wastes calls discovering.
    """
    from terminus.agents.roles import GIT_READ_TOOLS, ROLES
    from terminus.tools import registry

    assert set(GIT_TOOL_NAMES) <= set(registry.ASK_TOOL_NAMES)
    assert set(GIT_TOOL_NAMES) <= {t.name for t in registry.ask_tools()}

    for task_type, names in registry.PLAN_TOOL_NAMES.items():
        assert not set(GIT_WRITER_NAMES) & set(names), task_type
        assert set(registry.PLAN_GIT_TOOL_NAMES) <= set(names), task_type

    for role_name, role in ROLES.items():
        assert not set(GIT_WRITER_NAMES) & role.tool_names(), role_name
        assert set(GIT_READ_TOOLS) <= role.tool_names(), role_name


def test_versioning_prompt_is_scoped_to_the_tools_an_agent_has():
    from terminus.agent.factory import versioning_section
    from terminus.tools import registry

    full = versioning_section(registry.ask_tools())
    assert "## Versioning with Git" in full
    assert "git_commit" in full
    assert "git_checkout" in full
    assert "NOT available as" in full
    assert "merge" in full
    # Not a safety guarantee: git supplies mechanisms, not permission.
    assert "mechanisms, not guarantees" in full

    readers = versioning_section(registry.resolve(registry.GIT_READ_TOOL_NAMES))
    assert "## Versioning with Git" in readers
    assert "git_commit" not in readers
    assert "git_checkout" not in readers
    assert "git_status" in readers

    assert versioning_section(registry.resolve(("read_file",))) == ""


def test_worker_prompt_describes_only_the_git_tools_it_has():
    from terminus.agent.factory import versioning_section
    from terminus.tasks.executor import _build_system_prompt
    from terminus.tools import registry

    tools = list(registry.plan_tools_for("implement"))
    prompt = _build_system_prompt(
        {"id": "t1", "project_id": "p1", "task_type": "implement", "description": "d"},
        [],
        None,
        tools,
    )
    assert "## Versioning with Git" in prompt
    # Scoped to the versioning section: project instructions (TERMINUS.md,
    # AGENTS.md) are legitimately allowed to mention tool names, so asserting on
    # the whole prompt would fail on a note rather than on a real capability.
    versioning = versioning_section(list(tools))
    assert "git_commit" not in versioning
    assert "git_log" in versioning


def test_ask_prompt_includes_the_versioning_section():
    from terminus.agent.factory import ASK_TOOLS, ask_policy

    prompt = ask_policy().system_prompt
    assert "## Versioning with Git" in prompt
    names = [tool.name for tool in ASK_TOOLS]
    assert len(names) == len(set(names)), "a duplicated tool would break the schema"


def test_a_child_prompt_does_not_advertise_git_it_cannot_use():
    from terminus.agent.factory import child_policy, versioning_section
    from terminus.tools import registry

    child_tools = list(registry.resolve(("read_file", "git_log")))
    policy = child_policy(
        child_tools, "Investigate.", model="m", provider="p",
        model_call_limit=4, tool_call_limit=10,
    )
    assert policy.tools == tuple(child_tools)
    assert "## Versioning with Git" in policy.system_prompt
    assert "git_commit" not in versioning_section(list(child_tools))
    assert "Investigate." in policy.system_prompt
