"""AGENTS.md: the workspace's durable memory.

The invariants that matter are that it is never lost, never silently reset, and
never stale for a whole process. Everything else is a Markdown file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from terminus.agents_md import (
    AGENT_MD_TEMPLATE,
    AGENTS_MD_FILENAME,
    agents_md_path,
    agents_md_section,
    load_agents_md,
)
from terminus.workspace import WORKSPACE_ENV_VAR


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the workspace at tmp_path, which is how a session is attached."""
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# load / create
# ---------------------------------------------------------------------------

def test_it_lives_at_the_workspace_root(workspace):
    assert agents_md_path() == (workspace / AGENTS_MD_FILENAME).resolve()


def test_a_missing_file_is_created_from_the_template(workspace):
    assert not (workspace / AGENTS_MD_FILENAME).exists()
    assert load_agents_md() == AGENT_MD_TEMPLATE
    assert (workspace / AGENTS_MD_FILENAME).read_text(encoding="utf-8") == (
        AGENT_MD_TEMPLATE
    )


def test_an_existing_file_is_returned_as_is(workspace):
    (workspace / AGENTS_MD_FILENAME).write_text("mine\n", encoding="utf-8")
    assert load_agents_md() == "mine\n"


def test_an_existing_file_is_never_overwritten_by_repeated_loads(workspace):
    path = workspace / AGENTS_MD_FILENAME
    path.write_text("# mine\nkeep me\n", encoding="utf-8")
    for _ in range(5):
        assert load_agents_md() == "# mine\nkeep me\n"
    assert path.read_text(encoding="utf-8") == "# mine\nkeep me\n"


def test_loading_is_idempotent(workspace):
    first = load_agents_md()
    assert load_agents_md() == first == load_agents_md()


def test_a_file_written_after_creation_is_seen_immediately(workspace):
    """No stale cache: the next session sees what the last one learned."""
    load_agents_md()
    path = workspace / AGENTS_MD_FILENAME
    path.write_text("## Gotchas\n- the linter lies about line length\n", encoding="utf-8")
    assert "the linter lies" in load_agents_md()


def test_a_deleted_file_is_recreated(workspace):
    load_agents_md()
    (workspace / AGENTS_MD_FILENAME).unlink()
    assert load_agents_md() == AGENT_MD_TEMPLATE


def test_the_template_names_the_four_durable_categories(workspace):
    for heading in ("## Conventions", "## Decisions", "## Gotchas", "## Active Tasks"):
        assert heading in AGENT_MD_TEMPLATE
    assert "durable memory across sessions" in AGENT_MD_TEMPLATE
    assert "loads it at the start of every session" in AGENT_MD_TEMPLATE


# ---------------------------------------------------------------------------
# workspace isolation
# ---------------------------------------------------------------------------

def test_each_workspace_has_its_own_memory(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(first))
    (first / AGENTS_MD_FILENAME).write_text("first workspace\n", encoding="utf-8")

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(second))
    assert load_agents_md() == AGENT_MD_TEMPLATE

    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(first))
    assert load_agents_md() == "first workspace\n"


def test_a_path_is_resolved_through_the_workspace_abstraction(workspace):
    """Same containment rule as every other tool path, not a hand-rolled join."""
    from terminus.workspace import WorkspaceViolation, resolve_in_workspace

    assert resolve_in_workspace(AGENTS_MD_FILENAME, workspace=workspace) == (
        workspace / AGENTS_MD_FILENAME
    ).resolve()
    with pytest.raises(WorkspaceViolation):
        resolve_in_workspace(f"../{AGENTS_MD_FILENAME}", workspace=workspace)


# ---------------------------------------------------------------------------
# errors are surfaced, not swallowed
# ---------------------------------------------------------------------------

def test_an_unreadable_file_raises_rather_than_resetting_to_the_template(workspace):
    """The worst outcome would be a silent wipe of the agent's memory."""
    path = workspace / AGENTS_MD_FILENAME
    path.write_text("precious\n", encoding="utf-8")
    path.chmod(0o000)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("this host grants read access regardless of mode")
        with pytest.raises(OSError, match="Cannot read"):
            load_agents_md()
    finally:
        path.chmod(0o600)
    assert path.read_text(encoding="utf-8") == "precious\n"


def test_a_directory_in_the_files_place_raises_rather_than_being_ignored(workspace):
    (workspace / AGENTS_MD_FILENAME).mkdir()
    with pytest.raises(OSError):
        load_agents_md()


# ---------------------------------------------------------------------------
# concurrent creation
# ---------------------------------------------------------------------------

def test_two_sessions_starting_at_once_produce_one_file(workspace):
    """The race the exclusive create exists for.

    Real subprocesses, both pointed at the same workspace and neither having
    created the file yet. Exactly one may create it; the other must read back the
    winner's copy rather than overwrite it.
    """
    program = (
        "import os, sys;"
        "sys.path.insert(0, 'src');"
        "from terminus.agents_md import load_agents_md;"
        "sys.stdout.write(load_agents_md())"
    )
    env = {**os.environ, WORKSPACE_ENV_VAR: str(workspace)}
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", program],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(6)
    ]
    results = []
    for run in runs:
        out, err = run.communicate(timeout=120)
        assert run.returncode == 0, err
        results.append(out)

    path = workspace / AGENTS_MD_FILENAME
    assert all(result == AGENT_MD_TEMPLATE for result in results)
    assert path.read_text(encoding="utf-8") == AGENT_MD_TEMPLATE


# ---------------------------------------------------------------------------
# the section handed to the model
# ---------------------------------------------------------------------------

def test_the_section_carries_the_content_and_the_guidance():
    section = agents_md_section("## Gotchas\n- the linter lies\n")
    assert "## Agent memory (AGENTS.md)" in section
    assert "the linter lies" in section
    assert "Worth keeping" in section and "Not worth keeping" in section


def test_empty_content_produces_no_section():
    assert agents_md_section("") == ""
    assert agents_md_section("   \n ") == ""


# ---------------------------------------------------------------------------
# harness integration
# ---------------------------------------------------------------------------

def test_ask_context_contains_agents_md(workspace):
    from terminus.agent.factory import ask_policy

    (workspace / AGENTS_MD_FILENAME).write_text(
        "## Decisions\n- we chose SQLite over Postgres\n", encoding="utf-8"
    )
    prompt = ask_policy().system_prompt
    assert "## Agent memory (AGENTS.md)" in prompt
    assert "we chose SQLite over Postgres" in prompt


def test_a_new_session_sees_content_the_last_one_wrote(workspace):
    """The whole point: session 2 starts with what session 1 learned."""
    from terminus.agent.factory import ask_policy

    load_agents_md()
    (workspace / AGENTS_MD_FILENAME).write_text(
        "## Conventions\n- tests live in tests/\n", encoding="utf-8"
    )
    assert "tests live in tests/" in ask_policy().system_prompt


def test_a_child_agent_inherits_it(workspace):
    """Same workspace, so the same durable memory - the child is working here."""
    from terminus.agent.factory import child_policy
    from terminus.tools import registry

    (workspace / AGENTS_MD_FILENAME).write_text(
        "## Gotchas\n- the indexer skips .terminus\n", encoding="utf-8"
    )
    policy = child_policy(
        list(registry.resolve(("read_file",))),
        "Investigate.",
        model="m", provider="p", model_call_limit=4, tool_call_limit=10,
    )
    assert "the indexer skips .terminus" in policy.system_prompt


def test_a_plan_worker_receives_it(workspace):
    from terminus.tasks.executor import _build_system_prompt

    (workspace / AGENTS_MD_FILENAME).write_text(
        "## Active Tasks\n- finish the migration\n", encoding="utf-8"
    )
    prompt = _build_system_prompt(
        {"id": "t1", "project_id": "p1", "task_type": "implement", "description": "d"},
        [], None, [],
    )
    assert "finish the migration" in prompt


def test_it_is_not_folded_into_the_cached_static_prompt(workspace):
    """Guards the staleness bug this design exists to avoid.

    The static half of the /ask prompt is cached per workspace for the life of
    the process. If AGENTS.md were folded in there, a session that learned
    something would never see its own note until the process restarted.
    """
    from terminus.agent import factory
    from terminus.cache import get_cached_prompt

    load_agents_md()
    factory._build_system_prompt()          # populates the static cache
    cached = get_cached_prompt(f"{factory._SYSTEM_PROMPT_CACHE_KEY}:{factory.project_root()}")
    assert cached, "expected the static half to be cached"
    assert "## Agent memory" not in cached

    (workspace / AGENTS_MD_FILENAME).write_text(
        "## Gotchas\n- learned after the cache was warm\n", encoding="utf-8"
    )
    assert "learned after the cache was warm" in factory._build_system_prompt()


def test_building_context_creates_the_file_when_absent(workspace):
    from terminus.agent.factory import ask_policy

    assert not (workspace / AGENTS_MD_FILENAME).exists()
    ask_policy()
    assert (workspace / AGENTS_MD_FILENAME).exists()


# ---------------------------------------------------------------------------
# interaction with TERMINUS.md
# ---------------------------------------------------------------------------

def test_it_is_separate_from_terminus_md(workspace):
    """Both are supported; they are not the same file and neither replaces the other."""
    from terminus.context.environment import build_startup_context

    (workspace / "TERMINUS.md").write_text(
        "Always run the linter before committing.\n", encoding="utf-8"
    )
    (workspace / AGENTS_MD_FILENAME).write_text(
        "## Gotchas\n- the linter lies about line length\n", encoding="utf-8"
    )

    startup = build_startup_context(workspace)
    assert "Always run the linter before committing." in startup
    assert "the linter lies" not in startup

    from terminus.agent.factory import ask_policy

    prompt = ask_policy().system_prompt
    assert "Always run the linter before committing." in prompt, "TERMINUS.md still loads"
    assert "the linter lies about line length" in prompt, "AGENTS.md loads too"


# ---------------------------------------------------------------------------
# the model maintains it with the ordinary tools
# ---------------------------------------------------------------------------

def test_the_model_can_update_it_with_the_filesystem_tools(workspace):
    """No special write API, by design: edit_file is enough."""
    from terminus.permissions import (
        PermissionLevel,
        PermissionPolicy,
        get_permission_policy,
        set_permission_policy,
    )
    from terminus.tools.filesystem_tools import edit_file, read_file

    previous = get_permission_policy()
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None, deny_levels=(PermissionLevel.DESTRUCTIVE,),
    ))
    try:
        load_agents_md()
        note = "## Gotchas\n- the linter lies about line length\n"
        assert "Replaced 1 occurrence" in edit_file.invoke(
            {"file_path": AGENTS_MD_FILENAME, "old_text": "## Gotchas",
             "new_text": note.rstrip("\n")}
        )
        assert "the linter lies" in read_file.invoke({"file_path": AGENTS_MD_FILENAME})
        assert "the linter lies" in load_agents_md()
    finally:
        set_permission_policy(previous)


def test_the_harness_itself_never_rewrites_existing_content(workspace):
    """Only the model writes to this file. The harness creates it once."""
    path = workspace / AGENTS_MD_FILENAME
    path.write_text("hand written knowledge\n", encoding="utf-8")
    before = path.stat().st_mtime_ns

    for _ in range(3):
        load_agents_md()

    assert path.read_text(encoding="utf-8") == "hand written knowledge\n"
    assert path.stat().st_mtime_ns == before, "the file must not be touched"


def test_it_is_a_plain_file_in_the_workspace(workspace):
    """No sidecar state: no database, no index, no lock file."""
    load_agents_md()
    assert sorted(p.name for p in workspace.iterdir()) == [AGENTS_MD_FILENAME]


def test_it_is_inside_the_workspace_so_the_model_can_reach_it(workspace):
    """It is ordinary workspace state, reachable by the ordinary tools."""
    from terminus.permissions import PermissionPolicy, set_permission_policy
    from terminus.tools.filesystem_tools import file_exists

    load_agents_md()
    previous = PermissionPolicy()
    set_permission_policy(previous)
    assert file_exists.invoke({"file_path": AGENTS_MD_FILENAME}) == (
        f"File exists: {AGENTS_MD_FILENAME}"
    )
    assert Path(agents_md_path()).is_relative_to(workspace.resolve())
