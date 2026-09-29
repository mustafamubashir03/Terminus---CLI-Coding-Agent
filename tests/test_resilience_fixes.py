"""Tests for the Phase 5 resilience fixes found by the /ask audit.

Covers:
  * Qdrant / semantic-search failure degrades instead of ending the turn
  * an unknown skill returns an actionable string instead of raising
  * read_file is UTF-8 first (and round-trips with write_file/edit_file)
  * write_file is atomic and leaves no temp file behind
  * the tool budget is predictable (global + per-tool, continue not abort)
  * run_command is on /ask and NOT on /plan

These tests are about I/O behaviour, not permissions, so they install a policy
that allows writes explicitly. Without that they depended on whichever policy
another test file happened to leave installed.
"""

import asyncio
import os
import sys

import pytest

from terminus.permissions import (
    PermissionLevel,
    PermissionPolicy,
    get_permission_policy,
    set_permission_policy,
)
from terminus.tools import codebase_tool, filesystem_tools
from terminus.skills import skill_tools
from terminus.tools.filesystem_tools import edit_file, read_file, write_file
from terminus.skills.skill_tools import load_skill

PY = sys.executable


@pytest.fixture(autouse=True)
def writable_project():
    """Allow WRITE, refuse DESTRUCTIVE, and restore the previous policy after."""
    previous = get_permission_policy()
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    ))
    yield
    set_permission_policy(previous)


def invoke(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------------------
# semantic search degrades gracefully
# ---------------------------------------------------------------------------

def test_search_reports_unavailable_index_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(codebase_tool, "get_retriever", boom)
    out = invoke(codebase_tool.search_codebase, query="anything")
    assert "unavailable" in out.lower()
    # must steer the model to the tools that still work
    assert "grep" in out
    assert "read_file" in out
    # and must not be mistaken for "no matches"
    assert "no matches" not in out.lower()


def test_search_never_raises_for_any_transport_error(monkeypatch):
    for exc in (ConnectionError("dns"), TimeoutError("slow"), OSError("win 10054"),
                ValueError("bad payload")):
        # `_exc=exc` binds the loop variable now; a bare `exc` would be looked up
        # at call time, which only happens to be the right one here.
        def boom(*a, _exc=exc, **k):
            raise _exc

        monkeypatch.setattr(codebase_tool, "get_retriever", boom)
        out = invoke(codebase_tool.search_codebase, query="x")
        assert isinstance(out, str)
        assert "unavailable" in out.lower()


def test_search_empty_query_is_reported(monkeypatch):
    monkeypatch.setattr(codebase_tool, "get_retriever", lambda: pytest.fail("should not build"))
    assert "No search query" in invoke(codebase_tool.search_codebase, query="  ")


def test_search_no_results_suggests_grep(monkeypatch):
    monkeypatch.setattr(codebase_tool, "get_retriever", lambda: (lambda q, k: []))
    out = invoke(codebase_tool.search_codebase, query="zzz")
    assert "grep" in out


def test_search_returns_results_when_index_works(monkeypatch):
    chunks = [{
        "source": "src/a.py", "start_line": 1, "end_line": 2,
        "type": "function", "name": "a", "text": "def a(): ...",
    }]
    monkeypatch.setattr(codebase_tool, "get_retriever",
                        lambda: (lambda q, k: chunks))
    out = invoke(codebase_tool.search_codebase, query="a")
    assert "src/a.py" in out and "def a()" in out


# ---------------------------------------------------------------------------
# skills degrade gracefully
# ---------------------------------------------------------------------------

def test_unknown_skill_returns_actionable_string(monkeypatch):
    class Registry:
        def load_skill(self, name):
            raise skill_tools.SkillNotFoundError(f"no skill named {name}")

    monkeypatch.setattr(skill_tools, "_get_registry", lambda: Registry())
    out = invoke(load_skill, name="does-not-exist")
    assert isinstance(out, str)
    assert "not found" in out.lower()
    assert "does-not-exist" in out


def test_unknown_skill_does_not_raise(monkeypatch):
    class Registry:
        def load_skill(self, name):
            raise skill_tools.SkillNotFoundError("missing")

    monkeypatch.setattr(skill_tools, "_get_registry", lambda: Registry())
    # the whole point: no exception escapes
    assert isinstance(invoke(load_skill, name="missing"), str)


# ---------------------------------------------------------------------------
# encoding: read_file is UTF-8 first
# ---------------------------------------------------------------------------

def test_read_file_handles_utf8_that_windows_default_encoding_mangles(tmp_path):
    target = tmp_path / "unicode.py"
    text = "# caf\u00e9 na\u00efve \u2014 \u65e5\u672c\u8a9e \U0001f600\nvalue = '\u2713'\n"
    target.write_text(text, encoding="utf-8")

    out = invoke(read_file, file_path=str(target))
    assert "caf\u00e9" in out
    assert "\u65e5\u672c\u8a9e" in out
    assert "\U0001f600" in out
    assert "undecodable" not in out  # decoded cleanly, no replacement warning


def test_round_trip_write_then_read_preserves_text_exactly(tmp_path):
    target = tmp_path / "rt.py"
    text = "s = '\u00e9\u00e8\u00ea \u2014 \u4f60\u597d'\n"
    assert "successfully" in invoke(write_file, file_path=str(target), content=text)
    assert invoke(read_file, file_path=str(target)) == text


def test_edit_then_read_round_trips_unicode(tmp_path):
    target = tmp_path / "e.py"
    original = "name = \"caf\u00e9\"\n"
    invoke(write_file, file_path=str(target), content=original)
    out = invoke(edit_file, file_path=str(target), old_text="caf\u00e9", new_text="th\u00e9")
    assert "replaced" in out.lower()
    assert invoke(read_file, file_path=str(target)) == "name = \"th\u00e9\"\n"


def test_read_file_flags_but_still_reads_non_utf8(tmp_path):
    target = tmp_path / "latin.py"
    target.write_bytes(b"value = 'caf\xe9'\n")  # invalid UTF-8
    out = invoke(read_file, file_path=str(target))
    assert "not valid UTF-8" in out
    assert "value" in out  # still readable


def test_read_file_missing_and_empty_path():
    assert "not found" in invoke(read_file, file_path="no/such/file.py")
    assert "No file path" in invoke(read_file, file_path="  ")


# ---------------------------------------------------------------------------
# write_file atomicity
# ---------------------------------------------------------------------------

def test_write_file_leaves_no_temp_file(tmp_path):
    target = tmp_path / "a.txt"
    invoke(write_file, file_path=str(target), content="hi")
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_write_file_replaces_content_wholesale(tmp_path):
    target = tmp_path / "b.txt"
    invoke(write_file, file_path=str(target), content="first-longer-content")
    invoke(write_file, file_path=str(target), content="second")
    assert target.read_text(encoding="utf-8") == "second"


def test_write_file_failure_leaves_original_intact_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "c.txt"
    target.write_text("ORIGINAL", encoding="utf-8")

    real_replace = os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(filesystem_tools.os, "replace", boom)
    out = invoke(write_file, file_path=str(target), content="NEW")
    assert out.startswith("Could not write")
    assert "disk full" in out
    assert target.read_text(encoding="utf-8") == "ORIGINAL", "target must be untouched"
    assert list(tmp_path.iterdir()) == [target], "temp file must be cleaned up"
    monkeypatch.setattr(filesystem_tools.os, "replace", real_replace)


def test_write_file_permission_error_is_reported(tmp_path):
    out = invoke(write_file, file_path=str(tmp_path / "x" / "\0bad"), content="y")
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# tool budget is predictable
#
# The budgets are verified behaviourally in test_ask_agent_e2e.py (a runaway
# model is stopped after N real executions and the run still returns an
# answer). What is checked here is only that the wiring exists, by inspecting
# the assembled agent rather than the source text.
# ---------------------------------------------------------------------------

def _ask_middleware():
    import terminus.agent.factory as factory
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
    )

    captured = {}

    async def fake_checkpointer():
        return "CP"

    def fake_create_agent(llm, tools, system_prompt, checkpointer, middleware):
        captured["middleware"] = middleware
        return "AGENT"

    real = (factory.get_llm, factory.get_summarization_middleware,
            factory.get_checkpointer, factory.create_agent)
    factory.get_llm = lambda: "LLM"
    factory.get_summarization_middleware = lambda: "MW"
    factory.get_checkpointer = fake_checkpointer
    factory.create_agent = fake_create_agent
    try:
        asyncio.run(factory.build_agent())
    finally:
        (factory.get_llm, factory.get_summarization_middleware,
         factory.get_checkpointer, factory.create_agent) = real
    return captured["middleware"], ModelCallLimitMiddleware, ToolCallLimitMiddleware


def test_tool_budgets_are_registered_on_the_agent():
    middleware, ModelLimit, ToolLimit = _ask_middleware()

    model_limits = [m for m in middleware if isinstance(m, ModelLimit)]
    tool_limits = [m for m in middleware if isinstance(m, ToolLimit)]
    assert len(model_limits) == 1

    # one global budget plus the tighter per-tool search budget
    assert any(m.tool_name is None for m in tool_limits)
    assert any(m.tool_name == "search_codebase" for m in tool_limits)
    # and neither may abort the run
    assert all(m.exit_behavior == "continue" for m in tool_limits)


async def _async_cp():
    return "CP"


# ---------------------------------------------------------------------------
# /ask vs /plan isolation
# ---------------------------------------------------------------------------

def test_run_command_is_on_the_ask_agent():
    from terminus.agent.factory import ASK_TOOLS
    assert "run_command" in {t.name for t in ASK_TOOLS}


def test_the_two_run_commands_are_different_objects():
    from terminus.tools.terminal_tools import run_command as plan_run_command
    from terminus.tools.shell_tools import run_command as ask_run_command
    assert plan_run_command is not ask_run_command
    # only the /ask one is policy-gated
    assert "working_directory" in ask_run_command.args
