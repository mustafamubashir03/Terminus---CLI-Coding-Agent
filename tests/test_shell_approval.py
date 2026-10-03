"""Tests for the approval UX and for who owns the permission policy.

The security property under test: the *runtime* decides what may run, and the
decision depends only on the policy of the execution that is running - never on
anything the model supplies, and never on which agent happened to be built last.
"""

from __future__ import annotations

import asyncio
import builtins
import io
import sys

import pytest

from terminus.agent import factory
from terminus.execution import ask_context, execution_scope
from terminus.permissions import (
    PermissionLevel,
    PermissionPolicy,
    get_permission_policy,
)
from terminus.tools import shell_tools
from terminus.tools.filesystem_tools import write_file
from terminus.tools.shell_tools import run_command


def call(command, working_directory=None):
    return run_command.invoke({"command": command, "working_directory": working_directory})


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture(autouse=True)
def as_human(monkeypatch):
    """Pretend a human is at a terminal, so the approval UX is reachable.

    Tests that care about the no-terminal case install io.StringIO instead.
    """
    monkeypatch.setattr(sys, "stdin", FakeTTY())


# ---------------------------------------------------------------------------
# the interactive approver
# ---------------------------------------------------------------------------

@pytest.fixture
def answers(monkeypatch):
    """Feed a list of answers to the blocking input() prompt."""
    queue = list(answers.given)
    monkeypatch.setattr(builtins, "input", lambda *a: queue.pop(0))
    return queue


@pytest.mark.parametrize("given,expected", [
    (["y"], True),
    (["Y"], True),
    (["yes"], True),
    (["YES"], True),
    (["  yes  "], True),
])
def test_approver_accepts_affirmative_answers(monkeypatch, given, expected):
    monkeypatch.setattr(builtins, "input", lambda *a: given[0])
    assert factory.interactive_approver("npm install", "/x",
                                        PermissionLevel.WRITE, "needs approval") is expected


@pytest.mark.parametrize("given", [["n"], ["no"], [""], ["maybe"], ["yep"]])
def test_approver_treats_anything_else_as_no(monkeypatch, given):
    monkeypatch.setattr(builtins, "input", lambda *a: given[0])
    assert factory.interactive_approver("npm install", "/x",
                                        PermissionLevel.WRITE, "needs approval") is False


def test_approver_denies_on_eof(monkeypatch):
    def raise_eof(*a):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    assert factory.interactive_approver("rm -rf /", "/x",
                                        PermissionLevel.DESTRUCTIVE, "r") is False


def test_approver_denies_on_interrupt(monkeypatch):
    def raise_kb(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", raise_kb)
    assert factory.interactive_approver("rm -rf /", "/x",
                                        PermissionLevel.DESTRUCTIVE, "r") is False


def test_approval_prompt_starts_on_a_fresh_line(monkeypatch, capsys):
    """The answer may still be streaming, and _write adds no newline."""
    monkeypatch.setattr(builtins, "input", lambda *a: "y")
    factory.interactive_approver("npm install", "/x", PermissionLevel.WRITE, "because")
    out = capsys.readouterr().out
    assert out.startswith("\n"), "prompt must not be glued to streamed model text"
    assert "npm install" in out
    assert "/x" in out


def test_approval_prompt_shows_level_command_and_reason(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda *a: "n")
    factory.interactive_approver("rm -rf /", "/proj", PermissionLevel.DESTRUCTIVE, "wipes data")
    out = capsys.readouterr().out
    assert "destructive" in out
    assert "rm -rf /" in out
    assert "/proj" in out
    assert "wipes data" in out


# ---------------------------------------------------------------------------
# the policy belongs to the execution, not to the agent
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_build(monkeypatch):
    """Build an /ask agent without a real LLM, checkpointer or graph."""
    captured = {}

    monkeypatch.setattr(factory, "get_llm", lambda: "LLM")
    monkeypatch.setattr(factory, "get_summarization_middleware", lambda: "MW")
    monkeypatch.setattr(factory, "_build_system_prompt", lambda *_args, **_kwargs: "PROMPT")

    async def fake_checkpointer():
        return "CHECKPOINTER"

    monkeypatch.setattr(factory, "get_checkpointer", fake_checkpointer)

    def fake_create_agent(llm, tools, system_prompt, checkpointer, middleware):
        captured.update(tools=tools, system_prompt=system_prompt,
                        checkpointer=checkpointer, middleware=middleware)
        return "AGENT"

    monkeypatch.setattr(factory, "create_agent", fake_create_agent)

    from terminus.agent.factory import ask_policy

    async def run(policy=None):
        return await factory.build_agent(policy or ask_policy())

    run.captured = captured
    return run


def test_ask_policy_asks_a_human_when_interactive(monkeypatch):
    monkeypatch.setattr(sys, "stdin", FakeTTY())
    policy = factory.ask_permission_policy(interactive=True)
    assert policy.approver is factory.interactive_approver
    assert PermissionLevel.READ_ONLY in policy.auto_approve
    # WRITE too, or every ordinary edit would prompt
    assert PermissionLevel.WRITE in policy.auto_approve
    # nothing denied outright, so the human is actually asked
    assert policy.deny_levels == frozenset()


def test_ask_policy_denies_writes_when_non_interactive():
    policy = factory.ask_permission_policy(interactive=False)
    assert policy.approver is None, "no human is present to ask"
    assert policy.auto_approve == frozenset({PermissionLevel.READ_ONLY})
    assert PermissionLevel.WRITE in policy.deny_levels
    assert PermissionLevel.DESTRUCTIVE in policy.deny_levels


def test_a_blocking_prompt_is_never_used_without_a_terminal(monkeypatch):
    """interactive=True must not mean 'block forever' in a pipe or CI job."""
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    policy = factory.ask_permission_policy(interactive=True)
    assert policy.approver is None
    assert PermissionLevel.WRITE in policy.deny_levels


def test_the_scope_is_what_authorises_tools(tmp_path, workspace):
    """A non-interactive /ask run cannot write, because of the scope."""
    victim = tmp_path / "x.txt"
    with execution_scope(ask_context(factory.ask_permission_policy(interactive=False))):
        out = write_file.invoke({"file_path": str(victim), "content": "nope"})
    assert out.startswith("Refused:")
    assert not victim.exists()


def test_the_scope_is_restored_afterwards(tmp_path, workspace):
    outside = get_permission_policy()
    victim = tmp_path / "y.txt"
    with execution_scope(ask_context(factory.ask_permission_policy(interactive=False))):
        write_file.invoke({"file_path": str(victim), "content": "nope"})
    assert get_permission_policy() is outside
    # and with no scope at all the fail-closed default still refuses a write
    assert write_file.invoke(
        {"file_path": str(tmp_path / "z.txt"), "content": "nope"}
    ).startswith("Refused:")


def test_build_agent_does_not_touch_the_policy(stub_build):
    """Building an agent must not grant it anything."""
    before = get_permission_policy()
    asyncio.run(stub_build())
    assert get_permission_policy() is before


def test_build_agent_passes_the_real_tool_list(stub_build):
    asyncio.run(stub_build())
    assert stub_build.captured["tools"] == list(factory.ASK_TOOLS)


def test_a_policy_replaces_the_default_tool_list(stub_build):
    """The hook a delegated child uses to get a narrower toolset."""
    from terminus.agent.factory import AgentPolicy

    asyncio.run(stub_build(AgentPolicy(tools=(run_command,), system_prompt="P")))
    assert stub_build.captured["tools"] == [run_command]


def test_build_agent_still_installs_a_checkpointer(stub_build):
    """The /ask memory guarantee must survive this change."""
    asyncio.run(stub_build())
    assert stub_build.captured["checkpointer"] == "CHECKPOINTER"


def test_budget_middleware_is_registered(stub_build):
    asyncio.run(stub_build())
    middleware = stub_build.captured["middleware"]
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
    )

    model_limits = [m for m in middleware if isinstance(m, ModelCallLimitMiddleware)]
    tool_limits = [m for m in middleware if isinstance(m, ToolCallLimitMiddleware)]
    assert len(model_limits) == 1
    # one global budget plus the tighter per-tool search budget
    assert len(tool_limits) == 2
    assert any(m.tool_name is None for m in tool_limits)
    assert any(m.tool_name == "search_codebase" for m in tool_limits)
    # and they must not abort the run
    assert all(m.exit_behavior == "continue" for m in tool_limits)


def test_human_is_present_is_false_without_a_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert factory.human_is_present() is False


def test_human_is_present_is_true_for_a_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", FakeTTY())
    assert factory.human_is_present() is True


def test_approver_never_blocks_when_there_is_no_tty(monkeypatch):
    """The live smoke test hung here: input() on a pipe blocks forever."""
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    def explode(*a):
        raise AssertionError("input() must not be called without a terminal")

    monkeypatch.setattr(builtins, "input", explode)
    assert factory.interactive_approver("npm install", "/x",
                                        PermissionLevel.WRITE, "r") is False


# ---------------------------------------------------------------------------
# a blocking prompt is only safe when a human is there
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# policy is the only thing that matters
# ---------------------------------------------------------------------------

def test_identical_command_runs_or_not_purely_by_policy(tmp_path):
    """Same command, two policies, opposite outcomes - nothing else differs."""
    command = f'"{__import__("sys").executable}" -c "print(1)"'

    shell_tools.set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,), approver=None,
        deny_levels=(PermissionLevel.WRITE, PermissionLevel.DESTRUCTIVE)))
    assert call(command).startswith("Refused:")

    # `python -c` is an unrecognised executable, so it needs DESTRUCTIVE
    # approved. What this test asserts is that the *policy* decides, not where
    # the command happens to fall.
    shell_tools.set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE,
                      PermissionLevel.DESTRUCTIVE),
        approver=None, deny_levels=()))
    assert not call(command).startswith("Refused:")
