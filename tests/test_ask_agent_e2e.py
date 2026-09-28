"""End-to-end tests: drive the REAL /ask graph with a scripted model.

These are the "real scenarios" checks. Nothing here is mocked except the LLM
itself - the graph, the middleware, the tool node, the permission policy and
every tool are the production ones. The scripted model emits tool calls, so the
agent genuinely inspects, modifies, runs, observes and repairs.

Scenarios covered:
  1. inspect -> modify -> run -> observe -> verify (the full coding loop)
  2. a failing command is observed, repaired, and re-run until it passes
  3. Qdrant is down: the run continues and still completes
  4. a destructive command is refused and the agent recovers without it
  5. the global tool budget stops a runaway loop but still returns an answer
"""

from __future__ import annotations

import sys

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain.agents.middleware import ToolCallLimitMiddleware

from terminus.agent import factory
from terminus.permissions import PermissionLevel, PermissionPolicy
from terminus.tools import shell_tools

PY = sys.executable


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed list of messages, one per model call.

    GenericFakeChatModel already implements bind_tools as a no-op, which is all
    create_agent needs, so this is the smallest possible scriptable tool-calling
    model. The call counter lives outside the pydantic model to avoid field
    validation on assignment.
    """

    def __init__(self, script: list[AIMessage], counter: list[int] | None = None):
        super().__init__(messages=iter(script))
        object.__setattr__(self, "_script", list(script))
        object.__setattr__(self, "_counter", counter if counter is not None else [0])

    @property
    def calls(self) -> int:
        return self._counter[0]

    def bind_tools(self, tools, **kwargs):
        """The script already contains the tool calls; binding is a no-op."""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._counter[0] += 1
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except StopIteration:
            return super()._generate([], stop=stop, run_manager=run_manager, **kwargs)


def build(script, tmp_path, *, approve_writes: bool = True, tool_limit: int = 40):
    """Assemble a real /ask graph around a scripted model."""
    auto = (
        (PermissionLevel.READ_ONLY, PermissionLevel.WRITE)
        if approve_writes
        else (PermissionLevel.READ_ONLY,)
    )
    shell_tools.set_permission_policy(PermissionPolicy(
        auto_approve=auto,
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    ))

    from langchain.agents import create_agent

    from terminus.memory.short_term import get_summarization_middleware

    return create_agent(
        ScriptedModel(script),
        tools=list(factory.ASK_TOOLS),
        system_prompt="You are a coding agent working in a temporary directory.",
        checkpointer=None,
        middleware=[
            ToolCallLimitMiddleware(tool_name=None, run_limit=tool_limit,
                                    exit_behavior="continue"),
            ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=4,
                                    exit_behavior="continue"),
            get_summarization_middleware(),
        ],
    )


def transcript(agent, prompt="do the work"):
    """Run the graph and collect (tool_name, args) pairs the MODEL requested."""
    result = agent.invoke({"messages": [("user", prompt)]})
    calls = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            calls.append((call["name"], call["args"]))
    return result, calls


def executed(result) -> list:
    """Tool calls that actually ran.

    With exit_behavior="continue" the middleware does not stop the model from
    asking; it blocks the excess call and returns an error ToolMessage. So the
    number of tool_calls in the history is NOT the number of executions.
    """
    return [m for m in result["messages"]
            if m.type == "tool" and getattr(m, "status", None) == "success"]


def blocked(result) -> list:
    return [m for m in result["messages"]
            if m.type == "tool" and getattr(m, "status", None) == "error"]


# ---------------------------------------------------------------------------
# scenario 1 + 2: the full coding loop, including a failure and a repair
# ---------------------------------------------------------------------------

def test_full_coding_loop_inspect_modify_run_observe_verify(tmp_path):
    """Write a broken script, run it, read the failure, fix it, run it again."""
    script = tmp_path / "calc.py"
    script.write_text("print(2 + 2)\nprint('BUG')\n", encoding="utf-8")

    agent = build([
        # 1. inspect
        tool_call("read_file", {"file_path": str(script)}, "c1"),
        # 2. run it -> observe the output
        tool_call("run_command", {
            "command": f'"{PY}" "{script}"', "working_directory": str(tmp_path),
        }, "c2"),
        # 3. modify
        tool_call("edit_file", {
            "file_path": str(script), "old_text": "print('BUG')", "new_text": "print('FIXED')",
        }, "c3"),
        # 4. re-run to verify
        tool_call("run_command", {
            "command": f'"{PY}" "{script}"', "working_directory": str(tmp_path),
        }, "c4"),
        final("Fixed and verified."),
    ], tmp_path)

    result, calls = transcript(agent)
    assert [name for name, _ in calls] == [
        "read_file", "run_command", "edit_file", "run_command",
    ]

    # the file really was changed on disk
    assert "FIXED" in script.read_text(encoding="utf-8")
    assert "BUG" not in script.read_text(encoding="utf-8")

    # the second run's result really shows the fixed output
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    last = tool_messages[-1].content
    assert "FIXED" in last
    assert "exit code 0 (success)" in last


def test_failing_command_is_observed_and_repaired(tmp_path):
    """A non-zero exit is reported as data the agent can act on, not an error."""
    target = tmp_path / "check.py"
    target.write_text("raise SystemExit(7)\n", encoding="utf-8")

    agent = build([
        tool_call("run_command", {
            "command": f'"{PY}" "{target}"', "working_directory": str(tmp_path),
        }, "c1"),
        tool_call("write_file", {
            "file_path": str(target), "content": "print('all good')\n",
        }, "c2"),
        tool_call("run_command", {
            "command": f'"{PY}" "{target}"', "working_directory": str(tmp_path),
        }, "c3"),
        final("Repaired and passing."),
    ], tmp_path)

    result, calls = transcript(agent)
    assert [name for name, _ in calls] == ["run_command", "write_file", "run_command"]

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "exit code 7 (failure)" in tool_messages[0].content
    assert "exit code 0 (success)" in tool_messages[-1].content
    assert "all good" in tool_messages[-1].content


# ---------------------------------------------------------------------------
# scenario 3: retrieval is down, the run survives
# ---------------------------------------------------------------------------

def test_run_continues_when_the_vector_store_is_unavailable(tmp_path, monkeypatch):
    from terminus.tools import codebase_tool

    def boom(*a, **k):
        raise RuntimeError("ResponseHandlingException: connection reset")

    monkeypatch.setattr(codebase_tool, "get_retriever", boom)

    script = tmp_path / "ok.py"
    script.write_text("print('still working')\n", encoding="utf-8")

    agent = build([
        tool_call("search_codebase", {"query": "anything"}, "c1"),
        # the agent must be able to continue with the local tools
        tool_call("run_command", {
            "command": f'"{PY}" "{script}"', "working_directory": str(tmp_path),
        }, "c2"),
        final("Recovered without search."),
    ], tmp_path)

    result, calls = transcript(agent)
    assert [name for name, _ in calls] == ["search_codebase", "run_command"]
    assert result["messages"][-1].content == "Recovered without search."

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert "unavailable" in tool_messages[0].content
    assert "still working" in tool_messages[1].content


# ---------------------------------------------------------------------------
# scenario 4: the permission boundary holds inside a real run
# ---------------------------------------------------------------------------

def test_destructive_command_is_refused_and_the_agent_recovers(tmp_path):
    victim = tmp_path / "precious.txt"
    victim.write_text("do not delete", encoding="utf-8")

    agent = build([
        tool_call("run_command", {
            "command": f'del "{victim}"', "working_directory": str(tmp_path),
        }, "c1"),
        tool_call("read_file", {"file_path": str(victim)}, "c2"),
        final("Refused; file is intact."),
    ], tmp_path, approve_writes=False)

    result, calls = transcript(agent)
    assert [name for name, _ in calls] == ["run_command", "read_file"]
    assert victim.read_text(encoding="utf-8") == "do not delete", "must not be deleted"

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages[0].content.startswith("Refused:")
    assert "destructive" in tool_messages[0].content
    # the agent could still read the file afterwards
    assert "do not delete" in tool_messages[1].content


# ---------------------------------------------------------------------------
# scenario 5: the budget is a backstop, not a wall
# ---------------------------------------------------------------------------

def test_global_tool_budget_stops_a_runaway_loop_but_still_answers(tmp_path, monkeypatch):
    script = tmp_path / "ping.py"
    script.write_text("print('x')\n", encoding="utf-8")

    # Count what actually reached the operating system. The summarisation
    # middleware can compact the message history, so counting ToolMessages after
    # the fact is not reliable.
    real_run = shell_tools.subprocess.run
    spawned: list[str] = []

    def counting_run(command, **kwargs):
        spawned.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(shell_tools.subprocess, "run", counting_run)

    # a model that never stops asking (unique ids, as a real model would produce)
    loop = [tool_call("run_command", {
        "command": f'"{PY}" "{script}"', "working_directory": str(tmp_path),
    }, f"c{i}") for i in range(50)]
    agent = build(loop + [final("gave up")], tmp_path, tool_limit=5)

    result, _ = transcript(agent)
    assert len(spawned) == 5, "the global budget must stop real execution at 5"
    # and the run still ends with an answer rather than an exception
    assert result["messages"][-1].content == "gave up"


def test_search_codebase_has_its_own_tighter_budget(tmp_path, monkeypatch):
    from terminus.tools import codebase_tool

    monkeypatch.setattr(codebase_tool, "get_retriever",
                        lambda: (lambda q, k: [{
                            "source": "a.py", "start_line": 1, "end_line": 1,
                            "type": "f", "name": "a", "content": "x",
                        }]))
    loop = [tool_call("search_codebase", {"query": "q"}, f"s{i}") for i in range(20)]
    agent = build(loop + [final("done")], tmp_path, tool_limit=100)

    result, calls = transcript(agent)
    # the per-tool budget (4) bites before the global one (100)
    assert len(executed(result)) == 4
    assert len(blocked(result)) == 16
    assert result["messages"][-1].content == "done"
