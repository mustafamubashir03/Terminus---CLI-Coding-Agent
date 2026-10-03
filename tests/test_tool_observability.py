"""Tool execution is observable, and a refusal is not a result.

Two things are checked here, both through a real agent rather than by calling
callbacks by hand, because the claim is that the harness can see what happened -
not that a function exists.

* **Observability.** LangChain's ``on_tool_*`` hooks see every call ``ToolNode``
  makes, so ``ToolCallbackHandler`` records name, arguments, timing, outcome and
  the workspace paths a call reported changing. A model call and the tool calls
  it caused are already related by LangChain's own ``parent_run_id``, so a turn
  can be traced without Terminus inventing a second id scheme.

* **Error semantics.** Three states stay distinct: a tool ran and returned a
  result (including a permission *refusal*, which is a result the model is meant
  to react to); a tool was rejected, which LangGraph represents as a
  ``ToolMessage`` with ``status="error"``; and the run failed outright. Nothing
  here needs a Terminus-specific error protocol, because the framework already
  has one.
"""

from __future__ import annotations

import sys

import anyio
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from terminus.agent.orchestrator import Outcome, TurnResult
from terminus.observability.usage_tracker import (
    ToolCallbackHandler,
    ToolRecord,
    UsageCallbackHandler,
    mutated_paths,
)
from terminus.permissions import PermissionLevel, PermissionPolicy
from terminus.tools import registry
from terminus.workspace import project_root

PY = sys.executable


@pytest.fixture(autouse=True)
def writable_workspace(tmp_path, monkeypatch):
    """tmp_path is the workspace, and writes are pre-authorised.

    Without this the writes below would be *refused*, which is a perfectly good
    outcome but not the one most of these tests are about.
    """
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    from terminus.permissions import get_permission_policy, set_permission_policy

    previous = get_permission_policy()
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    ))
    yield tmp_path
    set_permission_policy(previous)


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final(text: str) -> AIMessage:
    return AIMessage(content=text)


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed script, one message per model call, sync or async.

    ``bind_tools`` is a no-op - the script already contains the tool calls, which
    is all ``create_agent`` needs. The closing answer repeats, because the graph
    may legitimately ask for another turn (the /ask orchestrator does once after
    a mutation, to insist the model look at its own work) and a model that simply
    stopped would turn that into a run failure.

    Both ``_generate`` and ``_astream`` are implemented here rather than relying on
    the base class: ``run_turn`` streams, and the base fake model does not refill
    its iterator for the async path, which surfaces as "No generations found in
    stream" and hides whatever the test was actually about.
    """

    def __init__(self, script, tail: int = 8):
        full = (list(script) + [script[-1]] * tail) if script else []
        super().__init__(messages=iter(full))
        # after super().__init__: pydantic rebuilds __dict__, so state set before
        # it is discarded. Private attrs are not declared as fields on purpose -
        # they are test scaffolding, not model configuration.
        object.__setattr__(self, "_full", full)
        object.__setattr__(self, "_pos", 0)

    def _next(self) -> AIMessage:
        position = self._pos
        object.__setattr__(self, "_pos", position + 1)
        return self._full[min(position, len(self._full) - 1)]

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        # A generation chunk, not a bare message: the streaming consumer reads
        # `.generation_info` off whatever it is handed, and a chunk has to be an
        # AIMessageChunk to be accepted at all.
        message = self._next()
        if not isinstance(message, AIMessageChunk):
            message = AIMessageChunk(
                content=message.content,
                tool_calls=message.tool_calls,
                id=message.id,
                response_metadata=message.response_metadata,
            )
        yield ChatGenerationChunk(message=message)


def drive(script, tools, config=None, prompt="do the work"):
    """Run a real graph over *script* and return its final state."""
    agent = create_agent(
        ScriptedModel(script),
        tools=list(tools),
        system_prompt="test",
        checkpointer=None,
    )
    return agent.invoke({"messages": [("user", prompt)]}, config=config or {})


def tool_messages(result) -> list:
    return [m for m in result["messages"] if m.type == "tool"]


# ---------------------------------------------------------------------------
# what the harness sees
# ---------------------------------------------------------------------------

def test_every_tool_call_is_recorded_with_its_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    seen = ToolCallbackHandler()
    result = drive(
        [
            call("write_file", {"file_path": "notes.md", "content": "hi"}, "c1"),
            call("read_file", {"file_path": "notes.md"}, "c2"),
            final("done"),
        ],
        registry.resolve(("write_file", "read_file")),
        config={"callbacks": [seen]},
    )
    assert [r.name for r in seen.records] == ["write_file", "read_file"]
    write, read = seen.records
    assert write.args["file_path"] == "notes.md"
    assert write.tool_call_id == "c1", "traceable back to the model's tool call"
    assert write.run_id
    assert write.duration_seconds >= 0.0
    assert write.status == "success"
    assert write.result
    assert read.result == "hi"
    assert all(m.status == "success" for m in tool_messages(result))


def test_a_tool_call_is_correlated_with_the_execution_that_caused_it(tmp_path, monkeypatch):
    """LangChain already parents a tool run; recording it is all that is needed.

    The parent is the graph node that ran the tool, not the model call itself -
    LangChain nests an LLM run and a tool run as siblings under their node. What
    matters is that the framework hands out a real parent id and Terminus keeps
    it, so a tool call can be traced back to the run that issued it.
    """
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    usage, tools = UsageCallbackHandler(kind="ask"), ToolCallbackHandler(kind="ask")
    drive(
        [
            call("write_file", {"file_path": "a.txt", "content": "x"}, "c1"),
            final("done"),
        ],
        registry.resolve(("write_file",)),
        config={"callbacks": [usage, tools]},
    )
    assert len(usage.records) == 2, "one model call to ask, one to answer"
    assert all(r.run_id for r in usage.records), "model calls must be identifiable"
    record = tools.records[0]
    assert record.parent_run_id, "the framework's parent link must be kept"
    assert record.parent_run_id != record.run_id, (
        "a tool run is its own execution, nested under something else"
    )
    # the two views of the same execution are distinguishable, not conflated
    assert record.run_id not in {r.run_id for r in usage.records}


def test_filesystem_mutations_are_surfaced(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    seen = ToolCallbackHandler()
    drive(
        [
            call("write_file", {"file_path": "src/app.py", "content": "x"}, "c1"),
            call("edit_file", {"file_path": "src/app.py",
                               "old_text": "x", "new_text": "y"}, "c2"),
            call("read_file", {"file_path": "src/app.py"}, "c3"),
            final("done"),
        ],
        registry.resolve(("write_file", "edit_file", "read_file")),
        config={"callbacks": [seen]},
    )
    # only the calls that changed something, workspace-relative, in order
    assert seen.files_changed() == ["src/app.py"]
    assert seen.files_changed() == ["src/app.py"], "deduplicated"


def test_mutated_paths_is_pure_and_usable_before_or_after_the_call():
    """The answer is a function of the call, so nothing has to be listening."""
    assert mutated_paths("write_file", {"file_path": "a/b.py"}) == ["a/b.py"]
    assert mutated_paths("write_file", {"file_path": "a/b.py",
                                        "content": "x" * 10_000}) == ["a/b.py"]
    assert mutated_paths("read_file", {"file_path": "a/b.py"}) == []
    assert mutated_paths("no_such_tool", {}) == []


def test_recorded_arguments_are_bounded(tmp_path, monkeypatch):
    """Telemetry must not become a content store for generated source."""
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    seen = ToolCallbackHandler()
    drive(
        [call("write_file", {"file_path": "big.py", "content": "x" * 50_000}, "c1"),
         final("done")],
        registry.resolve(("write_file",)),
        config={"callbacks": [seen]},
    )
    assert len(seen.records[0].args["content"]) < 1_000
    assert seen.records[0].args["content"].endswith("...[truncated]")


def test_a_failing_tool_is_recorded_as_a_failure_not_a_result(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    seen = ToolCallbackHandler()
    result = drive(
        [
            call("read_file", {"file_path": str(tmp_path.parent / "outside.txt")}, "c1"),
            final("moved on"),
        ],
        registry.resolve(("read_file",)),
        config={"callbacks": [seen]},
    )
    assert len(seen.records) == 1
    record = seen.records[0]
    assert record.status == "error"
    assert not record.ok
    assert "outside the workspace" in record.error
    assert seen.failures() == [record]
    assert seen.files_changed() == []
    # and the graph state says the same thing, so the model and the harness agree
    assert tool_messages(result)[0].status == "error"


def test_a_refusal_is_a_successful_result_not_a_failure(tmp_path, monkeypatch):
    """A refused write ran. Its result was a refusal; that is not an error."""
    from terminus.permissions import get_permission_policy, set_permission_policy

    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    previous = get_permission_policy()
    set_permission_policy(PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,),
        approver=None,
        deny_levels=(PermissionLevel.WRITE, PermissionLevel.DESTRUCTIVE),
    ))
    try:
        seen = ToolCallbackHandler()
        result = drive(
            [call("write_file", {"file_path": "no.txt", "content": "x"}, "c1"),
             final("refused")],
            registry.resolve(("write_file",)),
            config={"callbacks": [seen]},
        )
    finally:
        set_permission_policy(previous)

    record = seen.records[0]
    assert record.status == "success"
    assert "Refused:" in record.result
    assert seen.failures() == []
    assert seen.files_changed() == [], "a refused write changed nothing"
    assert tool_messages(result)[0].status == "success"
    assert not (tmp_path / "no.txt").exists()


def test_a_workspace_refusal_does_not_end_the_run(tmp_path, monkeypatch):
    """The whole point of an error ToolMessage: the loop continues."""
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    result = drive(
        [
            call("read_file", {"file_path": str(tmp_path.parent / "escape.txt")}, "c1"),
            call("read_file", {"file_path": "inside.txt"}, "c2"),
            final("recovered"),
        ],
        registry.resolve(("read_file",)),
    )
    (tmp_path / "inside.txt").write_text("reachable", encoding="utf-8")
    statuses = [m.status for m in tool_messages(result)]
    assert statuses[0] == "error"
    assert result["messages"][-1].content == "recovered"


# ---------------------------------------------------------------------------
# the /ask harness
# ---------------------------------------------------------------------------

def _policy(*names):
    """A minimal policy over *names*, so the turn's tools are exactly known."""
    from terminus.agent.factory import AgentPolicy

    return AgentPolicy(
        tools=registry.resolve(names),
        system_prompt="test",
        checkpoint=False,
        summarize=False,
        tool_call_limits=(),
        model_call_limit=8,
    )


def _script(escape_first: bool = False) -> list:
    """A turn that changes a file and then reads it back."""
    script = []
    if escape_first:
        script.append(call("read_file",
                           {"file_path": str(project_root().parent / "escape.txt")},
                           "c1"))
    script += [
        call("write_file", {"file_path": "pkg/module.py", "content": "x = 1\n"}, "w1"),
        call("read_file", {"file_path": "pkg/module.py"}, "r1"),
        final("wrote it and checked"),
    ]
    return script


@pytest.fixture
def captured_turn(monkeypatch):
    """Run the real ``run_turn`` and capture the config it hands the graph.

    Only ``build_agent`` is replaced, so the orchestrator, the permission scope,
    the middleware, the ToolNode and the callbacks are all the production ones.
    """
    import terminus.agent.orchestrator as orchestrator

    captured: dict = {}

    def install(script):
        async def fake_build_agent(policy):
            agent = create_agent(
                ScriptedModel(list(script)),
                tools=list(policy.tools),
                system_prompt=policy.system_prompt,
                checkpointer=None,
            )
            original_astream = agent.astream

            def astream(*args, **kwargs):
                captured["config"] = kwargs.get("config") or {}
                return original_astream(*args, **kwargs)

            agent.astream = astream
            return agent

        monkeypatch.setattr(orchestrator, "build_agent", fake_build_agent)

    def ask(*names, script=()):
        install(script)
        return anyio.run(
            lambda: orchestrator.run_turn(
                "do it", None, interactive=False, policy=_policy(*names)
            )
        )

    ask.captured = captured
    return ask


def test_an_ask_turn_observes_tool_calls(captured_turn):
    """The handler is on the run, so the turn can report what it did.

    Asserted on the config the graph is actually invoked with, because that is
    the only point at which "the harness is observing" is decided.
    """
    result = captured_turn("write_file", "read_file", script=_script())
    assert isinstance(result, TurnResult)
    callbacks = captured_turn.captured["config"]["callbacks"]
    tools_seen = [c for c in callbacks if isinstance(c, ToolCallbackHandler)]
    assert len(tools_seen) == 1, "exactly one tool observer per turn"
    usage = [c for c in callbacks if isinstance(c, UsageCallbackHandler)]
    assert len(usage) == 1, "tool observation rides beside usage, not instead of it"
    assert tools_seen[0].kind == usage[0].kind, "one execution, one label"


def test_a_turn_result_surfaces_what_the_handler_recorded(captured_turn, monkeypatch):
    """The orchestrator turns the handler's records into the caller's result.

    Driven through the real ``run_turn`` with a stand-in handler, because the
    streamed fake model cannot emit tool calls; what is under test is the
    orchestrator's own translation. Every test above already drives a real graph
    and a real handler.
    """
    import terminus.agent.orchestrator as orchestrator

    class StubHandler(ToolCallbackHandler):
        def files_changed(self):
            return ["pkg/module.py", "docs/plan.md"]

        def failures(self):
            return [ToolRecord(name="read_file", status="error",
                               error="outside the workspace")]

    monkeypatch.setattr(orchestrator, "ToolCallbackHandler", StubHandler)
    result = captured_turn("write_file", "read_file", script=[final("done")])

    assert result.outcome is Outcome.DONE
    assert result.files_changed == ("pkg/module.py", "docs/plan.md")
    assert result.tool_failures == ("read_file: outside the workspace",)


def test_a_turn_with_nothing_to_report_says_so(captured_turn):
    result = captured_turn("read_file", script=[final("just talked")])
    assert result.files_changed == ()
    assert result.tool_failures == ()
