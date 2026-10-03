"""The main agent's tool loop, end to end, with a deterministic fake model.

    main model -> tool call -> real spawn_agent tool -> real child ->
    ToolMessage -> main model -> final answer

No provider is contacted. The real agent graph, the real ``spawn_agent`` tool,
the real child machinery and the real execution scope are used; only the model
is scripted.

Three harness details are load-bearing, each found the hard way:

* **One model instance, bound to the module, patched at the right place.** The
  agent resolves the model through ``terminus.agent.factory.get_llm``, so that
  is the attribute to replace. Patching ``terminus.llm.factory.get_llm`` does
  not affect it, because ``factory`` holds its own reference.
* **One instance across all turns.** A model constructed per turn resets the
  script cursor and the agent re-issues the same tool call forever.
* **``bind_tools`` must be implemented.** ``BaseChatModel.bind_tools`` raises
  ``NotImplementedError``, and the agent binds before every model call.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, List

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from terminus.agents.spawn import _reset_write_scopes
from terminus.execution import (
    MAX_CHILDREN_PER_PARENT,
    ask_context,
    current_execution,
    execution_scope,
)
from terminus.observability.usage_tracker import clear_child_events
from terminus.permissions import PermissionLevel, PermissionPolicy

# --- scripted state, at module scope so no model state can be lost ----------

_SCRIPT: List[Any] = []
_CURSOR = 0
_OFFERED: List[str] = []
_RECEIVED: List[Any] = []
_CHILD_THREADS: List[str] = []
_SEEN_CONTEXTS: List[Any] = []
_PARENT_THREAD: dict = {}


def _reset(script: List[Any]) -> None:
    global _SCRIPT, _CURSOR, _OFFERED, _RECEIVED
    _SCRIPT, _CURSOR, _OFFERED, _RECEIVED = list(script), 0, [], []


class ScriptedModel(BaseChatModel):
    """Replays a fixed script of assistant messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "terminus-scripted"

    def bind_tools(self, tools, **_kwargs):
        for tool in tools or []:
            name = getattr(tool, "name", str(tool))
            if name not in _OFFERED:
                _OFFERED.append(name)
        return self

    def _step(self, messages, kwargs) -> ChatResult:
        global _CURSOR, _RECEIVED
        for tool in kwargs.get("tools") or []:
            name = getattr(tool, "name", str(tool))
            if name not in _OFFERED:
                _OFFERED.append(name)
        _RECEIVED = list(messages)
        message = _SCRIPT[min(_CURSOR, len(_SCRIPT) - 1)]
        _CURSOR += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._step(messages, kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._step(messages, kwargs)


def sync_async(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def _no_approval_policy() -> PermissionPolicy:
    return PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
        approver=None,
        deny_levels=(PermissionLevel.DESTRUCTIVE,),
    )


def _spin(count: int = 40) -> List[AIMessage]:
    """A model that keeps asking to delegate, emitting a fresh id each turn.

    The ids must be unique. A real model never reuses a tool_call_id, and
    replaying one id every turn is not a model that keeps delegating - it is a
    malformed transcript the agent cannot keep up with, which is a different
    thing entirely and does not test the budget.
    """
    return [_call("spin", "researcher", f"spin-{i}") for i in range(count)]


def _call(task: str, role: str = "researcher", call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "spawn_agent",
            "args": {"task": task, "role": role},
            "id": call_id,
        }],
    )


@pytest.fixture
def stub_child_agent(monkeypatch):
    """Stub the compiled agent, not the child's runtime.

    The child's ``ExecutionContext`` - its own identity, its own policy, its own
    thread - is created inside ``_default_runner``. Stubbing that function would
    delete the very thing under test, so only ``build_agent`` is replaced. The
    real child path therefore runs, on a different thread, under a different
    authority, and the parent and child are told apart by their thread id.
    """
    import terminus.agent.factory as factory
    from langchain_core.messages import AIMessage as _AI

    _reset_write_scopes()
    clear_child_events()
    _CHILD_THREADS.clear()
    _SEEN_CONTEXTS.clear()

    real_build_agent = factory.build_agent

    class _RecordingAgent:
        async def ainvoke(self, payload, config=None, **_kw):
            thread = (config or {}).get("configurable", {}).get("thread_id", "")
            context = current_execution()
            _CHILD_THREADS.append((thread, context))
            _SEEN_CONTEXTS.append(context)
            return {"messages": [_AI(content="child finished its work")]}

    is_child = []

    async def fake_build_agent(policy):
        # The parent runs ask_policy; the child runs child_policy, which is
        # built from a narrower tool list. That difference is how the two are
        # told apart here, and it is the same distinction the runtime makes.
        if len(policy.tools) == len(factory.ASK_TOOLS):
            return await real_build_agent(policy)
        is_child.append(sorted(t.name for t in policy.tools))
        return _RecordingAgent()

    monkeypatch.setattr(factory, "build_agent", fake_build_agent)
    yield
    _reset_write_scopes()
    clear_child_events()


async def _drive(script: List[Any], user: str, thread_id: str | None = None):
    """Patch in the scripted model and run one real agent turn.

    The thread id must be unique per run. The checkpointer is a *persistent*
    store, so reusing a fixed id replays the previous run's conversation into
    this one - which makes the model re-issue its tool call and the test observe
    an ever-growing pile of ToolMessages across runs.
    """
    import uuid

    import terminus.agent.factory as factory

    if thread_id is None:
        thread_id = f"loop-{uuid.uuid4().hex[:12]}"
    _PARENT_THREAD["thread_id"] = thread_id
    _reset(script)
    model = ScriptedModel()
    # factory holds its own reference to get_llm; replace that one.
    factory.get_llm = lambda: model  # type: ignore[assignment]
    agent = await asyncio.wait_for(factory.build_agent(factory.ask_policy()), timeout=60)
    context = ask_context(_no_approval_policy())
    with execution_scope(context):
        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": user}]},
                config={"configurable": {"thread_id": thread_id}},
            ),
            timeout=60,
        )
    from terminus.cli import shutdown_resources
    await shutdown_resources()
    return result, context


@sync_async
async def test_model_calls_spawn_agent_and_receives_the_child_result(stub_child_agent):
    result, context = await _drive(
        [
            _call("inspect the auth flow"),
            AIMessage(content="The subagent reported no issues."),
        ],
        "Audit the auth flow.",
    )
    messages = result["messages"]

    # spawn_agent was actually offered to the model and called by it. The exact
    # number of call records is LangGraph's state-merge bookkeeping; what matters
    # is that the model only ever reached for this tool, and that it then stopped.
    assert "spawn_agent" in _OFFERED, _OFFERED
    calls = [c["name"] for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    assert calls, "the model never called a tool"
    assert set(calls) == {"spawn_agent"}, calls

    # the child really executed, on its own thread, under its own authority
    child_threads = [t for t, _c in _CHILD_THREADS if t.startswith("child-")]
    assert child_threads, _CHILD_THREADS
    # The child's conversation thread must not be the parent's: sharing it would
    # replay the parent's transcript into the child.
    assert child_threads[0] != _PARENT_THREAD["thread_id"], "child reused the parent thread"
    assert _PARENT_THREAD["thread_id"].startswith("loop-")

    child_context = [c for t, c in _CHILD_THREADS if t.startswith("child-")][0]
    assert child_context.kind == "child"
    assert child_context.parent_agent_id, "the child has no parent identity"
    assert PermissionLevel.WRITE in child_context.policy.deny_levels

    # a ToolMessage came back carrying the child's report
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == "spawn_agent"
    assert "child finished its work" in str(tool_messages[0].content)

    # the parent model received it on its second turn, and answered
    assert _CURSOR == 2, "exactly two model turns: call, then answer"
    assert any(isinstance(m, ToolMessage) for m in _RECEIVED)
    assert messages[-1].content == "The subagent reported no issues."

    # and the parent execution paid for it
    assert context.budget.used == 1
    assert context.budget.remaining == MAX_CHILDREN_PER_PARENT - 1


@sync_async
async def test_the_child_does_not_receive_the_spawn_tool(stub_child_agent):
    """Recursion stays impossible even through the live tool loop."""
    from terminus.agents import AgentSpawner, ChildSpec

    await _drive(
        [_call("look around"), AIMessage(content="ok")], "go", "loop-no-spawn"
    )
    spawner = AgentSpawner(runner=lambda c: None, project_facts="")
    child = spawner.create(ChildSpec(task="look around", role="researcher"))
    assert "spawn_agent" not in child.tools
    # Bounded context: no transcript, and a hard cap.
    assert "cannot see the main conversation" in child.prompt()
    assert len(child.prompt()) < 8000


@sync_async
async def test_two_delegations_in_one_execution(stub_child_agent):
    result, context = await _drive(
        [
            _call("first area", "researcher", "a"),
            _call("second area", "reviewer", "b"),
            AIMessage(content="Combined."),
        ],
        "Cover two areas.",
        "loop-two",
    )
    assert len([t for t, _c in _CHILD_THREADS if t.startswith("child-")]) == 2
    assert context.budget.used == 2
    assert result["messages"][-1].content == "Combined."


@sync_async
async def test_a_spinning_model_is_stopped_by_the_budget(stub_child_agent):
    """A model that only ever delegates is refused, not left looping."""
    result, context = await _drive(
        _spin(), "spin"
    )
    assert context.budget.used == MAX_CHILDREN_PER_PARENT
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("Not spawned" in str(m.content) for m in tool_messages)
