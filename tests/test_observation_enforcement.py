"""A model that stops too early gets sent back to look, inside the same loop.

These build a real ``create_agent`` graph through the production
``run_turn``/``build_agent`` path, so they fail if the enforcement ever degrades
back into something that only appears to work.

Two things about the harness are worth stating, because getting either wrong makes
these tests pass for the wrong reason:

* **The model must be given permission to write.** ``write_file`` goes through the
  runtime permission policy like every other tool, and the default policy is
  fail-closed. Under it a write is *refused*, no mutating tool ever reaches the
  graph, and the rule under test never fires - which would look like a working
  test of a mechanism that was never exercised.
* **A tool-call message must carry text.** An ``AIMessage`` with empty content
  produces no content chunks when the graph streams it, and langchain-core raises
  ``No generation chunks were returned``. Real models emit text alongside tool
  calls, so the helpers here do too.
"""

import asyncio
import functools
import uuid

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from terminus.agent.observation import OBSERVATION_REQUEST
from terminus.agent.orchestrator import Outcome, run_turn
from terminus.permissions import PermissionLevel, PermissionPolicy


def sync_async(fn):
    """pytest-asyncio is not installed; this matches the suite's convention."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


#: The policy an interactive /ask turn runs under: read and write allowed, which
#: is what makes a mutation possible for the rule to react to.
ASK_LIKE = PermissionPolicy(
    auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
    approver=None,
    deny_levels=(PermissionLevel.DESTRUCTIVE,),
)


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed script and records what each call was shown.

    Both entry points are implemented from one position counter because
    ``run_turn`` streams. The base fake model only refills its iterator for
    ``_generate``: over ``_astream`` it raises "No generations found in stream", no
    tool ever runs, and ``_seen`` stays empty - so every assertion below would pass
    for the wrong reason, including the ones that assert something did *not*
    happen.
    """

    def __init__(self, script):
        super().__init__(messages=iter(script))
        object.__setattr__(self, "_script", list(script))
        object.__setattr__(self, "_seen", [])
        object.__setattr__(self, "_pos", 0)

    def bind_tools(self, tools, **kwargs):
        """The script already carries the tool calls; binding is a no-op."""
        return self

    def _next(self) -> AIMessage:
        position = self._pos
        object.__setattr__(self, "_pos", position + 1)
        # Past the end of the script, repeat the last reply, so the extra call the
        # graph makes after being sent back is answered rather than raising.
        return self._script[min(position, len(self._script) - 1)]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append(list(messages))
        message = self._next()
        if not isinstance(message, AIMessageChunk):
            message = AIMessageChunk(
                content=message.content,
                tool_calls=message.tool_calls,
                id=message.id,
                response_metadata=message.response_metadata,
            )
        yield ChatGenerationChunk(message=message)

    @property
    def calls(self) -> int:
        """How many model calls were actually made."""
        return len(self._seen)

    def saw(self, needle: str) -> bool:
        """Was *needle* shown on the last model call?"""
        return any(needle in str(getattr(m, "content", "")) for m in self._seen[-1])

    def times_saw(self, needle: str) -> int:
        """How many model calls were shown *needle*.

        Not a count of requests issued. Once the model has been asked, the request
        is part of its history, so every later call is shown it again - counting
        calls therefore cannot distinguish "asked once" from "asked twice".
        :meth:`asked_after_replying` is what measures that.
        """
        return sum(
            1
            for seen in self._seen
            if any(needle in str(getattr(m, "content", "")) for m in seen)
        )

    def asked_after_replying(self, needle: str) -> int:
        """How many times the model was asked *immediately after* answering.

        The request lands as a ``HumanMessage``, so a call that sees it as the
        most recent message is a model being sent back. Any call after that sees it
        further back in history, which is not a second request.
        """
        asks = 0
        for seen in self._seen:
            for message in reversed(seen):
                content = str(getattr(message, "content", ""))
                if needle in content:
                    asks += 1
                    break
                if content.strip():
                    break
        return asks


def _calls(tool: str, say: str, call_id: str, **arguments) -> AIMessage:
    return AIMessage(
        content=say, tool_calls=[{"name": tool, "args": arguments, "id": call_id}]
    )


def _wrote(path: str = "a.txt") -> AIMessage:
    return _calls(
        "write_file", say="Writing it now.", call_id="w1", file_path=path, content="x"
    )


def _read(path: str = "a.txt") -> AIMessage:
    return _calls("read_file", say="Reading it back.", call_id="r1", file_path=path)


def _edited(path: str = "a.txt") -> AIMessage:
    return _calls(
        "edit_file", say="Editing it.", call_id="e1",
        file_path=path, old_text="x", new_text="y",
    )


def _failing_read(path: str = "a.txt") -> AIMessage:
    """A read that cannot succeed: the path is outside the workspace."""
    return _calls("read_file", say="Reading it back.", call_id="r9",
                  file_path="../outside-workspace.txt")


def _says(text: str) -> AIMessage:
    return AIMessage(content=text)


async def _turn(script, question="do the thing", permission=ASK_LIKE):
    """Run one turn on a real graph around *script*, returning result and model."""
    import terminus.agent.factory as factory

    model = ScriptedModel(script)
    original = factory.get_llm
    factory.get_llm = lambda: model
    try:
        from terminus.execution import ask_context

        context = ask_context(permission) if permission else None
        result = await run_turn(
            question, f"obs-{uuid.uuid4().hex[:12]}", permission=context
        )
    finally:
        factory.get_llm = original
    return result, model


# --- 1. model-only response --------------------------------------------------


@sync_async
async def test_a_turn_that_mutates_nothing_is_never_sent_back(workspace):
    """Pure reasoning has nothing to verify."""
    result, model = await _turn([_says("The answer is 42.")])

    assert not model.saw(OBSERVATION_REQUEST)
    assert result.outcome is Outcome.DONE
    assert result.text == "The answer is 42."
    assert not result.unverified


# --- 2. read/observe without mutating ----------------------------------------


@sync_async
async def test_reading_only_is_never_sent_back(workspace):
    """Reading is observation, but there was nothing changed to observe."""
    result, model = await _turn([
        _read("notes.md"),
        _says("I found the notes you asked for."),
    ])

    assert not model.saw(OBSERVATION_REQUEST)
    assert result.outcome is Outcome.DONE
    assert not result.unverified


# --- 3. mutation followed by observation -------------------------------------


@sync_async
async def test_a_turn_that_observes_is_not_sent_back(workspace):
    """The normal path is untouched: write, read it back, then answer."""
    result, model = await _turn([
        _wrote(),
        _read(),
        _says("Wrote and verified a.txt."),
    ])

    assert not model.saw(OBSERVATION_REQUEST)
    assert result.outcome is Outcome.DONE
    assert not result.unverified
    assert "verified" in result.text


@sync_async
async def test_reading_before_writing_does_not_count_as_verification(workspace):
    """Verifying work that has not happened yet is not verifying it."""
    result, model = await _turn([
        _read(),
        _wrote(),
        _says("Wrote it."),
        _read(),
        _says("Now it is verified."),
    ])

    assert model.asked_after_replying(OBSERVATION_REQUEST) == 1
    assert result.outcome is Outcome.DONE
    assert not result.unverified


# --- 4. mutation without observation -----------------------------------------


@sync_async
async def test_the_model_is_actually_shown_the_request(workspace):
    """The load-bearing assertion: the text reaches the model, not just the log."""
    result, model = await _turn([
        _wrote(),                              # mutates
        _says("All done, tests pass."),        # claims success without looking
        _read(),                               # looks, once told
        _says("I wrote a.txt and read it back."),
    ])

    assert model.saw(OBSERVATION_REQUEST), (
        "the observation request never reached the model"
    )
    assert result.outcome is Outcome.DONE
    assert not result.unverified
    assert "read it back" in result.text


@sync_async
async def test_asking_is_not_a_veto(workspace):
    """Asked once. A model that insists on answering anyway gets to answer."""
    result, model = await _turn([
        _wrote(),
        _says("Done."),
        _says("I will not verify. Done."),
    ])

    assert result.outcome is Outcome.DONE
    assert "Done." in result.text
    assert model.asked_after_replying(OBSERVATION_REQUEST) == 1, "asked exactly once"
    # And it is reported honestly rather than silently dropped.
    assert result.unverified


@sync_async
async def test_a_second_mutation_re_arms_the_check(workspace):
    """Verification of the first change does not carry over to the second."""
    result, model = await _turn([
        _wrote(),
        _read(),
        _edited(),                             # a new change, nothing has looked at it
        _says("Edited it."),
        _read(),
        _says("Verified the edit too."),
    ])

    assert model.asked_after_replying(OBSERVATION_REQUEST) == 1
    assert not result.unverified


# --- 5. tool failure ---------------------------------------------------------


@sync_async
async def test_a_refused_read_still_counts_as_looking(workspace):
    """An observation that failed is still the model having tried to look.

    The alternative is worse than useless: a model blocked from reading would be
    sent back forever and the loop would end by budget rather than by an answer.
    """
    result, model = await _turn([
        _wrote(),
        _failing_read(),
        _says("I tried to read it back but could not, so I cannot claim it works."),
    ])

    assert not model.saw(OBSERVATION_REQUEST)
    assert not result.unverified
    assert "cannot claim" in result.text


# --- 6. multiple tool calls in one round -------------------------------------


@sync_async
async def test_several_mutations_in_one_round_count_as_one_mutation(workspace):
    """A batch of writes is still one thing to go and look at."""
    result, model = await _turn([
        _calls("write_file", say="Writing a.", call_id="w1", file_path="a.txt", content="x"),
        _calls("write_file", say="Writing b.", call_id="w2", file_path="b.txt", content="y"),
        _says("Both written."),
        _read("a.txt"),
        _read("b.txt"),
        _says("Both read back."),
    ])

    assert model.asked_after_replying(OBSERVATION_REQUEST) == 1
    assert result.outcome is Outcome.DONE
    assert not result.unverified


# --- 7. normal completion ----------------------------------------------------


@sync_async
async def test_a_verified_turn_reports_nothing_special(workspace):
    """The common case is ordinary: no request, no flag, a plain answer."""
    result, model = await _turn([
        _wrote(),
        _read(),
        _says("Done and checked."),
    ])

    assert result.outcome is Outcome.DONE
    assert not result.unverified
    assert result.tool_failures == ()
    assert model.asked_after_replying(OBSERVATION_REQUEST) == 0


# --- 8. budget / iteration termination ---------------------------------------


@sync_async
async def test_a_run_that_stops_on_budget_is_reported_as_such(workspace):
    """Looping for tools until the call limit ends the run, not the rule."""
    import terminus.agent.factory as factory
    from terminus.agent.factory import AgentPolicy

    budgeted = AgentPolicy(
        tools=factory.ASK_TOOLS,
        system_prompt="p",
        model_call_limit=3,
        tool_call_limits=((None, 50),),
        tool_limit_behaviour="end",
        summarize=False,
        checkpoint=False,
        verify_observations=True,
    )
    from terminus.execution import ask_context

    model = ScriptedModel([_wrote(), _read(), _read(), _read(), _read(), _read()])
    original = factory.get_llm
    factory.get_llm = lambda: model
    try:
        result = await run_turn(
            "go", f"obs-{uuid.uuid4().hex[:12]}",
            policy=budgeted, permission=ask_context(ASK_LIKE),
        )
    finally:
        factory.get_llm = original

    assert result.outcome is Outcome.BUDGET_EXHAUSTED
    assert "Model call limit" in result.text
    # It never claimed success, so there is nothing unverified to report.
    assert not result.unverified


@sync_async
async def test_the_check_does_not_loop_forever_against_an_insistent_model(workspace):
    """One request, however many times the model tries to stop without looking."""
    result, model = await _turn([
        _wrote(),
        _says("Done."),
        _says("Still done."),
        _says("Still done, honestly."),
        _says("Final answer, still done."),
    ])

    assert model.asked_after_replying(OBSERVATION_REQUEST) == 1
    assert result.outcome is Outcome.DONE
    assert result.unverified


# --- wiring: the middleware is on the production path ------------------------


def test_the_middleware_is_attached_by_the_production_builder(workspace, monkeypatch):
    """Not an isolated mechanism: it must be on the graph build_agent returns."""
    import asyncio as _asyncio

    import terminus.agent.factory as factory
    from terminus.agent.observation import ObservationMiddleware

    captured: dict = {}

    def fake_create_agent(model, **kwargs):
        captured["middleware"] = kwargs.get("middleware", [])
        return object()

    async def no_checkpointer():
        return None

    monkeypatch.setattr(factory, "create_agent", fake_create_agent)
    monkeypatch.setattr(factory, "get_llm", lambda: "LLM")
    monkeypatch.setattr(factory, "get_checkpointer", no_checkpointer)

    child = factory.child_policy(
        [t for t in factory.ASK_TOOLS if t.name == "write_file"],
        "child instructions",
        model=None, provider=None, model_call_limit=2, tool_call_limit=10,
    )
    for policy in (factory.ask_policy(), child):
        _asyncio.run(factory.build_agent(policy))
        assert any(isinstance(m, ObservationMiddleware) for m in captured["middleware"]), (
            "the observation middleware is not on the production agent"
        )


def test_a_policy_can_opt_out(workspace, monkeypatch):
    """A re-run attempt may be exempt; the switch has to actually disconnect it."""
    import asyncio as _asyncio

    import terminus.agent.factory as factory
    from terminus.agent.observation import ObservationMiddleware

    captured: dict = {}

    def fake_create_agent(model, **kwargs):
        captured["middleware"] = kwargs.get("middleware", [])
        return object()

    async def no_checkpointer():
        return None

    monkeypatch.setattr(factory, "create_agent", fake_create_agent)
    monkeypatch.setattr(factory, "get_llm", lambda: "LLM")
    monkeypatch.setattr(factory, "get_checkpointer", no_checkpointer)

    base = factory.ask_policy()
    exempt = type(base)(**{**base.__dict__, "verify_observations": False})
    _asyncio.run(factory.build_agent(exempt))
    assert not any(isinstance(m, ObservationMiddleware) for m in captured["middleware"])


# --- state bookkeeping -------------------------------------------------------


def test_the_flags_are_per_run_not_per_thread(workspace):
    """A second turn on one thread must be able to ask again.

    The flags live in graph state, which is checkpointed per thread. If they were
    not cleared at the start of each run, the second turn of a long conversation
    would inherit the first one's "already asked".
    """
    from terminus.agent.observation import (
        ASKED_KEY, MUTATED_KEY, OBSERVED_KEY, ObservationMiddleware,
    )

    middleware = ObservationMiddleware()
    cleared = middleware.before_agent({}, None)
    assert cleared == {MUTATED_KEY: False, OBSERVED_KEY: False, ASKED_KEY: False}

    # Once asked, it never asks again within the same run.
    state = {ASKED_KEY: True, MUTATED_KEY: True, OBSERVED_KEY: False, "messages": []}
    assert middleware.after_model(state, None) is None
