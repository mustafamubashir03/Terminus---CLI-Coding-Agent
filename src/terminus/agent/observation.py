"""Making the model look at its own work before it is allowed to claim success.

An agent that writes a file and then answers "done, all tests pass" has asserted
something it never observed. That is the commonest way a coding agent is
confidently wrong, and it is invisible from the outside because the claim is
well-formed.

The check belongs *inside* the loop. The model node runs, the model tries to stop,
and the last thing it sees before stopping is a request to go and look. Same
graph, same checkpointer, same model, same usage accounting - one execution path,
and the graph's own routing does the looping.

Four rules keep it from becoming an obstacle:

* **Ask once.** If the model has already been told, it gets to finish. A second
  turn is a request, not a veto, and a model that insists on answering without
  looking is entitled to answer.
* **Ask only when it mutated.** Pure reasoning and pure reading need no check.
* **Ask only if nothing observed it.** Reading a file back, running the tests, or
  looking at ``git diff`` all count. So does *failing* to observe: a refused read
  is still an attempt, and the model is the one that reports that it failed.
* **Read before writing does not count.** Verifying work that has not happened yet
  is not verifying.

``@hook_config(can_jump_to=["model"])`` is load-bearing, not decoration. LangChain
reads that metadata to decide whether the node gets a conditional edge back to the
model; without it a ``jump_to`` in the returned update is silently dropped and the
request is appended to a graph that has already decided to stop. The message is
recorded in state and never reaches the model - which looks like it works.
"""

from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage

from terminus.observability.logging import get_logger

logger = get_logger(__name__)

# Tools whose result means the agent has looked at the effect of its own work.
OBSERVING_TOOLS = frozenset({
    "read_file", "grep", "list_directory", "file_exists",
    "run_command", "search_codebase", "project_status",
    "git_status", "git_diff", "git_log",
})
"""The tools that count as having looked at the result of a mutation."""

MUTATING_TOOLS = frozenset({
    "write_file", "edit_file", "append_file", "delete_file",
    # git_commit writes history, git_checkout rewrites the working tree, and
    # git_branch(name) writes a ref. All three change state the model did not
    # produce by answering, so a turn that ends right after one has verified
    # nothing - even though commit and branch alone change no file.
    "git_commit", "git_checkout", "git_branch",
})
"""The tools that mean something changed and still has to be looked at.

The single definition of both sets. The orchestrator reads its verdict from the
state keys below rather than keeping a second copy of this logic, so "did it
change anything" has one answer in the process.
"""

OBSERVATION_REQUEST = (
    "Before answering: you changed files but have not read the result back or run "
    "anything since. Read what you changed, or run the project's checks, and report "
    "what you actually observed. If you could not verify it, say so explicitly "
    "rather than claiming it works."
)

MUTATED_KEY = "terminus_mutated"
OBSERVED_KEY = "terminus_observed"
ASKED_KEY = "terminus_asked"


class ObservationState(AgentState):
    """Agent state plus three facts about the current run.

    Deliberately not durable facts about the agent: these describe one run, and
    ``before_agent`` clears them, so a thread that asked once in an earlier turn
    can still ask again in a later one.
    """

    terminus_mutated: NotRequired[bool]
    terminus_observed: NotRequired[bool]
    terminus_asked: NotRequired[bool]


def observation_pending(state: Any) -> bool:
    """Did this run change something and never look at the result?

    Read from the same state keys the middleware writes, so the flag the
    orchestrator reports and the rule the graph enforced cannot disagree.
    """
    if not isinstance(state, dict):
        return False
    return bool(state.get(MUTATED_KEY)) and not bool(state.get(OBSERVED_KEY))


def _tool_names(messages: list[Any]) -> list[str]:
    """Every tool the model asked for, in order."""
    names: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                names.append(str(name))
    return names


def fold_tool_calls(messages: list[Any], mutated: bool, observed: bool) -> tuple[bool, bool]:
    """Fold a round's tool calls into "did it change anything, did it look".

    A mutation resets the observation, because reading a file back *before*
    rewriting it says nothing about the rewrite. A read only counts once there is
    something to read back.
    """
    for name in _tool_names(messages):
        if name in MUTATING_TOOLS:
            mutated, observed = True, False
        elif name in OBSERVING_TOOLS and mutated:
            observed = True
    return mutated, observed


class ObservationMiddleware(AgentMiddleware):
    """Send the model back to its own work, once, when it stops too early."""

    state_schema = ObservationState

    def before_agent(self, state: ObservationState, runtime: Any) -> dict[str, Any] | None:
        """Clear the per-run flags.

        Without this the flags ride along in the checkpointed thread, and the
        second turn of a conversation would inherit the first one's "already
        asked" and never be asked at all.
        """
        return {MUTATED_KEY: False, OBSERVED_KEY: False, ASKED_KEY: False}

    def before_model(self, state: ObservationState, runtime: Any) -> dict[str, Any] | None:
        """Account for the tool calls made so far.

        Runs before every model call, so by the time ``after_model`` asks whether
        the model is finishing, the round that just completed is already counted -
        including a round that raised, since a refused read is still the model
        having tried to look.
        """
        mutated, observed = fold_tool_calls(
            state.get("messages") or [],
            bool(state.get(MUTATED_KEY)),
            bool(state.get(OBSERVED_KEY)),
        )
        return {MUTATED_KEY: mutated, OBSERVED_KEY: observed}

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: ObservationState, runtime: Any) -> dict[str, Any] | None:
        """Catch the completion claim and turn the model back, once.

        The ``jump_to`` re-enters the model node through the graph's own routing
        rather than starting a second run here. That is what keeps the check
        inside the one execution path.
        """
        if state.get(ASKED_KEY):
            return None
        if not state.get(MUTATED_KEY) or state.get(OBSERVED_KEY):
            return None

        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        if messages[-1].tool_calls:
            return None

        logger.info("Model tried to finish after changing things without observing them")
        return {
            "messages": [HumanMessage(content=OBSERVATION_REQUEST)],
            ASKED_KEY: True,
            "jump_to": "model",
        }
