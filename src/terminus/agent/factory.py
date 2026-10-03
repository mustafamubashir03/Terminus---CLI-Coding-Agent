"""Building the /ask agent: its prompt, its tools, its budget, its policy.

Three things are assembled here, and the order matters.

**The prompt.** ``_build_system_prompt`` is a cached static half (identity,
environment snapshot, tool rules, skills catalogue) plus a live project-context
section that is rebuilt every turn. Caching the project section would pin a stale
view of the project for the whole session; rebuilding the static half each turn
would re-read the filesystem listing and TERMINUS.md for nothing.

**The tools.** ``ASK_TOOLS`` is the one /ask toolset, as data. A delegated child
gets a narrower set by name through ``tools_override``, intersected with its role
and with the parent's own tools - see ``terminus.agents.spawn``.

**The policy.** ``ask_permission_policy`` decides what this turn may do, based on
whether a human is actually present. Building an agent does not authorise it;
authority belongs to the execution, and the caller wraps the run in
``execution_scope``.

The model comes from ``terminus.llm.factory``, so /ask and its children share one
fallback chain rather than each having their own routing.
"""

from terminus.memory.short_term import get_summarization_middleware
from terminus.memory.short_term import get_checkpointer
from terminus.agent.observation import ObservationMiddleware
from terminus.llm.factory import get_chat_model, get_current_model_label, get_llm
from terminus.context.environment import build_startup_context
from terminus.workspace import project_root
from terminus.agents_md import agents_md_section, load_agents_md
from terminus.tools import registry
from terminus.observability.logging import get_logger
from langchain.agents import create_agent
from terminus.skills.skill_tools import build_skills_prompt
from terminus.project_context import project_prompt_section
from terminus.cache import get_cached_prompt, cache_prompt
from terminus.permissions import PermissionLevel, PermissionPolicy
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from typing import Any
from dataclasses import dataclass
from pathlib import Path
import sys

logger = get_logger(__name__)

_SYSTEM_PROMPT_CACHE_KEY = "ask_agent_system_prompt"

IDENTITY = """You are Terminus, a terminal coding agent. You answer questions about the codebase and make controlled changes to project files.
Be concise and practical.
Prefer inspecting the project with tools over guessing."""

TOOL_RULES = """## Tool rules (follow strictly)
1. Call 'search_codebase' with the most relevant query when required, or 'grep' when you know the exact text to look for.
2. Read the tool results carefully.
3. Once you have enough to answer, answer immediately. Do NOT repeat a search you have already done; only look again if what you have is not enough, or the user asks for more.
4. If you have no memory of an earlier part of this conversation, say so instead of searching the codebase for it.

## Changing files
- Inspect before you modify. Use 'read_file' to see the real contents, and 'grep' or 'search_codebase' to locate something, rather than guessing.
- Use 'edit_file' for a targeted change. It replaces 'old_text' with 'new_text' and only works when 'old_text' occurs exactly once, so copy the exact text from the file including its indentation. An empty 'new_text' deletes the matched snippet.
- Use 'write_file' when creating a new file, or when replacing a whole file's contents is the right thing to do. It overwrites the entire file.
- 'edit_file' will not create a file that does not already exist.
- After modifying, re-read with 'read_file' or 'grep' to confirm the change landed as intended.

ALWAYS give a final text answer."""

TOOL_GUIDE = """## Which tool to use
Local project:
- 'read_file' is authoritative for the current contents of a local file.
- 'grep' is exact text/regex search over local files.
- 'search_codebase' is semantic search over the indexed local project.
Public web:
- 'web_search' discovers candidate pages on the public web; it returns titles, URLs and descriptions, not full pages.
- 'web_fetch' reads one known URL and returns that page's Markdown.

Use 'web_search' then 'web_fetch' when you need external or current information. Prefer official or primary documentation when researching a library or API. Do not use the web tools for anything the local project already answers."""

WORKFLOW = """## Doing a coding task
A finished edit is not a finished task. Work in this order and keep going until you have actually observed the result:
1. Inspect - read the relevant files and find the project's own commands.
2. Understand - decide what must change.
3. Modify - prefer 'edit_file' over 'write_file'.
4. Run - use 'run_command' to execute what is relevant: the project's tests, linter, type checker, build, or a focused script.
5. Observe - read the exit code, stdout and stderr.
6. Repair - if it failed, decide whether a fix is needed and make it.
7. Verify - run it again until it passes.
8. Report - state what you ran and what the actual result was. Never claim a test passed unless you saw it pass.

Match the command to the project rather than guessing: look for pyproject.toml, package.json, Makefile, CI config, or a scripts/ directory and use the commands they define. After changing configuration, validate it. After changing behaviour, run a focused check.

'run_command' cannot grant itself permission. Read-only commands run directly; state-changing ones need runtime approval and may be refused. If a command is refused, do not try to work around it - tell the user what you wanted to run and why."""


DELEGATION = """## Delegating to a subagent

You can call `spawn_agent` to hand one self-contained task to a bounded subagent. It gets its own fresh context, its own role-restricted tools, and returns findings to you. It cannot see this conversation, cannot ask you anything mid-task, and cannot delegate further.

Delegate when the work would not fit your own context, and splitting it would genuinely widen your coverage:

- investigating several independent subsystems or competing hypotheses at once
- inspecting a set of candidate files when you do not yet know which one is relevant
- an independent review of work you have just produced

Prefer doing it yourself when the task is simple, local, or already within reach: a typo, a one-file change, a straightforward implementation, a trivial test. A subagent is a full extra agent run that starts cold, so delegating work you can simply finish is a waste rather than a parallelism win.

Do not delegate to raise the number of agents, and do not assume a multi-step task needs more than one of you.

Delegation is limited per execution, and hitting the limit is a refusal rather than a queue. Whatever you delegate, you remain responsible: integrate the findings and verify the result before reporting it as done. A subagent reporting success is a claim, not a conclusion."""


def human_is_present() -> bool:
    """True when there is actually someone at a terminal to answer a prompt.

    ``input()`` blocks forever on a pipe, a CI job or a background task, so an
    approval prompt is only safe when stdin is a real terminal. Callers that
    cannot be sure must get the deny-by-default policy instead of a hang.
    """
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def interactive_approver(command: str, working_directory: str,
                         level, reason: str) -> bool:
    """Ask the operator to approve one command (Phase 4 approval UX).

    Mirrors the existing blocking-input convention in tasks/approval.py. The
    model never reaches this function; only the runtime does.

    This is called from inside a tool while the /ask answer may still be
    streaming, and orchestrator._write does not terminate lines, so the leading
    newline is what stops the prompt from being glued onto the model's last
    streamed fragment.
    """
    if not human_is_present():
        # Belt and braces: never block a non-interactive caller.
        logger.warning(
            "Refusing %s command: no interactive terminal to approve it: %s",
            getattr(level, "value", level), command,
        )
        return False

    from rich.console import Console

    sys.stdout.flush()
    console = Console()
    console.print()
    console.print(f"[bold yellow]Command needs approval[/bold yellow] ({level.value})")
    console.print(f"  [bold]command[/bold]           : {command}")
    console.print(f"  [bold]working directory[/bold] : {working_directory}")
    console.print(f"  [bold]reason[/bold]            : {reason}")
    try:
        answer = input("  Approve this one command? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def workspace_section(workspace: Path | None = None) -> str:
    """Tell the model what a workspace is, in the terms it will act on.

    Concepts only. Every boundary named here is enforced in
    :mod:`terminus.workspace`, not by this text: the model is told where it is
    and what the filesystem means, and the runtime decides what it may reach.

    Kept short deliberately. A long section on filesystems would be instructions
    the model has to follow rather than facts it can rely on, and it would age
    worse than the code that enforces it.
    """
    root = workspace or project_root()
    return f"""## Workspace
You are working in one workspace: {root}
- Every file path you pass is resolved inside that workspace and is relative to it. A path that points outside it is rejected; do not try to reach files elsewhere on the machine.
- The filesystem is durable state, not scratch space. What you write is still there after this turn ends, and other agents or later sessions working in this same workspace can read it.
- So keep anything worth keeping in files rather than trying to hold it in your context: findings, notes, intermediate results, a plan you are working through. '.terminus/notes/', '.terminus/research/', '.terminus/plans/' exist for that and are yours to create.
- For knowledge that should outlive the task - conventions you inferred, a decision you made, a trap you fell into - AGENTS.md is the place. It is loaded into every session in this workspace, so a note there reaches work you have not started yet. Scratch files do not.
- When your context no longer holds what you need - a finding from earlier, the file you edited twenty calls ago - read it back from the file instead of guessing or searching for it again.
- Files are how work is handed over. Write down what the next step needs and let whoever picks it up - including you in a later session - read that instead of relying on what was said."""


def versioning_section(tools: Any = ()) -> str:
    """Tell the model what Git means here - only as far as its own tools reach.

    Built per agent from the tool names that agent actually has, rather than
    written once as a static section. A child agent inherits the /ask prompt, and
    a static section would tell a researcher it can checkpoint work it has no tool
    to checkpoint with; the model would then spend calls discovering the absence.
    The same reasoning applies in reverse: a worker that only sees the readers is
    told exactly that.

    Kept to what the tools do. Git supplies the mechanism; permissions and the
    workspace still decide what actually happens, and the text must not imply
    otherwise.
    """
    names = {getattr(tool, "name", str(tool)) for tool in tools}
    readers = [n for n in ("git_status", "git_diff", "git_log") if n in names]
    writers = [
        n for n in ("git_commit", "git_checkout", "git_branch") if n in names
    ]
    if not readers and not writers:
        return ""

    lines = ["## Versioning with Git", (
        "The workspace may be a Git repository. The filesystem is its state; Git "
        "is the history of that state, so recovery is a real operation rather "
        "than something you reconstruct from memory. Git tools report plainly "
        "when the workspace is not a repository; they never create one."
    )]
    if readers:
        lines.append(
            "- Inspect before you change anything: 'git_status' for what is "
            "modified, staged and untracked, 'git_diff' for the actual text, "
            "'git_log' for what the recent checkpoints were."
        )
    if "git_commit" in names:
        lines.append(
            "- 'git_commit' saves the whole workspace as a checkpoint. It stages "
            "everything first, so it records what is there, not only what you "
            "staged. Commit when a change is coherent and verified - not after "
            "every step."
        )
    if "git_branch" in names:
        lines.append(
            "- 'git_branch(\"name\")' creates a branch at the current commit; it "
            "does not switch to it. Use it to try something without touching the "
            "branch you started on. Merge is not available."
        )
    if "git_checkout" in names:
        lines.append(
            "- 'git_checkout' either moves you onto a branch or detaches HEAD at "
            "a commit; the result tells you which happened. Read the result "
            "before assuming where you are. If it fails because local changes "
            "would be overwritten, that work is still there - commit it or "
            "revert it yourself. Nothing is discarded for you."
        )
    if writers:
        lines.append(
            "- push, pull, fetch, merge, rebase and stash are NOT available as "
            "tools, and there is no tool for an arbitrary git command."
        )
    lines.append(
        "- Git gives you mechanisms, not guarantees. A checkpoint you created is "
        "only as good as the commit you actually made, and whether an operation "
        "was allowed is decided by the runtime, not by you."
    )
    return "\n".join(lines)


def _build_static_prompt() -> str:
    """The slow-moving half of the /ask system prompt, cached per workspace.

    The identity, the startup environment snapshot, the optional TERMINUS.md
    project instructions, the tool rules and the skills catalogue are folded into
    a single string. The filesystem listing and the TERMINUS.md read are cached
    against the *resolved project root*, so two workspaces in one process never
    share a snapshot. Keying this on a bare constant would hand Project A's
    directory listing and TERMINUS.md to an agent now working in Project B.

    Only genuinely static material is cached here. Project state changes while
    the process is alive, so it is appended fresh by ``_build_system_prompt``
    instead - caching it would pin a stale snapshot of the project for the whole
    session.

    ``build_skills_prompt`` already supplies its own heading and already returns
    "" when the registry is empty, so the catalogue is joined in as-is. A
    separate section header here used to double it up into ``==skills`` followed
    by ``=== Available Skills ===``, and left a stray header line in every
    prompt on a project with no skills installed.
    """
    workspace = project_root()
    key = f"{_SYSTEM_PROMPT_CACHE_KEY}:{workspace}"
    cached = get_cached_prompt(key)
    if cached is not None:
        return cached
    parts = [
        IDENTITY,
        build_startup_context(workspace),
        workspace_section(workspace),
        TOOL_RULES,
        TOOL_GUIDE,
        WORKFLOW,
        DELEGATION,
    ]
    catalogue = build_skills_prompt()
    if catalogue and catalogue.strip():
        parts.append(catalogue)
    return cache_prompt(key, "\n\n".join(parts))


def _build_system_prompt(tools: Any = ()) -> str:
    """Compose the /ask system prompt: cached static material plus live project state.

    ``build_agent`` is called once per question, so everything added here is
    rebuilt per turn: the project section from TaskStore, and AGENTS.md from the
    workspace. Both are read fresh precisely because they change while the process
    lives. Only the static half above is cached.

    ``tools`` are the tools *this* agent has. They are not part of the cached
    static half, because the versioning section depends on them and the cache is
    shared by every agent in the process - caching it would hand a child agent's
    prompt the parent's Git capabilities.

    create_agent turns this string into a SystemMessage that it prepends
    locally at each model call, so it never enters the agent's message state and
    is never duplicated into the checkpoint.
    """
    static = _build_static_prompt()
    parts = [
        static,
        versioning_section(tools),
        agents_md_section(load_agents_md()),
        project_prompt_section(),
    ]
    return "\n\n".join(part for part in parts if part)


# The /ask tool list, as data so tests (and a future child agent) can inspect it
# without parsing source or building a real graph.
#
# Which tools /ask gets is decided by name in terminus.tools.registry, which is
# also the single catalogue every other toolset in the project resolves through.
# The comment about what is deliberately absent lives there, next to the names.
ASK_TOOLS = registry.ask_tools()


def tools_by_name() -> dict[str, Any]:
    """The /ask toolset indexed by tool name.

    A child agent is configured with tool *names* - a role is a permission
    description, and names are what a prompt or a config file can express. This
    is where a name becomes the real tool, so a child can only ever be handed
    something that is genuinely in the parent's own toolset.
    """
    catalogue = registry.catalogue()
    return {tool.name: tool for tool in ASK_TOOLS if (tool.name in catalogue)}


DEFAULT_ASK_MODEL_CALLS = 16
"""Model calls one /ask turn may make before the run ends.

A turn is a question, not a project. Sixteen is enough to search, read, edit and
verify; beyond that the agent is looping, and ending the run lets it report what
it has rather than burning the budget.
"""


@dataclass(frozen=True)
class AgentPolicy:
    """Everything a caller can vary about one agent run.

    The loop itself is not here: LangGraph owns it. This is the complete set of
    differences between /ask, a child agent and a /plan worker, which is why
    there is one ``build_agent`` rather than three ways of assembling a graph.

    ``tool_call_limits`` is ``(tool_name, limit)`` pairs. A ``None`` name is the
    cap on every tool.
    """

    tools: tuple
    system_prompt: str
    model: str | None = None
    provider: str | None = None
    model_call_limit: int = DEFAULT_ASK_MODEL_CALLS
    tool_call_limits: tuple[tuple[str | None, int], ...] = (
        (None, 40),
        ("search_codebase", 4),
    )
    tool_limit_behaviour: str = "continue"
    summarize: bool = True
    checkpoint: bool = True
    verify_observations: bool = True
    """Send the model back once if it tries to claim success it never observed.

    On for every surface by default. A turn that mutates nothing - a research or
    review turn - is unaffected, because the rule only fires after a mutation.
    A /plan worker being re-run after a failure can set it False, because that
    attempt has already been observed by whoever read the log.
    """


async def build_agent(policy: AgentPolicy):
    """Assemble a LangGraph agent from *policy*.

    The one place a Terminus agent is constructed. Everything downstream - /ask,
    ``spawn_agent``, /plan workers - differs only in the policy it passes.

    Budgets, in the order they apply:

    * ``model_call_limit`` caps calls to the model node and, with
      ``exit_behavior="end"``, jumps the graph to its end rather than raising.
    * each ``tool_call_limits`` entry caps calls; with
      ``tool_limit_behaviour="continue"`` the excess comes back to the model as a
      tool error so it can adapt, and with ``"end"`` the run stops.

    The summarisation middleware makes its own model calls, which neither limit
    counts.
    """
    llm = (
        get_chat_model(policy.model or get_current_model_label() or "", policy.provider)
        if (policy.model or policy.provider)
        else get_llm()
    )

    middlewares = [
        ModelCallLimitMiddleware(run_limit=policy.model_call_limit, exit_behavior="end"),
        *(
            ToolCallLimitMiddleware(
                tool_name=name,
                run_limit=limit,
                exit_behavior=policy.tool_limit_behaviour,
            )
            for name, limit in policy.tool_call_limits
        ),
    ]
    if policy.summarize:
        middlewares.append(get_summarization_middleware())
    if policy.verify_observations:
        middlewares.append(ObservationMiddleware())

    checkpointer = await get_checkpointer() if policy.checkpoint else None
    logger.info(
        "Creating agent (tools=%d, model=%s)", len(policy.tools), getattr(llm, "model_name", "?")
    )
    return create_agent(
        llm,
        tools=list(policy.tools),
        system_prompt=policy.system_prompt,
        checkpointer=checkpointer,
        middleware=middlewares,
    )


def ask_policy() -> AgentPolicy:
    """The policy for one /ask turn.

    The prompt is rebuilt here rather than cached as a constant, because
    ``build_agent`` runs once per question and the project section must be read
    fresh each time; only the static half inside it is cached.
    """
    return AgentPolicy(tools=ASK_TOOLS, system_prompt=_build_system_prompt(ASK_TOOLS))


def child_policy(
    tools: list,
    instructions: str,
    *,
    model: str | None,
    provider: str | None,
    model_call_limit: int,
    tool_call_limit: int,
) -> AgentPolicy:
    """The policy for a delegated child: the parent's loop, a narrower self.

    A child keeps the shared identity, environment and tool rules - it is working
    in the same project under the same rules - and its own instructions are
    appended. What it does not inherit is the parent's *conversation*.

    Both limits come from the role's advertised budget, so the budget a
    ``/agents`` listing shows is the budget the child actually runs under.
    """
    return AgentPolicy(
        tools=tuple(tools),
        system_prompt=f"{_build_system_prompt(tools)}\n\n{instructions}",
        model=model,
        provider=provider,
        model_call_limit=model_call_limit,
        tool_call_limits=((None, tool_call_limit), ("search_codebase", 4)),
    )


def ask_permission_policy(interactive: bool = True) -> PermissionPolicy:
    """The permission policy for one /ask turn.

    The model can never approve itself: this is decided by the runtime from how
    /ask was entered, and enforced by the tools through the active execution.

    A blocking approval prompt is only safe when a human is actually there. In a
    pipe, a CI job or any background task ``input()`` would hang forever, so a
    non-terminal falls back to deny-by-default.

    WRITE is pre-authorised when interactive because ordinary source editing and
    ordinary build/test commands are the whole point of a coding agent;
    prompting for each one would make it unusable. DESTRUCTIVE always requires a
    human.
    """
    if interactive and not human_is_present():
        logger.info("No interactive terminal; shell approvals will be denied")
        interactive = False
    if interactive:
        return PermissionPolicy(
            auto_approve=(PermissionLevel.READ_ONLY, PermissionLevel.WRITE),
            approver=interactive_approver,
            deny_levels=(),
        )
    return PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,),
        approver=None,
        deny_levels=(PermissionLevel.WRITE, PermissionLevel.DESTRUCTIVE),
    )
