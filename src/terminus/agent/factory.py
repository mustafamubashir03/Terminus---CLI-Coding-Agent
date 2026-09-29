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
from terminus.llm.factory import get_chat_model, get_current_model_label, get_llm
from terminus.context.environment import build_startup_context
from terminus.workspace import project_root
from terminus.tools.codebase_tool import search_codebase
from terminus.observability.logging import get_logger
from langchain.agents import create_agent
from terminus.tools.filesystem_tools import (
    list_directory,
    read_file,
    file_exists,
    grep,
    write_file,
    edit_file,
)
from terminus.tools.web_tools import web_search, web_fetch
from terminus.tools.shell_tools import run_command
from terminus.skills.skill_tools import load_skill, build_skills_prompt
from terminus.project_context import project_prompt_section
from terminus.tools.project_status_tool import project_status
from terminus.tools.spawn_agent_tool import spawn_agent
from terminus.cache import get_cached_prompt, cache_prompt
from terminus.permissions import PermissionLevel, PermissionPolicy
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from typing import Any
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
        TOOL_RULES,
        TOOL_GUIDE,
        WORKFLOW,
        DELEGATION,
    ]
    catalogue = build_skills_prompt()
    if catalogue and catalogue.strip():
        parts.append(catalogue)
    return cache_prompt(key, "\n\n".join(parts))


def _build_system_prompt() -> str:
    """Compose the /ask system prompt: cached static material plus live project state.

    ``build_agent`` is called once per question, so the project section is read
    from TaskStore on each turn and stays current, while the expensive-to-build
    half is still served from cache.

    create_agent turns this string into a SystemMessage that it prepends
    locally at each model call, so it never enters the agent's message state and
    is never duplicated into the checkpoint.
    """
    project_section = project_prompt_section()
    if not project_section:
        return _build_static_prompt()
    return f"{_build_static_prompt()}\n\n{project_section}"


# The /ask tool list, as data so tests (and a future child agent) can inspect it
# without parsing source or building a real graph.
#
# Deliberately absent: 'delete_file', 'append_file', 'run_in_directory' and any
# external GitHub MCP server. 'run_command' is present but is policy-gated at
# call time by terminus.permissions - see tools/shell_tools.py.
ASK_TOOLS = (
    search_codebase,
    grep,
    list_directory,
    read_file,
    file_exists,
    write_file,
    edit_file,
    web_search,
    web_fetch,
    run_command,
    load_skill,
    project_status,
    spawn_agent,
    )


def tools_by_name() -> dict[str, Any]:
    """The agent toolset indexed by tool name.

    A child agent is configured with tool *names* (a role is a permission
    description, and names are what a prompt or a config file can express).
    This is where a name becomes the real tool, so a child can only ever be
    handed something that is genuinely in the parent's own toolset.
    """
    return {getattr(tool, "name", str(tool)): tool for tool in ASK_TOOLS}


DEFAULT_ASK_MODEL_CALLS = 16
"""Model calls one /ask turn may make before the run ends.

A turn is a question, not a project. Sixteen is enough to search, read, edit and
verify; beyond that the agent is looping, and ending the run lets it report what
it has rather than burning the budget.
"""


async def build_agent(tools_override: list | None = None, *, model: str | None = None,
                      provider: str | None = None, max_model_calls: int | None = None):
    """Create and return the /ask agent.

    Read/discovery tools, controlled write tools ('write_file', 'edit_file'),
    web research tools, and 'run_command' for real shell execution. Still no
    'delete_file', no 'append_file' and no external GitHub MCP server.

    ``tools_override`` is how a delegated child agent is built with a narrower
    tool set than /ask - see ``terminus.agents.spawn``. A child cannot widen its
    own reach: the names it gets are intersected with its role's and with what
    the parent has, in ``ChildAgent``.

    This function builds an agent; it does not authorise it. Permission lives
    with the execution (see terminus.execution), so the caller wraps the agent
    run in an :func:`execution_scope`. A child agent inherits its parent's
    policy simply by running inside the parent's scope - it cannot obtain a
    more permissive one by calling this function.

    Execution budget, precisely:
      * ModelCallLimitMiddleware(run_limit) caps calls to the MODEL node.
      * ToolCallLimitMiddleware(tool_name=None) caps calls to ALL tools per run
        and, with exit_behavior="continue", blocks the excess with a tool error
        and lets the agent finish, instead of ending the run.
      * ToolCallLimitMiddleware(tool_name="search_codebase") is a tighter
        per-tool cap on top of the global one.
      The summarisation middleware makes its own LLM calls, which are not
      counted by run_limit.

    ``model``/``provider`` override the configured route for this one agent,
    and go through the same ``get_chat_model`` router - so a child agent
    participates in the existing fallback chain rather than needing a model
    abstraction of its own. They are also how a child gets a *smaller* budget
    than /ask: a delegated task is a bounded unit of work, not a conversation.
    """
    llm = get_llm() if not (model or provider) else get_chat_model(
        model or get_current_model_label() or "", provider
    )
    full_prompt = _build_system_prompt()

    tools = list(tools_override) if tools_override else list(ASK_TOOLS)
    middlewares = [
        ModelCallLimitMiddleware(
            run_limit=max_model_calls or DEFAULT_ASK_MODEL_CALLS,
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(tool_name=None, run_limit=40, exit_behavior="continue"),
        ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=4, exit_behavior="continue"),
        get_summarization_middleware(),
    ]
    checkpointer = await get_checkpointer()
    logger.info(
        "Creating agent (tools=%d, model=%s)", len(tools), getattr(llm, "model_name", "?")
    )
    return create_agent(
        llm,
        tools=tools,
        system_prompt=full_prompt,
        checkpointer=checkpointer,
        middleware=middlewares
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
