"""The interactive REPL: a loop plus one command table.

The dispatch used to be a 125-line ``if``/``elif`` chain with the help text
written out separately underneath it. Two consequences: a new command had to be
added in two places and could silently drift from its own documentation, and
``session_id`` was threaded through unrelated branches.

``COMMANDS`` is now the single declaration. Each entry is a name, a summary and a
small async handler; ``/help`` is generated from the same list the dispatcher
walks, so it cannot fall behind.

Three commands are deliberately *not* in the table: ``/exit`` and ``/quit`` end
the loop and ``/clear`` resets the terminal. They are control flow rather than
work, and keeping them in the loop avoids inventing a "what do I return to say I
have finished" convention for every handler.

Imports of ``terminus.cli`` are deliberately function-local: ``cli`` re-exports
``terminus_cli_run`` from here, so a module-level import would be a cycle.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.prompt import Prompt

from terminus.observability.logging import get_logger

console = Console()
logger = get_logger(__name__)


@dataclass
class Repl:
    """What a command handler may see, and the one thing it may change.

    ``session_id`` is a field rather than a loop variable so a command that
    changes the session does so through the state it was handed instead of
    reaching into the caller's frame.
    """

    index: Any
    session_id: str


#: A handler takes the repl state and whatever followed the command name.
Handler = Callable[[Repl, str], Awaitable[None]]


@dataclass(frozen=True)
class Command:
    name: str
    summary: str
    handler: Handler
    takes_argument: bool = False

    @property
    def usage(self) -> str:
        return f"{self.name} <arg>" if self.takes_argument else self.name


async def _ask(repl: Repl, argument: str) -> None:
    from terminus.agent.orchestrator import handle_query

    if not argument:
        console.print("[bold red]Please enter a question[/bold red]")
        return
    console.print(f"[bold green]Question:[/bold green] {argument}")
    try:
        response = await handle_query(argument, repl.session_id)
        console.print(f"[bold blue]Response:[/bold blue] {response}")
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        console.print(f"[bold red]Query failed:[/bold red] {exc}")


async def _plan(repl: Repl, argument: str) -> None:
    from terminus.tasks.orchestrator import handle_plan_command

    if not argument:
        console.print("[bold red]Please enter a goal[/bold red]")
        return
    console.print(f"[bold green]Goal:[/bold green] {argument}")
    logger.info("User asked for a plan for the goal: %s", argument)
    response = await handle_plan_command(argument)
    console.print(f"[bold blue]Response:[/bold blue] {response}")


async def _show_session(repl: Repl, _argument: str) -> None:
    console.print(f"[bold green]Current Session:[/bold green] {repl.session_id}")


async def _new_session(repl: Repl, _argument: str) -> None:
    from terminus.memory.session import switch_session

    repl.session_id = str(uuid.uuid4())
    switch_session(repl.session_id)
    console.print(f"[bold green]New session created:[/bold green] {repl.session_id}")


async def _switch_session(repl: Repl, argument: str) -> None:
    from terminus.memory.session import switch_session

    if not argument:
        console.print("[bold red]Please enter a session ID[/bold red]")
        return
    repl.session_id = argument
    switch_session(argument)
    console.print(f"[bold green]Switched to session:[/bold green] {argument}")


async def _show_semantic_index(repl: Repl, _argument: str) -> None:
    from terminus.context.indexers.factory import show_index

    console.print("[bold green]Showing semantic index...[/bold green]")
    show_index(repl.index)


async def _show_index_migration(_repl: Repl, argument: str) -> None:
    from terminus.cli import show_index_migration

    # Read-only unless the operator explicitly asks for the destructive rebuild,
    # and even then it is refused while the collection holds other projects'
    # points.
    show_index_migration(
        rebuild_shared_collection="--rebuild-shared-collection" in argument
    )


async def _show_skill(_repl: Repl, argument: str) -> None:
    from terminus.skills.skill_tools import describe_skills

    console.print(f"[bold green]{describe_skills(argument or None)}[/bold green]")


async def _show_agents(_repl: Repl, _argument: str) -> None:
    from terminus.agents.roles import describe_roles
    from terminus.agents.spawn import recent_delegations

    console.print(f"[bold green]{describe_roles()}[/bold green]")
    tree = recent_delegations()
    if tree:
        console.print("\n[bold cyan]Recent delegations:[/bold cyan]")
        console.print(tree)
    else:
        console.print("[dim]No delegated agents have run in this process yet.[/dim]")


async def _task_status(_repl: Repl, _argument: str) -> None:
    from terminus.cli import show_task_status

    show_task_status()


async def _help(_repl: Repl, _argument: str) -> None:
    """Print the command list, generated from the table being dispatched on."""
    console.print("[bold green]Help:[/bold green]\n")
    width = max(len(c.usage) for c in COMMANDS)
    for command in COMMANDS:
        console.print(f"[yellow] {command.usage.ljust(width)}  {command.summary}[/yellow]")
    console.print(
        "\n[yellow] A bare message is not a command: prefix it with /ask or /plan.[/yellow]"
    )
    console.print(
        "[yellow] /index_migrate --rebuild-shared-collection rebuilds the shared "
        "collection (destructive)[/yellow]"
    )


#: The single source of truth for what the REPL accepts. This list is both the
#: dispatch order and the contents of /help, so the two cannot disagree.
COMMANDS: tuple[Command, ...] = (
    Command("/ask", "Ask a question about the codebase", _ask, takes_argument=True),
    Command("/plan", "Plan work for a goal; `/plan continue` resumes", _plan, takes_argument=True),
    Command("/task_status", "Show task status", _task_status),
    Command("/session", "Show the current session", _show_session),
    Command("/new_session", "Create a new session", _new_session),
    Command("/switch", "Switch to a session by id", _switch_session, takes_argument=True),
    Command("/skills", "List installed skills", _show_skill),
    Command("/skill", "Show one skill in detail", _show_skill, takes_argument=True),
    Command("/agents", "List child agent roles and recent delegations", _show_agents),
    Command("/show_semantic_index", "Show semantic index stats", _show_semantic_index),
    Command(
        "/index_migrate",
        "Report Qdrant project-scoping status",
        _show_index_migration,
        takes_argument=True,
    ),
    Command("/help", "Show this help", _help),
)

EXIT_COMMANDS = frozenset({"/exit", "/quit"})
CLEAR_COMMAND = "/clear"


def resolve(user_input: str) -> tuple[Command | None, str]:
    """The command *user_input* names, and the argument that followed it.

    The name is matched case-insensitively but the argument is taken from the
    original text, so a question or a session id keeps its capitalisation. The
    earlier chain lowercased the whole line first and then extracted ``/switch``'s
    argument, which quietly lowercased anything a user pasted in.
    """
    lowered = user_input.lower()
    for command in COMMANDS:
        if lowered == command.name:
            return command, ""
        if lowered.startswith(command.name + " "):
            return command, user_input[len(command.name):].strip()
    return None, ""


async def dispatch(repl: Repl, user_input: str) -> bool:
    """Run the command *user_input* names. False when it named no command."""
    command, argument = resolve(user_input)
    if command is None:
        logger.warning("Unrecognised input", extra={"query": user_input})
        console.print(
            "[bold red]Not a command[/bold red]\n"
            "Type [yellow]/help[/yellow] for the list, or "
            "[yellow]/ask <question>[/yellow] to ask one."
        )
        return False
    await command.handler(repl, argument)
    return True


async def _repl_loop(index) -> None:
    """Read a line, dispatch it, repeat until the user leaves."""
    from terminus.memory.session import get_current_session

    repl = Repl(index=index, session_id=get_current_session())
    while True:
        user_input = Prompt.ask("[bold green]Query >> [/bold green]").strip()
        lowered = user_input.lower()

        if not user_input:
            console.print("[bold red]Please enter a query[/bold red]")
            continue
        if lowered in EXIT_COMMANDS:
            console.print("[bold blue]Goodbye![/bold blue]")
            return
        if lowered == CLEAR_COMMAND:
            console.clear()
            continue
        if not await dispatch(repl, user_input):
            # A bare sentence is a question, not a typo. Treating it as one is
            # friendlier than the old "Invalid query", which read as though the
            # message itself were wrong.
            await _ask(repl, user_input)


async def terminus_cli_run() -> bool:
    """Start the session, run the REPL, and report on the way out.

    Returns False only when startup failed, which the caller uses to choose an
    exit code. A REPL that ran and was then quit is a success.
    """
    from terminus.cli import format_startup_error, initialize, _shutdown_report

    logger.info("Starting Terminus CLI")
    console.print("[bold blue]Welcome to Terminus![/bold blue]")
    console.print("Type [bold red]'/exit'[/bold red] or [bold red]'/quit'[/bold red] to quit")
    console.print("Type [bold cyan]'/clear'[/bold cyan] to clear the screen")
    try:
        try:
            _llm, _embedder, index, resolution = initialize()
        except Exception as exc:
            # A startup failure is a user-facing message, not a traceback.
            logger.error("Startup failed: %s", exc)
            console.print(
                f"[bold red]Startup failed:[/bold red] {format_startup_error(exc)}"
            )
            return False
        if resolution.fallback:
            # A substitution the user did not ask for has to be visible, not
            # merely logged.
            console.print(f"[bold yellow]{resolution.describe()}[/bold yellow]")
        await _repl_loop(index)
    finally:
        await _shutdown_report()
    return True
