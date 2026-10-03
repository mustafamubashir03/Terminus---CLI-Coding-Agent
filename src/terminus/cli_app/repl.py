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

import os
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from terminus.observability.logging import get_logger

console = Console()
logger = get_logger(__name__)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("terminus")
    except PackageNotFoundError:  # running from a source checkout
        return "0.1.0"


def _short_cwd() -> str:
    """The working directory, shortened so it fits one line.

    A home directory prefix is elided the way a shell prompt does it, because the
    tail is the part that identifies the project.
    """
    home = str(Path.home())
    here = str(Path.cwd())
    if here.startswith(home):
        return "~" + here[len(home):]
    return here


def _first_run_notice() -> None:
    """Tell someone with no provider how to get one, instead of failing later.

    The problem this solves: a fresh environment reached an interactive prompt
    that looked ready, and the first question failed with a provider error from
    the far end of a stack. Nothing on screen said the missing thing was a key.

    Deliberately a notice and not a gate. Refusing to open the session would also
    block `terminus config`, `terminus doctor` and `/help` - all of which work
    perfectly well with no credentials, and which the notice is telling the user
    to go and use. So Terminus opens, says what is missing, and waits.
    """
    try:
        import terminus.config as config_module
        from terminus.llm.providers import PROVIDERS
        from terminus.user_config import (
            apply_auto_provider,
            configured_provider_has_credential,
            load_env_files,
        )

        load_env_files()
        # Startup has already done this, so by the time the panel is reached the
        # provider is settled and a single key on the machine has been adopted.
        # Doing it again is harmless and keeps the panel honest when it is called
        # on its own.
        apply_auto_provider()
        if configured_provider_has_credential():
            return
        provider = str((config_module.CONFIG.get("llm") or {}).get("provider", ""))
    except Exception:  # never let a diagnostic break startup
        return

    # More than one key, or none: naming them is the useful part. A single key is
    # already adopted above, so this branch means there is a real choice to make.
    try:
        have = sorted(
            name
            for name, spec in PROVIDERS.items()
            if spec.env_keys and any(os.environ.get(k) for k in spec.env_keys)
        )
    except Exception:
        have = []

    if len(have) > 1:
        choices = "\n".join(
            f"  [dim]•[/dim] {name:<14} [dim]key found[/dim]" for name in have
        )
        body = (
            "No key for [cyan]%s[/cyan], the configured provider, and more than one\n"
            "other provider has a key. Which one to use is your call:\n\n"
            "%s\n\n"
            "  [bold]/config set llm.provider <name>[/bold]   choose one\n"
            "  [bold]/setup[/bold]                          or add a key for %s" % (provider, choices, provider)
        )
    else:
        body = (
            "No API key is configured, so questions cannot be answered yet.\n\n"
            "  [bold]/setup[/bold]     choose a provider and enter a key\n"
            "  [bold]/doctor[/bold]     see everything that is missing\n"
            "  [bold]/config show[/bold]  what Terminus would use right now\n\n"
            "[dim]Already have a key? Put GROQ_API_KEY (or another provider's key)\n"
            "in a .env file in this folder and start Terminus again - it will be\n"
            "used automatically.[/dim]"
        )

    console.print(
        Panel(
            body,
            title="[bold yellow]Not configured yet[/bold yellow]",
            subtitle="[dim]nothing to do if you already have a key in .env[/dim]",
            border_style="yellow",
            padding=(0, 2),
        )
    )


@dataclass
class Repl:
    """What a command handler may see, and the one thing it may change.

    ``session_id`` is a field rather than a loop variable so a command that
    changes the session does so through the state it was handed instead of
    reaching into the caller's frame.

    ``repo_path`` is the project directory, recorded rather than a live index.
    Opening the session must not wait on a vector store, so the index is
    resolved on first use by :meth:`index` - and only the commands that display
    the index ever ask for it. Retrieval builds its own store when a search
    actually happens, so nothing else needs it here.
    """

    repo_path: str
    session_id: str
    _index: Any = field(default=None, repr=False)
    _index_resolved: bool = field(default=False, repr=False)

    def index(self):
        """``(index, ResolvedBackend)``, built on first use and reused after.

        Reports an active fallback the moment the backend is actually resolved,
        which is the first point at which anyone can act on it. At startup the
        resolution is only a prediction from configuration - a remote store that
        is configured but unreachable cannot be detected without contacting it,
        and contacting it is exactly the cost this defers.
        """
        from terminus.context.indexers.factory import resolve_index

        if not self._index_resolved:
            self._index, resolution = resolve_index(self.repo_path)
            self._index_resolved = True
            if resolution.fallback:
                console.print(f"[bold yellow]{resolution.describe()}[/bold yellow]")
        return self._index


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
        # Logged at debug, not error: the message is already shown to the user
        # below, and printing it twice - once as a log line, once as output -
        # made every failure look like two separate problems.
        logger.debug("Query failed: %s", exc)
        console.print(f"[bold red]Query failed:[/bold red] {exc}")


async def _run_cli(args: list[str]) -> None:
    """Run a top-level Terminus command inside the session.

    The first-run notice tells people to run `terminus setup`, and a notice that
    can only be obeyed by leaving the program is a notice that gets read and
    then fails. Someone who pastes what they were told, at the prompt, is
    following the instructions exactly - so the instruction has to work there.

    Two things it cannot do, both handled rather than hidden:

    * It cannot prompt. The test runner that hosts the command replaces stdin, so
      an interactive command gets EOF and aborts - which is what printed
      "Aborted." and looked like a crash. An interactive command is refused with
      an instruction instead.
    * Its output is captured, so Rich's colour codes come back as literal
      ``[1;36m`` text. They are stripped rather than printed as gibberish.

    Nothing here is allowed to escape: a nested command that raises must not end
    the session.
    """
    from typer.testing import CliRunner

    from terminus.cli_app import app

    # `setup` without a provider given is the interactive flow, which needs a
    # real terminal. With --provider/--api-key/--key-stdin it is fully specified
    # and runs fine here.
    interactive = "setup" in args and not any(
        flag in args for flag in ("--provider", "--key-stdin", "--api-key")
    )

    try:
        result = CliRunner().invoke(app, args)
    except SystemExit:
        return
    except Exception as exc:  # a broken helper must not end the session
        logger.debug("In-session command %s failed", args, exc_info=True)
        console.print(f"[bold red]Could not run[/bold red] {' '.join(args)}: {exc}")
        return

    out = _strip_ansi(result.stdout or "")
    if out:
        console.print(out.rstrip())
    stderr = getattr(result, "stderr", None)
    if stderr:
        console.print(f"[dim]{_strip_ansi(str(stderr)).rstrip()}[/dim]")
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        logger.debug("In-session command %s failed", args, exc_info=result.exception)
    if interactive and result.exit_code not in (0, None):
        console.print(
            "\n[dim]For the interactive version, run "
            "[cyan]terminus setup[/cyan] in your shell - it has to read your "
            "typing, which this prompt cannot hand it. To do it right here, "
            "give it everything: "
            "[cyan]setup --provider groq --api-key <key>[/cyan][/dim]"
        )


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Remove escape sequences from captured output.

    Output captured from a nested command still contains the colour codes Rich
    wrote for a terminal. Printing that text again renders the codes literally -
    ``[1;36mTerminus setup[0m`` - which looks like the program is broken.
    """
    return _ANSI.sub("", text)


async def _setup(repl: Repl, argument: str) -> None:
    """Configure a provider, in the session, where the notice can be acted on."""
    console.print(
        "[dim]Choose a provider and enter a key. Press Ctrl+C at any prompt "
        "to stop without changing anything.[/dim]\n"
    )
    await _run_cli(["setup", *(argument.split() if argument else [])])


async def _doctor(repl: Repl, argument: str) -> None:
    await _run_cli(["doctor"])


async def _config(repl: Repl, argument: str) -> None:
    await _run_cli(["config", *(argument.split() if argument else ["show"])])


async def _login(repl: Repl, argument: str) -> None:
    await _run_cli(["providers", "login", *(argument.split() if argument else [])])


async def _logout(repl: Repl, argument: str) -> None:
    await _run_cli(["providers", "logout", *(argument.split() if argument else [])])


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
    try:
        index = repl.index()
    except Exception as exc:
        # One unreachable store must not end the session. Resolving the index is
        # deferred to this command, so this is the first point at which a remote
        # outage can be observed - and the session has to survive learning about
        # it. The detail is the same message the store's own error type carries.
        logger.warning("Could not open the semantic index", exc_info=True)
        console.print(f"[bold red]Could not open the semantic index:[/bold red] {exc}")
        return
    show_index(index)


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
        "\n[yellow] Just type a question and press enter - /ask is optional.[/yellow]"
    )
    console.print(
        "[yellow] /index_migrate --rebuild-shared-collection rebuilds the shared "
        "collection (destructive)[/yellow]"
    )
    # These are the same commands the shell has, and they work right here. They
    # are listed in slash form because that is unambiguous, but the shell form is
    # accepted too, so anything printed elsewhere in this program can be pasted
    # straight into this prompt.
    console.print(
        "\n[cyan]Configuration[/cyan]  [dim](these work here, or in your shell)[/dim]\n"
        "  [dim]/setup[/dim]              choose a provider and store a key\n"
        "  [dim]/doctor[/dim]             check that this installation can work\n"
        "  [dim]/config show[/dim]        what Terminus will use right now\n"
        "  [dim]/login --provider X[/dim]  store a key for a provider"
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
      # The configuration commands are in the session, not only in the shell,
      # because the first-run notice asks for them at the prompt. Kept at the end
      # so /help still lists the session commands first.
      Command("/setup", "Choose a provider and store a key", _setup, takes_argument=True),
      Command("/doctor", "Check that this installation can work", _doctor),
      Command("/config", "Show or change settings: /config show, /config set k v", _config,
              takes_argument=True),
      Command("/login", "Store a provider key: /login --provider groq", _login,
              takes_argument=True),
      Command("/logout", "Remove a stored key: /logout --provider groq", _logout,
              takes_argument=True),
  )

#: The ``terminus config`` subcommands, so the shell spelling can be told apart
#: from an English sentence that happens to start with the word "config".
_CONFIG_SUBCOMMANDS = frozenset({"show", "list", "get", "set", "unset"})

EXIT_COMMANDS = frozenset({"/exit", "/quit"})
CLEAR_COMMAND = "/clear"


def resolve(user_input: str) -> tuple[Command | None, str]:
    """The command *user_input* names in slash form, and its argument.

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


def as_session_command(user_input: str) -> tuple[Command | None, str]:
    """The command *user_input* names, accepting the shell spelling too.

    `resolve` matches the slash form. This also accepts `terminus setup` and a
    bare `setup`, because the startup notice and /help both tell people to run
    `terminus setup` and people type that at the prompt, because that is where
    they are. Sending it to the model as a question about the codebase is the
    most confusing thing this program could do with an instruction it just
    printed, so the words it printed are accepted as the commands they name.

    A bare phrase only counts when its first word is exactly a command, so
    "setup the tests" is still a question about the tests.
    """
    command, argument = resolve(user_input)
    if command is not None:
        return command, argument

    text = user_input.strip()
    if text.lower().startswith("terminus "):
        # They named the program, so this is a command line. Strip it and take
        # the rest as a command, including any arguments.
        text = text[len("terminus "):].strip()
    elif text.startswith("/"):
        return None, ""  # an explicit but unknown slash command: a typo
    if not text:
        return None, ""

    head, _, rest = text.partition(" ")
    lowered = head.lower()
    rest = rest.strip()

    # `config` and `providers` take a subcommand, and only a known one counts.
    # That is what lets "config set llm.provider groq" be a command while
    # "config file is missing a default" stays a question: the first has a verb
    # Terminus owns, the second does not.
    if lowered in ("config", "providers"):
        inner, _, remainder = rest.partition(" ")
        inner = inner.lower()
        if lowered == "config":
            if inner not in _CONFIG_SUBCOMMANDS:
                return None, ""
            command, _ = resolve("/config")
            return (command, (inner + " " + remainder).strip()) if command else (None, "")
        target = {"login": "/login", "logout": "/logout"}.get(inner)
        if target is None:
            return None, ""
        command, _ = resolve(target)
        return (command, remainder.strip()) if command else (None, "")

    # The rest take no arguments, so they are only commands when typed alone.
    # "setup the tests" has to remain a question about the tests.
    single = {"setup": "/setup", "doctor": "/doctor", "login": "/login",
              "logout": "/logout", "help": "/help"}.get(lowered)
    if single is None or rest:
        return None, ""
    command, _ = resolve(single)
    return (command, "") if command else (None, "")


async def dispatch(repl: Repl, user_input: str) -> bool:
    """Run the command *user_input* names. False when it named no command.

    A command that raises must not end the session. Losing an interactive
    conversation because one command had a bug in it is a much worse outcome
    than the bug, and the traceback belongs in the log rather than on the
    screen - a user at a prompt needs to know what to type next, not where a
    frame lives.
    """
    command, argument = as_session_command(user_input)
    if command is None:
        if user_input.startswith("/"):
            # A leading slash is an explicit request for a named command. Saying
            # "Unknown command" and then sending it to the model anyway - which
            # is what happened to /agent - is two contradictory answers to one
            # keystroke, and the second one spends real money.
            logger.debug("Unknown command %r", user_input)
            console.print(
                f"[bold red]Unknown command:[/bold red] {user_input.split()[0]}\n"
                "Type [yellow]/help[/yellow] for the list, or just type a "
                "question without the slash."
            )
            return True  # handled: do not also treat it as a question
        return False
    try:
        await command.handler(repl, argument)
    except Exception as exc:
        logger.error("Command %s failed", command.name, exc_info=True)
        console.print(
            f"[bold red]{command.name} failed:[/bold red] {exc}\n"
            "[dim]The session is still open. /help for the list.[/dim]"
        )
    return True


async def _repl_loop(repo_path: str) -> None:
    """Read a line, dispatch it, repeat until the user leaves."""
    from terminus.memory.session import get_current_session

    repl = Repl(repo_path=repo_path, session_id=get_current_session())
    while True:
        user_input = Prompt.ask("[bold cyan]terminus[/bold cyan] [dim]›[/dim]").strip()
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
            try:
                await _ask(repl, user_input)
            except Exception as exc:  # never let one question end the session
                logger.error("Question failed", exc_info=True)
                console.print(f"[bold red]Query failed:[/bold red] {exc}")


async def terminus_cli_run() -> bool:
    """Start the session, run the REPL, and report on the way out.

    Returns False only when startup failed, which the caller uses to choose an
    exit code. A REPL that ran and was then quit is a success.
    """
    from terminus.cli import _shutdown_report, initialize, start_sandbox

    logger.info("Starting Terminus CLI")
    try:
        try:
            # Cheap by construction: reads configuration and validates it. It does
            # not build a client, load an embedding model or open a vector store,
            # so the prompt is available essentially immediately.
            initialize()
        except Exception as exc:
            # A startup failure is a user-facing message, not a traceback.
            logger.error("Startup failed: %s", exc)
            console.print(
                "\n[bold red]Terminus could not start.[/bold red]\n"
                f"{exc}\n\n"
                "[dim]Run [cyan]terminus providers status[/cyan] for the full "
                "provider and backend report.[/dim]"
            )
            return False
        # The single header for the session. One panel, not a version line
        # followed by a greeting followed by three lines of instructions.
        console.print(
            Panel(
                f"[bold]{_short_cwd()}[/bold]\n"
                f"[dim]Type[/dim] [cyan]/help[/cyan] [dim]for commands, "
                f"[cyan]/exit[/cyan] [dim]to quit, or just ask a question.[/dim]",
                title=f"[bold blue]Terminus[/bold blue] {_version()}",
                subtitle="[dim]coding agent[/dim]",
                border_style="blue",
                padding=(0, 2),
            )
        )
        _first_run_notice()

        # One container for the session, before the first tool can run. It is
        # started here rather than lazily per command so that an agent's first
        # shell call is not also the thing that discovers Docker is missing, and
        # so a session that runs no commands pays no image pull.
        #
        # A failure here ends the session. The alternative - carry on without a
        # container - would run every command on the host under a session that
        # the operator believes is sandboxed, which is a downgrade nobody asked
        # for and only a warning stands between.
        from terminus.sandbox import SandboxUnavailable

        try:
            await start_sandbox()
        except SandboxUnavailable as exc:
            logger.error("Sandbox startup failed: %s", exc)
            console.print(
                "\n[bold red]Terminus could not start the sandbox.[/bold red]\n"
                f"{exc}\n\n"
                "[dim]Commands would run on the host without it, so the session "
                "is ending instead. Start Docker, build the image with "
                "[cyan]docker build -t terminus-sandbox:latest src/terminus/sandbox/Dockerfile[/cyan], "
                "or turn the boundary off deliberately with "
                "[cyan]terminus config set sandbox.enabled false[/cyan].[/dim]"
            )
            return False

        await _repl_loop(str(Path.cwd()))
    finally:
        await _shutdown_report()
    return True
