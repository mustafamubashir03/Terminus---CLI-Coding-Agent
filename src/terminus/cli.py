from terminus.tasks.orchestrator import handle_plan_command
import asyncio
from terminus.memory.session import switch_session,get_current_session
import uuid
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from terminus.context.indexers.factory import get_or_create_indexer,show_index
from terminus.env import load_project_env
from terminus.llm.factory import (
    aclose_llm_clients,
    format_provider_diagnostics,
    get_embedder,
    get_llm,
)
from terminus.agent.orchestrator import handle_query
from terminus.observability.logging import get_logger

console = Console()
logger = get_logger(__name__)

def show_task_status():
    """Display the current task progress for the latest approved project."""
    from terminus.config import CONFIG
    from terminus.tasks.task_store import TaskStore
    db_path = CONFIG.get("tasks", {}).get("db_path", ".terminus/tasks/tasks.db")
    store = TaskStore(db_path)
    project_id = store.get_latest_project()
    if not project_id:
        console.print("[bold yellow]No active project found.[/bold yellow]")
        return
    progress = store.get_progress(project_id)
    console.print("[bold green]Task Status:[/bold green]")
    for state, count in progress.items():
        console.print(f"{state.capitalize()}: {count}")
    tasks = store._get_all_tasks(project_id)
    if tasks:
        console.print("[bold cyan]Tasks Overview:[/bold cyan]")
        for t in tasks:
            console.print(f"- {t['id']}: {t['description']} ({t['status']})")

def initialize():
    logger.info("Initializing Terminus...")
    repo_path = Path.cwd()
    # A project .env is optional: it is searched for in parent directories, and a
    # missing file is not fatal because credentials may already be in the process
    # environment (CI, containers, an exported shell).
    load_project_env(repo_path)
    llm = get_llm()
    embedder = get_embedder()
    index = get_or_create_indexer(repo_path)
    logger.info("Terminus initialized successfully")
    return llm, embedder, index


def format_startup_error(exc: BaseException) -> str:
    """A short, actionable startup message, with no traceback for the user."""
    message = str(exc) or exc.__class__.__name__
    try:
        diagnostics = format_provider_diagnostics()
    except Exception:
        diagnostics = ""
    return f"{message}\n{diagnostics}" if diagnostics else message


async def shutdown_resources():
    """Release the process-wide resources Terminus owns.

    Each is closed independently and a failure in one is logged rather than
    raised, so a problem closing one client cannot prevent the others from being
    released.
    """
    from terminus.memory.short_term import close_checkpointer
    from terminus.mcp.terminus_mcp_client import close_terminus_mcp

    for label, close in (
        ("checkpointer", close_checkpointer),
        ("mcp", close_terminus_mcp),
        ("llm clients", aclose_llm_clients),
    ):
        try:
            await close()
        except Exception as exc:
            logger.warning("Error closing %s: %s", label, exc)


async def terminus_cli_run():
    logger.info("Starting Terminus CLI")
    console.print("[bold blue]Welcome to Terminus![/bold blue]")
    console.print("Type [bold red]'/exit'[/bold red] or [bold red]'/quit'[/bold red] to quit")
    console.print("Type [bold cyan]'/clear'[/bold cyan] to clear the screen")
    try:
        try:
            _llm, _embedder, index = initialize()
        except Exception as exc:
            # A startup failure is a user-facing message, not a traceback.
            logger.error("Startup failed: %s", exc)
            console.print(
                f"[bold red]Startup failed:[/bold red] {format_startup_error(exc)}"
            )
            return False
        while True:
            session_id = get_current_session()
            query = Prompt.ask("[bold green]Query >> [/bold green]")
            user_input = query.lower()
            if user_input == "":
                console.print("[bold red]Please enter a query[/bold red]")
                continue
            elif user_input in ["/exit","/quit"]:
                console.print("[bold blue]Goodbye![/bold blue]")
                break
            elif user_input == "/session":
                console.print(f"[bold green]Current Session:[/bold green] {session_id}")
            elif user_input == "/new_session":
                session_id = str(uuid.uuid4())
                switch_session(session_id)
                console.print(f"[bold green]New session created:[/bold green] {session_id}")
            elif user_input.startswith("/switch"):
                session_id = user_input.removeprefix("/switch ").strip()
                if not session_id:
                    console.print("[bold red]Please enter a session ID[/bold red]")
                    continue
                switch_session(session_id)
                console.print(f"[bold green]Switched to session:[/bold green] {session_id}")
            elif user_input == "/clear":
                console.clear()
                continue
            elif user_input.startswith("/ask"):
                question = query.removeprefix("/ask ").strip()
                if not question:
                    console.print("[bold red]Please enter a question[/bold red]")
                    continue
                console.print(f"[bold green]Question:[/bold green] {question}")
                try:
                    response = await handle_query(question, session_id)
                    console.print(f"[bold blue]Response:[/bold blue] {response}")
                except Exception as e:
                    logger.error(f"Query failed: {e}")
                    console.print(f"[bold red]Query failed:[/bold red] {e}")
            elif user_input.startswith("/show_semantic_index"):
                console.print("[bold green]Showing semantic index...[/bold green]")
                show_index(index)
            elif user_input.startswith("/plan"):
                goal = query.removeprefix("/plan ").strip()
                if not goal:
                    console.print("[bold red]Please enter a goal[/bold red]")
                    continue
                console.print(f"[bold green]Goal:[/bold green] {goal}")
                logger.info(f"User asked for a plan for the goal: {goal}")
                response = await handle_plan_command(goal)
                console.print(f"[bold blue]Response:[/bold blue] {response}")
            elif user_input.startswith("/task_status"):
                show_task_status()

            elif user_input.startswith("/help"):
                console.print("[bold green]Help:[/bold green]")
                console.print("\n[bold green]Commands:[/bold green]")
                console.print("[yellow] /ask <question> - Ask a question about codebase[/yellow]")
                console.print("[yellow] /plan <goal> - Create a new plan for the goal[/yellow]")
                console.print("[yellow] /plan continue - Resume the latest resumable project[/yellow]")
                console.print("[yellow] /task_status - Show task status[/yellow]")
                console.print("[yellow] /clear - Clear the screen[/yellow]")
                console.print("[yellow] /exit - Exit the CLI[/yellow]")
                console.print("[yellow] /quit - Exit the CLI[/yellow]")
                console.print("[yellow] /help - Show this help message[/yellow]")
                console.print("[yellow] /session - Show current session[/yellow]")
                console.print("[yellow] /new_session - Create a new session[/yellow]")
                console.print("[yellow] /switch <session_id> - Switch to a session[/yellow]")
                console.print("[yellow] /show_semantic_index - Show semantic index stats[/yellow]")
            else:
                logger.warning("Invalid query", extra={"query": query})
                console.print("[bold red]Invalid query[/bold red]")
                console.print("[yellow] Unknown command Try :  /ask 'Question Here' /clear")
                console.print("Use '/ask <question>' to ask a question about codebase")
                console.print("[bold yellow]show_semantic_index[/bold yellow] for showing semantic index")
        




    finally:
        # Process-wide resources are released here, in the one lifecycle the CLI
        # already has, rather than in each command handler.
        from terminus.observability.usage_tracker import get_summary

        summary = get_summary()
        if summary.records:
            console.print(
                "\n[bold magenta]Token Usage & Prompt-Caching Report[/bold magenta]"
            )
            console.print(summary.to_table())
            if summary.total_cached:
                console.print(
                    f"[bold green]Prompt cache saved {summary.savings_percent:.1f}% of "
                    f"input tokens "
                    f"({summary.total_cached:,} of {summary.total_input:,} cached)."
                )

        # Project ownership belongs to the process, not to one command, so it is
        # released from the lifecycle. Anything still held is reported rather
        # than claimed as released.
        from terminus.ownership import current_ownership

        still_held = set(current_ownership().owned())
        released = set(current_ownership().release_all())
        if still_held - released:
            console.print(
                f"[bold red]Warning:[/bold red] could not release ownership of "
                f"project(s) {', '.join(sorted(still_held - released))}. They may "
                f"process exits."
            )
        elif released:
            logger.info("Released project ownership: %s", ", ".join(sorted(released)))

        await shutdown_resources()


def run():
    """Sync entry point required by pyproject.toml scripts - bootstraps the async event loop."""
    if asyncio.run(terminus_cli_run()) is False:
        raise SystemExit(1)
