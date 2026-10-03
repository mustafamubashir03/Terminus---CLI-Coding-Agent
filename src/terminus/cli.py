"""The interactive session: a REPL of slash commands.

Reached by a bare ``terminus`` or ``terminus agent``, and kept as the entry point
for the REPL specifically. The non-interactive command tree lives in
``terminus.cli_app``; this module is the conversational surface, not the CLI's
argument handling.

The loop is a prompt, a slash command or a free-form question, and the commands
are ordinary ``if`` branches because the set is small and closed. Adding a
command means adding a branch and a line in ``/help``; there is no registry to
keep in sync.

``terminus_cli_run`` owns process startup and shutdown for the session: it
initialises the clients, and releases the checkpointer, the MCP client and the
LLM connections on the way out, each independently so one failure cannot strand
the rest.
"""

import asyncio
from pathlib import Path
from rich.console import Console
from terminus.env import load_project_env
from terminus.llm.factory import aclose_llm_clients, format_provider_diagnostics
from terminus.observability.logging import get_logger

console = Console()
logger = get_logger(__name__)

def show_task_status():
    """Display the current task progress for the latest approved project.

    Results and failures are shown, not just statuses. A persisted result that
    only ever reaches a judge is invisible to the person who asked for the work;
    the same bounded rendering the agents get is used here so there is one
    definition of what a task "did".
    """
    from terminus.config import CONFIG
    from terminus.project_context import collect_project_facts
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
    tasks = store.get_all_tasks(project_id)
    if tasks:
        console.print("[bold cyan]Tasks Overview:[/bold cyan]")
        for t in tasks:
            console.print(f"- {t['id']}: {t['description']} ({t['status']})")

    facts = collect_project_facts(store, project_id, include_skills=False)
    if not facts.recent:
        return
    console.print("[bold cyan]Results:[/bold cyan]")
    for task in facts.recent:
        if not (task.result or task.error):
            continue
        attempts = f", {task.attempts} attempt(s)" if task.attempts else ""
        label = "result" if task.result else "error"
        body = task.result or task.error
        console.print(f"- {task.id} ({task.status}{attempts}) {label}:")
        for line in body.splitlines():
            console.print(f"    {line}")

def show_index_migration(rebuild_shared_collection: bool = False):
    """Report whether the Qdrant collection is project-scoped, and optionally fix it.

    Read-only by default. The rebuild is destructive to a collection shared
    across projects, so it only runs on an explicit flag and only after
    ``migrate_collection`` has checked the collection is not provably another
    project's.
    """
    from terminus.context.indexers.migrate import (
        describe_plan,
        inspect_collection,
        migrate_collection,
    )

    if not rebuild_shared_collection:
        report = inspect_collection()
        console.print("[bold green]Qdrant project-scoping report:[/bold green]")
        for line in describe_plan(report):
            console.print(f"  {line}")
        if report.needs_migration:
            console.print(
                "[yellow]Nothing was changed. Re-run with "
                "--rebuild-shared-collection to rebuild (this empties the shared "
                "collection).[/yellow]"
            )
        return

    report = migrate_collection(rebuild_shared_collection=True)
    console.print("[bold green]Qdrant migration result:[/bold green]")
    for line in describe_plan(report):
        console.print(f"  {line}")


def initialize():
    """Prepare the process for a session, and return nothing expensive.

    Startup deliberately does *not* build a chat client, load an embedding model
    or open the vector store. Each of those is slow - a HuggingFace embedder
    loads weights, a remote store costs a network round trip, a provider SDK
    costs seconds of import time - and none of them is needed to accept the
    first question. Opening a session should be as cheap as reading a prompt.

    The pieces that genuinely must happen up front are the ones that are cheap
    and that would otherwise fail later and further from the cause: loading the
    project's ``.env``, resolving configuration, and checking that the
    configured vector provider/mode pair is one that can work at all. That last
    check is pure string validation, so a typo is still an immediate error rather
    than a confusing failure from inside a storage client.

    Everything expensive is built on first use by the code that needs it: the
    chat client by :mod:`terminus.llm.factory`, the embedder and store by the
    retriever the ``search_codebase`` tool calls.

    Returns the resolved vector-store description, which needs no I/O and is what
    the caller reports at startup.
    """
    from terminus.context.indexers.factory import configured, resolved_backend, validate
    from terminus.observability.logging import configure_tracing

    logger.info("Initializing Terminus...")
    repo_path = Path.cwd()
    # A project .env is optional: it is searched for in parent directories, and a
    # missing file is not fatal because credentials may already be in the process
    # environment (CI, containers, an exported shell).
    load_project_env(repo_path)

    # If exactly one provider has a key and the configured one does not, use it.
    # Done here rather than at config-import time because this is the first point
    # where the project's .env has been read, and a key that is sitting in a file
    # is a key the user has already provided. Making them run setup to connect
    # the two was a configuration ritual with no decision in it.
    try:
        from terminus.user_config import apply_auto_provider

        chosen = apply_auto_provider()
    except Exception:  # never let convenience break startup
        chosen = None
    if chosen:
        # Plain text: print() does not interpret Rich markup, so "[dim]...[/dim]"
        # would reach the terminal as literal brackets.
        logger.info("Auto-selected provider: %s", chosen)
        print(chosen)

    # Fail fast on a vector configuration that cannot work, without touching the
    # network or the disk.
    provider, mode = configured()
    validate(provider, mode)
    resolution = resolved_backend(provider, mode)

    # Tracing is reconciled last, deliberately. Its job is to guarantee nothing in
    # this process is traced, and it works by removing the variables LangChain
    # reads - so running it before the steps above was running it too early:
    # resolving the store re-reads the project's .env to find the cluster
    # endpoint, which put back every variable just removed, LANGSMITH_TRACING
    # included. Nothing above traces anything, so there is nothing to protect by
    # doing it sooner, and doing it last is the only order a later import cannot
    # undo.
    if configure_tracing():
        logger.info("LangSmith tracing enabled by configuration")

    logger.info("Terminus initialized successfully: %s", resolution.describe())
    return resolution


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


async def _shutdown_report() -> None:
    """Everything the process must give back on the way out.

    Token usage first, because it is the thing a user wants before they quit.
    Then project ownership, which belongs to the process rather than to any one
    command. Then the shared clients.

    Each step is independent: one that fails must not strand the others, so a
    failure is reported rather than raised.
    """
    from terminus.observability.usage_tracker import get_summary

    summary = get_summary()
    if summary.records:
        console.print("\n[bold magenta]Token Usage & Prompt-Caching Report[/bold magenta]")
        console.print(summary.to_table())
        if summary.total_cached:
            console.print(
                f"[bold green]Prompt cache saved {summary.savings_percent:.1f}% of "
                f"input tokens "
                f"({summary.total_cached:,} of {summary.total_input:,} cached)."
            )

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


from terminus.cli_app.repl import terminus_cli_run  # noqa: E402  (avoids an import cycle)


def run():
    """Sync entry point required by pyproject.toml scripts - bootstraps the async event loop."""
    if asyncio.run(terminus_cli_run()) is False:
        raise SystemExit(1)
