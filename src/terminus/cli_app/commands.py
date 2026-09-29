"""The non-provider commands: agent, config, sessions, usage, tools, skills, db.

Every command here is a thin adapter. Where a service already exists -
``session``, ``tools_by_name``, ``get_summary``, ``describe_skills``,
``handle_query`` - this module calls it and formats the result rather than
reimplementing the behaviour. The only logic added is presentation, argument
plumbing, and the one-run model override.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any
from collections.abc import Iterator

import typer

from terminus.cli_app import formatting, settings

config_app = typer.Typer(help="Inspect Terminus configuration.", no_args_is_help=False)
sessions_app = typer.Typer(help="View and switch sessions.", no_args_is_help=False)
tools_app = typer.Typer(help="Inspect the tools the agent can call.", no_args_is_help=False)
skills_app = typer.Typer(help="Inspect available skills and agent roles.", no_args_is_help=False)
db_app = typer.Typer(help="Locate Terminus data files.", no_args_is_help=False)
index_app = typer.Typer(help="Inspect the semantic index and its migration state.", no_args_is_help=False)
agent_app = typer.Typer(
    help="Run the agent: interactively, or once with -p.",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _json_flag(ctx: typer.Context, local: bool = False) -> bool:
    """True when JSON was asked for, globally (``--json``) or on the subcommand."""
    return bool((ctx.obj or {}).get("json", False)) or local


@contextlib.contextmanager
def model_override(model: str | None, provider: str | None = None) -> Iterator[None]:
    """Apply ``--model``/``--provider`` to this process only.

    ``get_llm_config()`` reads ``CONFIG["llm"]`` on every call, so mutating it
    here takes effect immediately and is restored on exit. Nothing is written to
    disk - that is the whole difference between ``agent --model`` and
    ``models set``.

    The model id is taken whole and never split on ``/``: provider-prefixed ids
    such as ``poolside/laguna-s-2.1:free`` and ``openai/gpt-oss-120b`` are
    ordinary values, and guessing at a split would select the wrong model. A
    provider therefore has to be named explicitly.
    """
    if not model and not provider:
        yield
        return
    from terminus.config import CONFIG

    llm = CONFIG.setdefault("llm", {})
    saved = dict(llm)
    if model:
        llm["model"] = model
    if provider:
        llm["provider"] = provider
    try:
        yield
    finally:
        llm.clear()
        llm.update(saved)


# --- config ----------------------------------------------------------------


@config_app.callback(invoke_without_command=True)
def config_root(ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show the effective configuration. Same as ``config list``."""
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(config_list, as_json=_json_flag(ctx, json_output))


@config_app.command("list")
def config_list(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Show the merged configuration and where it came from."""
    from terminus.config import CONFIG, CONFIG_SOURCE, CONFIG_SOURCE_KIND

    payload = {
        "source": str(CONFIG_SOURCE) if CONFIG_SOURCE else None,
        "source_kind": CONFIG_SOURCE_KIND,
        "writable": str(settings.config_path()),
        "config": CONFIG,
    }

    def render(d: dict[str, Any]) -> None:
        formatting.line(f"source   : {d['source'] or 'built-in defaults'} ({d['source_kind']})")
        formatting.line(f"writable : {d['writable']}")
        formatting.panel("Effective configuration", json.dumps(d["config"], indent=2, default=str))

    formatting.emit(payload, as_json=as_json, render=render)


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Dotted key, e.g. llm.model.")) -> None:
    """Read one setting, and report which layer supplies it."""
    if not settings.has_path(key):
        formatting.usage_error(
            f"Unknown setting '{key}'.",
            "Run: terminus config list",
        )
        raise typer.Exit(code=2)
    value = settings.get_value(key)
    source = settings.effective_source(key)
    formatting.line(f"{key} = {json.dumps(value, default=str)}   [dim]{source}[/dim]")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted key, e.g. llm.model."),
    value: str = typer.Argument(..., help="Value. Parsed as JSON, so numbers and booleans keep their type."),
) -> None:
    """Persist a setting to config.yaml."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    try:
        path, written = settings.set_value(key, parsed)
    except KeyError:
        formatting.usage_error(
            f"Unknown setting '{key}'. Nothing was written.",
            "Run: terminus config list",
        )
        raise typer.Exit(code=2)
    formatting.success(f"{key} = {json.dumps(written, default=str)}   ({path})")


# --- agent -----------------------------------------------------------------


@agent_app.callback(invoke_without_command=True)
def agent_root(
    ctx: typer.Context,
    prompt: str = typer.Argument(None, help="Prompt to run. Same as -p."),
    model: str = typer.Option(None, "--model", "-m", help="Model for this run only. Not persisted."),
    provider: str = typer.Option(None, "--provider", help="Provider for this run only. Not persisted."),
    session: str = typer.Option(None, "--session", "-s", help="Session id to use for this run."),
    plan: bool = typer.Option(False, "--plan", help="Plan for the goal instead of answering it."),
    show_usage: bool = typer.Option(False, "--usage", help="Print a token summary when the run finishes."),
) -> None:
    """Bare ``terminus agent`` opens the interactive session.

    With a prompt it runs once and exits, which is what you want in a script or
    a hook. Without one you get the REPL, with its slash commands.
    """
    if ctx.invoked_subcommand is not None:
        return
    if not prompt:
        if model or provider or session or plan:
            raise typer.UsageError("--model, --provider, --session and --plan all apply to a single prompt.")
        formatting.line("No prompt given - starting the interactive session. See terminus agent --help.")
        raise typer.Exit(code=formatting.launch_repl())

    import anyio

    from terminus.agent.orchestrator import handle_query
    from terminus.cli import format_startup_error, initialize, shutdown_resources
    from terminus.memory.session import get_current_session, switch_session

    with model_override(model, provider):
        thread = session or get_current_session()
        if session and session != get_current_session():
            with contextlib.suppress(Exception):
                switch_session(session)
        try:
            initialize()
        except Exception as exc:
            formatting.fail(f"Startup failed: {format_startup_error(exc)}")
            raise typer.Exit(code=1)

        async def _execute() -> str:
            if plan:
                from terminus.cli import handle_plan_command

                return await handle_plan_command(f"Create a plan for: {prompt}")
            return await handle_query(prompt, thread)

        try:
            answer = anyio.run(_execute)
        except Exception as exc:
            formatting.fail(f"Query failed: {exc}")
            raise typer.Exit(code=1)
        finally:
            with contextlib.suppress(Exception):
                anyio.run(shutdown_resources)

    formatting.line(answer)
    if show_usage:
        _print_usage(as_json=False)


# --- usage -----------------------------------------------------------------


def usage_payload() -> dict[str, Any]:
    from terminus.observability.usage_tracker import get_child_events, get_summary

    summary = get_summary()
    by_model: dict[str, dict[str, int]] = {}
    for record in summary.records:
        bucket = by_model.setdefault(record.model or "(unknown)", {"calls": 0, "input": 0, "output": 0})
        bucket["calls"] += 1
        bucket["input"] += record.input_tokens
        bucket["output"] += record.output_tokens
    return {
        "calls": len(summary.records),
        "input_tokens": summary.total_input,
        "output_tokens": summary.total_output,
        "cached_tokens": summary.total_cached,
        "billed_input_tokens": summary.total_billed_input,
        "errors": sorted({r.error_category for r in summary.records if r.error_category}),
        "by_model": by_model,
        "child_events": get_child_events(),
    }


def _print_usage(as_json: bool) -> None:
    def render(d: dict[str, Any]) -> None:
        formatting.line()
        formatting.line(f"[bold]tokens[/bold] in {d['input_tokens']}  out {d['output_tokens']}  "
                        f"cached {d['cached_tokens']}  calls {d['calls']}")
        if d["errors"]:
            formatting.warn(f"errors: {', '.join(d['errors'])}")

    formatting.emit(usage_payload(), as_json=as_json, render=render)


# --- sessions --------------------------------------------------------------


def _checkpointer_path() -> Path:
    from terminus.config import CONFIG

    return Path(CONFIG["memory"]["db_path"])


@sessions_app.callback(invoke_without_command=True)
def sessions_root(ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show the current session. Same as ``sessions list``."""
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(sessions_list, as_json=_json_flag(ctx, json_output))


@sessions_app.command("list")
def sessions_list(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Show the current session and the database that backs it.

    Terminus keeps a single current session in a small file rather than a
    catalogue of past sessions, so that is what is reported here instead of an
    invented history.
    """
    from terminus.memory.session import get_current_session

    payload = {"current": get_current_session(), "database": str(_checkpointer_path())}

    def render(d: dict[str, Any]) -> None:
        formatting.line(f"current : {d['current']}")
        formatting.line(f"database: {d['database']}")

    formatting.emit(payload, as_json=as_json, render=render)


@sessions_app.command("new")
def sessions_new(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Start a new session."""
    from terminus.memory.session import new_session

    session_id = new_session()
    formatting.emit(
        {"current": session_id},
        as_json=as_json,
        render=lambda d: formatting.success(f"New session: {d['current']}"),
    )


@sessions_app.command("switch")
def sessions_switch(session_id: str = typer.Argument(..., help="Session id to switch to.")) -> None:
    """Switch to an existing session id."""
    from terminus.memory.session import switch_session

    try:
        switch_session(session_id)
    except Exception as exc:
        formatting.usage_error(f"Could not switch session: {exc}")
        raise typer.Exit(code=1)
    formatting.success(f"Switched to session: {session_id}")


# --- tools -----------------------------------------------------------------


def tool_payload() -> list[dict[str, Any]]:
    from terminus.agent.factory import tools_by_name

    payload = []
    for name, tool in sorted(tools_by_name().items()):
        description = (getattr(tool, "description", "") or "").strip()
        payload.append({
            "name": name,
            "description": description.splitlines()[0] if description else "",
        })
    return payload


@tools_app.callback(invoke_without_command=True)
def tools_root(ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List the tools. Same as ``tools list``."""
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(tools_list, as_json=_json_flag(ctx, json_output))


@tools_app.command("list")
def tools_list(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """List the tools the agent can call."""
    tools = tool_payload()
    formatting.emit(
        {"tools": tools, "count": len(tools)},
        as_json=as_json,
        render=lambda d: formatting.show_table(
            "Tools",
            ["tool", "description"],
            [[t["name"], t["description"]] for t in d["tools"]],
        ),
    )


@tools_app.command("inspect")
def tools_inspect(
    name: str = typer.Argument(..., help="Tool to inspect."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show one tool in full."""
    from terminus.agent.factory import tools_by_name

    registry = tools_by_name()
    if name not in registry:
        formatting.usage_error(
            f"Unknown tool '{name}'.",
            f"Available: {', '.join(sorted(registry))}",
            "Run: terminus tools list",
        )
        raise typer.Exit(code=2)
    payload = {"name": name, "description": (getattr(registry[name], "description", "") or "").strip()}
    formatting.emit(
        payload,
        as_json=as_json,
        render=lambda d: formatting.panel(d["name"], d["description"] or "(no description)"),
    )


# --- skills ----------------------------------------------------------------


@skills_app.callback(invoke_without_command=True)
def skills_root(ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List skills. Same as ``skills list``.

    Use ``skills list <name>`` for one skill in full.
    """
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(skills_list, name=None, as_json=_json_flag(ctx, json_output))


@skills_app.command("list")
def skills_list(
    name: str = typer.Argument(None, help="Show one skill in full."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List skills, or show one in full.

    Delegates to the service the REPL's ``/skills`` command already uses, so the
    two cannot drift.
    """
    from terminus.skills.skill_tools import describe_skills

    text = describe_skills(name or None)
    if as_json:
        formatting.emit({"text": text}, as_json=True, render=lambda _d: None)
    else:
        formatting.panel(f"skills - {name}" if name else "skills", text)


@skills_app.command("agents")
def skills_agents(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Show the agent roles available for delegation."""
    from terminus.agents.roles import describe_roles
    from terminus.agents.spawn import recent_delegations

    payload = {"roles": describe_roles(), "recent_delegations": recent_delegations() or ""}
    formatting.emit(
        payload,
        as_json=as_json,
        render=lambda d: (
            formatting.panel("Agent roles", d["roles"]),
            (formatting.panel("Recent delegations", d["recent_delegations"])
             if d["recent_delegations"] else formatting.line("[dim]No delegations yet.[/dim]")),
        ),
    )


# --- db / index ------------------------------------------------------------


@db_app.callback(invoke_without_command=True)
def db_root(ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Print the session database path. Same as ``db path``."""
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(db_path, as_json=_json_flag(ctx, json_output))


@db_app.command("path")
def db_path(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Print the session database path, for scripts and ``sqlite3``."""
    path = _checkpointer_path()
    formatting.emit(
        {"path": str(path), "exists": path.exists()},
        as_json=as_json,
        render=lambda d: formatting.line(str(d["path"])),
    )


def _migration_report() -> dict[str, Any]:
    """The Qdrant scoping report, or a note that it does not apply.

    Only meaningful for Qdrant: Chroma isolates projects by directory rather than
    by a payload field, so there is no unscoped legacy data to report. Never
    passes ``rebuild_shared_collection``, so it is always read-only.
    """
    from terminus.context.indexers.factory import QDRANT, configured

    if configured()[0] != QDRANT:
        return {"applicable": False,
                "note": "chroma isolates projects by directory; no payload scoping to report"}
    try:
        from terminus.context.indexers.migrate import migrate_collection

        report = migrate_collection(rebuild_shared_collection=False)
        if isinstance(report, dict):
            return {"applicable": True, **report}
        return {"applicable": True, **(getattr(report, "__dict__", {}) or {"report": str(report)})}
    except Exception as exc:
        return {"applicable": True, "error": f"{type(exc).__name__}: {exc}"}


@index_app.command("status")
def index_status(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Show the configured and resolved vector backend, and check the store.

    Read-only. Reports the configuration *and* what it currently resolves to,
    which is the question that matters when a fallback is enabled: a
    ``provider=qdrant`` line above a ``resolved=chromadb`` line is the whole story
    of a silently-substituted index. Contacts the backend only to check
    reachability and collection health; never writes and never rebuilds.
    """
    from terminus.config import CONFIG
    from terminus.context.indexers.factory import configured, location_of, validate

    provider, mode = configured()
    validate(provider, mode)
    configured_block = {"provider": provider, "mode": mode, "location": location_of(provider)}

    health = _probe(provider, mode)
    payload = {
        "configured": configured_block,
        "reachable": health["reachable"],
        "detail": health["detail"],
        "fallback_enabled": bool(
            CONFIG.get("vector_store", {}).get("fallback_to_chroma", False)
        ),
        "collection": CONFIG.get(
            "qdrant" if provider == "qdrant" else "chromadb", {}
        ).get("collection_name", ""),
        "migration": _migration_report(),
    }

    def render(_d: dict[str, Any]) -> None:
        # soft_wrap: this is a report meant to be read at a glance, and a wrapped
        # "location=..." line is much harder to scan than a short one.
        for line in (
            f"  configured : provider={provider} mode={mode}",
            f"  location   : {configured_block['location']}",
            f"  collection : {payload['collection']}",
            f"  reachable  : {'yes' if payload['reachable'] else 'no'}",
            f"  detail     : {payload['detail']}",
            f"  fallback   : "
            f"{'chroma allowed if unreachable' if payload['fallback_enabled'] else 'disabled'}",
        ):
            formatting.console.print(line, soft_wrap=True)
        formatting.panel("Migration check (read-only)", json.dumps(payload["migration"], indent=2, default=str))

    formatting.emit(payload, as_json=as_json, render=render)


def _probe(provider: str, mode: str) -> dict[str, Any]:
    """Is the configured backend reachable right now? Never raises."""
    from terminus.context.indexers.errors import describe_failure
    from terminus.context.indexers.factory import QDRANT

    if provider != QDRANT:
        try:
            from terminus.context.indexers.semantic_chroma import chroma_persist_path

            path = chroma_persist_path()
            return {
                "reachable": True,
                "detail": f"local store at {path} ({'exists' if path.exists() else 'not built yet'})",
            }
        except Exception as exc:
            return {"reachable": False, "detail": f"{type(exc).__name__}: {exc}"}
    try:
        from terminus.context.indexers.qdrant_client import collection_name, create_qdrant_client

        client = create_qdrant_client()
        name = collection_name()
        existing = {c.name for c in client.get_collections().collections}
        if name not in existing:
            return {"reachable": True, "detail": f"collection {name!r} does not exist yet"}
        count = client.get_collection(collection_name=name).points_count
        return {"reachable": True, "detail": f"collection {name!r} holds {count or 0} point(s)"}
    except Exception as exc:
        return {"reachable": False, "detail": describe_failure(exc)}
