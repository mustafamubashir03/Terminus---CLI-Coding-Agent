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
    help="Run the agent: interactively, or once with a prompt (-p).",
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


@config_app.command("show")
def config_show(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Answer: what will Terminus actually use, and is it configured?

    The one command to run when the question is "why did it do that?". It
    resolves the route the same way the agent will, and reports each
    credential's presence and origin - never its value.
    """
    import terminus.config as config_module
    from terminus.llm.factory import get_provider_diagnostics

    # Re-resolve from disk rather than trusting the copy built at import. This
    # command exists to answer "what will Terminus use right now", and answering
    # from a snapshot taken whenever the module happened to be imported is only
    # accidentally correct: it is wrong in any process that wrote or removed a
    # config file after starting, which is exactly what `config set` does.
    CONFIG = config_module.load_config()
    # Same resolution startup applies, so this command and the agent can
    # never disagree about which provider is in use.
    try:
        from terminus.user_config import apply_auto_provider

        apply_auto_provider()
    except Exception:
        pass
    CONFIG_SOURCE = config_module.CONFIG_SOURCE
    CONFIG_SOURCE_KIND = config_module.CONFIG_SOURCE_KIND
    CONFIG_SOURCE_LAYERS = config_module.CONFIG_SOURCE_LAYERS
    from terminus.user_config import (
        configured_provider_has_credential,
        credential_status,
        has_usable_llm_credential,
    )

    llm = CONFIG.get("llm", {}) or {}
    diagnostics = get_provider_diagnostics()
    primary = diagnostics.get("primary_route") or {}
    fallbacks = diagnostics.get("fallback_routes") or []

    # Two different questions, deliberately kept apart. `usable` is the
    # blocking one: can a question be answered at all. `other_key_exists` is the
    # helpful one: they have a key, just not for the provider that is selected.
    usable = configured_provider_has_credential()
    payload: dict[str, Any] = {
        "usable": usable,
        "other_key_exists": has_usable_llm_credential() and not usable,
        "provider": llm.get("provider"),
        "model": llm.get("model"),
        "planner_model": llm.get("planner_model"),
        "judge_model": llm.get("judge_model"),
        "config_source": str(CONFIG_SOURCE) if CONFIG_SOURCE else "built-in defaults",
        "config_source_kind": CONFIG_SOURCE_KIND,
        "config_layers": [
            {"kind": kind, "path": str(path)} for kind, path in CONFIG_SOURCE_LAYERS
        ],
        "streaming": llm.get("streaming"),
        "fallbacks": [
            {
                "provider": f.get("provider"),
                "model": f.get("model"),
                "endpoint": f.get("endpoint"),
                "key_configured": f.get("api_key_configured"),
                "status": f.get("status"),
            }
            for f in fallbacks
        ],
        "exhausted": diagnostics.get("exhausted_providers"),
        "embeddings": CONFIG.get("embeddings", {}),
        "vector_store": {
            "provider": CONFIG.get("vector_store", {}).get("provider"),
            "mode": CONFIG.get("rag", {}).get("mode"),
            "fallback_to_chroma": CONFIG.get("vector_store", {}).get("fallback_to_chroma"),
        },
        "credentials": credential_status(),
        "tracing": bool((CONFIG.get("observability") or {}).get("tracing")),
    }

    def render(d: dict[str, Any]) -> None:
        # Lead with the one question a user actually has: can I ask a question
        # right now? Everything below is the detail behind this answer. A user
        # who has to read to the bottom of a table to learn their key is missing
        # is the problem this command exists to solve.
        usable = d["usable"]
        if usable:
            formatting.line(
                f"  [green]Ready.[/green] Asking a question will work: "
                f"{d['provider']} via {d['model']}."
            )
        else:
            formatting.line(
                f"  [red]Not ready.[/red] No usable API key for {d['provider']}, "
                f"so questions will fail."
            )
            formatting.line("  [cyan]terminus setup[/cyan] to fix it.")
        if d["other_key_exists"]:
            formatting.line(
                "  [dim]A key for a different provider is configured - "
                "[cyan]terminus config set llm.provider <name>[/cyan] may be "
                "all that is needed.[/dim]"
            )
        formatting.line("")
        formatting.heading("Route")
        formatting.line(f"  provider      : {d['provider']}")
        formatting.line(f"  model         : {d['model']}")
        if d["planner_model"] and d["planner_model"] != d["model"]:
            formatting.line(f"  planner model : {d['planner_model']}")
        if d["judge_model"] and d["judge_model"] != d["model"]:
            formatting.line(f"  judge model   : {d['judge_model']}")
        formatting.line(f"  streaming     : {d['streaming']}")
        layers = d["config_layers"]
        if len(layers) > 1:
            formatting.line("  config layers : (lowest first)")
            for layer in layers:
                formatting.line(f"    {layer['kind']:<8} {layer['path']}")
        else:
            formatting.line(f"  config source : {d['config_source']}")
        formatting.line("")
        formatting.heading("Primary")
        formatting.line(f"  endpoint      : {primary.get('endpoint')}")
        formatting.line(
            f"  api key       : "
            f"{'configured' if primary.get('api_key_configured') else 'MISSING'}"
        )
        # Named for what it measures. "status" next to a MISSING key reads as
        # "available" contradicting itself; this is only the rate-limit state.
        formatting.line(f"  rate limited  : {primary.get('status')}")
        formatting.line("")
        if d["fallbacks"]:
            formatting.heading("Fallbacks, in order")
            for f in d["fallbacks"]:
                state = "key ok" if f["key_configured"] else "NO KEY"
                formatting.line(
                    f"  {f['provider']:<14} {str(f['model'] or 'inherit'):<26} "
                    f"{state}, rate limited: {f['status']}"
                )
        else:
            formatting.line("  No fallbacks configured.")
        if d["exhausted"]:
            formatting.line("")
            formatting.line(f"  Currently set aside: {d['exhausted']}")
        formatting.line("")
        formatting.heading("Retrieval")
        vs = d["vector_store"]
        formatting.line(f"  store         : {vs['provider']} / {vs['mode']}")
        formatting.line(f"  chroma fallback: {'on' if vs['fallback_to_chroma'] else 'off'}")
        formatting.line(f"  embeddings    : {d['embeddings'].get('provider')} / {d['embeddings'].get('model')}")
        formatting.line("")
        formatting.heading("Credentials")
        for c in d["credentials"]:
            if c["configured"]:
                formatting.line(f"  {c['name']:<22} configured  ({c['source']})")
        missing = [c["name"] for c in d["credentials"] if not c["configured"]]
        if missing:
            formatting.line(f"  not set: {', '.join(missing)}")
        formatting.line("")
        formatting.line("  Values are never shown. 'terminus doctor' for a full check.")

    formatting.emit(payload, as_json=as_json, render=render)


@config_app.command("unset")
def config_unset(
    key: str = typer.Argument(..., help="Dotted key to remove, e.g. llm.model."),
    scope: str = typer.Option("project", "--scope", help="'project' or 'global'."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Remove a setting from a config file, so a lower layer takes effect again.

    Only meaningful for a file that actually sets the key. Removing it from the
    project file is how you fall back to your global default.
    """
    from terminus.cli_app.settings import unset_value

    try:
        path, removed = unset_value(key, scope=scope)
    except KeyError:
        formatting.usage_error(
            f"Unknown setting '{key}'.",
            "Run: terminus config show",
        )
        raise typer.Exit(code=2)
    except ValueError as exc:
        formatting.usage_error(str(exc))
        raise typer.Exit(code=2)

    def render(d: dict[str, Any]) -> None:
        if d["removed"]:
            formatting.success(f"Removed {d['key']} from {d['path']}")
        else:
            formatting.line(f"{d['key']} was not set in {d['path']}; nothing changed.")

    formatting.emit({"key": key, "path": str(path), "removed": removed}, as_json=as_json, render=render)


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
    value: str = typer.Argument(..., help="Value. Parsed as JSON, so numbers and booleans ke"),
    scope: str = typer.Option(
        "project", "--scope",
        help="'project' (default) writes this repository's config.yaml. "
             "'global' writes ~/.terminus/config.yaml, so it applies everywhere. "
             "A project value overrides a global one for the keys it names.",
    ),
) -> None:
    """Persist a setting to config.yaml."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    try:
        path, written = settings.set_value(key, parsed, scope=scope)
    except KeyError:
        formatting.usage_error(
            f"Unknown setting '{key}'. Nothing was written.",
            "Run: terminus config list",
        )
        raise typer.Exit(code=2)
    except ValueError as exc:
        formatting.usage_error(str(exc), "Use --scope project or --scope global.")
        raise typer.Exit(code=2)
    where = "every project" if scope == "global" else "this project"
    formatting.success(f"{key} = {json.dumps(written, default=str)}   ({path}, {where})")
    # Re-resolve in place, so the change applies to the rest of this session.
    # CONFIG is read once at import; without this, `config set llm.provider x`
    # in the middle of a session writes the file and then carries on using the
    # old provider for the next question - which reads as the command doing
    # nothing. Mutated rather than rebound, because a dozen modules hold a
    # reference to that exact dict.
    import terminus.config as config_module

    try:
        config_module.CONFIG.clear()
        config_module.CONFIG.update(config_module.load_config())
    except Exception as exc:  # never fail a successful write over a refresh
        formatting.warn(f"Saved, but this session is still using the old value: {exc}")


@agent_app.callback(invoke_without_command=True)
def agent_root(
    ctx: typer.Context,
    prompt: str = typer.Argument(None, metavar="[PROMPT]", help="Prompt to run, as a positional argument."),
    prompt_flag: str = typer.Option(
        None, "--prompt", "-p", metavar="TEXT",
        help="Prompt to run, as an option. Equivalent to the positional form.",
    ),
    model: str = typer.Option(None, "--model", "-m", help="Model for this run only. Not persisted."),
    provider: str = typer.Option(None, "--provider", help="Provider for this run only. Not persisted."),
    session: str = typer.Option(None, "--session", "-s", help="Session id to use for this run."),
    plan: bool = typer.Option(False, "--plan", help="Plan for the goal instead of answering it."),
    show_usage: bool = typer.Option(False, "--usage", help="Print a token summary when the run finishes."),
    dev: bool = typer.Option(
        False, "--dev",
        help="Full diagnostic logging for this run. Same as --log-level DEBUG.",
    ),
) -> None:
    """Run one prompt, or open the interactive session.

    The prompt can be given either way - positionally, or with -p/--prompt:

        terminus agent "where is the retry logic?"
        terminus agent -p "where is the retry logic?"

    With no prompt you get the REPL, with its slash commands. The -p form is
    the documented one because it stays unambiguous when the prompt begins with
    a dash; the positional form is kept because it reads better in a shell and
    because it is what this command has always accepted.
    """
    if ctx.invoked_subcommand is not None:
        return
    if prompt and prompt_flag:
        # Both spellings of the same argument, with two different values. Picking
        # one silently would run the wrong question.
        # format.usage_error rather than typer.UsageError: there is no
        # `typer.UsageError`, so the obvious spelling raised AttributeError and
        # the user got a traceback instead of this sentence.
        formatting.usage_error(
            "Give the prompt either positionally or with -p/--prompt, not both.",
            "terminus agent -p \"your prompt\"",
        )
        raise typer.Exit(code=2)
    prompt = prompt or prompt_flag
    if not prompt:
        if model or provider or session or plan:
            formatting.usage_error(
                "--model, --provider, --session and --plan all apply to a single prompt.",
                "terminus agent -p \"your prompt\" --model <model>",
            )
            raise typer.Exit(code=2)
        formatting.line("No prompt given - starting the interactive session. See terminus agent --help.")
        raise typer.Exit(code=formatting.launch_repl())

    import anyio

    from terminus.agent.orchestrator import handle_query
    from terminus.cli import format_startup_error, initialize, shutdown_resources
    from terminus.memory.session import get_current_session, switch_session
    from terminus.observability.logging import set_log_level

    if dev:
        # Also accepted here, because `--dev` before the subcommand is easy to
        # forget and the agent command is where the noise actually shows up.
        set_log_level("DEBUG")

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
                from terminus.tasks.orchestrator import handle_plan_command

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
