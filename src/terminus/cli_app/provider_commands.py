"""``terminus providers`` and ``terminus models``.

Both read the provider list from :mod:`terminus.llm.providers`, the single
place it is declared, and the live routing from
:func:`terminus.llm.factory.get_provider_diagnostics`. Nothing here re-declares
which providers exist, which credential each needs, or what the fallback order
is: if the router changes, these commands change with it, because they are
reading it rather than copying it.
"""

from __future__ import annotations

from typing import Any

import typer

from terminus.cli_app import formatting, settings
from terminus.llm import providers as provider_table

providers_app = typer.Typer(help="Manage providers and their credentials.", no_args_is_help=False)
models_app = typer.Typer(help="View and select models.", no_args_is_help=False)


def _json_flag(ctx: typer.Context, local: bool = False) -> bool:
    return bool((ctx.obj or {}).get("json", False)) or local


def _diagnostics() -> dict[str, Any]:
    from terminus.llm.factory import get_provider_diagnostics

    return get_provider_diagnostics()


def _known(provider: str) -> str | None:
    """Canonical lowercase name if *provider* is one we know, else None."""
    return provider.strip().lower() if provider_table.get(provider) else None


@providers_app.callback(invoke_without_command=True)
def providers_root(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Show configured providers. Equivalent to ``providers list``."""
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(providers_list, as_json=_json_flag(ctx, json_output))


@providers_app.command("list")
def providers_list(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """List providers Terminus knows about and whether each is configured."""
    active = _diagnostics().get("provider")
    entries = [
        {
            "provider": name,
            "label": spec.label,
            "auth_method": spec.auth_method,
            "api_key_configured": provider_table.has_credential(name),
            "credential_env": " or ".join(spec.env_keys),
            "active": name == active,
        }
        for name, spec in sorted(provider_table.PROVIDERS.items())
    ]
    formatting.emit(
        {"providers": entries, "active": active},
        as_json=as_json,
        render=lambda _d: formatting.show_table(
            "Providers",
            ["provider", "name", "configured", "auth", "role"],
            [
                [e["provider"], e["label"], "yes" if e["api_key_configured"] else "no",
                 e["auth_method"], "active" if e["active"] else "fallback"]
                for e in entries
            ],
        ),
    )


@providers_app.command("status")
def providers_status(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Show the live route: primary, fallbacks, timeouts and retry policy."""
    data = _diagnostics()
    routes = [data.get("primary_route") or {}, *(data.get("fallback_routes") or [])]
    payload = {
        "active_provider": data.get("provider"),
        "endpoint": data.get("endpoint"),
        "routes": routes,
        "fallbacks": data.get("fallbacks"),
        "retry_policy": data.get("retry_policy"),
        "exhausted": data.get("exhausted_providers"),
        "streaming": data.get("streaming"),
    }

    def render(_d: dict[str, Any]) -> None:
        formatting.heading("Route status")
        formatting.line(f"  active provider : {payload['active_provider']}")
        formatting.line(f"  endpoint        : {payload['endpoint']}")
        formatting.line(f"  streaming       : {payload['streaming']}")
        formatting.line(f"  retry policy    : {payload['retry_policy']}")
        formatting.table(
            "Routes", ["role", "provider", "endpoint", "configured", "status"],
            [
                [
                    "active" if route is routes[0] else "fallback" if routes else "",
                    route.get("provider", ""),
                    route.get("endpoint", ""),
                    "yes" if route.get("api_key_configured") else "no",
                    route.get("status", ""),
                ]
                for route in routes
            ],
        )

    formatting.emit(payload, as_json=as_json, render=render)


@providers_app.command("login")
def providers_login(
    provider: str = typer.Option(..., "--provider", "-p", help="Provider to configure."),
    method: str = typer.Option(None, "--method", "-m", help="Auth method; api-key is the only one Terminus supports."),
    api_key: str = typer.Option(None, "--api-key", help="Key as an argument. Visible in shell history - prefer the prompt."),
    scope: str = typer.Option(
        "global", "--scope",
        help="'global' (default) stores it for this user, in ~/.terminus, so it "
             "works in every project. 'project' stores it in this repository's .env.",
    ),
) -> None:
    """Store a provider credential.

    Prompts for the key without echoing it. ``--api-key`` is supported for
    scripting but is a poor choice interactively: it lands in shell history.

    Stored globally by default, so one login covers every project. Inside a
    repository the project's own ``.env`` still wins, which is what you want when
    a project deliberately uses a different account.
    """
    name = _known(provider)
    spec = provider_table.get(name) if name else None
    if spec is None:
        formatting.usage_error(
            f"Unknown provider '{provider}'.",
            f"Known providers: {', '.join(sorted(provider_table.PROVIDERS))}",
            "Run: terminus providers list",
        )
        raise typer.Exit(code=2)
    if method and method != spec.auth_method:
        formatting.usage_error(
            f"Provider '{name}' does not support --method {method}.",
            f"Supported: {spec.auth_method}",
        )
        raise typer.Exit(code=2)

    if api_key:
        value, warned = api_key, True
    else:
        value, warned = typer.prompt(f"{spec.label} API key", hide_input=True), False

    if not value.strip():
        formatting.fail("No key entered. Nothing was changed.")
        raise typer.Exit(code=1)

    # Written under the first accepted name for the provider. A provider with
    # two accepted names (google / google_genai) is read under either, so the
    # first is enough and there is no need to ask which one the user prefers.
    env_key = spec.primary_env
    from terminus.user_config import store_credential

    try:
        path = store_credential(env_key, value.strip(), scope=scope)
    except ValueError as exc:
        formatting.usage_error(str(exc), "Use --scope global or --scope project.")
        raise typer.Exit(code=2) from None
    where = "for this user, in every project" if scope == "global" else "for this project"
    formatting.success(f"Stored {env_key} {where}: {path}")
    if warned:
        formatting.warn("That key was passed on the command line and may persist in shell history.")
    formatting.line("  Check it took effect: terminus config show")


@providers_app.command("logout")
def providers_logout(
    provider: str = typer.Option(..., "--provider", "-p", help="Provider to disconnect."),
    scope: str = typer.Option(
        "both", "--scope",
        help="'both' (default) clears the global and project copies; 'global' or "
             "'project' clears just one. Removing from only one often leaves the "
             "other still supplying the key, which looks like logout did not work.",
    ),
) -> None:
    """Remove a stored provider credential."""
    name = _known(provider)
    spec = provider_table.get(name) if name else None
    if spec is None:
        formatting.usage_error(
            f"Unknown provider '{provider}'.",
            f"Known providers: {', '.join(sorted(provider_table.PROVIDERS))}",
        )
        raise typer.Exit(code=2)
    from terminus.user_config import clear_credential

    _, removed = clear_credential(spec.primary_env, scope=scope)
    if removed:
        formatting.success(f"Removed {spec.primary_env} (scope: {scope})")
        formatting.line("  Check it took effect: terminus config show")
    else:
        formatting.warn(f"No stored {spec.primary_env} found (scope: {scope}).")


@models_app.callback(invoke_without_command=True)
def models_root(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List models. ``terminus models`` is the same as ``models list``.

    Filtering is ``models list <provider>``. A bare filter on the group is not
    supported: Click resolves the first token after a group name as a
    subcommand, so ``models groq`` would be read as a command named ``groq``.
    """
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(models_list, provider=None, as_json=_json_flag(ctx, json_output))


@models_app.command("list")
def models_list(
    provider: str = typer.Argument(None, help="Only show models for this provider."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include route detail."),
) -> None:
    """List the models Terminus is configured to use, and what each provider offers.

    The active route and the fallback chain come from the live diagnostics. A
    provider whose catalogue Terminus does not enumerate still appears, with no
    models listed, rather than being omitted - the configured model is always
    shown, so the list is never silently wrong, only not exhaustive.
    """
    from terminus.config import CONFIG

    data = _diagnostics()
    llm = CONFIG.get("llm", {})
    active_provider = str(llm.get("provider") or data.get("provider") or "")
    active_model = str(llm.get("model") or data.get("requested_model") or "")

    entries: list[dict[str, Any]] = [{
        "provider": active_provider,
        "model": active_model,
        "role": "active",
        "endpoint": data.get("endpoint", ""),
    }]
    for fallback in data.get("fallbacks") or []:
        entries.append({
            "provider": fallback.get("provider", ""),
            "model": fallback.get("model", ""),
            "role": "fallback",
            "endpoint": fallback.get("endpoint", ""),
        })
    for name, spec in sorted(provider_table.PROVIDERS.items()):
        for model in spec.models:
            if any(e["provider"] == name and e["model"] == model for e in entries):
                continue
            entries.append({"provider": name, "model": model, "role": "available", "endpoint": ""})

    if provider:
        wanted = provider.strip().lower()
        entries = [e for e in entries if e["provider"] == wanted]

    columns = ["role", "provider", "model", "endpoint"] if verbose else ["role", "provider", "model"]
    formatting.emit(
        {"models": entries, "active": {"provider": active_provider, "model": active_model}},
        as_json=as_json,
        render=lambda _d: formatting.show_table(
            "Models",
            columns,
            [[e[column] for column in columns] for e in entries],
        ),
    )


@models_app.command("set")
def models_set(
    model: str = typer.Argument(..., help="Model id, taken whole. Ids contain '/', e.g. poolside/laguna-s-2.1:free."),
    provider: str = typer.Option(None, "--provider", "-p", help="Provider. Defaults to the configured one."),
) -> None:
    """Persist the default model. Use ``agent --model`` for a single run instead.

    The model id is stored whole, never split on ``/``: ids like
    ``poolside/laguna-s-2.1:free`` and ``openai/gpt-oss-120b`` are normal values,
    and a guessed split would persist the wrong model.
    """
    from terminus.config import CONFIG

    if not model.strip():
        formatting.usage_error("No model given.", "Usage: terminus models set <model> --provider <provider>")
        raise typer.Exit(code=2)
    chosen = provider.strip().lower() if provider else str(
        CONFIG.get("llm", {}).get("provider") or ""
    )
    if not provider_table.get(chosen):
        formatting.usage_error(
            f"Unknown provider '{chosen}'. Nothing was written.",
            f"Known providers: {', '.join(sorted(provider_table.PROVIDERS))}",
            "Run: terminus models list",
        )
        raise typer.Exit(code=2)

    settings.set_value("llm.model", model.strip())
    settings.set_value("llm.provider", chosen)
    formatting.success(f"Default model set to {chosen} / {model.strip()}")
    formatting.line("  This changed the persistent default. For one run only, use:")
    formatting.line("    terminus agent --provider <provider> --model <model> -p '...'")
