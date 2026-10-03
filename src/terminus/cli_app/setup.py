"""``terminus setup`` and ``terminus doctor``.

Two commands for the two moments that matter to a new user. ``setup`` is the
one-off: point Terminus at a provider and give it a key, once. ``doctor`` is
the "something is wrong, tell me what" command, for later.

``setup`` has two forms, because the two callers are different:

* interactive - the default, and the only form that can ask for a key without it
  ending up in the shell history;
* non-interactive - ``--provider``, ``--model``, ``--key-stdin``, for a script
  or a container that has no terminal to prompt at.

Both write to the same place: the developer's own ``~/.terminus``, by default, so
one setup covers every project they open afterwards. ``--scope project`` is there
for a repository that genuinely needs a different key.

The key is never echoed, never logged, and never written to ``config.yaml``.
"""

from __future__ import annotations

import getpass
import sys

import typer

from terminus.cli_app import formatting

setup_app = typer.Typer(
    help="Configure Terminus for first use.",
    no_args_is_help=False,
    invoke_without_command=True,
)

doctor_app = typer.Typer(
    help="Check that this installation can actually work.",
    no_args_is_help=False,
    invoke_without_command=True,
)

#: Shown during setup. Order is the order offered, cheapest and most forgiving
#: first - a free-tier OpenRouter model is a reasonable first stop because it
#: needs no card, and Groq is a strong free tier for speed.
RECOMMENDED: tuple[tuple[str, str, str], ...] = (
    ("openrouter", "poolside/laguna-s-2.1:free", "OpenRouter - many models, free tier available"),
    ("groq", "openai/gpt-oss-120b", "Groq - fast, generous free tier"),
    ("cerebras", "llama-3.3-70b", "Cerebras - very fast inference"),
    ("anthropic", "claude-sonnet-4-5", "Anthropic - strongest, needs a paid plan"),
    ("openai", "gpt-4o-mini", "OpenAI"),
    ("cohere", "command-r-plus-08-2024", "Cohere"),
)


def _read_key_interactively(env_key: str) -> str:
    """Prompt for a key without echoing it.

    ``getpass`` keeps it off the screen and out of the shell history, which is
    the entire reason to prefer this over ``--key``. It reads from the
    controlling terminal, so a piped stdin is not a credential.
    """
    formatting.line("")
    formatting.line(f"  Enter your {env_key}.")
    formatting.line("  It is not shown as you type, and is not stored in your shell history.")
    formatting.line("")
    try:
        value = getpass.getpass(f"  {env_key}: ")
    except (EOFError, KeyboardInterrupt):
        raise typer.Exit(code=1) from None
    return value.strip()


def _read_key_from_stdin() -> str:
    """Read a key from stdin, for a script that pipes one in.

    Stdin is read as a whole so the value can arrive with or without a trailing
    newline, which is what ``echo`` and ``cat`` produce respectively.
    """
    data = sys.stdin.readline()
    return data.strip()


@setup_app.callback(invoke_without_command=True)
def setup(
    ctx: typer.Context,
    provider: str = typer.Option(
        None, "--provider", help="Configure this provider non-interactively."
    ),
    model: str = typer.Option(None, "--model", help="Model to use with --provider."),
    key_stdin: bool = typer.Option(
        False, "--key-stdin",
        help="Read the API key from stdin instead of prompting. Use this in "
             "scripts; it keeps the key out of the process arguments.",
    ),
    api_key: str = typer.Option(
        None, "--api-key",
        help="Supply the key directly. SECURITY: this appears in your shell "
             "history and in the process list. Prefer the prompt, or --key-stdin.",
    ),
    scope: str = typer.Option(
        "global", "--scope",
        help="Where to store the key and settings: 'global' (this user, every "
             "project) or 'project' (this repository's .env / config.yaml).",
    ),
    force: bool = typer.Option(
        False, "--force", help="Run setup again even if a provider already works."
    ),
) -> None:
    """Set up a provider, a model and a key.

    Run with no arguments for the guided version. Everything it asks for is
    written to ``~/.terminus`` by default, so it applies to every project.
    """
    if ctx.invoked_subcommand is not None:
        return
    if scope not in ("global", "project"):
        formatting.usage_error(
            f"Unknown --scope {scope!r}.",
            "Use --scope global (default) or --scope project.",
        )
        raise typer.Exit(code=2)

    from terminus.config import CONFIG
    from terminus.user_config import configured_provider_has_credential

    already = configured_provider_has_credential()
    if already and not force:
        # Naming a provider is not consent to replace a working setup. This
        # guard used to read `not (provider or force)`, so a scripted
        # `setup --provider X` silently repointed a working installation.
        current = (CONFIG.get("llm") or {}).get("provider", "the current provider")
        formatting.heading("Terminus is already configured")
        formatting.line("")
        formatting.line(f"  A working key for [cyan]{current}[/cyan] is already stored,")
        formatting.line("  so nothing was changed.")
        formatting.line("")
        formatting.line("  [cyan]terminus config show[/cyan]      what Terminus will use")
        formatting.line("  [cyan]terminus doctor[/cyan]          check this installation")
        formatting.line("  [cyan]terminus setup --force[/cyan]   configure a different provider")
        formatting.line("")
        # A bare `terminus setup` on a configured install is a friendly "you are
        # done", so it succeeds. Being *asked* to configure a provider and then
        # quietly not doing it is a script that will carry on with the wrong
        # provider, so that has to be visible in the exit code.
        raise typer.Exit(code=1 if provider else 0)

    if provider:
        _setup_non_interactive(provider, model, key_stdin, api_key, scope)
        return

    _setup_interactive(scope)


def _refresh_live_config() -> None:
    """Re-resolve everything in this process after setup has written new files.

    ``CONFIG`` is built once at import, from whatever was on disk before setup
    ran, and the credential files are only read into the environment at startup.
    Left alone, everything setup says afterwards - and anything else in this
    process - still sees the old provider, so the confirmation it prints
    ("Configured: groq") is immediately contradicted by a "not ready" verdict.
    Loading the files we just wrote is also the only way setup can honestly
    report readiness at all.
    """
    try:
        import terminus.config as config_module
        from terminus.user_config import load_env_files

        config_module.CONFIG = config_module.load_config()
        load_env_files()
    except Exception:  # never let a refresh turn a good setup into a failure
        pass


def _setup_non_interactive(
    provider: str, model: str | None, key_stdin: bool, api_key: str | None, scope: str
) -> None:
    from terminus.llm.providers import get

    spec = get(provider)
    if spec is None:
        formatting.usage_error(
            f"Unknown provider {provider!r}.",
            "terminus setup            to see the providers offered",
        )
        raise typer.Exit(code=2)
    if not spec.env_keys:
        formatting.fail(f"Provider {provider!r} does not use an API key.")
        raise typer.Exit(code=1)

    if api_key:
        formatting.line("")
        formatting.line(
            "  WARNING: --api-key puts the secret in your shell history and in "
            "the process list."
        )
        formatting.line("  Use the prompt, or --key-stdin, to avoid that.")
        value = api_key
    elif key_stdin:
        value = _read_key_from_stdin()
    else:
        value = _read_key_interactively(spec.env_keys[0])

    if not value:
        formatting.fail("No key was provided, so nothing was changed.")
        raise typer.Exit(code=1)

    chosen = _apply(provider, model, value, spec.env_keys[0], scope)
    _refresh_live_config()
    formatting.heading("Configured")
    formatting.line("")
    formatting.line(f"  provider : {provider}")
    formatting.line(f"  model    : {chosen or 'unchanged'}")
    formatting.line(f"  key      : {spec.env_keys[0]} stored")
    formatting.line("  run 'terminus doctor' to verify, or just ask a question.")


def _setup_interactive(scope: str) -> None:
    from rich.prompt import Prompt

    from terminus.llm.providers import get

    formatting.heading("Terminus setup")
    formatting.line("")
    formatting.line("  No provider is configured yet. Pick one to get started.")
    formatting.line("")
    for index, (name, _model, blurb) in enumerate(RECOMMENDED, 1):
        formatting.line(f"    {index}. {name:<12} {blurb}")
    formatting.line("")

    choice = Prompt.ask(
        "  Provider",
        choices=[str(i) for i in range(1, len(RECOMMENDED) + 1)] + ["n"],
        default="1",
        show_choices=False,
    )
    if choice.lower() == "n":
        formatting.line("")
        formatting.line("  Nothing changed. Run 'terminus setup' when you're ready.")
        return
    provider = RECOMMENDED[int(choice) - 1][0]
    suggested_model = RECOMMENDED[int(choice) - 1][1]

    model = Prompt.ask("  Model", default=suggested_model)
    spec = get(provider)
    if spec is None or not spec.env_keys:
        formatting.fail(f"Provider {provider!r} does not use an API key.")
        raise typer.Exit(code=1)

    value = _read_key_interactively(spec.env_keys[0])
    if not value:
        formatting.fail("No key provided, so nothing was changed.")
        raise typer.Exit(code=1)

    _apply(provider, model, value, spec.env_keys[0], scope)
    _refresh_live_config()

    formatting.line("")
    formatting.line("  Optional: a fallback provider, so a rate limit does not end")
    formatting.line("  your question. 'terminus config set llm.fallbacks' takes a list.")
    formatting.line("")
    formatting.heading("Configured")
    formatting.line("")
    formatting.line(f"  provider : {provider}")
    formatting.line(f"  model    : {model}")
    formatting.line(f"  key      : {spec.env_keys[0]} stored")
    formatting.line("")
    formatting.line("  Start a session with 'terminus', or ask one question with:")
    formatting.line("    terminus agent -p \"what is this project?\"")
    formatting.line("  Check everything with 'terminus doctor'.")


def recommended_model(provider: str) -> str | None:
    """The model this project recommends for *provider*.

    Used whenever a provider is chosen without a model. Writing the provider
    and leaving the previous model in place produces a configuration that looks
    configured and cannot work - groq pointed at an OpenRouter model fails
    every request with model_not_found - which is the worst possible outcome
    for the command whose entire job is to make things work.
    """
    for name, model, _blurb in RECOMMENDED:
        if name == provider:
            return model
    return None


def _apply(provider: str, model: str, api_key: str, env_key: str, scope: str) -> str:
    """Write the key to the chosen store and point config at provider/model.

    Returns the model that ended up configured, so the caller can report the
    route it actually produced rather than the one it was asked for.
    """
    from terminus.cli_app.settings import set_value
    from terminus.user_config import store_credential

    store_credential(env_key, api_key, scope=scope)
    set_value("llm.provider", provider, scope=scope)
    chosen = model or recommended_model(provider)
    if chosen:
        set_value("llm.model", chosen, scope=scope)
    return chosen or ""


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_MARK = {PASS: "[bold green]PASS[/bold green]", WARN: "[bold yellow]WARN[/bold yellow]", FAIL: "[bold red]FAIL[/bold red]"}


def _line(status: str, label: str, detail: str = "") -> None:
    suffix = f"  [dim]{detail}[/dim]" if detail else ""
    formatting.line(f"  {_MARK[status]}  {label}{suffix}")


@doctor_app.callback(invoke_without_command=True)
def doctor(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Check that Terminus can work here, and say what is missing if it cannot."""
    if ctx.invoked_subcommand is not None:
        return
    from terminus.doctor import collect_report

    report = collect_report()
    if as_json:
        import json

        print(json.dumps(report, indent=2, default=str))
        raise typer.Exit(code=0 if report["ok"] else 1)

    formatting.heading("Terminus doctor")
    formatting.line("")
    for section, rows in report["sections"].items():
        formatting.line(f"  [bold]{section}[/bold]")
        for row in rows:
            _line(row["status"], row["label"], row.get("detail", ""))
        formatting.line("")

    summary = report["summary"]
    formatting.line(
        f"  {summary['pass']} passed, {summary['warn']} warnings, {summary['fail']} failures"
    )
    if not report["ok"]:
        formatting.line("")
        formatting.line("  Run 'terminus setup' if a provider is not configured.")
    formatting.line("")
    raise typer.Exit(code=0 if report["ok"] else 1)
