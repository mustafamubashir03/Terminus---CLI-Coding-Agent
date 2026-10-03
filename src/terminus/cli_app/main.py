"""Terminus command-line interface.

Layout:

* ``main``        - the Typer application, global flags, and the bare-invocation
                    REPL handoff.
* ``provider_commands`` - ``providers`` and ``models``.
* ``commands``    - everything else.
* ``settings``    - reading/writing config.yaml, and the credential file.
* ``formatting``  - Rich for humans, JSON for machines.

The rule this package follows: a command decides *what* to show and hands the
result to :mod:`terminus.cli_app.formatting`, which decides *how*. That is what
keeps ``--json`` parseable - a command never prints, so it can never leak Rich
markup into machine-readable output.
"""

from __future__ import annotations

import typer

from terminus.cli_app import formatting

__all__ = ["main", "app"]


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("terminus")
    except PackageNotFoundError:  # running from a source tree
        return "0.0.0"


app = typer.Typer(
    name="terminus",
    help=(
        "Terminus - a multi-agent coding assistant.\n\n"
        "Run [bold]terminus[/bold] with no arguments to open the interactive session, "
        "or [bold]terminus agent -p 'your prompt'[/bold] to run one prompt and exit."    ),
    add_completion=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _register() -> None:
    """Attach sub-apps. Imported lazily so ``terminus --help`` stays fast."""
    from terminus.cli_app.commands import (
        agent_app,
        config_app,
        db_app,
        index_app,
        sessions_app,
        skills_app,
        tools_app,
    )
    from terminus.cli_app.provider_commands import models_app, providers_app
    from terminus.cli_app.setup import doctor_app, setup_app

    # Names are explicit. Typer 0.27 warns that add_typer without a name drops
    # the sub-app callback, which flattens every leaf command onto the root.
    #
    # `setup` and `doctor` are registered first and are plain commands rather than
    # groups: they are what a user with no configuration needs, so they must not
    # be buried behind a subcommand they have to know the name of.
    for name, sub in (
        ("setup", setup_app),
        ("doctor", doctor_app),
        ("agent", agent_app),
        ("providers", providers_app),
        ("models", models_app),
        ("sessions", sessions_app),
        ("tools", tools_app),
        ("skills", skills_app),
        ("config", config_app),
        ("db", db_app),
        ("index", index_app),
    ):
        app.add_typer(sub, name=name)


def _version_callback(value: bool) -> None:
    if value:
        formatting.console.print(f"terminus {_version()}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True,
        help="Show the version and exit.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Make the current command emit JSON."),
    dev: bool = typer.Option(
        False, "--dev",
        help="Turn on full diagnostic logging: every provider attempt, every "
             "fallback decision, and third-party request logs. Equivalent to "
             "--log-level DEBUG. Intended for reporting a problem, not for "
             "normal use - the output is very noisy.",
    ),
    log_level: str = typer.Option(
        "WARNING", "--log-level",
        help="Log level: DEBUG, INFO, WARNING, ERROR. Defaults to WARNING; "
             "DEBUG also enables third-party request logging.",
    ),
) -> None:
    """Terminus - a multi-agent coding assistant.

    With no subcommand this opens the interactive session, the same REPL that
    ``terminus agent`` gives you. Subcommands are for when you want one specific
    thing and want it to print rather than prompt.
    """
    import os

    from terminus.config import load_config
    from terminus.observability.logging import set_log_level

    # Re-resolve configuration against the directory this command was run from.
    # CONFIG is populated once at import, which is the right answer for whatever
    # directory the process started in and the wrong one here: the project layer
    # is cwd-relative, so a command (or a test) that has moved must not be
    # answered from the import-time location. load_config() updates CONFIG in
    # place, so every module already holding a reference sees the current answer.
    load_config()

    ctx.obj = {"json": json_output}
    formatting.set_force_json(json_output)
    # --dev is a named thing people reach for when something is wrong, so it wins
    # over an explicitly-given --log-level rather than fighting it.
    effective = "DEBUG" if dev else log_level
    os.environ.setdefault("TERMINUS_LOG_LEVEL", effective.upper())
    set_log_level(effective)

    if ctx.invoked_subcommand is not None:
        return
    # No second banner here. The session prints its own header the moment it
    # starts, and telling someone who just typed `terminus` to "run terminus
    # agent to start the session" is telling them to do what they already did.
    raise typer.Exit(code=formatting.launch_repl())


@app.command("version")
def version_command(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Show the version."""
    formatting.emit(
        {"version": _version()},
        as_json=as_json,
        render=lambda d: formatting.line(d["version"]),
    )


_register()


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
