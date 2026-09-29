"""Presentation only: Rich for humans, JSON for machines.

Commands return plain data; this module decides how it is drawn. That split is
the point - a command never knows whether it is being read by a person or
piped into ``jq``, so ``--json`` is always clean and Rich markup can never leak
into machine-readable output.
"""

from __future__ import annotations

import json
from typing import Any
from collections.abc import Iterable, Sequence

from rich.console import Console
from rich.table import Table

console = Console()
error_console = Console(stderr=True)

JSON_FORMAT = "json"

# Set by the root callback for `terminus --json <command>`. A per-command
# --json is also accepted, and either one is enough. Holding it here means a
# command never has to thread the parent context through just to honour it, and
# a command added later inherits the behaviour for free.
_force_json = False


def set_force_json(value: bool) -> None:
    global _force_json
    _force_json = bool(value)


def emit(data: Any, *, as_json: bool, render) -> None:
    """Print *data* as JSON, or hand it to the Rich renderer.

    JSON goes to stdout and nothing else does in JSON mode, so the output is
    always parseable.
    """
    if as_json or _force_json:
        print(json.dumps(data, indent=2, default=str))
        return
    render(data)


def table(title: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> Table:
    built = Table(title=title, header_style="bold", show_lines=False)
    for column in columns:
        built.add_column(column, overflow="fold")
    for row in rows:
        built.add_row(*["" if cell is None else str(cell) for cell in row])
    return built


def show_table(title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    console.print(table(title, columns, rows))


def heading(text: str) -> None:
    console.print(f"[bold cyan]{text}[/bold cyan]")


def line(text: str = "") -> None:
    console.print(text)


def success(text: str) -> None:
    console.print(f"[green]{text}[/green]")


def warn(text: str) -> None:
    console.print(f"[yellow]{text}[/yellow]")


def fail(text: str) -> None:
    error_console.print(f"[bold red]{text}[/bold red]")


def usage_error(message: str, *hints: str) -> None:
    """A user-facing error with a next step, and no traceback."""
    fail(message)
    for hint in hints:
        error_console.print(f"  {hint}")


def panel(title: str, body: str) -> None:
    from rich.panel import Panel

    console.print(Panel(body, title=title, border_style="cyan"))


def launch_repl() -> int:
    """Hand off to the interactive session, mapping interrupts to a clean exit.

    The REPL prompts with ``Prompt.ask``, so end-of-input (a pipe, ``Ctrl-D``)
    and ``Ctrl-C`` arrive as exceptions. Both are a normal way to stop typing, so
    they exit 0 rather than surfacing a traceback - a bare ``terminus < /dev/null``
    should not look like a crash. Anything else still propagates.
    """
    import contextlib

    from terminus.cli import run

    try:
        with contextlib.suppress(EOFError):
            return int(run() or 0)
    except KeyboardInterrupt:
        line("")
    return 0
