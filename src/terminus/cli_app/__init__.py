"""Terminus command-line interface.

Public entry point is :func:`terminus.cli_app.main.main`, wired to the
``terminus`` console script. :mod:`terminus.cli` keeps the interactive REPL.
"""

from terminus.cli_app.main import app, main

__all__ = ["app", "main"]
