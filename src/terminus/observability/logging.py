"""Logging setup, in one place.

A single ``basicConfig`` at import time, and a ``get_logger`` that does *not*
pin a level on each module. Pinning a level here (the earlier behaviour) silently
defeated every level the process set later: a logger with its own level ignores
the root logger's, so ``terminus --log-level DEBUG`` changed nothing at all.
Modules now inherit the root level, which the CLI owns.
"""

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_log_level(level: str | int) -> None:
    """Set the level for every Terminus logger.

    Sets the root logger *and* the ``terminus`` package logger, so it takes
    effect regardless of whether a handler was already installed on either.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):  # an unknown name
            level = logging.WARNING
    logging.getLogger().setLevel(level)
    logging.getLogger("terminus").setLevel(level)


def get_logger(module_name: str) -> logging.Logger:
    """The logger for a module.

    Deliberately does not call ``setLevel``: see the module docstring.
    """
    return logging.getLogger(module_name)
