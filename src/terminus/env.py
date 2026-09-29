"""Finding the project's .env.

Searched upward from the working directory, so ``terminus`` can be started from a
subdirectory and still find the credentials belonging to the project above it.
A project file is optional throughout: credentials may already be in the process
environment, and a missing .env is not an error.

``TERMINUS_ENV_FILE`` overrides the search with an explicit path, which is what
you want for a checkout that keeps credentials somewhere other than beside the
code.

Loading never overrides an existing environment variable (``override=False``), so
a variable exported in the shell wins over the file. That ordering is the
conventional one and it means a surprising .env cannot silently replace a
deliberately-set value.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def find_project_env(start_path: str | Path | None = None) -> Path | None:
    start = Path(start_path or Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    configured = os.getenv("TERMINUS_ENV_FILE")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_project_env(start_path: str | Path | None = None) -> Path | None:
    env_file = find_project_env(start_path)
    if env_file is not None:
        load_dotenv(env_file, override=False)
    return env_file
