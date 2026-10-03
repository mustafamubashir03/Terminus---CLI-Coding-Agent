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
    """Load credentials for this process, and return the project file if there was one.

    Kept as the single entry point everything already calls, so adding the global
    credential store did not mean hunting down call sites. The returned path is
    still the project ``.env`` because that is what callers report on;
    ``user_config.load_env_files`` returns the full set.
    """
    from terminus.user_config import load_env_files

    return load_env_files(start_path).get("project")
