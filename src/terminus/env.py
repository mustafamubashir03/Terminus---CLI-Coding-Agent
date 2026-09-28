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
