from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "llm": {
        "provider": "openrouter",
        "model": "poolside/laguna-s-2.1:free",
        "planner_model": "poolside/laguna-s-2.1:free",
        "judge_model": "poolside/laguna-s-2.1:free",
        "request_timeout_seconds": 120,
        "max_retries": 0,
        "streaming": False,
        "fallbacks": [
            {"provider": "google_genai", "model": "gemini-3.5-flash-lite"},
            {"provider": "cohere", "model": "command-r-plus-08-2024"},
        ],
        "route_max_attempts": 2,
        "route_backoff_seconds": 5,
    },
    "memory": {
        "db_path": ".terminus/memory/terminus.db",
        "summarize_at_tokens": 4000,
        "max_messages": 20,
    },
    "tasks": {
        "db_path": ".terminus/tasks/tasks.db",
        "agent_timeout_seconds": 900,
        "judge_timeout_seconds": 300,
        "mcp_timeout_seconds": 120,
        "rate_limit_backoff_seconds": 20,
        "index_timeout_seconds": 300,
        # Opt-in concurrency. 1 is the default and stays serial until a project
        # explicitly raises this. Bound by MAX_CONCURRENT_TASKS in
        # terminus.tasks.orchestrator so config cannot fan out arbitrarily.
        "max_concurrent": 1,
    },
    "embeddings": {
        "provider": "huggingface",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "rag": {"mode": "semantic"},
    "vector_store": {
        "provider": "chromadb",
        "retrieval_mode": "semantic",
        "fallback_to_chroma": True,
    },
    "qdrant": {
        "collection_name": "terminus_hybrid",
        "timeout_seconds": 5,
    },
    "chromadb": {
        "persist_dir": ".terminus/chromadb/",
        "collection_name": "terminus",
    },
}

CONFIG_SOURCE: Path | None = None
CONFIG_SOURCE_KIND = "default"


def _merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config() -> dict:
    global CONFIG_SOURCE, CONFIG_SOURCE_KIND
    candidates = [
        (Path.cwd() / "config.yaml", "cwd"),
        (Path(__file__).parent / "config.yaml", "package"),
    ]
    for path, kind in candidates:
        if not path.exists():
            continue
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Unable to load configuration {path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration {path} must contain a YAML mapping")
        CONFIG_SOURCE = path.resolve()
        CONFIG_SOURCE_KIND = kind
        return _merge(DEFAULT_CONFIG, loaded)
    CONFIG_SOURCE = None
    CONFIG_SOURCE_KIND = "default"
    return deepcopy(DEFAULT_CONFIG)


CONFIG = load_config()
