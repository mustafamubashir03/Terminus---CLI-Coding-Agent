"""Configuration: the defaults, and the one place a config file is resolved.

Precedence, highest first:

    the current directory's ``config.yaml``
    the packaged ``config.yaml`` (shipped defaults for an installed Terminus)
    ``DEFAULT_CONFIG`` below

The first file that exists wins outright - it is *not* a merge of all of them.
That is deliberate. Layering three files would make it impossible to say where
any single value came from, and a user who edits their project config would have
to know which of the other layers was overriding them.

``CONFIG`` is the resolved result, built once at import. Everything else in
Terminus reads it rather than re-reading the file, so there is exactly one
resolved configuration per process. A write goes through
``terminus.cli_app.settings.set_value``, which edits that same file in place so
``terminus config set`` and a hand edit cannot disagree.
"""

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
        # A credential-free local store is the default, so a fresh install works
        # with nothing configured. Chroma is the default provider because it is
        # local by construction and needs no further decision; Qdrant is a
        # one-line change and defaults to its own local mode.
        "provider": "chromadb",
        "retrieval_mode": "semantic",
        # Off by default. Substituting a different backend because the
        # configured one failed makes retrieval non-deterministic, and it used to
        # rewrite `provider` and `rag.mode` in the global config so the swap
        # outlived the process that hit the failure. Opt in deliberately and the
        # swap is reported rather than silent.
        "fallback_to_chroma": False,
    },
    "qdrant": {
        "collection_name": "terminus_hybrid",
        "timeout_seconds": 5,
        # "local" or "cloud". Left empty on purpose: an absent value resolves by
        # terminus.context.indexers.qdrant_client.qdrant_mode() to cloud when
        # CLUSTER_ENDPOINT is set and local otherwise, so an existing cloud
        # configuration keeps working untouched.
        "mode": "",
        "path": ".terminus/qdrant",
        "url": "",
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
