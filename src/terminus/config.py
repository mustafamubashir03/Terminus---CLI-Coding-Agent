"""Configuration: the defaults, and the one place a config file is resolved.

Precedence, highest first:

    the current directory's ``config.yaml``
    ``~/.terminus/config.yaml``          (this developer's defaults, all projects)
    the packaged ``config.yaml``          (shipped defaults for an installed Terminus)
    ``DEFAULT_CONFIG`` below

The first file that exists wins outright - it is *not* a merge of all of them.
That is deliberate. Layering them would make it impossible to say where any
single value came from, and a user who edits their project config would have to
know which of the other layers was overriding them. The global file sits *below*
the project file for the same reason: a repository's own configuration is more
specific than a general default, and must not be surprised by it.

What each layer is for:

* project ``config.yaml`` - settings that belong to one repository: which vector
  store this project indexes into, project-specific model overrides.
* ``~/.terminus/config.yaml`` - settings that should follow the developer: their
  default provider and model. Written by ``terminus config set --global``.
* packaged ``config.yaml`` - the shipped defaults.

Credentials are deliberately **not** in any of these. They live in ``.env`` or
``~/.terminus/credentials.env``; see :mod:`terminus.user_config`.

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
    "observability": {
        # LangSmith tracing. Off unless asked for, deliberately: LangChain reads
        # LANGSMITH_TRACING straight out of the environment, so a key left in a
        # project's .env is enough to start shipping every prompt to a remote
        # service on every run - and to start printing that service's rate-limit
        # warnings into the middle of an answer when the quota runs out. Nothing
        # in Terminus needs tracing to work, so it earns an explicit switch.
        "tracing": False,
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
    },
    "chromadb": {
        "persist_dir": ".terminus/chromadb/",
        "collection_name": "terminus",
    },
}

CONFIG: dict = {}
CONFIG_SOURCE: Path | None = None
CONFIG_SOURCE_KIND = "default"
#: Every layer that actually contributed at least one key, lowest precedence
#: first, as (kind, path). ``CONFIG_SOURCE`` alone cannot express a merge, and
#: "your project config" is not a useful answer when the project only set one
#: key and inherited the provider from your global file.
CONFIG_SOURCE_LAYERS: list[tuple[str, Path]] = []


def _merge(base: dict, override: dict) -> dict:
    """Overlay ``override`` on ``base``, one key at a time.

    An override value of ``None`` is skipped rather than assigned. YAML reads a
    bare ``llm:`` (a section someone emptied out, or commented the contents of)
    as ``None``, and assigning that would replace the whole section - discarding
    the defaults and the user's global settings along with it, then failing
    later somewhere unrelated with a NoneType error. "No value here" has to mean
    "no opinion here", or the layering is not safe to hand-edit.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged.get(key), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_layer(path: Path) -> dict | None:
    """Parse one config file, or return None when it is not usable.

    A file that does not exist contributes nothing, which is what lets a project
    inherit from global instead of shadowing it.
    """
    if not path.is_file():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to load configuration {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration {path} must contain a YAML mapping")
    return loaded


def load_config() -> dict:
    """Layer every config file, lowest precedence first.

    Order is built-in defaults, then the packaged config, then the user's
    global ``~/.terminus/config.yaml``, then this project's ``config.yaml``.
    A later file overrides only the keys it actually mentions, so setting one
    value in a project no longer silently discards the rest of the user's
    global setup - which is what "project overrides global" has to mean for the
    hierarchy to be coherent.
    """
    global CONFIG_SOURCE, CONFIG_SOURCE_KIND, CONFIG_SOURCE_LAYERS
    candidates: list[tuple[Path, str]] = [
        (Path(__file__).parent / "config.yaml", "package"),
        (_global_config_candidate(), "global"),
        (Path.cwd() / "config.yaml", "cwd"),
    ]
    merged = deepcopy(DEFAULT_CONFIG)
    layers: list[tuple[str, Path]] = []
    for path, kind in candidates:
        loaded = _read_layer(path)
        if loaded is None:
            continue
        if loaded:
            merged = _merge(merged, loaded)
        layers.append((kind, path.resolve()))
    CONFIG_SOURCE_LAYERS = layers
    if layers:
        # The highest-precedence layer that said anything is the useful
        # single-file answer, and matches what CONFIG_SOURCE always meant.
        CONFIG_SOURCE_KIND, CONFIG_SOURCE = layers[-1]
    else:
        CONFIG_SOURCE = None
        CONFIG_SOURCE_KIND = "default"
    # Written through the module-level dict rather than only returned, so there is
    # exactly one live configuration in the process. Returning a fresh copy while
    # CONFIG kept the previous one let two readers disagree: a command that
    # re-loaded could report one provider while the code that resolved the
    # provider acted on another. A dozen modules hold a reference to CONFIG, so
    # the update is in place.
    CONFIG.clear()
    CONFIG.update(merged)
    return CONFIG


def _global_config_candidate() -> Path:
    """``~/.terminus/config.yaml``, resolved late so tests can move the home dir."""
    from terminus.user_config import global_config_path

    return global_config_path()


CONFIG = load_config()
