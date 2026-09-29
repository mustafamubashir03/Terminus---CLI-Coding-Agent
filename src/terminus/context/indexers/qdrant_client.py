"""Constructing a Qdrant client, and deciding where it points.

The one place that knows what a Qdrant client needs, so the indexers, the
retrievers and the reindexer stop disagreeing about it. They previously did:
the indexers passed a timeout and disabled version checks, the reindexer passed
neither, and the retrievers built their own through LangChain's classmethods.

Scope is deliberately narrow. This is a Qdrant helper, not a storage
abstraction: it does not know Chroma exists, and there is no backend interface
to implement. Local and cloud differ only in which arguments a ``QdrantClient``
is constructed with, so they are one function with a branch, not two classes.

Local mode
----------
``QdrantClient(path=...)`` runs an embedded Qdrant engine in-process against a
directory, with no server, no port and no credentials. That is what makes a
fresh install work with nothing configured.

Cloud mode
----------
``QdrantClient(url=..., api_key=...)`` is the pre-existing behaviour, unchanged.
Credentials stay in the environment; nothing here writes or reads a secret into
configuration.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from terminus.config import CONFIG
from terminus.env import load_project_env
from terminus.context.indexers.errors import (
    IndexerConfigurationError,
    configuration_error,
)
from terminus.observability.logging import get_logger
from terminus.workspace import project_root

logger = get_logger(__name__)

QDRANT_API_KEY_ENV = "QDRANT_API_KEY"
CLUSTER_ENDPOINT_ENV = "CLUSTER_ENDPOINT"

DEFAULT_LOCAL_PATH = ".terminus/qdrant"
DEFAULT_TIMEOUT_SECONDS = 5.0

LOCAL = "local"
CLOUD = "cloud"


def _config_source() -> str | None:
    from terminus.config import CONFIG_SOURCE

    return str(CONFIG_SOURCE) if CONFIG_SOURCE else None


def qdrant_mode() -> str:
    """Where this store lives: ``local`` or ``cloud``.

    Backwards-compatible resolution, in order:

    1. ``qdrant.mode`` when set - the explicit choice.
    2. ``cloud`` when ``CLUSTER_ENDPOINT`` is present. This is what keeps every
       existing Qdrant Cloud user on cloud: they configured an endpoint in the
       environment, and nothing about their setup says they want a local index.
    3. ``local`` otherwise, so a fresh install needs no credentials at all.

    Rule 2 is the important one. Defaulting to local when an endpoint happens to
    be set would silently strand a cloud user's data; defaulting to cloud when it
    is not set is the failure mode we are removing.
    """
    configured = str(CONFIG.get("qdrant", {}).get("mode", "") or "").strip().lower()
    if configured in {LOCAL, CLOUD}:
        return configured
    if os.getenv(CLUSTER_ENDPOINT_ENV):
        return CLOUD
    return LOCAL


def qdrant_local_path() -> Path:
    """The directory backing a local store, resolved against the project root.

    Resolution matters: ``qdrant.path`` is configured relative, and a relative
    path re-interprets against whatever the cwd is at call time, which is how
    project A's index ends up being read as project B's.
    """
    configured = Path(CONFIG.get("qdrant", {}).get("path", DEFAULT_LOCAL_PATH)).expanduser()
    if configured.is_absolute():
        return configured
    return project_root() / configured


def qdrant_location() -> str:
    """A short, log-safe description of where the store is.

    Never includes credentials, and for cloud only the host - enough to tell two
    clusters apart in a log without publishing a token or a full signed URL.
    """
    if qdrant_mode() == LOCAL:
        return f"local:{qdrant_local_path()}"
    load_project_env()
    endpoint = os.getenv(CLUSTER_ENDPOINT_ENV, "")
    return f"cloud:{endpoint or 'no-endpoint-configured'}"


def _timeout() -> float:
    return float(CONFIG.get("qdrant", {}).get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))


#: Local clients, keyed by resolved store path.
#:
#: A local Qdrant engine takes an exclusive single-writer lock on its storage
#: directory. Two clients on the same path in one process therefore fail with
#: ``AlreadyLocked`` - and that is not hypothetical: the indexer builds a client,
#: and the retriever builds one, in the same session. So the local client is
#: created once per directory and shared. Cloud clients are not cached: they hold
#: no on-disk lock, and pooling is the SDK's business rather than ours.
_local_clients: dict[str, Any] = {}
_local_lock = threading.Lock()


def _cached_local_client(path: Path):
    key = str(path)
    with _local_lock:
        client = _local_clients.get(key)
        if client is None:
            client = _create_local_client(key)
            _local_clients[key] = client
        return client


def _create_local_client(path: str):
    from qdrant_client import QdrantClient

    return QdrantClient(path=path, **_client_options())


def _client_options() -> dict[str, Any]:
    """Options every client gets, whoever constructs it."""
    return {
        "timeout": _timeout(),
        # Skips the client's startup version check, a network round trip before
        # the first query. Already being skipped deliberately; now consistent.
        "check_compatibility": False,
    }


def reset_local_clients() -> None:
    """Close and forget cached local clients. For tests and after a reconfigure."""
    with _local_lock:
        clients = list(_local_clients.values())
        _local_clients.clear()
    for client in clients:
        try:
            client.close()
        except Exception:  # pragma: no cover - close is best-effort
            pass


def cloud_settings() -> tuple[str, str]:
    """``(url, api_key)`` for a cloud store, or raise explaining what is missing.

    The project ``.env`` is loaded first, for the same reason the LLM factory
    does it: ``QDRANT_API_KEY`` and ``CLUSTER_ENDPOINT`` normally live there, and
    without loading it a correctly configured project would report "not set".
    Loading is idempotent and never overrides an already-set variable.

    Raises rather than returning a partial result: constructing a client with an
    empty URL produces a far more confusing failure several layers down.
    """
    load_project_env()

    url = (os.getenv(CLUSTER_ENDPOINT_ENV) or "").strip()
    if not url:
        raise IndexerConfigurationError(
            f"{CLUSTER_ENDPOINT_ENV} is not set, which qdrant.mode: cloud requires. "
            "Set it in your .env, or set qdrant.mode: local for a "
            "credential-free index. (Configuration: " + (_config_source() or "built-in defaults") + ")"
        )
    if not url.startswith(("http://", "https://")):
        raise configuration_error(
            f"{CLUSTER_ENDPOINT_ENV} must be an http(s) URL, got {url!r}",
            _config_source(),
        )
    api_key = (os.getenv(QDRANT_API_KEY_ENV) or "").strip()
    if not api_key:
        raise IndexerConfigurationError(
            f"{QDRANT_API_KEY_ENV} is not set, which qdrant.mode: cloud requires. "
            "Set it in your .env, or set qdrant.mode: local for a "
            "credential-free index. (Configuration: " + (_config_source() or "built-in defaults") + ")"
        )
    return url, api_key


def create_qdrant_client(**overrides: Any):
    """A ``QdrantClient`` for the configured location.

    ``overrides`` exist for the reindexer, which historically built its client
    without a timeout; they are accepted so its behaviour converges on the same
    defaults rather than staying divergent.

    A local client is cached per store directory - see :data:`_local_clients` for
    why. A cloud client is built fresh; it holds no on-disk lock.
    """
    from qdrant_client import QdrantClient

    if qdrant_mode() == LOCAL:
        path = qdrant_local_path()
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Qdrant: local store at %s", path)
        if not overrides:
            return _cached_local_client(path)
        return QdrantClient(path=str(path), **{**_client_options(), **overrides})

    url, api_key = cloud_settings()
    logger.info("Qdrant: cloud store at %s", url)
    return QdrantClient(
        url=url, api_key=api_key, **{**_client_options(), **overrides}
    )


def collection_name() -> str:
    name = str(CONFIG.get("qdrant", {}).get("collection_name", "") or "").strip()
    if not name:
        raise configuration_error("Qdrant collection_name is not configured", _config_source())
    return name


def embedding_dimensions() -> int:
    """The vector size the configured embedding model produces.

    Read from the embedder rather than hard-coded, because a local store
    records the dimension in its schema: a collection created at 384 cannot
    accept 1024, and the mismatch surfaces as an opaque server error much later
    if it is not checked up front.
    """
    from terminus.llm.factory import get_embedder

    embedder = get_embedder()
    probe = "dimension probe"
    for attribute in ("dimension", "embedding_dimension", "vector_size"):
        value = getattr(embedder, attribute, None)
        if isinstance(value, int):
            return value
    return len(embedder.embed_query(probe))
