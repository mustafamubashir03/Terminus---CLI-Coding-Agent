"""Process-local cache for retrieval resources.

Building a Qdrant client per query meant a new connection pool per
``search_codebase`` call, and the hybrid retriever also reloaded a BM25 model
each time. Both are wasteful and neither varies within a project, so they are
cached here rather than per module - which also keeps the two Qdrant retrievers
from each reaching into the other's private cache.

What the key must contain
-------------------------
Everything that changes which store answers the query:

    project   isolation boundary - two projects must never share a store
    location  local vs cloud changes the client the store wraps
    collection the collection name
    kind      dense vs hybrid - they differ in the sparse retriever they hold

``project`` is first and load-bearing. Omitting it is how a process that moved
between repositories would serve the first one's chunks to the second, which is
the same class of leak the payload filter exists to prevent.

Bounded by construction: one entry per (project, location, collection, kind), so
the worst case is a handful of clients, and stale entries for abandoned projects
are dropped by :func:`reset`.
"""

from __future__ import annotations

import threading
from typing import Any, Callable


_lock = threading.Lock()
_stores: dict[tuple, Any] = {}
_sparse: dict[tuple, Any] = {}


def project_key() -> str:
    from terminus.workspace import project_key as key

    return key()


def store_key(kind: str) -> tuple:
    from terminus.context.indexers.qdrant_client import collection_name, qdrant_location

    return (project_key(), qdrant_location(), collection_name(), kind)


def cached_store(kind: str, factory: Callable[[], Any]) -> Any:
    """The cached store for this project, building it on first use."""
    key = store_key(kind)
    with _lock:
        existing = _stores.get(key)
        if existing is not None:
            return existing
    built = factory()
    with _lock:
        # Re-check: another thread may have built it while we were outside the
        # lock. Keep whichever landed first so a project only ever has one store.
        return _stores.setdefault(key, built)


def cached_sparse(model_name: str, factory: Callable[[], Any]) -> Any:
    """The cached sparse embedder for this project and model."""
    key = (project_key(), model_name)
    with _lock:
        existing = _sparse.get(key)
        if existing is not None:
            return existing
    built = factory()
    with _lock:
        return _sparse.setdefault(key, built)


def reset() -> None:
    """Drop every cached resource.

    For tests and after an intentional reconfiguration - a backend switch, a
    collection rebuild. Not called on the normal path: a stale entry is bounded
    and keyed, so the cost of leaving it is one client per project.
    """
    with _lock:
        _stores.clear()
        _sparse.clear()
