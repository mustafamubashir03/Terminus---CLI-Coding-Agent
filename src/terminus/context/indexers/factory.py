"""Choosing and building the index for the configured vector store.

The decision this module makes is *which backend the user asked for*, and it
reports that decision rather than changing it.

It used to do the opposite. When the configured Qdrant failed, it rewrote
``CONFIG["vector_store"]["provider"]`` to ``chromadb``, forced ``rag.mode`` and
``retrieval_mode`` down to ``semantic``, and returned a Chroma collection. The
consequences were that a user who configured ``qdrant`` silently got Chroma, that
hybrid/BM25 retrieval silently disappeared, and that because the substitution
mutated process-global config, every later call - including ``get_retriever()``
on each ``search_codebase`` - kept using the substitute. The swap outlived the
failure that caused it.

So: the configured backend is the backend. If it cannot be used, this raises
:class:`VectorStoreUnavailableError` with the reason. A fallback exists for
backwards compatibility, but it is opt-in (``vector_store.fallback_to_chroma``),
it does not touch ``CONFIG``, and when it engages the caller receives a
:class:`ResolvedBackend` saying so - because the substitution is a decision
somebody has to make knowingly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from terminus.config import CONFIG, CONFIG_SOURCE
from terminus.context.indexers.errors import (
    IndexerConfigurationError,
    VectorStoreFailure,
    VectorStoreUnavailableError,
    classify_failure,
    configuration_error,
    qdrant_error_message,
)
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

QDRANT = "qdrant"
CHROMA_ALIASES = frozenset({"chromadb", "chroma"})

#: Failures meaning "not usable right now", as opposed to a mistake in the
#: request. Only these may engage an explicitly enabled fallback: a mistyped
#: collection or a rejected credential must surface, because answering it by
#: using a different database would hide the thing the operator needs to fix.
FALLBACK_PERMITTED = frozenset({
    VectorStoreFailure.DNS,
    VectorStoreFailure.CONNECTION_REFUSED,
    VectorStoreFailure.TLS,
    VectorStoreFailure.TIMEOUT,
    VectorStoreFailure.UNAVAILABLE,
    VectorStoreFailure.LOCAL_STORE,
})


def _config_source() -> str | None:
    return str(CONFIG_SOURCE) if CONFIG_SOURCE else None


def _chroma_alias(provider: str) -> bool:
    return provider in CHROMA_ALIASES


@dataclass(frozen=True)
class ResolvedBackend:
    """Which backend is actually in use, and how it differs from configuration.

    ``configured_*`` is what the config says; ``resolved_*`` is what was built.
    They differ only when an explicit fallback engaged. Carrying the difference
    here, rather than writing it into ``CONFIG``, is what makes the substitution
    visible for the life of the process without outliving it.

    ``mode`` is reported honestly. A Chroma fallback from a hybrid configuration
    resolves to ``semantic``, because Chroma has no BM25 vector here, and
    reporting hybrid would be a lie the model then reasons from.
    """

    provider: str
    mode: str
    location: str
    configured_provider: str
    configured_mode: str
    fallback: bool = False
    reason: str = ""

    def describe(self) -> str:
        line = (
            f"vector store: provider={self.provider} mode={self.mode} "
            f"location={self.location}"
        )
        if self.fallback:
            line += (
                f"\n  FALLBACK ACTIVE: configured provider={self.configured_provider} "
                f"mode={self.configured_mode} -> resolved provider={self.provider} "
                f"mode={self.mode}"
                f"\n  reason: {self.reason}"
                f"\n  Note: Chroma does not provide the configured hybrid/BM25 "
                f"retrieval. Results come from dense semantic search only."
            )
        return line

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "mode": self.mode,
            "location": self.location,
            "configured_provider": self.configured_provider,
            "configured_mode": self.configured_mode,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def configured() -> tuple[str, str]:
    """``(provider, mode)`` as written in the configuration."""
    return (
        str(CONFIG.get("vector_store", {}).get("provider", "")).lower(),
        str(CONFIG.get("rag", {}).get("mode", "")).lower(),
    )


def location_of(provider: str) -> str:
    """Where *provider* points, without contacting it."""
    if _chroma_alias(provider):
        from terminus.context.indexers.semantic_chroma import chroma_persist_path

        return f"local:{chroma_persist_path(Path.cwd())}"
    from terminus.context.indexers.qdrant_client import qdrant_location

    return qdrant_location()


def resolved_backend(
    provider: str,
    mode: str,
    *,
    configured_provider: str | None = None,
    configured_mode: str | None = None,
    fallback: bool = False,
    reason: str = "",
) -> ResolvedBackend:
    """The descriptor for a resolved provider/mode pair.

    ``configured_*`` defaults to the resolved pair, which is right when nothing
    was substituted. A fallback passes the original values explicitly, so the
    descriptor can say "you asked for qdrant/hybrid, you are getting
    chroma/semantic" rather than quietly reporting the substitute as the choice.
    """
    return ResolvedBackend(
        provider=provider,
        mode=mode,
        location=location_of(provider),
        configured_provider=configured_provider or provider,
        configured_mode=configured_mode or mode,
        fallback=fallback,
        reason=reason,
    )


def validate(provider: str, mode: str) -> None:
    """Reject a configuration that cannot work, naming the config source.

    Runs before any network or disk access, so a typo is a fast, clear error
    rather than a confusing failure from inside a storage client.
    """
    if not _chroma_alias(provider) and provider != QDRANT:
        raise configuration_error(
            f"Unknown vector_store.provider: {provider!r} (expected 'qdrant' or 'chromadb')",
            _config_source(),
        )
    if mode not in {"semantic", "hybrid"}:
        raise configuration_error(f"Unknown rag.mode: {mode!r}", _config_source())
    if mode == "hybrid" and not provider == QDRANT:
        raise configuration_error(
            "rag.mode 'hybrid' is only supported for qdrant; chromadb has no "
            "sparse/BM25 vector in this project. Use rag.mode: semantic, or "
            "switch vector_store.provider to qdrant.",
            _config_source(),
        )
    if provider != QDRANT:
        return
    from terminus.context.indexers.qdrant_client import collection_name, qdrant_mode

    collection_name()
    if qdrant_mode() != "cloud":
        return
    endpoint = os.getenv("CLUSTER_ENDPOINT", "")
    if endpoint and urlparse(endpoint).scheme not in {"http", "https"}:
        raise configuration_error("CLUSTER_ENDPOINT must be an http(s) URL", _config_source())
    if mode == "hybrid":
        retrieval = str(CONFIG.get("vector_store", {}).get("retrieval_mode", "hybrid")).lower()
        if retrieval not in {"dense", "sparse", "hybrid"}:
            raise configuration_error(
                f"Unsupported vector_store.retrieval_mode: {retrieval!r}", _config_source()
            )


def build_index(provider: str, mode: str, repo_path: str, force_reindex: bool):
    """Construct the index for an already-validated provider/mode pair."""
    if _chroma_alias(provider):
        from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index

        return get_or_create_chroma_index(repo_path, force_reindex=force_reindex)
    if mode == "semantic":
        from terminus.context.indexers.semantic_qdrant import get_or_create_qdrant_index

        return get_or_create_qdrant_index(repo_path, force_reindex=force_reindex)
    from terminus.context.indexers.hybrid_qdrant import get_or_create_qdrant_hybrid_index

    return get_or_create_qdrant_hybrid_index(repo_path, force_reindex=force_reindex)


def fallback_enabled() -> bool:
    return bool(CONFIG.get("vector_store", {}).get("fallback_to_chroma", False))


def get_or_create_index(repo_path: str, force_reindex: bool = False):
    """``(index, ResolvedBackend)`` for the configured store.

    Raises :class:`VectorStoreUnavailableError` when the configured backend
    cannot be used and no fallback is permitted. ``CONFIG`` is never modified,
    here or anywhere below.
    """
    provider, mode = configured()
    validate(provider, mode)
    resolved = resolved_backend(provider, mode)
    logger.info("Indexing %s -> %s", repo_path, resolved.describe())

    try:
        return build_index(provider, mode, str(repo_path), force_reindex), resolved
    except Exception as exc:
        if provider != QDRANT or not fallback_enabled():
            raise _unavailable(exc, repo_path, provider, mode) from exc
        failure = classify_failure(exc)
        if failure not in FALLBACK_PERMITTED:
            raise _unavailable(exc, repo_path, provider, mode) from exc
        return _fall_back_to_chroma(exc, repo_path, provider, mode)


def _fall_back_to_chroma(exc: Exception, repo_path: str, provider: str, mode: str):
    """Chroma for a reachability failure, with the substitution reported.

    Loud on purpose. The previous version of this path logged a warning and
    rewrote the global config, so the substitution was invisible to the user and
    permanent for the process.
    """
    reason = qdrant_error_message(
        exc, repo_path, provider, mode,
        CONFIG.get("qdrant", {}).get("collection_name"), _config_source(),
    ).splitlines()[-1].strip()
    logger.error(
        "vector_store.fallback_to_chroma is enabled and %s/%s is unreachable (%s). "
        "Falling back to repository-local Chroma semantic search. This is a "
        "SUBSTITUTION, not the configured backend, and hybrid/BM25 retrieval is "
        "not available on this path.", provider, mode, reason,
    )
    try:
        index = build_index("chromadb", "semantic", repo_path, False)
    except Exception as chroma_exc:
        raise VectorStoreUnavailableError(
            f"vector_store.fallback_to_chroma is enabled, but Chroma also failed "
            f"({type(chroma_exc).__name__}: {chroma_exc}). Neither backend is usable."
        ) from chroma_exc
    return index, resolved_backend(
        "chromadb",
        "semantic",
        configured_provider=provider,
        configured_mode=mode,
        fallback=True,
        reason=reason,
    )


def _unavailable(exc: Exception, repo_path: str, provider: str, mode: str) -> VectorStoreUnavailableError:
    error = VectorStoreUnavailableError(
        qdrant_error_message(
            exc, repo_path, provider, mode,
            CONFIG.get("qdrant", {}).get("collection_name"), _config_source(),
        )
    )
    error.__cause__ = exc
    return error


#: ``(repo_path, index, ResolvedBackend)`` for the first index this process built.
_resolved_index: tuple[str, Any, ResolvedBackend] | None = None


def resolve_index(repo_path: str, force_reindex: bool = False):
    """``(index, ResolvedBackend)``, built on first use and reused after that.

    Opening a session must not wait on a vector store. A remote backend costs a
    network round trip and a local one costs a lock and a manifest read, and
    neither is needed to accept the first question - only to answer one that
    searches the codebase. The retrieval path builds its own store when a search
    actually happens, so resolving here is purely for callers that want to
    *report* on the index.

    The result is memoised per repository so repeated calls - the REPL resolving
    it, then a command displaying it - do not reconnect each time.
    """
    global _resolved_index
    if _resolved_index is not None and _resolved_index[0] == repo_path:
        return _resolved_index[1], _resolved_index[2]
    index, resolved = get_or_create_index(repo_path, force_reindex)
    _resolved_index = (repo_path, index, resolved)
    return index, resolved


def show_index(index) -> None:
    """Print a summary of *index*, dispatching on the configured backend."""
    provider, mode = configured()
    validate(provider, mode)
    if _chroma_alias(provider):
        from terminus.context.indexers.semantic_chroma import show_chroma_semantic_index

        return show_chroma_semantic_index(index)
    if mode == "semantic":
        from terminus.context.indexers.semantic_qdrant import show_qdrant_semantic_index

        return show_qdrant_semantic_index(index)
    from terminus.context.indexers.hybrid_qdrant import show_qdrant_hybrid_index

    return show_qdrant_hybrid_index(index)


__all__ = [
    "CHROMA_ALIASES",
    "FALLBACK_PERMITTED",
    "IndexerConfigurationError",
    "QDRANT",
    "ResolvedBackend",
    "VectorStoreUnavailableError",
    "build_index",
    "configured",
    "fallback_enabled",
    "get_or_create_index",
    "location_of",
    "resolve_index",
    "resolved_backend",
    "show_index",
    "validate",
]
