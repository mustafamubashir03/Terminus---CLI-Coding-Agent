from __future__ import annotations

import os
from urllib.parse import urlparse

from terminus.config import CONFIG, CONFIG_SOURCE
from terminus.context.indexers.errors import (
    VectorStoreUnavailableError,
    configuration_error,
    is_transport_error,
    qdrant_error_message,
)
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def _config_source() -> str | None:
    return str(CONFIG_SOURCE) if CONFIG_SOURCE else None


def _validate_indexer_config(provider: str, mode: str) -> None:
    if provider not in {"chromadb", "chroma", "qdrant"}:
        raise configuration_error(f"Unknown indexer provider: {provider}", _config_source())
    if mode not in {"semantic", "hybrid"}:
        raise configuration_error(f"Unknown RAG mode: {mode}", _config_source())
    if mode == "hybrid" and provider != "qdrant":
        raise configuration_error(
            "Hybrid mode is only supported for qdrant", _config_source()
        )
    if provider != "qdrant":
        return
    collection = CONFIG.get("qdrant", {}).get("collection_name")
    if not collection:
        raise configuration_error(
            "Qdrant collection_name is not configured", _config_source()
        )
    endpoint = os.getenv("CLUSTER_ENDPOINT", "")
    if endpoint:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise configuration_error(
                "CLUSTER_ENDPOINT must be an HTTP(S) URL", _config_source()
            )
    if mode == "hybrid":
        retrieval_mode = CONFIG.get("vector_store", {}).get(
            "retrieval_mode", "hybrid"
        )
        if retrieval_mode not in {"dense", "sparse", "hybrid"}:
            raise configuration_error(
                f"Unsupported Qdrant retrieval mode: {retrieval_mode}",
                _config_source(),
            )


def _create_indexer(provider: str, mode: str, repo_path: str, force_reindex: bool):
    if mode == "semantic":
        if provider in {"chromadb", "chroma"}:
            from terminus.context.indexers.semantic_chroma import (
                get_or_create_chroma_index,
            )

            return get_or_create_chroma_index(
                repo_path, force_reindex=force_reindex
            )
        from terminus.context.indexers.semantic_qdrant import (
            get_or_create_qdrant_index,
        )

        return get_or_create_qdrant_index(repo_path, force_reindex=force_reindex)
    from terminus.context.indexers.hybrid_qdrant import (
        get_or_create_qdrant_hybrid_index,
    )

    return get_or_create_qdrant_hybrid_index(
        repo_path, force_reindex=force_reindex
    )


def _fallback_enabled() -> bool:
    return bool(CONFIG.get("vector_store", {}).get("fallback_to_chroma", True))


def _is_qdrant_unavailable(exc: Exception) -> bool:
    if is_transport_error(exc):
        return True
    message = str(exc)
    return "QDRANT_API_KEY" in message or "CLUSTER_ENDPOINT" in message


def _activate_chroma_fallback(
    repo_path: str, provider: str, mode: str, error: VectorStoreUnavailableError
):
    if not _fallback_enabled():
        raise error
    collection = CONFIG.get("qdrant", {}).get("collection_name")
    logger.warning(
        "%s Falling back to repository-local Chroma semantic indexing.",
        qdrant_error_message(
            error.__cause__ or error,
            repo_path,
            provider,
            mode,
            collection,
            _config_source(),
        ),
    )
    CONFIG.setdefault("vector_store", {})["provider"] = "chromadb"
    CONFIG["vector_store"]["retrieval_mode"] = "semantic"
    CONFIG.setdefault("rag", {})["mode"] = "semantic"
    CONFIG.setdefault("_runtime", {})["indexer_fallback"] = {
        "from_provider": provider,
        "from_mode": mode,
        "to_provider": "chromadb",
        "to_mode": "semantic",
    }
    from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index

    return get_or_create_chroma_index(repo_path)


def get_or_create_indexer(repo_path: str, force_reindex: bool = False):
    provider = str(CONFIG.get("vector_store", {}).get("provider", "")).lower()
    mode = str(CONFIG.get("rag", {}).get("mode", "")).lower()
    _validate_indexer_config(provider, mode)
    try:
        return _create_indexer(provider, mode, str(repo_path), force_reindex)
    except Exception as exc:
        if provider != "qdrant":
            raise
        error = VectorStoreUnavailableError(
            qdrant_error_message(
                exc,
                str(repo_path),
                provider,
                mode,
                CONFIG.get("qdrant", {}).get("collection_name"),
                _config_source(),
            )
        )
        error.__cause__ = exc
        if not _is_qdrant_unavailable(exc):
            raise error from exc
        return _activate_chroma_fallback(str(repo_path), provider, mode, error)


def show_index(index):
    provider = str(CONFIG.get("vector_store", {}).get("provider", "")).lower()
    mode = str(CONFIG.get("rag", {}).get("mode", "")).lower()
    _validate_indexer_config(provider, mode)
    if mode == "semantic":
        if provider == "qdrant":
            from terminus.context.indexers.semantic_qdrant import (
                show_qdrant_semantic_index,
            )

            return show_qdrant_semantic_index(index)
        from terminus.context.indexers.semantic_chroma import (
            show_chroma_semantic_index,
        )

        return show_chroma_semantic_index(index)
    from terminus.context.indexers.hybrid_qdrant import show_qdrant_hybrid_index

    return show_qdrant_hybrid_index(index)
