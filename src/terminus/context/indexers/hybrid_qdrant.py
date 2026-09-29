"""Hybrid (dense + BM25 sparse) index over the Qdrant store.

Hybrid mode is Qdrant-only, and that is a property of the retriever rather than
an accident: BM25 sparse vectors are a Qdrant feature here, and Chroma's
collection has no equivalent. The indexer factory rejects hybrid for any other
provider so the combination cannot be requested by configuration.

Everything else mirrors the dense indexer - same client helper, same
incremental path, same legacy-point reporting - so the two differ only in the
sparse retriever they pass in.
"""

from __future__ import annotations

from langchain_core.documents import Document

from terminus.context.indexers.code_parser import get_source_files, parse_file
from terminus.context.indexers import qdrant_client
from terminus.context.indexers.qdrant_client import collection_name, qdrant_location
from terminus.context.indexers.qdrant_store import write_documents
from terminus.context.qdrant_scope import (
    chunk_metadata,
    ensure_project_payload_index,
    ensure_source_payload_index,
    unscoped_points_present,
)
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

SPARSE_MODEL = "Qdrant/bm25"


def _configured_mode() -> str:
    from terminus.config import CONFIG

    return str(CONFIG.get("vector_store", {}).get("retrieval_mode", "hybrid")).lower()


def retrieval_mode():
    """The configured dense/sparse/hybrid mix, as LangChain's enum.

    An unrecognised value resolves to HYBRID rather than raising: the
    configuration is validated in the indexer factory, so by the time this runs
    the value is already known-good, and a default here keeps the retrievers
    usable without re-validating.
    """
    from langchain_qdrant import RetrievalMode

    return {
        "dense": RetrievalMode.DENSE,
        "sparse": RetrievalMode.SPARSE,
        "hybrid": RetrievalMode.HYBRID,
    }.get(_configured_mode(), RetrievalMode.HYBRID)


def sparse_retriever():
    """The BM25 sparse embedder.

    Loading it is not free, so callers that search repeatedly should hold onto
    the result; the retrievers do.
    """
    from langchain_qdrant import FastEmbedSparse

    return FastEmbedSparse(model_name=SPARSE_MODEL)


def get_or_create_qdrant_hybrid_index(repo_path: str, *, force_reindex: bool = False):
    """Index the codebase for hybrid dense+sparse search."""
    name = collection_name()
    client = qdrant_client.create_qdrant_client()
    logger.info(
        "Qdrant hybrid index: collection=%s location=%s mode=%s",
        name, qdrant_location(), retrieval_mode(),
    )

    existing = {c.name for c in client.get_collections().collections}
    if name in existing and (client.get_collection(collection_name=name).points_count or 0) > 0:
        ensure_project_payload_index(client, name)
        ensure_source_payload_index(client, name)
        if unscoped_points_present(client, name):
            from terminus.context.indexers.semantic_qdrant import _warn_unscoped

            _warn_unscoped(name)
        if force_reindex:
            from terminus.context.indexers.reindexer import full_reindex

            logger.info("Force reindex requested - wiping and rebuilding")
            return full_reindex(repo_path)[0]
        from terminus.context.indexers.reindexer import incremental_reindex

        store, result = incremental_reindex(repo_path)
        if result.files_added or result.files_modified or result.files_deleted:
            logger.info("Incremental reindex: %s", result)
        return store

    logger.info("Loading codebase from: %s", repo_path)
    files = get_source_files(repo_path)
    documents: list[Document] = []
    for filepath in files:
        try:
            chunks = parse_file(filepath)
        except (SyntaxError, ValueError) as exc:
            logger.warning("Skipping %s: %s: %s", filepath, "parse", exc)
            continue
        for chunk in chunks:
            documents.append(Document(page_content=chunk.content, metadata=chunk_metadata(chunk)))

    store = write_documents(
        client, documents, name, sparse_embedding=sparse_retriever()
    )
    from terminus.context.indexers.freshness import Manifest, _now_iso

    manifest = Manifest(repo_path=repo_path, last_full_index=_now_iso())
    manifest.files = Manifest.snapshot_directory(repo_path)
    manifest.save()
    logger.info("Hybrid indexing completed. Indexed %d files into %s", len(files), name)
    return store


def show_qdrant_hybrid_index(store) -> None:
    """Print the first chunks in the hybrid index."""
    from terminus.context.indexers.semantic_qdrant import show_qdrant_semantic_index

    show_qdrant_semantic_index(store)
