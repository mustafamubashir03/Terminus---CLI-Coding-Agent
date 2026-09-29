"""Incremental and full reindexing engine.

Orchestrates the freshness-diff → delete stale → re-chunk → re-embed flow.
Supports Qdrant (semantic / hybrid) and ChromaDB (semantic) backends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from terminus.config import CONFIG
from terminus.context.indexers.code_parser import parse_file
from terminus.context.indexers.freshness import (
    DiffResult,
    FileSnapshot,
    Manifest,
    _now_iso,
)
from terminus.observability.logging import get_logger
from terminus.context.qdrant_scope import chunk_metadata

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReindexResult:
    """Statistics returned by an incremental or full reindex run."""
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"+{self.files_added} new, ~{self.files_modified} modified, "
            f"-{self.files_deleted} deleted, "
            f"{self.chunks_added} chunks added, {self.chunks_removed} removed "
            f"in {self.elapsed_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Document builder (shared by all providers)
# ---------------------------------------------------------------------------

def _parse_files_to_documents(filepaths: list[str]) -> tuple[list[Document], int]:
    """Parse *filepaths* into langchain ``Document`` objects.

    Returns ``(documents, chunk_count)``.
    """
    docs: list[Document] = []
    skipped = 0
    for filepath in filepaths:
        try:
            chunks = parse_file(filepath)
        except (SyntaxError, ValueError, OSError) as exc:
            logger.warning(f"Skipping {filepath}: {exc}")
            skipped += 1
            continue
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk.content,
                    # Scoped by the shared helper, so an incremental reindex
                    # tags points exactly like a full index. Omitting this
                    # would quietly produce unscoped points.
                    metadata=chunk_metadata(chunk),
                )
            )
    if skipped:
        logger.debug(f"Skipped {skipped} unparseable files")
    return docs, len(docs)


# ---------------------------------------------------------------------------
# Provider helpers: Qdrant
# ---------------------------------------------------------------------------

def _qdrant_client_and_cfg():
    """Return ``(QdrantClient, collection_name)``.

    Delegates to the shared client helper, so the reindexer stops being the one
    place that built a Qdrant client without a timeout and with the version
    check left on. ``api_key``/``endpoint`` are no longer returned: the upsert
    helper needs a client, not credentials, and returning them is what kept this
    function pinned to cloud.
    """
    from terminus.context.indexers import qdrant_client
    from terminus.context.indexers.qdrant_client import collection_name

    return qdrant_client.create_qdrant_client(), collection_name()


def _qdrant_delete_points(client: Any, collection: str, filepaths: list[str]) -> int:
    """Delete all points whose ``metadata.source`` matches any path.

    Returns the total number of points removed (via ``count``).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    total_deleted = 0
    for fp in filepaths:
        try:
            # Count how many points exist for this source path
            count_before = client.count(
                collection_name=collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source", match=MatchValue(value=fp)
                        )
                    ]
                ),
                exact=True,
            )
            points_count = getattr(count_before, "count", 0) or 0

            client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source", match=MatchValue(value=fp)
                        )
                    ]
                ),
            )
            total_deleted += points_count
            if points_count:
                logger.debug(f"Deleted {points_count} points for {fp}")
        except Exception as exc:
            logger.warning(f"Failed to delete points for {fp}: {exc}")
    return total_deleted


def _ensure_source_payload_index(client: Any, collection: str) -> None:
    """Create a KEYWORD payload index on ``metadata.source`` if missing.

    Required for ``FieldCondition`` filter-based deletions to work in
    Qdrant.  Without it, filtered deletes raise ``Bad request`` and stale
    points are never purged.
    """
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name="metadata.source",
            field_schema="keyword",
        )
        logger.info("Created payload index on metadata.source")
    except Exception as exc:
        # "Bad request: index already exists" is the common case and is fine.
        # Any other error is logged but not fatal — the delete may still
        # succeed if Qdrant falls back to a full scan.
        err_str = str(exc).lower()
        if "already" not in err_str and "exist" not in err_str:
            logger.warning(f"Payload index creation warning: {exc}")
        else:
            logger.debug(f"Payload index already exists: {exc}")


def _qdrant_upsert_documents(
    docs: list[Document], embedder: Any, sparse: Any, retrieval_mode: Any,
    client: Any, collection: str, batch_size: int = 50,
) -> None:
    """Upsert *docs* into a Qdrant collection.

    Goes through the constructor plus ``add_documents`` rather than
    ``from_documents``, because the installed ``langchain-qdrant`` forwards a
    ``client=`` argument into its HTTP client and rejects it - which makes the
    classmethod unusable for a local store, and inconsistent with the indexers
    even for cloud.
    """
    if not docs:
        return
    from langchain_qdrant import QdrantVectorStore

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embedder,
        sparse_embedding=sparse,
        retrieval_mode=retrieval_mode,
    )
    store.add_documents(docs, batch_size=batch_size)


def _qdrant_wipe_collection(client: Any, collection: str) -> None:
    """Delete ALL points from a Qdrant collection (keeps schema)."""
    from qdrant_client.models import Filter

    try:
        client.delete(
            collection_name=collection,
            points_selector=Filter(must=[]),  # match all
        )
        logger.info(f"Wiped all points from {collection}")
    except Exception as exc:
        logger.warning(f"Wipe failed (may already be empty): {exc}")


# ---------------------------------------------------------------------------
# Provider helpers: ChromaDB
# ---------------------------------------------------------------------------

def _chroma_collection(repo_path: str | None = None):
    """Return ``(chromadb.Collection, chromadb_client)``."""
    import chromadb

    base = Path(repo_path or Path.cwd()).expanduser().resolve()
    persist_dir = Path(CONFIG["chromadb"]["persist_dir"]).expanduser()
    if not persist_dir.is_absolute():
        persist_dir = base / persist_dir
    coll_name = CONFIG["chromadb"]["collection_name"]
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=coll_name)
    return collection, client



def _chroma_delete_points(collection: Any, filepaths: list[str]) -> int:
    """Delete ChromaDB documents by source path."""
    total = 0
    for fp in filepaths:
        try:
            results = collection.get(where={"source": fp})
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                total += len(ids)
        except Exception as exc:
            logger.warning(f"Failed to delete ChromaDB points for {fp}: {exc}")
    return total


def _chroma_upsert_documents(
    docs: list[Document], embedder: Any, collection: Any, batch_size: int = 50,
) -> int:
    """Upsert *docs* into a ChromaDB collection.  Returns chunk count."""
    if not docs:
        return 0
    count = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        for doc in batch:
            m = doc.metadata
            doc_id = f"{m['source']}:{m['start_line']}-{m['end_line']}"
            ids.append(doc_id)
            embeddings.append(embedder.embed_query(doc.page_content))
            documents.append(doc.page_content)
            metadatas.append(m)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        count += len(batch)
    return count


def _chroma_wipe_collection(collection: Any) -> None:
    """Delete all documents from a ChromaDB collection."""
    try:
        results = collection.get()
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info(f"Wiped {len(ids)} points from ChromaDB collection")
    except Exception as exc:
        logger.warning(f"ChromaDB wipe failed: {exc}")


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def _get_provider() -> str:
    return CONFIG["vector_store"]["provider"]


def _get_mode() -> str:
    return CONFIG["rag"]["mode"]


# ---------------------------------------------------------------------------
# Core: incremental reindex
# ---------------------------------------------------------------------------

def incremental_reindex(repo_path: str) -> tuple[Any, ReindexResult]:
    """Diff-based reindex: only process files that changed since last index.

    Returns ``(vector_store_or_collection, ReindexResult)``.
    """
    t0 = time.time()
    result = ReindexResult()
    provider = _get_provider()

    # 1. Load previous manifest and snapshot current state (fast stat pass).
    #    Only files whose mtime/size changed get a SHA-256 hash inside
    #    ``compute_diff`` — untouched files never touch the disk content.
    manifest = Manifest.load(repo_path)

    # If there is no baseline manifest but the collection already has points,
    # the index content predates freshness tracking.  We cannot trust the
    # existing points to match the current files, so rebuild cleanly instead
    # of appending duplicates on top of a possibly-stale index.
    if not manifest.files:
        logger.info(
            "No manifest baseline found - falling back to a full rebuild "
            "(avoids duplicate points from a pre-freshness index)"
        )
        return full_reindex(repo_path)

    current = Manifest.snapshot_directory_fast(repo_path)

    # 2. Compute diff (hashes only stat-changed files)
    diff = manifest.compute_diff(current, repo_path)

    if not diff.has_changes:
        logger.info("Index is up to date — no changes detected")
        # Still need to return the existing vector store
        vs = _connect_existing(provider, repo_path)
        result.elapsed_seconds = time.time() - t0
        return vs, result

    # 3. Parse new + modified files into documents
    files_to_index = diff.added + diff.modified
    docs, chunk_count = _parse_files_to_documents(files_to_index)

    # 4. Delete points for deleted + modified files (modified = old points
    #    removed first, then new points added)
    files_to_delete = diff.deleted + diff.modified

    # 5. Apply changes to the provider
    if provider == "qdrant":
        _apply_qdrant_incremental(
            diff, docs, chunk_count, files_to_delete, result
        )
    elif provider in ("chromadb", "chroma"):
        _apply_chroma_incremental(
            repo_path, diff, docs, chunk_count, files_to_delete, result
        )
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")

    # 6. Merge manifest: keep full snapshots for unchanged files, use the
    #    freshly-hashed entries for added/modified files.  Deleted files
    #    are naturally dropped because they no longer exist in ``current``.
    updated: dict[str, FileSnapshot] = {}
    for path, snap in current.items():
        if path in diff.unchanged and path in manifest.files:
            updated[path] = manifest.files[path]
        else:
            updated[path] = snap
    manifest.files = updated
    manifest.last_full_index = _now_iso()
    manifest.save()

    result.files_added = len(diff.added)
    result.files_modified = len(diff.modified)
    result.files_deleted = len(diff.deleted)
    result.chunks_added = chunk_count
    result.elapsed_seconds = time.time() - t0

    logger.info(f"Incremental reindex complete: {result}")
    vs = _connect_existing(provider, repo_path)
    return vs, result


def _apply_qdrant_incremental(
    diff: DiffResult,
    docs: list[Document],
    chunk_count: int,
    files_to_delete: list[str],
    result: ReindexResult,
) -> None:
    """Apply incremental changes to a Qdrant collection."""
    from terminus.context.indexers.hybrid_qdrant import (
        get_retrieval_mode,
    )
    from terminus.llm.factory import get_embedder

    client, collection = _qdrant_client_and_cfg()
    embedder = get_embedder()

    # Ensure metadata.source is indexed so filtered deletions work
    _ensure_source_payload_index(client, collection)

    # Delete stale points
    deleted_count = _qdrant_delete_points(client, collection, files_to_delete)
    result.chunks_removed = deleted_count
    logger.info(f"Deleted {deleted_count} stale points from Qdrant")

    # Upsert new documents
    if docs:
        mode = _get_mode()
        if mode == "hybrid":
            from langchain_qdrant import FastEmbedSparse

            sparse = FastEmbedSparse(model_name="Qdrant/bm25")
            retrieval_mode = get_retrieval_mode()
            _qdrant_upsert_documents(
                docs, embedder, sparse, retrieval_mode,
                client, collection,
            )
        else:
            from langchain_qdrant import QdrantVectorStore

            # Constructor + add_documents, not from_documents: the installed
            # langchain-qdrant rejects an injected client in the classmethods.
            QdrantVectorStore(
                client=client,
                collection_name=collection,
                embedding=embedder,
            ).add_documents(docs, batch_size=50)
        logger.info(f"Upserted {chunk_count} new/updated chunks into Qdrant")


def _apply_chroma_incremental(
    repo_path: str,
    diff: DiffResult,
    docs: list[Document],
    chunk_count: int,
    files_to_delete: list[str],
    result: ReindexResult,
) -> None:
    """Apply incremental changes to a ChromaDB collection."""
    from terminus.llm.factory import get_embedder

    collection, _client = _chroma_collection(repo_path)
    embedder = get_embedder()

    # Delete stale points
    deleted_count = _chroma_delete_points(collection, files_to_delete)
    result.chunks_removed = deleted_count
    logger.info(f"Deleted {deleted_count} stale points from ChromaDB")

    # Upsert new documents
    added = _chroma_upsert_documents(docs, embedder, collection)
    logger.info(f"Upserted {added} new/updated chunks into ChromaDB")


# ---------------------------------------------------------------------------
# Core: full reindex
# ---------------------------------------------------------------------------

def full_reindex(repo_path: str) -> tuple[Any, ReindexResult]:
    """Wipe-and-rebuild: re-index every source file from scratch.

    Returns ``(vector_store_or_collection, ReindexResult)``.
    """
    t0 = time.time()
    result = ReindexResult()
    provider = _get_provider()

    # 1. Snapshot current directory
    current = Manifest.snapshot_directory(repo_path)

    # 2. Parse ALL files
    all_files = list(current.keys())
    docs, chunk_count = _parse_files_to_documents(all_files)

    # 3. Wipe existing collection and rebuild
    if provider == "qdrant":
        _apply_qdrant_full(docs, chunk_count, result)
    elif provider in ("chromadb", "chroma"):
        _apply_chroma_full(repo_path, docs, chunk_count, result)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")

    # 4. Create fresh manifest
    manifest = Manifest(
        repo_path=repo_path,
        files=current,
        last_full_index=_now_iso(),
    )
    manifest.save()

    result.files_added = len(all_files)
    result.files_modified = 0
    result.files_deleted = 0
    result.chunks_added = chunk_count
    result.elapsed_seconds = time.time() - t0

    logger.info(f"Full reindex complete: {result}")
    vs = _connect_existing(provider, repo_path)
    return vs, result


def _apply_qdrant_full(
    docs: list[Document], chunk_count: int, result: ReindexResult,
) -> None:
    """Full wipe-and-rebuild for Qdrant."""
    from terminus.context.indexers.hybrid_qdrant import get_retrieval_mode
    from terminus.llm.factory import get_embedder

    client, collection = _qdrant_client_and_cfg()
    embedder = get_embedder()

    # Ensure metadata.source is indexed so filtered deletions work
    _ensure_source_payload_index(client, collection)

    # Wipe
    _qdrant_wipe_collection(client, collection)

    # Rebuild
    mode = _get_mode()
    if mode == "hybrid":
        from langchain_qdrant import FastEmbedSparse

        sparse = FastEmbedSparse(model_name="Qdrant/bm25")
        retrieval_mode = get_retrieval_mode()
        _qdrant_upsert_documents(
            docs, embedder, sparse, retrieval_mode,
            client, collection,
        )
    else:
        from langchain_qdrant import QdrantVectorStore

        if docs:
            # Constructor + add_documents, not from_documents: the installed
            # langchain-qdrant rejects an injected client in the classmethods.
            QdrantVectorStore(
                client=client,
                collection_name=collection,
                embedding=embedder,
            ).add_documents(docs, batch_size=50)
    result.chunks_added = chunk_count
    logger.info(f"Full Qdrant rebuild: {chunk_count} chunks")


def _apply_chroma_full(
    repo_path: str,
    docs: list[Document],
    chunk_count: int,
    result: ReindexResult,
) -> None:
    """Full wipe-and-rebuild for ChromaDB."""
    from terminus.llm.factory import get_embedder

    collection, _client = _chroma_collection(repo_path)
    embedder = get_embedder()

    # Wipe
    _chroma_wipe_collection(collection)

    # Rebuild
    added = _chroma_upsert_documents(docs, embedder, collection)
    result.chunks_added = added
    logger.info(f"Full ChromaDB rebuild: {added} chunks")


# ---------------------------------------------------------------------------
# Connect to existing collection (for returning to callers)
# ---------------------------------------------------------------------------

def _connect_existing(provider: str, repo_path: str) -> Any:
    """Return a connected vector store / collection (read-only, no reindex)."""
    if provider == "qdrant":
        from langchain_qdrant import QdrantVectorStore

        from terminus.context.indexers.qdrant_client import collection_name
        from terminus.context.indexers.qdrant_store import open_qdrant_store
        from terminus.llm.factory import get_embedder

        collection = collection_name()
        embedder = get_embedder()
        mode = _get_mode()

        # The shared opener, so this read-only path honours qdrant.mode and the
        # shared timeout like every other path.
        base = open_qdrant_store(collection)
        kwargs: dict[str, Any] = {
            "client": base.client,
            "collection_name": collection,
            "embedding": embedder,
        }
        if mode == "hybrid":
            from langchain_qdrant import FastEmbedSparse, RetrievalMode

            kwargs["sparse_embedding"] = FastEmbedSparse(
                model_name="Qdrant/bm25"
            )
            kwargs["retrieval_mode"] = RetrievalMode.HYBRID

        return QdrantVectorStore.from_existing_collection(**kwargs)
    if provider in ("chromadb", "chroma"):
        collection, _ = _chroma_collection(repo_path)
        return collection
    return None
