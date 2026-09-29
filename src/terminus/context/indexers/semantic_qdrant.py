"""Semantic (dense-only) index over the Qdrant store.

One indexer for both local and cloud: the client comes from
:mod:`terminus.context.indexers.qdrant_client`, which is the only place that
decides which, and everything after client construction is shared.

An existing non-empty collection is refreshed incrementally rather than
rebuilt, using the freshness manifest. Legacy points that predate project
scoping are reported, never silently repaired - see
:mod:`terminus.context.qdrant_scope` and :mod:`terminus.context.indexers.migrate`
for why that repair cannot be automatic.
"""

from __future__ import annotations

from langchain_core.documents import Document

from terminus.context.indexers.code_parser import get_source_files, parse_file
from terminus.context.indexers.errors import VectorStoreFailure
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


def get_or_create_qdrant_index(repo_path: str, *, force_reindex: bool = False):
    """Index the codebase for dense semantic search.

    ``force_reindex=True`` wipes and rebuilds. That empties the whole collection,
    which is shared across projects, so it stays an explicit operator decision.
    """
    name = collection_name()
    client = qdrant_client.create_qdrant_client()
    logger.info(
        "Qdrant dense index: collection=%s location=%s", name, qdrant_location()
    )

    existing = {c.name for c in client.get_collections().collections}
    if name in existing and (client.get_collection(collection_name=name).points_count or 0) > 0:
        ensure_project_payload_index(client, name)
        ensure_source_payload_index(client, name)
        if unscoped_points_present(client, name):
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
    documents = _documents(files)
    store = write_documents(client, documents, name)
    _save_manifest(repo_path)
    logger.info("Semantic indexing completed. Indexed %d files into %s", len(files), name)
    return store


def _documents(files) -> list[Document]:
    """Parse *files* into scoped documents, skipping unparseable ones.

    A file that will not parse is a warning, never a failure: one bad file must
    not cost the user the whole index.
    """
    documents: list[Document] = []
    for filepath in files:
        try:
            chunks = parse_file(filepath)
        except (SyntaxError, ValueError) as exc:
            logger.warning("Skipping %s: %s: %s", filepath, VectorStoreFailure.PARSE.value, exc)
            continue
        for chunk in chunks:
            documents.append(Document(page_content=chunk.content, metadata=chunk_metadata(chunk)))
    return documents


def _save_manifest(repo_path: str) -> None:
    """Record the indexed state so the next run can diff instead of re-parsing."""
    from terminus.context.indexers.freshness import Manifest, _now_iso

    manifest = Manifest(repo_path=repo_path, last_full_index=_now_iso())
    manifest.files = Manifest.snapshot_directory(repo_path)
    manifest.save()


def _warn_unscoped(name: str) -> None:
    logger.warning(
        "Collection %r holds points with no project payload (indexed before project "
        "scoping). Retrieval is filtered by project, so these points are not "
        "returned and search will look empty. Rebuild the index to restore them; "
        "note a full reindex empties this shared collection, which may hold other "
        "projects' points.",
        name,
    )


def show_qdrant_semantic_index(store) -> None:
    """Print the first chunks in the dense index."""
    from rich.console import Console

    from terminus.context.qdrant_scope import METADATA_PAYLOAD_KEY

    console = Console()
    client = store.client
    name = collection_name()
    points, _ = client.scroll(collection_name=name, with_payload=True, with_vectors=True)
    console.print("Collection: %s" % name)
    console.print("Total points: %d" % len(points))
    for index, point in enumerate(points, 1):
        metadata = (point.payload or {}).get(METADATA_PAYLOAD_KEY) or {}
        console.print("[bold cyan]Chunk %d:[/bold cyan] %s" % (index, metadata))
        console.print("File: %s" % metadata.get("source", "?"))
        console.print("Name: %s" % metadata.get("name", "?"))
        console.print("Lines: %s-%s" % (metadata.get("start_line", "?"), metadata.get("end_line", "?")))
        console.print("-" * 50)
