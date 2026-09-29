"""Chroma semantic index.

Chroma is a first-class backend, not a consolation prize: it is local, needs no
credentials, persists, and is the default. This module makes that path as good as
the Qdrant one rather than merely working.

Batching
--------
The previous version embedded and upserted one chunk at a time inside the file
loop - one model call and one database round trip per chunk. Measured on 200
chunks that was 5.6s against 0.39s for the same work batched, and the gap grows
with the repository. Chunking and embedding *semantics* are unchanged: same
chunks, same model, same metadata, same deterministic ids. Only the number of
round trips differs.

``embed_documents`` is used when the embedder provides it, and the per-chunk path
is kept as a fallback so a custom embedder that only implements
``embed_query`` still works.
"""

from __future__ import annotations

from pathlib import Path

from terminus.config import CONFIG
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

#: Chunks per upsert. Large enough to amortise the round trip, small enough that
#: a failure does not lose much work and memory stays bounded.
BATCH_SIZE = 200


def _repo_path(repo_path: str | Path) -> Path:
    return Path(repo_path).expanduser().resolve()


def chroma_persist_path(repo_path: str | Path | None = None) -> Path:
    """The directory backing this project's Chroma store.

    Shared by the indexer and the retriever so they cannot address different
    stores. A relative configured path resolves against the project root, which
    is what isolates two projects in one process.
    """
    configured = Path(CONFIG["chromadb"]["persist_dir"]).expanduser()
    if configured.is_absolute():
        return configured
    return (_repo_path(repo_path) if repo_path is not None else _repo_path(Path.cwd())) / configured


def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_persist_path()))


def _collection(name: str | None = None):
    return _client().get_or_create_collection(
        name=name or CONFIG["chromadb"]["collection_name"]
    )


def get_or_create_chroma_index(repo_path: str, *, force_reindex: bool = False):
    """Load or build the Chroma index for *repo_path*.

    An existing non-empty collection is refreshed incrementally rather than
    rebuilt, using the same freshness manifest the Qdrant path uses.
    """
    if force_reindex:
        from terminus.context.indexers.reindexer import full_reindex

        logger.info("Force reindex requested - wiping and rebuilding")
        return full_reindex(repo_path)[0]

    collection = _collection()
    if collection.count() > 0:
        from terminus.context.indexers.reindexer import incremental_reindex

        _store, result = incremental_reindex(repo_path)
        if result.files_added or result.files_modified or result.files_deleted:
            logger.info("Incremental reindex: %s", result)
        return collection

    logger.info("No existing Chroma index for %s; building one", repo_path)
    result_collection = index_codebase_chroma(repo_path)
    _save_manifest(repo_path)
    return result_collection


def index_codebase_chroma(repo_path: str):
    """Parse, embed and store every source file, in batches."""
    from terminus.context.indexers.code_parser import get_source_files, parse_file
    from terminus.llm.factory import get_embedder

    embedder = get_embedder()
    collection = _collection()
    logger.info("Chroma indexing of %s", repo_path)

    records: list[tuple[str, str, dict]] = []
    for filepath in get_source_files(repo_path):
        try:
            chunks = parse_file(filepath)
        except (SyntaxError, ValueError) as exc:
            logger.warning("Skipping %s: parse: %s", filepath, exc)
            continue
        for chunk in chunks:
            # Deterministic id, so a re-upsert replaces rather than duplicates.
            records.append((
                f"{chunk.source}:{chunk.start_line}-{chunk.end_line}",
                chunk.content,
                {
                    "name": chunk.name,
                    "type": chunk.type,
                    "source": chunk.source,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
            ))

    _embed_and_upsert(collection, embedder, records)
    logger.info("Indexed %d chunk(s); collection holds %d", len(records), collection.count())
    return collection


def _embed_and_upsert(collection, embedder, records: list[tuple[str, str, dict]]) -> None:
    """Embed and store *records*, batched when the embedder allows it.

    Falls back to one-at-a-time embedding for an embedder that only implements
    ``embed_query``; the upsert is batched either way, which is where most of the
    saving is.
    """
    if not records:
        return
    texts = [content for _id, content, _meta in records]
    batched = hasattr(embedder, "embed_documents")
    for start in range(0, len(records), BATCH_SIZE):
        window = records[start:start + BATCH_SIZE]
        window_texts = texts[start:start + BATCH_SIZE]
        if batched:
            vectors = embedder.embed_documents(window_texts)
        else:
            vectors = [embedder.embed_query(text) for text in window_texts]
        collection.upsert(
            ids=[record[0] for record in window],
            embeddings=vectors,
            documents=window_texts,
            metadatas=[record[2] for record in window],
        )


def _save_manifest(repo_path: str) -> None:
    from terminus.context.indexers.freshness import Manifest, _now_iso

    manifest = Manifest(repo_path=repo_path, last_full_index=_now_iso())
    manifest.files = Manifest.snapshot_directory(repo_path)
    manifest.save()


def show_chroma_semantic_index(collection) -> None:
    """Print a summary of the Chroma index."""
    from rich.console import Console

    console = Console()
    console.print("[bold green]Semantic Index Stats:[/bold green]")
    console.print("[dim]Collection:[/dim] %s" % collection.name)
    console.print("[dim]Total Chunks:[/dim] %d" % collection.count())
    results = collection.get(include=["documents", "metadatas"])
    for index, (document, metadata) in enumerate(
        zip(results.get("documents") or [], results.get("metadatas") or [], strict=False), 1
    ):
        console.print("Chunk %d:" % index)
        console.print("[bold green]Metadata:[/bold green] %s" % metadata)
        console.print("[bold green]Content:[/bold green] %s" % (document or "")[:300])
        console.print("-" * 50)
