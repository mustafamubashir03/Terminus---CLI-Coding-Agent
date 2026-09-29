"""Chroma retrieval.

Isolation here is structural rather than filtered: each project has its own
persistent directory, so the collection *is* the boundary and no payload filter
is needed. That is a legitimate alternative to Qdrant's shared collection plus
a filter, and it is why Chroma's result has no ``project`` key.

The collection is cached per resolved persist directory, which is what stops a
process that moved between projects from serving the previous project's chunks.
"""

from __future__ import annotations

from terminus.context.retrievers.retrieved import RetrievedChunk
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def _collection():
    import chromadb

    from terminus.context.indexers.semantic_chroma import chroma_persist_path
    from terminus.context.retrievers.cache import cached_store

    from terminus.config import CONFIG

    return cached_store(
        f"chroma:{chroma_persist_path()}",
        lambda: chromadb.PersistentClient(path=str(chroma_persist_path())).get_or_create_collection(
            name=CONFIG["chromadb"]["collection_name"]
        ),
    )


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The *k* nearest chunks from this project's collection.

    ``score`` is ``None``: the Chroma query path used here does not return
    distances, and inventing one from a ranking position would be a number that
    looks comparable to a Qdrant score and is not.
    """
    from terminus.llm.factory import get_embedder

    embedder = get_embedder()
    results = _collection().query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=k,
        include=["documents", "metadatas"],
    )
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    chunks = [
        RetrievedChunk(
            text=document,
            source=metadata["source"],
            name=metadata["name"],
            type=metadata["type"],
            start_line=metadata["start_line"],
            end_line=metadata["end_line"],
            score=None,
        )
        for document, metadata in zip(documents, metadatas, strict=False)
        if metadata is not None
    ]
    logger.info("Retrieved %d chunk(s) for query: %s", len(chunks), query)
    return chunks
