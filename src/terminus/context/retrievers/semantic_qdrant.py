"""Dense Qdrant retrieval.

Thin by design: the store comes from
:func:`terminus.context.indexers.qdrant_store.open_qdrant_store` and is cached
per project by :mod:`terminus.context.retrievers.cache`, and the result shape is
:func:`RetrievedChunk`. Everything specific to Qdrant - client construction,
collection, location - lives in the indexer layer, so this module is the same
size as its Chroma counterpart.
"""

from __future__ import annotations

from terminus.context.qdrant_scope import project_filter
from terminus.context.retrievers.cache import cached_store
from terminus.context.retrievers.retrieved import RetrievedChunk
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def _store():
    from terminus.context.indexers.qdrant_store import open_qdrant_store

    return cached_store("dense", open_qdrant_store)


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The *k* most similar chunks, scoped to this project."""
    results = _store().similarity_search_with_score(query, k=k, filter=project_filter())
    chunks = [
        RetrievedChunk(
            text=document.page_content,
            source=document.metadata["source"],
            name=document.metadata["name"],
            type=document.metadata["type"],
            start_line=document.metadata["start_line"],
            end_line=document.metadata["end_line"],
            score=float(score),
        )
        for document, score in results
    ]
    logger.info("Retrieved %d dense chunk(s) for query: %s", len(chunks), query)
    return chunks
