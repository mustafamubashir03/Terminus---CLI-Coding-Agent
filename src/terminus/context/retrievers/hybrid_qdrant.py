"""Hybrid Qdrant retrieval: dense vectors blended with BM25 sparse.

Same shape as the dense retriever, plus a sparse embedder. Both resources are
cached per project, which is where the win is: the BM25 model is loaded once
rather than per query.

The store is rebuilt around the cached client so the sparse retriever and the
retrieval mode are attached. The expensive parts - the client and the model - are
reused; only this thin wrapper is new each call.
"""

from __future__ import annotations

from terminus.context.indexers.hybrid_qdrant import SPARSE_MODEL, retrieval_mode, sparse_retriever
from terminus.context.indexers.qdrant_client import collection_name
from terminus.context.qdrant_scope import project_filter
from terminus.context.retrievers.cache import cached_sparse, cached_store
from terminus.context.retrievers.retrieved import RetrievedChunk
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def _store():
    from terminus.context.indexers.qdrant_store import open_qdrant_store

    return cached_store("hybrid", open_qdrant_store)


def _sparse():
    return cached_sparse(SPARSE_MODEL, sparse_retriever)


def _hybrid_store():
    from langchain_qdrant import QdrantVectorStore

    base = _store()
    return QdrantVectorStore(
        client=base.client,
        collection_name=collection_name(),
        embedding=base.embedding,
        sparse_embedding=_sparse(),
        retrieval_mode=retrieval_mode(),
    )


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """The *k* best chunks by the configured dense/sparse blend, project-scoped."""
    results = _hybrid_store().similarity_search_with_score(
        query, k=k, filter=project_filter()
    )
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
    logger.info("Retrieved %d hybrid chunk(s) for query: %s", len(chunks), query)
    return chunks
