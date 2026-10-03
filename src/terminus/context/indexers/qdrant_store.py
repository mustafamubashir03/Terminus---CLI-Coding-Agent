"""Building the Qdrant vector store, for both local and cloud.

The LangChain compatibility note
-------------------------------
``langchain-qdrant`` 1.1.0's ``QdrantVectorStore.from_documents`` and
``from_existing_collection`` both forward a ``client=`` argument into the HTTP
``ApiClient`` constructor, which rejects it::

    TypeError: Client.__init__() got an unexpected keyword argument 'client'

Only the plain constructor accepts an injected client. So a local store - which
has to be built from a ``QdrantClient(path=...)`` and cannot be described by a
URL at all - has to go through::

    QdrantVectorStore(client=client, collection_name=..., embedding=...)
    store.add_documents(documents, batch_size=...)

That is what this module does, for local and cloud alike, so there is one code
path instead of two. Cloud keeps the URL/api-key construction; local adds
``path``; everything after client construction is identical.

The collection is created explicitly when missing, because a local store has no
server to auto-create it and the constructor expects it to exist.
"""

from __future__ import annotations

from typing import Any

from terminus.context.indexers import qdrant_client
from terminus.context.indexers.qdrant_client import (
    collection_name,
    embedding_dimensions,
    qdrant_mode,
)
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 50

SPARSE_VECTOR_NAME = "langchain-sparse"


def _ensure_collection(client: Any, name: str, *, sparse: bool) -> bool:
    """Create *name* if absent. True when the collection already had points.

    Returns whether the collection existed and was non-empty, which is the
    caller's signal to do an incremental reindex rather than a full one.

    A sparse-capable collection is created with the ``langchain-sparse`` schema
    from the start. Adding it later is not possible in Qdrant without recreating
    the collection, so a dense-only collection that later needs hybrid retrieval
    is permanently unusable.
    """
    from qdrant_client import models

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(collection_name=name)
        if (info.points_count or 0) > 0:
            return True
    if name not in existing:
        kwargs: dict[str, Any] = {
            "collection_name": name,
            "vectors_config": models.VectorParams(
                size=embedding_dimensions(), distance=models.Distance.COSINE
            ),
        }
        if sparse:
            kwargs["sparse_vectors_config"] = {
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                ),
            }
        client.create_collection(**kwargs)
        logger.info(
            "Created Qdrant collection %r (mode=%s, sparse=%s)", name, qdrant_mode(), sparse
        )
    return False


def open_qdrant_store(collection: str | None = None):
    """A ``QdrantVectorStore`` bound to the configured collection.

    Used by the retrievers. The collection is not created here: a search against
    a missing collection should surface as a real "collection not found" from
    the indexer, not be papered over by creating an empty one.
    """
    from langchain_qdrant import QdrantVectorStore

    from terminus.llm.factory import get_embedder

    name = collection or collection_name()
    client = qdrant_client.create_qdrant_client()
    return QdrantVectorStore(client=client, collection_name=name, embedding=get_embedder())


def write_documents(
    client: Any,
    documents: list,
    name: str,
    *,
    sparse_embedding: Any = None,
    retrieval_mode: Any = None,
) -> Any:
    """Upsert *documents*, creating the collection when it is absent.

    ``sparse_embedding`` and ``retrieval_mode`` belong on the *constructor*.
    ``add_documents`` forwards unknown keywords to ``client.upsert``, which
    accepts and discards them, so passing ``sparse_embedding`` there writes
    dense vectors into a collection the hybrid retriever then refuses - and the
    collection cannot gain a sparse schema afterwards.
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    from terminus.llm.factory import get_embedder

    mode = retrieval_mode or RetrievalMode.DENSE
    wants_sparse = mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID)

    _ensure_collection(client, name, sparse=wants_sparse)
    store = QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=get_embedder(),
        sparse_embedding=sparse_embedding if wants_sparse else None,
        retrieval_mode=mode,
    )
    store.add_documents(documents, batch_size=BATCH_SIZE)
    return store
