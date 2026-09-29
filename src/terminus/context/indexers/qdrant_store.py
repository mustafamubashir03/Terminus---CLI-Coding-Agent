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


def _ensure_collection(client: Any, name: str) -> bool:
    """Create *name* if absent. True when the collection already had points.

    Returns whether the collection existed and was non-empty, which is the
    caller's signal to do an incremental reindex rather than a full one.
    """
    from qdrant_client import models

    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        info = client.get_collection(collection_name=name)
        if (info.points_count or 0) > 0:
            return True
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=embedding_dimensions(), distance=models.Distance.COSINE
            ),
        )
        logger.info("Created Qdrant collection %r (mode=%s)", name, qdrant_mode())
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
) -> Any:
    """Upsert *documents*, creating the collection when it is absent.

    ``sparse_embedding`` is set only for hybrid mode; a dense-only store must not
    receive one, or the collection ends up with a sparse vector name the dense
    retriever never populates.
    """
    from langchain_qdrant import QdrantVectorStore

    from terminus.llm.factory import get_embedder

    _ensure_collection(client, name)
    store = QdrantVectorStore(
        client=client, collection_name=name, embedding=get_embedder()
    )
    options: dict[str, Any] = {}
    if sparse_embedding is not None:
        options["sparse_embedding"] = sparse_embedding
    store.add_documents(documents, batch_size=BATCH_SIZE, **options)
    return store
