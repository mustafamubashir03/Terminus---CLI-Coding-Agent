from langchain_qdrant import FastEmbedSparse
from langchain_qdrant import QdrantVectorStore,RetrievalMode
from qdrant_client import QdrantClient
from terminus.config import CONFIG
from terminus.llm.factory import get_embedder
from terminus.observability.logging import get_logger
import os

logger = get_logger(__name__)

RETRIVAL_MODE_MAP = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid":RetrievalMode.HYBRID
}

def _get_retrieval_mode()->RetrievalMode:
    """Get the retrieval mode from the configuration"""
    mode = CONFIG["vector_store"].get("retrieval_mode","hybrid")
    return RETRIVAL_MODE_MAP.get(mode,RetrievalMode.HYBRID)

_collection_cache = None

def _get_collection():
    """Returns a cached ChromaDB collection, opening it only once per process."""
    global _collection_cache
    if _collection_cache is None:
        cluster_endpoint = os.getenv("CLUSTER_ENDPOINT")
        api_key = os.getenv("QDRANT_API_KEY")
        collection_name = CONFIG["qdrant"]["collection_name"]
        client = QdrantClient(url=cluster_endpoint, api_key=api_key)
        _collection_cache = client.get_collection(collection_name=collection_name)
        logger.info("Qdrant collection opened and cached")
    return _collection_cache

def retrieve(query: str, k: int = 5) -> list[dict]:
    """ Embed the query and finds k most similar chunks"""
    embedder = get_embedder()
    retrieval_mode = _get_retrieval_mode()
    sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
    vector_store = QdrantVectorStore.from_existing_collection(collection_name=CONFIG["qdrant"]["collection_name"], embedding=embedder,retrieval_mode=retrieval_mode, sparse_embedding=sparse_embedding, url=os.getenv("CLUSTER_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY"))
    results = vector_store.similarity_search_with_score(query, k=k)
    chunks = []
    for doc, score in results:
        chunks.append({
            "content": doc.page_content,
            "source": doc.metadata["source"],
            "name": doc.metadata["name"],
            "type": doc.metadata["type"],
            "start_line": doc.metadata["start_line"],
            "end_line": doc.metadata["end_line"],
            "score": score
        })
        logger.debug(f"Retrieved {doc.metadata['type']} {doc.metadata['source']}:{doc.metadata['name']}")
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query}")
    return chunks
