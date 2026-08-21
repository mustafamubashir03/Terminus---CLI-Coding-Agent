import chromadb
from terminus.config import CONFIG
from terminus.llm.factory import get_embedder
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

_collection_cache: chromadb.Collection | None = None

def _get_collection() -> chromadb.Collection:
    """Returns a cached ChromaDB collection, opening it only once per process."""
    global _collection_cache
    if _collection_cache is None:
        chroma_client = chromadb.PersistentClient(path=CONFIG["chromadb"]["persist_dir"])
        _collection_cache = chroma_client.get_or_create_collection(name=CONFIG["chromadb"]["collection_name"])
        logger.info("ChromaDB collection opened and cached")
    return _collection_cache

def retrieve(query: str, k: int = 5) -> list[dict]:
    """ Embed the query and finds k most similar chunks"""
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    chunks = []
    for doc, meta in zip(docs, metas):
        chunks.append({
            "document": doc,
            "source": meta["source"],
            "name": meta["name"],
            "type": meta["type"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
        })
        logger.debug(f"Retrieved {meta['type']} {meta['source']}:{meta['name']}")
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query}")
    return chunks