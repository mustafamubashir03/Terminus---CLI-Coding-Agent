import chromadb
from pathlib import Path
from terminus.config import CONFIG
from terminus.llm.factory import get_embedder
from terminus.observability.logging import get_logger
from terminus.workspace import project_root

logger = get_logger(__name__)

_collection_cache: chromadb.Collection | None = None
_collection_cache_key: str | None = None


def _persist_dir() -> str:
    """Absolute persist directory for the *current* project.

    Mirrors context/indexers/semantic_chroma._chroma_persist_path so the
    indexer and the retriever always address the same store. A relative
    configured path is resolved against the project root, so two projects in one
    process get two different collections.
    """
    configured = Path(CONFIG["chromadb"]["persist_dir"]).expanduser()
    if configured.is_absolute():
        return str(configured)
    return str(project_root() / configured)


def _get_collection() -> chromadb.Collection:
    """Return the project's ChromaDB collection, opening it once per project.

    The cache is keyed by the resolved persist directory. A single unkeyed
    module-level handle would be reused after a change of project and would
    return the *previous* project's chunks, i.e. cross-project data leakage
    through the semantic index.
    """
    global _collection_cache, _collection_cache_key
    key = _persist_dir()
    if _collection_cache is None or _collection_cache_key != key:
        chroma_client = chromadb.PersistentClient(path=key)
        _collection_cache = chroma_client.get_or_create_collection(
            name=CONFIG["chromadb"]["collection_name"]
        )
        _collection_cache_key = key
        logger.info("ChromaDB collection opened for %s", key)
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
            "content": doc,
            "source": meta["source"],
            "name": meta["name"],
            "type": meta["type"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
        })
        logger.debug(f"Retrieved {meta['type']} {meta['source']}:{meta['name']}")
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query}")
    return chunks
