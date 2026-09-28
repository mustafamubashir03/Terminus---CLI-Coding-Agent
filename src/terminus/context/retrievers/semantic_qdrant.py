from langchain_qdrant import QdrantVectorStore
from terminus.config import CONFIG
from terminus.llm.factory import get_embedder
from terminus.observability.logging import get_logger
import os

logger = get_logger(__name__)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """ Embed the query and finds k most similar chunks"""
    embedder = get_embedder()
    vector_store = QdrantVectorStore.from_existing_collection(collection_name=CONFIG["qdrant"]["collection_name"], embedding=embedder, url=os.getenv("CLUSTER_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY"))
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
