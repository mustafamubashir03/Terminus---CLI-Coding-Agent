from terminus.observability.logging import get_logger
from terminus.config import CONFIG
logger = get_logger(__name__)

def get_retriever():
    """ Get the retriever based on the vector store configuration"""
    vector_store = CONFIG["vector_store"]["provider"]
    mode = CONFIG["rag"]["mode"]
    logger.info(f"Using vector store: {vector_store} with mode: {mode}")
    if vector_store == "chroma":
        if mode == "semantic":
            from .semantic_chroma import retrieve
            return retrieve
        else:
            raise ValueError(f"Hybrid mode is only supported for qdrant")
    elif vector_store == "qdrant":
        if mode == "semantic":
            from .semantic_qdrant import retrieve
            return retrieve
        elif mode == "hybrid":
            from .hybrid_qdrant import retrieve
            return retrieve
    else:
        raise ValueError(f"Vector store not found: {vector_store}")