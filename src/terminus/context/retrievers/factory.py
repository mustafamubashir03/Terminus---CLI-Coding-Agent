from terminus.observability.logging import get_logger
from terminus.config import CONFIG
logger = get_logger(__name__)

def get_retriever():
    """ Get the retriever based on the vector store configuration"""
    vector_store = CONFIG["vector_store"]["provider"]
    logger.info(f"Using vector store: {vector_store}")
    if vector_store == "chroma":
        from .semantic_chroma import retrieve
        return  retrieve
    elif vector_store == "qdrant":
        from .semantic_qdrant import retrieve
        return retrieve
    else:
        raise ValueError(f"Vector store not found: {vector_store}")