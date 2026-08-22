from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index
from terminus.config import CONFIG
def get_or_create_indexer(repo_path:str):
    provider = CONFIG["vector_store"]["provider"]

    if provider == "chromadb":
        from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index
        return get_or_create_chroma_index(repo_path)
    elif provider == "qdrant":
        from terminus.context.indexers.semantic_qdrant import get_or_create_qdrant_index
        return get_or_create_qdrant_index(repo_path)
    else:
        raise ValueError(f"Unknown indexer provider: {provider}")

def show_index():
    provider = CONFIG["vector_store"]["provider"]
    if provider == "qdrant":
        from terminus.context.indexers.semantic_qdrant import show_qdrant_semantic_index
        return show_qdrant_semantic_index()
    if provider == "chromadb":
        from terminus.context.indexers.semantic_chroma import show_chroma_semantic_index
        return show_chroma_semantic_index()
    else:
        raise ValueError(f"Unknown indexer provider: {provider}")   
    