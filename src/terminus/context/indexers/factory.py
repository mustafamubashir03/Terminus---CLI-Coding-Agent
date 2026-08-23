from terminus.config import CONFIG
def get_or_create_indexer(repo_path:str):
    provider = CONFIG["vector_store"]["provider"]
    mode = CONFIG["rag"]["mode"]
    
    if mode == "semantic":
        if provider == "chromadb":
            from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index
            return get_or_create_chroma_index(repo_path)
        elif provider == "qdrant":
            from terminus.context.indexers.semantic_qdrant import get_or_create_qdrant_index
            return get_or_create_qdrant_index(repo_path)
        else:
            raise ValueError(f"Unknown indexer provider: {provider}")
    elif mode == "hybrid":
        if provider == "qdrant":
            from terminus.context.indexers.hybrid_qdrant import get_or_create_qdrant_hybrid_index
            return get_or_create_qdrant_hybrid_index(repo_path)
        else:
            raise ValueError(f"Hybrid mode is only supported for qdrant")

def show_index(index):
    provider = CONFIG["vector_store"]["provider"]
    mode = CONFIG["rag"]["mode"]
    if mode == "semantic":
        if provider == "qdrant":
            from terminus.context.indexers.semantic_qdrant import show_qdrant_semantic_index
            return show_qdrant_semantic_index(index)
        elif provider == "chromadb":
            from terminus.context.indexers.semantic_chroma import show_chroma_semantic_index
            return show_chroma_semantic_index(index)
    elif mode == "hybrid":
        if provider == "qdrant":
            from terminus.context.indexers.hybrid_qdrant import show_qdrant_hybrid_index
            return show_qdrant_hybrid_index(index)
        else:
            raise ValueError(f"Hybrid mode is only supported for qdrant")
    else:
        raise ValueError(f"Unknown indexer provider: {provider}")   
    