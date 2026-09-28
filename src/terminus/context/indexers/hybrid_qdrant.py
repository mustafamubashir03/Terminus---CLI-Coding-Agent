from langchain_core.documents import Document
import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient

from terminus.observability.logging import get_logger
from terminus.config import CONFIG
from terminus.llm.factory import get_embedder
from terminus.context.indexers.code_parser import get_source_files,parse_file

logger = get_logger(__name__)

RETRIEVAL_MODE_MAP = {
    "dense": RetrievalMode.DENSE,
    "sparse": RetrievalMode.SPARSE,
    "hybrid":RetrievalMode.HYBRID
}

def get_retrieval_mode()->RetrievalMode:
    mode = CONFIG["vector_store"].get("retrieval_mode","hybrid")
    return RETRIEVAL_MODE_MAP.get(mode,RetrievalMode.HYBRID)

def get_or_create_qdrant_hybrid_index(repo_path: str, *, force_reindex: bool = False) -> QdrantVectorStore:
    """Index the codebase for Hybrid search.

    If the collection already has points, an incremental freshness check is
    performed and only changed/new files are re-indexed.  Pass
    *force_reindex=True* to wipe and rebuild everything from scratch.
    """
    collection_name = CONFIG["qdrant"]["collection_name"]
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        logger.error("QDRANT_API_KEY not found in .env file")
        raise ValueError("QDRANT_API_KEY not found in .env file")
    cluster_endpoint = os.getenv("CLUSTER_ENDPOINT")
    if not cluster_endpoint:
        logger.error("CLUSTER_ENDPOINT not found in .env file")
        raise ValueError("CLUSTER_ENDPOINT not found in .env file")
    embedder = get_embedder()
    retrieval_mode = get_retrieval_mode()
    timeout = float(CONFIG.get("qdrant", {}).get("timeout_seconds", 5))
    client = QdrantClient(
        url=cluster_endpoint,
        api_key=api_key,
        timeout=timeout,
        check_compatibility=False,
    )
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        info = client.get_collection(collection_name=collection_name)
        if (info.points_count or 0) > 0:
            if force_reindex:
                from terminus.context.indexers.reindexer import full_reindex

                logger.info("Force reindex requested ΓÇö wiping and rebuilding")
                return full_reindex(repo_path)[0]

            from terminus.context.indexers.reindexer import incremental_reindex

            vector_store, result = incremental_reindex(repo_path)
            if result.files_added or result.files_modified or result.files_deleted:
                logger.info(f"Incremental reindex: {result}")
            return vector_store

    # Collection missing or empty ΓÇö full initial index
    logger.info(f"Loading codebase from: {repo_path}")
    files = get_source_files(repo_path)
    docs = []

    for filepath in files:
        try:
            chunks = parse_file(filepath)
        except (SyntaxError, ValueError) as e:
            logger.error(f"Skipping {filepath} due to parsing error: {e}")
            continue
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk.content,
                metadata={
                    "source": chunk.source,
                    "name": chunk.name,
                    "type": chunk.type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                },
            ))
            logger.debug(f"Embedded and stored {chunk.source}:{chunk.start_line}-{chunk.end_line}")

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedder,
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=retrieval_mode,
        collection_name=collection_name,
        url=cluster_endpoint,
        api_key=api_key,
        batch_size=50,
    )

    # Create manifest so next startup does incremental diff
    from terminus.context.indexers.freshness import Manifest, _now_iso

    # Ensure metadata.source is indexed so future incremental deletions work
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.source",
            field_schema="keyword",
        )
    except Exception:
        pass  # "already exists" is the expected case

    manifest = Manifest(repo_path=repo_path, last_full_index=_now_iso())
    manifest.files = Manifest.snapshot_directory(repo_path)
    manifest.save()

    logger.info(
        f"Hybrid indexing completed. Indexed {len(files)} files "
        f"into {vector_store.collection_name}"
    )
    return vector_store


def show_qdrant_hybrid_index(vector_store: QdrantVectorStore)->None:
    """Show the qdrant semantic index stats"""
    from rich.console import Console
    console = Console()
    client = vector_store.client
    collection_name = CONFIG["qdrant"]["collection_name"]
    results = client.scroll(collection_name=collection_name, with_payload=True,with_vectors=True)
    points = results[0]
    console.print(f"Total points: {len(points)}")
    
    for i,point in enumerate(points):
        payload = point.payload
        console.print(f"[bold cyan] Chunk {i+1}:[/bold cyan] {payload}")
        if payload is None:
            console.print("[yellow]No payload for this point, skipping.[/yellow]")
            continue
        console.print(f"File: {payload['metadata']['source']}")
        console.print(f"Name: {payload['metadata']['name']}")
        console.print(f"Lines: {payload['metadata']['start_line']}-{payload['metadata']['end_line']}")
        console.print(
            f"\n[bold]Code:[/bold]\n"
            f"[code]{payload.get('page_content', '')[:300]}[/code]...\n"
        )

        raw_vec = point.vector
        if isinstance(raw_vec, dict):
            dense = next((v for v in raw_vec.values() if isinstance(v, list) and v and isinstance(v[0], float)), None)
            embedding: list[float] | None = dense  # type: ignore[assignment]
        elif isinstance(raw_vec, list) and raw_vec and isinstance(raw_vec[0], list):
            embedding = raw_vec[0]  
        else:
            embedding = raw_vec  # type: ignore[assignment]
        if embedding is not None:
            console.print(
                f"[bold green]Embedding [{len(embedding)}]: "
                f"{', '.join(f'{v:.4f}' for v in embedding[:5])} ..."
            )
        else:
            console.print("[yellow]No embedding available for this point.[/yellow]")
        console.print("-" * 50)
    
    
    
