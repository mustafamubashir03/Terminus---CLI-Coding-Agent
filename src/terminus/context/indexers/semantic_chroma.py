from pathlib import Path
from rich.console import Console
import chromadb

from terminus.config import CONFIG
from terminus.context.indexers.code_parser import get_source_files, parse_file
from terminus.llm.factory import get_embedder
from terminus.observability.logging import get_logger

logger = get_logger(__name__)
console = Console()


def _repo_path(repo_path: str | Path) -> Path:
    return Path(repo_path).expanduser().resolve()


def _chroma_persist_path(repo_path: str | Path) -> Path:
    configured = Path(CONFIG["chromadb"]["persist_dir"]).expanduser()
    if configured.is_absolute():
        return configured
    return _repo_path(repo_path) / configured


def get_or_create_chroma_index(repo_path: str, *, force_reindex: bool = False) -> chromadb.Collection:
    """Create or load the ChromaDB index.

    If the collection already has documents, an incremental freshness check is
    performed and only changed/new files are re-indexed.  Pass
    *force_reindex=True* to wipe and rebuild everything from scratch.
    """
    repo_path = str(_repo_path(repo_path))
    persist_dir = str(_chroma_persist_path(repo_path))
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=CONFIG["chromadb"]["collection_name"]
    )
    if collection.count() > 0:
        if force_reindex:
            from terminus.context.indexers.reindexer import full_reindex

            logger.info("Force reindex requested ΓÇö wiping and rebuilding")
            return full_reindex(repo_path)[0]

        from terminus.context.indexers.reindexer import incremental_reindex

        _vs, result = incremental_reindex(repo_path)
        if result.files_added or result.files_modified or result.files_deleted:
            logger.info(f"Incremental reindex: {result}")
        return collection

    logger.info(f"No existing index found. Initializing new index for {repo_path}")
    console.print(f"[yellow]No index found. Initializing new index for [/yellow]{repo_path}")
    result_coll = index_codebase_chroma(repo_path)

    # Create manifest so next startup does incremental diff
    from terminus.context.indexers.freshness import Manifest, _now_iso

    manifest = Manifest(repo_path=repo_path, last_full_index=_now_iso())
    manifest.files = Manifest.snapshot_directory(repo_path)
    manifest.save()

    return result_coll

def index_codebase_chroma(repo_path: str)->chromadb.Collection:
    """Parse all python files and store their embeddings and docstrings in Chroma. Returns the ChromaDB collection """
    embedder = get_embedder()
    repo_path = str(_repo_path(repo_path))
    persist_dir = str(_chroma_persist_path(repo_path))
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=CONFIG["chromadb"]["collection_name"]
    )
    logger.info(f"Starting semantic indexing of {repo_path}...")

    files = get_source_files(repo_path)

    for filepath in files:
        try:
            chunks = parse_file(filepath)
        except(SyntaxError, ValueError) as e:
            logger.warning(f"Skipping {filepath} due to parsing error: {e}")
            continue

        for chunk in chunks:
            embedding = embedder.embed_query(chunk.content)
            doc_id = f"{chunk.source}:{chunk.start_line}-{chunk.end_line}"

            collection.upsert(
                embeddings=[embedding],
                documents=[chunk.content],
                metadatas=[{
                    "name": chunk.name,
                    "type": chunk.type,
                    "source": chunk.source,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line
                }],
                ids=[doc_id]
            )
            logger.debug(f"Embedded and stored {doc_id}")
    logger.info(f"Indexed {len(files)} files with {collection.count()} chunks")
    return collection

        
    
def show_chroma_semantic_index(collection:chromadb.Collection)->None:
    """Show the semantic index stats"""
    console.print("[bold green]Semantic Index Stats:[/bold green]")
    console.print(f"[dim]Collection:[/dim] {collection.name}")
    console.print(f"[dim]Total Chunks:[/dim] {collection.count()}")
    results = collection.get(include=['documents','metadatas',"embeddings"])
    docs = results['documents'] or []
    metas = results['metadatas'] or []
    embs = results['embeddings'] or []
    for i,(doc, meta, emb) in enumerate(zip(docs, metas, embs)):
        console.print(f"Chunk {i+1}:\n")
        console.print(f"[bold green]Metadata:[/bold green] {meta}")
        console.print(f"[bold green]Content:[/bold green] {doc}")
        console.print(f"[bold green]Embedding:[/bold green] {emb[:20]}...")
        console.print("-"*50)


