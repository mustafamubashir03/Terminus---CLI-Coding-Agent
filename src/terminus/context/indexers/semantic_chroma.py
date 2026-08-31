from pathlib import Path
from rich.console import Console
import chromadb

from terminus.config import CONFIG
from terminus.context.indexers.code_parser import get_source_files, parse_file
from terminus.llm.factory import get_embedder
from terminus.observability.logging import get_logger

logger = get_logger(__name__)
console = Console()

def get_or_create_chroma_index(repo_path: str)->chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path=CONFIG["chromadb"]["persist_dir"])
    collection = chroma_client.get_or_create_collection(
        name=CONFIG["chromadb"]["collection_name"]
    )
    if collection.count() > 0:
        logger.info(f"Found existing index with {collection.count()} chunks.")
        console.print(f"[dim]Loading existing index-{collection.count()} chunks [/dim]")
        return collection
    repo_path = str(Path.cwd())
    logger.info(f"No existing index found. Initializing new index for {repo_path}")
    console.print(f"[yellow]No index found. Initializing new index for [/yellow]{repo_path}")
    return index_codebase_chroma(repo_path)

def index_codebase_chroma(repo_path: str)->chromadb.Collection:
    """Parse all python files and store their embeddings and docstrings in Chroma. Returns the ChromaDB collection """
    embedder = get_embedder()
    chroma_client = chromadb.PersistentClient(path=CONFIG["chromadb"]["persist_dir"])
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


