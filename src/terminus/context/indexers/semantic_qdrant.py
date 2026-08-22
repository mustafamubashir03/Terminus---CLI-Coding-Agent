from langchain_core.documents import Document
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from terminus.observability.logging import get_logger
from terminus.config import CONFIG
from terminus.llm.factory import get_embedder
from terminus.context.indexers.code_parser import get_source_files,parse_file

logger = get_logger(__name__)

def get_or_create_qdrant_index(repo_path: str)->QdrantVectorStore:
    """Index the codebase for semantic search"""
    collection_name = CONFIG["qdrant"]["collection_name"]
    api_key= os.getenv("QDRANT_API_KEY")
    if(not api_key):
        logger.error("QDRANT_API_KEY not found in .env file")
        raise ValueError("QDRANT_API_KEY not found in .env file")
    cluster_endpoint =os.getenv("CLUSTER_ENDPOINT")
    if(not cluster_endpoint):
        logger.error("CLUSTER_ENDPOINT not found in .env file")
        raise ValueError("CLUSTER_ENDPOINT not found in .env file")
    client = QdrantClient(url=cluster_endpoint, api_key=api_key)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        info = client.get_collection(collection_name=collection_name)
        if info.points_count > 0:
            logger.info("Collection already exists with points. Skipping indexing")
            return QdrantVectorStore.from_existing_collection(
                collection_name=collection_name,
                client=client,
                url=cluster_endpoint,
                api_key=api_key,
            )
    logger.info(f"Loading codebase from: {repo_path}")
    files = get_source_files(repo_path)
    docs = []

    for filepath in files:
        try:
            chunks = parse_file(filepath)
        except(SyntaxError, ValueError) as e:
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
                    "end_line": chunk.end_line
                }
            ))
            logger.debug(f"Embedded and stored {chunk.source}:{chunk.start_line}-{chunk.end_line}")
    embedder = get_embedder()
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name=collection_name,
        url=cluster_endpoint,
        api_key=api_key,
        batch_size=50
    )
    logger.info(f"Semantic indexing completed. Indexed {len(files)} files into {vector_store.collection.count()} chunks")
    return vector_store


def show_qdrant_semantic_index(vector_store: QdrantVectorStore)->None:
    """Show the qdrant semantic index stats"""
    from rich.console import Console
    console = Console()
    client = vector_store.client
    collection_name = CONFIG["qdrant"]["collection_name"]
    results = client.scroll(collection_name=collection_name, with_payload=True)
    points = results[0]
    console.print(f"Total points: {len(points)}")
    
    for i,point in enumerate(points):
        payload = point.payload
        embedding = point.vector
        console.print(f"[bold cyan] Chunk {i+1}:[/bold cyan] {payload}")
        console.print(f"File: {payload['source']}")
        console.print(f"Name: {payload['name']}")
        console.print(f"Lines: {payload['start_line']}-{payload['end_line']}")
        console.print(f"\n[bold]Code:[/bold]\n[code]{payload.get('page_content','')[0:300]}[/code]...\n")
        console.print(f"[bold green] Embedding [{len(embedding)}]: {', '.join(f'{v: .4f}' for v in embedding[:5])} ...")
        console.print("-" * 50)
    
    
    
