import chromadb

from terminus.config import CONFIG
from terminus.context.indexers.code_parser import get_source_files, parse_file
from terminus.llm.factory import get_embedder
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

def index_codebase(repo_path: str)->chromadb.Collection:
    """Parse all python files and store their embeddings and docstrings in Chroma. Returns the ChromaDB collection """
    
    embedder = get_embedder()
    chroma_client = chromadb.PersistentClient(path=CONFIG["chromadb"]["persist_dir"])
    collection = chroma_client.get_or_create_collection(
        name=CONFIG["chromadb"]["collection_name"],
        embedding_function=embedder
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

        
    