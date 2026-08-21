from langchain.tools import tool
from terminus.context.retrievers.semantic_chroma import retrieve
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

@tool
def search_codebase(query: str) -> str:
    """ Retrieves chunks from the codebase based on the query"""

    chunks = retrieve(query,k=5)
    if not chunks:
        return "No relevant code found"
        
    results = []
    for chunk in chunks:
        results.append(
            f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"Type: {chunk['type']}\n"
            f"Name: {chunk['name']}\n"
            f"Code:\n{chunk['document']}"
        )
    return "\n---\n".join(results)
    
    
    