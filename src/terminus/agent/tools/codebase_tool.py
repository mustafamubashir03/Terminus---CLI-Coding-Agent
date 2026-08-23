from terminus.context.retrievers.factory import get_retriever
from langchain.tools import tool
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

@tool
def search_codebase(query: str) -> str:
    """ Retrieves chunks from the codebase based on the query if requires."""
    retrieve = get_retriever()
    chunks = retrieve(query,k=5)
    if not chunks:
        return "No relevant code found"
        
    results = []
    for chunk in chunks:
        results.append(
            f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"Type: {chunk['type']}\n"
            f"Name: {chunk['name']}\n"
            f"Code:\n{chunk['content']}\n\n"
        )
    return "\n---\n".join(results)
    
    
    