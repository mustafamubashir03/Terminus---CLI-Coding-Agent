from langchain.tools import tool

from terminus.context.retrievers.factory import get_retriever
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


@tool
def search_codebase(query: str) -> str:
    """ Retrieves chunks from the codebase based on the query if requires.

    This is semantic/hybrid search over a pre-built index. Use 'grep' for exact
    text matching and 'read_file' for a known path. If the index is
    unavailable, fall back to those tools.
    """
    if not query or not query.strip():
        return "No search query provided"
    query = query.strip()

    try:
        retrieve = get_retriever()
        chunks = retrieve(query, k=5)
    except Exception as exc:
        # Semantic search is an accelerator, not the source of truth. A broken or
        # unreachable vector store must not end the turn: report it and let the
        # model continue with grep / read_file / list_directory. The configured
        # backend is not substituted here - that decision was already made, and
        # reported, when the index was built.
        logger.warning("Semantic search unavailable: %s: %s", type(exc).__name__, exc)
        return (
            "Semantic search is currently unavailable "
            f"({type(exc).__name__}), so no indexed results could be returned. "
            "Use 'grep' for exact text search, 'read_file' for a known path, and "
            "'list_directory' to explore. Do not treat this as an empty result set."
        )

    if not chunks:
        return "No relevant code found in the index. Try 'grep' for an exact string."

    results = []
    for chunk in chunks:
        results.append(
            f"File: {chunk['source']} (lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"Type: {chunk['type']}\n"
            f"Name: {chunk['name']}\n"
            f"Code:\n{chunk['text']}\n\n"
        )
    return "\n---\n".join(results)
