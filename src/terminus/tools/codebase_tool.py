from langchain.tools import tool

from terminus.context.retrievers.factory import get_retriever
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

#: Whether this process has already refreshed the index. One refresh per process
#: is enough: the reindex is manifest-driven, so with nothing changed it is a
#: fast no-op, and doing it per search would put a filesystem walk in front of
#: every query.
_index_refreshed = False


def _refresh_index_once() -> None:
    """Bring the semantic index up to date, at most once per process.

    This used to happen in ``cli.initialize``. Moving it out of startup is what
    took the boot from ~28s to well under a second, but it must not be dropped:
    the retriever only *queries*, so a project that has never been indexed
    returns nothing forever, and ``search_codebase`` reports "no relevant code
    found" while the model quietly answers from ``grep`` instead. The user sees
    a working agent and never learns semantic search is inert.

    So the work moves to the first search rather than to every search, and rather
    than to startup. Failures are logged and ignored: the caller already degrades
    to a clear "search is unavailable" message, and a reindex that cannot run is
    not a reason to refuse the query.
    """
    global _index_refreshed
    if _index_refreshed:
        return
    _index_refreshed = True  # set first: one failed attempt must not retry per query
    try:
        from pathlib import Path

        from terminus.context.indexers.factory import get_or_create_index

        get_or_create_index(str(Path.cwd()))
    except Exception as exc:
        logger.warning("Index refresh before search failed: %s: %s", type(exc).__name__, exc)


@tool
def search_codebase(query: str) -> str:
    """ Retrieves chunks from the codebase based on the query if requires.

    This is semantic/hybrid search over the project index, which is refreshed
    automatically on the first search of a session. Use 'grep' for exact
    text matching and 'read_file' for a known path. If the index is
    unavailable, fall back to those tools.
    """
    if not query or not query.strip():
        return "No search query provided"
    query = query.strip()

    _refresh_index_once()

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
