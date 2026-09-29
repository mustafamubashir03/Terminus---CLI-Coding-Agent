"""Picking the retriever for the configured backend.

Returns the ``retrieve`` function itself rather than an object. The choice is
re-read from configuration on every call, so a backend change takes effect on the
next search without a restart, and callers depend on one plain function
signature - ``retrieve(query, k) -> list[RetrievedChunk]`` - rather than on a
retriever class hierarchy.

All four backends honour that signature and produce the same result shape; see
:mod:`terminus.context.retrievers.retrieved`.
"""

from __future__ import annotations

from terminus.config import CONFIG
from terminus.context.indexers.factory import CHROMA_ALIASES, validate
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


def get_retriever():
    """The ``retrieve`` function for the configured provider and mode."""
    provider = str(CONFIG["vector_store"]["provider"]).lower()
    mode = str(CONFIG["rag"]["mode"]).lower()
    validate(provider, mode)
    logger.debug("Retriever: provider=%s mode=%s", provider, mode)

    if provider in CHROMA_ALIASES:
        # validate() has already rejected hybrid for a non-Qdrant provider, so
        # reaching here with mode != semantic is unreachable by configuration.
        from terminus.context.retrievers.semantic_chroma import retrieve

        return retrieve
    if mode == "hybrid":
        from terminus.context.retrievers.hybrid_qdrant import retrieve

        return retrieve
    from terminus.context.retrievers.semantic_qdrant import retrieve

    return retrieve
