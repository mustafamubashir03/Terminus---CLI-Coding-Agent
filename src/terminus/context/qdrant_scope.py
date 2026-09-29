"""The project-scope contract for Qdrant, in one place.

Qdrant collections are shared by every project: ``CONFIG["qdrant"]
["collection_name"]`` names a single collection that all workspaces write into.
That makes the payload field carrying the project identity the only thing
standing between project A's source code and project B's ``search_codebase``
results, and it has to be written by every indexer and demanded by every
retriever.

Both halves used to be inlined per module, which is exactly how the semantic
(dense) indexer and retriever came to be left out of the hybrid pair. So the
field name, the identifier written, the identifier matched, and the payload
shape live here, and both sides call in. There is no framework here: one
constant, one metadata helper, one filter.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models

from terminus.observability.logging import get_logger
from terminus.workspace import PROJECT_METADATA_KEY, project_key

logger = get_logger(__name__)

METADATA_PAYLOAD_KEY = "metadata"
"""The payload key LangChain nests document metadata under.

Hard-coded in ``QdrantVectorStore._build_payloads`` and not configurable
through the constructor arguments Terminus uses, so it is a constant here for
the same reason the field name is.
"""


def payload_key(field: str) -> str:
    """Qualify a document-metadata field as a Qdrant payload path.

    ``QdrantVectorStore._build_payloads`` nests everything a Document carries
    under the ``metadata`` payload key, so a Document whose metadata has
    ``project`` is stored at ``metadata.project``. Filtering on the bare name
    matches nothing, which looks identical to an empty index and is how a
    supposedly-scoped search can silently return zero results forever.
    """
    return f"{METADATA_PAYLOAD_KEY}.{field}"


def project_filter() -> models.Filter:
    """Restrict a search to points indexed for the current project.

    Built unconditionally, including when the field is absent from the stored
    payload. An old unscoped collection therefore matches nothing and reads as
    empty rather than as a leak; reindexing repopulates it. The alternative -
    quietly widening the filter when a collection looks legacy - would restore
    the cross-project leak this exists to close.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key=payload_key(PROJECT_METADATA_KEY),
                match=models.MatchValue(value=project_key()),
            )
        ]
    )


def unscoped_points_present(client: Any, collection: str, sample: int = 8) -> bool:
    """True if a non-empty collection holds points written before scoping.

    Used to make a legacy collection visible instead of silently inert. It
    samples a handful of points rather than counting the collection, because the
    answer only needs to distinguish "everything is tagged" from "at least one
    point is not", and this runs on the index path.

    This only reports. It never rebuilds: the collection is shared, legacy points
    carry no owner, so they cannot be attributed to a project and deleted
    selectively. Only a full reindex can clear them, and that is the operator's
    decision to make, not something to trigger behind their back.
    """
    try:
        points, _ = client.scroll(
            collection_name=collection, limit=max(1, sample), with_payload=True
        )
    except Exception:
        # An unreadable collection is a different problem; do not guess here.
        return False
    if not points:
        return False
    for point in points:
        payload = point.payload or {}
        metadata = payload.get(METADATA_PAYLOAD_KEY) or {}
        if not isinstance(metadata, dict) or PROJECT_METADATA_KEY not in metadata:
            return True
    return False


def _create_payload_index(client: Any, collection: str, field: str) -> bool:
    """Create a keyword payload index, tolerating only "already exists".

    The one place a payload index is created, because this is the one place that
    gets it right. The indexers used to wrap this in a bare
    ``except Exception: pass``, which swallowed a rejected or malformed index
    request just as happily as the expected "already exists" - so a genuinely
    broken index looked identical to a redundant one, and searches silently
    scanned instead of using an index.

    Returns True when the index was created, False when it already existed.
    A real failure is re-raised, so it surfaces.
    """
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema="keyword",
        )
        return True
    except Exception as exc:
        text = str(exc).lower()
        if "already" in text or "exist" in text:
            return False
        raise


def ensure_project_payload_index(client: Any, collection: str) -> None:
    """Create the keyword payload index the project filter is served from.

    Qdrant filters an unindexed payload field by scanning, which is correct but
    gets slower as the shared collection grows. The existing code already does
    this for ``metadata.source`` for the same reason.
    """
    try:
        if _create_payload_index(client, collection, payload_key(PROJECT_METADATA_KEY)):
            logger.info("Created payload index on %s", payload_key(PROJECT_METADATA_KEY))
    except Exception as exc:
        # A payload index is a performance optimisation, not correctness: Qdrant
        # answers the filter by scanning without it. So a failure is reported and
        # indexing continues, rather than failing the whole index over speed.
        logger.warning(
            "Could not index the project payload field (%s); searches will still be "
            "correct but will scan rather than use an index", exc,
        )


def ensure_source_payload_index(client: Any, collection: str) -> None:
    """Create the keyword index on ``metadata.source``.

    Required for correctness, not only speed: the incremental reindex deletes
    changed files by filtering on this field, and without the index Qdrant
    rejects the filtered delete.
    """
    try:
        _create_payload_index(client, collection, payload_key("source"))
    except Exception as exc:
        logger.warning(
            "Could not index the source payload field (%s); incremental reindex may "
            "be unable to delete changed files until it succeeds", exc,
        )


def chunk_metadata(chunk: Any) -> dict[str, Any]:
    """Payload for one parsed chunk, scoped to the current project.

    The single place the shared collection is populated, so every indexer tags
    its points identically. The returned dict is Document metadata - flat, and
    un-namespaced; LangChain adds the ``metadata.`` prefix when it writes the
    payload.
    """
    return {
        "source": chunk.source,
        "name": chunk.name,
        "type": chunk.type,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        PROJECT_METADATA_KEY: project_key(),
    }
