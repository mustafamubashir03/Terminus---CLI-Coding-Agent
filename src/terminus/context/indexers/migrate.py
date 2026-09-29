"""Operator-facing migration of a Qdrant collection to project-scoped points.

Why this exists
---------------
Retrieval is project-filtered, so a point with no project payload is invisible.
Collections indexed before project scoping exist, and the automatic paths
cannot repair them:

* ``incremental_reindex`` returns early when no file changed, so an unchanged
  manifest leaves the legacy points untagged forever;
* ``full_reindex`` rebuilds correctly but **empties the whole collection**, and
  the collection is shared, so it would also destroy other projects' points.

Legacy points carry no owner, so they cannot be attributed to a project and
deleted selectively. There is no safe automatic repair, and guessing ownership
would risk deleting another workspace's data. So the decision belongs to an
operator, and this module makes it explicit and informed rather than leaving a
user with a silently empty search.

The report is read-only and safe by default. Destroying anything requires an
explicit flag, and even then the collection's sharing is reported first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from terminus.config import CONFIG
from terminus.context.qdrant_scope import (
    METADATA_PAYLOAD_KEY,
    PROJECT_METADATA_KEY,
)
from terminus.observability.logging import get_logger
from terminus.workspace import project_key

logger = get_logger(__name__)

SCAN_PAGE_SIZE = 256
"""Points fetched per scroll page while counting.

A count has to walk the collection, so it is paged and bounded: the report stays
responsive on a large shared collection instead of trying to load it all to
find out what is in it.
"""


@dataclass
class CollectionReport:
    """What is actually in the collection, as opposed to what we assume."""

    collection: str = ""
    reachable: bool = False
    error: str = ""
    total_points: int = 0
    scoped_points: int = 0
    unscoped_points: int = 0
    foreign_points: int = 0
    this_project_points: int = 0
    sampled: bool = False
    owners: dict[str, int] = field(default_factory=dict)

    @property
    def needs_migration(self) -> bool:
        return self.unscoped_points > 0

    @property
    def is_shared(self) -> bool:
        """Another project owns points here.

        Unscoped points have no owner at all, so this deliberately reports only
        what can actually be proven. Legacy points are reported separately and
        never counted as shared.
        """
        return self.foreign_points > 0

    @property
    def exclusively_ours(self) -> bool:
        """Everything present is tagged for this project, and nothing legacy.

        Only then is a rebuild provably safe: it cannot destroy another
        project's data, because there is none.
        """
        return (
            self.reachable
            and self.total_points > 0
            and not self.needs_migration
            and not self.is_shared
        )


def _client():
    from qdrant_client import QdrantClient

    api_key = os.environ.get("QDRANT_API_KEY")
    endpoint = os.environ.get("CLUSTER_ENDPOINT")
    if not api_key or not endpoint:
        raise ValueError(
            "QDRANT_API_KEY and CLUSTER_ENDPOINT must be set to inspect the index"
        )
    return QdrantClient(
        url=endpoint,
        api_key=api_key,
        timeout=float(CONFIG.get("qdrant", {}).get("timeout_seconds", 5)),
        check_compatibility=False,
    )


def inspect_collection(
    collection: str | None = None, *, page_size: int = SCAN_PAGE_SIZE
) -> CollectionReport:
    """Report what a collection holds. Read-only; never mutates anything.

    An unreachable Qdrant is reported as such rather than raised, because the
    most useful thing this can tell an operator is "I could not check".
    """
    name = collection or CONFIG.get("qdrant", {}).get("collection_name", "")
    report = CollectionReport(collection=name)
    if not name:
        report.error = "No collection name is configured."
        return report

    try:
        client = _client()
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    try:
        if not client.collection_exists(name):
            report.reachable = True
            report.error = "Collection does not exist yet."
            return report

        info = client.get_collection(collection_name=name)
        report.reachable = True
        total = int(info.points_count or 0)
        report.total_points = total
        mine = project_key()

        offset = None
        seen = 0
        while True:
            points, offset = client.scroll(
                collection_name=name,
                limit=page_size,
                offset=offset,
                with_payload=True,
            )
            if not points:
                break
            for point in points:
                seen += 1
                payload = point.payload or {}
                metadata = payload.get(METADATA_PAYLOAD_KEY) or {}
                owner = (
                    metadata.get(PROJECT_METADATA_KEY)
                    if isinstance(metadata, dict)
                    else None
                )
                if not owner:
                    report.unscoped_points += 1
                else:
                    report.scoped_points += 1
                    report.owners[str(owner)] = report.owners.get(str(owner), 0) + 1
                    if str(owner) == mine:
                        report.this_project_points += 1
                    else:
                        report.foreign_points += 1
            if offset is None:
                break
        # Scrolling counts what is really there, which is the number that
        # matters; the collection's own counter is kept for comparison.
        report.sampled = seen < total
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
    return report


def describe_plan(report: CollectionReport) -> list[str]:
    """Exactly what a rebuild would delete and recreate, in plain words.

    Shown before anything destructive, so the operator is consenting to a
    specific description rather than to a verb.
    """
    if not report.reachable:
        return [f"Cannot inspect {report.collection}: {report.error}"]
    if not report.total_points:
        return [f"{report.collection} is empty. Nothing to migrate."]
    if not report.needs_migration:
        return [
            f"{report.collection} holds {report.total_points} point(s), all already "
            f"project-scoped. No migration needed."
        ]

    lines = [
        f"Rebuilding {report.collection} would DELETE all {report.total_points} "
        f"point(s) and recreate {report.total_points} point(s) from the current files.",
        f"  unscoped (no project owner): {report.unscoped_points}",
        f"  scoped to this project:      {report.this_project_points}",
    ]
    if report.is_shared:
        lines.append(
            f"  scoped to OTHER projects:    {report.foreign_points}  <-- these "
            f"would be destroyed"
        )
        for owner, count in sorted(
            report.owners.items(), key=lambda kv: -kv[1]
        ):
            if owner != project_key():
                lines.append(f"      {owner}  ({count} point(s))")
    else:
        lines.append(
            "  no points owned by another project were found, but unscoped points "
            "have no owner at all, so their origin cannot be proven."
        )
    if report.needs_migration and not report.is_shared:
        lines.append(
            "Ownership of the unscoped points cannot be proven, so a rebuild is "
            "NOT provably safe for other projects."
        )
    return lines


def migrate_collection(
    collection: str | None = None, *, rebuild_shared_collection: bool = False
) -> CollectionReport:
    """Inspect, and rebuild only when explicitly asked to.

    ``rebuild_shared_collection`` is required for anything destructive, and it
    is refused outright when the collection holds other projects' points: an
    operator asking to rebuild a shared collection still has to say so twice,
    because the second acknowledgement is the one that matters. This never runs
    as a side effect of indexing or asking a question.
    """
    report = inspect_collection(collection)
    if not report.reachable:
        logger.warning("Qdrant migration aborted: %s", report.error)
        return report
    if not report.needs_migration:
        return report

    if not rebuild_shared_collection:
        logger.warning(
            "Collection %s has %d unscoped point(s). Nothing was changed. "
            "Re-run with rebuild_shared_collection=True to rebuild.",
            report.collection,
            report.unscoped_points,
        )
        return report

    if report.is_shared:
        logger.error(
            "Refusing to rebuild %s: it holds points owned by %d other project(s). "
            "A full rebuild deletes the entire collection. Re-run with "
            "rebuild_shared_collection=True to proceed anyway.",
            report.collection,
            len({o for o in report.owners if o != project_key()}),
        )
        return report

    for line in describe_plan(report):
        logger.warning("%s", line)
    logger.warning(
        "Rebuilding shared collection %s from current files.", report.collection
    )
    from terminus.context.indexers.reindexer import full_reindex

    full_reindex(str(os.getcwd()))
    after = inspect_collection(collection)
    logger.warning(
        "Rebuild complete for %s: %d point(s), %d unscoped remaining.",
        after.collection,
        after.total_points,
        after.unscoped_points,
    )
    return after


def payload_indexes(client: Any, collection: str) -> list[str]:
    """Which payload fields Qdrant has an index on. Diagnostic only."""
    try:
        schema = client.get_collection(collection_name=collection).payload_schema
    except Exception:
        return []
    return sorted(str(key) for key in (schema or {}))
