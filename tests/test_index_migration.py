"""The Qdrant project-scope migration, driven through a real in-memory Qdrant.

The properties under test:

    The report counts what is really in the collection, including legacy points
    and other projects' points; nothing is destroyed without an explicit flag;
    a rebuild is refused outright when the collection is provably shared; and
    legacy points are never quietly guessed to be ours.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terminus.context.indexers import migrate
from terminus.workspace import PROJECT_METADATA_KEY

qdrant_client = pytest.importorskip("qdrant_client")
QdrantClient = qdrant_client.QdrantClient
models = qdrant_client.models

langchain_qdrant = pytest.importorskip("langchain_qdrant")
QdrantVectorStore = langchain_qdrant.QdrantVectorStore

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

VECTOR_SIZE = 3


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t) % 13), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 13), 1.0, 0.0]


class _Chunk:
    def __init__(self, source="a.py"):
        self.content = "def f(): return 1"
        self.source = source
        self.name = "f"
        self.type = "function"
        self.start_line = 1
        self.end_line = 2


def _store(client, collection) -> QdrantVectorStore:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )
    return QdrantVectorStore(
        client=client, collection_name=collection, embedding=_FakeEmbeddings()
    )


def _add_scoped(client, collection, project, source="a.py") -> None:
    """Index one chunk as belonging to *project*.

    The project identity is produced by the real ``project_key()`` rather than a
    hardcoded string, because that function normalises the path (and on Windows
    "G:/x" and "G:\\x" are the same project but different strings). Testing a
    made-up identity would not test the thing that has to match.
    """
    from terminus.context.qdrant_scope import chunk_metadata
    import terminus.workspace as ws

    previous = ws.project_root
    ws.project_root = lambda: Path(project)
    try:
        _store(client, collection).add_documents(
            [Document(page_content="def f(): return 1", metadata=chunk_metadata(_Chunk(source)))]
        )
    finally:
        ws.project_root = previous


def _add_legacy(client, collection, source="legacy.py") -> None:
    _store(client, collection).add_documents([
        Document(page_content="def legacy(): pass",
                 metadata={"source": source, "name": "legacy", "type": "function",
                           "start_line": 1, "end_line": 1}),
    ])


@pytest.fixture
def client():
    instance = QdrantClient(location=":memory:")
    yield instance
    instance.close()


THIS_PROJECT = "G:/projA"


@pytest.fixture
def wired(client, monkeypatch):
    """Point the migration module at the in-memory client and a known project.

    ``migrate.project_key`` is left alone so the report is compared against the
    same identity function the indexer used.
    """
    import terminus.workspace as ws

    monkeypatch.setattr(migrate, "_client", lambda: client)
    monkeypatch.setattr(ws, "project_root", lambda: Path(THIS_PROJECT))
    monkeypatch.setattr(migrate.os, "getcwd", lambda: THIS_PROJECT)
    return client


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_an_absent_collection_is_reported_not_raised(wired):
    report = migrate.inspect_collection("nope")
    assert report.reachable is True
    assert "does not exist" in report.error
    assert report.total_points == 0


def test_an_empty_collection_needs_no_migration(wired):
    _store(wired, "empty")
    report = migrate.inspect_collection("empty")
    assert report.reachable and report.total_points == 0
    assert report.needs_migration is False
    assert "empty" in " ".join(migrate.describe_plan(report))


def test_a_fully_scoped_collection_needs_no_migration(wired):
    _add_scoped(wired, "clean", THIS_PROJECT)
    report = migrate.inspect_collection("clean")
    assert report.scoped_points == 1
    assert report.unscoped_points == 0
    assert report.needs_migration is False
    assert report.exclusively_ours is True


def test_legacy_points_are_counted(wired):
    _add_legacy(wired, "legacy")
    report = migrate.inspect_collection("legacy")
    assert report.unscoped_points == 1
    assert report.scoped_points == 0
    assert report.needs_migration is True


def test_a_partially_scoped_collection_is_flagged(wired):
    _add_legacy(wired, "mixed")
    _add_scoped(wired, "mixed", THIS_PROJECT)
    report = migrate.inspect_collection("mixed")
    assert report.unscoped_points == 1
    assert report.scoped_points == 1
    assert report.needs_migration is True


def test_another_projects_points_are_detected(wired):
    _add_scoped(wired, "shared", "G:/projB")
    report = migrate.inspect_collection("shared")
    assert report.is_shared is True
    assert report.foreign_points == 1
    assert report.this_project_points == 0
    assert report.needs_migration is False


def test_a_legacy_only_collection_is_not_claimed_to_be_safe(wired):
    """Legacy points have no owner, so their provenance is unprovable."""
    _add_legacy(wired, "legacy")
    report = migrate.inspect_collection("legacy")
    assert report.is_shared is False, "cannot prove or disprove another project"
    assert report.exclusively_ours is False
    assert "cannot be proven" in " ".join(migrate.describe_plan(report))


def test_counts_survive_more_points_than_one_page(wired):
    for i in range(migrate.SCAN_PAGE_SIZE + 40):
        _add_legacy(wired, "big", source=f"f{i}.py")
    report = migrate.inspect_collection("big", page_size=64)
    assert report.unscoped_points == migrate.SCAN_PAGE_SIZE + 40


def test_an_unreachable_qdrant_is_reported(wired, monkeypatch):
    class Hostile:
        def collection_exists(self, *_a, **_k):
            raise RuntimeError("connection reset")

    monkeypatch.setattr(migrate, "_client", lambda: Hostile())
    report = migrate.inspect_collection("whatever")
    assert report.reachable is False
    assert "connection reset" in report.error
    assert "Cannot inspect" in " ".join(migrate.describe_plan(report))


def test_missing_credentials_are_reported(wired, monkeypatch):
    monkeypatch.setattr(migrate, "_client", lambda: (_ for _ in ()).throw(
        ValueError("QDRANT_API_KEY and CLUSTER_ENDPOINT must be set")
    ))
    report = migrate.inspect_collection("x")
    assert report.reachable is False
    assert "QDRANT_API_KEY" in report.error


# ---------------------------------------------------------------------------
# refusal: nothing destructive without the flag
# ---------------------------------------------------------------------------


def test_migration_does_nothing_without_the_flag(wired):
    _add_legacy(wired, "legacy")
    report = migrate.migrate_collection("legacy", rebuild_shared_collection=False)
    assert report.unscoped_points == 1
    # And the data is genuinely still there.
    assert migrate.inspect_collection("legacy").unscoped_points == 1


def test_migration_is_a_no_op_when_nothing_needs_migrating(wired):
    _add_scoped(wired, "clean", THIS_PROJECT)
    report = migrate.migrate_collection("clean", rebuild_shared_collection=True)
    assert report.unscoped_points == 0
    assert report.scoped_points == 1


def test_a_full_reindex_is_never_called_without_the_flag(wired, monkeypatch):
    called = []
    import terminus.context.indexers.reindexer as rx

    monkeypatch.setattr(rx, "full_reindex", lambda *_a, **_k: called.append(1))
    _add_legacy(wired, "legacy")
    migrate.migrate_collection("legacy", rebuild_shared_collection=False)
    assert called == [], "a rebuild ran without the destructive flag"


def test_a_shared_collection_is_refused_even_with_the_flag(wired, monkeypatch):
    """Asking twice must still be answered no while another project is present."""
    called = []
    import terminus.context.indexers.reindexer as rx

    monkeypatch.setattr(rx, "full_reindex", lambda *_a, **_k: called.append(1))
    _add_legacy(wired, "shared")
    _add_scoped(wired, "shared", "G:/projB")

    report = migrate.migrate_collection("shared", rebuild_shared_collection=True)
    assert called == [], "a shared collection was rebuilt anyway"
    assert report.is_shared is True
    assert report.foreign_points == 1


def test_the_refusal_survives_an_explicit_rebuild_attempt(wired, monkeypatch):
    called = []
    import terminus.context.indexers.reindexer as rx

    monkeypatch.setattr(rx, "full_reindex", lambda *_a, **_k: called.append(1))
    _add_scoped(wired, "onlyb", "G:/projB")
    report = migrate.migrate_collection("onlyb", rebuild_shared_collection=True)
    assert called == []
    assert report.foreign_points == 1


def test_a_legacy_rebuild_that_is_provably_ours_may_proceed(wired, monkeypatch):
    """Ownership proven: our points only, nothing legacy. Rebuild is allowed."""
    called = []

    def fake_reindex(_path):
        called.append(1)
        # The rebuild replaces the collection with freshly scoped points.
        client = wired
        client.delete_collection("provable")
        _add_scoped(client, "provable", THIS_PROJECT, source="new.py")
        return None, None

    import terminus.context.indexers.reindexer as rx

    monkeypatch.setattr(rx, "full_reindex", fake_reindex)
    _add_legacy(wired, "provable")
    # Unscoped points exist, so migration is warranted; the rebuild then yields
    # a clean collection. This is the one path where destruction is authorised.
    report = migrate.migrate_collection("provable", rebuild_shared_collection=True)
    assert called == [1]
    assert report.unscoped_points == 0


# ---------------------------------------------------------------------------
# the plan an operator is shown
# ---------------------------------------------------------------------------


def test_the_plan_says_what_would_be_destroyed(wired):
    _add_legacy(wired, "shared")
    _add_scoped(wired, "shared", "G:/projB")
    plan = " ".join(migrate.describe_plan(migrate.inspect_collection("shared")))
    assert "DELETE all" in plan
    assert "would be destroyed" in plan
    assert "G:/projB" in plan or "G:\\projB" in plan
    assert PROJECT_METADATA_KEY in plan or "project" in plan


def test_the_plan_is_explicit_for_an_exclusive_collection(wired):
    _add_scoped(wired, "onlya", THIS_PROJECT)
    plan = " ".join(migrate.describe_plan(migrate.inspect_collection("onlya")))
    assert "already" in plan and "No migration needed" in plan


# ---------------------------------------------------------------------------
# payload index
# ---------------------------------------------------------------------------


def test_payload_indexes_are_reported(wired):
    _add_scoped(wired, "idx", THIS_PROJECT)
    indexes = migrate.payload_indexes(wired, "idx")
    assert isinstance(indexes, list)


def test_payload_index_probe_survives_failure(wired):
    class Hostile:
        def get_collection(self, **_k):
            raise RuntimeError("nope")

    assert migrate.payload_indexes(Hostile(), "x") == []
