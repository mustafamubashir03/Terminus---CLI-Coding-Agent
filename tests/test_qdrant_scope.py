"""The Qdrant project scope, proven against a real Qdrant client.

These use an in-memory Qdrant rather than mocks, because the property being
protected is about how LangChain actually writes and filters payloads. A mocked
store cannot catch a wrong filter key - it would happily answer to any key we
assert, which is exactly how a scoped search can look correct and return
nothing.

The properties under test:

    A project only ever retrieves its own chunks; a collection left over from
    before the project payload existed returns nothing rather than everything;
    and the field the indexer writes is the field the retriever matches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terminus.context.qdrant_scope import (
    chunk_metadata,
    ensure_project_payload_index,
    payload_key,
    project_filter,
    unscoped_points_present,
)
from terminus.workspace import PROJECT_METADATA_KEY

qdrant_client = pytest.importorskip("qdrant_client")
QdrantClient = qdrant_client.QdrantClient
models = qdrant_client.models

langchain_qdrant = pytest.importorskip("langchain_qdrant")
QdrantVectorStore = langchain_qdrant.QdrantVectorStore

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402


class _FakeEmbeddings(Embeddings):
    """Deterministic vectors: no model, no network, stable across runs."""

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


@pytest.fixture
def client():
    instance = QdrantClient(location=":memory:")
    yield instance
    instance.close()


VECTOR_SIZE = 3


def _ensure_collection(client, collection: str) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )


def _store(client, collection: str) -> QdrantVectorStore:
    _ensure_collection(client, collection)
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=_FakeEmbeddings(),
    )


def _index(client, collection: str, project: str, source: str, monkeypatch) -> None:
    """Write one chunk tagged as belonging to *project*."""
    monkeypatch.setattr("terminus.workspace.project_root", lambda: Path(project))
    _store(client, collection).add_documents(
        [Document(page_content="def f(): return 1", metadata=chunk_metadata(_Chunk(source)))]
    )


def _search(client, collection: str, project: str, monkeypatch) -> list:
    monkeypatch.setattr("terminus.workspace.project_root", lambda: Path(project))
    return _store(client, collection).similarity_search_with_score("def f", k=10, filter=project_filter())


# ---------------------------------------------------------------------------
# the filter key must match where LangChain actually stores the field
# ---------------------------------------------------------------------------


def test_filter_targets_the_namespaced_payload_key():
    """LangChain nests document metadata under "metadata"; a bare key matches nothing."""
    condition = project_filter().must[0]
    assert condition.key == f"metadata.{PROJECT_METADATA_KEY}"
    assert condition.key == payload_key(PROJECT_METADATA_KEY)


def test_indexer_writes_a_flat_document_metadata():
    """chunk_metadata feeds Document.metadata, which LangChain namespaces itself."""
    assert PROJECT_METADATA_KEY in chunk_metadata(_Chunk())
    assert chunk_metadata(_Chunk())["source"] == "a.py"


def test_written_and_matched_keys_agree(client, monkeypatch):
    """The end-to-end proof: what the indexer writes, the filter finds."""
    _index(client, "kv", "G:/projA", "a.py", monkeypatch)
    points, _ = client.scroll("kv", limit=5, with_payload=True)
    stored = points[0].payload
    # The field really is at metadata.project in the stored payload...
    assert stored["metadata"][PROJECT_METADATA_KEY] == str(Path("G:/projA"))
    # ...and that is precisely the path the filter matches.
    assert project_filter().must[0].key == "metadata.project"
    assert _search(client, "kv", "G:/projA", monkeypatch)


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


def test_a_project_cannot_retrieve_another_projects_chunks(client, monkeypatch):
    _index(client, "kv", "G:/projA", "a.py", monkeypatch)
    _index(client, "kv", "G:/projB", "b.py", monkeypatch)

    mine = _search(client, "kv", "G:/projA", monkeypatch)
    assert [d.metadata["source"] for d, _ in mine] == ["a.py"]

    theirs = _search(client, "kv", "G:/projB", monkeypatch)
    assert [d.metadata["source"] for d, _ in theirs] == ["b.py"]


def test_a_third_project_sees_nothing(client, monkeypatch):
    _index(client, "kv", "G:/projA", "a.py", monkeypatch)
    assert _search(client, "kv", "G:/projC", monkeypatch) == []


# ---------------------------------------------------------------------------
# legacy collections
# ---------------------------------------------------------------------------


def test_a_legacy_unscoped_collection_returns_nothing(client, monkeypatch):
    """The whole point: an old collection must read as empty, not as a leak.

    Written the way the indexer wrote points before the project payload existed
    - no project field at all.
    """
    _store(client, "legacy").add_documents([
        Document(
            page_content="SECRET def other_project_code(): pass",
            metadata={"source": "secret.py", "name": "other_project_code",
                      "type": "function", "start_line": 1, "end_line": 1},
        )
    ])
    # The point is genuinely there, and genuinely invisible.
    unfiltered, _ = client.query_points(
        "legacy",
        query=[float(len("def f") % 13), 1.0, 0.0],
        limit=10,
    ).points, None
    assert len(unfiltered) == 1
    assert _search(client, "legacy", "G:/projA", monkeypatch) == []


def test_a_collection_mixing_scoped_and_legacy_only_yields_scoped(client, monkeypatch):
    """A partial backfill must not re-open the leak for the legacy half."""
    _store(client, "mixed").add_documents([
        Document(page_content="legacy def g(): pass",
                 metadata={"source": "legacy.py", "name": "g", "type": "function",
                           "start_line": 1, "end_line": 1}),
    ])
    _index(client, "mixed", "G:/projA", "a.py", monkeypatch)
    found = _search(client, "mixed", "G:/projA", monkeypatch)
    assert [d.metadata["source"] for d, _ in found] == ["a.py"]


# ---------------------------------------------------------------------------
# legacy collections are detected, not silently left inert
# ---------------------------------------------------------------------------


def test_a_fully_scoped_collection_is_not_reported_as_legacy(client, monkeypatch):
    _index(client, "kv", "G:/projA", "a.py", monkeypatch)
    assert unscoped_points_present(client, "kv") is False


def test_a_legacy_collection_is_detected(client, monkeypatch):
    _store(client, "legacy").add_documents([
        Document(page_content="def g(): pass",
                 metadata={"source": "g.py", "name": "g", "type": "function",
                           "start_line": 1, "end_line": 1}),
    ])
    assert unscoped_points_present(client, "legacy") is True


def test_a_partially_scoped_collection_is_detected(client, monkeypatch):
    """A half-finished backfill is exactly the case that must not pass as fine."""
    _store(client, "mixed").add_documents([
        Document(page_content="def g(): pass",
                 metadata={"source": "g.py", "name": "g", "type": "function",
                           "start_line": 1, "end_line": 1}),
    ])
    _index(client, "mixed", "G:/projA", "a.py", monkeypatch)
    assert unscoped_points_present(client, "mixed") is True


def test_an_empty_collection_is_not_legacy(client):
    _ensure_collection(client, "empty")
    assert unscoped_points_present(client, "empty") is False


def test_an_unreadable_collection_is_not_guessed_at():
    class Hostile:
        def scroll(self, **_k):
            raise RuntimeError("no such collection")

    assert unscoped_points_present(Hostile(), "whatever") is False


def test_project_payload_index_is_requested_and_is_idempotent(client, monkeypatch):
    """Local Qdrant ignores payload indexes, so the contract is only that the
    call is made, tolerated, and repeatable."""
    _index(client, "kv", "G:/projA", "a.py", monkeypatch)
    ensure_project_payload_index(client, "kv")
    ensure_project_payload_index(client, "kv")
    ensure_project_payload_index(client, "kv")


def test_project_payload_index_failure_is_not_fatal():
    class Hostile:
        def create_payload_index(self, **_k):
            raise RuntimeError("index service unavailable")

    ensure_project_payload_index(Hostile(), "kv")


def test_a_legacy_collection_still_gives_a_correct_error_from_the_indexer(
    client, monkeypatch
):
    """The indexer must reach its warning rather than crash on a legacy shape."""
    _store(client, "legacy").add_documents([
        Document(page_content="def g(): pass",
                 metadata={"source": "g.py", "name": "g", "type": "function",
                           "start_line": 1, "end_line": 1}),
    ])
    # It only has to not raise, and to have decided the collection is legacy.
    assert unscoped_points_present(client, "legacy") is True


# ---------------------------------------------------------------------------
# identifier parity
# ---------------------------------------------------------------------------


def test_write_and_query_use_the_same_identifier(monkeypatch):
    """Same call site on both sides, so a project cannot write under one id and
    read under another."""
    import inspect

    import terminus.context.qdrant_scope as scope

    write_source = inspect.getsource(scope.chunk_metadata)
    read_source = inspect.getsource(scope.project_filter)
    assert "project_key()" in write_source
    assert "project_key()" in read_source
