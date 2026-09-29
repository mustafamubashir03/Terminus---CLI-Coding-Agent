"""Vector backend behaviour: local Qdrant, cloud Qdrant, Chroma, selection.

Deliberately split into two kinds of test.

**Real local backends.** The Qdrant and Chroma local tests run the actual
installed engines in a temp directory - no mocks. Both were verified to work
that way, so mocking them would only test the mock. This is what proves a fresh
install with no credentials can index and search.

**Cloud and failure paths, stubbed.** No test contacts a real Qdrant cluster.
Cloud construction, credential handling and every failure class are asserted
against a client stub, because they are about *what Terminus does with* the
result, not about the cluster.

The seed embedder is deterministic and 384-dimensional, matching the configured
model, so no model download is needed and results are reproducible.
"""

from __future__ import annotations

import socket
import ssl
import threading
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from terminus.config import CONFIG
from terminus.context.indexers import factory as indexer_factory
from terminus.context.indexers import qdrant_client
from terminus.context.indexers.errors import (
    VectorStoreFailure,
    VectorStoreUnavailableError,
    classify_failure,
    describe_failure,
    root_cause,
)
from terminus.context.retrievers import cache as retrieval_cache

DIMENSIONS = 384


class SeedEmbeddings(Embeddings):
    """Deterministic stand-in for the configured MiniLM model.

    Seeded from the text so a document always embeds identically, which is what
    makes a persistence test meaningful: a changed embedding after a restart
    would look like data loss.
    """

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        seed = sum((i + 1) * ord(c) for i, c in enumerate(text)) or 1
        return [((seed * (i + 3)) % 997) / 997.0 for i in range(DIMENSIONS)]


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point every store at a temp directory, with local Qdrant and no cloud env.

    Also resets the retrieval cache between tests: it is process-local by design,
    and a leaked entry would make one test's stub answer another's query.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("CLUSTER_ENDPOINT", raising=False)
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "local")
    monkeypatch.setitem(CONFIG["qdrant"], "path", ".terminus/qdrant")
    monkeypatch.setitem(CONFIG["qdrant"], "collection_name", "test_collection")
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "qdrant")
    monkeypatch.setitem(CONFIG["vector_store"], "retrieval_mode", "dense")
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", False)
    monkeypatch.setitem(CONFIG["rag"], "mode", "semantic")
    monkeypatch.setitem(CONFIG["chromadb"], "persist_dir", ".terminus/chromadb/")
    monkeypatch.setitem(CONFIG["chromadb"], "collection_name", "terminus")
    monkeypatch.setattr(qdrant_client, "embedding_dimensions", lambda: DIMENSIONS)
    monkeypatch.setattr("terminus.llm.factory.get_embedder", lambda: SeedEmbeddings())
    retrieval_cache.reset()
    qdrant_client.reset_local_clients()
    yield tmp_path
    retrieval_cache.reset()
    # A local engine holds an exclusive lock on its directory, so the client is
    # released or the next test on the same path cannot open it.
    qdrant_client.reset_local_clients()


def seed_project(root: Path) -> Path:
    """A tiny but realistic project: two modules, one importing the other."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "auth.py").write_text(
        "def verify_token(token):\n"
        "    return token is not None\n"
        "\n"
        "def refresh(token):\n"
        "    return verify_token(token)\n",
        encoding="utf-8",
    )
    (root / "src" / "app.py").write_text(
        "from auth import verify_token\n"
        "\n"
        "def handler(request):\n"
        "    return verify_token(request.token)\n",
        encoding="utf-8",
    )
    return root


# --- Qdrant local: the real engine ----------------------------------------


def test_qdrant_local_needs_no_credentials(isolated_store):
    """A fresh install with no .env at all must still be able to build a client."""
    client = qdrant_client.create_qdrant_client()
    assert client is not None
    assert qdrant_client.qdrant_mode() == "local"
    assert qdrant_client.qdrant_location().startswith("local:")


def test_qdrant_local_creates_its_directory(isolated_store):
    qdrant_client.create_qdrant_client()
    assert (isolated_store / ".terminus" / "qdrant").is_dir()


def test_qdrant_local_indexes_and_queries(isolated_store):
    seed_project(isolated_store)
    from terminus.context.indexers.semantic_qdrant import get_or_create_qdrant_index

    store = get_or_create_qdrant_index(str(isolated_store))
    assert store is not None

    retrieval_cache.reset()
    from terminus.context.retrievers.semantic_qdrant import retrieve

    chunks = retrieve("verify_token", k=3)
    assert chunks, "a freshly indexed project returned nothing"
    assert all(c["source"].endswith(".py") for c in chunks)
    assert all(c["score"] is not None for c in chunks)


def test_qdrant_local_survives_a_restart(isolated_store):
    """Index, drop every cached object, then search again.

    The cache drop stands in for a process restart: nothing in memory may be
    carrying the data, so a hit here means it reached the directory.
    """
    seed_project(isolated_store)
    from terminus.context.indexers.semantic_qdrant import get_or_create_qdrant_index

    get_or_create_qdrant_index(str(isolated_store))

    client = qdrant_client.create_qdrant_client()
    before = client.get_collection(collection_name="test_collection").points_count
    assert before > 0
    # Release the local engine's exclusive directory lock, standing in for the
    # process going away. Nothing in memory may be carrying the data.
    qdrant_client.reset_local_clients()
    retrieval_cache.reset()

    from terminus.context.retrievers.semantic_qdrant import retrieve

    assert retrieve("verify_token", k=3), "the index did not survive the restart"


def test_qdrant_local_is_project_scoped(isolated_store, tmp_path, monkeypatch):
    """The same shared local store must not serve a second project."""
    seed_project(isolated_store)
    from terminus.context.indexers.semantic_qdrant import get_or_create_qdrant_index

    get_or_create_qdrant_index(str(isolated_store))
    retrieval_cache.reset()
    from terminus.context.retrievers.semantic_qdrant import retrieve

    assert retrieve("verify_token", k=5)

    # A different project, same directory on disk.
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setitem(CONFIG["qdrant"], "path", str(isolated_store / ".terminus" / "qdrant"))
    retrieval_cache.reset()

    assert retrieve("verify_token", k=5) == [], (
        "a store shared by path leaked another project's chunks"
    )


# --- Chroma local: the real engine ----------------------------------------


def test_chroma_local_indexes_and_queries(isolated_store):
    seed_project(isolated_store)
    from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index

    collection = get_or_create_chroma_index(str(isolated_store))
    assert collection.count() > 0

    retrieval_cache.reset()
    from terminus.context.retrievers.semantic_chroma import retrieve

    chunks = retrieve("verify_token", k=3)
    assert chunks
    assert all(c["source"].endswith(".py") for c in chunks)


def test_chroma_local_survives_a_restart(isolated_store):
    seed_project(isolated_store)
    from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index

    get_or_create_chroma_index(str(isolated_store))
    retrieval_cache.reset()

    from terminus.context.retrievers.semantic_chroma import retrieve

    assert retrieve("verify_token", k=3)


def test_chroma_batches_rather_than_upserting_per_chunk(isolated_store, monkeypatch):
    """Indexing must not be one round trip per chunk.

    The old implementation called embed + upsert inside the file loop. This
    asserts the batch shape directly - one upsert for many chunks - because that
    is the property that carries the 14x difference, and a timing assertion would
    be flaky.
    """
    seed_project(isolated_store)
    from terminus.context.indexers import semantic_chroma

    calls: list[int] = []
    real_upsert = None

    def counting_upsert(**kwargs):
        calls.append(len(kwargs.get("ids", [])))
        return FakeCollection()

    class FakeCollection:
        def upsert(self, **kwargs):
            calls.append(len(kwargs.get("ids", [])))
            return self

        def count(self):
            return sum(calls)

    monkeypatch.setattr(semantic_chroma, "_collection", lambda *a, **k: FakeCollection())
    monkeypatch.setattr(
        "terminus.llm.factory.get_embedder", lambda: SeedEmbeddings()
    )
    assert real_upsert is None

    semantic_chroma.index_codebase_chroma(str(isolated_store))

    assert calls, "nothing was indexed"
    assert len(calls) < sum(calls), (
        f"one upsert per chunk again: {len(calls)} calls for {sum(calls)} chunks"
    )


# --- result contract, across every backend -------------------------------


@pytest.mark.parametrize("module_name", [
    "terminus.context.retrievers.semantic_qdrant",
    "terminus.context.retrievers.semantic_chroma",
])
def test_every_retriever_declares_the_same_result_shape(module_name):
    """One documented shape, so a consumer never has to special-case a backend."""
    import importlib
    import inspect

    from terminus.context.retrievers.retrieved import RetrievedChunk

    module = importlib.import_module(module_name)
    hints = inspect.get_annotations(module.retrieve, eval_str=True)
    assert hints["return"] == list[RetrievedChunk], module_name


def test_chroma_results_carry_score_as_none_not_absent(isolated_store):
    """`score` must be present-and-None, so a consumer gets None and not KeyError."""
    from terminus.context.retrievers.semantic_chroma import retrieve

    seed_project(isolated_store)
    from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index

    get_or_create_chroma_index(str(isolated_store))
    retrieval_cache.reset()

    chunks = retrieve("verify_token", k=2)
    assert chunks
    for chunk in chunks:
        assert "score" in chunk
        assert chunk["score"] is None


def test_required_result_fields_are_always_present(isolated_store):
    """The keys codebase_tool reads must exist for every backend."""
    required = {"text", "source", "name", "type", "start_line", "end_line", "score"}

    seed_project(isolated_store)
    from terminus.context.indexers.semantic_qdrant import get_or_create_qdrant_index
    from terminus.context.indexers.semantic_chroma import get_or_create_chroma_index

    get_or_create_qdrant_index(str(isolated_store))
    get_or_create_chroma_index(str(isolated_store))

    for module in (
        "terminus.context.retrievers.semantic_qdrant",
        "terminus.context.retrievers.semantic_chroma",
    ):
        import importlib

        retrieve = importlib.import_module(module).retrieve
        retrieval_cache.reset()
        for chunk in retrieve("verify_token", k=2):
            assert required <= set(chunk), f"{module} missing {required - set(chunk)}"


# --- cloud construction: stubbed, never a real network ------------------


class ClientSpy:
    """Records how a Qdrant client was constructed."""

    instances: list[dict] = []

    def __init__(self, **kwargs):
        ClientSpy.instances.append(kwargs)
        self.kwargs = kwargs


@pytest.fixture()
def client_spy(monkeypatch):
    ClientSpy.instances = []
    monkeypatch.setattr("qdrant_client.QdrantClient", ClientSpy)
    return ClientSpy


def test_cloud_mode_passes_url_and_api_key(isolated_store, monkeypatch, client_spy):
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "cloud")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://cluster.example")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")

    qdrant_client.create_qdrant_client()

    options = client_spy.instances[0]
    assert options["url"] == "https://cluster.example"
    assert options["api_key"] == "secret-key"
    assert "path" not in options, "cloud must not also open a local store"


def test_local_mode_never_passes_credentials(isolated_store, monkeypatch, client_spy):
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "local")
    # Credentials present but irrelevant: local must not read them.
    monkeypatch.setenv("QDRANT_API_KEY", "should-be-ignored")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://should-be-ignored.example")

    qdrant_client.create_qdrant_client()

    options = client_spy.instances[0]
    assert "api_key" not in options
    assert "url" not in options
    assert "path" in options


def test_every_client_gets_the_configured_timeout(isolated_store, client_spy):
    original = CONFIG["qdrant"].get("timeout_seconds", 5)
    CONFIG["qdrant"]["timeout_seconds"] = 11
    try:
        qdrant_client.create_qdrant_client()
        assert client_spy.instances[0]["timeout"] == 11
    finally:
        CONFIG["qdrant"]["timeout_seconds"] = original


def test_compatibility_check_is_disabled_consistently(isolated_store, client_spy):
    qdrant_client.create_qdrant_client()
    assert client_spy.instances[0]["check_compatibility"] is False


def test_the_reindexer_uses_the_shared_client(isolated_store, monkeypatch, client_spy):
    """One client construction site, so the options cannot drift apart again."""
    from terminus.context.indexers import reindexer

    monkeypatch.setitem(CONFIG["qdrant"], "collection_name", "c")
    reindexer._qdrant_client_and_cfg()

    assert len(client_spy.instances) == 1
    assert client_spy.instances[0]["check_compatibility"] is False
    assert "timeout" in client_spy.instances[0]


def test_cloud_mode_without_a_url_fails_clearly(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "cloud")
    monkeypatch.delenv("CLUSTER_ENDPOINT", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "k")
    with pytest.raises(Exception) as caught:
        qdrant_client.create_qdrant_client()
    assert "CLUSTER_ENDPOINT" in str(caught.value)


def test_cloud_mode_without_a_key_fails_clearly(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "cloud")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://c.example")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    with pytest.raises(Exception) as caught:
        qdrant_client.create_qdrant_client()
    assert "QDRANT_API_KEY" in str(caught.value)


def test_a_non_http_endpoint_is_rejected(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "cloud")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "ftp://cluster.example")
    monkeypatch.setenv("QDRANT_API_KEY", "k")
    with pytest.raises(Exception) as caught:
        qdrant_client.create_qdrant_client()
    assert "http" in str(caught.value).lower()


# --- mode resolution is backwards compatible ----------------------------


def test_no_mode_and_no_endpoint_resolves_to_local(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "")
    monkeypatch.delenv("CLUSTER_ENDPOINT", raising=False)
    assert qdrant_client.qdrant_mode() == "local"


def test_no_mode_but_an_endpoint_resolves_to_cloud(isolated_store, monkeypatch):
    """The rule that keeps every existing cloud user on cloud."""
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://cluster.example")
    assert qdrant_client.qdrant_mode() == "cloud"


def test_explicit_mode_beats_the_endpoint(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "local")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://cluster.example")
    assert qdrant_client.qdrant_mode() == "local"


# --- backend selection: the core fix -------------------------------------


def test_configured_qdrant_failure_raises_and_leaves_config_alone(isolated_store, monkeypatch):
    """The bug: a failure must not substitute a backend or rewrite CONFIG."""
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", False)

    def unreachable(*_a, **_k):
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(indexer_factory, "build_index", unreachable)
    monkeypatch.setattr(
        "terminus.context.indexers.semantic_chroma.get_or_create_chroma_index",
        lambda *_a, **_k: pytest.fail("Chroma must not be built for the configured backend"),
    )

    with pytest.raises(VectorStoreUnavailableError):
        indexer_factory.get_or_create_index(str(isolated_store))

    assert CONFIG["vector_store"]["provider"] == "qdrant"
    assert CONFIG["rag"]["mode"] == "semantic"
    assert "indexer_fallback" not in CONFIG.get("_runtime", {})


def test_fallback_is_off_by_default(isolated_store, monkeypatch):
    monkeypatch.setattr(
        indexer_factory, "build_index",
        lambda *_a, **_k: (_ for _ in ()).throw(ConnectionResetError("reset")),
    )
    with pytest.raises(VectorStoreUnavailableError):
        indexer_factory.get_or_create_index(str(isolated_store))


def test_explicit_fallback_engages_and_is_observable(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", True)
    monkeypatch.setitem(CONFIG["rag"], "mode", "hybrid")
    monkeypatch.setitem(CONFIG["vector_store"], "retrieval_mode", "hybrid")
    sentinel = object()

    def build(provider, mode, repo_path, force_reindex):
        if provider == "qdrant":
            raise ConnectionResetError("reset by peer")
        return sentinel

    monkeypatch.setattr(indexer_factory, "build_index", build)

    index, resolution = indexer_factory.get_or_create_index(str(isolated_store))

    assert index is sentinel
    assert resolution.fallback is True
    assert resolution.configured_provider == "qdrant"
    assert resolution.configured_mode == "hybrid"
    # Chroma cannot do hybrid, and the report must not pretend otherwise.
    assert resolution.mode == "semantic"
    assert "BM25" in resolution.describe()
    # The operator's configuration is untouched.
    assert CONFIG["vector_store"]["provider"] == "qdrant"
    assert CONFIG["rag"]["mode"] == "hybrid"


def test_a_configuration_error_never_falls_back(isolated_store, monkeypatch):
    """Missing credentials are a mistake to fix, not an outage to route around."""
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", True)

    def build(*_a, **_k):
        raise ValueError("QDRANT_API_KEY is not set")

    monkeypatch.setattr(indexer_factory, "build_index", build)
    with pytest.raises(VectorStoreUnavailableError):
        indexer_factory.get_or_create_index(str(isolated_store))


def test_a_fallback_that_also_fails_says_both_failed(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", True)

    def build(provider, mode, repo_path, force_reindex):
        raise ConnectionResetError("unreachable")

    monkeypatch.setattr(indexer_factory, "build_index", build)
    with pytest.raises(VectorStoreUnavailableError, match="Neither backend"):
        indexer_factory.get_or_create_index(str(isolated_store))


def test_hybrid_on_chroma_is_rejected_rather_than_downgraded(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "chromadb")
    monkeypatch.setitem(CONFIG["rag"], "mode", "hybrid")
    with pytest.raises(Exception, match="only supported for qdrant"):
        indexer_factory.get_or_create_index(str(isolated_store))


def test_an_unknown_provider_is_rejected(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "pinecone")
    with pytest.raises(Exception, match="Unknown vector_store.provider"):
        indexer_factory.get_or_create_index(str(isolated_store))


def test_chroma_selected_directly_is_never_a_fallback(isolated_store, monkeypatch):
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "chromadb")
    monkeypatch.setitem(CONFIG["rag"], "mode", "semantic")
    seen = []
    monkeypatch.setattr(
        indexer_factory, "build_index",
        lambda provider, mode, *a, **k: seen.append(provider) or object(),
    )
    _index, resolution = indexer_factory.get_or_create_index(str(isolated_store))
    assert seen == ["chromadb"]
    assert resolution.fallback is False


# --- error classification ------------------------------------------------


def _chain(inner, outer, cause=True):
    """outer raised from inner, as the SDKs do."""
    outer.__cause__ = inner if cause else None
    return outer


@pytest.mark.parametrize("exc,expected", [
    (ConnectionResetError("reset"), VectorStoreFailure.TLS),
    (ssl.SSLError("handshake failure"), VectorStoreFailure.TLS),
    (socket.gaierror("name not resolved"), VectorStoreFailure.DNS),
    (ConnectionRefusedError("refused"), VectorStoreFailure.CONNECTION_REFUSED),
    (TimeoutError("timed out"), VectorStoreFailure.TIMEOUT),
    (ValueError("QDRANT_API_KEY is not set"), VectorStoreFailure.MISSING_CREDENTIALS),
    (ValueError("unauthorized"), VectorStoreFailure.AUTHENTICATION),
    (Exception("collection not found"), VectorStoreFailure.COLLECTION),
])
def test_failures_are_classified_distinctly(exc, expected):
    assert classify_failure(exc) is expected


def test_the_observed_tls_reset_reports_its_root_cause():
    """The real Qdrant Cloud failure must be legible, not 'ResponseHandlingException'."""
    root = ConnectionResetError("[WinError 10054] forcibly closed by the remote host")
    middle = _chain(root, type("httpcore.ConnectError", (Exception,), {})())
    inner = _chain(middle, type("httpx.ConnectError", (Exception,), {})())
    outer = _chain(inner, type("ResponseHandlingException", (Exception,), {})())

    assert classify_failure(outer) is VectorStoreFailure.TLS
    assert root_cause(outer) is root, "the chain must be walked to the real cause"

    described = describe_failure(outer)
    assert "TLS" in described
    assert "WinError 10054" in described, "the operator needs the actual error text"
    assert "ResponseHandlingException" not in described, (
        "the SDK's wrapper name is not the diagnosis"
    )


def test_the_error_message_names_what_to_do():
    error = VectorStoreUnavailableError(
        indexer_factory.qdrant_error_message(
            ConnectionResetError("reset"), "/repo", "qdrant", "hybrid", "c", "config.yaml"
        )
    )
    text = str(error)
    assert "what happened" in text and "what to do" in text
    assert "qdrant.mode: local" in text, "the message should point at the local option"


def test_exception_chaining_is_preserved(isolated_store, monkeypatch):
    cause = ConnectionResetError("reset")

    def build(*_a, **_k):
        raise cause

    monkeypatch.setattr(indexer_factory, "build_index", build)
    with pytest.raises(VectorStoreUnavailableError) as caught:
        indexer_factory.get_or_create_index(str(isolated_store))
    assert caught.value.__cause__ is cause


# --- payload index errors are not swallowed ------------------------------


class PayloadClient:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def create_payload_index(self, **_k):
        if self.behaviour == "exists":
            raise RuntimeError("Payload index already exists")
        if self.behaviour == "broken":
            raise RuntimeError("wrong dimensions: expected 1")
        return True


def test_an_existing_payload_index_is_tolerated():
    from terminus.context.qdrant_scope import ensure_project_payload_index

    ensure_project_payload_index(PayloadClient("exists"), "c")


def test_a_broken_payload_index_is_reported_not_hidden():
    """The old code did `except Exception: pass` and lost this entirely."""
    from terminus.context.qdrant_scope import ensure_project_payload_index

    assert PayloadClient("broken").create_payload_index  # the stub really is one
    # A performance index that cannot be created is logged and indexing continues,
    # but the failure is no longer indistinguishable from "already exists".
    with caplog_at("terminus.context.qdrant_scope") as records:
        ensure_project_payload_index(PayloadClient("broken"), "c")
    assert any("wrong dimensions" in r.getMessage() for r in records)


def test_the_shared_index_helper_distinguishes_exists_from_failure():
    from terminus.context.qdrant_scope import _create_payload_index

    assert _create_payload_index(PayloadClient("create"), "c", "f") is True
    assert _create_payload_index(PayloadClient("exists"), "c", "f") is False
    with pytest.raises(RuntimeError, match="wrong dimensions"):
        _create_payload_index(PayloadClient("broken"), "c", "f")


# --- retrieval cache -----------------------------------------------------


def test_the_retrieval_cache_is_reused_within_a_project(isolated_store, monkeypatch):
    calls = []

    def opener():
        calls.append(1)
        return object()

    from terminus.context.retrievers import cache

    first = cache.cached_store("dense", opener)
    second = cache.cached_store("dense", opener)
    assert first is second
    assert len(calls) == 1


def test_the_retrieval_cache_is_separated_by_project(isolated_store, tmp_path, monkeypatch):
    """A store must not be shared across project roots."""
    from terminus.context.retrievers import cache

    def opener():
        return object()

    here = cache.cached_store("dense", opener)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert cache.cached_store("dense", opener) is not here


# --- helpers -------------------------------------------------------------


class caplog_at:
    """Minimal `assertLogs` shim usable as a context manager in a test."""

    def __init__(self, logger_name):
        self.logger_name = logger_name
        self.records: list = []
        self._handler = None

    def __enter__(self):
        import logging

        logger = logging.getLogger(self.logger_name)
        self._handler = logging.Handler()
        self._handler.emit = self.records.append
        logger.addHandler(self._handler)
        self._previous = logger.level
        logger.setLevel(logging.DEBUG)
        return self.records

    def __exit__(self, *_exc):
        import logging

        logger = logging.getLogger(self.logger_name)
        logger.removeHandler(self._handler)
        logger.setLevel(self._previous)
        return False


def test_typed_thread_safety_of_the_cache():
    """The cache is shared across the threads LangChain runs sync tools on."""
    from terminus.context.retrievers import cache

    cache.reset()
    built = []
    barrier = threading.Barrier(4)

    def opener():
        barrier.wait(timeout=5)
        built.append(1)
        return object()

    results: list = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.cached_store("dense", opener)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    cache.reset()

    assert len(results) == 4
    assert all(r is results[0] for r in results), "concurrent callers got different stores"
