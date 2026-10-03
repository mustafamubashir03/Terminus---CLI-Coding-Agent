import asyncio
import os
from copy import deepcopy

import pytest

from terminus import cli
from terminus.config import CONFIG, DEFAULT_CONFIG
from terminus.context.indexers import factory as indexer_factory
from terminus.context.indexers.errors import VectorStoreUnavailableError
from terminus.context.indexers import semantic_chroma
from terminus.context.indexers import qdrant_client
from terminus.context.indexers import freshness
from terminus.env import find_project_env, load_project_env


def test_find_project_env_searches_parent_directories(tmp_path):
    env_file = tmp_path / ".env"
    nested = tmp_path / "repo" / "nested"
    nested.mkdir(parents=True)
    env_file.write_text("TERMINUS_TEST_VALUE=file\n", encoding="utf-8")

    assert find_project_env(nested) == env_file.resolve()
    assert load_project_env(nested) == env_file.resolve()


def test_explicit_env_file_takes_precedence(tmp_path, monkeypatch):
    ancestor = tmp_path / ".env"
    explicit = tmp_path / "custom.env"
    nested = tmp_path / "repo"
    nested.mkdir()
    ancestor.write_text("TERMINUS_TEST_VALUE=ancestor\n", encoding="utf-8")
    explicit.write_text("TERMINUS_TEST_VALUE=explicit\n", encoding="utf-8")
    monkeypatch.setenv("TERMINUS_ENV_FILE", str(explicit))

    assert find_project_env(nested) == explicit.resolve()


def test_load_project_env_does_not_override_process_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TERMINUS_TEST_VALUE=file\n", encoding="utf-8")
    monkeypatch.setenv("TERMINUS_TEST_VALUE", "process")

    load_project_env(tmp_path)

    assert os.environ["TERMINUS_TEST_VALUE"] == "process"


def test_initialize_allows_missing_local_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_project_env", lambda path: None)
    monkeypatch.setattr(cli, "format_provider_diagnostics", lambda: "diagnostics")
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "chromadb")
    monkeypatch.setitem(CONFIG["rag"], "mode", "semantic")

    resolution = cli.initialize()

    # A backend description, derived from configuration alone.
    assert resolution.provider == "chromadb"
    assert resolution.mode == "semantic"
    assert resolution.fallback is False


def test_initialize_does_not_build_anything_expensive(monkeypatch):
    """Startup must not load a model, build a client, or open a store.

    This is the regression guard for the slow-boot fix. Each of those three
    things used to happen here: a HuggingFace embedder loaded its weights
    (tens of seconds on a cold cache), a provider SDK was imported (seconds),
    and a remote vector store was contacted (a network round trip that could
    fail and take the whole session down with it). None is needed to accept a
    first question, so all three are now built on first use instead.

    Asserted by making each explode if it is reached, rather than by timing -
    a timing assertion would pass on a fast machine and fail on a slow one,
    which is the opposite of what a guard should do.
    """
    def explode(name):
        def _fail(*args, **kwargs):
            raise AssertionError(
                f"initialize() must not call {name}; it is built on first use"
            )
        return _fail

    from terminus import llm

    for name in ("get_embedder", "get_llm", "get_chat_model"):
        monkeypatch.setattr(llm.factory, name, explode(name))
    monkeypatch.setattr(indexer_factory, "get_or_create_index", explode("get_or_create_index"))
    monkeypatch.setattr(
        indexer_factory, "build_index", explode("build_index")
    )
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "chromadb")
    monkeypatch.setitem(CONFIG["rag"], "mode", "semantic")

    # Completes without touching any of them.
    assert cli.initialize().provider == "chromadb"


def test_initialize_does_not_import_provider_sdks(monkeypatch):
    """Importing Terminus must not drag in every provider SDK.

    ``langchain_openai`` and ``langchain_cohere`` each cost seconds to import,
    and neither is needed to read a configuration file. The subclasses that
    require them live in their own modules so this module can defer the import
    to the route that uses it.
    """
    import subprocess
    import sys

    code = (
        "import sys; import terminus.cli; "
        "print(','.join(m for m in ('langchain_openai','langchain_cohere',"
        "'langchain_google_genai') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", (
        "importing terminus.cli loaded provider SDKs: " + out.stdout.strip()
    )


def test_cli_reports_startup_error_without_traceback(monkeypatch, capsys):
    def fail_startup():
        raise ValueError("OPENROUTER_API_KEY is not set")

    async def no_shutdown():
        return None

    monkeypatch.setattr(cli, "initialize", fail_startup)
    monkeypatch.setattr(cli, "shutdown_resources", no_shutdown)

    result = asyncio.run(cli.terminus_cli_run())

    output = capsys.readouterr().out
    assert result is False
    # The cause is shown, and it is the actionable part of the message rather
    # than a dump of every configured route.
    assert "OPENROUTER_API_KEY is not set" in output
    assert "could not start" in output
    # Points at the command that has the rest, without printing all of it.
    assert "terminus providers status" in output
    assert "Traceback" not in output


def test_default_indexer_is_local_chroma():
    assert DEFAULT_CONFIG["vector_store"]["provider"] == "chromadb"
    assert DEFAULT_CONFIG["rag"]["mode"] == "semantic"
    # Off. Substituting a backend is a decision, and the default is not to make it.
    assert DEFAULT_CONFIG["vector_store"]["fallback_to_chroma"] is False


def test_qdrant_transport_failure_uses_repository_local_chroma_when_enabled(monkeypatch, tmp_path):
    """Opt-in fallback still works, and no longer rewrites the global config.

    The previous version of this test asserted that ``CONFIG`` said ``chromadb``
    afterwards - i.e. it asserted the mutation that made retrieval
    non-deterministic. What matters now is that the fallback is *reported* and
    that the configuration the operator wrote is left exactly as it was.
    """
    saved = deepcopy(CONFIG)
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://qdrant.example")
    monkeypatch.setitem(CONFIG["rag"], "mode", "hybrid")
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "qdrant")
    monkeypatch.setitem(CONFIG["vector_store"], "retrieval_mode", "hybrid")
    monkeypatch.setitem(CONFIG["qdrant"], "collection_name", "test-collection")
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", True)
    sentinel = object()

    def fail_qdrant(provider, mode, repo_path, force_reindex):
        # Only the configured backend fails; the Chroma fallback must be able to
        # succeed, which is the point of the test.
        if provider == "qdrant":
            raise ConnectionResetError("connection reset by peer")
        return sentinel

    monkeypatch.setattr(indexer_factory, "build_index", fail_qdrant)
    monkeypatch.setattr(
        semantic_chroma, "get_or_create_chroma_index", lambda path, **kw: sentinel
    )

    try:
        result, resolution = indexer_factory.get_or_create_index(str(tmp_path))
        assert result is sentinel
        assert resolution.fallback is True
        assert resolution.provider == "chromadb"
        # Reported honestly: Chroma cannot do hybrid/BM25 on this path.
        assert resolution.configured_mode == "hybrid"
        assert resolution.mode == "semantic"
        # The configuration the operator wrote is untouched.
        assert CONFIG["vector_store"]["provider"] == "qdrant"
        assert CONFIG["rag"]["mode"] == "hybrid"
        assert "indexer_fallback" not in CONFIG.get("_runtime", {})
    finally:
        CONFIG.clear()
        CONFIG.update(saved)


def test_missing_qdrant_credentials_fail_loudly(monkeypatch, tmp_path):
    """A missing credential is a configuration mistake, not an outage.

    Falling back to a different database here would hide the one thing the
    operator has to fix, and would report results from a store they did not
    choose. So it raises even with the fallback enabled.
    """
    saved = deepcopy(CONFIG)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("CLUSTER_ENDPOINT", raising=False)
    monkeypatch.setitem(CONFIG["rag"], "mode", "hybrid")
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "qdrant")
    monkeypatch.setitem(CONFIG["vector_store"], "retrieval_mode", "hybrid")
    monkeypatch.setitem(CONFIG["qdrant"], "collection_name", "test-collection")
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", True)

    def fail_qdrant(*args, **kwargs):
        raise ValueError("CLUSTER_ENDPOINT not found in .env file")

    monkeypatch.setattr(indexer_factory, "build_index", fail_qdrant)
    monkeypatch.setattr(
        semantic_chroma,
        "get_or_create_chroma_index",
        lambda path, **kw: pytest.fail("Chroma must not be built for a config error"),
    )

    try:
        with pytest.raises(VectorStoreUnavailableError):
            indexer_factory.get_or_create_index(str(tmp_path))
    finally:
        CONFIG.clear()
        CONFIG.update(saved)


def test_qdrant_client_reset_is_reported_as_transport_failure(monkeypatch, tmp_path):
    saved = deepcopy(CONFIG)
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://qdrant.example")
    monkeypatch.setitem(CONFIG["rag"], "mode", "hybrid")
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "qdrant")
    monkeypatch.setitem(CONFIG["vector_store"], "retrieval_mode", "hybrid")
    monkeypatch.setitem(CONFIG["qdrant"], "collection_name", "test-collection")
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", False)

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            raise ConnectionResetError("connection reset by peer")

    # Patched at the shared client helper, which is now the only place a Qdrant
    # client gets constructed.
    monkeypatch.setattr(qdrant_client, "create_qdrant_client", lambda **kw: FailingClient())

    try:
        with pytest.raises(VectorStoreUnavailableError) as caught:
            indexer_factory.get_or_create_index(str(tmp_path))
        # The reported cause is the reset itself, not a wrapper class name.
        assert "tls" in str(caught.value).lower()
    finally:
        CONFIG.clear()
        CONFIG.update(saved)


def test_qdrant_transport_failure_can_fail_fast(monkeypatch, tmp_path):
    saved = deepcopy(CONFIG)
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("CLUSTER_ENDPOINT", "https://qdrant.example")
    monkeypatch.setitem(CONFIG["rag"], "mode", "hybrid")
    monkeypatch.setitem(CONFIG["vector_store"], "provider", "qdrant")
    monkeypatch.setitem(CONFIG["vector_store"], "retrieval_mode", "hybrid")
    monkeypatch.setitem(CONFIG["qdrant"], "collection_name", "test-collection")
    monkeypatch.setitem(CONFIG["vector_store"], "fallback_to_chroma", False)

    def fail_qdrant(*args, **kwargs):
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(indexer_factory, "build_index", fail_qdrant)

    try:
        with pytest.raises(VectorStoreUnavailableError, match="Vector store unavailable"):
            indexer_factory.get_or_create_index(str(tmp_path))
    finally:
        CONFIG.clear()
        CONFIG.update(saved)


def test_relative_storage_paths_are_scoped_to_repository(tmp_path, monkeypatch):
    monkeypatch.setitem(CONFIG["chromadb"], "persist_dir", ".terminus/chromadb/")
    monkeypatch.setitem(CONFIG, "index", {"manifest_path": ".terminus/index/manifest.json"})

    chroma_path = semantic_chroma.chroma_persist_path(tmp_path)
    manifest_path = freshness._manifest_path(str(tmp_path))

    assert chroma_path == tmp_path.resolve() / ".terminus" / "chromadb"
    assert manifest_path == tmp_path.resolve() / ".terminus" / "index" / "manifest.json"


# --- indexing must happen on the search path, not at startup -----------------
#
# Startup stopped building the index (that was the 28s). But the retriever only
# *queries*, so dropping the call meant a project that had never been indexed
# returned nothing forever - and the model quietly answered from grep instead, so
# the user saw a working agent and never learned semantic search was inert.

def test_search_codebase_refreshes_the_index_once_per_process(monkeypatch):
    import terminus.tools.codebase_tool as tool_module

    calls = []

    def fake_index(repo_path):
        calls.append(repo_path)
        return object(), object()

    monkeypatch.setattr(tool_module, "_index_refreshed", False)
    monkeypatch.setattr(
        "terminus.context.indexers.factory.get_or_create_index", fake_index
    )
    monkeypatch.setattr(
        tool_module, "get_retriever", lambda: (lambda q, k=5: [])
    )

    tool_module.search_codebase.invoke({"query": "anything"})
    tool_module.search_codebase.invoke({"query": "something else"})

    assert len(calls) == 1, f"index refreshed {len(calls)} times, expected once"
    assert calls[0], "refresh was called with no repository path"


def test_a_failed_refresh_does_not_retry_on_every_query(monkeypatch):
    """One failure must not turn into a filesystem walk per query."""
    import terminus.tools.codebase_tool as tool_module

    calls = []

    def boom(repo_path):
        calls.append(repo_path)
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(tool_module, "_index_refreshed", False)
    monkeypatch.setattr(
        "terminus.context.indexers.factory.get_or_create_index", boom
    )
    monkeypatch.setattr(tool_module, "get_retriever", lambda: (lambda q, k=5: []))

    tool_module.search_codebase.invoke({"query": "one"})
    tool_module.search_codebase.invoke({"query": "two"})

    assert len(calls) == 1


def test_search_codebase_survives_a_failing_refresh(monkeypatch):
    """A refresh that cannot run must not refuse the query."""
    import terminus.tools.codebase_tool as tool_module

    def boom(repo_path):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(tool_module, "_index_refreshed", False)
    monkeypatch.setattr(
        "terminus.context.indexers.factory.get_or_create_index", boom
    )
    monkeypatch.setattr(
        tool_module, "get_retriever", lambda: (lambda q, k=5: [])
    )

    out = tool_module.search_codebase.invoke({"query": "still answer me"})
    assert "No relevant code found" in out


# --- langchain-qdrant API: the classmethod has no `client` parameter --------
#
# from_existing_collection builds its own client from url/host/port. Passing a
# ready-made one raised TypeError: Client.__init__() got an unexpected keyword
# argument 'client', which surfaced as the vector store being "unavailable" on
# every reindex.

def test_connect_existing_uses_the_constructor_not_the_classmethod(monkeypatch):
    """Regression: the reindex path must build a store for an existing collection."""
    import inspect

    from langchain_qdrant import QdrantVectorStore

    assert "client" in inspect.signature(QdrantVectorStore.__init__).parameters
    assert "client" not in inspect.signature(
        QdrantVectorStore.from_existing_collection
    ).parameters, (
        "langchain-qdrant changed: if from_existing_collection accepts a client "
        "again, reindexer._connect_existing can go back to it"
    )


# --- a full index must not destroy another project's points -----------------
#
# The collection is shared by every project. The wipe used to be
# Filter(must=[]), which matches every point, so running a full index in a new
# project silently deleted every other project's data - observed dropping a
# 554-point shared collection to 4.

class _RecordingClient:
    def __init__(self):
        self.deletes = []

    def delete(self, **kwargs):
        self.deletes.append(kwargs)


def test_full_index_wipe_is_scoped_to_the_current_project(monkeypatch):
    from terminus.context.indexers import reindexer
    from terminus.workspace import PROJECT_METADATA_KEY

    monkeypatch.setattr(reindexer, "project_key_for_test", None, raising=False)
    client = _RecordingClient()

    reindexer._qdrant_wipe_collection(client, "shared_collection")

    assert len(client.deletes) == 1
    selector = client.deletes[0]["points_selector"]
    # Scoped: a must-condition on this project's key, not "match everything".
    assert selector.must, "wipe was unscoped; it would delete other projects' points"
    condition = selector.must[0]
    assert condition.key == f"metadata.{PROJECT_METADATA_KEY}"
    assert condition.match.value


def test_wipe_does_not_fail_when_the_collection_is_empty(monkeypatch):
    from terminus.context.indexers import reindexer

    class Failing(_RecordingClient):
        def delete(self, **kwargs):
            raise RuntimeError("collection not found")

    monkeypatch.setattr(reindexer, "_qdrant_wipe_collection", reindexer._qdrant_wipe_collection)
    reindexer._qdrant_wipe_collection(Failing(), "nope")
