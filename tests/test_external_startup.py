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
    llm = object()
    embedder = object()
    index = object()
    monkeypatch.setattr(cli, "get_llm", lambda: llm)
    monkeypatch.setattr(cli, "format_provider_diagnostics", lambda: "diagnostics")
    monkeypatch.setattr(cli, "get_embedder", lambda: embedder)
    resolution = indexer_factory.resolved_backend("chromadb", "semantic")
    monkeypatch.setattr(cli, "get_or_create_index", lambda path: (index, resolution))

    # The resolution travels with the index so a fallback can be reported.
    assert cli.initialize() == (llm, embedder, index, resolution)


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
    assert "Startup failed:" in output
    assert "OPENROUTER_API_KEY is not set" in output
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
