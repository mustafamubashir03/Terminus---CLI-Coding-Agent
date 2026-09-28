from __future__ import annotations

from collections.abc import Iterator


class IndexerConfigurationError(RuntimeError):
    pass


class VectorStoreUnavailableError(RuntimeError):
    pass


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transport_error(exc: BaseException) -> bool:
    for current in _exception_chain(exc):
        if isinstance(current, (ConnectionError, TimeoutError, OSError)):
            return True
        name = type(current).__name__.lower()
        module = type(current).__module__.lower()
        if any(token in name for token in ("connect", "timeout", "transport", "network", "remote_protocol")):
            return True
        if module.startswith("grpc"):
            return True
    return False


def qdrant_error_message(
    exc: BaseException,
    repo_path: str,
    provider: str,
    mode: str,
    collection: str | None,
    config_source: str | None,
) -> str:
    source = config_source or "built-in defaults"
    collection_text = collection or "not configured"
    detail = str(exc).strip()
    reason = (
        detail
        if isinstance(exc, ValueError)
        and ("QDRANT_API_KEY" in detail or "CLUSTER_ENDPOINT" in detail)
        else type(exc).__name__
    )
    reason = reason or type(exc).__name__
    return (
        f"Vector store unavailable for repository {repo_path} "
        f"(provider={provider}, mode={mode}, collection={collection_text}, config={source}): "
        f"{reason}. Check the vector-store endpoint and network access, or set "
        f"vector_store.provider to chromadb for local semantic indexing."
    )


def configuration_error(message: str, config_source: str | None) -> IndexerConfigurationError:
    source = config_source or "built-in defaults"
    return IndexerConfigurationError(f"{message} (configuration: {source})")
