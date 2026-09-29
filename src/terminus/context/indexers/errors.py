"""Failure classification for the indexers.

Two jobs.

**Classify** a failure well enough that the operator knows which of the ten
distinct things went wrong, because "vector database failed" is not actionable
and collapses a missing credential, a suspended cluster and a corrupted local
store into one indistinguishable outcome.

**Report the cause, not the wrapper.** An SDK wraps the interesting exception
one or more layers down: the Qdrant client's transport raises
``httpx.ConnectError`` -> ``httpcore.ConnectError`` -> ``ConnectionResetError``,
and surfaces the lot as ``ResponseHandlingException``. Reporting only the outer
name is what produced the message ``...: ResponseHandlingException. Check the
vector-store endpoint and network access...`` when the actual fault was a TLS
handshake reset. The chain is walked and the *root* is named.

Nothing here is certain about the network. A TLS reset tells us the connection
was cut during the handshake, not why. The messages say what was observed.
"""

from __future__ import annotations

import socket
import ssl
from collections.abc import Iterator
from enum import Enum

from terminus.observability.logging import get_logger

logger = get_logger(__name__)


class IndexerError(RuntimeError):
    """Base for the two failures an indexer reports deliberately."""


class IndexerConfigurationError(IndexerError):
    """The configuration is wrong. Nothing was attempted."""


class VectorStoreUnavailableError(IndexerError):
    """The configured vector store could not be used.

    Raised instead of substituting a different backend. The caller decides
    whether that is fatal; it is never resolved by quietly using Chroma.
    """


class VectorStoreFailure(str, Enum):
    """The distinct ways a vector store can fail.

    Kept as an enum so tests can assert on the class of failure rather than on
    message text, and so a new failure mode is a deliberate addition rather
    than a new string that happens to differ.
    """

    MISSING_CREDENTIALS = "missing_credentials"
    AUTHENTICATION = "authentication"
    MISCONFIGURED = "misconfigured"
    DNS = "dns"
    CONNECTION_REFUSED = "connection_refused"
    TLS = "tls"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    COLLECTION = "collection"
    LOCAL_STORE = "local_store"
    EMBEDDING = "embedding"
    PARSE = "parse"
    UNKNOWN = "unknown"


#: A failure that means "this store is not usable right now", as opposed to a
#: mistake in the request. Used to decide whether an *explicitly enabled*
#: fallback is allowed to engage at all.
RECOVERABLE_FAILURES = frozenset({
    VectorStoreFailure.DNS,
    VectorStoreFailure.CONNECTION_REFUSED,
    VectorStoreFailure.TLS,
    VectorStoreFailure.TIMEOUT,
    VectorStoreFailure.UNAVAILABLE,
    VectorStoreFailure.LOCAL_STORE,
})


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def root_cause(exc: BaseException) -> BaseException:
    """The innermost exception in *exc*'s chain.

    Reported in preference to the wrapper because the wrapper is the SDK's
    choice of vocabulary and the cause is the fault.
    """
    current = exc
    while True:
        nxt = current.__cause__ or current.__context__
        if nxt is None or nxt is current:
            return current
        current = nxt


def is_transport_error(exc: BaseException) -> bool:
    """True when the chain contains a connection/timeout style failure.

    Kept for compatibility with the existing predicate. Prefer
    :func:`classify_failure`, which distinguishes the cases this folds together.
    """
    return classify_failure(exc) in RECOVERABLE_FAILURES


def classify_failure(exc: BaseException) -> VectorStoreFailure:
    """Which kind of failure *exc* represents."""
    for current in _exception_chain(exc):
        if isinstance(current, ssl.SSLError):
            return VectorStoreFailure.TLS
        if isinstance(current, socket.gaierror):
            return VectorStoreFailure.DNS
        if isinstance(current, (socket.timeout, TimeoutError)):
            return VectorStoreFailure.TIMEOUT
        if isinstance(current, ConnectionRefusedError):
            return VectorStoreFailure.CONNECTION_REFUSED
        if isinstance(current, ConnectionResetError):
            # A connection accepted then cut mid-handshake. Distinguished from a
            # plain refusal, and from a timeout, because the remedy differs: a
            # reset points at the far end, not at this host's firewall.
            return VectorStoreFailure.TLS
        if isinstance(current, ssl.SSLCertVerificationError):
            return VectorStoreFailure.TLS
    if isinstance(exc, OSError):
        return VectorStoreFailure.LOCAL_STORE

    text = str(exc).lower()
    if "qdrant_api_key" in text or "cluster_endpoint" in text or "not set" in text:
        return VectorStoreFailure.MISSING_CREDENTIALS
    if isinstance(exc, IndexerConfigurationError):
        return VectorStoreFailure.MISCONFIGURED
    for marker in ("unauthorized", "forbidden", "invalid api key", "invalid_api_key", "403", "401"):
        if marker in text:
            return VectorStoreFailure.AUTHENTICATION
    for marker in ("not found", "already exists", "wrong dimensions", "vector name",
                   "collection", "schema"):
        if marker in text:
            return VectorStoreFailure.COLLECTION
    for marker in ("name or service not known", "nodename nor servname",
                   "getaddrinfo", "temporary failure in name resolution"):
        if marker in text:
            return VectorStoreFailure.DNS
    if "timed out" in text or "timeout" in text:
        return VectorStoreFailure.TIMEOUT

    name = type(exc).__name__.lower()
    module = type(exc).__module__.lower()
    if module.startswith("grpc"):
        return VectorStoreFailure.UNAVAILABLE
    if any(token in name for token in ("connect", "network", "remote_protocol", "unavailable")):
        return VectorStoreFailure.UNAVAILABLE
    if "timeout" in name:
        return VectorStoreFailure.TIMEOUT
    return VectorStoreFailure.UNKNOWN


#: Human phrasing per failure. Deliberately describes what was observed and
#: stops there: an SDK cannot tell us *why* a far end reset a handshake.
_FAILURE_TEXT: dict[VectorStoreFailure, str] = {
    VectorStoreFailure.MISSING_CREDENTIALS: "credentials are not configured",
    VectorStoreFailure.AUTHENTICATION: "the credentials were rejected",
    VectorStoreFailure.MISCONFIGURED: "the configuration is not usable",
    VectorStoreFailure.DNS: "the endpoint hostname could not be resolved",
    VectorStoreFailure.CONNECTION_REFUSED: "the endpoint refused the connection",
    VectorStoreFailure.TLS: "the TLS handshake failed or the connection was reset during it",
    VectorStoreFailure.TIMEOUT: "the request timed out",
    VectorStoreFailure.UNAVAILABLE: "the endpoint could not be reached",
    VectorStoreFailure.COLLECTION: "the collection or its schema is not usable",
    VectorStoreFailure.LOCAL_STORE: "the local store could not be opened",
    VectorStoreFailure.EMBEDDING: "the embedding model could not be used",
    VectorStoreFailure.PARSE: "a source file could not be parsed",
    VectorStoreFailure.UNKNOWN: "failed for an unrecognised reason",
}

#: What the operator can do about each. Same reasoning: observation plus the
#: documented next step, not a guess at the vendor's infrastructure.
_FAILURE_ADVICE: dict[VectorStoreFailure, str] = {
    VectorStoreFailure.MISSING_CREDENTIALS:
        "set the credential in your .env, or switch qdrant.mode to 'local' for a "
        "credential-free index",
    VectorStoreFailure.AUTHENTICATION:
        "check the API key and that it is valid for this cluster",
    VectorStoreFailure.MISCONFIGURED:
        "run `terminus config list` to see the effective configuration",
    VectorStoreFailure.DNS:
        "check the endpoint hostname and this machine's DNS; a typo and an "
        "interrupted connection look the same here",
    VectorStoreFailure.CONNECTION_REFUSED:
        "confirm the endpoint is running and reachable on that port",
    VectorStoreFailure.TLS:
        "the far end accepted the TCP connection and then closed it during the "
        "TLS handshake. That points at the endpoint, not at this machine: the "
        "cluster may be suspended, deleted, or no longer provisioned for this "
        "account. A local index avoids it entirely (qdrant.mode: local)",
    VectorStoreFailure.TIMEOUT:
        "raise qdrant.timeout_seconds, or check for a slow or overloaded endpoint",
    VectorStoreFailure.UNAVAILABLE:
        "check network access to the endpoint; if it is a remote cluster, "
        "confirm it still exists",
    VectorStoreFailure.COLLECTION:
        "run `terminus index status` to inspect the collection before reindexing",
    VectorStoreFailure.LOCAL_STORE:
        "the local store directory may be locked by another process or corrupt; "
        "move it aside to rebuild",
    VectorStoreFailure.EMBEDDING:
        "check that the configured embedding model is available locally",
    VectorStoreFailure.PARSE:
        "a file was skipped during indexing; the rest of the index is unaffected",
    VectorStoreFailure.UNKNOWN:
        "rerun with --log-level DEBUG for the full traceback",
}


def describe_failure(exc: BaseException) -> str:
    """One line naming the observed failure and the root cause."""
    kind = classify_failure(exc)
    cause = root_cause(exc)
    detail = str(cause).strip() or type(cause).__name__
    return f"{kind.value}: {_FAILURE_TEXT[kind]} ({type(cause).__name__}: {detail})"


def failure_advice(kind: VectorStoreFailure) -> str:
    return _FAILURE_ADVICE[kind]


def qdrant_error_message(
    exc: BaseException,
    repo_path: str,
    provider: str,
    mode: str,
    collection: str | None,
    config_source: str | None,
) -> str:
    """The full, user-facing failure for an unusable vector store.

    States what was observed, where it happened, and what to do. It no longer
    advises switching to Chroma: choosing a different backend is the
    operator's decision, and a fallback that exists to make that decision for
    them is the bug this replaced.
    """
    source = config_source or "built-in defaults"
    collection_text = collection or "not configured"
    kind = classify_failure(exc)
    return (
        f"Vector store unavailable for repository {repo_path} "
        f"(provider={provider}, mode={mode}, collection={collection_text}, config={source})\n"
        f"  what happened : {describe_failure(exc)}\n"
        f"  what to do    : {failure_advice(kind)}"
    )


def configuration_error(message: str, config_source: str | None) -> IndexerConfigurationError:
    source = config_source or "built-in defaults"
    return IndexerConfigurationError(f"{message} (configuration: {source})")
