"""Logging setup, in one place.

A single ``basicConfig`` at import time, and a ``get_logger`` that does *not*
pin a level on each module. Pinning a level here (the earlier behaviour) silently
defeated every level the process set later: a logger with its own level ignores
the root logger's, so ``terminus --log-level DEBUG`` changed nothing at all.
Modules now inherit the root level, which the CLI owns.

**Third-party loggers are held at WARNING unless DEBUG was explicitly asked for.**
At INFO, ``httpx`` narrates every request as a timestamped line - during a model
download that is forty lines of noise in front of the answer the user asked for.
Opting into DEBUG means you wanted the firehose, so nothing is held back then.
"""

import logging

#: Loggers that are informative at DEBUG and pure noise at INFO. Anything here is
#: only allowed to speak when the process is at DEBUG.
_CHATTY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "sentence_transformers",
    "transformers",
    "openai",
    "google",
    "google_genai",
    "langchain",
    "langchain_core",
    "langsmith",
    "langsmith.client",
    "qdrant_client",
    "chromadb",
    "sentencepiece",
    "filelock",
    "asyncio",
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def set_log_level(level: str | int) -> None:
    """Set the level for every Terminus logger.

    Sets the root logger *and* the ``terminus`` package logger, so it takes
    effect regardless of whether a handler was already installed on either.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.WARNING
    logging.getLogger().setLevel(level)
    logging.getLogger("terminus").setLevel(level)
    # Third-party loggers are floored at ERROR unless DEBUG was explicitly asked
    # for, so their warnings cannot reach the user. That is not only about
    # volume: a library warning is a condition the user cannot act on and that
    # Terminus already reports through its own error path. LangSmith, for
    # instance, warns once per trace when the account's monthly quota is spent -
    # pure noise, and it had been flooding stderr mid-answer. Their errors still
    # show, because those can matter.
    for name in _CHATTY_LOGGERS:
        logging.getLogger(name).setLevel(
            logging.DEBUG if level <= logging.DEBUG else logging.ERROR
        )


def configure_tracing() -> bool:
    """Make LangSmith tracing follow ``observability.tracing``. Returns the result.

    LangChain decides whether to trace by reading ``LANGSMITH_TRACING`` (and the
    older ``LANGCHAIN_TRACING_V2``) out of the environment at call time, and it
    does that with no involvement from Terminus. So a project ``.env`` carrying a
    LangSmith key is enough to start sending every prompt, tool call and file
    path to a remote service - and, once that account's quota is spent, to start
    printing a rate-limit warning per trace into the middle of an answer.

    Both directions are reconciled here rather than only silencing the logs:

    * tracing off (the default) removes the variables LangChain looks at, so
      nothing is exported no matter what the ``.env`` says;
    * tracing on sets them, so a deliberate switch is all that is needed.

    The API key is left untouched either way, so enabling tracing is still just
    a matter of supplying the key. Suppressing the loggers in
    :func:`set_log_level` stays as a second line of defence, for the case where
    tracing is switched on and the remote service complains.
    """
    import os

    try:
        from terminus.config import CONFIG

        enabled = bool((CONFIG.get("observability") or {}).get("tracing", False))
    except Exception:  # pragma: no cover - never let observability block startup
        return False

    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        if enabled:
            os.environ[name] = "true"
        else:
            os.environ.pop(name, None)
    return enabled


def get_logger(module_name: str) -> logging.Logger:
    """The logger for a module.

    Deliberately does not call ``setLevel``: see the module docstring.
    """
    return logging.getLogger(module_name)
