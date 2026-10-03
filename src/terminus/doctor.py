"""Collecting the health report behind ``terminus doctor``.

Kept out of the CLI module so it can be tested without a terminal, and so the
rules about what counts as a failure live in one readable place.

The classification policy is deliberate and is the thing worth arguing about
here. A missing **optional** service is a WARN, never a FAIL: not having LangSmith
configured, or not having a Qdrant cluster, does not stop anyone from asking a
question. Only the things Terminus genuinely cannot do without are FAIL - and
"genuinely cannot" means a provider key (no LLM means no answer) and a readable
project.

Everything here reports presence and provenance. No secret is read into the
result, and any error text is passed through
:func:`terminus.user_config.redact` before it is returned, so a provider that
helpfully echoes the key back in an exception message cannot leak it through the
diagnostic.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _row(status: str, label: str, detail: str = "") -> dict[str, Any]:
    return {"status": status, "label": label, "detail": detail}


def collect_report() -> dict[str, Any]:
    """Gather the whole report. Never raises; a broken check is a FAIL row."""
    from terminus.user_config import (
        apply_auto_provider,
        configured_provider_has_credential,
        global_config_path,
        has_usable_llm_credential,
        redact,
    )

    # Resolve the provider exactly as startup does. Reporting FAIL on an install
    # that would in fact answer questions is worse than any missing check.
    try:
        apply_auto_provider()
    except Exception:
        pass

    sections: dict[str, list[dict[str, Any]]] = {
        "Installation": _installation(),
        "Configuration": _configuration(global_config_path),
        "Provider": _provider(configured_provider_has_credential, has_usable_llm_credential),
        "Credentials": _credentials(),
        "Retrieval": _retrieval(redact),
        "Optional": _optional(),
    }

    rows = [row for group in sections.values() for row in group]
    summary = {
        "pass": sum(1 for r in rows if r["status"] == PASS),
        "warn": sum(1 for r in rows if r["status"] == WARN),
        "fail": sum(1 for r in rows if r["status"] == FAIL),
    }
    return {"ok": summary["fail"] == 0, "summary": summary, "sections": sections}


def _installation() -> list[dict[str, Any]]:
    rows = [_row(PASS, "Terminus", _version())]
    rows.append(_row(PASS, "Python", f"{platform.python_version()} ({sys.executable})"))
    rows.append(_row(PASS, "Platform", f"{platform.system()} {platform.release()}"))
    return rows


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("terminus")
    except Exception:
        return "unknown"


def _configuration(global_config_path: Path) -> list[dict[str, Any]]:
    from terminus.config import CONFIG_SOURCE, CONFIG_SOURCE_KIND

    where = str(CONFIG_SOURCE) if CONFIG_SOURCE else "built-in defaults"
    kind = CONFIG_SOURCE_KIND
    label = {
        "cwd": "project config.yaml",
        "global": "your global config.yaml",
        "package": "packaged config.yaml",
        "default": "built-in defaults",
    }.get(kind, kind)
    return [
        _row(PASS, "Configuration source", f"{label} ({where})"),
        _row(PASS, "Global config", str(global_config_path())),
    ]


def _provider(configured_ok, any_key) -> list[dict[str, Any]]:
    from terminus.config import CONFIG

    llm = CONFIG.get("llm", {})
    provider = str(llm.get("provider", "?"))
    model = str(llm.get("model", "?"))
    rows = [_row(PASS, "Provider", provider), _row(PASS, "Model", model)]

    if configured_ok():
        rows.append(_row(PASS, "Provider key", "present for the configured provider"))
    elif any_key():
        rows.append(
            _row(
                WARN,
                "Provider key",
                "no key for the configured provider, but another provider has one",
            )
        )
        rows.append(_row(WARN, "  fix", "terminus setup, or terminus config set llm.provider"))
    else:
        rows.append(_row(FAIL, "Provider key", "not configured"))
        rows.append(_row(FAIL, "  fix", "terminus setup"))

    fallbacks = llm.get("fallbacks") or []
    if fallbacks:
        names = ", ".join(str(f.get("provider")) for f in fallbacks if isinstance(f, dict))
        rows.append(_row(PASS, "Fallbacks", names))
    else:
        rows.append(_row(WARN, "Fallbacks", "none configured; a rate limit will end the request"))
    return rows


def _credentials() -> list[dict[str, Any]]:
    from terminus.user_config import credential_status

    rows: list[dict[str, Any]] = []
    for entry in credential_status():
        if entry["configured"]:
            rows.append(
                _row(PASS, entry["name"], f"configured ({entry['source']})")
            )
        else:
            rows.append(_row(WARN, entry["name"], "not set"))
    return rows


def _retrieval(redact) -> list[dict[str, Any]]:
    """Vector store reachability. Optional: a WARN unless it is the default path."""
    from terminus.config import CONFIG

    rows: list[dict[str, Any]] = []
    provider = str(CONFIG.get("vector_store", {}).get("provider", "?"))
    mode = str(CONFIG.get("rag", {}).get("mode", "?"))
    embeds = CONFIG.get("embeddings", {})
    rows.append(
        _row(PASS, "Vector store", f"{provider} / {mode}")
    )
    rows.append(
        _row(PASS, "Embeddings", f"{embeds.get('provider')} / {embeds.get('model')}")
    )

    try:
        # Ask the store whether it is there. Do not resolve the index: that
        # builds one, and a health check that quietly indexes a repository -
        # loading an embedding model, calling an embedding API, writing vectors -
        # is not a health check. Someone runs `doctor` to find out whether
        # things work, not to make them work.
        location = _probe_store_readonly()
        if location is None:
            rows.append(
                _row(WARN, "Vector store reachable", "not built yet (the first search builds it)")
            )
        else:
            rows.append(_row(PASS, "Vector store reachable", location))
    except Exception as exc:
        # Not a failure: Terminus answers questions without search.
        rows.append(_row(WARN, "Vector store reachable", redact(str(exc)).splitlines()[0][:90]))
    return rows


def _probe_store_readonly() -> str | None:
    """Describe the configured store without creating or changing anything.

    Returns a short label for the store, or None when it is not there yet.
    """
    from terminus.config import CONFIG
    from terminus.context.indexers.qdrant_client import (
        collection_name,
        create_qdrant_client,
        qdrant_mode,
    )

    if qdrant_mode() == "cloud":
        client = create_qdrant_client()
        name = collection_name()
        try:
            if not client.collection_exists(name):
                return None
            count = getattr(client.get_collection(name), "points_count", None)
            return f"qdrant cloud {name}" + (f", {count} points" if count is not None else "")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    persist = str((CONFIG.get("chromadb") or {}).get("persist_dir") or ".terminus/chromadb/")
    local = Path.cwd() / persist
    return f"chroma {local}" if local.exists() else None


def _optional() -> list[dict[str, Any]]:
    from terminus.config import CONFIG

    rows: list[dict[str, Any]] = []
    tracing = bool((CONFIG.get("observability") or {}).get("tracing"))
    if tracing:
        rows.append(_row(PASS, "LangSmith tracing", "enabled"))
    else:
        rows.append(_row(WARN, "LangSmith tracing", "off (default)"))
    if os.environ.get("LANGSMITH_API_KEY"):
        rows.append(_row(PASS, "LangSmith key", "configured"))
    return rows
