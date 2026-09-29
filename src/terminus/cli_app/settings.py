"""Reading and writing Terminus's configuration, and storing credentials.

Two responsibilities, deliberately separate:

* **Preferences** (``llm.model``, ``llm.provider``, ...) live in ``config.yaml``
  and are written back through the same file :mod:`terminus.config` already
  reads. Reusing that file is what keeps ``models set`` and a hand-edited
  ``config.yaml`` in agreement - there is only ever one source.
* **Credentials** never go in configuration. They go in a ``.env`` file, which is
  the mechanism Terminus already loads at startup, so a key written by
  ``providers login`` is usable on the very next command with no restart and no
  new loading code.

Precedence, highest first - and the top entry is the reason ``--model`` behaves
differently from ``models set``:

    explicit CLI option   (this run only, never persisted)
    project config.yaml
    package config.yaml
    environment / .env
    built-in defaults
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from terminus.config import CONFIG, CONFIG_SOURCE, CONFIG_SOURCE_KIND
from terminus.llm.providers import credential_env_names

#: Every environment variable Terminus reads a credential from. Derived from the
#: provider table so it cannot fall behind the router; this used to be a fourth
#: hand-written list, which is how `providers login` and `providers list` came to
#: disagree with each other about which providers exist.
ENV_KEYS = credential_env_names()

SECRET_MASK = "********"


def config_path() -> Path:
    """The configuration file a write should go to.

    Always the current directory's ``config.yaml``, which is exactly the file
    :func:`terminus.config.load_config` looks for first. Resolving it from the
    import-time ``CONFIG_SOURCE`` instead would send a write to whatever
    directory the process happened to start in, which is a surprising place to
    find your configuration edited.
    """
    return Path.cwd() / "config.yaml"


PLAIN_SCALAR = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./:@+=-]*$")
# Words YAML would read back as a non-string type.
NOT_A_PLAIN_STRING = {"true", "false", "null", "yes", "no", "on", "off", "~"}


def _scalar(value: Any) -> str:
    """Render a Python value as a YAML scalar.

    Strings are emitted plain when that is unambiguous and JSON-quoted
    otherwise. ``yaml.safe_dump`` is not usable here: for a value such as
    ``poolside/laguna-s-2.1:free`` it appends a ``...`` document-end marker,
    which parses on its own but poisons the next append.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if PLAIN_SCALAR.match(text) and text.lower() not in NOT_A_PLAIN_STRING:
        return text
    return json.dumps(text)


def _find_key(lines: list[str], parts: list[str]) -> int | None:
    """Index of the line that defines the final key, or None.

    Tracks nesting by indentation. List items are skipped: their ``-`` prefix
    means they are not dict keys, and a dotted path never descends into one.
    """
    stack: list[tuple[int, str]] = []
    for index, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = re.match(r"^(\s*)([A-Za-z_][\w\-]*)\s*:", raw)
        if not match:
            continue
        indent = len(match.group(1))
        key = match.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        if [name for _i, name in stack] == parts:
            return index
    return None


def set_value(dotted: str, value: Any) -> tuple[Path, Any]:
    """Persist ``dotted=value`` into the configuration file.

    Edits the one line in place rather than re-serialising the document.
    A round-trip through ``yaml.safe_dump`` would silently delete every comment
    in ``config.yaml`` - and those comments carry the reasoning behind the
    tuning values, so losing them is a real cost, not cosmetic.

    Returns the file and the value as written. Raises on an unknown key rather
    than inventing one, so a typo is an error instead of a setting that is
    silently ignored until the next run.
    """
    if not has_path(dotted):
        raise KeyError(dotted)
    parts = dotted.split(".")
    if isinstance(value, (dict, list)):
        raise TypeError(f"{dotted} must be a scalar, not a {type(value).__name__}")

    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    index = _find_key(lines, parts)

    if index is not None:
        indent = len(lines[index]) - len(lines[index].lstrip())
        lines[index] = f"{' ' * indent}{parts[-1]}: {_scalar(value)}"
    else:
        # A key the defaults define but this file does not: add the missing
        # parents, then the key itself. The last line written always carries a
        # value, so every parent written before it has a child to nest under.
        missing: list[str] = []
        for depth in range(len(parts), 0, -1):
            if _find_key(lines, parts[:depth]) is None:
                missing.insert(0, parts[depth - 1])
        if len(missing) == len(parts):
            base_indent = 0
        else:
            parent = _find_key(lines, parts[: len(parts) - len(missing)])
            base_indent = len(lines[parent]) - len(lines[parent].lstrip()) + 2
        for offset, name in enumerate(missing[:-1]):
            lines.append(f"{' ' * (base_indent + 2 * offset)}{name}:")
        leaf_indent = base_indent + 2 * (len(missing) - 1)
        lines.append(f"{' ' * leaf_indent}{parts[-1]}: {_scalar(value)}")
    if lines and lines[-1] == "":
        lines.pop()

    updated = "\n".join(lines).rstrip() + "\n"
    # Never leave a file behind that does not parse: a syntax error here would
    # break every later run, including the one the user runs to fix it.
    try:
        yaml.safe_load(updated)
    except yaml.YAMLError as exc:
        raise ValueError(f"Refusing to write a config file that would not parse: {exc}") from exc
    path.write_text(updated, encoding="utf-8")
    return path, value


def get_value(dotted: str) -> Any:
    """Read a dotted key out of the effective configuration."""
    value: Any = CONFIG
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def has_path(dotted: str) -> bool:
    """True when the effective configuration supplies this dotted key."""
    return _has_path(CONFIG, dotted)


def effective_source(dotted: str) -> str:
    """Which layer currently supplies a setting. Shown by ``config get``."""
    for candidate, kind in (
        (Path.cwd() / "config.yaml", "project"),
        (CONFIG_SOURCE if CONFIG_SOURCE_KIND == "package" else None, "package"),
    ):
        if candidate is None or not candidate.exists():
            continue
        try:
            loaded = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _has_path(loaded, dotted):
            return kind
    return "default" if _has_path(CONFIG, dotted) else "unset"


def _has_path(document: Any, dotted: str) -> bool:
    cursor = document
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True


# --- credentials -----------------------------------------------------------


def credentials_file() -> Path:
    """Where ``providers login`` writes a key.

    The project's ``.env``, because that is already the file Terminus loads at
    startup - so a key written here works immediately and needs no new loading
    path. ``.env`` is gitignored. A credential is never written to
    ``config.yaml``, which people commit.
    """
    return Path.cwd() / ".env"


def store_credential(env_key: str, value: str, *, to_stdout: bool = False) -> Path:
    """Write ``ENV_KEY=value`` into the ``.env`` file, replacing any existing entry."""
    path = credentials_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pattern = re.compile(rf"^\s*{re.escape(env_key)}\s*=")
    lines = [line for line in existing if not pattern.match(line)]
    lines.append(f"{env_key}={value}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    if to_stdout:
        os.environ[env_key] = value
    return path


def clear_credential(env_key: str) -> tuple[Path, bool]:
    """Remove a key from the ``.env`` file. Returns the file and whether it was there."""
    path = credentials_file()
    if not path.exists():
        return path, False
    pattern = re.compile(rf"^\s*{re.escape(env_key)}\s*=")
    existing = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in existing if not pattern.match(line)]
    removed = len(kept) != len(existing)
    if removed:
        path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
        os.environ.pop(env_key, None)
    return path, removed


def credential_status() -> list[dict[str, Any]]:
    """Presence only. The value is never read into the result."""
    return [
        {
            "name": key,
            "configured": bool(os.environ.get(key)),
            "source": ".env" if _in_env_file(key) else ("environment" if os.environ.get(key) else ""),
        }
        for key in ENV_KEYS
    ]


def _in_env_file(key: str) -> bool:
    path = credentials_file()
    if not path.exists():
        return False
    return any(re.match(rf"^\s*{re.escape(key)}\s*=", line) for line in path.read_text(encoding="utf-8").splitlines())


def mask(value: str) -> str:
    return SECRET_MASK if value else ""
