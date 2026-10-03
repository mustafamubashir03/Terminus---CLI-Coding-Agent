"""Global configuration and credentials, in one place.

Terminus previously had exactly one place to put a key: the project's ``.env``.
That works, but it means ``terminus providers login`` inside one repository
leaves every other repository unauthenticated, which is the opposite of what
"log in to a tool" means. It is also a footgun in the other direction: a key
lives in a file sitting in the working tree, where it is one `git add -A` away
from being committed.

So there are now two stores, and the rule between them is the ordinary one:

* the project ``.env`` is specific to a repository, and wins inside it;
* ``~/.terminus/credentials.env`` belongs to the developer, and applies
  everywhere.

``~/.terminus`` is not invented here - it is already where user skills live
(``skills/registry.py``). Keeping one directory means one thing to back up and
one thing to reason about.

Precedence, highest first, for a credential:

    1. the process environment      (exported by the user, or set by CI)
    2. the project ``.env``         (repository-specific; gitignored)
    3. ``TERMINUS_ENV_FILE``        (an explicit override of the above)
    4. ``~/.terminus/credentials.env``  (this module's store)
    5. unset

1 wins because :func:`load_env_files` uses ``override=False`` throughout, so a
value already in the environment is never replaced. 2 beats 4 because the
project file is read *first* and the global file only fills what is still
missing.

No credential is ever written to ``config.yaml``: people commit that file, and
a secret in it is a leak that looks like configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from terminus.observability.logging import get_logger

logger = get_logger(__name__)

#: Filename inside ``~/.terminus``. Separate from ``config.yaml`` on purpose: one
#: is committed by developers on purpose, the other must never be.
CREDENTIALS_FILENAME = "credentials.env"
CONFIG_FILENAME = "config.yaml"

ENV_KEYS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "FIREWORKS_API_KEY",
    "CEREBRAS_API_KEY",
    "ANTHROPIC_API_KEY",
    "QDRANT_API_KEY",
    "CLUSTER_ENDPOINT",
)

SECRET_MASK = "********"

_FILES_LOADED = False

#: Which of ``ENV_KEYS`` are endpoints rather than secrets. Showing the host of
#: a cluster endpoint is useful and reveals nothing; masking it would just make
#: the diagnostic less useful.
_NON_SECRET_KEYS = frozenset({"CLUSTER_ENDPOINT"})


def user_config_dir() -> Path:
    """``~/.terminus``, created on demand."""
    return Path.home() / ".terminus"


def global_config_path() -> Path:
    return user_config_dir() / CONFIG_FILENAME


def global_credentials_path() -> Path:
    return user_config_dir() / CREDENTIALS_FILENAME


def project_env_write_path() -> Path:
    """Where ``--scope project`` should write.

    ``TERMINUS_ENV_FILE`` wins, so a checkout that keeps credentials somewhere
    other than beside the code keeps them there. Otherwise the ``.env`` beside
    the working directory, not the one found by searching upward: writing a
    secret into a parent directory you merely happen to be standing in is not
    something to do quietly.
    """
    configured = os.getenv("TERMINUS_ENV_FILE")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``KEY=value`` file. No interpolation, no export, no shell syntax."""
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return found
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            found[key] = value
    return found


def load_env_files(start_path: str | Path | None = None) -> dict[str, Path]:
    """Populate ``os.environ`` from the credential files, in precedence order.

    Returns the files that were actually read, so a caller can report where a
    value came from rather than guessing.

    The order below is the whole design: the project file is read first so that
    the global file, read second with ``override=False``, can only supply values
    the project did not. Reading them the other way round would make the global
    store beat the repository, which is backwards - a developer who deliberately
    puts a different key in a repository means it.
    """
    from terminus.env import find_project_env
    from dotenv import load_dotenv

    read: dict[str, Path] = {}

    project_env = find_project_env(start_path)
    if project_env is not None:
        load_dotenv(project_env, override=False)
        read["project"] = project_env

    global_env = global_credentials_path()
    if global_env.is_file():
        for key, value in _read_env_file(global_env).items():
            if not os.environ.get(key):
                os.environ[key] = value
                read.setdefault("global", global_env)

    return read


def credential_source(env_key: str) -> str:
    """Where the *current* value of *env_key* came from, without reading the secret.

    Returns ``environment``, ``project .env``, ``global credentials``, or ``""``.

    Compares the live value against what each file holds, rather than just
    checking that the key appears in a file. Those differ in exactly the case
    that matters: an exported shell variable and a stale project ``.env`` line
    for the same key both "exist", but the exported one is the one in effect -
    and a precedence report that named the wrong one would be worse than none.
    """
    current = os.environ.get(env_key)
    if not current:
        return ""

    from terminus.env import find_project_env

    project_env = find_project_env()
    if project_env is not None and _read_env_file(project_env).get(env_key) == current:
        return "project .env"
    if _read_env_file(global_credentials_path()).get(env_key) == current:
        return "global credentials"
    return "environment"


def credential_status() -> list[dict[str, Any]]:
    """Presence and provenance for every credential Terminus understands.

    The value is never read into the result. Only ``configured`` and ``source``.
    """
    _ensure_loaded()
    return [
        {
            "name": key,
            "configured": bool(os.environ.get(key)),
            "secret": key not in _NON_SECRET_KEYS,
            "source": credential_source(key),
        }
        for key in ENV_KEYS
    ]


def mask(value: str) -> str:
    """A value safe to print. Empty in, empty out, so blanks stay visually blank."""
    return SECRET_MASK if value else ""


def redact(text: str) -> str:
    """Strip any known credential value out of a string before it is shown.

    Used on exception text and diagnostic output. Deliberately paranoid: it masks
    every credential currently in the environment, not just the one being talked
    about, because the one that leaks is rarely the one expected.
    """
    if not text:
        return text
    out = text
    for key in ENV_KEYS:
        if key in _NON_SECRET_KEYS:
            continue
        value = os.environ.get(key)
        if value and len(value) >= 8:
            out = out.replace(value, SECRET_MASK)
    return out


def _write_env_file(path: Path, updates: dict[str, str]) -> None:
    """Rewrite *path* with *updates* applied, preserving comments and order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    applied: set[str] = set()
    lines: list[str] = []
    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in updates:
                lines.append(f"{key}={updates[key]}")
                applied.add(key)
                continue
        lines.append(line)
    for key, value in updates.items():
        if key not in applied:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    _restrict_to_owner(path)


def _restrict_to_owner(path: Path) -> None:
    """Make a credential file readable only by its owner.

    On POSIX that is a 0600 mode, and the umask is irrelevant because the mode
    is set explicitly. On Windows the mode bits barely mean anything, so this
    locks the file down through its ACL instead - otherwise the global store is
    a plaintext API key readable by every local account, which is a real
    downgrade from the project ``.env`` it replaces.

    Best effort by design: a filesystem that refuses to be told is not a reason
    to fail a ``login`` that already succeeded, so problems are logged and the
    write stands.
    """
    try:
        if os.name == "nt":
            _restrict_windows(path)
        else:
            os.chmod(path, 0o600)
    except Exception as exc:  # pragma: no cover - platform specific
        logger.debug("Could not restrict permissions on %s: %s", path, exc)


def _restrict_windows(path: Path) -> None:  # pragma: no cover - Windows only
    """Replace the file's ACL with a single entry granting only the current user."""
    import subprocess

    user = os.environ.get("USERNAME")
    domain = os.environ.get("USERDOMAIN")
    if not user:
        return
    account = f"{domain}\\{user}" if domain else user
    # icacls returns 0 on success; the grant list is reset first so a permissive
    # entry inherited from the parent directory is actually removed.
    subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{account}:(R,W)"],
        capture_output=True,
        check=False,
    )


def _remove_env_key(path: Path, env_key: str, delete_if_empty: bool = False) -> bool:
    """Delete *env_key*'s line, keeping comments and the order of everything else.

    ``_write_env_file`` cannot do this: it merges a set of assignments into the
    lines already present, so a key that is simply not mentioned keeps its old
    value. Passing the remaining keys to it therefore looked like it removed the
    credential while leaving it on disk, and ``providers logout`` would report
    success and change nothing.

    ``delete_if_empty`` is reserved for the global store, which Terminus owns.
    A project ``.env`` is the user's file - it may be referenced by a compose
    file, a CI step, or another tool - so it is emptied, never removed.
    """
    if not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    removed = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            if stripped.partition("=")[0].strip() == env_key:
                removed = True
                continue
        kept.append(line)
    if not removed:
        return False
    body = "\n".join(kept).rstrip()
    if not body:
        if delete_if_empty:
            path.unlink()
        else:
            path.write_text("", encoding="utf-8")
        return True
    path.write_text(body + "\n", encoding="utf-8")
    return True


def store_credential(env_key: str, value: str, scope: str = "global") -> Path:
    """Persist ``ENV_KEY=value`` to the chosen store, and use it immediately.

    ``scope="global"`` writes ``~/.terminus/credentials.env`` and is the default,
    because a login should follow the developer rather than the repository.
    ``scope="project"`` writes the repository's ``.env`` for a deliberate,
    repository-local key.

    Sets the value in the process environment too, so the current invocation
    picks it up without a restart.
    """
    if scope == "project":
        path = project_env_write_path()
    elif scope == "global":
        path = global_credentials_path()
    else:
        raise ValueError(f"unknown credential scope: {scope!r}")
    _write_env_file(path, {env_key: value})
    # Deliberately not exported into this process's environment. Every entry
    # point re-reads the credential files, so the very next command sees it -
    # and setting it here would silently outrank a project .env, breaking the
    # precedence this module documents.
    return path


def clear_credential(env_key: str, scope: str = "both") -> tuple[Path, bool]:
    """Remove *env_key*. Returns the file and whether anything was removed.

    ``scope="both"`` is the default because a stale key in either store is the
    confusing case: the user removes it and it keeps working.
    """
    targets: list[Path] = []
    if scope in ("global", "both"):
        targets.append(global_credentials_path())
    if scope in ("project", "both"):
        targets.append(project_env_write_path())
    if scope not in ("global", "project", "both"):
        raise ValueError(f"unknown credential scope: {scope!r}")

    removed = False
    for path in targets:
        if _remove_env_key(path, env_key, delete_if_empty=path == global_credentials_path()):
            removed = True
    if removed:
        os.environ.pop(env_key, None)
    return (targets[0] if targets else global_credentials_path()), removed


def _providers_with_credentials() -> list[str]:
    """Every provider that currently has a usable key, in registry order."""
    from terminus.llm.providers import PROVIDERS

    _ensure_loaded()
    return [
        name
        for name, spec in PROVIDERS.items()
        if spec.env_keys and any(os.environ.get(key) for key in spec.env_keys)
    ]


def resolve_auto_provider() -> tuple[str, str] | None:
    """The provider and model to use instead, when the choice is unambiguous.

    Returns None unless *all* of these hold:

    * the configured provider has no key, so nothing works as configured;
    * exactly one provider does have a key, so there is no guessing involved.

    This is the case that made configuration feel like a ritual: one key sitting
    in ``.env``, a default pointing at a different provider, and a tool that
    refused to connect the two. When there is only one key on the machine there
    is nothing to disambiguate, so the answer is not "run these three commands" -
    it is the provider that key belongs to.

    The model comes from that provider's own registry, never from the configured
    one: pointing groq at an OpenRouter model fails every request with
    model_not_found, which looks exactly like a broken key.
    """
    from terminus.config import CONFIG
    from terminus.llm.providers import PROVIDERS

    llm = CONFIG.setdefault("llm", {})
    configured = str(llm.get("provider", ""))
    # The full loader, project file included. The light _ensure_loaded() only
    # reads the global store, which is right for a hot-path check but wrong here:
    # `config show` and `doctor` run without a startup pass, and a key in the
    # project's .env is the most common place for one to be. Reading only the
    # global file made both report "not ready" for an install that works.
    try:
        load_env_files()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not read credential files: %s", exc)
    have = _providers_with_credentials()
    if not have:
        return None
    if configured in have:
        return None
    if len(have) > 1:
        # Two or more keys: which one to use is a real decision, so make it
        # explicitly rather than picking for them.
        return None
    name = have[0]
    models = list(PROVIDERS[name].models or ())
    return name, (models[0] if models else str(llm.get("model", "")))


def apply_auto_provider() -> str | None:
    """Adopt :func:`resolve_auto_provider` into the live config.

    Returns a short line describing what was chosen, for the caller to show, or
    None when nothing changed. A silent decision is worse than no decision: a
    user who cannot see why their provider is not the one they configured will
    assume the tool ignored them.
    """
    chosen = resolve_auto_provider()
    if chosen is None:
        return None
    import terminus.config as config_module

    provider, model = chosen
    llm = config_module.CONFIG.setdefault("llm", {})
    previous = str(llm.get("provider", ""))
    llm["provider"] = provider
    if model:
        llm["model"] = model
    if previous:
        return (
            f"Using {provider}"
            + (f" via {model}" if model else "")
            + f" instead: it is the only provider with a key. "
            f"To pin a different one: terminus config set llm.provider {previous}"
        )
    return f"Using {provider}" + (f" via {model}" if model else "")


def has_usable_llm_credential() -> bool:
    """True when at least one provider Terminus knows about has a key.

    Deliberately about *any* provider rather than the configured one: a user who
    has a Groq key but is still on the OpenRouter default should be offered the
    choice, not told they are unconfigured.
    """
    from terminus.llm.providers import PROVIDERS
    _ensure_loaded()

    for spec in PROVIDERS.values():
        for key in spec.env_keys:
            if os.environ.get(key):
                return True
    return False


def _ensure_loaded() -> None:
    """Read the *global* credential file into this process, once, on first need.

    A readiness check that only looked at ``os.environ`` reported "no key" in
    any command that had not happened to load the files first, which is every
    standalone command - `doctor` reported FAIL on a working install.

    Only the global store, deliberately. Loading the *project* file here would
    mean a readiness check walked upward from the working directory and pulled
    in whatever `.env` it found above it, so asking "do I have a key?" could
    silently import unrelated settings from a directory the user never named.
    The project's own file is loaded by the startup path, where it belongs.
    """
    global _FILES_LOADED
    if _FILES_LOADED:
        return
    _FILES_LOADED = True  # set first: a broken file must not retry forever
    try:
        global_env = global_credentials_path()
        if not global_env.is_file():
            return
        for key, value in _read_env_file(global_env).items():
            # Never displace a value the process already has: the environment
            # and the project file both outrank the global store.
            os.environ.setdefault(key, value)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not read the global credential store: %s", exc)


def configured_provider_has_credential() -> bool:
    """True when the *currently configured* provider can actually be called."""
    from terminus.config import CONFIG
    from terminus.llm.providers import has_credential

    _ensure_loaded()
    return has_credential(str((CONFIG.get("llm") or {}).get("provider", "")))


def _is_secret(env_key: str) -> bool:
    return env_key not in _NON_SECRET_KEYS
