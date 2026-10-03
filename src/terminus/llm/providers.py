"""Every provider Terminus can talk to, declared once.

Why this module exists
----------------------
The provider list used to be written out five separate times: once as an
if/elif chain in ``llm/factory._build_model_direct``, once as a ``key_env`` map
in ``llm/factory.get_provider_diagnostics``, once as a ``key_names`` tuple in that
same function, once as ``PROVIDER_INFO`` in the CLI, and once as ``ENV_KEYS`` in
the CLI's settings module. Five lists, five chances to forget a provider - and a
provider present in four of them still fails, because the one that actually
constructs the client is the one that matters.

So the declaration lives here, and everything else reads it. Adding a provider is
now one entry in :data:`PROVIDERS` plus, if its wire format is not OpenAI
Chat Completions, one builder function in ``llm/factory``.

What is data and what is code
-----------------------------
The *identity* of a provider - its name, its credential environment variables,
its endpoint, its wire format, the models it is known to serve - is data, and is
in the table. *Constructing* a client is code, because the SDKs genuinely differ
(OpenAI-compatible vs Google Generate Content vs Cohere). Splitting the two is
what keeps the table readable without pretending every provider is the same.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    """One provider's identity. Not its client - see :mod:`terminus.llm.factory`."""

    name: str
    label: str
    env_keys: tuple[str, ...]
    """Credential environment variables, in the order they are tried.

    A provider with more than one accepts either, which is how ``google`` and
    ``google_genai`` share a key without the caller having to know which name the
    user happened to choose.
    """

    base_url: str = ""
    """The base URL to construct a client against, or "" when not applicable."""

    endpoint: str = "provider-defined"
    """The full request URL, for diagnostics. Not used to build the client."""

    api_kind: str = "provider-defined"
    """Which wire format: ``chat_completions`` or ``google_generate_content``."""

    models: tuple[str, ...] = ()
    """Models this provider is known to serve, for the ``models list`` command.

    Empty means "we do not enumerate this provider's catalogue"; the configured
    model is still shown, so the list is never silently wrong, only not
    exhaustive.
    """

    auth_method: str = "api-key"

    key_required: bool = True
    """Whether a missing credential should be an error.

    False for a local server that accepts any key: refusing to start because
    ``localhost`` has no API key would be a rule, not a safety property.
    """

    base_url_env: str = ""
    """Environment variable that overrides :attr:`base_url`.

    For endpoints that are not at a fixed public address - a model server on the
    developer's own machine, on whatever port it happens to be using.
    """

    @property
    def primary_env(self) -> str:
        """The variable named in error messages when nothing is configured."""
        return self.env_keys[0] if self.env_keys else ""

    def resolved_base_url(self) -> str:
        """:attr:`base_url`, or the override if the environment sets one."""
        import os

        override = os.getenv(self.base_url_env, "").strip() if self.base_url_env else ""
        return override or self.base_url


PROVIDERS: dict[str, Provider] = {
    spec.name: spec
    for spec in (
        Provider(
            name="openrouter",
            label="OpenRouter",
            env_keys=("OPENROUTER_API_KEY",),
            base_url="https://openrouter.ai/api/v1",
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_kind="chat_completions",
            models=("poolside/laguna-s-2.1:free",),
        ),
        Provider(
            name="google_genai",
            label="Google GenAI",
            env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_kind="google_generate_content",
            models=("gemini-3.5-flash-lite",),
        ),
        Provider(
            # An alias for the same provider and the same key. Kept because
            # configuration files in the wild use both names.
            name="google",
            label="Google (alias)",
            env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_kind="google_generate_content",
            models=("gemini-3.5-flash-lite",),
        ),
        Provider(
            name="cohere",
            label="Cohere",
            env_keys=("COHERE_API_KEY",),
            models=("command-r-plus-08-2024",),
        ),
        Provider(
            name="groq",
            label="Groq",
            env_keys=("GROQ_API_KEY",),
            base_url="https://api.groq.com/openai/v1",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_kind="chat_completions",
            # Confirmed live against GET /openai/v1/models on 2026-09-28.
            models=("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"),
        ),
        Provider(
            name="openai",
            label="OpenAI",
            env_keys=("OPENAI_API_KEY",),
            base_url="https://api.openai.com/v1",
            endpoint="https://api.openai.com/v1/chat/completions",
            api_kind="chat_completions",
        ),
        Provider(name="fireworks", label="Fireworks", env_keys=("FIREWORKS_API_KEY",)),
        Provider(name="cerebras", label="Cerebras", env_keys=("CEREBRAS_API_KEY",)),
        Provider(name="anthropic", label="Anthropic", env_keys=("ANTHROPIC_API_KEY",)),
        Provider(
            name="ollama",
            label="Ollama (local)",
            env_keys=("OLLAMA_API_KEY",),
            base_url="http://localhost:11434/v1",
            endpoint="http://localhost:11434/v1/chat/completions",
            base_url_env="OLLAMA_BASE_URL",
            api_kind="chat_completions",
            models=("qwen3:8b",),
            key_required=False,
        ),
    )
}

"""Provider name -> :class:`Provider`. The one declaration of the list."""


#: Providers Terminus constructs itself rather than delegating to
#: ``langchain.chat_models.init_chat_model``.
BUILT_LOCALLY = frozenset(
    {"cohere", "openrouter", "google_genai", "google", "groq", "ollama"}
)

NON_LLM_CREDENTIALS = ("QDRANT_API_KEY", "CLUSTER_ENDPOINT")
"""Credentials Terminus uses that are not LLM provider keys.

Kept beside the provider table so "which environment variables does Terminus
read" has one answer, rather than the LLM keys being in one module and the rest
scattered through the Qdrant and diagnostics code.
"""


def get(provider: str) -> Provider | None:
    """The :class:`Provider` for *provider*, or None if it is not known."""
    return PROVIDERS.get((provider or "").lower())


def credential_env_names() -> tuple[str, ...]:
    """Every environment variable that can hold a Terminus credential.

    Ordered by provider, then the non-LLM ones, so ``providers list`` and
    diagnostics enumerate the same set without a second list to maintain.
    """
    names: list[str] = []
    for spec in PROVIDERS.values():
        names.extend(spec.env_keys)
    names.extend(NON_LLM_CREDENTIALS)
    return tuple(dict.fromkeys(names))


def has_credential(provider: str) -> bool:
    """Can *provider* be called right now?

    The single answer to "is this provider usable", so the CLI table, the setup
    wizard and the REPL's start-up check cannot disagree about whether a provider
    is available.
    """
    import os

    spec = get(provider)
    if spec is None:
        return False
    if any(os.environ.get(key) for key in spec.env_keys):
        return True
    return not spec.key_required
