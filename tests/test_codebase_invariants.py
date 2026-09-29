"""Invariants that keep duplicated knowledge from coming back.

Most of what follows is a guard against a specific duplication that existed and
was fixed. They are cheap, they read as statements about the architecture rather
than about any one function, and without them nothing would stop a second copy
of the provider list or the delegation bound being added next time.

Nothing here needs a network call or a credential.
"""

from __future__ import annotations

import ast
import logging
import pathlib

import pytest

from terminus.llm import providers as provider_table

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "terminus"


# --- the provider list is declared exactly once ---------------------------


def test_every_declared_provider_has_a_builder():
    """A provider Terminus routes must be constructible.

    A name in the table with no builder is a provider that appears in
    ``providers list``, offers ``providers login``, and then fails at the first
    call with "Unknown LLM provider" - the worst combination, because the CLI
    invited the user to configure it.
    """
    from terminus.llm.factory import _BUILDERS

    missing = set(provider_table.PROVIDERS) - set(_BUILDERS)
    assert not missing, f"declared but not constructible: {sorted(missing)}"


def test_builders_only_name_declared_providers():
    """The reverse direction: a builder for a provider nobody declared."""
    from terminus.llm.factory import _BUILDERS

    unknown = set(_BUILDERS) - set(provider_table.PROVIDERS)
    assert not unknown, f"built but not declared: {sorted(unknown)}"


def test_every_provider_declares_its_credential():
    """``providers login`` cannot work without an environment variable."""
    for name, spec in provider_table.PROVIDERS.items():
        assert spec.env_keys, f"{name} declares no credential"
        assert all(key.isupper() for key in spec.env_keys), name


def test_provider_lookup_is_case_insensitive():
    assert provider_table.get("OpenRouter") is provider_table.get("openrouter")
    assert provider_table.get("nope") is None


def test_credentials_env_names_are_deduplicated_and_complete():
    names = provider_table.credential_env_names()
    assert len(names) == len(set(names)), "a credential is listed twice"
    for spec in provider_table.PROVIDERS.values():
        for key in spec.env_keys:
            assert key in names
    for key in provider_table.NON_LLM_CREDENTIALS:
        assert key in names


def test_cli_and_router_agree_on_the_provider_list():
    """The CLI used to keep a fourth copy of this list, and it had drifted."""
    from terminus.cli_app import settings

    assert set(settings.ENV_KEYS) == set(provider_table.credential_env_names())


def test_diagnostics_enumerate_exactly_the_declared_credentials(monkeypatch):
    from terminus.llm.factory import get_provider_diagnostics

    reported = set(get_provider_diagnostics()["credential_presence"])
    assert reported == set(provider_table.credential_env_names())


def test_diagnostics_never_include_a_credential_value(monkeypatch):
    """Presence only, always. This is the property that makes diagnostics safe
    to print or paste into an issue."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-super-secret-value")
    from terminus.llm.factory import get_provider_diagnostics

    rendered = str(get_provider_diagnostics())
    assert "gsk-super-secret-value" not in rendered


@pytest.mark.parametrize("name,expected", [
    ("groq", "https://api.groq.com/openai/v1/chat/completions"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
    ("openai", "https://api.openai.com/v1/chat/completions"),
    ("fireworks", "provider-defined"),
])
def test_endpoints_come_from_the_table(name, expected):
    from terminus.llm.factory import _provider_endpoint

    assert _provider_endpoint(name) == expected


def test_a_missing_credential_names_every_accepted_variable(monkeypatch):
    """The message has to tell the user which key to set."""
    from terminus.llm.factory import _build_model_direct, get_llm_config

    # Set to "" rather than deleted: the builder loads the project .env, and
    # load_dotenv(override=False) will not overwrite a variable that is already
    # present - not even an empty one. Deleting would let the real .env put the
    # key back and the test would stop testing anything.
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(key, "")
    with pytest.raises(ValueError) as caught:
        _build_model_direct("gemini-3.5-flash-lite", "google_genai", get_llm_config())
    assert "GEMINI_API_KEY or GOOGLE_API_KEY is not set" in str(caught.value)


def test_an_alias_provider_accepts_either_credential(monkeypatch):
    """``google`` and ``google_genai`` are the same provider under two names, so
    either variable has to work."""
    from terminus.llm.factory import _require_key

    spec = provider_table.get("google")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-the-alias")
    assert _require_key("google", spec) == "from-the-alias"

    monkeypatch.setenv("GEMINI_API_KEY", "from-the-primary")
    assert _require_key("google", spec) == "from-the-primary"


def test_unknown_provider_is_refused():
    from terminus.llm.factory import _build_model_direct, get_llm_config

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        _build_model_direct("m", "not-a-provider", get_llm_config())


# --- the delegation bound is declared exactly once ------------------------


def test_spawner_ceiling_is_the_budget_ceiling():
    """The spawner and the budget used to declare ``8`` independently, so their
    ceilings could drift apart with nothing failing."""
    from terminus.agents.spawn import MAX_CHILD_AGENTS
    from terminus.execution import MAX_CHILDREN_PER_PARENT

    assert MAX_CHILD_AGENTS == MAX_CHILDREN_PER_PARENT
    from terminus.agents.spawn import AgentSpawner

    spawner = AgentSpawner(max_children=999)
    assert spawner.max_children == MAX_CHILDREN_PER_PARENT, (
        "the spawner must not be able to exceed the budget that enforces the bound"
    )


# --- failure classification survives re-wrapping -------------------------


def test_quota_horizon_survives_rewrapping():
    """ProviderCallError is re-classified when it propagates. Copying the field
    list by hand once dropped ``reset_in_seconds``, leaving a failure marked
    ``exhausted`` with no idea when the route becomes usable again."""
    from terminus.tasks.errors import (
        FailureInfo,
        ProviderCallError,
        classify_failure,
    )

    inner = FailureInfo(
        category="rate_limit",
        retryable=True,
        status_code=429,
        message="daily limit reached",
        provider="openrouter",
        model="some/model",
        exhausted=True,
        reset_in_seconds=21_600.0,
    )
    rewrapped = classify_failure(ProviderCallError("openrouter", "some/model", inner))

    assert rewrapped.exhausted is True
    assert rewrapped.reset_in_seconds == 21_600.0
    assert rewrapped.provider == "openrouter"
    assert rewrapped.status_code == 429


def test_retry_in_text_is_parsed_once_for_both_consumers():
    """``_retry_after`` and the reset horizon used to carry separate copies of
    the same regex, which is how they ended up disagreeing."""
    from terminus.tasks.errors import _RETRY_IN_TEXT

    assert _RETRY_IN_TEXT.search("rate limited, retry in 30s")
    assert _RETRY_IN_TEXT.search("retry after 2.5 s")
    assert not _RETRY_IN_TEXT.search("no hint here")


# --- logging -------------------------------------------------------------


def test_set_log_level_reaches_module_loggers():
    """get_logger used to pin INFO on every module, which silently defeated
    ``terminus --log-level DEBUG`` - a logger with its own level ignores the
    root's."""
    from terminus.observability.logging import get_logger, set_log_level

    previous = logging.getLogger().level
    try:
        set_log_level("DEBUG")
        assert get_logger("terminus.tests.anything").isEnabledFor(logging.DEBUG)
        set_log_level("WARNING")
        assert not get_logger("terminus.tests.anything").isEnabledFor(logging.INFO)
    finally:
        set_log_level(previous)


def test_get_logger_does_not_pin_a_level():
    from terminus.observability.logging import get_logger, set_log_level

    set_log_level("ERROR")
    assert get_logger("terminus.tests.pinned").getEffectiveLevel() == logging.ERROR


# --- ownership -----------------------------------------------------------


def test_project_ownership_releases_when_the_body_raises(tmp_path):
        """A failure inside the run must not leave a project unownable."""
        from terminus.ownership import current_ownership
        from terminus.tasks.orchestrator import project_ownership

        # The lock directory is derived from the database path, so this must be a
        # real path under tmp_path. A bare ":memory:" has a parent of ".", which
        # would put the lock file in whatever directory the test ran from - in
        # practice, the repository root.
        class _Store:
            db_path = str(tmp_path / "tasks.db")

        project_id = "invariant-test-project"
        with pytest.raises(RuntimeError, match="boom"):
            with project_ownership(_Store(), project_id):
                assert current_ownership().holds(project_id)
                raise RuntimeError("boom")
        assert not current_ownership().holds(project_id)


# --- hygiene the audit found and fixed -----------------------------------


def test_no_mojibake_in_source_comments_or_logs():
    """A UTF-8 en dash had been re-encoded as cp1252 in four files, including
    inside logger messages, so it reached the terminal."""
    suspects = "\u0393\u00c7\u00f6\ufffd"
    offenders = []
    for path in SRC.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            offenders.append(f"{path}: not valid utf-8")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(char in line for char in suspects):
                offenders.append(f"{path}:{number}")
    assert not offenders, "mojibake found: " + ", ".join(offenders)


#: Modules a new engineer has to read to understand the main path. These are the
#: ones where a docstring pays for itself, so they are required. The rule is
#: deliberately not "every module" - forcing a docstring onto a forty-line
#: retriever adapter produces restating-the-code noise, which is worse than
#: having none.
SPINE_MODULES = (
    "config.py",
    "env.py",
    "cli.py",
    "agent/factory.py",
    "agent/orchestrator.py",
    "memory/short_term.py",
    "memory/session.py",
    "tasks/approval.py",
)


@pytest.mark.parametrize("relative", SPINE_MODULES)
def test_spine_modules_explain_themselves(relative):
    """Each module on the main path states what it is and why it exists.

    Asserted non-empty rather than wordy on purpose. A docstring that only
    restates the code is the thing this audit removed, and it should not be
    invited back in.
    """
    text = (SRC / relative).read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(text)) or ""
    assert len(docstring.strip()) >= 40, f"{relative} needs a real module docstring"
