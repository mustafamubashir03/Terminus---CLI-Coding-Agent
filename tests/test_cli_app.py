"""Tests for the Typer CLI.

Two rules keep these tests honest:

* Anything that could write configuration runs in ``tmp_path`` with a
  monkeypatched cwd. ``config set`` and ``providers login`` would otherwise edit
  the developer's real ``config.yaml`` and ``.env``.
* No test needs a credential or a network call. Every assertion is about parsing,
  wiring, precedence and formatting - the parts that can be checked
  deterministically.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from terminus.cli_app import app
from terminus.cli_app import formatting, settings

runner = CliRunner()


def run(*args: str, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


def only_json(result) -> object:
    """Assert stdout is exactly one JSON document and return it."""
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


# --- structure and help ---------------------------------------------------


def test_help_lists_every_command_group():
    result = run("--help")
    assert result.exit_code == 0
    for group in ("agent", "providers", "models", "sessions", "tools", "skills", "config", "db", "index"):
        assert group in result.output, f"{group} missing from root help"


def test_short_help_flag_works():
    assert run("-h").exit_code == 0


def test_version_flag_and_command_agree():
    from_importlib = run("version", "--json")
    from_flag = run("--version")
    assert only_json(from_importlib)["version"]
    assert only_json(from_importlib)["version"] in from_flag.output


def test_unknown_command_is_rejected():
    result = run("definitely-not-a-command")
    assert result.exit_code != 0


def test_every_subcommand_has_help():
    """No subcommand should be a dead end."""
    groups = {
        "agent": [],
        "providers": ["list", "status", "login", "logout"],
        "models": ["list", "set"],
        "sessions": ["list", "new", "switch"],
        "tools": ["list", "inspect"],
        "skills": ["list", "agents"],
        "config": ["list", "get", "set"],
        "db": ["path"],
        "index": ["status"],
    }
    for group, leaves in groups.items():
        assert run(group, "--help").exit_code == 0, group
        for leaf in leaves:
            assert run(group, leaf, "--help").exit_code == 0, f"{group} {leaf}"


# --- json contract ---------------------------------------------------------


@pytest.mark.parametrize("args", [
    ("version",),
    ("providers", "list"),
    ("providers",),
    ("models", "list"),
    ("models",),
    ("tools", "list"),
    ("tools",),
    ("db", "path"),
    ("db",),
    ("sessions",),
    ("config",),
    ("skills",),
])
def test_json_output_parses_and_has_no_markup(args):
    result = run(*args, "--json")
    assert result.exit_code == 0, result.output
    json.loads(result.output)  # one clean document
    assert "[bold" not in result.output, "Rich markup leaked into JSON output"


def test_global_json_flag_also_works():
    """`terminus --json tools list` must behave like `terminus tools list --json`."""
    assert json.loads(run("--json", "tools", "list").output)["count"] >= 0


def test_db_path_is_usable_by_a_script():
    """`db path` must print only the path, so it can be piped into sqlite3."""
    result = run("db", "path")
    assert result.exit_code == 0
    assert result.output.strip()
    assert "terminus.db" in result.output


# --- config ---------------------------------------------------------------


def test_config_get_resolves_a_dotted_key():
    result = run("config", "get", "llm.model")
    assert result.exit_code == 0
    assert "llm.model" in result.output


def test_config_get_rejects_unknown_key_without_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("config", "get", "llm.nonexistent")
    assert result.exit_code == 2
    assert not (tmp_path / "config.yaml").exists()


def test_config_set_rejects_unknown_key_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("config", "set", "llm.nonexistent", "1")
    assert result.exit_code == 2
    assert not (tmp_path / "config.yaml").exists(), "a rejected key must not create or modify a config file"


def test_config_set_persists_a_known_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("config", "set", "llm.max_retries", "7")
    assert result.exit_code == 0
    written = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "max_retries" in written


def test_config_set_preserves_comments_and_formatting(tmp_path, monkeypatch):
    """Regression: a write must not re-serialise the file.

    An earlier implementation round-tripped through yaml.safe_dump, which
    deleted every comment in the real config.yaml. The comments explain why the
    tuning values are what they are, so a settings command that erases them is
    destructive even though the parsed result is identical.
    """
    monkeypatch.chdir(tmp_path)
    original = (
        "llm:\n"
        "  provider: openrouter\n"
        "  # why streaming is on: the REPL prints per-token chunks\n"
        "  streaming: true\n"
        "  max_retries: 0\n"
    )
    (tmp_path / "config.yaml").write_text(original, encoding="utf-8")
    assert run("config", "set", "llm.max_retries", "5").exit_code == 0
    updated = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "the REPL prints per-token chunks" in updated, "comment was lost"
    assert "max_retries: 5" in updated
    assert "provider: openrouter" in updated, "an unrelated key was lost"


def test_config_set_targets_the_current_directory(tmp_path, monkeypatch):
    """A write goes to cwd/config.yaml, matching what load_config reads first."""
    monkeypatch.chdir(tmp_path)
    assert run("config", "set", "llm.max_retries", "5").exit_code == 0
    assert (tmp_path / "config.yaml").exists()


def test_config_set_adds_a_missing_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("llm:\n  provider: openrouter\n", encoding="utf-8")
    assert run("config", "set", "llm.request_timeout_seconds", "42").exit_code == 0
    assert _load(tmp_path / "config.yaml")["llm"]["request_timeout_seconds"] == 42
    assert _load(tmp_path / "config.yaml")["llm"]["provider"] == "openrouter"


def test_config_set_adds_a_missing_top_level_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("llm:\n  provider: openrouter\n", encoding="utf-8")
    assert run("config", "set", "memory.max_messages", "7").exit_code == 0
    assert _load(tmp_path / "config.yaml")["memory"]["max_messages"] == 7


def test_config_set_refuses_a_non_scalar(tmp_path, monkeypatch):
    """Writing a dict would need a whole block; refuse rather than corrupt."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("llm:\n  provider: openrouter\n", encoding="utf-8")
    result = run("config", "set", "llm.reasoning", '{"effort": "high"}')
    assert result.exit_code != 0
    assert "provider: openrouter" in (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_config_set_keeps_json_types(tmp_path, monkeypatch):
    """`true` and `3` must round-trip as a bool and an int, not as strings."""
    monkeypatch.chdir(tmp_path)
    run("config", "set", "llm.streaming", "false")
    run("config", "set", "llm.max_retries", "3")
    document = json.loads(json.dumps(_load(tmp_path / "config.yaml")))
    assert document["llm"]["streaming"] is False
    assert document["llm"]["max_retries"] == 3


def _load(path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_config_list_reports_its_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = only_json(run("config", "list", "--json"))
    assert "config" in payload and "source_kind" in payload


# --- model override: one run, never persisted ---------------------------


def test_model_override_changes_then_restores():
    from terminus.config import CONFIG

    before = dict(CONFIG["llm"])
    with pytest.raises(RuntimeError):
        from terminus.cli_app.commands import model_override

        with model_override("some/model-id", "groq"):
            assert CONFIG["llm"]["model"] == "some/model-id"
            assert CONFIG["llm"]["provider"] == "groq"
            raise RuntimeError("boom")
    assert dict(CONFIG["llm"]) == before, "override must be undone even when the body raises"


def test_model_override_takes_the_model_id_whole():
    """Provider-prefixed ids contain '/'; splitting one would pick a wrong model."""
    from terminus.config import CONFIG

    from terminus.cli_app.commands import model_override

    with model_override("poolside/laguna-s-2.1:free"):
        assert CONFIG["llm"]["model"] == "poolside/laguna-s-2.1:free"


def test_model_override_does_not_write_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from terminus.cli_app.commands import model_override

    with model_override("some/model-id", "groq"):
        pass
    assert not (tmp_path / "config.yaml").exists()


def test_agent_without_prompt_is_a_usage_error():
    result = run("agent", "--model", "some/model")
    assert result.exit_code != 0


# --- providers and credentials -------------------------------------------


def test_providers_list_reports_presence_not_secrets():
    payload = only_json(run("providers", "list", "--json"))
    assert payload["providers"]
    for entry in payload["providers"]:
        assert isinstance(entry["api_key_configured"], bool)
        assert "api_key" not in entry or entry["api_key"] in (None, "")


def test_diagnostics_never_expose_a_key():
    """The whole point of presence-only is that a value cannot leak from here."""
    from terminus.llm.factory import get_provider_diagnostics

    rendered = json.dumps(get_provider_diagnostics(), default=str)
    for marker in ("api_key\":", "sk-", "Bearer "):
        assert marker not in rendered, f"diagnostics leaked {marker!r}"


def test_login_rejects_an_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("providers", "login", "-p", "nope", "--api-key", "x")
    assert result.exit_code == 2
    assert not (tmp_path / ".env").exists()


def test_login_rejects_an_unsupported_method(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("providers", "login", "-p", "groq", "-m", "oauth", "--api-key", "x")
    assert result.exit_code == 2
    assert not (tmp_path / ".env").exists()


def test_login_writes_the_key_to_env_not_config(tmp_path, monkeypatch):
    """A credential must never land in a file people commit."""
    monkeypatch.chdir(tmp_path)
    result = run("providers", "login", "-p", "groq", "--api-key", "test-key-value")
    assert result.exit_code == 0
    assert "test-key-value" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert not (tmp_path / "config.yaml").exists(), "credential must not reach config.yaml"


def test_login_replaces_an_existing_key_rather_than_appending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "first")
    run("providers", "login", "-p", "groq", "--api-key", "second")
    lines = [ln for ln in (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
             if ln.startswith("GROQ_API_KEY=")]
    assert lines == ["GROQ_API_KEY=second"]


def test_login_prompts_without_echoing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("providers", "login", "-p", "cohere", input="prompted-secret\n")
    assert result.exit_code == 0
    assert "prompted-secret" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert "prompted-secret" not in result.output, "a prompted secret must not be echoed back"


def test_logout_removes_the_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "to-remove")
    result = run("providers", "logout", "-p", "groq")
    assert result.exit_code == 0
    assert "to-remove" not in (tmp_path / ".env").read_text(encoding="utf-8")


def test_logout_on_a_missing_key_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run("providers", "logout", "-p", "groq").exit_code == 0


def test_login_with_an_empty_key_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("providers", "login", "-p", "groq", input="\n")
    assert result.exit_code == 1
    assert not (tmp_path / ".env").exists()


# --- models ---------------------------------------------------------------


def test_models_list_is_derived_from_the_live_route():
    """No hard-coded model table: the active row must match the diagnostics."""
    from terminus.llm.factory import get_provider_diagnostics

    diagnostics = get_provider_diagnostics()
    payload = only_json(run("models", "list", "--json"))
    active = next(m for m in payload["models"] if m["role"] == "active")
    assert active["provider"] == diagnostics["provider"]
    assert active["model"] == diagnostics["requested_model"]


def test_models_list_filters_by_provider():
    payload = only_json(run("models", "list", "groq", "--json"))
    assert payload["models"]
    assert {m["provider"] for m in payload["models"]} == {"groq"}


def test_models_set_takes_the_id_whole(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("models", "set", "poolside/laguna-s-2.1:free", "--provider", "openrouter")
    assert result.exit_code == 0
    llm = _load(tmp_path / "config.yaml")["llm"]
    assert llm["model"] == "poolside/laguna-s-2.1:free", "the id must not be split on '/'"
    assert llm["provider"] == "openrouter"


def test_models_set_rejects_an_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("models", "set", "some-model", "--provider", "nope")
    assert result.exit_code == 2
    assert not (tmp_path / "config.yaml").exists()


# --- sessions, tools, skills, index --------------------------------------


@pytest.fixture()
def isolated_sessions(tmp_path, monkeypatch):
    """Point the session store at a temp directory.

    The real session helpers read and write files under the configured memory
    path. Without this, a test that merely *reads* the current session creates
    one, and one that switches overwrites the developer's - which then fails an
    unrelated session-isolation test in the same run.
    """
    from terminus.config import CONFIG

    memory = dict(CONFIG["memory"])
    CONFIG["memory"]["db_path"] = str(tmp_path / "memory" / "terminus.db")
    yield tmp_path
    CONFIG["memory"].clear()
    CONFIG["memory"].update(memory)


def test_sessions_shows_a_current_session_and_a_database(isolated_sessions):
    payload = only_json(run("sessions", "--json"))
    assert payload["current"]
    assert "terminus.db" in payload["database"]


def test_sessions_new_creates_a_session(isolated_sessions):
    first = only_json(run("sessions", "new", "--json"))["current"]
    second = only_json(run("sessions", "new", "--json"))["current"]
    assert first != second, "each new session needs its own id"


def test_sessions_switch_reports_success(isolated_sessions):
    assert run("sessions", "switch", "test-session-id").exit_code == 0
    assert only_json(run("sessions", "--json"))["current"] == "test-session-id"


def test_tools_list_matches_the_agent_registry():
    from terminus.agent.factory import tools_by_name

    payload = only_json(run("tools", "list", "--json"))
    assert payload["count"] == len(tools_by_name())


def test_tools_inspect_shows_a_real_tool():
    from terminus.agent.factory import tools_by_name

    name = next(iter(sorted(tools_by_name())))
    payload = only_json(run("tools", "inspect", name, "--json"))
    assert payload["name"] == name


def test_tools_inspect_rejects_an_unknown_tool():
    result = run("tools", "inspect", "no-such-tool")
    assert result.exit_code == 2
    assert "tools list" in result.output, "the error should point at the command that lists them"


def test_skills_list_uses_the_same_service_as_the_repl():
    from terminus.skills.skill_tools import describe_skills

    rendered = run("skills", "list").output
    expected = describe_skills(None)
    # Rich draws a panel border, so compare content, not the frame. Every skill
    # the service reports must reach the screen, or the two views have drifted.
    assert "Installed skills (13)" in " ".join(rendered.split())
    for line in expected.splitlines():
        entry = line.strip()
        if not entry.startswith("*"):
            continue
        name = entry.lstrip("* ").split(" (")[0].strip()
        assert name in rendered, f"{name} missing from `skills list`"


def test_config_set_survives_a_model_id_containing_a_colon(tmp_path, monkeypatch):
    """Regression: yaml.safe_dump appends a `...` marker to such values.

    The first write parsed, so nothing looked wrong. The second write landed
    after that marker and produced a file no longer loadable - and
    load_config() raises on a bad file, so the user is locked out.
    """
    monkeypatch.chdir(tmp_path)
    run("config", "set", "llm.model", "poolside/laguna-s-2.1:free")
    assert run("config", "set", "llm.provider", "openrouter").exit_code == 0
    from terminus.config import load_config

    assert load_config()["llm"]["model"] == "poolside/laguna-s-2.1:free"


def test_skills_agents_shows_roles():
    assert run("skills", "agents").exit_code == 0


def test_index_status_is_read_only(tmp_path, monkeypatch):
    """`index status` must never be able to rebuild a shared collection."""
    from terminus.context.indexers import migrate as migrate_module

    seen: dict[str, bool] = {}
    original = migrate_module.migrate_collection

    def spy(*args, **kwargs):
        seen["rebuild"] = kwargs.get("rebuild_shared_collection")
        return original(*args, **kwargs)

    monkeypatch.setattr(migrate_module, "migrate_collection", spy)
    assert run("index", "status").exit_code == 0
    assert seen["rebuild"] is False


# --- the REPL handoff ----------------------------------------------------


def test_bare_invocation_starts_the_repl(monkeypatch):
    """`terminus` with no arguments must reach the session, not print help."""
    started: list[bool] = []
    monkeypatch.setattr(formatting, "launch_repl", lambda: started.append(True) or 0)
    monkeypatch.setattr("terminus.cli.run", lambda: 0)
    assert run().exit_code == 0
    assert started == [True]


def test_agent_bare_starts_the_repl(monkeypatch):
    started: list[bool] = []
    monkeypatch.setattr(formatting, "launch_repl", lambda: started.append(True) or 0)
    assert run("agent").exit_code == 0
    assert started == [True]


def test_launch_repl_treats_end_of_input_as_a_clean_exit(monkeypatch):
    """`terminus < /dev/null` should not look like a crash."""

    def raises_eof():
        raise EOFError

    monkeypatch.setattr("terminus.cli.run", raises_eof)
    assert formatting.launch_repl() == 0


def test_launch_repl_treats_interrupt_as_a_clean_exit(monkeypatch):
    def raises_interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr("terminus.cli.run", raises_interrupt)
    assert formatting.launch_repl() == 0


# --- settings unit tests -------------------------------------------------


def test_has_path_walks_nested_keys():
    assert settings.has_path("llm.model")
    assert not settings.has_path("llm.definitely_not_here")
    assert not settings.has_path("definitely.not.here")


def test_get_value_reads_through():
    assert settings.get_value("llm.provider") is not None


def test_credential_status_never_returns_values():
    for entry in settings.credential_status():
        assert set(entry) == {"name", "configured", "source"}
        assert isinstance(entry["configured"], bool)


def test_mask_hides_a_value():
    assert settings.mask("secret") == "********"
    assert settings.mask("") == ""
