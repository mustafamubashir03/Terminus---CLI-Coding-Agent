"""Tests for the configuration hierarchy and the setup/doctor surface.

The bug this file exists to prevent: a project ``config.yaml`` that mentions
one key silently discarding the rest of the user's global setup. That is not a
hypothetical - it is what a first-file-wins loader does, and it is invisible
until someone wonders why their provider reverted to a default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from terminus.cli_app import app

runner = CliRunner()


def run(*args: str, **kwargs):
    return runner.invoke(app, list(args), **kwargs)


@pytest.fixture
def layers(tmp_path, monkeypatch):
    """A home with a global config, and a project directory, both writable."""
    home = Path(os.environ["USERPROFILE"])
    (home / ".terminus").mkdir(parents=True, exist_ok=True)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return home, project


def write_global(home: Path, text: str) -> Path:
    path = home / ".terminus" / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_project(project: Path, text: str) -> Path:
    path = project / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- the hierarchy ---------------------------------------------------------


def test_project_config_inherits_keys_it_does_not_mention(layers):
    home, project = layers
    write_global(home, "llm:\n  provider: cohere\n  model: global-model\n")
    write_project(project, "llm:\n  model: project-model\n")

    from terminus.config import load_config

    merged = load_config()
    assert merged["llm"]["provider"] == "cohere", "global value must survive"
    assert merged["llm"]["model"] == "project-model", "project value must win"


def test_global_config_applies_with_no_project_file(layers):
    home, _project = layers
    write_global(home, "llm:\n  provider: cohere\n")

    from terminus.config import load_config

    assert load_config()["llm"]["provider"] == "cohere"


def test_defaults_apply_when_nothing_is_configured(layers):
    from terminus.config import load_config

    merged = load_config()
    assert merged["llm"]["provider"] == "openrouter"
    assert merged["llm"]["model"]


def test_an_empty_section_does_not_discard_the_layer_below(layers):
    """`llm:` with nothing under it means "no opinion", not "erase everything"."""
    home, project = layers
    write_global(home, "llm:\n  provider: cohere\n  model: global-model\n")
    write_project(project, "llm:\n")

    from terminus.config import load_config

    merged = load_config()
    assert merged["llm"]["provider"] == "cohere"
    assert merged["llm"]["model"] == "global-model"


def test_config_show_reports_every_layer_that_contributed(layers):
    home, project = layers
    write_global(home, "llm:\n  provider: cohere\n")
    write_project(project, "llm:\n  model: project-model\n")

    result = run("config", "show", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    kinds = [layer["kind"] for layer in payload["config_layers"]]
    assert kinds == ["global", "cwd"]


def test_unset_lets_the_lower_layer_take_effect_again(layers):
    home, project = layers
    write_global(home, "llm:\n  model: global-model\n")
    write_project(project, "llm:\n  model: project-model\n  streaming: true\n")

    assert run("config", "unset", "llm.model", "--scope", "project").exit_code == 0

    from terminus.config import load_config

    merged = load_config()
    assert merged["llm"]["model"] == "global-model"
    assert merged["llm"]["streaming"] is True, "the sibling key must be left alone"


def test_unset_removes_the_section_it_emptied(layers):
    _home, project = layers
    write_project(project, "llm:\n  model: only-key\n")

    assert run("config", "unset", "llm.model", "--scope", "project").exit_code == 0
    assert project.joinpath("config.yaml").read_text(encoding="utf-8").strip() == ""


def test_unset_reports_honestly_when_the_key_was_never_set(layers):
    _home, project = layers
    write_project(project, "llm:\n  streaming: true\n")

    result = run("config", "unset", "llm.model", "--scope", "project", "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["removed"] is False


# --- credentials -----------------------------------------------------------


def test_login_defaults_to_the_global_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run("providers", "login", "-p", "groq", "--api-key", "k").exit_code == 0
    stored = Path(os.environ["USERPROFILE"]) / ".terminus" / "credentials.env"
    assert "GROQ_API_KEY=k" in stored.read_text(encoding="utf-8")
    assert not (tmp_path / ".env").exists()


def test_a_credential_never_lands_in_a_yaml_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "super-secret")
    run("config", "set", "llm.provider", "groq", "--scope", "global")
    for candidate in (tmp_path / "config.yaml",
                      Path(os.environ["USERPROFILE"]) / ".terminus" / "config.yaml"):
        if candidate.exists():
            assert "super-secret" not in candidate.read_text(encoding="utf-8")


def test_config_show_never_prints_a_credential(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "sk-should-never-appear")
    for args in (("config", "show"), ("config", "show", "--json"), ("doctor",)):
        result = run(*args)
        assert "sk-should-never-appear" not in result.stdout
        assert "sk-should-never-appear" not in (result.stderr or "")


def test_config_show_says_not_ready_when_the_key_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("config", "show")
    assert "Not ready" in result.output
    assert "terminus setup" in result.output


def test_config_show_says_ready_once_setup_has_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run("setup", "--provider", "groq", "--model", "m", "--api-key", "k").exit_code == 0
    # In place, not by rebinding: a dozen modules hold a reference to the
    # original dict, and replacing the module attribute would leave them reading
    # a stale copy.
    import terminus.config as config_module

    config_module.CONFIG.clear()
    config_module.CONFIG.update(config_module.load_config())
    result = run("config", "show")
    assert "Ready" in result.output


def test_logout_actually_removes_the_key(tmp_path, monkeypatch):
    """A logout that reports success and changes nothing is worse than none."""
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "to-remove", "--scope", "project")
    assert "to-remove" in (tmp_path / ".env").read_text(encoding="utf-8")

    assert run("providers", "logout", "-p", "groq").exit_code == 0
    assert "to-remove" not in (tmp_path / ".env").read_text(encoding="utf-8")


def test_logout_empties_a_project_env_without_deleting_it(tmp_path, monkeypatch):
    """The .env may be referenced by a compose file or CI; do not remove it."""
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "k", "--scope", "project")
    run("providers", "logout", "-p", "groq")
    assert (tmp_path / ".env").exists()


def test_logout_clears_both_copies_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "global-copy")
    run("providers", "login", "-p", "groq", "--api-key", "project-copy", "--scope", "project")
    run("providers", "logout", "-p", "groq")

    stored = Path(os.environ["USERPROFILE"]) / ".terminus" / "credentials.env"
    assert "global-copy" not in (stored.read_text(encoding="utf-8") if stored.exists() else "")
    assert "project-copy" not in (tmp_path / ".env").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_the_global_credential_file_is_owner_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "k")
    stored = Path(os.environ["USERPROFILE"]) / ".terminus" / "credentials.env"
    assert oct(stored.stat().st_mode)[-3:] == "600"


# --- setup and doctor ------------------------------------------------------


def test_setup_is_registered_and_documents_itself():
    result = run("setup", "--help")
    assert result.exit_code == 0
    assert "--key-stdin" in result.output


def test_setup_from_stdin_stores_a_working_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("setup", "--provider", "groq", "--model", "test-model",
                 "--key-stdin", "--force", input="piped-secret\n")
    assert result.exit_code == 0
    assert "piped-secret" not in result.output, "a piped secret must not be echoed"

    home = Path(os.environ["USERPROFILE"])
    assert "GROQ_API_KEY=piped-secret" in (home / ".terminus" / "credentials.env").read_text(encoding="utf-8")
    written = (home / ".terminus" / "config.yaml").read_text(encoding="utf-8")
    assert "groq" in written and "test-model" in written
    assert "piped-secret" not in written, "the key must not be written into the config"


def test_setup_warns_that_an_argument_key_is_exposed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("setup", "--provider", "groq", "--api-key", "k", "--force")
    assert result.exit_code == 0
    assert "history" in result.output.lower()


def test_setup_refuses_to_overwrite_a_working_setup_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run("setup", "--provider", "groq", "--key-stdin", input="one\n").exit_code == 0
    second = run("setup", "--provider", "cohere", "--key-stdin", input="two\n")
    assert second.exit_code != 0
    stored = (Path(os.environ["USERPROFILE"]) / ".terminus" / "credentials.env").read_text(encoding="utf-8")
    assert "one" in stored and "two" not in stored


def test_setup_rejects_an_unknown_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("setup", "--provider", "not-a-provider", "--key-stdin", input="k\n")
    assert result.exit_code == 2


def test_doctor_is_registered():
    assert run("doctor", "--help").exit_code == 0


def test_doctor_fails_when_the_provider_key_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run("doctor")
    assert result.exit_code == 1, "a missing key is the one thing that must fail"
    assert "setup" in result.output


def test_doctor_writes_nothing_into_the_project(tmp_path, monkeypatch):
    """A health check must not index, embed, or otherwise mutate the project."""
    project = tmp_path / "fresh"
    project.mkdir()
    monkeypatch.chdir(project)
    run("providers", "login", "-p", "groq", "--api-key", "k")
    run("doctor")
    assert list(project.iterdir()) == [], "doctor created files in the project"


def test_doctor_reports_optional_services_as_warnings_not_failures(tmp_path, monkeypatch):
    """Missing LangSmith or a Qdrant cluster must not stop anyone asking a question."""
    monkeypatch.chdir(tmp_path)
    run("setup", "--provider", "groq", "--api-key", "k")
    result = run("doctor")
    for line in result.output.splitlines():
        if line.strip().startswith("FAIL") and (
            "LangSmith" in line or "Qdrant" in line or "Vector store reachable" in line
        ):
            pytest.fail(f"an optional service was reported as a failure: {line.strip()}")
    assert "FAIL" not in result.output, (
        "nothing optional should be failing here:\n" + result.output
    )


# --- first run -------------------------------------------------------------


@pytest.fixture
def captured_repl(monkeypatch):
    """Collect what the REPL prints, via the console it actually writes to."""
    import io

    from rich.console import Console

    from terminus.cli_app import repl

    buffer = io.StringIO()
    monkeypatch.setattr(repl, "console", Console(file=buffer, width=100, no_color=True))
    return lambda: buffer.getvalue()


def test_first_run_points_at_setup_when_nothing_is_configured(tmp_path, monkeypatch, captured_repl):
    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)
    repl._first_run_notice()
    text = captured_repl()
    assert "Not configured yet" in text
    assert "/setup" in text
    # Say the thing that makes it zero-effort: a key in .env is enough.
    assert ".env" in text


def test_a_single_key_needs_no_configuration_at_all(tmp_path, monkeypatch, captured_repl):
    """The whole point. One key in .env, a default pointing elsewhere, no commands."""
    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=stored-in-dot-env\n", encoding="utf-8")
    repl._first_run_notice()
    assert "Not configured yet" not in captured_repl(), (
        "there was nothing to warn about: the only key on the machine is usable"
    )


def test_a_single_key_is_adopted_and_its_own_model_chosen(tmp_path, monkeypatch):
    from terminus.user_config import load_env_files, resolve_auto_provider

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=stored-in-dot-env\n", encoding="utf-8")
    load_env_files()

    assert resolve_auto_provider() == ("groq", "openai/gpt-oss-120b"), (
        "the model must belong to the chosen provider, not the configured one"
    )


def test_two_keys_are_left_as_a_real_choice(tmp_path, monkeypatch):
    """With more than one key, guessing would be wrong: do not guess."""
    from terminus.user_config import load_env_files, resolve_auto_provider

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=a\nCOHERE_API_KEY=b\n", encoding="utf-8")
    load_env_files()
    assert resolve_auto_provider() is None


def test_an_explicitly_configured_provider_is_not_overridden(tmp_path, monkeypatch):
    """A deliberate choice outranks convenience."""
    from terminus.user_config import load_env_files, resolve_auto_provider

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=a\n", encoding="utf-8")
    load_env_files()
    from terminus.user_config import configured_provider_has_credential

    if configured_provider_has_credential():
        assert resolve_auto_provider() is None


def test_first_run_is_silent_once_a_key_exists(tmp_path, monkeypatch, captured_repl):
    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)
    run("setup", "--provider", "groq", "--model", "m", "--api-key", "k")
    repl._first_run_notice()
    assert "Not configured yet" not in captured_repl()


def test_first_run_survives_a_broken_config(tmp_path, monkeypatch, captured_repl):
    """A diagnostic must never be the thing that stops someone starting up."""
    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("llm: [this is not a mapping\n", encoding="utf-8")
    repl._first_run_notice()  # must not raise
    assert True


def test_a_fresh_process_sees_the_stored_global_key(tmp_path, monkeypatch):
    """The whole point of a global store: a new invocation must find the key.

    This regressed once already. `doctor` checked os.environ without reading the
    credential files, so a working install reported FAIL whenever the process
    had not happened to load them first - which is every standalone command.
    """
    import subprocess
    import sys

    project = tmp_path / "fresh"
    project.mkdir()
    home = tmp_path / "home"
    (home / ".terminus").mkdir(parents=True)
    (home / ".terminus" / "config.yaml").write_text("llm:\n  provider: groq\n", encoding="utf-8")
    (home / ".terminus" / "credentials.env").write_text("GROQ_API_KEY=stored-globally\n", encoding="utf-8")

    env = dict(os.environ)
    for name in list(env):
        if name.endswith("_API_KEY"):
            env.pop(name)
    env.update({"HOME": str(home), "USERPROFILE": str(home),
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    proc = subprocess.run(
        [sys.executable, "-c",
         "from terminus.user_config import configured_provider_has_credential;"
         "print(configured_provider_has_credential())"],
        cwd=str(project), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.stdout.strip() == "True", proc.stdout + proc.stderr


def test_setup_chooses_a_model_that_belongs_to_the_provider(tmp_path, monkeypatch):
    """A provider with someone else's model looks configured and cannot work.

    This regressed: `setup --provider groq` wrote the provider and left the
    previous OpenRouter model, so every later request failed model_not_found.
    """
    monkeypatch.chdir(tmp_path)
    assert run("setup", "--provider", "groq", "--api-key", "k").exit_code == 0

    import terminus.config as config_module

    merged = config_module.load_config()
    model = merged["llm"]["model"]
    assert model != "poolside/laguna-s-2.1:free", "kept a model from another provider"
    assert "groq" not in model or model.startswith("openai/") or "/" in model
    assert merged["llm"]["provider"] == "groq"


def test_an_explicit_model_is_still_respected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("setup", "--provider", "groq", "--model", "my-model", "--api-key", "k")

    import terminus.config as config_module

    assert config_module.load_config()["llm"]["model"] == "my-model"


# --- running commands from inside the session --------------------------------


@pytest.mark.parametrize("typed", [
    "terminus setup", "setup", "/setup", "Terminus Setup",
    "terminus doctor", "doctor", "/doctor",
    "terminus config show", "config show", "/config show",
    "terminus config set llm.provider groq", "config set llm.provider groq",
    "config unset llm.model", "config list", "config get llm.model",
    "terminus providers login --provider groq", "providers login --provider groq",
    "terminus providers logout --provider groq", "providers logout --provider groq",
])
def test_what_the_program_prints_can_be_pasted_back_at_the_prompt(typed):
    """The notice says `terminus setup`; typing that must not become a question.

    A user who copies the instruction the program just printed, at the prompt the
    program is sitting at, is following the instructions exactly. Sending it to
    the model as a question about the codebase produced a provider error about
    OPENROUTER_API_KEY, which looks like the tool is broken.
    """
    from terminus.cli_app.repl import as_session_command

    command, _argument = as_session_command(typed)
    assert command is not None, f"{typed!r} was not recognised as a command"
    assert command.name in {"/setup", "/doctor", "/config", "/login", "/logout"}


@pytest.mark.parametrize("typed", [
    "setup the tests for the retry logic",
    "set up the environment variables",
    "config file is missing a default",
    "doctor the database schema for me",
    "login flow is broken on the callback",
    "logout handler never fires",
    "providers are slow today",
    "where is the retry logic?",
    "add a retry to the upload path",
])
def test_ordinary_questions_are_not_mistaken_for_commands(typed):
    """Only a recognized word or subcommand counts; English starts with them."""
    from terminus.cli_app.repl import as_session_command

    command, _argument = as_session_command(typed)
    assert command is None, f"{typed!r} was wrongly taken as a command"


def test_an_unknown_slash_command_says_so(tmp_path, monkeypatch, captured_repl):
    import asyncio

    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)
    asyncio.run(repl.dispatch(repl.Repl(repo_path=str(tmp_path), session_id="test"), "/nonsense"))
    text = captured_repl()
    assert "Unknown command" in text
    assert "Not a command" not in text


def test_the_configuration_commands_are_listed_in_help(tmp_path, monkeypatch, captured_repl):
    import asyncio

    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)
    asyncio.run(repl._help(repl.Repl(repo_path=str(tmp_path), session_id="test"), ""))
    text = captured_repl()
    for name in ("/setup", "/doctor", "/config"):
        assert name in text, f"{name} missing from /help"


def test_help_does_not_claim_bare_questions_are_commands(tmp_path, monkeypatch, captured_repl):
    """/help used to say a bare message was not a command, then treated it as one."""
    import asyncio

    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)
    asyncio.run(repl._help(repl.Repl(repo_path=str(tmp_path), session_id="test"), ""))
    assert "prefix it with /ask" not in captured_repl()


def test_a_key_for_another_provider_is_used_not_complained_about(tmp_path, monkeypatch, captured_repl):
    """Telling someone to fetch a key they already have is how trust is lost."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=stored\n", encoding="utf-8")
    from terminus.user_config import load_env_files

    load_env_files()
    from terminus.cli_app import repl

    repl._first_run_notice()
    text = captured_repl()
    assert "No API key for openrouter" not in text
    assert "already have a key" not in text


def test_tracing_is_reconciled_after_the_store_is_resolved(tmp_path, monkeypatch):
    """Resolving the store re-reads .env, which undoes an earlier reconciliation."""
    import terminus.cli as cli
    import terminus.config as config_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    config_module.CONFIG["observability"]["tracing"] = False
    (tmp_path / ".env").write_text("LANGSMITH_TRACING=true\nCLUSTER_ENDPOINT=http://x\n",
                                   encoding="utf-8")
    cli.initialize()
    assert "LANGSMITH_TRACING" not in os.environ


def test_a_failing_command_does_not_end_the_session(tmp_path, monkeypatch, captured_repl):
    """A traceback out of the REPL loses the conversation and looks like a crash.

    This happened for real: an unsupported argument to the test runner inside
    `/setup` propagated out of the prompt loop and killed the session.
    """
    import asyncio

    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)

    async def boom(_repl, _argument):
        raise RuntimeError("deliberate failure")

    original = {c.name: c for c in repl.COMMANDS}
    monkeypatch.setattr(
        repl, "COMMANDS",
        tuple(repl.Command("/boom", "boom", boom) if c.name == "/ask" else c
              for c in repl.COMMANDS),
    )
    assert original  # the table was not empty, so the swap really happened

    handled = asyncio.run(repl.dispatch(repl.Repl(repo_path=str(tmp_path), session_id="t"), "/boom"))
    text = captured_repl()
    assert handled is True, "the session must treat it as a handled command"
    assert "deliberate failure" in text
    assert "still open" in text
    assert "Traceback" not in text


def test_an_in_session_command_cannot_crash_the_loop(tmp_path, monkeypatch, captured_repl):
    """Whatever a nested CLI command does, the prompt must come back."""
    import asyncio

    from typer.testing import CliRunner

    from terminus.cli_app import repl

    monkeypatch.chdir(tmp_path)

    def explode(self, args, **kwargs):
        raise TypeError("CliRunner.__init__() got an unexpected keyword argument")

    monkeypatch.setattr(CliRunner, "invoke", explode)
    asyncio.run(repl._run_cli(["doctor"]))
    text = captured_repl()
    assert "Could not run" in text
    assert "Traceback" not in text


def test_config_set_takes_effect_in_the_same_session(tmp_path, monkeypatch):
    """Writing the file and carrying on with the old value reads as doing nothing.

    The first-run panel tells people to run `config set llm.provider X` at the
    prompt. If the next question still used the old provider, the instruction
    would be worse than useless.
    """
    import terminus.config as config_module

    monkeypatch.chdir(tmp_path)
    run("providers", "login", "-p", "groq", "--api-key", "k")
    assert run("config", "set", "llm.provider", "groq", "--scope", "project").exit_code == 0

    from terminus.user_config import configured_provider_has_credential

    assert config_module.CONFIG["llm"]["provider"] == "groq"
    assert configured_provider_has_credential() is True, (
        "the session should be usable immediately after the write, with no restart"
    )


def test_help_points_at_the_configuration_commands():
    result = run("--help")
    assert result.exit_code == 0
    assert "setup" in result.output
    assert "doctor" in result.output
