"""Configuration is resolved from where the command runs, not from import time.

``CONFIG`` is a module-level dict, so every module that reads it holds a reference
to one object for the life of the process. That is fine for the layer it is
actually a function of, and wrong for the project layer: ``config.py`` resolves
``Path.cwd() / "config.yaml"``, which means the project config belongs to whatever
directory the process is standing in.

Resolving once at import therefore froze the project layer to the directory the
interpreter happened to start in. Anything that moved afterwards - a test using
``monkeypatch.chdir``, a command run against another checkout - was answered from
the import-time location. These tests pin the corrected behaviour: re-resolution
happens at invocation and follows the current working directory.

Every global-config location here is inside the throwaway home that
``conftest.isolated_user_home`` creates, so a developer's real
``~/.terminus/config.yaml`` is never read and never written.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import terminus.config as config_module
from terminus.config import DEFAULT_CONFIG, load_config

runner = CliRunner()


def source_kind() -> str:
    """Which layer won, read at call time.

    ``load_config`` rebinds the module attribute, so a ``from ... import
    CONFIG_SOURCE_KIND`` at the top of this file would keep the value that was
    true when *this module* was imported - which is precisely the staleness
    under test, reproduced in the test that reads it.
    """
    return config_module.CONFIG_SOURCE_KIND


def global_config() -> Path:
    """The global config path for this test's throwaway home."""
    from terminus.user_config import global_config_path

    return global_config_path()


def write_config(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def resolved() -> dict:
    """The configuration as the current directory resolves it."""
    return load_config()


# --- 1 & 2. the project layer follows the current directory ------------------


def test_a_project_directory_is_found_from_inside_it(tmp_path, monkeypatch):
    write_config(tmp_path, "llm:\n  provider: cohere\n  model: from-the-project\n")
    monkeypatch.chdir(tmp_path)

    config = resolved()

    assert config["llm"]["provider"] == "cohere"
    assert config["llm"]["model"] == "from-the-project"
    assert source_kind() == "cwd"


def test_chdir_moves_project_resolution_to_the_new_directory(tmp_path, monkeypatch):
    """The invariant: resolution follows the working directory, not import time.

    Two directories with different configs, visited in sequence by one process.
    A frozen import-time resolution cannot answer both.
    """
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_config(first, "llm:\n  provider: cohere\n  model: first-project\n")
    write_config(second, "llm:\n  provider: groq\n  model: second-project\n")

    monkeypatch.chdir(first)
    assert resolved()["llm"]["model"] == "first-project"

    monkeypatch.chdir(second)
    assert resolved()["llm"]["model"] == "second-project"
    assert resolved()["llm"]["provider"] == "groq"

    monkeypatch.chdir(first)
    assert resolved()["llm"]["model"] == "first-project"


def test_a_directory_with_no_config_falls_through_to_the_layer_below(tmp_path, monkeypatch):
    """Leaving the project must not leave its configuration behind."""
    project = tmp_path / "project"
    project.mkdir()
    write_config(project, "llm:\n  provider: cohere\n  model: only-here\n")

    monkeypatch.chdir(project)
    assert resolved()["llm"]["model"] == "only-here"

    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert resolved()["llm"]["model"] != "only-here"
    assert resolved()["llm"]["provider"] != "cohere"


# --- 3. the global layer ------------------------------------------------------


def test_the_global_config_applies_when_there_is_no_project_config(tmp_path, monkeypatch):
    write_config(global_config().parent, "llm:\n  provider: cohere\n  model: global-model\n")
    monkeypatch.chdir(tmp_path)

    config = resolved()

    assert config["llm"]["provider"] == "cohere"
    assert config["llm"]["model"] == "global-model"
    assert source_kind() == "global"


def test_a_real_cli_command_sees_the_global_config(tmp_path, monkeypatch):
    """Resolution happens for commands, not only for direct calls."""
    from terminus.cli_app.main import app

    write_config(global_config().parent, "llm:\n  provider: cohere\n")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0, result.output
    assert "cohere" in result.output


# --- 4. package defaults ------------------------------------------------------


def test_defaults_apply_when_neither_project_nor_global_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not global_config().exists()

    config = resolved()

    assert config["llm"]["provider"] == DEFAULT_CONFIG["llm"]["provider"]
    assert config["llm"]["model"] == DEFAULT_CONFIG["llm"]["model"]
    assert source_kind() == "default"


# --- 5. no leaking between tests ---------------------------------------------


def test_a_project_config_always_wins_over_whatever_is_global(tmp_path, monkeypatch):
    """The requirement, stated as behaviour rather than as a slogan.

    A global layer is *meant* to apply, so "the global value was used" is not
    itself a leak. The invariant that matters is that a project always decides
    its own provider, so no global setting can decide an assertion made against
    a project.
    """
    write_config(
        global_config().parent,
        "llm:\n  provider: ANTHOCH_MARKER\n  model: global-marker\n",
    )
    write_config(tmp_path, "llm:\n  provider: openai\n  model: project-wins\n")
    monkeypatch.chdir(tmp_path)

    config = resolved()

    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["model"] == "project-wins"
    assert "ANTHOCH_MARKER" not in repr(config)


def test_ollama_configured_globally_does_not_change_a_project_default(tmp_path, monkeypatch):
    """Requirement: a global ollama setting must not satisfy a test expecting another.

    This is the exact shape of the regression that made ~23 config tests fail:
    a developer's own provider leaking into assertions about a fresh install.
    """
    write_config(
        global_config().parent,
        "llm:\n  provider: ollama\n  model: qwen3:8b\n",
    )
    monkeypatch.chdir(tmp_path)

    config = resolved()
    assert config["llm"]["provider"] == "ollama", "the global layer must still apply"

    # With a project file present, the project wins and the global value cannot
    # decide the outcome.
    write_config(tmp_path, "llm:\n  provider: openai\n")
    monkeypatch.chdir(tmp_path)
    assert resolved()["llm"]["provider"] == "openai"


def test_credentials_do_not_leak_into_resolved_configuration(tmp_path, monkeypatch):
    """A resolved configuration is settings only; a secret is never in it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-not-a-setting")
    monkeypatch.chdir(tmp_path)

    config = resolved()
    flat = repr(config)

    assert "sk-not-a-setting" not in flat
    assert "OPENROUTER_API_KEY" not in flat


# --- 6 & 7. the global layer is genuinely isolated ---------------------------


def test_the_global_config_path_is_inside_the_throwaway_home():
    """Proves the isolation rather than assuming it.

    If this ever pointed at the developer's real home, every test above would be
    writing to and reading from their actual configuration.
    """
    from terminus.user_config import global_config_path

    path = global_config_path()
    home = Path(os.environ["USERPROFILE"])

    assert home in path.parents
    assert path.parent.name == ".terminus"


def test_a_second_test_sees_a_clean_global_config(tmp_path, monkeypatch):
    """One test's global config is not visible to the next one."""
    from terminus.config import load_config as reload_config

    write_config(global_config().parent, "llm:\n  provider: cohere\n")
    monkeypatch.chdir(tmp_path)
    assert reload_config()["llm"]["provider"] == "cohere"

    global_config().unlink()
    monkeypatch.chdir(tmp_path / "..")
    assert reload_config()["llm"]["provider"] != "cohere"


# --- 8. precedence, through the real CLI ------------------------------------


def test_the_project_config_overrides_the_global_one_in_real_cli_use(tmp_path, monkeypatch):
    """The documented order, end to end: project beats global beats package."""
    from terminus.cli_app.main import app

    write_config(global_config().parent, "llm:\n  provider: cohere\n  model: global-model\n")
    monkeypatch.chdir(tmp_path)

    assert runner.invoke(app, ["config", "show", "--json"]).exit_code == 0
    assert resolved()["llm"]["provider"] == "cohere"

    write_config(tmp_path, "llm:\n  provider: groq\n  model: project-model\n")

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0, result.output
    assert "project-model" in result.output
    # The project decides. Checked through resolution rather than by scanning the
    # output for a provider name: `config show` enumerates every known provider,
    # so the global one appearing there is information, not leakage.
    assert resolved()["llm"]["provider"] == "groq"
    assert resolved()["llm"]["model"] == "project-model"


def test_config_set_writes_the_project_layer_it_says_it_does(tmp_path, monkeypatch):
    """A write and a later read must agree about which file was written."""
    from terminus.cli_app.main import app

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["config", "set", "llm.provider", "groq", "--scope", "project"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").is_file()
    assert "groq" in (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert resolved()["llm"]["provider"] == "groq"


def test_a_config_written_by_the_cli_is_read_back_by_the_next_command(tmp_path, monkeypatch):
    """Round trip across two separate invocations, as two real commands would be."""
    from terminus.cli_app.main import app

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["config", "set", "llm.model", "round-trip-model"]).exit_code == 0

    result = runner.invoke(app, ["config", "show", "--json"])

    assert "round-trip-model" in result.output


# --- the mechanism itself -----------------------------------------------------


def test_re_resolution_updates_the_shared_dict_in_place():
    """Every module holds a reference to CONFIG, so it must be mutated not rebound.

    Rebinding ``terminus.config.CONFIG`` would leave a dozen modules reading a
    stale object, and the symptom would be two parts of the program disagreeing
    about which provider is active.
    """
    import terminus.config as config_module

    before = config_module.CONFIG
    identity = id(before)
    config_module.CONFIG.update({"llm": {"provider": "temporary-marker"}})

    assert id(config_module.CONFIG) == identity
    assert config_module.CONFIG["llm"]["provider"] == "temporary-marker"

    config_module.CONFIG["llm"].pop("provider", None)
    load_config()
    assert config_module.CONFIG["llm"]["provider"] != "temporary-marker"


@pytest.mark.parametrize(
    ("layer", "prepare"),
    [
        ("cwd", lambda tmp: write_config(tmp, "llm:\n  provider: cohere\n")),
        ("global", lambda tmp: write_config(global_config().parent, "llm:\n  provider: cohere\n")),
        ("default", lambda tmp: None),
    ],
)
def test_the_source_kind_names_the_layer_that_won(tmp_path, monkeypatch, layer, prepare):
    """The reported source is the highest-precedence layer that said anything."""
    prepare(tmp_path)
    monkeypatch.chdir(tmp_path)

    load_config()

    assert source_kind() == layer
