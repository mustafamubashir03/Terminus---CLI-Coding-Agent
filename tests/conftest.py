"""Session-wide guard rails for the test suite.

Two of these exist because a test that writes outside its tmp_path is a test
that destroys something real, and the failure shows up on a machine rather than
in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Make ``tmp_path`` this test's workspace, and yield it.

    Filesystem tools resolve every path inside the workspace and refuse
    anything outside it, so a test that exercises them against ``tmp_path`` has
    to say that ``tmp_path`` *is* the workspace. ``TERMINUS_WORKSPACE`` is the
    supported way to attach to a directory the process did not start in, so
    setting it here is exercising the real mechanism rather than bypassing it -
    and it is undone by monkeypatch like any other environment change.
    """
    monkeypatch.setenv("TERMINUS_WORKSPACE", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def isolated_user_home(tmp_path_factory, monkeypatch):
    """Point ``~`` at a throwaway directory for every test.

    ``providers login`` defaults to the global scope, which writes to
    ``~/.terminus``. Without this, running the suite would overwrite the
    developer's own credentials and global config, and no test would fail -
    the damage would just appear the next time they used the tool. ``Path.home``
    reads ``HOME`` on POSIX and ``USERPROFILE`` on Windows, so both are set.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    # Expanduser on Windows consults these, and some code paths read them
    # directly rather than going through Path.home().
    monkeypatch.setenv("HOMEDRIVE", Path(home).drive)
    monkeypatch.setenv("HOMEPATH", str(home)[2:])
    yield home


@pytest.fixture(autouse=True)
def no_inherited_provider_keys(monkeypatch):
    """Guarantee no test can accidentally pass because the host has a real key.

    Without this a credential-exporting test reads a perfectly valid provider
    key from the developer's environment and reports success while testing
    nothing, and a test asserting a key is absent passes or fails depending on
    whose laptop it runs on.
    """
    for name in list(os.environ):
        if name.endswith("_API_KEY") or name in {"CLUSTER_ENDPOINT", "LANGSMITH_API_KEY"}:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def close_shared_resources():
    """Release the shared async resources once the suite is done.

    The memory checkpointer and any LLM clients hold aiosqlite connections, whose
    worker threads are non-daemon. One leaked open connection is enough to stop
    the interpreter from exiting, which turns a finished run into a hang: pytest
    prints its summary and then sits there forever. The CLI closes these on
    shutdown; a test session has to do the same or it never terminates.
    """
    yield

    import asyncio

    from terminus.llm.factory import aclose_llm_clients
    from terminus.memory.short_term import close_checkpointer

    async def release():
        await close_checkpointer()
        await aclose_llm_clients()

    try:
        asyncio.run(release())
    except Exception:
        pass


@pytest.fixture(autouse=True)
def restore_global_config_state():
    """Undo config and credential state that a test changed in-process.

    ``CONFIG`` is a module-level dict resolved once at import, and a dozen
    modules do ``from terminus.config import CONFIG`` at module scope, so each
    one holds a reference to that exact dict. ``setup`` deliberately
    re-resolves it in the process that ran it, which is right for the real CLI
    (one process, one command) and wrong for a test suite, where the next test
    inherits a provider it never asked for.

    Restored by mutating the dict in place rather than rebinding the module
    attribute: rebinding leaves every module that already imported the old
    object pointing at the mutated one, and the suite then disagrees with itself
    about which provider is active.
    """
    import copy

    import terminus.config as config_module
    import terminus.user_config as user_config

    saved_config = copy.deepcopy(config_module.CONFIG)
    saved_source = config_module.CONFIG_SOURCE
    saved_kind = config_module.CONFIG_SOURCE_KIND
    saved_layers = config_module.CONFIG_SOURCE_LAYERS
    saved_loaded = user_config._FILES_LOADED
    try:
        yield
    finally:
        config_module.CONFIG.clear()
        config_module.CONFIG.update(saved_config)
        config_module.CONFIG_SOURCE = saved_source
        config_module.CONFIG_SOURCE_KIND = saved_kind
        config_module.CONFIG_SOURCE_LAYERS = saved_layers
        user_config._FILES_LOADED = saved_loaded


os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")


@pytest.fixture(autouse=True)
def isolated_module_state():
    """Undo per-test state that outlives the test that created it.

    Two separate leaks show up as failures in tests that have nothing to do with
    the file that caused them, so both are closed here rather than at each site:

    * a plain assignment to a module attribute - ``factory.get_llm = ...`` reads
      better than a fixture, but the replacement outlives the test;
    * a permission policy or execution scope left installed in its ContextVar,
      which the fail-closed tests then read as ambient authority.

    Restoring the module namespace and the ContextVars after every test is what
    makes the suite order-independent.
    """
    import terminus.agent.factory as factory
    import terminus.execution as execution
    import terminus.permissions as permissions

    watched = {
        factory: {k: v for k, v in vars(factory).items() if not k.startswith("__")},
        execution: {k: v for k, v in vars(execution).items() if not k.startswith("__")},
        permissions: {k: v for k, v in vars(permissions).items() if not k.startswith("__")},
    }
    policy_before = permissions.get_permission_policy()
    execution_before = execution.current_execution()
    try:
        yield
    finally:
        for module, before in watched.items():
            for name, value in before.items():
                if vars(module).get(name) is not value:
                    setattr(module, name, value)
        if permissions.get_permission_policy() is not policy_before:
            permissions._policy.set(policy_before)
        if execution.current_execution() is not execution_before:
            execution._current.set(execution_before)
