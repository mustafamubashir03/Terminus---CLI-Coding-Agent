"""State ownership: project / workspace / session / thread / checkpoint / task.

The invariant under test:

    Two projects never share conversational, task or retrieval state, and a
    project is only ever executed in the workspace it was created for.

These tests use real TempProjects, real SQLite databases and a real (tiny)
LangGraph over the production checkpointer. Only the LLM is absent.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from terminus.workspace import project_key, project_root


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class MsgState(TypedDict):
    messages: Annotated[list, add_messages]


def _echo(state: MsgState):
    return state


def _tiny_graph():
    g = StateGraph(MsgState)
    g.add_node("echo", _echo)
    g.add_edge(START, "echo")
    return g.compile()


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run the test body as if Terminus had been started in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _make_plan():
    from terminus.tasks.planner import ExecutionPlan, PlannedTask, TaskType
    return ExecutionPlan(
        project_name="P", goal_summary="g", tech_stack=["python"],
        total_estimated_hours=1.0,
        tasks=[PlannedTask(
            id="task__001", title="t", description="d",
            task_type=TaskType.IMPLEMENT, depends_on=[],
            estimated_minutes=5, output_files=[], acceptance_criteria=["done"],
        )],
        risks=[], assumptions=[],
    )


def _store():
    from terminus.config import CONFIG
    from terminus.tasks.task_store import TaskStore
    path = Path(CONFIG["tasks"]["db_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return TaskStore(str(path))


# ---------------------------------------------------------------------------
# project identity
# ---------------------------------------------------------------------------

def test_project_root_is_the_cwd(in_tmp_cwd):
    assert project_root() == in_tmp_cwd.resolve()
    assert project_key() == str(in_tmp_cwd.resolve())


def test_two_projects_have_distinct_identities(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    monkeypatch.chdir(a)
    key_a = project_key()
    monkeypatch.chdir(b)
    assert project_key() != key_a


def test_state_paths_are_per_project(tmp_path, monkeypatch):
    """Every configured .terminus path is relative, so isolation is by cwd."""
    from terminus.config import CONFIG
    from terminus.memory import session as S

    for key in ("memory", "tasks"):
        assert not os.path.isabs(CONFIG[key]["db_path"])
    assert not os.path.isabs(CONFIG["chromadb"]["persist_dir"])

    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    monkeypatch.chdir(a)
    session_a = str(S._session_file().resolve())
    monkeypatch.chdir(b)
    assert str(S._session_file().resolve()) != session_a


# ---------------------------------------------------------------------------
# session identity
# ---------------------------------------------------------------------------

def test_switch_session_creates_the_state_directory(tmp_path, monkeypatch):
    """Regression: switch_session used to raise FileNotFoundError."""
    from terminus.memory import session as S

    monkeypatch.setitem(S.CONFIG["memory"], "db_path",
                        str(tmp_path / "missing_dir" / "terminus.db"))
    session_file = S._session_file()
    assert not session_file.parent.exists()

    S.switch_session("manual-id")          # must not raise
    assert session_file.read_text().strip() == "manual-id"


def test_new_session_and_switch_session_behave_the_same(tmp_path, monkeypatch):
    from terminus.memory import session as S
    monkeypatch.setitem(S.CONFIG["memory"], "db_path",
                        str(tmp_path / "d" / "terminus.db"))
    created = S.new_session()
    assert S.get_current_session() == created
    S.switch_session("other")
    assert S.get_current_session() == "other"


def test_sessions_are_isolated_from_each_other(in_tmp_cwd, tmp_path):
    """Two thread ids must not see each other's messages."""
    async def run():
        from terminus.memory.short_term import close_checkpointer, get_checkpointer
        cp = await get_checkpointer()
        app = _tiny_graph()
        app.checkpointer = cp
        await app.ainvoke({"messages": [HumanMessage("A1-FACT")]},
                          config={"configurable": {"thread_id": "A1"}})
        await app.ainvoke({"messages": [HumanMessage("A2-FACT")]},
                          config={"configurable": {"thread_id": "A2"}})
        one = await cp.aget({"configurable": {"thread_id": "A1"}})
        two = await cp.aget({"configurable": {"thread_id": "A2"}})
        await close_checkpointer()
        return one, two

    one, two = asyncio.run(run())
    text_one = str(one["channel_values"]["messages"])
    text_two = str(two["channel_values"]["messages"])
    assert "A1-FACT" in text_one and "A2-FACT" not in text_one
    assert "A2-FACT" in text_two and "A1-FACT" not in text_two


def test_a_new_session_starts_empty(in_tmp_cwd):
    async def run():
        from terminus.memory.short_term import close_checkpointer, get_checkpointer
        cp = await get_checkpointer()
        app = _tiny_graph()
        app.checkpointer = cp
        fresh = await cp.aget({"configurable": {"thread_id": "brand-new"}})
        await close_checkpointer()
        return fresh

    assert asyncio.run(run()) is None


# ---------------------------------------------------------------------------
# process restart
# ---------------------------------------------------------------------------

_SUBPROCESS = """
import asyncio, os, sys
sys.path.insert(0, {src!r})
os.chdir({ws!r})
from langchain_core.messages import HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

class S(TypedDict):
    messages: Annotated[list, add_messages]

def echo(state):
    return state

async def main():
    from terminus.memory.short_term import close_checkpointer, get_checkpointer
    cp = await get_checkpointer()
    g = StateGraph(S); g.add_node("echo", echo); g.add_edge(START, "echo")
    app = g.compile(); app.checkpointer = cp
    cfg = {{"configurable": {{"thread_id": {thread!r}}}}}
    if {write}:
        await app.ainvoke({{"messages": [HumanMessage("PERSISTED-FACT")]}}, config=cfg)
        print("WROTE")
    else:
        cp_now = await cp.aget(cfg)
        msgs = (cp_now or {{}}).get("channel_values", {{}}).get("messages", []) or []
        print("READ", len(msgs), "PERSISTED-FACT" in str(msgs))
    await close_checkpointer()

asyncio.run(main())
"""


def _run_child(ws: Path, thread: str, write: bool) -> str:
    import terminus
    src = str(Path(terminus.__file__).resolve().parents[1])
    code = _SUBPROCESS.format(src=src, ws=str(ws), thread=thread, write=write)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300, cwd=str(ws))
    assert proc.returncode == 0, f"child failed: {proc.stderr[-2000:]}"
    return proc.stdout


def test_conversation_survives_a_process_restart(tmp_path):
    """A separate OS process must see the checkpoint written by the first."""
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / ".terminus" / "memory").mkdir(parents=True)

    assert "WROTE" in _run_child(ws, "sess-1", write=True)
    out = _run_child(ws, "sess-1", write=False)
    assert "READ" in out
    assert "True" in out, "the checkpoint did not survive the restart"


def test_a_different_session_does_not_see_it_after_restart(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / ".terminus" / "memory").mkdir(parents=True)
    _run_child(ws, "sess-1", write=True)
    out = _run_child(ws, "sess-2", write=False)
    assert "READ 0 False" in out


# ---------------------------------------------------------------------------
# cross-project isolation
# ---------------------------------------------------------------------------

def test_same_thread_id_in_two_projects_is_isolated(tmp_path, monkeypatch):
    """Project B reusing Project A's thread id must not see A's history."""
    from terminus.memory.short_term import close_checkpointer, get_checkpointer

    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    (a / ".terminus" / "memory").mkdir(parents=True)
    (b / ".terminus" / "memory").mkdir(parents=True)

    async def run():
        results = {}
        for root, fact in ((a, "ALPHA-IN-A"), (b, "GAMMA-IN-B")):
            monkeypatch.chdir(root)
            await close_checkpointer()          # force a fresh connection
            cp = await get_checkpointer()
            app = _tiny_graph()
            app.checkpointer = cp
            await app.ainvoke({"messages": [HumanMessage(fact)]},
                              config={"configurable": {"thread_id": "shared-id"}})
            state = await cp.aget({"configurable": {"thread_id": "shared-id"}})
            results[root] = str(state["channel_values"]["messages"])
            await close_checkpointer()
        return results

    results = asyncio.run(run())
    assert "ALPHA-IN-A" in results[a] and "GAMMA-IN-B" not in results[a]
    assert "GAMMA-IN-B" in results[b] and "ALPHA-IN-A" not in results[b]


def test_task_state_is_per_project(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()

    monkeypatch.chdir(a)
    store_a = _store()
    pid_a = store_a.create_project("goal-a", _make_plan())

    monkeypatch.chdir(b)
    store_b = _store()
    assert store_b.get_latest_project() is None, "B must not see A's projects"

    monkeypatch.chdir(a)
    assert _store().get_latest_project() == pid_a


# ---------------------------------------------------------------------------
# workspace identity for projects (the wrong-workspace bug)
# ---------------------------------------------------------------------------

def test_project_records_its_workspace(in_tmp_cwd):
    store = _store()
    pid = store.create_project("goal", _make_plan())
    assert store.get_project_workspace(pid) == str(project_root())


def test_workspace_matches_is_true_in_the_right_directory(in_tmp_cwd):
    store = _store()
    pid = store.create_project("goal", _make_plan())
    assert store.workspace_matches(pid) is True


def _copy_state(src: Path, dst: Path) -> None:
    """Replicate the real failure: a project directory copied/moved elsewhere.

    The copied .terminus carries the plan and its task state, so the project is
    still offered as resumable - which is exactly when it must be refused.
    """
    import shutil
    shutil.copytree(src / ".terminus", dst / ".terminus")


def test_workspace_matches_is_false_in_a_copied_workspace(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    monkeypatch.chdir(a)
    pid = _store().create_project("goal", _make_plan())
    _copy_state(a, b)

    monkeypatch.chdir(b)
    store_b = _store()
    assert store_b.get_latest_project() == pid, "copied state is still offered"
    assert store_b.get_project_workspace(pid) == str(a.resolve())
    assert store_b.workspace_matches(pid) is False


def test_workspace_matches_is_true_in_the_original_workspace(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    monkeypatch.chdir(a)
    pid = _store().create_project("goal", _make_plan())
    _copy_state(a, b)
    monkeypatch.chdir(a)
    assert _store().workspace_matches(pid) is True


def test_legacy_projects_without_a_workspace_are_still_allowed(tmp_path, monkeypatch):
    """A NULL workspace means 'unknown', not 'wrong' - do not break old data."""
    monkeypatch.chdir(tmp_path)
    store = _store()
    pid = store.create_project("goal", _make_plan())
    with store.conn() as conn:
        conn.execute("UPDATE projects SET workspace=NULL WHERE id=?", (pid,))
    assert store.get_project_workspace(pid) is None
    assert store.workspace_matches(pid) is True


def test_workspace_column_is_migrated_into_an_old_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / ".terminus" / "tasks" / "tasks.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    # a database from before workspace identity existed
    con = sqlite3.connect(str(db))
    con.execute("""CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT,
                   goal TEXT, plan_json TEXT, status TEXT,
                   created_at REAL DEFAULT (unixepoch()))""")
    con.execute("""CREATE TABLE tasks (id TEXT NOT NULL, project_id TEXT NOT NULL,
                   title TEXT, description TEXT, task_type TEXT,
                   status TEXT DEFAULT 'pending', depends_on TEXT DEFAULT '[]',
                   output_files TEXT DEFAULT '[]', acceptance_criteria TEXT DEFAULT '[]',
                   result TEXT, error TEXT, retry_count INTEGER DEFAULT 0,
                   max_retries INTEGER DEFAULT 3, execution_order INTEGER DEFAULT 0,
                   started_at REAL, created_at REAL DEFAULT (unixepoch()),
                   completed_at REAL, PRIMARY KEY (project_id, id))""")
    con.execute("INSERT INTO projects(id,name,goal,status) VALUES ('old','n','g','approved')")
    con.commit()
    con.close()

    from terminus.tasks.task_store import TaskStore
    store = TaskStore(str(db))
    columns = {r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(projects)")}
    assert "workspace" in columns
    assert store.get_latest_project() == "old"


def test_plan_continue_refuses_the_wrong_workspace(tmp_path, monkeypatch, capsys):
    """The concrete failure: executing A's plan against B's files."""
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()

    monkeypatch.chdir(a)
    pid = _store().create_project("goal", _make_plan())
    _copy_state(a, b)

    monkeypatch.chdir(b)
    asyncio.run(__import__(
        "terminus.tasks.orchestrator", fromlist=["x"]).handle_plan_command("continue"))
    out = capsys.readouterr().out
    # rich hard-wraps console output, possibly mid-path, so assert on the
    # message and the behaviour rather than on a rendered path.
    assert "different workspace" in " ".join(out.split())
    assert "Run Terminus from" in " ".join(out.split())

    # nothing may have executed
    tasks = _store().get_all_tasks(pid)
    assert all(t["status"] == "pending" for t in tasks), "a task ran in the wrong tree"


def test_plan_continue_still_works_in_the_right_workspace(tmp_path, monkeypatch):
    a = tmp_path / "A"
    a.mkdir()
    monkeypatch.chdir(a)
    pid = _store().create_project("goal", _make_plan())
    # same directory expressed differently must still match
    assert _store().workspace_matches(pid, str(a) + os.sep) is True
    assert _store().workspace_matches(pid, str(a).upper() if os.name == "nt"
                                      else str(a)) is True


# ---------------------------------------------------------------------------
# cache scoping (the staleness bugs)
# ---------------------------------------------------------------------------

def test_system_prompt_cache_is_scoped_per_workspace(tmp_path, monkeypatch):
    from terminus.agent.factory import _build_system_prompt

    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    (a / "A_MARKER.txt").write_text("a", encoding="utf-8")
    (b / "B_MARKER.txt").write_text("b", encoding="utf-8")

    monkeypatch.chdir(a)
    assert "A_MARKER" in _build_system_prompt()
    monkeypatch.chdir(b)
    prompt_b = _build_system_prompt()
    assert "B_MARKER" in prompt_b
    assert "A_MARKER" not in prompt_b, "Project A's listing leaked into B"
    monkeypatch.chdir(a)
    assert "A_MARKER" in _build_system_prompt()


def test_terminus_md_is_read_per_workspace(tmp_path, monkeypatch):
    from terminus.context.environment import build_startup_context

    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    (a / "TERMINUS.md").write_text("RULES_FOR_A", encoding="utf-8")
    (b / "TERMINUS.md").write_text("RULES_FOR_B", encoding="utf-8")

    assert "RULES_FOR_A" in build_startup_context(a)
    assert "RULES_FOR_B" in build_startup_context(b)
    assert "RULES_FOR_A" not in build_startup_context(b)


def test_semantic_collection_cache_is_scoped_per_workspace(tmp_path, monkeypatch):
    """Regression: an unkeyed cached collection leaked across projects.

    The cache now lives in ``retrievers.cache`` and is keyed on the project root,
    so two projects in one process get two collections. The assertion is
    unchanged: switching project must open a *new* store, not reuse the last one.
    """
    import chromadb

    from terminus.config import CONFIG
    from terminus.context.indexers.semantic_chroma import chroma_persist_path
    from terminus.context.retrievers import cache as retrieval_cache
    from terminus.context.retrievers.cache import cached_store

    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    opened = []
    retrieval_cache.reset()

    class FakeCollection:
        name = "terminus"

    class FakeClient:
        def __init__(self, path):
            opened.append(path)

        def get_or_create_collection(self, name):
            return FakeCollection()

    monkeypatch.setattr(chromadb, "PersistentClient", FakeClient)
    monkeypatch.setitem(CONFIG["chromadb"], "persist_dir", ".terminus/chromadb/")

    def get():
        return cached_store(
            f"chroma:{chroma_persist_path()}",
            lambda: chromadb.PersistentClient(
                path=str(chroma_persist_path())
            ).get_or_create_collection(name=CONFIG["chromadb"]["collection_name"]),
        )

    monkeypatch.chdir(a)
    get()
    assert opened[-1].startswith(str(a.resolve()))

    monkeypatch.chdir(b)
    get()
    assert len(opened) == 2, "the cache was reused across projects"
    assert opened[-1].startswith(str(b.resolve()))

    # and it is still cached within one project
    monkeypatch.chdir(a)
    get()
    assert len(opened) == 2, "the cache was dropped within a single project"
    retrieval_cache.reset()


# ---------------------------------------------------------------------------
# worker identity
# ---------------------------------------------------------------------------

def test_worker_prompt_states_its_project_workspace_and_task(in_tmp_cwd):
    from terminus.tasks.executor import _build_system_prompt

    store = _store()
    pid = store.create_project("goal", _make_plan())
    task = store.get_all_tasks(pid)[0]
    prompt = _build_system_prompt(task, [])
    assert pid in prompt
    assert task["id"] in prompt
    assert str(project_root()) in prompt


def test_worker_prompt_carries_no_ask_conversation(tmp_path, monkeypatch):
    """Isolation is deliberate: the worker gets identity, not /ask history."""
    from terminus.tasks.executor import _build_system_prompt
    monkeypatch.chdir(tmp_path)
    store = _store()
    pid = store.create_project("goal", _make_plan())
    task = store.get_all_tasks(pid)[0]
    prompt = _build_system_prompt(task, [{"id": "dep", "result": "DEP-RESULT"}])
    assert "DEP-RESULT" in prompt, "dependency results must cross the boundary"
    assert "conversation" not in prompt.lower()


# ---------------------------------------------------------------------------
# permission policy ownership
# ---------------------------------------------------------------------------

def test_the_policy_belongs_to_the_execution_not_the_process(in_tmp_cwd):
    from terminus.execution import ask_context, execution_scope
    from terminus.permissions import PermissionLevel, PermissionPolicy, get_permission_policy

    strict = PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,), approver=None,
        deny_levels=(PermissionLevel.WRITE, PermissionLevel.DESTRUCTIVE),
    )
    before = get_permission_policy()
    with execution_scope(ask_context(strict)):
        assert get_permission_policy() is strict
    assert get_permission_policy() is before


def test_a_child_inherits_rather_than_widening(in_tmp_cwd):
    """A future child must not obtain more permission by building an agent.

    Inheritance is now structural: a child runs inside the parent's execution
    scope, and building an agent changes nothing.
    """
    import terminus.agent.factory as factory
    from terminus.execution import ask_context, execution_scope
    from terminus.permissions import (
        PermissionLevel,
        PermissionPolicy,
        get_permission_policy,
    )

    strict = PermissionPolicy(
        auto_approve=(PermissionLevel.READ_ONLY,), approver=None,
        deny_levels=(PermissionLevel.WRITE, PermissionLevel.DESTRUCTIVE),
    )
    seen = {}

    def fake_create_agent(llm, tools, system_prompt, checkpointer, middleware):
        # what the agent would see if it ran right now
        seen["policy"] = get_permission_policy()
        return "AGENT"

    real = (factory.create_agent, factory.get_llm,
            factory.get_summarization_middleware, factory.get_checkpointer)
    factory.create_agent = fake_create_agent
    factory.get_llm = lambda: "LLM"
    factory.get_summarization_middleware = lambda: "MW"
    factory.get_checkpointer = lambda: _async_value("CP")
    try:
        with execution_scope(ask_context(strict)):
            asyncio.run(factory.build_agent())
    finally:
        (factory.create_agent, factory.get_llm,
         factory.get_summarization_middleware, factory.get_checkpointer) = real

    assert seen["policy"] is strict, "the child must see the parent's policy"
    # and it is still the strict one, not a fresh default
    assert PermissionLevel.WRITE in seen["policy"].deny_levels


# ---------------------------------------------------------------------------
# the skills catalogue is per-project state too
# ---------------------------------------------------------------------------

def test_skills_catalogue_is_scoped_per_workspace(tmp_path, monkeypatch):
    """Regression: the registry and its prompt were cached under a bare key."""
    from terminus.skills import skill_tools

    a, b = tmp_path / "A", tmp_path / "B"
    for root, name in ((a, "alpha_skill"), (b, "bravo_skill")):
        d = root / ".terminus" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: for {root.name}\n"
            f"when_to_use: in {root.name}\n---\n\nBody.\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(skill_tools, "_registry", None)
    monkeypatch.setattr(skill_tools, "_registry_project", None)

    monkeypatch.chdir(a)
    in_a = skill_tools.build_skills_prompt()
    assert "alpha_skill" in in_a
    assert "bravo_skill" not in in_a

    monkeypatch.chdir(b)
    in_b = skill_tools.build_skills_prompt()
    assert "bravo_skill" in in_b
    assert "alpha_skill" not in in_b, "Project A's skills leaked into B"

    monkeypatch.chdir(a)
    assert "alpha_skill" in skill_tools.build_skills_prompt()


async def _async_value(v):
    return v
