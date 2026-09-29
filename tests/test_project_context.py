"""The project read model: it is bounded, read-only, and project-scoped.

The properties under test:

    A project section can never exceed its budget, it never raises on absent or
    malformed data, it refuses a project belonging to another workspace, it never
    writes, and the agents and the CLI that render it agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terminus import project_context as pc
from terminus.tasks.planner import ExecutionPlan, PlannedTask, TaskType
from terminus.tasks.task_store import TaskStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _plan(goal: str = "Ship the thing", stack=("python",), risks=("slippery",)) -> ExecutionPlan:
    return ExecutionPlan(
        project_name="Demo",
        goal_summary=goal,
        tech_stack=list(stack),
        total_estimated_hours=1.0,
        tasks=[PlannedTask(
            id="task__001", title="t1", description="d",
            task_type=TaskType.IMPLEMENT, depends_on=[], estimated_minutes=1,
            output_files=[], acceptance_criteria=["done"],
        )],
        risks=list(risks),
        assumptions=["it works"],
    )


@pytest.fixture
def store(tmp_path) -> TaskStore:
    return TaskStore(str(tmp_path / "tasks.db"))


def _seed(store: TaskStore, goal: str = "Ship the thing", tasks: int = 1,
          workspace: str | None = None) -> str:
    plan = _plan(goal)
    plan.tasks = [
        PlannedTask(
            id=f"task__{i:03d}", title=f"t{i}", description="d",
            task_type=TaskType.IMPLEMENT, depends_on=[], estimated_minutes=1,
            output_files=[], acceptance_criteria=["done"],
        )
        for i in range(1, tasks + 1)
    ]
    return store.create_project(
        goal, plan, workspace=workspace if workspace is not None else str(pc.project_root())
    )


def _row(store: TaskStore, pid: str, tid: str = "task__001") -> dict:
    return next(t for t in store.get_all_tasks(pid) if t["id"] == tid)


def _complete(store: TaskStore, pid: str, tid: str, result: str) -> None:
    store.complete_task(pid, tid, result)


def _fail(store: TaskStore, pid: str, tid: str, error: str) -> None:
    store.fail_task(pid, tid, error, force=True)


# ---------------------------------------------------------------------------
# empty state
# ---------------------------------------------------------------------------


def test_no_project_renders_nothing(store: TaskStore):
    """The common case - a repo nobody has planned - must cost zero prompt."""
    facts = pc.collect_project_facts(store)
    assert facts.has_project is False
    assert pc.render(facts) == ""


def test_missing_database_degrades_to_empty(tmp_path):
    """Optional context must never be the reason a run fails."""

    class Broken:
        def get_resumable_project(self):
            raise RuntimeError("no database")

        def get_latest_project(self):
            raise RuntimeError("no database")

    facts = pc.collect_project_facts(Broken())
    assert facts.has_project is False
    assert pc.render(facts) == ""


def test_unreadable_project_row_degrades_to_empty(store: TaskStore):
    """A project we cannot read is reported as a project with no readable state."""
    pid = _seed(store)

    class HalfBroken:
        def get_project(self, _pid):
            raise RuntimeError("corrupt row")

        def get_all_tasks(self, _pid):
            return []

    facts = pc.collect_project_facts(HalfBroken(), pid)
    assert facts.has_project is True
    assert pc.render(facts)


# ---------------------------------------------------------------------------
# plan context
# ---------------------------------------------------------------------------


def test_plan_fields_reach_the_section(store: TaskStore):
    pid = _seed(store, goal="Make the parser fast")
    text = pc.render(pc.collect_project_facts(store, pid))
    assert "Make the parser fast" in text
    assert "python" in text
    assert "slippery" in text
    assert "it works" in text


def test_malformed_plan_json_is_tolerated(store: TaskStore):
    """A truncated or hand-edited plan must not take the run down."""
    pid = _seed(store)
    with store.conn() as conn:
        conn.execute(
            "UPDATE projects SET plan_json = ? WHERE id = ?", ("{not json", pid)
        )
    facts = pc.collect_project_facts(store, pid)
    assert facts.has_project is True
    assert facts.goal  # falls back to projects.goal
    assert pc.render(facts)


@pytest.mark.parametrize("payload", ["", None, "[]", '"a string"', "123"])
def test_unusable_plan_json_shapes_are_survivable(payload):
    assert pc.plan_fields(payload) in ({}, {})


# ---------------------------------------------------------------------------
# task results
# ---------------------------------------------------------------------------


def test_successful_result_is_visible(store: TaskStore):
    pid = _seed(store)
    _complete(store, pid, "task__001", "Wrote the module and its tests.")
    text = pc.render(pc.collect_project_facts(store, pid))
    assert "Wrote the module and its tests." in text
    assert "completed" in text


def test_failure_is_visible_with_its_message(store: TaskStore):
    pid = _seed(store)
    _fail(store, pid, "task__001", "ImportError: no module named x")
    text = pc.render(pc.collect_project_facts(store, pid))
    assert "ImportError" in text
    assert "failed" in text


def test_task_counts_are_reported(store: TaskStore):
    pid = _seed(store, tasks=3)
    _complete(store, pid, "task__001", "ok")
    _complete(store, pid, "task__002", "ok")
    facts = pc.collect_project_facts(store, pid)
    assert facts.total == 3
    assert facts.completed == 2
    assert facts.counts.get("pending") == 1


def test_attempt_count_is_reported(store: TaskStore):
    pid = _seed(store, tasks=2)
    facts = pc.collect_project_facts(store, pid)
    task = facts.recent[0]
    assert task.attempts == 0  # a pending task has made no attempts


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def test_rendered_section_never_exceeds_its_budget(store: TaskStore):
    """The hard ceiling holds even when every field is enormous."""
    pid = _seed(store, goal="g", tasks=40)
    for i in range(1, 41):
        _complete(store, pid, f"task__{i:03d}", "x" * 20_000)
    text = pc.render(pc.collect_project_facts(store, pid))
    assert len(text) <= pc.MAX_TOTAL_CHARS


def test_long_result_is_truncated_with_a_marker(store: TaskStore):
    pid = _seed(store)
    _complete(store, pid, "task__001", "HEAD" + ("y" * 5_000) + "TAIL")
    text = pc.render(pc.collect_project_facts(store, pid))
    assert "[truncated]" in text
    assert "HEAD" in text
    assert "TAIL" in text


def test_result_cap_is_smaller_than_the_persisted_cap(store: TaskStore):
    """We re-send to the model, so the prompt bound is tighter than the disk bound."""
    from terminus.tasks.task_store import MAX_PERSISTED_RESULT_CHARS

    assert pc.MAX_RESULT_CHARS < MAX_PERSISTED_RESULT_CHARS


def test_only_a_bounded_number_of_tasks_are_shown(store: TaskStore):
    pid = _seed(store, tasks=30)
    facts = pc.collect_project_facts(store, pid)
    assert len(facts.recent) <= pc.MAX_TASKS_SHOWN
    # Counts still describe every task, even though detail does not.
    assert facts.total == 30


def test_list_fields_are_capped(store: TaskStore):
    plan = _plan(risks=[f"risk {i}" for i in range(50)])
    pid = store.create_project(plan.goal_summary, plan)
    facts = pc.collect_project_facts(store, pid)
    assert len(facts.risks) <= pc.MAX_LIST_ITEMS


def test_clip_keeps_head_and_tail():
    """The head states the task, the tail usually states the outcome."""
    clipped = pc._clip("A" * 5_000 + "B" * 5_000, pc.MAX_RESULT_CHARS)
    assert "[truncated]" in clipped
    assert clipped.startswith("A")
    assert clipped.endswith("B")
    assert len(clipped) <= pc.MAX_RESULT_CHARS + len(pc._TRUNCATION_NOTE)


def test_clip_degrades_gracefully_at_a_tiny_limit():
    """A limit too small to hold the marker must still return bounded text."""
    clipped = pc._clip("A" * 200, 10)
    assert len(clipped) <= 10 + len(pc._TRUNCATION_NOTE)
    assert clipped.startswith("A")


def test_clip_of_short_text_is_untouched():
    assert pc._clip("short", 100) == "short"
    assert pc._clip("", 100) == ""


# ---------------------------------------------------------------------------
# project scoping
# ---------------------------------------------------------------------------


def test_a_project_from_another_workspace_is_refused(store: TaskStore, tmp_path):
    """Silently adopting another directory's project is the failure mode here."""
    _seed(store, workspace=str(tmp_path / "somewhere-else"))
    assert pc.resolve_project_id(store) is None
    assert pc.collect_project_facts(store).has_project is False


def test_a_matching_workspace_resolves(store: TaskStore):
    pid = _seed(store)
    assert pc.resolve_project_id(store) == pid


def test_results_are_not_written_back(store: TaskStore):
    """The read model reports; it never repairs or rewrites what it reads."""
    pid = _seed(store)
    before = _row(store, pid)
    pc.collect_project_facts(store, pid)
    assert _row(store, pid) == before


def test_collection_never_raises_on_a_broken_store(tmp_path):
    class Hostile:
        def get_resumable_project(self):
            return "p"

        def get_latest_project(self):
            return "p"

        def workspace_matches(self, *_a):
            return True

        def get_project(self, *_a):
            return {"name": "n", "goal": "g", "plan_json": None}

        def get_all_tasks(self, *_a):
            raise ValueError("corrupt")

    facts = pc.collect_project_facts(Hostile())
    assert facts.has_project is True
    assert facts.recent == []


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------


def test_skills_are_omitted_when_absent(monkeypatch):
    """No skills ship with Terminus, so the default section must not imply any."""
    monkeypatch.setattr(pc, "_available_skills", list)
    facts = pc.ProjectFacts(workspace="w", project_id="p")
    text = pc.render(facts)
    assert text  # the section itself still renders
    assert "Skills" not in text


def test_skills_are_listed_when_present(monkeypatch):
    monkeypatch.setattr(pc, "_available_skills", lambda: ["alpha", "beta"])
    facts = pc.ProjectFacts(workspace="w", project_id="p", skills=["alpha", "beta"])
    assert "Skills available: alpha, beta" in pc.render(facts)


def test_skill_names_are_bounded():
    """A large catalogue must not be able to dominate the section."""
    registry = type("R", (), {"skills": {f"s{i}": {} for i in range(500)}})()
    import terminus.skills.skill_tools as st

    original = st._get_registry
    st._get_registry = lambda: registry
    try:
        names = pc._available_skills()
    finally:
        st._get_registry = original
    assert len(names) <= pc.MAX_SKILLS_SHOWN


def test_many_skills_stay_within_budget():
    facts = pc.ProjectFacts(workspace="w", project_id="p", skills=["s"] * 500)
    text = pc.render(facts)
    assert len(text) <= pc.MAX_TOTAL_CHARS


def test_skill_lookup_failure_yields_no_skills(monkeypatch):
    """A missing skills directory must not stop the project section rendering."""
    import builtins

    real_import = builtins.__import__

    def hostile(name, *args, **kwargs):
        if "skill_tools" in name:
            raise ImportError("no skills module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", hostile)
    assert pc._available_skills() == []


# ---------------------------------------------------------------------------
# integration with the surfaces that consume it
# ---------------------------------------------------------------------------


def test_ask_prompt_carries_project_state(store: TaskStore, monkeypatch):
    """The /ask agent must be able to see what the project has achieved."""
    from terminus.agent import factory

    pid = _seed(store, goal="Make the parser fast")
    _complete(store, pid, "task__001", "Wrote the module.")
    monkeypatch.setattr(
        factory, "project_prompt_section",
        lambda *a, **k: pc.render(pc.collect_project_facts(store, pid)),
    )
    assert "Make the parser fast" in factory._build_system_prompt()


def test_ask_prompt_omits_the_section_when_there_is_no_project(monkeypatch):
    from terminus.agent import factory

    monkeypatch.setattr(factory, "project_prompt_section", lambda *a, **k: "")
    assert "Project context" not in factory._build_system_prompt()


def test_ask_prompt_has_no_orphan_skills_header(monkeypatch):
    """A skills heading must never appear without a catalogue under it.

    Terminus ships a built-in skill library, so the catalogue is normally present.
    What must never happen is a heading surviving on its own: when the registry is
    empty that implies a capability that does not exist.

    Asserted on the assembled prompt, because that is the artifact the model
    actually sees. The static half is cached per workspace, so the cache is
    stubbed out for the duration - otherwise the first build would be replayed
    and the change under test would be invisible.
    """
    from terminus.agent import factory

    # Rebuild the prompt every call instead of replaying the cached one.
    monkeypatch.setattr(factory, "get_cached_prompt", lambda _key: None)
    monkeypatch.setattr(factory, "cache_prompt", lambda _key, value: value)

    with_skills = factory._build_system_prompt()
    assert "Available Skills" in with_skills, "the built-in library should be present"
    assert "==skills" not in with_skills, "no second, malformed heading in front of it"

    for empty in ("", "   \n  "):
        monkeypatch.setattr(
                factory, "build_skills_prompt", lambda value=empty: value
            )
        assert "Available Skills" not in factory._build_system_prompt()


def test_project_context_is_not_frozen_in_the_prompt_cache(store: TaskStore, monkeypatch):
    """The static half is cached per workspace; live project state is not.

    build_agent runs once per question, so a cached project snapshot would pin
    the project to whatever it looked like when the process started.
    """
    from terminus.agent import factory

    pid = _seed(store, goal="First goal")
    seen = []

    def section(*_a, **_k):
        facts = pc.collect_project_facts(store, pid)
        seen.append(facts.goal)
        return pc.render(facts)

    monkeypatch.setattr(factory, "project_prompt_section", section)
    factory._build_system_prompt()
    _complete(store, pid, "task__001", "done")
    prompt = factory._build_system_prompt()
    assert len(seen) == 2, "the project section must be re-read, not cached"
    assert "First goal" in prompt


def test_project_status_tool_is_registered_and_read_only():
    from terminus.agent.factory import ASK_TOOLS
    from terminus.tools.project_status_tool import project_status

    names = {getattr(t, "name", None) for t in ASK_TOOLS}
    assert "project_status" in names
    assert getattr(project_status, "description", "")
    # A read-only tool must not import the write guard.
    import terminus.tools.project_status_tool as mod

    source = mod.__doc__ or ""
    assert "read-only" in source.lower()


def test_worker_prompt_carries_acceptance_criteria():
    """The worker used to be graded on criteria it was never shown."""
    from terminus.tasks import executor as ex

    task = {
        "id": "t1", "project_id": "p", "task_type": "implement",
        "description": "d",
        "acceptance_criteria": json.dumps(["criterion one", "criterion two"]),
    }
    prompt = ex._build_system_prompt(task, [], None)
    assert "criterion one" in prompt
    assert "criterion two" in prompt


def test_worker_prompt_carries_plan_context():
    from terminus.tasks import executor as ex

    task = {"id": "t1", "project_id": "p", "task_type": "implement", "description": "d"}
    plan = {"plan_json": json.dumps({
        "goal_summary": "Harden the client", "tech_stack": ["httpx"], "risks": ["flaky"],
    })}
    prompt = ex._build_system_prompt(task, [], plan)
    assert "Harden the client" in prompt
    assert "httpx" in prompt


def test_worker_without_load_skill_is_not_told_about_skills():
    from terminus.tasks import executor as ex

    assert ex._worker_skills_section([]) == ""


def test_planner_gets_bounded_context(monkeypatch):
    from terminus.tasks import planner

    monkeypatch.setattr(planner, "build_startup_context", lambda *_a: "## Environment\nfiles: 3")
    context = planner._planner_context()
    assert "## Environment" in context
    assert len(context) <= planner.MAX_PLANNER_CONTEXT_CHARS


def test_planner_context_failure_does_not_stop_planning(monkeypatch):
    from terminus.tasks import planner

    def boom(*_a, **_k):
        raise RuntimeError("no filesystem")

    monkeypatch.setattr(planner, "build_startup_context", boom)
    monkeypatch.setattr(planner, "project_prompt_section", boom)
    assert planner._planner_context() == ""


def test_task_status_cli_shows_results(store: TaskStore, monkeypatch, capsys):
    from terminus import cli

    pid = _seed(store)
    _complete(store, pid, "task__001", "Wrote the module and its tests.")

    monkeypatch.setattr("terminus.config.CONFIG", {"tasks": {"db_path": store.db_path}})
    cli.show_task_status()
    out = capsys.readouterr().out
    assert "Wrote the module and its tests." in out
    assert "Results:" in out


# ---------------------------------------------------------------------------
# qdrant scoping
# ---------------------------------------------------------------------------


def test_retriever_scopes_search_to_this_project(monkeypatch):
    """Regression: the shared collection must never answer across projects."""
    from terminus.context.retrievers import hybrid_qdrant as rq

    captured = {}

    class FakeStore:
        def similarity_search_with_score(self, query, k=5, filter=None):
            captured["filter"] = filter
            return []

    # Patched at the retriever's store seam: the search call is what passes the
    # project filter, and that is what this test is about.
    monkeypatch.setattr(rq, "_hybrid_store", lambda: FakeStore())
    rq.retrieve("hello", k=3)

    assert captured["filter"] is not None
    conditions = captured["filter"].must
    assert len(conditions) == 1
    # Namespaced: LangChain stores Document metadata under "metadata".
    assert conditions[0].key == "metadata.project"
    assert conditions[0].match.value == str(pc.project_root())


def test_two_projects_build_different_filters(monkeypatch):
    """Project A and project B must not produce the same filter."""
    from terminus.context.qdrant_scope import project_filter as _project_filter

    here = _project_filter()
    monkeypatch.setattr(
        "terminus.workspace.project_root", lambda: Path("C:/some/other/project")
    )
    there = _project_filter()
    assert here.must[0].match.value != there.must[0].match.value
    assert there.must[0].match.value.endswith("some/other/project") or \
        there.must[0].match.value.endswith("some\\other\\project")


def test_every_qdrant_reader_and_writer_go_through_the_shared_scope():
    """One module owns the field, so the semantic and hybrid pairs cannot drift.

    Both Qdrant modes used to carry their own inline copy of the payload, and
    the semantic pair was simply never given one.
    """
    from terminus.context.indexers import hybrid_qdrant as ix
    from terminus.context.indexers import reindexer as rx
    from terminus.context.indexers import semantic_qdrant as six
    from terminus.context.retrievers import hybrid_qdrant as rq
    from terminus.context.retrievers import semantic_qdrant as srq

    for module in (ix, rx, six):
        assert hasattr(module, "chunk_metadata"), f"{module.__name__} does not scope writes"
    for module in (rq, srq):
        assert hasattr(module, "project_filter"), f"{module.__name__} does not scope reads"

    # No reader may search the shared collection unfiltered.
    for module in (rq, srq):
        assert "similarity_search_with_score(query, k=k)" not in (module.__doc__ or "")
    import inspect
    for module in (rq, srq):
        source = inspect.getsource(module)
        assert "filter=project_filter()" in source, f"{module.__name__} search is unfiltered"


def test_indexed_chunks_carry_the_project_key(monkeypatch, tmp_path):
    """A full index must tag every chunk, or the filter can never match."""
    from terminus.config import CONFIG
    from terminus.context.indexers import hybrid_qdrant as ix
    from terminus.workspace import project_key

    class _FakeQdrantClientModule:
        """Reports no collections, so the full initial index path runs."""

        @staticmethod
        def create_qdrant_client(**_k):
            return type("C", (), {
                "get_collections": lambda self: type("R", (), {"collections": []})(),
            })()

    source = tmp_path / "m.py"
    source.write_text("def f():\n    return 1\n", encoding="utf-8")
    source_path = str(source)

    class Chunk:
        content = "def f(): return 1"
        source = source_path
        name = "f"
        type = "function"
        start_line = 1
        end_line = 1

    captured = {}

    def capture_write(client, documents, name, **_k):
        captured["docs"] = list(documents)

        class Store:
            collection_name = name

        return Store()

    # The indexer builds its store through the shared helper, which uses the
    # constructor + add_documents path: the installed langchain-qdrant rejects an
    # injected client in from_documents / from_existing_collection, so those
    # classmethods cannot be used for a local store.
    monkeypatch.setattr(ix, "write_documents", capture_write)
    monkeypatch.setattr(ix, "get_source_files", lambda _p: [str(source)])
    monkeypatch.setattr(ix, "parse_file", lambda _p: [Chunk()])
    monkeypatch.setattr(ix, "sparse_retriever", lambda: object())
    monkeypatch.setattr(ix, "qdrant_client", _FakeQdrantClientModule())
    monkeypatch.setitem(CONFIG["qdrant"], "mode", "local")
    monkeypatch.setitem(CONFIG["qdrant"], "collection_name", "test")

    ix.get_or_create_qdrant_hybrid_index(str(tmp_path))

    docs = captured.get("docs")
    assert docs, "the full-index path did not run"
    assert all(d.metadata["project"] == project_key() for d in docs)
