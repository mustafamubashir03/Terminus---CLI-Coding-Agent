"""The checkpointer's lifecycle, across event loops.

``AsyncSqliteSaver`` is built for the loop it was constructed on: it captures that
loop and guards its writes with an ``asyncio.Lock``, which binds on first
contention. Handing one to a different loop raises ``RuntimeError: ... is bound to
a different event loop``. The CLI runs one loop for the life of the process and
never sees it; anything that calls ``asyncio.run`` more than once does.

These drive a real checkpointed graph rather than the saver directly, because that
is the situation that failed: a compiled graph resumed on a second event loop. Each
test builds its own graph and its own database file, so none depends on another
having run first.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import TypedDict

import pytest

from terminus.config import CONFIG
from terminus.memory import short_term


class Counter(TypedDict):
    n: int


@pytest.fixture(autouse=True)
def isolated_checkpointer(tmp_path, monkeypatch):
    """A private database file, no cached saver, and a closed connection afterwards."""
    db_path = tmp_path / "memory" / "t.db"
    monkeypatch.setitem(CONFIG["memory"], "db_path", str(db_path))
    for name in ("_checkpointer", "_db_conn", "_checkpointer_loop"):
        monkeypatch.setattr(short_term, name, None)
    yield db_path
    try:
        asyncio.run(short_term.close_checkpointer())
    except Exception:
        pass


def build_graph():
    """A one-node graph, compiled per run because the checkpointer is per loop."""
    from langgraph.graph import END, START, StateGraph

    def bump(state: Counter) -> Counter:
        return {"n": state["n"] + 1}

    graph = StateGraph(Counter)
    graph.add_node("bump", bump)
    graph.add_edge(START, "bump")
    graph.add_edge("bump", END)
    return graph


async def _run(graph, thread: str, n: int | None) -> int:
    """Run *graph* on this loop against *thread*, via the shared checkpointer.

    ``n=None`` sends no input for the channel, which is how LangGraph is asked to
    resume: input channels are merged over the checkpoint, so passing ``n`` again
    would overwrite the saved value rather than continue from it.
    """
    compiled = graph.compile(checkpointer=await short_term.get_checkpointer())
    payload = {} if n is None else {"n": n}
    result = await compiled.ainvoke(payload, config={"configurable": {"thread_id": thread}})
    return result["n"]


# ---------------------------------------------------------------------------
# the loop contract
# ---------------------------------------------------------------------------

def test_one_loop_reuses_one_saver():
    """Same loop, same object: /ask, a child and a worker share one connection."""

    async def run():
        first = await short_term.get_checkpointer()
        second = await short_term.get_checkpointer()
        return first, second

    first, second = asyncio.run(run())
    assert first is second, "the same event loop must not churn connections"


def test_a_second_loop_gets_its_own_saver():
    """The invariant the fix exists for: never hand a saver to a foreign loop.

    Asserted on object identity rather than on a raised error, because whether the
    library's lock actually complains depends on two operations happening to
    contend - which is timing. Identity is the contract.
    """
    first = asyncio.run(short_term.get_checkpointer())
    second = asyncio.run(short_term.get_checkpointer())

    assert first is not second
    assert first.loop is not second.loop


def test_three_loops_in_a_row_each_work():
    """Repeated loop changes must not accumulate an unusable cached saver."""
    graph = build_graph()
    for index in range(3):
        assert asyncio.run(_run(graph, f"thread-{index}", 1)) == 2


# ---------------------------------------------------------------------------
# the original failure: a checkpointed graph resumed on a second loop
# ---------------------------------------------------------------------------

def test_a_graph_resumes_on_a_second_event_loop(isolated_checkpointer):
    """The smallest realistic reproduction, and it must not raise.

    The same thread, driven from two independent loops, the second time with no
    input so it resumes. Without a loop-keyed saver the second call reuses a saver
    built on the first loop and the run fails on the saver's lock. Counting to 3
    rather than 2 proves the resume was real and not a fresh start.
    """
    graph = build_graph()

    assert asyncio.run(_run(graph, "shared-thread", 1)) == 2
    assert asyncio.run(_run(graph, "shared-thread", None)) == 3
    assert asyncio.run(_run(graph, "shared-thread", None)) == 4


def test_state_written_on_one_loop_is_readable_on_the_next(isolated_checkpointer):
    """State survives the saver being replaced: it lives in the file, not the object."""
    graph = build_graph()
    assert asyncio.run(_run(graph, "carried", 40)) == 41
    assert asyncio.run(_run(graph, "carried", None)) == 42


def test_separate_threads_stay_separate_across_loops(isolated_checkpointer):
    graph = build_graph()
    asyncio.run(_run(graph, "alpha", 1))
    asyncio.run(_run(graph, "beta", 10))
    # alpha ran once (1 -> 2) and resumes to 3; beta ran once (10 -> 11) and
    # resumes to 12. If the threads shared state these could not both hold.
    assert asyncio.run(_run(graph, "alpha", None)) == 3, "alpha resumed its own state"
    assert asyncio.run(_run(graph, "beta", None)) == 12, "beta did not inherit alpha's"


def test_the_database_file_is_created_where_configured(isolated_checkpointer):
    asyncio.run(_run(build_graph(), "file-check", 1))
    assert isolated_checkpointer.is_file()
    assert isolated_checkpointer.parent.is_dir()


def test_checkpoints_are_readable_by_plain_sqlite(isolated_checkpointer):
    """Durability is a property of the file, not of the aiosqlite wrapper."""
    asyncio.run(_run(build_graph(), "durable", 1))

    connection = sqlite3.connect(isolated_checkpointer)
    try:
        rows = connection.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id = ?",
            ("durable",),
        ).fetchall()
    finally:
        connection.close()
    assert rows == [("durable",)]


# ---------------------------------------------------------------------------
# the existing shutdown path
# ---------------------------------------------------------------------------

def test_close_drops_the_cache_and_the_next_call_reconnects(isolated_checkpointer):
    first = asyncio.run(short_term.get_checkpointer())

    asyncio.run(short_term.close_checkpointer())
    assert short_term._checkpointer is None
    assert short_term._db_conn is None
    assert short_term._checkpointer_loop is None

    assert asyncio.run(_run(build_graph(), "after-close", 1)) == 2
    assert asyncio.run(short_term.get_checkpointer()) is not first


def test_close_is_safe_when_nothing_was_opened():
    asyncio.run(short_term.close_checkpointer())


def test_short_term_is_the_only_place_a_saver_is_built():
    """One connection per process, so one module has to own its lifecycle."""
    import terminus

    root = Path(terminus.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "AsyncSqliteSaver(" in path.read_text(encoding="utf-8")
        and path.name != "short_term.py"
    ]
    assert not offenders, (
        f"another module builds its own saver: {offenders}. The cached one in "
        "memory/short_term.py is the only connection Terminus keeps."
    )
