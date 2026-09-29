"""The delegation benchmark is deterministic and its conclusion is recorded.

The benchmark is the reason this primitive has a ``min_agents``-style
expectation at all. It is not here to prove delegation is good; it is here so the
claim is falsifiable and so a future change to budgets, context assembly or fan-out
that starts paying for itself in agent count can be noticed.

Run ``python benchmarks/delegation_benchmark.py`` for the readable table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.delegation_benchmark import (  # noqa: E402
    TASKS,
    _verdict,
    run_benchmark,
)


@pytest.fixture(scope="module")
def report():
    return run_benchmark()


def test_every_task_runs_under_both_strategies(report):
    assert len(report["rows"]) == len(TASKS)
    for row in report["rows"]:
        assert row["single"]["strategy"] == "single"
        assert row["delegated"]["strategy"] == "delegated"


def test_delegation_never_lowers_recall(report):
    """Fan-out cannot lose information, because the parent aggregates everything."""
    for row in report["rows"]:
        assert row["delegated"]["recall"] >= row["single"]["recall"], row["task"]


def test_delegation_only_earns_its_cost_on_multi_area_work(report):
    """The recorded result, asserted so a regression is visible.

    Delegation pays for itself exactly when the problem does not fit in one
    agent's context budget. On the other four tasks it spends more agents,
    more tool calls and more context for identical recall.
    """
    verdicts = {row["task"]: _verdict(row) for row in report["rows"]}
    assert verdicts["multi-area-bug"] == "delegation helps"
    wasted = [task for task, verdict in verdicts.items() if "costs more" in verdict]
    assert set(wasted) == {
        "unfamiliar-subarea", "frontend-review", "simple-typo", "implement-and-review",
    }
    assert not [t for t, v in verdicts.items() if v == "delegation hurts"]


def test_the_helping_case_is_where_one_context_cannot_reach(report):
    """Sanity: the win comes from budget, not from a lucky prompt."""
    row = next(r for r in report["rows"] if r["task"] == "multi-area-bug")
    assert row["single"]["recall"] < 1.0, "the single agent should be unable to finish"
    assert row["delegated"]["recall"] == 1.0


def test_delegation_always_costs_more_agents(report):
    """There is no free delegation: the parent is an extra execution."""
    for row in report["rows"]:
        assert row["delegated"]["agents"] > row["single"]["agents"], row["task"]


def test_the_benchmark_is_reproducible():
    first = run_benchmark()
    second = run_benchmark()
    for a, b in zip(first["rows"], second["rows"], strict=False):
        assert a["single"]["findings"] == b["single"]["findings"]
        assert a["delegated"]["findings"] == b["delegated"]["findings"]
        assert a["delegated"]["recall"] == b["delegated"]["recall"]


def test_a_task_that_needs_no_agent_is_marked_as_such():
    """`simple-typo` is the case that must stay single-agent forever."""
    task = next(t for t in TASKS if t.name == "simple-typo")
    assert task.min_agents == 0
    assert task.max_children == 1
