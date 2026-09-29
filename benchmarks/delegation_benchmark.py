"""Does delegation actually make Terminus better?

An A/B over a simulated repository with known ground truth, comparing:

    A. one agent, one context, one budget
    B. a parent that delegates to bounded children and aggregates

The simulation is deliberate and the harness is deterministic. The question is
not whether a real model reasons better - it is whether the *orchestration shape*
buys anything, and that is a property of budgets, context and fan-out rather
than of model quality. A child can only find what its own context contains, so
delegation's value is exactly the value of splitting one budget across several
focused contexts, and its cost is exactly the cost of running several agents.

No network, no model, no timing luck. Numbers here are reproducible.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# a simulated repository with known answers
# ---------------------------------------------------------------------------

# file -> the facts a competent agent could discover by reading it
REPO: dict[str, list[str]] = {
    "src/auth/oauth.py": ["oauth:token_exchange retries 3x on any 5xx"],
    "src/auth/session.py": ["session:cookie lacks SameSite on the callback"],
    "src/billing/invoice.py": ["billing:invoice totals recomputed per row"],
    "src/billing/tax.py": ["tax:rate fetched from the network per line item"],
    "src/api/routes.py": ["api:route missing auth middleware"],
    "src/web/dashboard.tsx": ["ui:dashboard renders a 200-row table unvirtualized"],
    "src/web/table.tsx": ["ui:table sorts on every keystroke"],
    "src/web/forms.tsx": ["ui:form labels are placeholders only"],
    "src/worker/queue.py": ["worker:no backoff on retry"],
}

TASKS: list["BenchmarkTask"] = []


@dataclass
class BenchmarkTask:
    name: str
    kind: str
    question: str
    areas: list[str]
    ground_truth: list[str]
    files: list[str]
    single_context: int
    """How many files a single agent can afford to read."""
    child_context: int
    """How many files one child can afford to read."""
    single_calls: int = 0
    child_calls: int = 0
    min_agents: int = 0
    max_agents: int = 1
    max_children: int = 1


TASKS = [
    BenchmarkTask(
        name="unfamiliar-subarea",
        kind="investigation",
        question="The OAuth login fails intermittently in production. Find the cause.",
        areas=["auth"],
        ground_truth=[
            "oauth:token_exchange retries 3x on any 5xx",
            "session:cookie lacks SameSite on the callback",
        ],
        files=["src/auth/oauth.py", "src/auth/session.py", "src/billing/tax.py"],
        single_context=2, child_context=2, min_agents=1, max_agents=2, max_children=2,
    ),
    BenchmarkTask(
        name="multi-area-bug",
        kind="investigation",
        question="Checkout is slow and sometimes returns a 500. Find the causes.",
        areas=["billing", "api", "auth"],
        ground_truth=[
            "billing:invoice totals recomputed per row",
            "tax:rate fetched from the network per line item",
            "api:route missing auth middleware",
            "oauth:token_exchange retries 3x on any 5xx",
        ],
        files=[
            "src/billing/invoice.py", "src/billing/tax.py", "src/api/routes.py",
            "src/auth/oauth.py", "src/worker/queue.py", "src/web/table.tsx",
        ],
        # One agent can only read 2 of the 6 candidate files.
        single_context=2, child_context=2, min_agents=3, max_agents=3, max_children=3,
    ),
    BenchmarkTask(
        name="frontend-review",
        kind="review",
        question="Review the dashboard for interface and performance problems.",
        areas=["ui"],
        ground_truth=[
            "ui:dashboard renders a 200-row table unvirtualized",
            "ui:table sorts on every keystroke",
            "ui:form labels are placeholders only",
        ],
        files=["src/web/dashboard.tsx", "src/web/table.tsx", "src/web/forms.tsx",
               "src/auth/session.py"],
        single_context=3, child_context=2, min_agents=1, max_agents=2, max_children=2,
    ),
    BenchmarkTask(
        name="simple-typo",
        kind="simple",
        question="Fix the typo in the copyright banner.",
        areas=["ui"],
        ground_truth=[],
        files=["src/web/forms.tsx"],
        single_context=1, child_context=1, min_agents=0, max_agents=1, max_children=1,
    ),
    BenchmarkTask(
        name="implement-and-review",
        kind="implement",
        question="Add a new API route and have it reviewed.",
        areas=["api", "auth"],
        ground_truth=[
            "api:route missing auth middleware",
            "oauth:token_exchange retries 3x on any 5xx",
        ],
        files=["src/api/routes.py", "src/auth/oauth.py", "src/billing/tax.py"],
        single_context=3, child_context=2, min_agents=1, max_agents=2, max_children=2,
    ),
]


# ---------------------------------------------------------------------------
# the two strategies
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    strategy: str
    task: str
    agents: int = 0
    tool_calls: int = 0
    context_chars: int = 0
    findings: list[str] = field(default_factory=list)
    duplicated: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    @property
    def found(self) -> int:
        return len(self.findings)

    def recall(self, truth: Sequence[str]) -> float:
        if not truth:
            return 1.0
        hit = sum(1 for t in truth if t in self.findings)
        return hit / len(truth)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["found"] = self.found
        return data


def _read_file(path: str) -> str:
    """The only capability an agent has. Costs one tool call."""
    facts = REPO.get(path, [])
    body = "\n".join(f"- {f}" for f in facts)
    return f"{path}\n{body}\n"


def _solver(paths: Sequence[str], budget: int) -> tuple[list[str], int, int]:
    """Read up to *budget* of *paths*; return findings, calls, context size."""
    findings: list[str] = []
    chars = 0
    calls = 0
    for path in paths[:budget]:
        chunk = _read_file(path)
        chars += len(chunk)
        calls += 1
        for fact in REPO.get(path, []):
            if fact not in findings:
                findings.append(fact)
    return findings, calls, chars


def run_single(task: BenchmarkTask) -> RunMetrics:
    """A: one agent, one context, one budget. No delegation."""
    started = time.perf_counter()
    findings, calls, chars = _solver(task.files, task.single_context)
    return RunMetrics(
        strategy="single", task=task.name, agents=1, tool_calls=calls,
        context_chars=chars, findings=findings,
        seconds=time.perf_counter() - started,
    )


async def run_delegated(
    task: BenchmarkTask, spawn: Callable[..., "asyncio.Future"]
) -> RunMetrics:
    """B: a parent delegates to bounded children and aggregates."""
    started = time.perf_counter()
    children = min(task.max_children, max(1, len(task.files)))
    # Split the file list into contiguous slices, one per child, so each child
    # gets a genuinely focused context rather than a copy of everything.
    slices: list[list[str]] = [[] for _ in range(children)]
    for index, path in enumerate(task.files):
        slices[index % children].append(path)

    calls = 0
    chars = 0
    seen: list[str] = []
    duplicated: list[str] = []
    for child_paths in slices:
        findings, child_calls, child_chars = await spawn(child_paths, task.child_context)
        calls += child_calls
        chars += child_chars
        for fact in findings:
            if fact in seen:
                duplicated.append(fact)
            else:
                seen.append(fact)

    return RunMetrics(
        strategy="delegated", task=task.name, agents=children + 1,
        tool_calls=calls, context_chars=chars, findings=seen, duplicated=duplicated,
        seconds=time.perf_counter() - started,
    )


async def _fake_spawn(paths: Sequence[str], budget: int):
    return _solver(paths, budget)


def run_benchmark() -> dict:
    """Run every task under both strategies and summarise."""
    rows: list[dict] = []
    for task in TASKS:
        single = run_single(task)
        delegated = asyncio.run(run_delegated(task, _fake_spawn))
        for metrics in (single, delegated):
            metrics.recall = task.ground_truth and len(
                [t for t in task.ground_truth if t in metrics.findings]
            ) / len(task.ground_truth) or 1.0
        rows.append({
            "task": task.name,
            "kind": task.kind,
            "min_agents": task.min_agents,
            "single": {**single.as_dict(), "recall": single.recall},
            "delegated": {**delegated.as_dict(), "recall": delegated.recall},
        })
    return {"rows": rows}


def _verdict(row: dict) -> str:
    s, d = row["single"], row["delegated"]
    if d["recall"] > s["recall"]:
        return "delegation helps"
    if d["recall"] < s["recall"]:
        return "delegation hurts"
    if d["agents"] > s["agents"]:
        return "delegation costs more for nothing"
    return "tie"


if __name__ == "__main__":  # pragma: no cover
    report = run_benchmark()
    header = (
        f"{'task':22s} {'strat':10s} {'agents':>6s} {'calls':>6s} "
        f"{'ctx':>6s} {'found':>5s} {'recall':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in report["rows"]:
        for key in ("single", "delegated"):
            m = row[key]
            print(
                f"{row['task']:22s} {m['strategy']:10s} {m['agents']:6d} "
                f"{m['tool_calls']:6d} {m['context_chars']:6d} {m['found']:5d} "
                f"{m['recall']:6.0%}"
            )
        print(f"{'':22s} -> {_verdict(row)}")
    print()
    wins = [r for r in report["rows"] if _verdict(r) == "delegation helps"]
    losses = [r for r in report["rows"] if _verdict(r) == "delegation hurts"]
    waste = [r for r in report["rows"] if _verdict(r) == "delegation costs more for nothing"]
    print(f"helps:  {len(wins)}  {[r['task'] for r in wins]}")
    print(f"hurts:  {len(losses)}  {[r['task'] for r in losses]}")
    print(f"waste:  {len(waste)}  {[r['task'] for r in waste]}")
    print(f"json:   {json.dumps({'verdicts': {r['task']: _verdict(r) for r in report['rows']}})}")
