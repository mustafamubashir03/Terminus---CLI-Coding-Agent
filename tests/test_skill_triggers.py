"""Trigger evaluation for the built-in library.

Precision is the property that matters. A skill loaded for the wrong reason
costs context on every call and actively degrades output, so a library that fires
on everything is worse than a small one that fires accurately. These are
behavioural checks against realistic prompts, not unit tests of the scorer.
"""

from __future__ import annotations

import pytest

from terminus.skills.matcher import match_skills
from terminus.skills.registry import SkillRegistry, builtin_skills_dir

# (prompt, must_include, must_not_include)
POSITIVE = [
    ("Build a polished SaaS dashboard in React", {"frontend-design", "react-best-practices"}, set()),
    ("Redesign our marketing landing page so it looks modern", {"frontend-design"}, {"web-design-guidelines"}),
    ("Review my UI for accessibility and keyboard problems", {"web-design-guidelines"}, set()),
    ("This React component re-renders on every keystroke", {"react-best-practices"}, set()),
    ("The Next.js bundle is too large, cut the first-load JS", {"react-best-practices"}, set()),
    ("It works locally but fails in CI, and it is intermittent", {"systematic-debugging"}, set()),
    ("Production is throwing 500s since the last deploy", {"systematic-debugging"}, set()),
    ("Write a test for the invoice rounding logic", {"test-driven-development"}, set()),
    ("Add a regression test so this bug cannot come back", {"test-driven-development"}, set()),
    ("Implement this Figma design in the dashboard", {"figma-implement-design"}, {"web-design-guidelines"}),
    ("Build an MCP server exposing our order tools", {"mcp-builder"}, set()),
    ("Create a new skill for our release process", {"skill-creator"}, set()),
    ("Take a screenshot and check the button works in the browser", {"webapp-testing"}, set()),
    ("Reproduce this browser bug and confirm the fix", {"webapp-testing"}, set()),
    # Added with the adapted third-party library.
    ("Design the database schema for invoices and line items", {"database"}, set()),
    ("Write a migration that backfills the new tenant_id column", {"database"}, set()),
    ("This query is slow, add an index on orders by tenant", {"database"}, set()),
    ("Add row level security so tenants cannot see each other rows", {"database"}, set()),
    ("Add animations, the transitions feel janky", {"motion-polish"}, set()),
    ("Make it feel like Linear, add some life to this", {"motion-polish"}, set()),
    ("Stagger the list items as they load", {"motion-polish"}, set()),
    ("Which component library should I use for this dashboard?", {"ui-patterns"}, set()),
    ("Build a settings page with grouped sections", {"ui-patterns"}, set()),
    ("What should the empty state of this table say?", {"ui-patterns"}, set()),
    ("Write tailwind classes for a responsive 3 column grid", {"tailwindcss-development"}, set()),
    ("Add a dark mode variant to this card", {"tailwindcss-development"}, set()),
]

NEGATIVE = [
    "Fix MongoDB connection timeout in Python",
    "Add a REST endpoint for invoices",
    "Bump the dependency versions in requirements.txt",
    "Rename a variable for consistency",
    "Document the API in the README",
    "Configure GitHub Actions to cache pip",
    "Update the Dockerfile base image",
    "Set up logging for the worker process",
    "Increase the connection pool size in the database config",
    "Refactor the billing module into smaller services",
    "Add a Redis cache in front of the user lookup",
    "Configure the load balancer health check",
]


@pytest.fixture(scope="module")
def library():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    return reg


@pytest.mark.parametrize("prompt,expected,forbidden", POSITIVE, ids=[p[0][:40] for p in POSITIVE])
def test_a_relevant_prompt_selects_the_right_skill(library, prompt, expected, forbidden):
    selected = {m.name for m in match_skills(library, prompt) if m.selected}
    missing = expected - selected
    assert not missing, f"{prompt!r} did not select {sorted(missing)} (got {sorted(selected)})"
    leaked = selected & forbidden
    assert not leaked, f"{prompt!r} wrongly selected {sorted(leaked)}"


@pytest.mark.parametrize("prompt", NEGATIVE, ids=[p[:40] for p in NEGATIVE])
def test_an_unrelated_prompt_selects_nothing(library, prompt):
    """Backend, infrastructure and documentation work must not load a UI skill."""
    selected = {m.name for m in match_skills(library, prompt) if m.selected}
    assert not selected, f"{prompt!r} wrongly selected {sorted(selected)}"


@pytest.mark.parametrize(
    "prompt,forbidden",
    [
        ("Build a polished React dashboard", {"motion-polish", "database"}),
        ("Add a dark mode variant to this card", {"frontend-design"}),
        ("Review my UI for accessibility", {"ui-patterns", "motion-polish"}),
        ("Add a REST endpoint for invoices", {"database", "ui-patterns"}),
    ],
    ids=["generic-polish", "tailwind-not-a-design-call", "audit-is-not-composition", "backend"],
)
def test_the_new_skills_do_not_fire_on_unrelated_work(library, prompt, forbidden):
    """A generic adjective must not select a skill that happens to be named after it."""
    selected = {m.name for m in match_skills(library, prompt) if m.selected}
    leaked = selected & forbidden
    assert not leaked, f"{prompt!r} wrongly selected {sorted(leaked)}"


def test_frontend_and_audit_skills_coexist_on_a_compound_request(library):
    """'Redesign it and audit it' genuinely wants both, and must get both."""
    selected = {
        m.name
        for m in match_skills(library, "Redesign the settings page and audit its accessibility")
        if m.selected
    }
    assert "frontend-design" in selected
    assert "web-design-guidelines" in selected


def test_an_explicit_named_skill_is_always_honoured(library):
    selected = {
        m.name
        for m in match_skills(library, "do something unrelated", explicit=["mcp-builder"])
        if m.selected
    }
    assert selected == {"mcp-builder"}


def test_naming_a_skill_overrides_its_own_non_trigger(library):
    """A user asking for a skill by name beats the matcher's own judgement."""
    selected = {
        m.name
        for m in match_skills(library, "build me a new dashboard", explicit=["web-design-guidelines"])
        if m.selected
    }
    assert "web-design-guidelines" in selected


def test_selection_stays_within_the_per_execution_budget(library):
    selected = [
        m for m in match_skills(
            library, "React frontend design audit MCP testing debugging skills", threshold=0.0
        ) if m.selected
    ]
    from terminus.skills.registry import MAX_SKILLS_PER_TASK

    assert len(selected) <= MAX_SKILLS_PER_TASK


def test_the_library_has_no_skill_nobody_can_reach(library):
    """Every built-in must fire on at least one realistic prompt."""
    fired = {m.name for m in match_skills(library, "anything", include_all=True, threshold=0.0)}
    for prompt, _expected, _forbidden in POSITIVE:
        fired |= {m.name for m in match_skills(library, prompt) if m.selected}
    unreachable = set(library.refs) - fired
    assert not unreachable, f"no realistic prompt reaches {sorted(unreachable)}"
