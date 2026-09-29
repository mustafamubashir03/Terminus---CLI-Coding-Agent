"""Skills: discovery across sources, provenance, precedence, matching, and bounds.

The properties under test:

    A skill is found wherever it is installed, its origin is reported, a name
    claimed by several roots resolves to one winner deterministically, a
    malformed skill is skipped rather than fatal, explicit selection always
    wins, automatic selection is explainable and bounded, and nothing Terminus
    ships can blow the context budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from terminus.skills import matcher
from terminus.skills.matcher import (
    DEFAULT_THRESHOLD,
    match_skills,
    render_selection,
)
from terminus.skills.registry import (
    MAX_CATALOGUE_ENTRIES,
    MAX_SKILL_CHARS,
    MAX_SKILLS_PER_TASK,
    MAX_SKILLS_TOTAL_CHARS,
    SkillNotFoundError,
    SkillRegistry,
    builtin_skills_dir,
)


def write_skill(root: Path, name: str, *, description: str = "does a thing",
                body: str = "Body text.", when: str = "", tags: str = "",
                extra: str = "") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    front = [f"name: {name}", f"description: {description}"]
    if when:
        front.append(f"when_to_use: {when}")
    if tags:
        front.append(f"tags: {tags}")
    if extra:
        front.append(extra)
    (d / "SKILL.md").write_text(
        "---\n" + "\n".join(front) + "\n---\n\n" + body + "\n", encoding="utf-8"
    )
    return d


@pytest.fixture
def roots(tmp_path):
    return {
        "project": tmp_path / "project" / ".terminus" / "skills",
        "user": tmp_path / "user" / "skills",
        "builtin": tmp_path / "builtin",
    }


@pytest.fixture
def registry(roots):
    reg = SkillRegistry(sources=roots)
    reg.load()
    return reg


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_a_skill_is_found_in_each_source(roots):
    for source in ("project", "user", "builtin"):
        write_skill(roots[source], f"only_{source}", description=f"works in {source}")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert set(reg.refs) == {"only_project", "only_user", "only_builtin"}
    assert reg.refs["only_user"].source == "user"


def test_a_directory_without_skill_md_is_ignored(roots):
    (roots["project"] / "empty_dir").mkdir(parents=True)
    write_skill(roots["project"], "real", description="a real one")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert set(reg.refs) == {"real"}


def test_a_hidden_directory_is_ignored(roots):
    write_skill(roots["project"], ".git")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert reg.refs == {}


def test_all_builtin_skills_load(roots):
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    expected = {
        # First-party curated library.
        "frontend-design", "webapp-testing", "systematic-debugging",
        "test-driven-development", "react-best-practices",
        "web-design-guidelines", "figma-implement-design", "skill-creator",
        "mcp-builder",
        # Adapted from third-party sources, reviewed before inclusion.
        "database", "motion-polish", "ui-patterns", "tailwindcss-development",
    }
    assert set(reg.refs) == expected
    assert not reg.load_errors


# ---------------------------------------------------------------------------
# malformed / invalid
# ---------------------------------------------------------------------------


def test_a_malformed_skill_is_skipped_not_fatal(roots):
    bad = roots["project"] / "broken"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text("no frontmatter at all", encoding="utf-8")
    write_skill(roots["project"], "good", description="fine")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert "good" in reg.refs
    assert "broken" not in reg.refs
    assert reg.load_errors, "a skipped skill must be reported"


def test_invalid_frontmatter_is_reported(roots):
    bad = roots["project"] / "badyaml"
    bad.mkdir(parents=True)
    (bad / "SKILL.md").write_text("---\nname: [unclosed\n---\nBody\n", encoding="utf-8")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert reg.refs == {}
    assert reg.load_errors


def test_a_skill_with_no_name_falls_back_to_its_directory(roots):
    d = roots["project"] / "dirname"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\ndescription: x\n---\nBody\n", encoding="utf-8")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert "dirname" in reg.refs


def test_a_missing_root_is_not_an_error(roots):
    reg = SkillRegistry(sources={**roots, "user": roots["user"] / "nope"})
    reg.load()
    assert reg.refs == {}
    assert not reg.load_errors


# ---------------------------------------------------------------------------
# provenance and precedence
# ---------------------------------------------------------------------------


def test_duplicate_resolves_project_over_user_over_builtin(roots):
    for source in ("builtin", "user", "project"):
        write_skill(roots[source], "shared", description=f"from {source}")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert reg.refs["shared"].source == "project"
    assert reg.refs["shared"].description == "from project"


def test_a_shadowed_copy_is_recorded_not_discarded(roots):
    write_skill(roots["builtin"], "shared", description="builtin version")
    write_skill(roots["user"], "shared", description="user version")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert reg.refs["shared"].source == "user"
    assert [s.source for s in reg.shadowed["shared"]] == ["builtin"]


def test_dedupe_is_deterministic_across_runs(roots):
    for source in ("builtin", "user", "project"):
        write_skill(roots[source], "shared", description=f"from {source}")
    winners = set()
    for _ in range(5):
        reg = SkillRegistry(sources=roots)
        reg.load()
        winners.add(reg.refs["shared"].source)
    assert winners == {"project"}


def test_provenance_records_source_and_repository(roots):
    write_skill(roots["project"], "mine", description="x", extra="repository: acme/skills")
    reg = SkillRegistry(sources=roots)
    reg.load()
    ref = reg.refs["mine"]
    assert ref.provenance.startswith("project:")
    assert "acme/skills" in ref.provenance


def test_only_builtins_are_trusted(roots):
    write_skill(roots["builtin"], "b", description="x")
    write_skill(roots["user"], "u", description="x")
    write_skill(roots["project"], "p", description="x")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert reg.refs["b"].is_trusted is True
    assert reg.refs["u"].is_trusted is False
    assert reg.refs["p"].is_trusted is False


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_optional_metadata_is_parsed(roots):
    write_skill(roots["project"], "meta", description="x",
                extra="version: 2.1.0\nlicense: MIT")
    reg = SkillRegistry(sources=roots)
    reg.load()
    ref = reg.refs["meta"]
    assert ref.version == "2.1.0"
    assert ref.license == "MIT"
    assert ref.tags == []


def test_tags_may_be_a_list_or_a_string(roots):
    write_skill(roots["project"], "a", description="x", tags="[ui, design]")
    write_skill(roots["project"], "b", description="x", tags="ui, design")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert reg.refs["a"].tags == reg.refs["b"].tags == ["ui", "design"]


def test_version_and_license_are_optional(roots):
    write_skill(roots["project"], "bare", description="x")
    reg = SkillRegistry(sources=roots)
    reg.load()
    assert reg.refs["bare"].version == ""
    assert reg.refs["bare"].license == ""


# ---------------------------------------------------------------------------
# executable content
# ---------------------------------------------------------------------------


def test_executable_helpers_are_classified_but_listed_not_run(roots):
    d = write_skill(roots["project"], "withscript", description="x")
    (d / "scripts").mkdir()
    (d / "scripts" / "run.sh").write_text("echo hi", encoding="utf-8")
    (d / "references").mkdir()
    (d / "references" / "notes.md").write_text("notes", encoding="utf-8")
    reg = SkillRegistry(sources=roots)
    reg.load()
    ref = reg.refs["withscript"]
    assert ref.executable_files == ["scripts/run.sh"]
    assert ref.support_files == ["references/notes.md"]


def test_loading_a_skill_with_a_script_says_it_is_not_run(roots):
    d = write_skill(roots["project"], "withscript", description="x")
    (d / "scripts").mkdir()
    (d / "scripts" / "run.sh").write_text("echo hi", encoding="utf-8")
    reg = SkillRegistry(sources=roots)
    reg.load()
    text = reg.load_skill("withscript")
    assert "NOT run automatically" in text
    assert "permission approval" in text


def test_a_missing_skill_raises_with_the_available_names(roots):
    write_skill(roots["project"], "present", description="x")
    reg = SkillRegistry(sources=roots)
    reg.load()
    with pytest.raises(SkillNotFoundError) as caught:
        reg.load_skill("absent")
    assert "present" in str(caught.value)


# ---------------------------------------------------------------------------
# catalogue and bounds
# ---------------------------------------------------------------------------


def test_catalogue_is_metadata_only(registry):
    write_skill(registry.sources["project"], "shown", description="a described skill",
                body="UNIQUE_BODY_MARKER")
    registry.load()
    catalogue = registry.build_skills_prompt()
    assert "shown" in catalogue
    assert "UNIQUE_BODY_MARKER" not in catalogue, "bodies must not enter the catalogue"


def test_catalogue_is_empty_string_when_there_are_no_skills(registry):
    assert registry.build_skills_prompt() == ""


def test_catalogue_entry_count_is_capped(registry):
    for i in range(MAX_CATALOGUE_ENTRIES + 15):
        write_skill(registry.sources["project"], f"s{i:03d}", description=f"skill {i}")
    registry.load()
    catalogue = registry.build_skills_prompt()
    assert catalogue.count("Skill: ") <= MAX_CATALOGUE_ENTRIES
    assert "more skills installed" in catalogue


def test_a_skill_body_is_clipped(registry):
    write_skill(registry.sources["project"], "huge", description="x", body="Z" * 50_000)
    registry.load()
    text = registry.load_skill("huge")
    assert len(text) <= MAX_SKILL_CHARS + 400
    assert "omitted" in text


def test_the_selected_block_respects_the_total_budget(registry):
    for name in ("one", "two", "three"):
        write_skill(registry.sources["project"], name, description="x", body="Q" * 6000)
    registry.load()
    request = "one two three"
    matches = match_skills(registry, request, threshold=0.0)
    block = render_selection(matches, registry)
    assert len(block) <= MAX_SKILLS_TOTAL_CHARS
    assert "truncated" in block


def test_selection_respects_the_per_execution_skill_count(registry):
    for i in range(8):
        write_skill(registry.sources["project"], f"sk{i}", description="alpha beta gamma")
    registry.load()
    matches = match_skills(registry, "alpha beta gamma", threshold=0.1)
    assert len([m for m in matches if m.selected]) <= MAX_SKILLS_PER_TASK


def test_a_skipped_budget_overrun_is_explained(registry):
    for i in range(6):
        write_skill(registry.sources["project"], f"sk{i}", description="alpha beta")
    registry.load()
    matches = match_skills(registry, "alpha beta", threshold=0.1)
    dropped = [m for m in matches if not m.selected]
    assert dropped
    assert any("budget" in " ".join(m.reasons) for m in dropped)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------


def test_matching_returns_an_explainable_reason(registry):
    write_skill(registry.sources["project"], "reacty", description="use for react work",
                tags="[frontend]")
    registry.load()
    best = match_skills(registry, "fix the react component", include_all=True)[0]
    assert best.name == "reacty"
    assert best.reasons, "a selection must say why"


def test_an_explicit_skill_is_selected_even_with_no_signal(registry):
    write_skill(registry.sources["project"], "explicit_one", description="unrelated words")
    registry.load()
    matches = match_skills(registry, "completely different request", explicit=["explicit_one"])
    assert any(m.selected and m.explicit for m in matches)


def test_explicit_selection_beats_a_non_trigger(registry):
    d = write_skill(registry.sources["project"], "guarded", description="a skill",
                    extra="negative_triggers: [build]")
    d.mkdir(exist_ok=True)
    registry.load()
    auto = match_skills(registry, "build a thing")
    assert not [m for m in auto if m.name == "guarded" and m.selected]
    forced = match_skills(registry, "build a thing", explicit=["guarded"])
    assert [m for m in forced if m.selected and m.name == "guarded"]


def test_an_explicit_skill_that_does_not_exist_is_reported(registry):
    registry.load()
    matches = match_skills(registry, "x", explicit=["nonexistent"])
    missing = [m for m in matches if m.name == "nonexistent"]
    assert missing and not missing[0].selected
    assert "no such skill" in " ".join(missing[0].reasons)


def test_a_non_discriminative_name_word_gives_no_bonus(registry):
    for name in ("figma-design", "other-design"):
        write_skill(registry.sources["project"], name, description="totally different topic")
    registry.load()
    match = [m for m in match_skills(registry, "design", include_all=True) if m.name == "figma-design"][0]
    assert not any("request names 'design'" in r for r in match.reasons)


def test_a_discriminative_name_word_does_give_a_bonus(registry):
    write_skill(registry.sources["project"], "kubernetes", description="cluster work")
    registry.load()
    match = match_skills(registry, "fix the kubernetes deployment", include_all=True)[0]
    assert any("request names 'kubernetes'" in r for r in match.reasons)


def test_framework_signals_are_honoured(registry):
    write_skill(registry.sources["project"], "fw", description="framework guidance",
                tags="[nextjs]")
    registry.load()
    plain = [m for m in match_skills(registry, "improve the app", include_all=True) if m.name == "fw"]
    signalled = [m for m in match_skills(registry, "improve the app", frameworks=["nextjs"], include_all=True) if m.name == "fw"]
    assert signalled[0].score > plain[0].score


def test_stopwords_do_not_create_matches(registry):
    write_skill(registry.sources["project"], "tool", description="a utility thing")
    registry.load()
    assert not [m for m in match_skills(registry, "the and for with") if m.selected]


def test_conflicts_are_reported_between_overlapping_skills(registry):
    write_skill(registry.sources["project"], "alpha_ui", description="design interface layout")
    write_skill(registry.sources["project"], "beta_ui", description="design interface layout")
    registry.load()
    matches = match_skills(registry, "design interface layout", threshold=0.1)
    assert matcher.detect_conflicts(matches)


def test_two_unrelated_skills_are_not_a_conflict(registry):
    write_skill(registry.sources["project"], "aa", description="graph theory algorithms")
    write_skill(registry.sources["project"], "bb", description="kitchen bread baking")
    registry.load()
    matches = match_skills(registry, "algorithms", threshold=0.1)
    assert not matcher.detect_conflicts(matches)


def test_threshold_of_zero_selects_nothing_for_an_empty_request(registry):
    write_skill(registry.sources["project"], "x", description="y")
    registry.load()
    assert not [m for m in match_skills(registry, "") if m.selected]
    assert DEFAULT_THRESHOLD > 0


def test_inflected_wording_still_matches():
    """A description saying 'redesigning' must answer to a request saying 'redesign'."""
    assert "redesign" in matcher._tokens("redesigning")
    assert "audit" in matcher._tokens("audits")
    assert "test" in matcher._tokens("testing")


# ---------------------------------------------------------------------------
# the shipped library
# ---------------------------------------------------------------------------


def test_every_builtin_has_the_fields_that_matter():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    for name, ref in reg.refs.items():
        assert ref.name == name
        assert ref.description, f"{name} has no description"
        assert len(ref.description) > 60, f"{name} description is too weak to match on"
        assert ref.body.strip(), f"{name} has an empty body"
        assert len(ref.body) <= MAX_SKILL_CHARS, f"{name} exceeds the skill budget"
        assert ref.license, f"{name} records no licence"
        assert ref.version, f"{name} records no version"


def test_every_builtin_states_when_not_to_use_it():
    """A description with no boundary cannot be trusted to select precisely."""
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    for name, ref in reg.refs.items():
        lowered = ref.description.lower()
        assert "do not" in lowered or "not use" in lowered, f"{name} states no non-trigger"


def test_no_builtin_ships_an_executable_helper():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    assert all(not ref.executable_files for ref in reg.refs.values())


def test_audit_and_creation_skills_stay_distinct():
    """frontend-design creates; web-design-guidelines inspects."""
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    create = reg.refs["frontend-design"].description.lower()
    audit = reg.refs["web-design-guidelines"].description.lower()
    assert "reviewing" in audit or "auditing" in audit
    assert "does not" in audit
    assert "redesign" in create


def test_figma_skill_is_design_to_code_only():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    text = (reg.refs["figma-implement-design"].description + reg.refs["figma-implement-design"].body).lower()
    assert "design-to-code" in text
    assert "does not write to figma" in text or "not write to figma" in text


def test_webapp_testing_does_not_pretend_to_have_browser_tooling():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    body = reg.refs["webapp-testing"].body.lower()
    assert "no browser tooling" in body or "not available" in body
    assert "do not claim" in body or "do not describe" in body


def test_skill_creator_teaches_evaluation_not_just_authoring():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    body = reg.refs["skill-creator"].body.lower()
    assert "negative" in body
    assert "baseline" in body


def test_tdd_skill_does_not_force_tdd_everywhere():
    reg = SkillRegistry(sources={"builtin": builtin_skills_dir()})
    reg.load()
    body = reg.refs["test-driven-development"].body.lower()
    assert "wrong tool" in body or "exploratory" in body
