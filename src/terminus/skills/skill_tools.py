from __future__ import annotations

from pathlib import Path
from langchain.tools import tool

from terminus.config import CONFIG
from terminus.skills.registry import (
    MAX_REFERENCES_PER_SKILL,
    MAX_SKILLS_PER_TASK,
    SkillNotFoundError,
    SkillRef,
    SkillRegistry,
)
from terminus.cache import get_cached_prompt, cache_prompt
from terminus.observability.logging import get_logger
from terminus.workspace import project_key

logger = get_logger(__name__)


_registry: SkillRegistry | None = None
_registry_project: str | None = None
_SKILLS_PROMPT_CACHE_KEY = "skills_prompt"


def _get_registry() -> SkillRegistry:
    """The skills catalogue for the current project, parsed once per project.

    Three roots are searched: the project's own ``.terminus/skills``, the user's
    personal ``~/.terminus/skills``, and the curated library that ships with
    Terminus. Project skills stay project-scoped - the registry is rebuilt when
    the working project changes, so one repository's conventions can never leak
    into another. User and built-in skills are global by design and are not
    project data.

    ``skills_dir`` is a relative path, so it is resolved against the project
    root here. The registry is rebuilt if the project changes, otherwise a
    process that moved between projects would keep serving the first project's
    skills.
    """
    global _registry, _registry_project
    project = project_key()
    if _registry is None or _registry_project != project:
        configured = Path(CONFIG.get("skills", {}).get("skills_dir", ".terminus/skills"))
        if not configured.is_absolute():
            configured = Path(project) / configured
        registry = SkillRegistry(configured)
        if CONFIG.get("skills", {}).get("user_skills_dir"):
            user_dir = Path(CONFIG["skills"]["user_skills_dir"])
            if not user_dir.is_absolute():
                user_dir = Path.home() / user_dir
            registry.sources["user"] = user_dir
        registry.load()
        if registry.load_errors:
            # Malformed skills are skipped, never fatal, but they should be
            # visible: a user who installed a broken skill needs to know.
            logger.warning(
                "Skipped %d malformed skill file(s): %s",
                len(registry.load_errors),
                ", ".join(sorted(registry.load_errors)[:3]),
            )
        _registry = registry
        _registry_project = project
    return _registry


def build_skills_prompt() -> str:
    """Return the metadata-only catalogue appended to the agent system prompt.

    Called from agent/factory.build_agent so the model knows which skills exist
    without paying for their bodies. The registry parses every SKILL.md under the
    skills directory, and the result is cached per project: two projects in one
    process must not see each other's skills.
    """
    key = f"{_SKILLS_PROMPT_CACHE_KEY}:{project_key()}"
    cached = get_cached_prompt(key)
    if cached is not None:
        return cached
    return cache_prompt(key, _get_registry().build_skills_prompt())


def describe_skills(name: str | None = None) -> str:
    """Human-facing inventory, for the /skills and /skill commands.

    Reports provenance and what a skill ships, because a user deciding whether
    to trust a skill needs more than its name. Executable helpers are surfaced
    explicitly and are always described as not-run.
    """
    registry = _get_registry()
    refs = registry.refs
    if not refs:
        return "No skills are installed.\n\nAdd one by creating a directory with a SKILL.md:\n  <project>/.terminus/skills/<name>/SKILL.md"

    def render(ref: SkillRef) -> str:
        lines = [f"name:        {ref.name}", f"source:      {ref.provenance}"]
        if ref.version:
            lines.append(f"version:     {ref.version}")
        if ref.license:
            lines.append(f"license:     {ref.license}")
        if ref.tags:
            lines.append(f"tags:        {', '.join(ref.tags)}")
        lines.append(f"trusted:     {'built-in' if ref.is_trusted else 'user-supplied (untrusted content)'}")
        lines.append(f"description: {ref.description or '(none)'}")
        if ref.meta.get("when_to_use"):
            lines.append(f"triggers:    {ref.meta['when_to_use']}")
        lines.append(f"body:        {len(ref.body)} chars")
        if ref.support_files:
            lines.append(f"references:  {', '.join(ref.support_files[:MAX_REFERENCES_PER_SKILL])}")
        if ref.executable_files:
            lines.append(
                f"executables: {', '.join(ref.executable_files[:MAX_REFERENCES_PER_SKILL])} "
                "(not run automatically)"
            )
        if ref.root.name != ref.name:
            lines.append(f"note:        directory is '{ref.root.name}' but name is '{ref.name}'")
        shadowed = registry.shadowed.get(ref.name)
        if shadowed:
            lines.append(
                f"shadows:     {', '.join(f'{s.source}:{s.root.name}' for s in shadowed)}"
            )
        return "\n".join(lines)

    if name:
        ref = refs.get(name)
        if ref is None:
            return f"Skill not found: {name}\n\nInstalled: {', '.join(sorted(refs))}"
        return render(ref)

    lines = [f"Installed skills ({len(refs)}):"]
    for ref in sorted(refs.values(), key=lambda r: (r.source != "builtin", r.name)):
        marker = "*" if ref.is_trusted else " "
        suffix = " [shadowed]" if ref.name in registry.shadowed else ""
        lines.append(f" {marker} {ref.name} ({ref.source}){suffix}")
        if ref.description:
            lines.append(f"     {ref.description.splitlines()[0][:120]}")
    lines.append("")
    lines.append("  * built-in (ships with Terminus)")
    lines.append(f"Use /skill <name> for full detail. {MAX_SKILLS_PER_TASK} skills load at most per execution.")
    if registry.load_errors:
        lines.append("")
        lines.append(f"Skipped {len(registry.load_errors)} malformed skill(s).")
    return "\n".join(lines)


@tool
def load_skill(name: str) -> str:
    """
    Load the full instructions for a skill by name.
    Call this when the user's request matches one of the skills listed in your system prompt.
    Returns the skill's step-by-step instructions and lists any available support files.
    (scripts, templates, resources) that you can read using the read_file tool.
    """
    logger.info(f"Loading skill {name}")
    # A wrong skill name is an expected operational mistake, not a bug: return
    # the list of valid names so the model can correct itself, instead of
    # raising and ending the whole agent turn.
    try:
        return _get_registry().load_skill(name)
    except SkillNotFoundError as exc:
        logger.warning(f"Skill {name} not found")
        return (
            f"Skill not found: {name}. {exc} "
            "If this project defines no skills, carry on without one."
        )


