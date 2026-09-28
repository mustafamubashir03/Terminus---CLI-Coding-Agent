from __future__ import annotations

from pathlib import Path
from langchain.tools import tool

from terminus.config import CONFIG
from terminus.skills.registry import SkillRegistry, SkillNotFoundError
from terminus.cache import get_cached_prompt, cache_prompt
from terminus.observability.logging import get_logger
from terminus.workspace import project_key

logger = get_logger(__name__)


_registry: SkillRegistry | None = None
_registry_project: str | None = None
_SKILLS_PROMPT_CACHE_KEY = "skills_prompt"


def _get_registry() -> SkillRegistry:
    """The skills catalogue for the current project, parsed once per project.

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
        _registry = SkillRegistry(configured)
        _registry.load()
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
   
   
