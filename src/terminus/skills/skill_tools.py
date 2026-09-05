from __future__ import annotations

from pathlib import Path
from langchain.tools import tool

from terminus.config import CONFIG
from terminus.skills.registry import SkillRegistry, SkillNotFoundError
from terminus.observability.logging import get_logger

logger = get_logger(__name__)



_registry: SkillRegistry | None = None

def _get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        skills_config = CONFIG.get("skills", {})
        skills_path = skills_config.get("path", skills_config.get("skills_dir", ".terminus/skills"))
        _registry = SkillRegistry(Path(skills_path))
        _registry.load()
    return _registry 



def build_skills_prompt() -> str:
    """
    Trigger registry initialisation and return the metadata-only prompt snippet.

    This is called once inside build_agent() in factory.py. and its return value
    is appended to SYSTEM_PROMPT before the agent is created. The LLM therefore always
    knows what skills exist, without paying the token cost of their bodies.
    Flow :
       factory.py:build_agent()
        - build_skills_prompt()
           - _get_registry()
              - SkillRegistry.load() → finds and parses skill.json files in skills_directory.
            - SkillRegistry.build_skills_prompt()
                returns compact metadata-only string like:
                skill_name_1: skill_description_1
                skill_name_2: skill_description_2
                ... appended to SYSTEM_PROMPT in order to make them discoverable by the LLM
    """
    return _get_registry().build_skills_prompt()
    

@tool
def load_skill(name : str)->str:
    """
    Load the sfull instructions for a skill by name.
    Call this when the user's request matches one of the skills listed in your system prompt.
    Returns the skill's steps-by-step instructions andlists any availbale support files.
    (scripts, templates, resources) that you can read using read_file tool.
    """ 
    logger.info(f"Loading skill {name}")
    try:
        return _get_registry().load_skill(name)
    except SkillNotFoundError:
        logger.error(f"Skill {name} not found")
        raise
   
   