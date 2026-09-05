from __future__ import annotations
import re
from pathlib import Path

from terminus.observability.logging import get_logger
logger = get_logger(__name__)


SKILL_FILENAME="SKILL.md"

class SkillNotFoundError(Exception):
    """Raise when a skill is not found"""
    pass

class SkillRegistry:
    """ 
    Owns the in-memory catalog of all skills found in skills_dir.
    
    Expected layout on disk:
    skills_dir/
        python_debug/
            SKILL.md - required -- frontmatter + instructions
            scripts/ - optional - executable helpers
            templates/ - optional -- output templates
            resources/ - optional -- reference docs
        write_tests/
            SKILL.md
            templates/
                pytest_template.py


    After calling load(), the internal _skills dict looks like:
      {
        "python_debug":{
            "meta": {"name":"python_debug","description":"...",....}
            "body": "You are an expert Python debugger. Follow these steps...",
            "skills_dir":Path(".terminus/skills/python_debug") 
            },
            "write_tests":{           
                "meta": {"name":"write_tests","description":"...",....},
                "body": "Generate pytest unit tests...",
                "skills_dir":Path(".terminus/skills/write_tests") 
                }
      
      }

      "meta" is parsed from the YAML frontmatter block (name, description, when_to_use ,).
      "body" is everything after the frontmatter - the actual instructions sent to the LLM.
      "skill_dir" is kept so we can later discover support files inside that folder.
    """
    def __init__(self,skills_dir:Path = Path(".terminus/skills"))->None:
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}


    def load(self)->None:
        """
        Load skills from the skills directory
        """
        self.skills.clear()
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory {self.skills_dir} does not exist")
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / SKILL_FILENAME
            if not skill_file.exists():
                continue

            try:
                meta, body = self._parse_skill_markdown(skill_file)
                name = meta.get("name", skill_dir.name)
                if not meta.get("when_to_use"):
                    logger.warning(f"Skill {name} has no when_to_use field")
                self.skills[name] = {
                    "meta": meta,
                    "body": body,
                    "skills_dir": skill_dir
                }
                logger.info(f"Loaded skill: {name}")
            except Exception as e:
                logger.error(f"Failed to load skill {skill_dir.name}: {e}")

    def build_skills_prompt(self)->str:
        """
        Returns a compact string appended to the agent's system prompt at startup.
        This string lists all available skills and when to use them.
        
         """
        if not self.skills:
             return ""

        lines = ["=== Available Skills ===\n"]
        for name, skill in sorted(self.skills.items()):
            meta = skill["meta"]
            desc = meta.get("description", "No description")
            when = meta.get("when_to_use")  
            lines.append(f"Skill: {name}")
            lines.append(f"Description: {desc}")
            if when:
                lines.append(f"When to use: {when}")
            lines.append("")

        lines.append(
            "\n When the user's request matches a skill, call load_skill(name) to load the skill"
        )
        lines.append(
            "\n Always call get_all_skills() at the start to refresh the list"
        )
        return "\n".join(lines)


    def load_skill(self,name:str)->dict[str,str]:
        """Load a skill by name. Returns the full SKILL.md body for the named skill, plus a listing of any support files inside the skill's folder.
            The agent calls this (via the load_skill tool in skill_tools.py) when the user's request matches a skill. THe returned context becomes
            part of the prompt for the *next* LLM call (the one that actually performs the task).
            Example return value:

                You are an expert Python debugger.Follow these steps...
                1. Ask for the full traceback if not provided.
                ..
                ---Support Files---
                /path/to/skills/python_debug/scripts/run_debugger.py
                /path/to/skills/python_debug/templates/error_report.txt
                

            The agent then uses this information, along with the original user request,
            to generate the final response.

        """
        if name not in self.skills:
            available_skills = ", ".join(self.skills.keys())
            raise SkillNotFoundError(f"Skill {name} not found. Available skills: {available_skills}")

        skill = self.skills.get(name)
        body = skill.get("body", "No body found")
        skill_dir: Path = skill.get("skills_dir")

        if not skill_dir or not skill_dir.exists():
            return body

        support_files = self._list_support_files(skill_dir)
        result = body
        if support_files:
            result += "\n\n---Support Files---\n" 
            result += "\n".join(f" {p}" for p in support_files)
            result += (
                "\n You can read any of these files using the read_file tool"
                "if the skill instruction reference them."
            )
        return result

    @property
    def skill_names(self):
        return list(self.skills.keys())





    def _parse_skill_markdown(self,skill_file:Path)->tuple[dict,str]:
        """
        Parse a skill markdown file and return the metadata and body
        """
        try:
            content = skill_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        except Exception as e:
            raise ValueError(f"Failed to read {skill_file}: {e}")

        if "---\n" not in content:
            raise ValueError(f"No YAML frontmatter found in {skill_file}")

        _, _, rest = content.partition("---\n")
        yaml_block, _, body = rest.partition("---\n")
        import yaml

        meta = yaml.safe_load(yaml_block)
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid YAML frontmatter in {skill_file}")

        if "name" not in meta:
            meta["name"] = skill_file.parent.name

        return meta, body.strip()   

    def _list_support_files(self,skill_dir:Path)->list[Path]:
        """
        List all support files in the skill directory
        """
        support_files = []
        for item in skill_dir.rglob("*"):
            if item.is_file() and item != skill_dir / SKILL_FILENAME:
                support_files.append(item)
        return support_files    

    def get_all_skills(self)->list[dict]:
        """Get all skills"""
        return list(self.skills.values())
    
                



    
        