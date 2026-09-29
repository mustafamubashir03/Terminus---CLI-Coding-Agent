from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from terminus.observability.logging import get_logger
logger = get_logger(__name__)


SKILL_FILENAME="SKILL.md"

# --- context budget -------------------------------------------------------
# A skill is knowledge loaded into a prompt, so every dimension of it is
# bounded. These are deliberately explicit constants rather than tuning
# parameters: a skill system that can grow the prompt without limit is a
# context-management bug waiting to happen.
MAX_SKILL_CHARS = 8000
"""Characters of SKILL.md body that may be injected for a single skill."""

MAX_SKILLS_TOTAL_CHARS = 6000
"""Characters for the whole selected-skills block, however many skills are in it.

A per-skill cap alone is not a budget: three skills at 5,000 characters each is
15,000 characters of prompt growth, which is exactly the failure a skills system
is supposed to avoid. This is the ceiling on the block, and skills share it in
proportion to their size, so a second selection cannot quietly double the cost.
"""

MAX_SKILLS_PER_TASK = 3
"""How many skill bodies may be loaded into one execution at once."""

MAX_REFERENCES_PER_SKILL = 12
"""Support files listed alongside a skill before the listing itself is cut."""

MAX_CATALOGUE_ENTRIES = 60
"""Upper bound on skills described in the system-prompt catalogue.

The catalogue is metadata only, so this can be generous; it exists so a user who
installs hundreds of skills does not grow every single prompt without limit.
"""

SKILL_SOURCES = ("project", "user", "builtin")
"""Search order, lowest precedence last.

project > user > builtin. A repository's own conventions outrank a user's
personal default, which outranks what Terminus ships. None of them outrank
system safety, which is enforced by the permission layer rather than here.
"""

# Lower number = visited first = wins a duplicate name. Deliberately the same
# order as SKILL_SOURCES, which is written highest-precedence-first.
_SOURCE_PRECEDENCE = {name: index for index, name in enumerate(SKILL_SOURCES)}


class SkillNotFoundError(Exception):
    """Raise when a skill is not found"""


@dataclass(frozen=True)
class SkillRef:
    """One discovered skill, with where it came from and what it contains.

    ``body`` is the instruction text and is only read when a skill is actually
    selected, so discovery stays cheap. ``executable_files`` is populated for
    inspection only - nothing here ever runs them.
    """

    name: str
    description: str
    body: str
    root: Path
    source: str
    meta: dict = field(default_factory=dict)
    support_files: list[str] = field(default_factory=list)
    executable_files: list[str] = field(default_factory=list)
    load_error: str = ""

    @property
    def version(self) -> str:
        return str(self.meta.get("version") or "")

    @property
    def license(self) -> str:
        return str(self.meta.get("license") or "")

    @property
    def tags(self) -> list[str]:
        """Tags as a list, however the author wrote them.

        Accepts a YAML list, a comma-separated string, and the string
        repr of a list - which is what an inline flow sequence can degrade to
        depending on the quoting. Normalising here means a matcher never has to
        care how the frontmatter was written.
        """
        raw = self.meta.get("tags") or []
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, list):
                    raw = parsed
            if isinstance(raw, str):
                raw = [part.strip(" []'\"") for part in raw.split(",")]
        return [str(t).strip() for t in raw if str(t).strip()]

    @property
    def provenance(self) -> str:
        repository = self.meta.get("repository") or self.meta.get("source_repository")
        base = f"{self.source}:{self.root.name}"
        return f"{base} ({repository})" if repository else base

    @property
    def is_trusted(self) -> bool:
        """Built-ins ship with Terminus; anything else is user-supplied content.

        Trust here never grants execution. It only decides whether a skill may be
        auto-selected without the user asking, and whether a mismatch in a
        support file is reported loudly.
        """
        return self.source == "builtin"

    def truncated_body(self, limit: int = MAX_SKILL_CHARS) -> str:
        """The body, bounded, without cutting mid-sentence where avoidable."""
        text = self.body or ""
        if len(text) <= limit:
            return text
        head = text[:limit].rstrip()
        return (
            f"{head}\n\n[... {len(text) - limit} more characters of this skill were "
            f"omitted to stay within the {limit}-character skill budget. Read "
            f"{self.root / SKILL_FILENAME} directly for the full text.]"
        )


def builtin_skills_dir() -> Path:
    """Where the curated library that ships with Terminus lives.

    A package directory next to this module rather than anything under the
    user's repository, so the built-ins are always present and always
    identifiable as built-ins.
    """
    return Path(__file__).resolve().parent / "builtin"


def user_skills_dir() -> Path:
    """Where a user installs their own skills, independent of any project.

    Deliberately outside the project root: user skills are a personal default
    that should be available in every project, and must not become part of a
    repository by accident.
    """
    return Path.home() / ".terminus" / "skills"


def _is_under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


class SkillRegistry:
    """
    The in-memory catalog of skills found under the configured roots.

    On-disk layout, one directory per skill::

        <root>/<name>/SKILL.md     required - YAML frontmatter + body
        <root>/<name>/references/  optional - supporting documentation
        <root>/<name>/scripts/     optional - executable helpers (never auto-run)
        <root>/<name>/assets/      optional - templates and fixtures

    Three roots are searched, in decreasing precedence::

        project   <project_root>/.terminus/skills
        user      ~/.terminus/skills
        builtin   terminus/skills/builtin  (ships with Terminus)

    ``self.skills`` maps the skill's frontmatter ``name`` to::

        {"meta": {...frontmatter...}, "body": "instructions", "skills_dir": Path}

    and ``self.refs`` holds the same skills as :class:`SkillRef` values with
    provenance attached. A name defined in more than one root resolves to the
    highest-precedence definition, and the shadowed copies stay reachable
    through :meth:`shadowed` so a user can see what they overrode.
    """
    def __init__(self,skills_dir:Path = Path(".terminus/skills"), *,
                 sources: dict[str,Path] | None = None) -> None:
        self.skills_dir = skills_dir
        self.skills: dict[str, dict] = {}
        self.refs: dict[str, SkillRef] = {}
        self.shadowed: dict[str, list[SkillRef]] = {}
        self.load_errors: dict[str, str] = {}
        # Explicit roots win; otherwise project + user + built-in are searched.
        self.sources: dict[str, Path] = dict(sources) if sources else {
            "project": Path(skills_dir),
            "user": user_skills_dir(),
            "builtin": builtin_skills_dir(),
        }


    def load(self)->None:
        """
        Discover skills across every configured root.

        Roots are visited in increasing precedence and a name is claimed by the
        first root that defines it, so the highest-precedence copy wins without
        any content comparison: if a user shadows a built-in, that is a
        deliberate act on their part. The displaced copy is recorded rather than
        discarded so it can be reported.
        """
        self.skills.clear()
        self.refs.clear()
        self.shadowed.clear()
        self.load_errors.clear()

        for source in sorted(self.sources, key=lambda s: _SOURCE_PRECEDENCE.get(s, 0)):
            root = self.sources[source]
            if not Path(root).exists():
                logger.debug("Skill source %s has no directory at %s", source, root)
                continue
            for skill_dir in sorted(Path(root).iterdir(), key=lambda p: p.name):
                if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                    continue
                ref = self._read_skill(skill_dir, source)
                if ref is None:
                    continue
                if ref.name in self.refs:
                    # A lower-precedence root already claimed this name.
                    self.shadowed.setdefault(ref.name, []).append(ref)
                    logger.info(
                        "Skill %s from %s is shadowed by %s (%s wins)",
                        ref.name, source, self.refs[ref.name].source, ref.source,
                    )
                    continue
                self.refs[ref.name] = ref

        for ref in self.refs.values():
            self.skills[ref.name] = {
                "meta": dict(ref.meta),
                "body": ref.body,
                "skills_dir": ref.root,
                "source": ref.source,
                "provenance": ref.provenance,
            }

    def match(self, request: str, **signals) -> list:
        """Rank installed skills against a request. See :mod:`terminus.skills.matcher`."""
        from terminus.skills.matcher import match_skills

        return match_skills(self, request, **signals)

    def _read_skill(self, skill_dir: Path, source: str) -> SkillRef | None:
        """Parse one skill directory, or record why it was skipped.

        A malformed skill must never take the catalogue down: the registry is
        read on every prompt, so one bad file has to be a warning, not an
        exception.
        """
        skill_file = skill_dir / SKILL_FILENAME
        if not skill_file.exists():
            return None
        try:
            meta, body = self._parse_skill_markdown(skill_file)
        except Exception as exc:
            logger.warning("Skipping malformed skill %s: %s", skill_dir, exc)
            self.load_errors[str(skill_dir)] = f"{type(exc).__name__}: {exc}"
            return None

        name = str(meta.get("name") or skill_dir.name).strip()
        if not name:
            logger.warning("Skipping skill %s: empty name", skill_dir)
            self.load_errors[str(skill_dir)] = "empty name"
            return None

        support, executable = self._classify_support_files(skill_dir)
        return SkillRef(
            name=name,
            description=str(meta.get("description") or "").strip(),
            body=body,
            root=skill_dir,
            source=source,
            meta=meta,
            support_files=support,
            executable_files=executable,
        )

    def _classify_support_files(self, skill_dir: Path) -> tuple[list[str], list[str]]:
        """Split a skill's extra files into inert references and executables.

        Classified, never run. A skill that ships a script is telling the model
        a shell command may help, but running it stays a normal tool call under
        the normal permission policy - installing a skill must not be the same
        thing as executing it.
        """
        support: list[str] = []
        executable: list[str] = []
        script_dirs = {"scripts", "bin", "tools"}
        script_suffixes = {".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".exe"}
        for item in sorted(skill_dir.rglob("*")):
            if not item.is_file() or item.name == SKILL_FILENAME:
                continue
            rel = item.relative_to(skill_dir).as_posix()
            parts = rel.split("/")
            if parts[0] in script_dirs or item.suffix.lower() in script_suffixes:
                executable.append(rel)
            else:
                support.append(rel)
        return support, executable

    def build_skills_prompt(self)->str:
        """
        Returns a compact string appended to the agent's system prompt at startup.
        This string lists all available skills and when to use them.

        Metadata only, and bounded twice over: the catalogue is capped at
        :data:`MAX_CATALOGUE_ENTRIES` entries and each description is itself
        clipped. A user who installs a hundred skills still gets a small,
        predictable prompt; the full text of a skill is only ever paid for when
        that skill is actually selected.
        """
        if not self.skills:
             return ""

        lines = ["=== Available Skills ===\n"]
        names = sorted(self.skills)[:MAX_CATALOGUE_ENTRIES]
        for name in names:
            skill = self.skills[name]
            meta = skill["meta"]
            desc = str(meta.get("description", "No description")).strip()
            if not desc:
                desc = "(no description; do not select automatically)"
            desc = desc if len(desc) <= 300 else desc[:297].rstrip() + "..."
            when = meta.get("when_to_use")
            lines.append(f"Skill: {name}")
            lines.append(f"Description: {desc}")
            if when:
                lines.append(f"When to use: {str(when)[:300]}")
            if skill.get("source") and skill.get("source") != "project":
                lines.append(f"Source: {skill['source']}")
            lines.append("")

        if len(self.skills) > len(names):
            lines.append(
                f"(+{len(self.skills) - len(names)} more skills installed; "
                f"use the skills command to list them)"
            )
            lines.append("")

        lines.append(
            "\n When the user's request matches a skill, call load_skill(name) to load the skill"
        )
        return "\n".join(lines)


    def load_skill(self,name:str)->str:
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

        The body is clipped to :data:`MAX_SKILL_CHARS` and the file listing to
        :data:`MAX_REFERENCES_PER_SKILL`, so a skill cannot expand a prompt
        without limit. Files under a script directory are listed separately and
        labelled: they are reported, never run.
        """
        ref = self.refs.get(name)
        if ref is None:
            available_skills = ", ".join(sorted(self.skills))
            raise SkillNotFoundError(f"Skill {name} not found. Available skills: {available_skills}")

        result = ref.truncated_body()
        header = f"# Skill: {name} (source: {ref.source})"
        result = f"{header}\n{result}"

        if ref.support_files:
            shown = ref.support_files[:MAX_REFERENCES_PER_SKILL]
            extra = len(ref.support_files) - len(shown)
            result += "\n\n---Reference Files---\n" + "\n".join(f" {p}" for p in shown)
            if extra > 0:
                result += f"\n (+{extra} more reference file(s) not listed)"
            result += (
                "\n Read any of these with the read_file tool if the instructions "
                "reference them."
            )
        if ref.executable_files:
            shown = ref.executable_files[:MAX_REFERENCES_PER_SKILL]
            result += (
                "\n\n---Executable Helpers (NOT run automatically)---\n"
                + "\n".join(f" {p}" for p in shown)
                + "\n These are listed for your information. Running one is a normal "
                "shell tool call and still requires the usual permission approval; "
                "loading a skill never runs it for you."
            )
        return result

    def _parse_skill_markdown(self,skill_file:Path)->tuple[dict,str]:
        """
        Parse a skill markdown file and return the metadata and body
        """
        try:
            content = skill_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        except Exception as e:
            # `from e` so the underlying OSError stays in the traceback; the
            # message alone would lose why the read failed.
            raise ValueError(f"Failed to read {skill_file}: {e}") from e

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
