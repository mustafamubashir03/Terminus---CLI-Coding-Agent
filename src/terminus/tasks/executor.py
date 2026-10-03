"""Machinery for running one task's worker agent and judging its output.

The lifecycle itself - validate the workspace, apply this execution's
permissions, run, verify, decide retry - belongs to
:func:`terminus.tasks.worker.execute_task`. This module owns the parts: which
tools a task type gets, what a worker is told, how its stream is bounded, and
how its output is verified.
"""

import asyncio
import json
import re
from pathlib import Path

from pydantic import BaseModel
from terminus.observability.logging import get_logger
from terminus.llm.factory import get_chat_model
from terminus.llm.text import message_text
from terminus.skills.skill_tools import build_skills_prompt
from terminus.skills.matcher import detect_conflicts
from terminus.skills.registry import MAX_SKILLS_PER_TASK, MAX_SKILLS_TOTAL_CHARS
from terminus.project_context import plan_fields
from terminus.mcp.terminus_mcp_client import get_terminus_mcp_tools
from terminus.agents_md import agents_md_section, load_agents_md
from terminus.tools import registry
from terminus.context.environment import build_startup_context
from terminus.agent.factory import AgentPolicy, build_agent, versioning_section
from terminus.workspace import project_root

from terminus.config import CONFIG
from terminus.cache import get_cached_prompt, cache_prompt, retry_async
from terminus.observability.usage_tracker import (
    ToolCallbackHandler,
    UsageCallbackHandler,
    record,
)

# Bounds for external operations. Every long-running call below is wrapped so a
# hung provider/tool/agent cannot leave a task permanently IN_PROGRESS.
_AGENT_STREAM_TIMEOUT_SECONDS = CONFIG.get("tasks", {}).get("agent_timeout_seconds", 900)
_JUDGE_TIMEOUT_SECONDS = CONFIG.get("tasks", {}).get("judge_timeout_seconds", 300)
_MCP_TOOLS_TIMEOUT_SECONDS = CONFIG.get("tasks", {}).get("mcp_timeout_seconds", 120)
_RATE_LIMIT_BACKOFF_SECONDS = CONFIG.get("tasks", {}).get("rate_limit_backoff_seconds", 30)

logger = get_logger(__name__)

_JUDGE_SYSTEM_PROMPT_CACHE_KEY = "judge_system_prompt"

# Static system prompt for the LLM-as-a-judge. Cached because it is token-heavy
# and identical across every judge invocation within a process.
JUDGE_SYSTEM_PROMPT = (
    "You are a stringent judge evaluating whether a task's output meets all its acceptance criteria.\n"
    "A task may primarily produce files and code, in which case its textual output can be short; "
    "you must NOT reject an output for being short or terse. Judge only whether the acceptance "
    "criteria are actually satisfied by the output."
)

_WORKER_MODEL_CALLS = 12
"""Model calls one task attempt may make before the run ends."""

_DEFAULT_TOOL_NAMES = ("search_codebase",)

# Code repository tools are read-only and are intentionally excluded (they are
# suffix-aliased by GitHub's MCP ""_get_content" tool and slow down every
# task with unused network I/O).
_GITHUB_TOOL_SUFFIX = "_get_content"


def _sanitize_tool_name(name: str) -> str:
    """Replace spaces and special chars with underscores for API compatibility.

    Some providers (Nvidia/OpenRouter) reject tool names containing spaces or
    non-alphanumeric characters. This ensures all tool names are safe.
    """
    return name.replace(" ", "_").replace("-", "_")


def _dedupe(tools: list) -> list:
    """Drop tools whose names collide, keeping the first occurrence, and warn."""
    seen: set[str] = set()
    result = []
    for t in tools:
        raw_name = getattr(t, "name", None) or str(t)
        sanitized = _sanitize_tool_name(raw_name)
        if raw_name != sanitized:
            t.name = sanitized
        if sanitized in seen:
            logger.warning(
                f"Duplicate tool name '{sanitized}' in task tools; keeping first. "
                f"An MCP tool may be shadowing a built-in."
            )
            continue
        seen.add(sanitized)
        result.append(t)
    return result


def _filter_github_code_read_tools(tools: list) -> list:
    """Drop GitHub MCP ''*_get_content'' tools which invoke heavy network I/O.

    Every other agent model passes text directly; only the code-read tools are
    removed to keep the tool list lean for cheaper / weaker models.
    """
    return [t for t in tools if not getattr(t, "name", "").endswith(_GITHUB_TOOL_SUFFIX)]


_CACHED_TOOLS: dict | None = None


async def _tool_plans() -> dict:
    """Build and cache the per-task-type tool lists for the process.

    Which Terminus tools each task type gets is declared as *names* in
    ``terminus.tools.registry``, the same catalogue /ask resolves through, so a
    tool cannot exist for one surface and be silently missing from the other.
    Only the MCP tools are genuinely dynamic, and they are appended after the
    named ones.
    """
    global _CACHED_TOOLS
    if _CACHED_TOOLS is not None:
        return _CACHED_TOOLS
    mcp_tools = await asyncio.wait_for(
        get_terminus_mcp_tools(), timeout=_MCP_TOOLS_TIMEOUT_SECONDS
    )
    mcp_tools = _filter_github_code_read_tools(mcp_tools)

    _CACHED_TOOLS = {
        task_type: _dedupe([
            *registry.resolve(
                (*_DEFAULT_TOOL_NAMES, *names), extra=mcp_tools
            )
        ])
        for task_type, names in registry.PLAN_TOOL_NAMES.items()
    }
    return _CACHED_TOOLS

def _as_str_list(value) -> list[str]:
    """Normalise a stored list field to a list of strings.

    ``output_files`` and ``acceptance_criteria`` are persisted as JSON, but rows
    written by older versions (or by hand) hold either a JSON array or a
    comma-separated string. Every reader goes through here so the tolerance for
    both shapes is defined once.
    """
    if not value:
        return []
    if isinstance(value, str):
        if value.strip().startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return [str(item) for item in parsed] if isinstance(parsed, list) else []
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _load_plan(project_id: str | None) -> dict | None:
    """Read this task's plan-level context, or None.

    A worker is told which task it is running, but a task title is meaningless
    without the project goal it serves. This is the read-only lookup that closes
    that gap; it never writes, and any failure degrades to "no plan context"
    rather than blocking execution. A worker only ever runs inside an existing
    project, so a missing database here is already an error elsewhere.
    """
    if not project_id:
        return None
    try:
        from terminus.config import CONFIG
        from terminus.tasks.task_store import TaskStore

        db_path = CONFIG.get("tasks", {}).get("db_path", ".terminus/tasks/tasks.db")
        return TaskStore(db_path).get_project(project_id)
    except Exception:
        return None


def _build_system_prompt(
    task: dict, dep_outputs: list[dict], plan: dict | None = None, tools: list | None = None
) -> str:
    prompt = "You are tasked with executing the following subtask:\n"
    prompt += f"Task Type: {task.get('task_type')}\n"
    prompt += f"Description: {task.get('description')}\n"

    # Identity, not history. A worker is a fresh agent with no conversation, so
    # it must be told which project, workspace and task it is acting for. This is
    # the minimum context contract; the /ask conversation and the planner's
    # scratch conversation stay isolated on purpose.
    prompt += f"Project ID: {task.get('project_id')}\n"
    prompt += f"Task ID: {task.get('id')}\n"
    prompt += f"Workspace (your working directory, the only tree you may edit): {project_root()}\n"

    # The plan this task belongs to. Without it a worker optimises for its own
    # description and cannot tell whether it is serving the actual goal.
    if plan:
        fields = plan_fields(plan.get("plan_json"))
        goal = fields.get("goal_summary") or plan.get("goal") or ""
        if goal:
            prompt += f"\nProject goal you are working toward: {goal}\n"
        stack = [str(s) for s in (fields.get("tech_stack") or [])][:5]
        if stack:
            prompt += f"Project tech stack: {', '.join(stack)}\n"
        risks = [str(r) for r in (fields.get("risks") or [])][:5]
        if risks:
            prompt += f"Known project risks: {'; '.join(risks)}\n"

    # Acceptance criteria are the contract between this worker and the judge that
    # grades it. They were persisted and enforced, but never shown to the worker,
    # so the agent was graded on requirements it could not read. They must be
    # stated in the same words the judge will use.
    criteria = _as_str_list(task.get("acceptance_criteria"))
    if criteria:
        prompt += "\nYour work must satisfy every one of these acceptance criteria:\n"
        for criterion in criteria:
            prompt += f"- {criterion}\n"
        prompt += (
            "Make each one true in the files you produce, and state in your final "
            "summary how each is satisfied.\n"
        )

    output_files = _as_str_list(task.get("output_files"))
    if output_files:
        prompt += "\nThe files you MUST produce as your deliverables (write their exact contents):\n"
        for f in output_files:
            prompt += f"- {f}\n"
        prompt += (
            "\nUse the write_file tool to create these paths exactly as listed. "
            "After writing each file, include its content in your final text "
            "summary so it can be verified.\n"
        )

    if dep_outputs:
        prompt += "\nHere are the results from dependencies that you must use:\n"
        for dep in dep_outputs:
            prompt += f"- Dependency Task {dep.get('id')}:\n{dep.get('result')}\n"
    # The project own instructions. A worker writes files, so it is the agent
    # most bound by them; it was also the one agent that never saw them, because
    # it builds this prompt from scratch rather than reusing the /ask assembly.
    workspace_notes = build_startup_context(project_root())
    if workspace_notes:
        prompt += "\n" + workspace_notes

    # A /plan worker is the clearest case of the workspace being shared durable
    # state rather than scratch: it writes files that a later task, a reviewer or
    # a judge reads, and it reads files earlier tasks produced. One paragraph, and
    # only the facts - every path it can reach is already enforced by
    # terminus.workspace rather than by this sentence.
    prompt += (
        "\nPaths you pass to file tools are relative to that workspace, and a path "
        "outside it is rejected. The workspace is shared and durable: earlier tasks "
        "have already written into it, later tasks and the reviewer will read what "
        "you write, and your own findings belong in files if they need to outlive "
        "this attempt. Use '.terminus/notes/' for working notes rather than relying "
        "on anything you cannot point at.\n"
    )
    # Version state a worker can read but not change. The section is derived from
    # this worker's own tool list rather than written unconditionally, so a worker
    # that somehow lost the git tools is never told it has them.
    versioning = versioning_section(tools or ())
    if versioning:
        prompt += f"\n{versioning}\n"

    # Durable knowledge from earlier sessions and earlier tasks in this same
    # workspace. Read per task rather than cached: a worker that started from a
    # stale copy would carry an outdated convention into files it is about to
    # write, and the reviewer and judge read those files.
    memory = agents_md_section(load_agents_md())
    if memory:
        prompt += f"\n{memory}\n"

    prompt += "\nEnsure you output exactly what is required to complete this task."
    return prompt

def _worker_skills_section(tools: list) -> str:
    """The skills catalogue, only where the worker can actually act on it.

    Two things have to hold at once: the catalogue must be non-empty, and this
    worker's tool set must contain ``load_skill``. Task types "test",
    "integrate" and "configure" are built without it, so the old unconditional
    header advertised a tool those workers did not have, and advertised a
    catalogue that is empty by default anyway.
    """
    if not any(getattr(tool, "name", None) == "load_skill" for tool in tools):
        return ""
    catalogue = build_skills_prompt()
    if not catalogue or not catalogue.strip():
        return ""
    return f"\n\n======SKILLS======\n\n{catalogue}"


def _worker_selected_skills(task: dict) -> str:
    """Skills chosen for this specific task, injected as instructions.

    The catalogue above only names what exists; this is the bounded decision of
    which of them apply to *this* task, with the reason recorded, so the worker
    does not have to spend its budget deciding whether a skill is relevant. The
    task's own acceptance criteria and deliverables are fed to the matcher as
    well as its description, because those describe the work better than the
    title does.
    """
    try:
        from terminus.skills.matcher import match_skills, render_selection
        from terminus.skills.skill_tools import _get_registry

        registry = _get_registry()
        # The description and title describe the work. The task_type is excluded
        # on purpose: it is one of six fixed words, and as query text it matched
        # skill names - "implement" is in "figma-implement-design", so every
        # implementation task in any language pulled in the Figma skill.
        query = " ".join(str(task.get(field) or "") for field in ("description", "title"))
        criteria = _as_str_list(task.get("acceptance_criteria"))
        # Only genuine framework facts are passed as signals. The task_type is
        # deliberately not one of them: "implement" is a substring of
        # "figma-implement-design", so feeding it in selected the Figma skill for
        # any implementation task in any language.
        known_frameworks = ("react", "next.js", "nextjs", "vue", "svelte", "angular", "tailwind")
        signals = {
            "frameworks": [c for c in criteria if any(f in c.lower() for f in known_frameworks)],
        }
        matches = match_skills(registry, query, limit=MAX_SKILLS_PER_TASK, **signals)
        block = render_selection(matches, registry)
        if not block.strip():
            return ""
        conflicts = detect_conflicts(matches)
        if conflicts:
            # Never concatenate contradictory instructions silently. Say so and
            # let the project conventions in the prompt win, per precedence.
            block += (
                "\n\nNOTE: these skills overlap ("
                + "; ".join(conflicts)
                + "). Where they disagree, the project's own conventions and the "
                "task's acceptance criteria take precedence over any skill."
            )
        block = f"\n\n======SELECTED SKILLS======\n{block}"
        # The budget covers what is actually injected, header and conflict note
        # included, so the cap is real rather than approximately real.
        if len(block) > MAX_SKILLS_TOTAL_CHARS:
            block = (
                block[: MAX_SKILLS_TOTAL_CHARS - 60].rstrip()
                + "\n[skills block truncated]"
            )
        return block
    except Exception as exc:
        # Skills are optional context; never let selection fail a task.
        logger.debug("Skill selection skipped for task: %s", exc)
        return ""

class Verdict(BaseModel):
    passed: bool
    reason: str


def _verdict_from_text(text: str) -> Verdict | None:
    match = re.search(r"\b(true|false|passed|failed)\b", text, re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).strip().lower()
    prefix = text[max(0, match.start() - 5):match.start()].lower()
    passed = token in {"true", "passed"} and not prefix.endswith("not ")
    reason = (text[: match.start()] + " " + text[match.end():]).strip()
    reason = re.sub(r"\s+", " ", reason)[:300].strip()
    return Verdict(passed=passed, reason=reason or text[:300])


def _read_output_file_contents(file_paths: list) -> str:
    """Return the contents of the task's declared output files (if they exist).

    The judge verifies acceptance criteria against the *actual* deliverable
    files, not just the agent's final text summary (which can be terse and
    would otherwise cause false rejections).
    """
    import os

    files = []
    for fp in file_paths:
        if isinstance(fp, str) and fp.strip():
            files.append(fp.strip())
    if not files:
        return ""

    parts = []
    for fp in files:
        if os.path.isfile(fp):
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(8000)
                parts.append(f"--- Content of {fp} ---\n{content}\n")
            except Exception:
                parts.append(f"--- {fp} exists but could not be read ---\n")
        else:
            parts.append(f"--- {fp} DOES NOT EXIST ON DISK ---\n")
    return "\n".join(parts)


def _output_file_snapshot(file_paths: list[str]) -> dict[str, tuple]:
    snapshot = {}
    for file_path in file_paths:
        try:
            stat = Path(file_path).stat()
            snapshot[file_path] = (True, stat.st_mtime_ns, stat.st_size)
        except OSError:
            snapshot[file_path] = (False, 0, 0)
    return snapshot


def _deliverables_written_since(
    snapshot: dict[str, tuple], file_paths: list[str]
) -> bool:
    if not file_paths:
        return False
    current = _output_file_snapshot(file_paths)
    return all(current[path][0] for path in file_paths) and any(
        current[path] != snapshot.get(path) for path in file_paths
    )


def _deliverable_summary(file_paths: list[str]) -> str:
    contents = _read_output_file_contents(file_paths)
    if contents:
        return contents
    return "\n".join(f"Created deliverable: {path}" for path in file_paths)


async def judge_task(task: dict, output: str, output_file_contents: str = "") -> Verdict:
    """Verify a worker's output against the task's acceptance criteria.

    Split out from the agent run so tasks/worker.py owns the lifecycle
    (execute, then verify) while this module owns the machinery of each step.
    A rejection is returned as Verdict(passed=False), not raised, so the
    caller decides whether to retry.
    """
    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"].get("judge_model", CONFIG["llm"]["model"])
    # get_chat_model applies llm.request_timeout_seconds + llm.max_retries so a
    # hung judge request fails fast and transient errors retry (bounded).
    llm = get_chat_model(model, model_provider=provider)

    criteria = _as_str_list(task.get("acceptance_criteria"))
    if not criteria:
        raise ValueError(f"Task {task.get('id')} has no acceptance criteria")

    system_prompt = get_cached_prompt(_JUDGE_SYSTEM_PROMPT_CACHE_KEY)
    if system_prompt is None:
        system_prompt = cache_prompt(
            _JUDGE_SYSTEM_PROMPT_CACHE_KEY, JUDGE_SYSTEM_PROMPT
        )
    user_message = f"Task Description: {task.get('description')}\n\n"
    user_message += "Acceptance Criteria:\n"
    for c in criteria:
        user_message += f"- {c}\n"
    user_message += f"\nAgent Output Summary:\n{output}\n\n"
    if output_file_contents:
        user_message += f"\nActual deliverable file contents on disk:\n{output_file_contents}\n\n"
    user_message += (
        "Did the output pass all acceptance criteria? "
        "Verify against the actual file contents on disk when present. "
        "Call the Verdict tool exactly once with strict true/false and provide a reason."
    )

    judge_handler = UsageCallbackHandler(kind="judge")

    tool_choice = "auto" if provider.lower() == "openrouter" else "any"
    judge_llm = llm.bind_tools([Verdict], tool_choice=tool_choice)

    async def _run_judge():
        return await asyncio.wait_for(
            judge_llm.ainvoke(
                [
                    ("system", system_prompt),
                    ("human", user_message),
                ],
                config={"callbacks": [judge_handler]},
            ),
            timeout=_JUDGE_TIMEOUT_SECONDS,
        )

    try:
        response = await retry_async(
            _run_judge,
            max_retries=3,
            base_delay=_RATE_LIMIT_BACKOFF_SECONDS,
        )
    finally:
        record(judge_handler.records, "judge")

    tool_calls = getattr(response, "tool_calls", None) or []
    for tc in tool_calls:
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if isinstance(args, dict):
            try:
                return Verdict.model_validate(args)
            except Exception:
                continue

    # Some providers (Cohere) answer structured-outputs with plain text once the
    # input grows large, even with a forced tool choice. Fall back to reading
    # the verdict from the text so the judge is robust across providers.
    text = getattr(response, "content", "") or ""
    if isinstance(text, list):
        text = " ".join(s.get("text", "") if isinstance(s, dict) else str(s) for s in text)
    verdict = _verdict_from_text(text)
    if verdict is not None:
        return verdict

    raise ValueError(
        f"Judge returned no structured verdict for task {task['id']}: {text[:200]!r}"
    )

def _log_tool_activity(task_id, tools_seen) -> None:
    """Report a worker's tool activity: what it touched, and what failed.

    Read from the handler's records rather than from graph state, because the
    records exist even when the stream raised before a final state arrived.
    """
    changed = tools_seen.files_changed()
    if changed:
        logger.info(
            "Task %s touched workspace paths: %s", task_id, ", ".join(changed)
        )
    for failure in tools_seen.failures():
        logger.warning(
            "Task %s tool %s failed: %s", task_id, failure.name, failure.error
        )


async def _run_worker_agent(
    task: dict,
    dep_outputs: list[dict],
    feedback: str = "",
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """Build a fresh worker agent for one task and return (answer, deliverables).

    The second element is the on-disk content of the task's declared output
    files, so the judge can verify the real deliverable rather than the worker's
    own summary.

    No checkpointer and no thread id: a worker is a single bounded attempt, not
    a conversation. Its durable state is the task row and the result.

    ``provider``/``model`` default to the active runtime configuration, so the
    worker and /ask route through the same llm/factory stack and the same
    fallbacks. A caller may pass them explicitly to pin a route.
    """
    provider = provider or CONFIG["llm"]["provider"]
    model = model or CONFIG["llm"]["model"]
    task_id = task.get("id", "?")

    tool_map = await _tool_plans()
    tools = tool_map.get(
        task.get("task_type", ""), list(registry.resolve(_DEFAULT_TOOL_NAMES))
    )

    system_prompt = _build_system_prompt(
        task, dep_outputs, _load_plan(task.get("project_id")), tools
    )
    system_prompt += _worker_selected_skills(task)
    if feedback:
        system_prompt += f"\n\nPrevious attempt feedback:\n{feedback}"
    system_prompt += _worker_skills_section(tools)
    logger.info("Building worker agent for task %s: %s", task_id, task.get("description"))

    agent = await build_agent(AgentPolicy(
        tools=tuple(tools),
        system_prompt=system_prompt,
        model=model,
        provider=provider,
        model_call_limit=_WORKER_MODEL_CALLS,
        tool_call_limits=(("search_codebase", 4), ("write_file", 30)),
        tool_limit_behaviour="end",
        summarize=False,
        checkpoint=False,
    ))
    output_files = _as_str_list(task.get("output_files"))
    user_message = (
        f"{task.get('description')}\n\n"
        "The project directory may be empty - there is no existing code to read.\n"
        "You must CREATE all output files from scratch.\n"
        "Do not spend time listing directories. Go directly to writing the output files.\n"
    )
    if feedback:
        user_message += f"\n\nPrevious attempt feedback:\n{feedback}"
    if output_files:
        user_message += "\nDeliverable file paths (write these exactly):\n"
        for f in output_files:
            user_message += f"- {f}\n"
        user_message += "\nCreate directories as needed for these paths."
    user_message += (
        "\n\nAfter writing each deliverable file, give your final text answer that "
        "summarizes what you created AND quotes each file's key contents so they "
        "can be verified against the acceptance criteria."
    )

    handler = UsageCallbackHandler(kind="executor")
    # Tool observation rides the same config as usage. The framework's
    # on_tool_* hooks see every call ToolNode makes, so a worker gets the same
    # record of what it touched that /ask does, without a second execution path.
    tools_seen = ToolCallbackHandler(kind="executor")
    initial_snapshot = _output_file_snapshot(output_files)

    async def _stream_agent():
        final_state = None
        async for step in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="values",
            config={"callbacks": [handler, tools_seen]},
        ):
            last_msg = step["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", None)
            if tool_calls:
                names = [
                    tc.get("name", "?") if isinstance(tc, dict)
                    else getattr(tc, "name", "?")
                    for tc in tool_calls
                ]
                logger.info("Task %s tool calls: %s", task_id, names)
            final_state = step
        if final_state is not None:
            final_state["_terminus_deliverables_written"] = _deliverables_written_since(
                initial_snapshot, output_files
            )
        return final_state

    logger.info("Task %s: agent stream START", task_id)
    try:
        final_state = await asyncio.wait_for(
            _stream_agent(), timeout=_AGENT_STREAM_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Task {task_id} agent stream timed out after "
            f"{_AGENT_STREAM_TIMEOUT_SECONDS}s (the LLM or a tool failed to respond in time)"
        ) from None
    finally:
        record(handler.records, "executor")
        _log_tool_activity(task_id, tools_seen)
    logger.info("Task %s: agent stream END", task_id)

    messages = final_state.get("messages", []) if final_state else []
    deliverables_written = bool(
        final_state and final_state.get("_terminus_deliverables_written")
    )
    if not messages and not deliverables_written:
        raise ValueError(f"Agent produced no messages for task {task_id} (empty stream)")

    output = next(
        (
            message_text(msg)
            for msg in reversed(messages)
            if type(msg).__name__ == "AIMessage" and message_text(msg).strip()
        ),
        None,
    )
    if not output or not output.strip():
        if not deliverables_written:
            raise ValueError(f"Agent returned an empty response for task {task_id}")
        output = _deliverable_summary(output_files)
        logger.info("Task %s completed from verified deliverable files", task_id)

    logger.info("Task output generated for %s (%s chars)", task_id, len(output))
    return output, _read_output_file_contents(output_files)
