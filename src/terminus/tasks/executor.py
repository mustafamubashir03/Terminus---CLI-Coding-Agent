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
from langchain.agents import create_agent
from terminus.llm.factory import get_chat_model
from terminus.llm.text import message_text
from terminus.tools.codebase_tool import search_codebase
from terminus.tools.terminal_tools import run_command,run_in_directory
from terminus.skills.skill_tools import load_skill,build_skills_prompt
from terminus.mcp.terminus_mcp_client import get_terminus_mcp_tools
from terminus.tools.filesystem_tools import list_directory,read_file,write_file,delete_file,file_exists,append_file
from terminus.workspace import project_root

from terminus.config import CONFIG
from terminus.cache import get_cached_prompt, cache_prompt, retry_async
from terminus.observability.usage_tracker import UsageCallbackHandler, record

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

# Simple tools for agent

_DEFAULT_TOOLS = [search_codebase]

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
    """Build and cache the per-task-type tool lists for the process."""
    global _CACHED_TOOLS
    if _CACHED_TOOLS is not None:
        return _CACHED_TOOLS
    mcp_tools = await asyncio.wait_for(
        get_terminus_mcp_tools(), timeout=_MCP_TOOLS_TIMEOUT_SECONDS
    )
    mcp_tools = _filter_github_code_read_tools(mcp_tools)
    filesystem_tools = [list_directory, read_file, write_file, delete_file, file_exists, append_file]

    _CACHED_TOOLS = {
        "design": _dedupe([*_DEFAULT_TOOLS, *filesystem_tools, load_skill]),
        "implement": _dedupe([*_DEFAULT_TOOLS, *filesystem_tools, *mcp_tools, run_command, run_in_directory, load_skill]),
        "test": _dedupe([*_DEFAULT_TOOLS, *filesystem_tools, *mcp_tools, run_command, run_in_directory]),
        "review": _dedupe([*_DEFAULT_TOOLS, *filesystem_tools, load_skill]),
        "integrate": _dedupe([*_DEFAULT_TOOLS, *filesystem_tools, *mcp_tools, run_command, run_in_directory]),
        "configure": _dedupe([*_DEFAULT_TOOLS, *filesystem_tools, *mcp_tools, run_command, run_in_directory]),
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


def _build_system_prompt(task: dict, dep_outputs: list[dict]) -> str:
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

    prompt += "\nEnsure you output exactly what is required to complete this task."
    return prompt

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

    llm = get_chat_model(model, model_provider=provider)
    tool_map = await _tool_plans()
    tools = tool_map.get(task.get("task_type", ""), _DEFAULT_TOOLS)

    system_prompt = _build_system_prompt(task, dep_outputs)
    if feedback:
        system_prompt += f"\n\nPrevious attempt feedback:\n{feedback}"
    system_prompt = f"{system_prompt}\n\n======SKILLS======\n\n{build_skills_prompt()}"
    logger.info("Building worker agent for task %s: %s", task_id, task.get("description"))

    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
    )

    middlewares = [
        ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=4, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="write_file", run_limit=30, exit_behavior="end"),
    ]
    agent = create_agent(llm, tools=tools, system_prompt=system_prompt, middleware=middlewares)

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
    initial_snapshot = _output_file_snapshot(output_files)

    async def _stream_agent():
        final_state = None
        async for step in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="values",
            config={"callbacks": [handler]},
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
