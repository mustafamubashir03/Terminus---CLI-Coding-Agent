from pydantic import BaseModel
from terminus.observability.logging import get_logger
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from terminus.mcp.terminus_mcp_client import get_terminus_mcp_tools
from terminus.tools.codebase_tool import search_codebase
from terminus.tools.terminal_tools import run_command,run_in_directory
from terminus.skills.skill_tools import load_skill,build_skills_prompt
from terminus.tools.filesystem_tools import list_directory,read_file,write_file,delete_file,file_exists,append_file

from terminus.config import CONFIG
import json

logger = get_logger(__name__)

# Simple tools for agent

_DEFAULT_TOOLS = [search_codebase]


async def _get_tools_by_type():
    mcp_tools = await get_terminus_mcp_tools()
    filesystem_tools = [list_directory, read_file, write_file, delete_file, file_exists, append_file]

    return {
        "design": [
            *_DEFAULT_TOOLS,
            *filesystem_tools,
            load_skill,
        ],
        "implement": [
            *_DEFAULT_TOOLS,
            *filesystem_tools,
            *mcp_tools,
            run_command,
            run_in_directory,
            load_skill,
        ],
        "test": [
            *_DEFAULT_TOOLS,
            *filesystem_tools,
            *mcp_tools,
            run_command,
            run_in_directory,
        ],
        "review": [
            *_DEFAULT_TOOLS,
            *filesystem_tools,
            load_skill,
        ],
        "integrate": [
            *_DEFAULT_TOOLS,
            *filesystem_tools,
            *mcp_tools,
            run_command,
            run_in_directory,
        ],
        "configure": [
            *_DEFAULT_TOOLS,
            *filesystem_tools,
            *mcp_tools,
            run_command,
            run_in_directory,
        ],
    }

def _build_system_prompt(task: dict, dep_outputs: list[dict]) -> str:
    prompt = f"You are tasked with executing the following subtask:\n"
    prompt += f"Task Type: {task.get('task_type')}\n"
    prompt += f"Description: {task.get('description')}\n"
    
    if dep_outputs:
        prompt += "\nHere are the results from dependencies that you must use:\n"
        for dep in dep_outputs:
            prompt += f"- Dependency Task {dep.get('id')}:\n{dep.get('result')}\n"
            
    prompt += "\nEnsure you output exactly what is required to complete this task."
    return prompt

class Verdict(BaseModel):
    passed: bool
    reason: str

async def _judge_task(task: dict, output: str) -> Verdict:
    """Evaluate the output against the acceptance criteria."""
    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"]["model"]
    llm = init_chat_model(model, model_provider=provider)
    
    criteria = task.get("acceptance_criteria", [])
    if isinstance(criteria, str):
        try:
            criteria = json.loads(criteria)
        except json.JSONDecodeError:
            criteria = []
    
    system_prompt = "You are a stringent judge evaluating whether a task's output meets all its acceptance criteria."
    user_message = f"Task Description: {task.get('description')}\n\n"
    user_message += "Acceptance Criteria:\n"
    for c in criteria:
        user_message += f"- {c}\n"
    user_message += f"\nOutput:\n{output}\n\n"
    user_message += "Did the output pass all acceptance criteria? Answer with strict true/false and provide a reason."

    judge_agent = create_agent(llm, tools=[], system_prompt=system_prompt, response_format=Verdict)
    result = await judge_agent.ainvoke({"messages": [{"role": "user", "content": user_message}]})
    return result["structured_response"]

async def run_subtask_agent(task: dict, dep_outputs: list[dict] | None = None) -> str:
    """
    Build a fresh agent for a single task and invoke it.
    After the agent returns, an LLM judge verifies the output against acceptance_criteria.
    If it fails, raise ValueError so the orchestrator's existing retry logic kicks in automatically.
    """
    provider = CONFIG["llm"]["provider"]
    model = CONFIG["llm"]["model"]

    llm = init_chat_model(model, model_provider=provider)
    tool_map = await _get_tools_by_type()
    tools = tool_map.get(
        task.get("task_type", ""),
        _DEFAULT_TOOLS
    )
    skills_prompt = build_skills_prompt()
    system_prompt = _build_system_prompt(task, dep_outputs or [])
    system_prompt = f"{system_prompt}\n\n======SKILLS======\n\n{skills_prompt}"
    logger.info(
        f"Building agent for task {task['id']}: {task['description']}"
    )
    
    agent = create_agent(llm, tools=tools, system_prompt=system_prompt)
    user_message = (
        f"{task['description']}\n\n"
        "The project directory may be empty - there is no existing code to read.\n"
        "You must CREATE all output files from scratch.\n"
        "Do not spend time listing directories. Go directly to writing the output files."
    )
    final_state = None
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="values",
    ):
        last_msg = step["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if tool_calls:
            tool_names = [tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?") for tc in tool_calls]
            logger.info(f"Task {task['id']} tool calls: {tool_names}")
        final_state = step
     
    def _get_content(msg) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # LANGSMITH RICH OUTPUT (e.g. Markdown) CAN COME AS A LIST OF TEXT/IMAGE/FILE BLOBS. 
            # COLLAPSE TO A SINGLE STRING.
            return "\n".join(
                s.get("text", "") if isinstance(s, dict) else str(s) for s in content
            )
        return content or ""

    # The final AI Message can be empty for reasoning models(reasoning tokens are internal).
    # Walk backwards to find the last message with content.
    output = next(
        (_get_content(msg) for msg in reversed(final_state["messages"]) if type(msg).__name__ == "AIMessage" and _get_content(msg).strip()),
        None
    )
    
    if not output or not output.strip():
        raise ValueError(f"Agent returned an empty response for task {task['id']}")
    
    logger.info(
        f"Task output generated for {task['id']} ({len(output)} chars)"
    )

    ## LLM as a judge
    verdict = await _judge_task(task, output)
    logger.info(
        f"Judge verdict for task {task['id']}: passed={verdict.passed}, reason={verdict.reason[:200]}"
    )
    if not verdict.passed:
        raise ValueError(f"Judge rejected output for task {task['id']}: {verdict.reason}")
    return output
