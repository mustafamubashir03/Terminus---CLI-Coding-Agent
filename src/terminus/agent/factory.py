from terminus.llm.factory import get_llm
from terminus.agent.tools import search_codebase
from terminus.observability.logging import get_logger
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a code assistant. Your job is to answer questions about the codebase.

RULES (follow strictly):
1. Call 'search_codebase' with the most relevant query.
2. Read the tool results carefully.
3. Immediately provide your final answer based on those results. Do NOT call the tool again.
4. If the results don't contain the answer, say so - do NOT keep searching with different queries.

ALWAYS give a final text answer after at most 2 tool calls."""

def build_agent():
    """ Create and return a langchain agent"""
    llm = get_llm()
    tools = [search_codebase]
    logger.info("Creating agent")
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=4, exit_behavior="end")
]
    )