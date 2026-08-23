from terminus.memory.short_term import get_summarization_middleware
from terminus.memory.short_term import get_checkpointer
from terminus.llm.factory import get_llm
from terminus.agent.tools import search_codebase
from terminus.observability.logging import get_logger
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a code assistant. Your job is to answer questions about the codebase.If you got no memory of past converstation of user just say so no need to retrieve data by searching codebase when no required.

RULES (follow strictly):
1. Call 'search_codebase' with the most relevant query when required.
2. Read the tool results carefully.
3. Immediately provide your final answer based on those results. Do NOT call the tool again only if the information retrieved is not enough to answer the question or if the user asks for more information.

ALWAYS give a final text answer."""

def build_agent():
    """ Create and return a langchain agent"""
    llm = get_llm()
    tools = [search_codebase]
    logger.info("Creating agent")
    checkpointer = get_checkpointer()
    middlewares = [
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="search_codebase", run_limit=4, exit_behavior="end"),
        get_summarization_middleware()
    ]
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=middlewares
    )