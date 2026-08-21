from terminus.agent.factory import build_agent
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

def handle_query(question:str)->str:
    """ Handles the query and returns the response """
    logger.info(f"Handling query: {question}")
    agent = build_agent()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return response["messages"][-1].content
