from terminus.mcp.terminus_mcp_config import load_terminus_mcp_config
from langchain_mcp_adapters.client import MultiServerMCPClient
from terminus.observability.logging import get_logger

logger = get_logger(__name__)


async def get_terminus_mcp_tools():
    """ Connect to all configured MCP servers and return their tools"""
    configs = load_terminus_mcp_config()
    logger.info(f"Connecting to MCP servers: {configs.keys()}")
    client = MultiServerMCPClient(configs)
    tools = await client.get_tools()
    logger.info(f"Connected to {len(tools)} tools")
    return tools