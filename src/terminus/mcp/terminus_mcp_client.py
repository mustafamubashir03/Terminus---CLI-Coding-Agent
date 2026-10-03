"""Shared MCP client connection.

MCP servers (e.g. the GitHub stdio server) run as subprocesses, and spawning
them costs seconds. Every agent/task previously opened a brand-new
MultiServerMCPClient, so a /plan run with 8 tasks paid 8 subprocess spawns plus
8 x 26-tool inventories - a major source of the CLI's sequential slowness.
This module keeps ONE client per process and hands the same tools to every
agent. Failures degrade to no MCP tools instead of aborting the agent.
"""

from terminus.mcp.terminus_mcp_config import load_terminus_mcp_config
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

_mcp_client = None
_mcp_tools: list | None = None
_mcp_lock = None  # asyncio.Lock, created lazily on the running loop


def _get_lock():
    global _mcp_lock
    if _mcp_lock is None:
        import asyncio

        _mcp_lock = asyncio.Lock()
    return _mcp_lock


async def get_terminus_mcp_tools() -> list:
    """Return the tools of all configured MCP servers, connecting at most once.

    Returns ``[]`` when no servers are configured or when a server cannot be
    reached, so MCP availability never blocks or breaks the agents that merely
    use the filesystem.
    """
    global _mcp_client, _mcp_tools
    if _mcp_tools is not None:
        return _mcp_tools

    async with _get_lock():
        if _mcp_tools is not None:
            return _mcp_tools

        configs = load_terminus_mcp_config()
        if not configs:
            _mcp_tools = []
            return _mcp_tools

        logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(configs)
        try:
            tools = await client.get_tools()
        except Exception:
            logger.exception("Failed to connect to MCP servers; continuing without them")
            # No cleanup call: langchain-mcp-adapters >= 0.1.0 removed async
            # context manager support, and ``__aexit__`` is a synchronous stub
            # that raises NotImplementedError. Calling it (let alone awaiting it -
            # it is not a coroutine) is both wrong and unnecessary, because the
            # stdio subprocesses are owned by this process and die with it. This
            # is the same reasoning as close_terminus_mcp below.
            _mcp_client = None
            _mcp_tools = []
            return _mcp_tools

        _mcp_client = client
        _mcp_tools = tools
        logger.info(f"Connected to {len(tools)} MCP tools")
        return _mcp_tools


async def close_terminus_mcp() -> None:
    """Close the shared MCP client, if one was opened (idempotent).

    langchain-mcp-adapters >= 0.1.0 removed async context manager support.
    The stdio subprocesses are killed when the parent process exits, so we
    just drop our reference to allow garbage collection.
    """
    global _mcp_client, _mcp_tools
    _mcp_client = None
    _mcp_tools = None