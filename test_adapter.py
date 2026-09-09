import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    config = {
        "filesystem": {
            "command": r"G:\nodejs\npx.CMD",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                r"G:\terminus-testing\frontend-testing",
            ],
            "transport": "stdio",
        }
    }

    print("Creating client...")

    client = MultiServerMCPClient(config)

    print("Getting tools...")

    tools = await client.get_tools()

    print("TOOLS:", [tool.name for tool in tools])


asyncio.run(main())