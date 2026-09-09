import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server = StdioServerParameters(
        command=r"G:\nodejs\npx.CMD",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            r"G:\terminus-testing\frontend-testing",
        ],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.initialize()
            print("INITIALIZED:", result)

            tools = await session.list_tools()
            print("TOOLS:", [tool.name for tool in tools.tools])


asyncio.run(main())