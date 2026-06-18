"""Live MCP stdio smoke test: spawn `python -m dopest_clip` as a real MCP server, do the
protocol handshake, list tools + resources, and actually CALL a tool over MCP.

Proves the MCP-first face works end-to-end over the wire (not just in-process registration).
Run from the repo root:  .venv\\Scripts\\python.exe tests\\mcp_smoke.py
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "dopest_clip"], cwd=".")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print("server:", info.serverInfo.name, info.serverInfo.version)

            tools = (await session.list_tools()).tools
            print("tools registered:", len(tools))
            print("sample tools:", ", ".join(t.name for t in tools[:6]))

            resources = (await session.list_resources()).resources
            print("learn resources:", len(resources))

            # read a learn:// resource over MCP
            ov = await session.read_resource("learn://overview")
            body = ov.contents[0].text
            print("learn://overview bytes:", len(body))

            # actually CALL a tool over the protocol
            r = await session.call_tool("list_providers", {})
            print("call list_providers ok, first 90 chars:", r.content[0].text[:90])

            # call an editing tool (no deps): list_projects
            r2 = await session.call_tool("list_projects", {})
            print("call list_projects ok, first 90 chars:", r2.content[0].text[:90])

    print("MCP STDIO ROUND-TRIP: OK")


if __name__ == "__main__":
    asyncio.run(main())
