"""A discovered tool still succeeds after the MCP server restarts, without
rediscovering it.

Each discovered tool reopens its own connection per call, so an application
does not need to rebuild its agent graph when a long-lived MCP service is
redeployed.
"""

import pytest

from langchain.mcp import MCPAdapter

from server_process import free_port, start_scratch_server, stop_scratch_server


@pytest.mark.asyncio
async def test_tool_call_survives_server_restart():
    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    proc = start_scratch_server(port)
    try:
        async with MCPAdapter(url) as adapter:
            tools = await adapter.list_tools()
        echo = next(t for t in tools if t.name == "echo")

        before = await echo.ainvoke({"text": "before restart"})
        assert before[0]["text"] == "before restart"

        stop_scratch_server(proc)
        proc = start_scratch_server(port)

        # Same tool object discovered before the restart. No new MCPAdapter,
        # no new list_tools() call.
        after = await echo.ainvoke({"text": "after restart"})
        assert after[0]["text"] == "after restart"
    finally:
        stop_scratch_server(proc)
