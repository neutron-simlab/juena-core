"""CP0b, row 3: a tool call succeeds after the MCP server restarts, without
rebuilding the agent. VITESS execution is a long-lived internal Compose
service; if a redeploy meant every discovered tool went stale, the app would
need to rebuild its agent graph on every MCP restart. It does not have to:
each discovered tool reopens its own connection per call.
"""

import pytest

from langchain.mcp import MCPAdapter

from conftest import _free_port, start_scratch_server, stop_scratch_server


@pytest.mark.asyncio
async def test_tool_call_survives_server_restart():
    port = _free_port()
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
