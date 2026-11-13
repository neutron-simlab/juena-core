"""A discovered tool still succeeds after the ``async with MCPAdapter(...)``
discovery block has exited.

``juena_core.mcp`` hands back the discovered tools and keeps no adapter alive,
so it relies on each tool being usable on its own after discovery.
"""

import pytest

from langchain.mcp import MCPAdapter


@pytest.mark.asyncio
async def test_tool_survives_discovery_block_exit(scratch_mcp_server):
    async with MCPAdapter(scratch_mcp_server) as adapter:
        tools = await adapter.list_tools()
    echo = next(t for t in tools if t.name == "echo")

    # Outside the `async with` block. The discovery context has already exited.
    result = await echo.ainvoke({"text": "still alive"})
    assert result[0]["text"] == "still alive"
