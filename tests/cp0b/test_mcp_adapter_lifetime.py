"""CP0b, row 2: a discovered tool must still succeed after the
``async with MCPAdapter(...)`` discovery block has exited. This is the
contract that inverted when ``langchain-mcp-adapters`` (whose client raised on
``__aenter__``) was replaced by ``langchain.mcp`` (whose adapter *is* an async
context manager). If this regresses, CP5's "discovered tools are stateless
after discovery" note is wrong and must be revised before it is relied on.
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
