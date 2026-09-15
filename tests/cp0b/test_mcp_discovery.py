"""CP0b, row 1 and row 6 of the ten-check table.

Discovery: ``async with MCPAdapter(target) as adapter: await adapter.list_tools()``
returns the expected tools. Structured content: a FastMCP tool's
``structuredContent`` is reachable at ``message.artifact["structured_content"]``
without parsing model-visible text.
"""

import pytest

from langchain.mcp import MCPAdapter


@pytest.mark.asyncio
async def test_discovery_returns_expected_tools(scratch_mcp_server):
    async with MCPAdapter(scratch_mcp_server) as adapter:
        tools = await adapter.list_tools()
    assert {t.name for t in tools} == {"echo", "add"}


@pytest.mark.asyncio
async def test_structured_content_reachable_via_artifact(scratch_mcp_server):
    async with MCPAdapter(scratch_mcp_server) as adapter:
        tools = await adapter.list_tools()
    add = next(t for t in tools if t.name == "add")

    tool_call = {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1", "type": "tool_call"}
    message = await add.ainvoke(tool_call)

    assert message.artifact["structured_content"] == {"total": 5}


@pytest.mark.asyncio
async def test_discovery_over_real_http(scratch_mcp_url):
    """The deployed shape: a bare URL string, not an in-process server object."""
    async with MCPAdapter(scratch_mcp_url) as adapter:
        tools = await adapter.list_tools()
    assert {t.name for t in tools} == {"echo", "add"}
