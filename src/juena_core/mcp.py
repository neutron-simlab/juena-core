"""Stub for 01/CP5, behind the ``[mcp]`` extra. New in core — there is no
``juena/mcp.py`` to port. **The only module in either application stack that
imports ``langchain.mcp``** (01-EXTRACT-JUENA-CORE.md, CP5): the namespace is
beta, and containing every import here means a breaking release is one
module to fix rather than a search across two applications.

Contract, measured in 01/CP0b (``tests/cp0b/``, 11 passing tests) rather than
assumed:

- ``MCPAdapter`` is an async context manager; ``list_tools()`` discovers
  tools that remain callable after the discovery block exits — each reopens
  its own connection per call, which is also what survives an MCP server
  restart without rebuilding anything here.
- The config shape is ``{"mcpServers": {...}}`` (or a bare URL for one
  server); transport is inferred, there is no ``transport`` key.
- ``message.artifact["structured_content"]`` carries the FastMCP tool's
  ``structuredContent``.

Two contracts this module keeps that are policy, not library behaviour, and
differ by caller:

- **An optional integration returns ``None`` on failure** — nobody's product
  depends on it, so its tools should not exist rather than exist and fail
  (mirrors ``indexing/bootstrap.py``'s survey pattern, copied not
  abstracted, per 00-BOUNDARY.md, decision 4).
- **VITESS execution is not optional.** 03/CP3 makes the MCP service healthy
  before construction instead of degrading, because there the tools are the
  product, not a nice-to-have.

``discover_tools``' exact signature is 01/CP5's to finalize; this stub fixes
its degrade-on-failure contract, not its ergonomics.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

__all__ = ["discover_tools"]


async def discover_tools(target: object) -> list[BaseTool] | None:
    raise NotImplementedError("juena_core.mcp.discover_tools lands in 01/CP5")
