"""The MCP connection layer, behind the ``[mcp]`` extra.

New in core — there is no ``juena/mcp.py`` to port; the shape comes from
``juena/tools/context7.py``, which ran the retired ``langchain-mcp-adapters``.
**This is the only module in either application stack that imports
``langchain.mcp``** (01-EXTRACT-JUENA-CORE.md, CP5). The namespace is beta, so
containing every import here makes a breaking release one module to fix rather
than a search across two applications.

The library contract below was measured in 01/CP0b (``tests/cp0b/``) against
the installed release, not assumed:

- ``MCPAdapter`` is an async context manager, and ``list_tools()`` returns
  tools that **stay callable after the discovery block exits** — each reopens
  its own connection per call. That is also why a tool survives an MCP server
  restart with no agent rebuild (``test_mcp_reconnection.py``).
- The config shape is ``{"mcpServers": {...}}`` — FastMCP's ``MCPConfig`` —
  and the transport is inferred from each entry. There is no ``transport``
  key any more. One server may be passed as a bare URL string.
- Discovery is ``await adapter.list_tools()``, not ``get_tools()``.

Because discovered tools are self-contained, this module hands back tools and
nothing else: there is no adapter to keep alive, and
:func:`~juena_core.server.agent.registry.shutdown_agents` has no MCP cleanup
to perform. Calling ``__aexit__`` outside the block that entered it would be
wrong in any case.

**The two functions differ by policy, not by library behaviour**, and the
difference is a decision rather than an inconsistency:

:func:`discover_optional_tools`
    For an integration nobody's product depends on. If the server is
    unreachable its tools should not exist, rather than exist and fail on
    first use, so this returns ``None`` and logs. juena-chatbot's Context7 is
    this case.

:func:`discover_tools`
    For an integration that *is* the product. VITESS execution has nothing to
    degrade to, so a failure raises and 03/CP3 makes the service healthy
    before it builds the agent at all.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from juena_core.log import get_logger

__all__ = ["MCPUnavailableError", "discover_tools", "discover_optional_tools"]

logger = get_logger(__name__)


class MCPUnavailableError(RuntimeError):
    """An MCP server whose tools are required could not be reached."""


def _load_adapter_class() -> type[Any]:
    """Import ``MCPAdapter`` on first use.

    Deferred so that importing this module does not require the ``[mcp]``
    extra, and so a missing install is a condition a caller can report rather
    than an ImportError at startup. Importing ``langchain.mcp`` also raises
    ``LangChainBetaWarning`` once per process; core does not filter it —
    letting the warning surface is how anyone notices the namespace changed.
    """

    from langchain.mcp import MCPAdapter

    return MCPAdapter


async def discover_tools(target: Any, *, label: str = "MCP") -> list[BaseTool]:
    """Discover the tools an MCP server offers, raising if it cannot be reached.

    Args:
        target: Anything ``MCPAdapter`` accepts — a URL string, an
            ``{"mcpServers": {...}}`` config, or a ``FastMCP`` instance.
        label: Name used in errors and logs, e.g. ``"VITESS"``.

    Raises:
        MCPUnavailableError: the extra is not installed, or discovery failed.
    """

    try:
        adapter_cls = _load_adapter_class()
    except ImportError as exc:
        raise MCPUnavailableError(
            f"{label} tools require the MCP extra. Install juena-core[mcp]."
        ) from exc

    try:
        async with adapter_cls(target) as adapter:
            tools = list(await adapter.list_tools())
    except Exception as exc:
        raise MCPUnavailableError(f"{label} MCP discovery failed: {exc}") from exc

    logger.info("Discovered %d %s MCP tool(s)", len(tools), label)
    return tools


async def discover_optional_tools(target: Any, *, label: str) -> list[BaseTool] | None:
    """Discover an optional integration's tools, or ``None`` if unavailable.

    Never raises. An optional integration that is down should leave the agent
    with one capability fewer, not with a tool that fails the first time a
    model reaches for it.
    """

    try:
        return await discover_tools(target, label=label)
    except MCPUnavailableError as exc:
        logger.warning("%s", exc)
        return None
