"""The MCP connection layer, behind the ``[mcp]`` extra.

MCP (Model Context Protocol) lets an agent use tools served by another
process. Import ``langchain.mcp`` only in this module: the namespace is beta,
so keeping every import here makes a breaking release one module to fix.

How the installed library behaves, checked rather than assumed:

- ``MCPAdapter`` is an async context manager, and ``list_tools()`` returns
  tools that stay callable after the discovery block exits: each opens its
  own connection per call. So a tool survives an MCP server restart with no
  agent rebuild.
- The config shape is ``{"mcpServers": {...}}`` (FastMCP's ``MCPConfig``),
  and the transport is inferred from each entry; there is no ``transport``
  key. One server may be passed as a bare URL string.
- Discovery is ``await adapter.list_tools()``, not ``get_tools()``.

Because discovered tools are self-contained, this module hands back tools and
nothing else: there is no adapter to keep alive, and
:func:`~juena_core.server.agent.registry.shutdown_agents` has no MCP cleanup
to perform. Calling ``__aexit__`` outside the block that entered it would be
wrong in any case.

The two discovery functions differ only in what they do on failure:

:func:`discover_optional_tools`
    For an integration the application can run without. If the server is
    unreachable its tools should not exist, rather than exist and fail on
    first use, so this logs and returns ``None``.

:func:`discover_tools`
    For an integration the application cannot work without. There is
    nothing to fall back to, so a failure raises.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from juena_core.log import get_logger

__all__ = ["MCPUnavailableError", "discover_tools", "discover_optional_tools", "call_personal_tool"]

logger = get_logger(__name__)


class MCPUnavailableError(RuntimeError):
    """An MCP server whose tools are required could not be reached.

    ``status_code`` holds the HTTP status of the failed request, when there
    was one.
    """

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


async def call_personal_tool(target: Any, name: str, arguments: dict[str, Any]) -> Any:
    """Use a request-owned MCP connection without retaining its auth configuration.

    Do not interpolate upstream errors: they may contain authentication headers.

    Args:
        target: Anything ``MCPAdapter`` accepts, including the request's
            credentials.
        name: Tool name, matched exactly or as the ending ``_<name>``.
        arguments: Arguments for the tool call.

    Returns:
        The tool's structured content, or its content blocks as dicts.

    Raises:
        MCPUnavailableError: Any failure, with a fixed message.
    """
    try:
        async with _load_adapter_class()(target) as adapter:
            tools = await adapter.list_tools()
            selected = next(t for t in tools if t.name == name or t.name.endswith("_" + name))
            result = await adapter.client.call_tool(selected.name, arguments, raise_on_error=True)
            return result.structured_content or [block.model_dump(exclude_none=True) for block in result.content]
    except Exception as error:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        raise MCPUnavailableError("The documentation service could not complete the request.",
                                  status_code=status_code) from None


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

    Returns:
        The discovered tools.

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

    Args:
        target: Anything ``MCPAdapter`` accepts, as for :func:`discover_tools`.
        label: Name used in the warning and logs.

    Returns:
        The discovered tools, or ``None`` when discovery failed.
    """

    try:
        return await discover_tools(target, label=label)
    except MCPUnavailableError as exc:
        logger.warning("%s", exc)
        return None
