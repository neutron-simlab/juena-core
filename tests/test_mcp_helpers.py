"""CP5 policy around LangChain's built-in MCP adapter."""

from __future__ import annotations

from typing import Any

import pytest

from juena_core import mcp


class _Adapter:
    events: list[tuple[str, Any]] = []

    def __init__(self, target: Any) -> None:
        self.target = target

    async def __aenter__(self) -> _Adapter:
        self.events.append(("enter", self.target))
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        self.events.append(("exit", self.target))

    async def list_tools(self) -> list[Any]:
        self.events.append(("list", self.target))
        return ["tool-a", "tool-b"]


@pytest.mark.asyncio
async def test_required_discovery_uses_adapter_context_and_returns_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Adapter.events = []
    monkeypatch.setattr(mcp, "_load_adapter_class", lambda: _Adapter)

    tools = await mcp.discover_tools("http://mcp.invalid/mcp", label="VITESS")

    assert tools == ["tool-a", "tool-b"]
    assert _Adapter.events == [
        ("enter", "http://mcp.invalid/mcp"),
        ("list", "http://mcp.invalid/mcp"),
        ("exit", "http://mcp.invalid/mcp"),
    ]


@pytest.mark.asyncio
async def test_required_discovery_raises_when_the_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> type[Any]:
        raise ImportError("fastmcp is not installed")

    monkeypatch.setattr(mcp, "_load_adapter_class", missing)

    with pytest.raises(mcp.MCPUnavailableError, match=r"juena-core\[mcp\]"):
        await mcp.discover_tools("http://mcp.invalid/mcp", label="VITESS")


@pytest.mark.asyncio
async def test_optional_discovery_degrades_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise mcp.MCPUnavailableError("Context7 is down")

    monkeypatch.setattr(mcp, "discover_tools", unavailable)

    assert (
        await mcp.discover_optional_tools(
            "http://mcp.invalid/mcp",
            label="Context7",
        )
        is None
    )


@pytest.mark.asyncio
async def test_required_discovery_wraps_adapter_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenAdapter(_Adapter):
        async def list_tools(self) -> list[Any]:
            raise ConnectionError("connection refused")

    monkeypatch.setattr(mcp, "_load_adapter_class", lambda: BrokenAdapter)

    with pytest.raises(
        mcp.MCPUnavailableError,
        match="VITESS MCP discovery failed: connection refused",
    ):
        await mcp.discover_tools("http://mcp.invalid/mcp", label="VITESS")
