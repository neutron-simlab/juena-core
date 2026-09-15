"""Shared fixtures for CP0b's dependency-contract tests.

These tests exercise real library behaviour against a throwaway FastMCP server
(``scratch_server.py``) and, for the checkpointing contract, a real Postgres
started with ``docker compose -f tests/compose.postgres.yml up -d --wait``.
Nothing here is part of juena_core's public surface.
"""

import pytest

from server_process import SCRATCH_SERVER, free_port, start_scratch_server, stop_scratch_server


@pytest.fixture
def scratch_mcp_url():
    """A scratch FastMCP server reachable over real HTTP, in its own process.

    Used by the checks that need an actual network round trip (discovery by
    URL, reconnection after restart) rather than the in-process ``FastMCP``
    instance other checks pass directly as the adapter's target.
    """
    port = free_port()
    proc = start_scratch_server(port)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        stop_scratch_server(proc)


@pytest.fixture
def scratch_mcp_server():
    """The scratch ``FastMCP`` instance itself, for in-process checks that
    don't need a real socket: ``MCPAdapter`` accepts a ``FastMCP`` server
    object directly as its target.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("cp0b_scratch_server", SCRATCH_SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mcp
