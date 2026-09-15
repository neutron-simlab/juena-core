"""Shared fixtures for CP0b's dependency-contract tests.

These tests exercise real library behaviour against a throwaway FastMCP server
(``scratch_server.py``) and, for the checkpointing contract, a real Postgres
started with ``docker compose -f tests/compose.postgres.yml up -d --wait``.
Nothing here is part of juena_core's public surface.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

SCRATCH_SERVER = Path(__file__).with_name("scratch_server.py")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                headers={"Accept": "application/json, text/event-stream"},
                timeout=1,
            )
            if r.status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001 - retried until the deadline
            last_error = exc
        time.sleep(0.2)
    raise TimeoutError(f"scratch MCP server never became ready: {last_error}")


def start_scratch_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, str(SCRATCH_SERVER)],
        env={**os.environ, "CP0B_PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_ready(f"http://127.0.0.1:{port}/mcp")
    return proc


def stop_scratch_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def scratch_mcp_url():
    """A scratch FastMCP server reachable over real HTTP, in its own process.

    Used by the checks that need an actual network round trip (discovery by
    URL, reconnection after restart) rather than the in-process ``FastMCP``
    instance other checks pass directly as the adapter's target.
    """
    port = _free_port()
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
