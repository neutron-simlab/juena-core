"""Process helpers for CP0b's throwaway HTTP MCP server."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

SCRATCH_SERVER = Path(__file__).with_name("scratch_server.py")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                headers={"Accept": "application/json, text/event-stream"},
                timeout=1,
            )
            if response.status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001 - retried until the deadline
            last_error = exc
        time.sleep(0.2)
    raise TimeoutError(f"scratch MCP server never became ready: {last_error}")


def start_scratch_server(port: int) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen(
        [sys.executable, str(SCRATCH_SERVER)],
        env={**os.environ, "CP0B_PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_ready(f"http://127.0.0.1:{port}/mcp")
    except BaseException:
        stop_scratch_server(proc)
        raise
    return proc


def stop_scratch_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
