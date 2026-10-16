"""Disposable real-filesystem API and deterministic SSE for Chromium tests only."""
import asyncio
import base64
import json
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
import uvicorn
from deepagents.backends.utils import create_file_data
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from juena_core.clients.base import BaseAgentClient
from juena_core.filesystem import api as routes
from juena_core.filesystem.service import ConversationFilesystem
from juena_core.filesystem.storage import WorkspaceStore
from juena_core.server.chat.repository import ChatNotFoundError
from juena_core.server.database.connection import get_db_session
from juena_core.server.streaming.own_task import AgentStreamResponse

API = "http://127.0.0.1:19621"
ORIGIN = "http://127.0.0.1:19620"


@st.cache_resource
def setup():
    temporary = tempfile.TemporaryDirectory(prefix="juena-browser-")
    filesystem = ConversationFilesystem(WorkspaceStore(Path(temporary.name)), identity_secret="browser-tests-secret-at-least-32-characters")
    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=[ORIGIN], allow_credentials=True,
                       allow_methods=["GET", "PUT"], allow_headers=["Content-Type"])
    stats = {"started": 0, "completed": 0, "cancelled": 0, "filesystem_requests": 0}
    locks = ExitStack()
    expired = False
    paused = False

    def principal(request: Request):
        if expired or request.cookies.get("juena_session") != "browser-test-cookie":
            raise HTTPException(401, "Session expired")
        return SimpleNamespace(id="alice")

    async def owner(session, user_id, thread_id, *, agent_id):
        if user_id != "alice" or thread_id not in ("one", "two"):
            raise ChatNotFoundError(thread_id)
        return SimpleNamespace(agent_id="test")

    async def snapshot(config):
        return SimpleNamespace(values={"files": {
            "/inputs/message.txt": create_file_data("pasted input"),
            "/findings/note.md": create_file_data("specialist observation"),
        }})

    async def agent(_agent_id):
        return SimpleNamespace(aget_state=snapshot)

    async def pending(*args):
        return object() if paused else None

    async def session():
        yield object()

    routes.get_owned_chat = owner
    routes.get_agent = agent
    routes.first_pending_interrupt = pending
    api.dependency_overrides[get_db_session] = session
    api.include_router(routes.build_filesystem_router(principal, filesystem, browser_origins=(ORIGIN,)))

    def seed():
        for thread in ("one", "two"):
            files = filesystem.for_thread("alice", thread)
            files.ensure()
            for name, content in {
                "analysis.py": f"value = '{thread}'\n".encode(),
                "outputs/nested/result.txt": b"uncached result\n",
                "outputs/other.txt": b"other file\n",
                "preview.html": b"<script>window.unsafePreview = true</script>",
                "notebook.ipynb": b'{"cells": []}',
                "picture.png": base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1sAAAAASUVORK5CYII="),
                "binary.bin": b"\x00\xff\x01",
                ".scratch/temp.py": b"temporary",
                "large.txt": b"x" * (1024 * 1024 + 1),
                **{f"many/file_{n:03}.txt": str(n).encode() for n in range(105)},
            }.items():
                files.store.put(files.workspace_id, "/workspace/" + name, content)
    seed()

    @api.middleware("http")
    async def count_files(request, call_next):
        if "/filesystem" in request.url.path:
            stats["filesystem_requests"] += 1
        return await call_next(request)

    @api.get("/chats/{thread_id}")
    async def chat(thread_id: str, request: Request):
        principal(request)
        if thread_id not in ("one", "two"):
            raise HTTPException(404)
        return {"thread_id": thread_id, "agent_id": "test"}

    @api.get("/test/stats")
    def observe():
        return stats

    @api.post("/test/control/{action}")
    def control(action: str):
        nonlocal expired, paused
        files = filesystem.for_thread("alice", "one")
        if action == "reset":
            locks.close(); expired = paused = False
            stats.update(started=0, completed=0, cancelled=0, filesystem_requests=0)
            seed()
        elif action == "lock":
            locks.enter_context(files.guard(operation=True, shared=True))
            locks.enter_context(files.guard(execution=True))
        elif action == "unlock":
            locks.close()
        elif action == "expire":
            expired = True
        elif action == "pause":
            paused = True
        elif action == "unpause":
            paused = False
        elif action == "mutate":
            files.store.put(files.workspace_id, "/workspace/analysis.py", b"agent changed this\n")
        return {"ok": True}

    @api.post("/test/stream")
    async def stream(request: Request):
        payload = await request.json()
        files = filesystem.for_thread("alice", payload["thread_id"])
        stats["started"] += 1
        async def events():
            with files.guard(operation=True, shared=True):
                try:
                    for index in range(60):
                        yield {"data": json.dumps({"type": "token", "content": f"{index} "})}
                        await asyncio.sleep(0.2)
                    files.store.put(files.workspace_id, "/workspace/outputs/nested/finished.txt", b"created at completion")
                    stats["completed"] += 1
                    yield {"data": json.dumps({"type": "token", "content": "STREAM FINISHED"})}
                except asyncio.CancelledError:
                    stats["cancelled"] += 1
                    raise
        return AgentStreamResponse(events(), on_close=lambda: None)

    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=19621, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    # Keep temporary storage alive for this process only.
    api.state.temporary = temporary
    return BaseAgentClient(API, agent="test", timeout=30)
