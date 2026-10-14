"""Authenticated filesystem endpoints using the conversation's canonical paths."""

from __future__ import annotations

__all__ = ['SaveFileInput', 'build_filesystem_router']

import asyncio
from contextlib import asynccontextmanager, nullcontext
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from juena_core.filesystem.service import ConversationFilesystem, MAX_EDIT_BYTES
from juena_core.filesystem.storage import RevisionConflict, WorkspaceBusy, WorkspaceDeleted, WorkspaceOwnershipError, assert_safe_path
from juena_core.server.agent.registry import get_agent
from juena_core.server.chat.repository import ChatNotFoundError, get_owned_chat
from juena_core.server.database.checkpointer import get_checkpointer
from juena_core.server.database.connection import get_db_session
from juena_core.server.identity import Principal, PrincipalDependency
from juena_core.server.interrupts import ResumeError, first_pending_interrupt


class SaveFileInput(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    text: str = Field(max_length=MAX_EDIT_BYTES)
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")


def _browser_visible(path: str) -> bool:
    path = str(assert_safe_path(path))
    return not any(path == root or path.startswith(root + "/") for root in (
        "/workspace/conversation_history", "/workspace/.scratch/conversation_history",
    ))


def _browser_path(path: str) -> str:
    if not _browser_visible(path):
        raise PermissionError("Internal conversation history is not available in the file browser")
    return str(assert_safe_path(path))


def build_filesystem_router(principal: PrincipalDependency, filesystem: ConversationFilesystem,
                            *, browser_origins: tuple[str, ...] = ()) -> APIRouter:
    router = APIRouter(prefix="/threads/{thread_id}/filesystem", tags=["filesystem"])

    @asynccontextmanager
    async def access(thread_id: str, user: Principal, session: AsyncSession, *, write: bool = False):
        try:
            chat = await get_owned_chat(session, user.id, thread_id, agent_id=None)
            files = filesystem.for_thread(str(user.id), thread_id)
            # Reads are protected against deletion by their shared files lock.
            # Reserving an operation as well would make Save race with previews.
            async with files.aguard(operation=True) if write else nullcontext():
                # A deletion may have completed between authorization and the lock.
                await get_owned_chat(session, user.id, thread_id, agent_id=None)
                config = {"configurable": {"thread_id": thread_id, "user_id": str(user.id)}}
                if write:
                    await asyncio.to_thread(filesystem.assert_idle, files.workspace_id)
                    agent = await get_agent(chat.agent_id)
                    if await first_pending_interrupt(agent, config) is not None:
                        raise WorkspaceBusy("Answer the pending approval or question before editing files")
                    files.store.mark_paused(files.workspace_id, False)
                checkpoint = await get_checkpointer().aget(config)
                state = (checkpoint or {}).get("channel_values", {}).get("files", {})
                if files.store.deleted(files.workspace_id):
                    raise WorkspaceDeleted("Conversation workspace was deleted")
                async with files.aguard(shared=True):
                    migrate = files.needs_migration(state)
                if migrate:
                    async with files.aguard():
                        await asyncio.to_thread(files.migrate, state)
                async with files.aguard(shared=not write):
                    yield files, state
        except ChatNotFoundError:
            raise HTTPException(404, {"code": "conversation_not_found", "message": "Conversation not found"}) from None
        except FileNotFoundError:
            raise HTTPException(404, {"code": "file_not_found", "message": "File or folder not found"}) from None
        except RevisionConflict as exc:
            raise HTTPException(409, {"code": "revision_conflict", "message": str(exc)}) from exc
        except (WorkspaceBusy, ResumeError) as exc:
            raise HTTPException(409, {"code": "workspace_busy", "message": str(exc)}) from exc
        except WorkspaceDeleted as exc:
            raise HTTPException(403, {"code": "workspace_deleted", "message": str(exc)}) from exc
        except WorkspaceOwnershipError as exc:
            raise HTTPException(403, {"code": "workspace_forbidden", "message": str(exc)}) from exc
        except PermissionError as exc:
            raise HTTPException(403, {"code": "path_read_only" if write else "path_forbidden", "message": str(exc)}) from exc
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("")
    async def list_files(thread_id: str, response: Response, path: str = Query("/", max_length=4096),
                         hidden: bool = False, offset: int = Query(0, ge=0),
                         user: Principal = Depends(principal), session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
        async with access(thread_id, user, session) as (files, state):
            response.headers["Cache-Control"] = "private, no-store"
            page = await asyncio.to_thread(files.list, _browser_path(path), state, hidden=hidden, offset=offset)
            page["entries"] = [entry for entry in page["entries"] if _browser_visible(entry["path"])]
            return page

    @router.get("/file")
    async def read_file(thread_id: str, response: Response, path: str = Query(..., max_length=4096),
                        user: Principal = Depends(principal), session: AsyncSession = Depends(get_db_session)) -> dict[str, Any]:
        async with access(thread_id, user, session) as (files, state):
            response.headers["Cache-Control"] = "private, no-store"
            return await asyncio.to_thread(files.preview, _browser_path(path), state)

    @router.get("/download")
    async def download_file(thread_id: str, path: str = Query(..., max_length=4096),
                            user: Principal = Depends(principal), session: AsyncSession = Depends(get_db_session)) -> Response:
        async with access(thread_id, user, session) as (files, state):
            # Bound transfers independently of the much smaller editor limit.
            data = await asyncio.to_thread(files.read, _browser_path(path), state, limit=100 * 1024 * 1024 + 1)
            if len(data) > 100 * 1024 * 1024:
                raise HTTPException(413, "File exceeds the 100 MiB download limit")
            return Response(data, media_type="application/octet-stream", headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(PurePosixPath(path).name)}",
                "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
            })

    @router.put("/file")
    async def save_file(thread_id: str, payload: SaveFileInput, request: Request,
                        user: Principal = Depends(principal), session: AsyncSession = Depends(get_db_session)) -> dict[str, str]:
        # Python clients send no Origin. Browser writes must come from one of
        # the application's explicitly configured public origins, even when a
        # reverse proxy makes the UI and API appear same-origin to the browser.
        origin = request.headers.get("origin")
        if (origin is not None and origin not in browser_origins) or (
            origin is None and "sec-fetch-site" in request.headers
        ):
            raise HTTPException(403, {"code": "origin_not_allowed", "message": "Browser origin is not allowed"})
        async with access(thread_id, user, session, write=True) as (files, _state):
            revision = await asyncio.to_thread(files.save, _browser_path(payload.path), payload.text, payload.revision)
            return {"path": payload.path, "revision": revision}

    return router
