"""The browser and agent operate on the same durable conversation files."""

import hashlib
import io
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from deepagents.backends.utils import create_file_data
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from juena_core.filesystem import service
from juena_core.filesystem.api import build_filesystem_router
from juena_core.filesystem.backend import ConversationBackend
from juena_core.filesystem.service import ConversationFilesystem, MAX_EDIT_BYTES
from juena_core.filesystem.storage import RevisionConflict, WorkspaceBusy, WorkspaceStore
from juena_core.server.chat.inputs import prepare_code_chat_turn_inputs
from juena_core.server.chat.repository import ChatNotFoundError
from juena_core.server.database.connection import get_db_session


@pytest.fixture
def filesystem(tmp_path, monkeypatch):
    fs = ConversationFilesystem(WorkspaceStore(tmp_path / "files"), identity_secret="test-secret-that-is-at-least-32-characters")
    monkeypatch.setattr(service, "_filesystem", fs)
    return fs


@pytest.fixture
def files(filesystem):
    result = filesystem.for_thread("alice", "conversation")
    result.ensure()
    return result


def runtime_backend(monkeypatch, *, writable=True, root="/workspace", directories_only=False):
    runtime = SimpleNamespace(context={"user_id": "alice", "thread_id": "conversation"}, state={"files": {}})
    monkeypatch.setattr("juena_core.filesystem.backend.get_runtime", lambda: runtime)
    return ConversationBackend(writable=writable, root=root, directories_only=directories_only)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [b"value = 1\n", b"\xef\xbb\xbfvalue = 1\r\n"])
async def test_upload_browser_edit_and_agent_read_share_bytes(files, monkeypatch, content):
    class Graph:
        async def aget_state(self, **kwargs):
            return SimpleNamespace(values={"files": {}})

    prepared = await prepare_code_chat_turn_inputs(
        Graph(), {}, "Inspect this script", [UploadFile(io.BytesIO(content), filename="analysis.py")],
        closing_note="Use shared files", thread_files=files,
    )
    path = prepared.attachments[0].staged_path
    assert path == "/workspace/uploads/analysis.py"
    assert path not in prepared.files_update
    assert "/inputs/uploads/analysis.py" not in prepared.files_update
    assert path in prepared.message_override
    backend = runtime_backend(monkeypatch)
    assert backend.read("/uploads/analysis.py").file_data["content"].encode("utf-8") == content.replace(b"\r\n", b"\n")
    assert files.read(path, {}) == content
    preview = files.preview(path, {})
    files.save(path, "value = 2\n", preview["revision"])
    assert backend.read("/uploads/analysis.py").file_data["content"] == "value = 2\n"
    assert files.read(path, {}) == b"value = 2\n"


def test_nested_agent_file_is_visible_without_artifact_registration(files, monkeypatch):
    backend = runtime_backend(monkeypatch)
    assert backend.write("/scripts/nested/analysis.py", "print(42)").error is None
    assert files.list("/workspace/scripts/nested", {})["entries"][0]["path"] == "/workspace/scripts/nested/analysis.py"
    assert files.read("/workspace/scripts/nested/analysis.py", {}) == b"print(42)"
    assert backend.edit("/scripts/nested/analysis.py", "42", "43").error is None
    assert files.preview("/workspace/scripts/nested/analysis.py", {})["text"] == "print(43)"
    assert backend.grep("43", "/scripts/nested/analysis.py").matches == [
        {"path": "/scripts/nested/analysis.py", "line": 1, "text": "print(43)"}
    ]


def test_research_is_read_only_and_supervisor_can_only_discover(files, monkeypatch):
    files.add_upload("script.py", b"print(1)")
    research = runtime_backend(monkeypatch, writable=False)
    assert research.read("/uploads/script.py").file_data["content"] == "print(1)"
    assert research.write("/uploads/new.py", "x").error
    assert research.edit("/uploads/script.py", "1", "2").error
    supervisor = runtime_backend(monkeypatch, directories_only=True)
    assert supervisor.ls("/uploads").entries[0]["path"] == "/uploads/script.py"
    assert supervisor.read("/uploads/script.py").error
    assert supervisor.download_files(["/uploads/script.py"])[0].error


def test_saved_revision_prevents_overwrite_and_limits_do_not_truncate(files):
    path = files.add_upload("script.py", b"initial")
    preview = files.preview(path, {})
    files.save(path, "changed", preview["revision"])
    with pytest.raises(RevisionConflict):
        files.save(path, "stale draft", preview["revision"])
    with pytest.raises(ValueError, match="1 MiB"):
        files.save(path, "x" * (MAX_EDIT_BYTES + 1), hashlib.sha256(b"changed").hexdigest())
    assert files.read(path, {}) == b"changed"


def test_migration_preserves_conflicts_and_later_edits(files):
    files.add_upload("script.py", b"already here")
    legacy = {"/inputs/uploads/script.py": create_file_data("legacy")}
    aliases = files.migrate(legacy)
    path = aliases["/inputs/uploads/script.py"]
    assert path == "/workspace/uploads/script_legacy2.py"
    preview = files.preview(path, {})
    files.save(path, "user edit", preview["revision"])
    assert files.migrate(legacy) == aliases
    assert files.read("/inputs/uploads/script.py", {}) == b"user edit"
    assert files.read("/workspace/uploads/script.py", {}) == b"already here"


def test_names_are_deduplicated_and_ownership_isolated(filesystem, files):
    assert files.add_upload("data.csv", b"a") == "/workspace/uploads/data.csv"
    assert files.add_upload("data.csv", b"b") == "/workspace/uploads/data_2.csv"
    for owner, thread in (("bob", "conversation"), ("alice", "other")):
        other = filesystem.for_thread(owner, thread)
        other.ensure()
        with pytest.raises(FileNotFoundError):
            other.read("/workspace/uploads/data.csv", {})


@pytest.mark.parametrize("path", ["/workspace/../secret", "/etc/passwd", "//workspace/file", "/workspace/evil\\name", "/workspace/evil\x00name"])
def test_invalid_paths_cannot_be_read(files, path):
    with pytest.raises((ValueError, PermissionError)):
        files.read(path, {})


def test_symlinks_hardlinks_and_special_files_are_not_exposed(files, tmp_path):
    outside = tmp_path / "secret"
    outside.write_text("private")
    root = files.path("/workspace")
    (root / "link").symlink_to(outside)
    (root / "linked_dir").symlink_to(tmp_path, target_is_directory=True)
    os.link(outside, root / "hardlink")
    os.mkfifo(root / "pipe")
    assert not files.list("/workspace", {})["entries"]
    for path in ("link", "linked_dir/secret", "hardlink", "pipe"):
        with pytest.raises(ValueError):
            files.read(f"/workspace/{path}", {})


def test_only_scratch_expires_and_active_or_paused_work_is_protected(files):
    path = files.add_upload("permanent.py", b"permanent")
    scratch = files.path("/workspace/.scratch/temp.py")
    scratch.write_text("temporary")
    root = files.store.workspace_root(files.workspace_id)
    os.utime(root, (1, 1))
    with files.guard(operation=True, shared=True):
        assert files.store.cleanup_expired(set()) == 0
    files.store.mark_paused(files.workspace_id, True)
    os.utime(root, (1, 1))
    assert files.store.cleanup_expired(set()) == 0
    files.store.mark_paused(files.workspace_id, False)
    os.utime(root, (1, 1))
    assert files.store.cleanup_expired(set()) == 1
    assert not scratch.exists()
    assert files.read(path, {}) == b"permanent"
    files.ensure()
    assert files.path("/workspace/.scratch").is_dir()


def test_cross_process_file_locks_protect_execution_and_deletion(files):
    # Separate open descriptors contend even inside one process, just as API/worker do.
    with files.guard(execution=True):
        with pytest.raises(WorkspaceBusy):
            with files.guard():
                pass
    with files.guard(operation=True, shared=True):
        with pytest.raises(WorkspaceBusy):
            with files.guard(operation=True):
                pass


def test_api_writes_preserve_worker_group_access_under_default_umask(files):
    previous = os.umask(0o022)
    try:
        with files.guard():
            path = files.add_upload("editable.py", b"data")
        lock = files.store.root / ".locks" / f"{files.workspace_id}.files"
        assert files.path(path).stat().st_mode & 0o770 == 0o660
        assert lock.stat().st_mode & 0o770 == 0o660
    finally:
        os.umask(previous)


def test_nested_outputs_preserve_relative_paths(files):
    since = datetime.now(UTC)
    files.store.put(files.workspace_id, "/workspace/outputs/a/result.csv", b"a")
    files.store.put(files.workspace_id, "/workspace/outputs/b/result.csv", b"b")
    assert files.store.artifacts_since(files.workspace_id, since).accepted == ("a/result.csv", "b/result.csv")


def test_state_routes_expose_only_inputs_and_findings(files):
    state = {"/inputs/code.py": create_file_data("code"), "/findings/topic.md": create_file_data("notes"),
             "/memories/private": create_file_data("secret")}
    assert files.list("/findings", state)["entries"][0]["writable"] is False
    assert files.preview("/findings/topic.md", state)["text"] == "notes"
    with pytest.raises(PermissionError):
        files.read("/memories/private", state)
    with pytest.raises(PermissionError):
        files.save("/findings/topic.md", "overwrite", "a" * 64)


@pytest.fixture
def api(filesystem, files, monkeypatch):
    from juena_core.filesystem import api as module
    async def owner(session, user_id, thread_id, *, agent_id):
        if user_id != "alice" or thread_id != "conversation":
            raise ChatNotFoundError(thread_id)
        return SimpleNamespace(agent_id="test")
    async def checkpoint(config):
        return {"channel_values": {"files": {"/findings/note.md": create_file_data("a finding")}}}
    async def agent(agent_id):
        return object()
    async def pending(*args):
        return None
    async def session():
        yield object()
    monkeypatch.setattr(module, "get_owned_chat", owner)
    monkeypatch.setattr(module, "get_checkpointer", lambda: SimpleNamespace(aget=checkpoint))
    monkeypatch.setattr(module, "get_agent", agent)
    monkeypatch.setattr(module, "first_pending_interrupt", pending)
    app = FastAPI()
    app.include_router(build_filesystem_router(lambda: SimpleNamespace(id="alice"), filesystem, browser_origins=("http://localhost:9501",)))
    app.dependency_overrides[get_db_session] = session
    return TestClient(app)


def test_api_reads_saves_conflicts_and_downloads(api, files):
    path = files.add_upload("script.py", b"a = 1")
    url = "/threads/conversation/filesystem"
    assert api.get(url, params={"path": "/workspace/uploads"}).json()["entries"][0]["path"] == path
    preview = api.get(url + "/file", params={"path": path}).json()
    payload = {"path": path, "text": "a = 2", "revision": preview["revision"]}
    assert api.put(url + "/file", json=payload).status_code == 200
    assert api.put(url + "/file", json=payload).status_code == 409
    download = api.get(url + "/download", params={"path": path})
    assert download.content == b"a = 2"
    assert download.headers["x-content-type-options"] == "nosniff"
    assert api.get("/threads/other/filesystem").status_code == 404
    assert api.get(url + "/file", params={"path": "/etc/passwd"}).status_code == 403


def test_api_refuses_saves_during_execution_and_pending_approval(api, files, monkeypatch):
    path = files.add_upload("script.py", b"a = 1")
    payload = {"path": path, "text": "changed", "revision": files.preview(path, {})["revision"]}
    with files.guard(operation=True, shared=True):
        assert api.put("/threads/conversation/filesystem/file", json=payload).status_code == 409
    async def paused(*args):
        return object()
    monkeypatch.setattr("juena_core.filesystem.api.first_pending_interrupt", paused)
    assert api.put("/threads/conversation/filesystem/file", json=payload).status_code == 409
    assert files.read(path, {}) == b"a = 1"


def test_queued_execution_survives_loss_of_api_process_lock(api, filesystem, files, monkeypatch):
    path = files.add_upload("script.py", b"original")
    monkeypatch.setattr(filesystem, "active_workspaces", lambda: {files.workspace_id})
    result = api.put("/threads/conversation/filesystem/file", json={
        "path": path, "text": "changed", "revision": files.preview(path, {})["revision"],
    })
    assert result.status_code == 409
    assert files.read(path, {}) == b"original"


@pytest.mark.parametrize("directory", ["/workspace/conversation_history", "/workspace/.scratch/conversation_history"])
def test_browser_excludes_internal_history_even_with_temporary_files(api, files, directory):
    from pathlib import PurePosixPath
    path = directory + "/session.md"
    files.store.put(files.workspace_id, path, b"internal history")
    base = "/threads/conversation/filesystem"
    listing = api.get(base, params={"path": str(PurePosixPath(directory).parent), "hidden": True}).json()
    assert directory not in {entry["path"] for entry in listing["entries"]}
    assert api.get(base, params={"path": directory, "hidden": True}).status_code == 403
    for suffix in ("/file", "/download"):
        assert api.get(base + suffix, params={"path": path}).status_code == 403
    assert api.put(base + "/file", json={
        "path": path, "text": "overwrite", "revision": files.preview(path, {})["revision"],
    }).status_code == 403


@pytest.mark.asyncio
async def test_actual_graph_runtime_reads_workspace_and_state_inputs(files):
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from juena_core.server.agent.runtime_model_middleware import RuntimeModelContext

    class State(TypedDict):
        files: dict
        observed: list[str]

    files.add_upload("code.py", b"shared")
    workspace = ConversationBackend()
    inputs = ConversationBackend(root="/inputs")

    async def read(state):
        one = await workspace.aread("/uploads/code.py")
        two = await inputs.aread("/current_message.txt")
        assert one.error is None, one.error
        assert two.error is None, two.error
        return {"observed": [one.file_data["content"], two.file_data["content"]]}

    graph = StateGraph(State, context_schema=RuntimeModelContext)
    graph.add_node("read", read)
    graph.add_edge(START, "read")
    graph.add_edge("read", END)
    result = await graph.compile().ainvoke(
        {"files": {"/inputs/current_message.txt": create_file_data("pasted")}},
        context=RuntimeModelContext(user_id="alice", thread_id="conversation", provider="test", model="test"),
    )
    assert result["observed"] == ["shared", "pasted"]


def test_deleted_workspace_cannot_be_recreated_by_late_worker(files):
    from juena_core.filesystem.storage import WorkspaceOwnershipError
    files.add_upload("code.py", b"code")
    with files.guard(operation=True), files.guard():
        files.store.delete(files.workspace_id, files.tenant_id)
    with pytest.raises(WorkspaceOwnershipError, match="deleted"):
        files.ensure()


def test_api_conflicts_have_distinct_codes_and_reads_never_wait_for_execution(api, files):
    path = files.add_upload("code.py", b"before")
    url = "/threads/conversation/filesystem"
    payload = {"path": path, "text": "after", "revision": files.preview(path, {})["revision"]}
    with files.guard(execution=True):
        response = api.get(url, params={"path": "/workspace"})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "workspace_busy"
    assert api.put(url + "/file", json=payload).status_code == 200
    response = api.put(url + "/file", json=payload)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "revision_conflict"
    assert api.get(url + "/file", params={"path": "/workspace/missing"}).json()["detail"]["code"] == "file_not_found"
    assert api.get("/threads/other/filesystem").json()["detail"]["code"] == "conversation_not_found"


@pytest.mark.parametrize("headers,allowed", [
    ({"Origin": "http://localhost:9501"}, True),
    ({"Origin": "https://untrusted.example"}, False),
    ({"Origin": "null"}, False),
    ({"Sec-Fetch-Site": "same-origin"}, False),
    ({}, True),  # Authenticated Python client.
])
def test_browser_saves_require_an_allowed_origin(api, files, headers, allowed):
    path = files.add_upload("code.py", b"before")
    response = api.put("/threads/conversation/filesystem/file", headers=headers, json={
        "path": path, "text": "after", "revision": files.preview(path, {})["revision"],
    })
    assert response.status_code == (200 if allowed else 403)
    assert files.read(path, {}) == (b"after" if allowed else b"before")


def test_parallel_agent_reads_and_browser_listing_share_locks(api, files, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event
    files.add_upload("code.py", b"one\ntwo\n")
    backend = runtime_backend(monkeypatch)
    entered, release = Barrier(5), Event()
    original = files.__class__.read
    def held_read(self, *args, **kwargs):
        entered.wait(timeout=3)
        assert release.wait(timeout=3)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(files.__class__, "read", held_read)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(backend.read, "/uploads/code.py") for _ in range(4)]
        try:
            entered.wait(timeout=3)
            response = api.get("/threads/conversation/filesystem", params={"path": "/workspace/uploads"})
            assert response.status_code == 200, response.text
        finally:
            release.set()
        assert all(future.result().error is None for future in futures)


@pytest.mark.asyncio
async def test_short_write_contention_waits_without_blocking_event_loop(files):
    import asyncio
    async def reader():
        async with files.aguard(shared=True):
            return files.read("/workspace/uploads/code.py", {})
    files.add_upload("code.py", b"original")
    with files.guard():
        task = asyncio.create_task(reader())
        await asyncio.sleep(0.05)
        assert not task.done()
    assert await asyncio.wait_for(task, 1) == b"original"
    # A cancelled waiter must never acquire or retain a lock in another thread.
    with files.guard():
        task = asyncio.create_task(reader())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    with files.guard(timeout=0):
        pass


@pytest.mark.asyncio
async def test_missing_legacy_alias_does_not_break_listing_migration_or_staging(files):
    from juena_core.sandbox.runtime import stage_runtime_inputs
    old = "/inputs/uploads/legacy.py"
    state = {old: create_file_data("legacy"), "/inputs/message.txt": create_file_data("hello")}
    target = files.migrate(state)[old]
    files.path(target).unlink()
    assert files.list("/inputs/uploads", state)["entries"] == []
    with pytest.raises(FileNotFoundError):
        files.read(old, state)
    # Retain the migration record: deletion must not resurrect the old checkpoint copy.
    assert files.migrate(state)[old] == target
    assert not files.path(target).exists()
    await stage_runtime_inputs(user_id="alice", thread_id="conversation", files=state)
    assert (files.store.inputs_dir(files.workspace_id) / "message.txt").read_bytes() == b"hello"
    assert not (files.store.inputs_dir(files.workspace_id) / "uploads/legacy.py").exists()


def test_input_alias_listing_uses_metadata_without_loading_contents(files, monkeypatch):
    state = {"/inputs/uploads/data.txt": create_file_data("large contents")}
    files.migrate(state)
    monkeypatch.setattr(files, "read", lambda *a, **kw: pytest.fail("Listing loaded file contents"))
    assert files.list("/inputs/uploads", state)["entries"][0]["size"] == len("large contents")


def test_windows_multiline_edit_and_text_view_use_universal_newlines(files, monkeypatch):
    backend = runtime_backend(monkeypatch)
    path = files.add_upload("windows.py", b"one\r\ntwo\r\nthree\r\n")
    assert backend.read("/uploads/windows.py").file_data["content"] == "one\ntwo\nthree\n"
    assert files.read(path, {}) == b"one\r\ntwo\r\nthree\r\n"
    assert backend.edit("/uploads/windows.py", "one\ntwo", "four\nfive").error is None
    assert files.read(path, {}) == b"four\nfive\nthree\n"


def test_cleanup_and_delete_repair_readonly_directories(files):
    scratch = files.path("/workspace/.scratch/locked")
    scratch.mkdir()
    (scratch / "temp").write_text("disposable")
    scratch.chmod(0o500)
    root = files.store.workspace_root(files.workspace_id)
    os.utime(root, (1, 1))
    assert files.store.cleanup_expired(set()) == 1
    durable = files.path("/workspace/locked")
    durable.mkdir()
    (durable / "file").write_text("durable")
    durable.chmod(0o500)
    files.store.delete(files.workspace_id, files.tenant_id)
    assert not root.exists()
    assert files.store.deleted(files.workspace_id)


def test_cleanup_continues_after_one_workspace_fails(filesystem, files, monkeypatch):
    other = filesystem.for_thread("alice", "other")
    other.ensure()
    for entry in (files, other):
        entry.path("/workspace/.scratch/temp").write_text("temp")
        os.utime(entry.store.workspace_root(entry.workspace_id), (1, 1))
    original = files.store._remove_tree
    def remove(root):
        if files.workspace_id in root.parts:
            raise PermissionError("cannot remove this workspace")
        original(root)
    monkeypatch.setattr(files.store, "_remove_tree", remove)
    assert files.store.cleanup_expired(set()) == 1
    assert not other.path("/workspace/.scratch").exists()
    assert files.path("/workspace/.scratch/temp").exists()


def test_failed_delete_does_not_leave_live_chat_tombstoned(files, monkeypatch):
    path = files.add_upload("keep.txt", b"keep")
    def fail(root):
        raise PermissionError("unrecoverable filesystem error")
    monkeypatch.setattr(files.store, "_remove_tree", fail)
    with pytest.raises(PermissionError):
        files.store.delete(files.workspace_id, files.tenant_id)
    assert not files.store.deleted(files.workspace_id)
    files.ensure()
    assert files.read(path, {}) == b"keep"


@pytest.mark.parametrize("secret", ["", "too short"])
def test_filesystem_requires_stable_strong_identity_even_without_execution(tmp_path, secret):
    with pytest.raises(ValueError, match="at least 32"):
        ConversationFilesystem(WorkspaceStore(tmp_path), identity_secret=secret)


def test_api_permission_failures_have_non_authentication_codes(api, files):
    base = "/threads/conversation/filesystem"
    path = files.add_upload("code.py", b"before")
    payload = {"path": path, "text": "after", "revision": files.preview(path, {})["revision"]}
    denied = api.put(base + "/file", headers={"Origin": "https://untrusted.example"}, json=payload)
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "origin_not_allowed"
    payload["path"] = "/inputs/message.txt"
    assert api.put(base + "/file", json=payload).json()["detail"]["code"] == "path_read_only"
    files.store.delete(files.workspace_id, files.tenant_id)
    assert api.get(base).json()["detail"]["code"] == "workspace_deleted"


def test_quota_accounting_reuses_total_and_recounts_after_external_execution(files, monkeypatch):
    from unittest.mock import Mock
    measure = Mock(wraps=files.store._directory_size)
    monkeypatch.setattr(files.store, "_directory_size", measure)
    files.store.limit_bytes = 20
    for i in range(5):
        files.store.put(files.workspace_id, f"/workspace/{i}.txt", b"a")
    assert measure.call_count == 1
    files.store.invalidate_usage(files.workspace_id)
    files.path("/workspace/from-command.txt").write_bytes(b"x" * 15)
    with pytest.raises(ValueError, match="quota"):
        files.store.put(files.workspace_id, "/workspace/extra.txt", b"a")
    assert measure.call_count == 2
    files.store.put(files.workspace_id, "/workspace/0.txt", b"")
    assert files.read("/workspace/from-command.txt", {}) == b"x" * 15


def test_save_waits_for_preview_without_confusing_it_with_an_agent_turn(api, files, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import sleep
    path = files.add_upload("code.py", b"before")
    payload = {"path": path, "text": "after", "revision": files.preview(path, {})["revision"]}
    entered, release = Event(), Event()
    original = files.__class__.preview
    def held(self, *args):
        entered.set()
        assert release.wait(3)
        return original(self, *args)
    monkeypatch.setattr(files.__class__, "preview", held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(api.get, "/threads/conversation/filesystem/file", params={"path": path})
        try:
            assert entered.wait(3)
            save = pool.submit(api.put, "/threads/conversation/filesystem/file", json=payload)
            sleep(0.1)
            assert not save.done()
        finally:
            release.set()
        assert read.result().status_code == 200
        result = save.result()
        assert result.status_code == 200, result.text
