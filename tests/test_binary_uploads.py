"""Binary uploads preserve bytes without publishing partial batches or graph blobs."""

import io
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from juena_core.filesystem.service import ConversationFilesystem
from juena_core.filesystem.storage import WorkspaceStore
from juena_core.schema.upload_limits import UploadPolicy, validate_attachments
from juena_core.server.agent.input_handler import AgentInputHandler
from juena_core.server.chat.inputs import normalize_uploaded_attachments, prepare_code_chat_turn_inputs


TIFF = b"II\x2a\x00\x08\x00\x00\x00\xff\x00"
POLICY = UploadPolicy(binary_suffixes=frozenset({".tif", ".tiff"}), max_binary_bytes=64, max_total_bytes=96)


@pytest.fixture
def files(tmp_path):
    fs = ConversationFilesystem(WorkspaceStore(tmp_path / "files"), identity_secret="test-secret-that-is-at-least-32-characters")
    result = fs.for_thread("alice", "conversation")
    result.ensure()
    return result


def upload(name, content, size=None):
    return UploadFile(io.BytesIO(content), filename=name, size=size)


@pytest.mark.asyncio
async def test_mixed_uploads_preserve_bytes_and_metadata_without_binary_graph_state(files):
    async def state(**kwargs):
        return SimpleNamespace(values={"files": {}})
    source = [upload("../../detector.TIFF", TIFF), upload("curve.DAT", b"\xef\xbb\xbf1 2\r\n")]
    prepared = await prepare_code_chat_turn_inputs(
        SimpleNamespace(aget_state=state), {}, "inspect", source,
        closing_note="Inspect TIFFs", thread_files=files, upload_policy=POLICY,
    )
    assert prepared.attachments[0].staged_path == "/workspace/uploads/detector.TIFF"
    assert prepared.attachments[0].char_count is None
    assert prepared.attachments[0].size_bytes == len(TIFF)
    assert prepared.attachments[0].media_type == "image/tiff"
    assert files.read(prepared.attachments[0].staged_path, {}) == TIFF
    assert files.read(prepared.attachments[1].staged_path, {}) == b"\xef\xbb\xbf1 2\r\n"
    assert "None chars" not in prepared.message_override
    assert all(not p.startswith(("/inputs/uploads/", "/workspace/uploads/")) for p in prepared.files_update)
    assert prepared.workspace_files == prepared.files_update
    human = AgentInputHandler.build_input("inspect", display_attachments=prepared.attachments)["messages"][0]
    assert human.additional_kwargs["juena_attachments"][0]["size_bytes"] == len(TIFF)
    assert all(u.file.closed for u in source)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [upload("fake.tif", b"not tiff"), upload("binary.dat", b"\xff\x00")])
async def test_failed_validation_publishes_nothing_and_closes_every_upload(files, bad):
    source = [upload("good.tif", TIFF), bad, upload("remaining.txt", b"text")]
    with pytest.raises(ValueError):
        await normalize_uploaded_attachments(source, thread_files=files, upload_policy=POLICY)
    assert not files.path("/workspace/uploads").exists()
    assert all(u.file.closed for u in source)


@pytest.mark.asyncio
async def test_quota_failure_rolls_back_batch_preserving_existing_files(files):
    files.add_upload("keep.dat", b"old")
    files.store.limit_bytes = 20
    with pytest.raises(ValueError, match="quota"):
        await normalize_uploaded_attachments(
            [upload("a.tif", TIFF), upload("b.tif", TIFF)], thread_files=files, upload_policy=POLICY,
        )
    assert [p.name for p in files.path("/workspace/uploads").iterdir()] == ["keep.dat"]
    assert files.read("/workspace/uploads/keep.dat", {}) == b"old"
    files.add_upload("later.dat", b"ok")


@pytest.mark.asyncio
async def test_repeated_upload_names_are_deduplicated_across_turns(files):
    for expected in ("sample.tif", "sample_2.tif"):
        result, updates = await normalize_uploaded_attachments([upload("sample.tif", TIFF)], thread_files=files, upload_policy=POLICY)
        assert result[0].staged_path == f"/workspace/uploads/{expected}"
        assert updates == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("known_size", [True, False])
async def test_actual_file_and_batch_boundaries_are_enforced(files, known_size):
    for length in (63, 64):
        await normalize_uploaded_attachments([upload("ok.tif", TIFF + b"x" * (length - len(TIFF)), length if known_size else None)], thread_files=files, upload_policy=POLICY)
    with pytest.raises(ValueError, match="per-file"):
        await normalize_uploaded_attachments([upload("large.tif", TIFF + b"x" * 55, 65 if known_size else None)], thread_files=files, upload_policy=POLICY)
    for length in (48, 49):
        batch = [upload(f"{i}.tif", TIFF + b"x" * (length - len(TIFF)), length if known_size else None) for i in range(2)]
        if length == 48:
            await normalize_uploaded_attachments(batch, thread_files=files, upload_policy=POLICY)
        else:
            with pytest.raises(ValueError, match="combined"):
                await normalize_uploaded_attachments(batch, thread_files=files, upload_policy=POLICY)


@pytest.mark.asyncio
async def test_interrupted_upload_does_not_publish_previous_file(files):
    broken = upload("broken.tif", TIFF)
    async def fail(size):
        raise OSError("interrupted upload")
    broken.read = fail
    first = upload("first.tif", TIFF)
    with pytest.raises(OSError, match="interrupted"):
        await normalize_uploaded_attachments([first, broken], thread_files=files, upload_policy=POLICY)
    assert not files.path("/workspace/uploads").exists()
    assert first.file.closed and broken.file.closed


@pytest.mark.asyncio
async def test_dat_defaults_require_utf8_and_keep_text_limit():
    for name in ("curve.dat", "curve.DAT"):
        result, updates = await normalize_uploaded_attachments([upload(name, b"1 2\n")])
        assert result[0].char_count == 4
        assert len(updates) == 1
    with pytest.raises(ValueError, match="UTF-8"):
        await normalize_uploaded_attachments([upload("binary.dat", b"\xff\x00")])
    with pytest.raises(ValueError, match="per-file"):
        await normalize_uploaded_attachments([upload("large.dat", b"x" * (1024**2 + 1))])
    assert validate_attachments([("ok.dat", 1024**2)]) == []


@pytest.mark.asyncio
async def test_binary_uploads_require_opt_in_and_filesystem():
    with pytest.raises(ValueError, match="unsupported extension"):
        await normalize_uploaded_attachments([upload("image.tif", TIFF)])
    with pytest.raises(ValueError, match="filesystem"):
        await normalize_uploaded_attachments([upload("image.tif", TIFF)], upload_policy=POLICY)


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [b"MM\x00\x2a\x00\x00\x00\x08", b"II\x2b\x00\x08\x00\x00\x00" + b"\x00" * 8, b"MM\x00\x2b\x00\x08\x00\x00" + b"\x00" * 8])
async def test_tiff_header_variants(files, header):
    result, _ = await normalize_uploaded_attachments([upload("image.tif", header)], thread_files=files, upload_policy=POLICY)
    assert files.read(result[0].staged_path, {}) == header
