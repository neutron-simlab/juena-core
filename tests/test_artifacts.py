"""Artifact validation, ownership, persistence, and download metadata."""

import io
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from juena_core.artifacts import (
    MAX_UNDELIVERED_NOTICES_PER_TURN,
    ArtifactStore,
    set_artifact_store_for_tests,
)
from juena_core.artifacts import (
    MAX_IMAGE_ARTIFACTS_PER_TURN,
    MAX_RECORD_ARTIFACTS_PER_TURN,
)
from juena_core.server.api.endpoints import build_api_router
from juena_core.server.identity import Principal


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", tmp_path / "audit.jsonl")


def test_stores_inline_png_and_downloadable_script_for_owner(tmp_path) -> None:
    store = _store(tmp_path)
    plot = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="sine plot.png",
        content=_png(),
    )
    script = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="run-1",
        filename="analysis.py",
        content=b"print('done')\n",
    )

    assert (plot.kind, plot.mime_type, plot.width, plot.height) == (
        "image",
        "image/png",
        12,
        8,
    )
    assert (script.kind, script.mime_type) == ("file", "text/x-python")
    assert store.get("user-a", script.artifact_id)[1] == b"print('done')\n"
    assert store.get("user-b", script.artifact_id) is None

    refs = store.claim_for_message("user-a", "thread-a")
    assert [item["artifact_id"] for item in refs] == [plot.artifact_id, script.artifact_id]
    assert store.claim_for_message("user-a", "thread-a") == []


def test_artifact_group_survives_persistence_and_message_delivery(tmp_path) -> None:
    store = _store(tmp_path)
    artifact = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id="graph-run-1",
        filename="result.dat",
        content=b"result\n",
        group_id="simulation-run-1",
        group_label="Simulation proof · narrow",
    )

    assert artifact.group_id == "simulation-run-1"
    assert artifact.group_label == "Simulation proof · narrow"
    assert store.claim_for_message("user-a", "thread-a")[0] == artifact.model_dump(
        mode="json"
    )


def test_execution_records_do_not_consume_the_result_budget(tmp_path) -> None:
    """The production failure, in one assertion.

    A long debugging session writes two records per approved command. Before
    they were budgeted apart, six commands filled a shared 12-file allowance and
    the plot that was the point of the turn arrived to find no room.
    """
    store = _store(tmp_path)
    for index in range(MAX_RECORD_ARTIFACTS_PER_TURN):
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename="generated-command.sh",
            content=f"echo {index}\n".encode(),
            category="record",
        )

    plot = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="fit.png",
        content=_png(),
    )

    assert plot.kind == "image"
    assert [ref.artifact_id for ref in store.peek_result_refs("user-a", "thread-a")] == [
        plot.artifact_id
    ]

    with pytest.raises(ValueError, match="execution records"):
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename="generated-command.sh",
            content=b"echo over\n",
            category="record",
        )


def test_image_budget_reports_its_reason(tmp_path) -> None:
    store = _store(tmp_path)
    for _ in range(MAX_IMAGE_ARTIFACTS_PER_TURN):
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename="plot.png",
            content=_png(),
        )

    with pytest.raises(ValueError, match=f"maximum of {MAX_IMAGE_ARTIFACTS_PER_TURN} plots"):
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename="one-too-many.png",
            content=_png(),
        )


def test_byte_budget_rejects_an_oversized_turn(tmp_path, monkeypatch) -> None:
    """Bytes are the backstop now that downloadable files are not counted."""
    monkeypatch.setattr("juena_core.artifacts.MAX_ARTIFACT_BYTES_PER_TURN", 10)
    store = _store(tmp_path)
    store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="first.txt",
        content=b"123456",
    )

    with pytest.raises(ValueError, match="MB file budget"):
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename="overflow.txt",
            content=b"12345",
        )


def test_downloadable_files_are_not_capped_by_count(tmp_path) -> None:
    """A download button costs almost nothing, so only bytes constrain these."""
    store = _store(tmp_path)

    refs = [
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename=f"series-{index}.csv",
            content=b"q,I\n0.1,42\n",
        )
        for index in range(30)
    ]

    assert len({ref.artifact_id for ref in refs}) == 30


def test_scientific_dat_output_is_validated_as_utf8_text(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="jscatter-fit.dat",
        content=b"q I eI\n0.1 42 0.2\n",
    )

    assert (ref.kind, ref.mime_type) == ("file", "text/plain")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        store.register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename="invalid.dat",
            content=b"\xff\xfe",
        )


def test_begin_turn_discards_a_stale_pending_window(tmp_path) -> None:
    """A cancelled stream never reaches the drain, so the next turn clears it."""
    store = _store(tmp_path)
    store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="orphan.png",
        content=_png(),
    )

    store.begin_turn("user-a", "thread-a")

    assert store.claim_for_message("user-a", "thread-a") == []
    assert store.drain_events("user-a", "thread-a") == []


def test_undelivered_notice_queue_is_bounded(tmp_path) -> None:
    store = _store(tmp_path)
    overflow = 3
    for index in range(MAX_UNDELIVERED_NOTICES_PER_TURN + overflow):
        store.note_undelivered(
            "user-a",
            "thread-a",
            f"result-{index}.txt",
            "not allowed",
        )

    preview = store.peek_undelivered("user-a", "thread-a")
    notices = store.drain_undelivered("user-a", "thread-a")

    assert preview == notices
    assert len(notices) == MAX_UNDELIVERED_NOTICES_PER_TURN + 1
    assert notices[-1] == (
        f"{overflow} additional output files",
        "Details omitted because this report is bounded",
    )
    assert store.drain_undelivered("user-a", "thread-a") == []


@pytest.mark.parametrize("filename", ["result.html", "plot.svg", "program.exe", "data.zip"])
def test_rejects_active_or_unapproved_download_types(tmp_path, filename: str) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _store(tmp_path).register_artifact(
            user_id="user-a",
            thread_id="thread-a",
            run_id=None,
            filename=filename,
            content=b"not trusted",
        )


def test_only_inline_plot_event_contains_bytes(tmp_path) -> None:
    store = _store(tmp_path)
    plot = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="plot.png",
        content=_png(),
    )
    script = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="analysis.py",
        content=b"print('done')\n",
    )

    plot_event, script_event = store.drain_events("user-a", "thread-a")
    assert plot_event["artifact_id"] == plot.artifact_id
    assert plot_event["content_base64"]
    assert script_event["artifact_id"] == script.artifact_id
    assert "content_base64" not in script_event
    persisted = store.claim_for_message("user-a", "thread-a")
    assert all("content_base64" not in item for item in persisted)


def test_thread_deletion_removes_files(tmp_path) -> None:
    store = _store(tmp_path)
    ref = store.register_artifact(
        user_id="user-a",
        thread_id="thread-a",
        run_id=None,
        filename="result.txt",
        content=b"result\n",
    )

    store.delete_thread("user-a", "thread-a")

    assert store.get("user-a", ref.artifact_id) is None


def _principal(user_id: UUID, name: str) -> Principal:
    return Principal(
        id=user_id,
        subject=name,
        issuer="test-issuer",
        email=None,
        display_name=name,
    )


def test_artifact_endpoint_enforces_authenticated_owner(tmp_path) -> None:
    """The download route is the only way a stored file leaves the server.

    ``get_artifact`` is a closure inside ``build_api_router`` now, so this
    drives the real route rather than the function: the store is asked for
    ``str(user.id)``, and a caller who is not that user must not be able to
    tell an artifact they do not own from one that does not exist.
    """

    owner_id = uuid4()
    other_id = uuid4()
    caller = {"principal": _principal(owner_id, "owner")}
    store = _store(tmp_path)
    set_artifact_store_for_tests(store)
    ref = store.register_artifact(
        user_id=str(owner_id),
        thread_id="thread-a",
        run_id=None,
        filename="analysis.py",
        content=b"print('ok')\n",
    )

    app = FastAPI()
    app.include_router(build_api_router(lambda: caller["principal"]))

    try:
        with TestClient(app) as client:
            owned = client.get(f"/artifacts/{ref.artifact_id}")
            caller["principal"] = _principal(other_id, "other")
            refused = client.get(f"/artifacts/{ref.artifact_id}")
    finally:
        set_artifact_store_for_tests(None)

    assert owned.status_code == 200
    assert owned.content == b"print('ok')\n"
    assert owned.headers["content-disposition"].startswith("attachment;")
    assert refused.status_code == 404
