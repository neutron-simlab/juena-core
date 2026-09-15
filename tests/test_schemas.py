"""CP1 schema contracts."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from juena_core.schema.agents import AskUserSchema, SpecialistReport
from juena_core.schema.interrupts import (
    ArtifactRef,
    ClarificationResumeInput,
    ExecutionEvidence,
)
from juena_core.schema.llm_models import (
    BlabladorModelName,
    Provider,
    get_blablador_model_display_name,
    get_default_model_for_provider,
    get_models_for_provider,
)
from juena_core.schema.server import ChatMessage, CreateChatInput
from juena_core.schema.upload_limits import (
    DEFAULT_SUFFIXES,
    DEFAULT_TEXT_READABLE_FILE_TYPES,
    is_text_readable_filename,
    validate_attachments,
)


def test_model_catalogue_preserves_curated_blablador_models() -> None:
    models = get_models_for_provider(Provider.BLABLADOR)

    assert models == [model.value for model in BlabladorModelName]
    assert get_default_model_for_provider(Provider.BLABLADOR) == models[0]
    assert (
        get_blablador_model_display_name(BlabladorModelName.MINIMAX_M27.value)
        == "MiniMax-M2.7"
    )


def test_chat_message_container_defaults_are_not_shared() -> None:
    first = ChatMessage(type="ai", content="first")
    second = ChatMessage(type="ai", content="second")

    first.custom_data["answer"] = 42
    first.tool_calls.append({"name": "search", "args": {}, "id": None})

    assert second.custom_data == {}
    assert second.tool_calls == []


def test_create_chat_requires_agent_identity() -> None:
    with pytest.raises(ValidationError):
        CreateChatInput()

    created = CreateChatInput(agent_id="vitess")
    assert created.agent_id == "vitess"


def test_specialist_report_strips_finding_and_forbids_extra_fields() -> None:
    report = SpecialistReport(status="completed", finding="  measured result  ")
    assert report.finding == "measured result"

    with pytest.raises(ValidationError):
        SpecialistReport(status="completed", finding="ok", invented=True)


def test_ask_user_strips_question() -> None:
    assert AskUserSchema(question="  Which sample?  ").question == "Which sample?"
    with pytest.raises(ValidationError):
        AskUserSchema(question="   ")


def test_execution_evidence_derives_attempt_and_success() -> None:
    completed = ExecutionEvidence(command="run", status="completed", exit_code=0)
    skipped = ExecutionEvidence(command="run", status="skipped")

    assert completed.attempted is True
    assert completed.succeeded is True
    assert skipped.attempted is False
    assert skipped.succeeded is False


def test_generic_interrupt_schemas_round_trip() -> None:
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        filename="result.dat",
        mime_type="text/plain",
        kind="file",
        size=12,
        caption="result",
        created_at=datetime(2026, 9, 15),
    )
    resume = ClarificationResumeInput(
        kind="clarification",
        thread_id="thread-1",
        interrupt_id="interrupt-1",
        answer="Use sample 42",
    )

    assert artifact.filename == "result.dat"
    assert resume.kind == "clarification"


def test_default_upload_policy_stays_available_to_the_ui() -> None:
    assert "txt" in DEFAULT_TEXT_READABLE_FILE_TYPES
    assert ".txt" in DEFAULT_SUFFIXES
    assert is_text_readable_filename("notes.TXT")
    assert not is_text_readable_filename("sample.h5")


def test_application_can_supply_binary_domain_extensions_and_limits() -> None:
    allowed = frozenset({".dat", ".h5"})

    assert is_text_readable_filename("sample.H5", allowed_suffixes=allowed)
    assert validate_attachments(
        [("sample.h5", 2_000), ("result.dat", None)],
        allowed_suffixes=allowed,
        max_bytes=2_000,
        max_files=2,
    ) == []
    errors = validate_attachments(
        [("sample.h5", 2_001), ("notes.txt", 10)],
        allowed_suffixes=allowed,
        max_bytes=2_000,
        max_files=1,
    )
    assert len(errors) == 3
