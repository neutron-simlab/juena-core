"""CP2 contracts for shared specialist and supervisor assembly."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from juena_core.agents import specialist_runtime
from juena_core.agents.backends import build_supervisor_backend


def _runtime_settings(**overrides):  # noqa: ANN003, ANN202
    values = {
        "EXECUTE_TIMEOUT_SECONDS": 600,
        "FALLBACK_PROVIDER": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _specialist() -> dict:
    return {
        "name": "test-specialist",
        "description": "Test one bounded objective.",
        "runnable": RunnableLambda(lambda state: state),
    }


def _summarizer_model() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="summary")])


def _names(items) -> list[str]:  # noqa: ANN001
    return [type(item).__name__ for item in items]


def test_resilience_middleware_order_and_pinned_limits() -> None:
    stack = specialist_runtime.resilience_middleware(
        [object()], model_call_limit=11, tool_call_limit=22
    )

    assert _names(stack) == [
        "ModelFallbackMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    ]
    assert stack[1].max_retries == specialist_runtime.MODEL_MAX_RETRIES == 3
    assert stack[2].max_retries == specialist_runtime.TOOL_MAX_RETRIES == 2
    assert stack[-1].tool_name == "ask_user"


def test_supervisor_stack_has_one_exact_extra_splice_point() -> None:
    class FirstExtra:
        pass

    class SecondExtra:
        pass

    stack = specialist_runtime.build_supervisor_middleware(
        backend=build_supervisor_backend(InMemoryStore()),
        summarizer_model=_summarizer_model(),
        fallback_models=[object()],
        subagents=[_specialist()],
        extra=[FirstExtra(), SecondExtra()],
    )

    assert _names(stack) == [
        "RepeatedToolCallMiddleware",
        "FilesystemMiddleware",
        "MemoryMiddleware",
        "SubAgentMiddleware",
        "_DeepAgentsSummarizationMiddleware",
        "PatchToolCallsMiddleware",
        "FirstExtra",
        "SecondExtra",
        "RuntimeModelMiddleware",
        "ModelFallbackMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
        "ToolCallLimitMiddleware",
    ]


def test_specialist_execution_and_approval_are_a_paired_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExecutionMiddleware:
        pass

    monkeypatch.setattr(
        specialist_runtime, "settings", lambda: _runtime_settings()
    )
    execution = ExecutionMiddleware()
    stack = specialist_runtime.build_specialist_middleware(
        backend=specialist_runtime.build_specialist_backend(),
        summarizer_model=_summarizer_model(),
        fallback_models=[],
        filesystem_tool_descriptions={},
        specialist_name="test-specialist",
        execution_middleware=[execution],
        interrupt_on={
            "execute": {
                "allowed_decisions": ["approve", "edit", "reject"],
                "description": "Review execution.",
            }
        },
    )

    assert stack[-2] is execution
    assert isinstance(stack[-1], HumanInTheLoopMiddleware)
    assert _names(stack[:3]) == [
        "SpecialistOutcomeMiddleware",
        "RepeatedToolCallMiddleware",
        "FilesystemMiddleware",
    ]


@pytest.mark.parametrize(
    ("execution_middleware", "interrupt_on"),
    [([object()], None), ((), {"execute": True})],
)
def test_partial_execution_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    execution_middleware,
    interrupt_on,
) -> None:
    monkeypatch.setattr(
        specialist_runtime, "settings", lambda: _runtime_settings()
    )

    with pytest.raises(ValueError, match="supplied together"):
        specialist_runtime.build_specialist_middleware(
            backend=specialist_runtime.build_specialist_backend(),
            summarizer_model=_summarizer_model(),
            fallback_models=[],
            filesystem_tool_descriptions={},
            specialist_name="test-specialist",
            execution_middleware=execution_middleware,
            interrupt_on=interrupt_on,
        )


def test_unattended_specialist_has_neither_questions_nor_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        specialist_runtime, "settings", lambda: _runtime_settings()
    )
    stack = specialist_runtime.build_specialist_middleware(
        backend=specialist_runtime.build_specialist_backend(),
        summarizer_model=_summarizer_model(),
        fallback_models=[],
        filesystem_tool_descriptions={},
        specialist_name="test-specialist",
        unattended=True,
    )

    assert not any(
        getattr(item, "tool_name", None) == "ask_user" for item in stack
    )
    with pytest.raises(ValueError, match="unattended"):
        specialist_runtime.build_specialist_middleware(
            backend=specialist_runtime.build_specialist_backend(),
            summarizer_model=_summarizer_model(),
            fallback_models=[],
            filesystem_tool_descriptions={},
            specialist_name="test-specialist",
            execution_middleware=[object()],
            interrupt_on={"execute": True},
            unattended=True,
        )


def _write_skill(skills_dir: Path, name: str = "plot-report") -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Produce a bounded artifact report.\n"
        "---\n\n"
        "# Plot report\n",
        encoding="utf-8",
    )


def test_only_valid_authored_skills_mount_the_skills_route(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    authored = tmp_path / "authored"
    _write_skill(authored)

    assert specialist_runtime.has_authored_skills(empty) is False
    assert specialist_runtime.has_authored_skills(authored) is True
    assert "/skills/" not in specialist_runtime.build_specialist_backend(
        skills_dir=empty
    ).routes
    assert "/skills/" in specialist_runtime.build_specialist_backend(
        skills_dir=authored
    ).routes


def test_fallback_model_uses_only_configured_available_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, str]] = []
    monkeypatch.setattr(
        specialist_runtime,
        "settings",
        lambda: _runtime_settings(FALLBACK_PROVIDER=" OPENAI "),
    )
    monkeypatch.setattr(
        specialist_runtime,
        "get_available_providers",
        lambda: {"openai": True},
    )
    monkeypatch.setattr(
        specialist_runtime, "get_default_model", lambda provider: "fallback-model"
    )
    monkeypatch.setattr(
        specialist_runtime,
        "build_chat_model",
        lambda **kwargs: built.append(kwargs) or object(),
    )

    assert len(specialist_runtime.build_fallback_models()) == 1
    assert built == [{"provider": "openai", "model": "fallback-model"}]
