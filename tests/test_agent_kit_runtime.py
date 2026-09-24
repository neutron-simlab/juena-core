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


def _filesystem_tool_names(stack: list) -> set[str]:
    middleware = next(
        item for item in stack if type(item).__name__ == "FilesystemMiddleware"
    )
    return {tool.name for tool in middleware.tools}


def _supervisor_stack(**overrides) -> list:
    return specialist_runtime.build_supervisor_middleware(
        backend=build_supervisor_backend(InMemoryStore()),
        summarizer_model=_summarizer_model(),
        fallback_models=[],
        subagents=[_specialist()],
        **overrides,
    )


def test_a_supervisor_gets_every_filesystem_tool_unless_it_asks_otherwise() -> None:
    """juena-chatbot's supervisor relies on the default; it must not move."""

    names = _filesystem_tool_names(_supervisor_stack())

    assert {"execute", "delete"} <= names


def test_a_supervisor_can_drop_the_tools_its_backend_cannot_serve() -> None:
    """`execute` and `delete` are a dead affordance on a state backend.

    `vitess-ai`'s two agents run VITESS through one trusted MCP gateway. Neither
    tool can do anything against `SupervisorStateBackend` -- `execute` answers
    that the backend implements no sandbox protocol -- and `execute` is the tool
    a weaker model reaches for when it decides to run the binary itself, which
    is the one thing a single execution path exists to prevent.
    """

    names = _filesystem_tool_names(
        _supervisor_stack(
            filesystem_tools=("read_file", "write_file", "edit_file", "ls", "glob", "grep")
        )
    )

    assert names == {"read_file", "write_file", "edit_file", "ls", "glob", "grep"}


def test_a_supervisor_refuses_a_bare_string_of_tool_names() -> None:
    with pytest.raises(ValueError, match="each be read as a tool name"):
        _supervisor_stack(filesystem_tools="read_file")


def _specialist_stack(monkeypatch: pytest.MonkeyPatch, **overrides) -> list:
    monkeypatch.setattr(specialist_runtime, "settings", lambda: _runtime_settings())
    return specialist_runtime.build_specialist_middleware(
        backend=specialist_runtime.build_specialist_backend(),
        summarizer_model=_summarizer_model(),
        fallback_models=[],
        filesystem_tool_descriptions={},
        specialist_name="test-specialist",
        **overrides,
    )


def test_a_specialist_gets_every_filesystem_tool_unless_it_asks_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default is what juena-chatbot's specialists rely on; it must not move."""

    names = _filesystem_tool_names(_specialist_stack(monkeypatch))

    assert {"ls", "read_file", "write_file", "edit_file", "glob", "grep"} <= names


def test_an_application_can_narrow_a_specialist_to_the_tools_it_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A specialist whose job is a conversation and one tool call should have two.

    `vitess-ai`'s five module specialists were each being bound eleven tools,
    `execute` and `delete` among them, which is context spent on tools they will
    never use and, on a weaker model, an invitation to use them.
    """

    names = _filesystem_tool_names(
        _specialist_stack(monkeypatch, filesystem_tools=["read_file"])
    )

    assert names == {"read_file"}


def test_a_specialist_can_have_no_filesystem_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`read_file` is mandatory in an allowlist, so `None` is the only way to none.

    `vitess-ai`'s module specialists hand off through a typed state channel, not
    through `/findings/`. None of them can write a file, so the `read_file` they
    were left with could never have anything to read -- and the prompt then had
    to describe a tool that did nothing.
    """

    stack = _specialist_stack(monkeypatch, filesystem_tools=None)

    assert not [
        item for item in stack if type(item).__name__ == "FilesystemMiddleware"
    ]
    assert [type(item).__name__ for item in stack][:3] == [
        "SpecialistOutcomeMiddleware",
        "RepeatedToolCallMiddleware",
        "_DeepAgentsSummarizationMiddleware",
    ]


def test_a_bare_string_of_tool_names_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Sequence[str]` accepted `"read_file"` and bound eight one-letter tools."""

    with pytest.raises(ValueError, match="not the bare string"):
        _specialist_stack(monkeypatch, filesystem_tools="read_file")


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


def test_explicit_fallback_models_preserve_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, str]] = []
    monkeypatch.setattr(
        specialist_runtime,
        "get_available_providers",
        lambda: {"blablador": True},
    )
    monkeypatch.setattr(
        specialist_runtime,
        "build_chat_model",
        lambda **kwargs: built.append(kwargs) or object(),
    )

    fallbacks = specialist_runtime.build_fallback_models(
        [(" BLABLADOR ", "mimo-pro"), ("blablador", "gpt-oss")]
    )

    assert len(fallbacks) == 2
    assert built == [
        {"provider": "blablador", "model": "mimo-pro"},
        {"provider": "blablador", "model": "gpt-oss"},
    ]
