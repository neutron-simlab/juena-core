"""Shared middleware and backend assembly for supervisors and specialists."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from juena_core.agents.backends import (
    FINDINGS_PREFIX,
    MEMORY_SOURCES,
    MEMORY_SYSTEM_PROMPT,
    SUPERVISOR_FILESYSTEM_TOOL_DESCRIPTIONS,
    FindingsStateBackend,
    ReadOnlyFilesystemBackend,
    ReadOnlyInputsStateBackend,
)
from juena_core.agents.delegation import with_delegation_boundary
from juena_core.agents.loop_guard import RepeatedToolCallMiddleware
from juena_core.agents.specialist_outcome import SpecialistOutcomeMiddleware
from juena_core.config import settings
from juena_core.llms_providers import (
    build_chat_model,
    get_available_providers,
    get_default_model,
)
from juena_core.log import get_logger
from juena_core.server.agent.runtime_model_middleware import RuntimeModelMiddleware

logger = get_logger(__name__)

__all__ = [
    "SUPERVISOR_MODEL_CALL_LIMIT",
    "SUPERVISOR_TOOL_CALL_LIMIT",
    "SPECIALIST_MODEL_CALL_LIMIT",
    "SPECIALIST_TOOL_CALL_LIMIT",
    "BACKGROUND_MODEL_CALL_LIMIT",
    "BACKGROUND_TOOL_CALL_LIMIT",
    "ASK_USER_CALL_LIMIT",
    "MODEL_MAX_RETRIES",
    "TOOL_MAX_RETRIES",
    "UNATTENDED_NOTICE",
    "SPECIALIST_TASK_DESCRIPTION",
    "PromptResourceError",
    "load_markdown",
    "has_authored_skills",
    "build_specialist_backend",
    "resilience_middleware",
    "build_fallback_models",
    "build_specialist_middleware",
    "build_supervisor_middleware",
]

SUPERVISOR_MODEL_CALL_LIMIT = 50
SUPERVISOR_TOOL_CALL_LIMIT = 100
SPECIALIST_MODEL_CALL_LIMIT = 60
SPECIALIST_TOOL_CALL_LIMIT = 150
# Background work gets a wider budget because nobody is watching a spinner.
# The application's wall-clock timeout remains the real bound.
BACKGROUND_MODEL_CALL_LIMIT = 150
BACKGROUND_TOOL_CALL_LIMIT = 400
# Questions allowed per graph invocation. A resume starts a new invocation.
ASK_USER_CALL_LIMIT = 3
MODEL_MAX_RETRIES = 3
# Pinned because an upstream default change must not silently change policy.
TOOL_MAX_RETRIES = 2

_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

UNATTENDED_NOTICE = (
    "You are running unattended, as a background job the user started and then "
    "carried on with their conversation. Nobody is watching this run: `ask_user` "
    "is not available to you here, and neither is an execution backend. Decide "
    "with the evidence you can gather, and put whatever you could not settle in "
    "`limitations` -- an honest gap there is worth far more than a guess, "
    "because the supervisor can act on it and cannot act on a guess.\n\n"
    "Your report is the deliverable. The `/findings/` rule is unchanged here: "
    "write a file only if this objective asked you to, and if it did, the file "
    "itself comes back with the report for whoever is delegated to next."
)

SPECIALIST_TASK_DESCRIPTION = """Delegate a substantive domain task to a specialist.
Give the specialist a self-contained objective with only the relevant user context and
saved preferences. Prefer one specialist; use more only for an explicit comparison or
clearly independent aspects that need different expertise. Resolve an ambiguous request
with ask_user before delegating rather than guessing on the user's behalf.

Every result arrives as a `<specialist_report>` block paired with a
`<verified_by_server>` block. The second block is written by the server from actual
execution evidence and the artifact store; it overrides anything the report claims.

Available specialist types:
{available_agents}
"""


class PromptResourceError(RuntimeError):
    """Raised when a required packaged prompt cannot be loaded."""


def load_markdown(package: str, filename: str) -> str:
    """Load one required Markdown resource and fail loudly when packaging is broken."""

    try:
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError) as exc:
        raise PromptResourceError(
            f"Unable to load prompt resource {package}:{filename}"
        ) from exc
    if not text.strip():
        raise PromptResourceError(f"Prompt resource {package}:{filename} is empty")
    return text.rstrip() + "\n"


def _valid_skill_file(skill_file: Path) -> bool:
    """Perform the minimum Agent Skills validation needed before enabling middleware."""

    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError:
        return False
    if not text.startswith("---\n"):
        return False
    try:
        _marker, frontmatter, body = text.split("---", 2)
        metadata = yaml.safe_load(frontmatter)
    except (ValueError, yaml.YAMLError):
        return False
    if not isinstance(metadata, dict):
        return False
    name = metadata.get("name")
    description = metadata.get("description")
    return bool(
        isinstance(name, str)
        and name == skill_file.parent.name
        and len(name) <= 64
        and _SKILL_NAME_RE.fullmatch(name)
        and isinstance(description, str)
        and description.strip()
        and len(description) <= 1024
        and body.strip()
    )


def has_authored_skills(skills_dir: Path | None) -> bool:
    """Return whether a specialist owns at least one valid `SKILL.md`."""

    if skills_dir is None or not skills_dir.is_dir():
        return False
    return any(
        child.is_dir() and _valid_skill_file(child / "SKILL.md")
        for child in skills_dir.iterdir()
    )


def build_specialist_backend(
    *,
    repo_cache_root: Path | None = None,
    skills_dir: Path | None = None,
) -> CompositeBackend:
    """Build isolated staged-input, repository, and optional skill routes."""

    routes: dict[str, Any] = {FINDINGS_PREFIX: FindingsStateBackend()}
    if repo_cache_root is not None:
        routes["/repos/"] = ReadOnlyFilesystemBackend(
            root_dir=repo_cache_root,
            virtual_mode=True,
            label="read-only repository",
        )
    if has_authored_skills(skills_dir):
        routes["/skills/"] = ReadOnlyFilesystemBackend(
            root_dir=skills_dir,
            virtual_mode=True,
            label="read-only skill",
        )
    return CompositeBackend(default=ReadOnlyInputsStateBackend(), routes=routes)


def resilience_middleware(
    fallback_models: Sequence[Any],
    *,
    model_call_limit: int,
    tool_call_limit: int,
    ask_user_bound: bool = True,
) -> list[Any]:
    """Build retry, fallback, and per-invocation budget middleware in nesting order."""

    middleware: list[Any] = []
    if fallback_models:
        middleware.append(ModelFallbackMiddleware(*fallback_models))
    middleware.extend(
        [
            ModelRetryMiddleware(max_retries=MODEL_MAX_RETRIES, on_failure="error"),
            ToolRetryMiddleware(max_retries=TOOL_MAX_RETRIES),
            ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="continue"),
        ]
    )
    if ask_user_bound:
        middleware.append(
            ToolCallLimitMiddleware(
                tool_name="ask_user",
                run_limit=ASK_USER_CALL_LIMIT,
                exit_behavior="continue",
            )
        )
    return middleware


def build_fallback_models() -> list[Any]:
    """Build deployment-level fallbacks independently of the request model."""

    fallback_provider = (settings().FALLBACK_PROVIDER or "").strip().lower()
    if not fallback_provider:
        return []
    if not get_available_providers().get(fallback_provider, False):
        logger.warning(
            "FALLBACK_PROVIDER '%s' is not configured; running without a model fallback",
            fallback_provider,
        )
        return []
    fallback_model = get_default_model(fallback_provider)
    if not fallback_model:
        return []
    logger.info("Model fallback enabled: %s/%s", fallback_provider, fallback_model)
    return [build_chat_model(provider=fallback_provider, model=fallback_model)]


def build_specialist_middleware(
    *,
    backend: BackendProtocol,
    summarizer_model: Any,
    fallback_models: Sequence[Any],
    filesystem_tool_descriptions: Mapping[str, str],
    specialist_name: str,
    skills_dir: Path | None = None,
    skills_label: str = "",
    filesystem_tools: Sequence[str] | Literal["all"] = "all",
    execution_middleware: Sequence[Any] = (),
    interrupt_on: dict[str, Any] | None = None,
    unattended: bool = False,
) -> list[Any]:
    """Build the common specialist stack while keeping agent choices local.

    Execution middleware and its approval policy are an all-or-nothing pair.
    Unattended specialists receive neither that pair nor `ask_user`.

    `filesystem_tools` narrows what `FilesystemMiddleware` exposes. It defaults
    to every tool the backend supports -- eight of them, including `execute`
    and `delete` -- which is right for a research specialist that works in a
    workspace and wrong for one whose whole job is a conversation and a single
    validation call. A specialist bound to ten tools it will never use spends
    context on them and, on a weaker model, reaches for them. `read_file` is
    required in any allowlist by the middleware itself.
    """

    has_execution = bool(execution_middleware)
    has_interrupt = interrupt_on is not None
    if has_execution != has_interrupt:
        raise ValueError(
            "execution_middleware and interrupt_on must be supplied together"
        )
    if unattended and (has_execution or has_interrupt):
        raise ValueError("unattended specialists cannot mount an execution backend")

    middleware: list[Any] = [
        SpecialistOutcomeMiddleware(specialist_name=specialist_name),
        RepeatedToolCallMiddleware(),
    ]
    if has_authored_skills(skills_dir):
        middleware.append(
            SkillsMiddleware(
                backend=backend,
                sources=[("/skills/", skills_label)],
            )
        )
    middleware.extend(
        [
            FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=dict(filesystem_tool_descriptions),
                max_execute_timeout=settings().EXECUTE_TIMEOUT_SECONDS,
                tools="all" if filesystem_tools == "all" else list(filesystem_tools),
            ),
            create_summarization_middleware(summarizer_model, backend),
            PatchToolCallsMiddleware(),
            *resilience_middleware(
                fallback_models,
                model_call_limit=(
                    BACKGROUND_MODEL_CALL_LIMIT
                    if unattended
                    else SPECIALIST_MODEL_CALL_LIMIT
                ),
                tool_call_limit=(
                    BACKGROUND_TOOL_CALL_LIMIT
                    if unattended
                    else SPECIALIST_TOOL_CALL_LIMIT
                ),
                ask_user_bound=not unattended,
            ),
        ]
    )
    if has_execution:
        middleware.extend(execution_middleware)
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    return middleware


def build_supervisor_middleware(
    *,
    backend: BackendProtocol,
    summarizer_model: Any,
    fallback_models: Sequence[Any],
    subagents: Sequence[dict[str, Any]],
    task_description: str = SPECIALIST_TASK_DESCRIPTION,
    extra: Sequence[Any] = (),
    memory_system_prompt: str = MEMORY_SYSTEM_PROMPT,
    model_call_limit: int = SUPERVISOR_MODEL_CALL_LIMIT,
    tool_call_limit: int = SUPERVISOR_TOOL_CALL_LIMIT,
) -> list[Any]:
    """Return the canonical supervisor middleware stack.

    `extra` is the sole splice point. It sits after
    `PatchToolCallsMiddleware` and before `RuntimeModelMiddleware`, preserving
    the established nesting order while letting an application add genuinely
    application-owned cross-cutting behavior.
    """

    return [
        RepeatedToolCallMiddleware(),
        FilesystemMiddleware(
            backend=backend,
            custom_tool_descriptions=SUPERVISOR_FILESYSTEM_TOOL_DESCRIPTIONS,
        ),
        MemoryMiddleware(
            backend=backend,
            sources=MEMORY_SOURCES,
            system_prompt=memory_system_prompt,
        ),
        SubAgentMiddleware(
            backend=backend,
            subagents=with_delegation_boundary(list(subagents)),
            system_prompt=None,
            task_description=task_description,
        ),
        create_summarization_middleware(summarizer_model, backend),
        PatchToolCallsMiddleware(),
        *extra,
        RuntimeModelMiddleware(),
        *resilience_middleware(
            fallback_models,
            model_call_limit=model_call_limit,
            tool_call_limit=tool_call_limit,
        ),
    ]
