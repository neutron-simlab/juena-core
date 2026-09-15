"""Schemas shared by the agent HTTP server and its clients."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from juena_core.schema.llm_models import Provider

__all__ = [
    "UserInput",
    "StreamInput",
    "ToolCall",
    "ChatMessage",
    "HealthStatus",
    "CreateChatInput",
    "UpdateChatInput",
]


class UserInput(BaseModel):
    """Basic user input for an agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["Hello, how can you help me?"],
    )
    thread_id: str | None = Field(
        description="Thread ID used to continue a multi-turn conversation.",
        default=None,
        max_length=255,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    provider: Provider | None = Field(
        description="LLM provider to use.",
        default=None,
        examples=["openai", "blablador"],
    )
    model: str | None = Field(
        description="Provider-specific LLM model name.",
        default=None,
        examples=["gpt-4o-mini", "GPT-OSS-120b"],
    )


class StreamInput(UserInput):
    """User input for a streamed agent response."""


class ToolCall(TypedDict):
    """A request emitted by a model to call a tool."""

    name: str
    args: dict[str, Any]
    id: str | None
    type: NotRequired[Literal["tool_call"]]


class ChatMessage(BaseModel):
    """One message in a persisted or streamed conversation."""

    type: Literal["human", "ai", "tool", "custom", "system"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom", "system"],
    )
    id: str | None = Field(
        description="Stable identifier copied from the underlying message.",
        default=None,
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: list[ToolCall] = Field(
        description="Tool calls in the message.",
        default_factory=list,
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message responds to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="ID of this single invocation, used for feedback and tracing.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    thread_id: str | None = Field(
        description="ID of the whole conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    response_metadata: dict[str, Any] = Field(
        description="Response metadata such as headers or token counts.",
        default_factory=dict,
    )
    custom_data: dict[str, Any] = Field(
        description="Custom message data.",
        default_factory=dict,
    )
    timestamp: datetime | None = Field(
        description="Timestamp when the message was created.",
        default=None,
    )


class HealthStatus(BaseModel):
    """Health-check response."""

    status: Literal["ok", "error"] = Field(description="Overall health status.")
    timestamp: datetime = Field(
        description="Timestamp of the health check.",
        default_factory=datetime.now,
    )
    version: str = Field(description="Service version.", default="0.1.0")
    uptime: float | None = Field(
        description="Service uptime in seconds.",
        default=None,
    )
    details: dict[str, Any] = Field(
        description="Additional health-check details.",
        default_factory=dict,
    )


class CreateChatInput(BaseModel):
    """Request body for creating an agent-owned conversation."""

    agent_id: str = Field(min_length=1, max_length=64)
    thread_id: str | None = Field(default=None, max_length=255)
    title: str = Field(default="New Chat", max_length=500)


class UpdateChatInput(BaseModel):
    """Request body for renaming or re-summarising a conversation."""

    title: str | None = Field(default=None, max_length=500)
    summary: str | None = Field(default=None, max_length=10_000)
