"""Apply foreground corrections inside each graph, before its next model call."""

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.config import get_config
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from typing_extensions import Annotated, NotRequired

from juena_core.server.run_control import control_from

__all__ = ["SteeringMiddleware"]


class SteeringState(AgentState):
    steering_run_id: NotRequired[str]
    steering_applied_ids: NotRequired[list[str]]
    # Visible in the transcript, but never input to a model or summarizer.
    unapplied_steering: Annotated[list[AnyMessage], add_messages]


class SteeringMiddleware(AgentMiddleware[SteeringState]):
    """Persist each correction once per graph; acknowledge model inclusion."""

    state_schema = SteeringState

    def __init__(self, recipient: str):
        self.recipient = recipient

    def before_model(self, state, runtime):
        control = control_from(runtime.context)
        if control is None:
            return None
        control.check()
        seen = (
            set(state.get("steering_applied_ids", []))
            if state.get("steering_run_id") == control.id
            else set()
        )
        messages = control.messages(seen)
        if not messages:
            return None
        return {
            "messages": messages,
            "steering_run_id": control.id,
            "steering_applied_ids": [
                *seen,
                *(m.additional_kwargs["steering"]["instruction_id"] for m in messages),
            ],
        }

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)

    def _model_control(self, request):
        control = control_from(request.runtime.context)
        if control is not None:
            control.check()
            namespace = get_config().get("configurable", {}).get("checkpoint_ns", "")
            ids = {
                m.additional_kwargs["steering"]["instruction_id"]
                for m in request.messages
                if m.additional_kwargs.get("steering", {}).get("run_id") == control.id
            }
            control.applied(ids, f"{self.recipient}:{namespace}")

    def wrap_model_call(self, request, handler):
        self._model_control(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._model_control(request)
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        control = control_from(request.runtime.context)
        if control is not None:
            control.check()
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        control = control_from(request.runtime.context)
        if control is not None:
            control.check()
        return await handler(request)
