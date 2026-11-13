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
        """Initialize the middleware.

        Args:
            recipient: Name of this graph, recorded when its model receives a
                correction.
        """
        self.recipient = recipient

    def before_model(self, state, runtime):
        """Add the corrections this graph has not seen yet to its messages.

        Args:
            state: The agent state.
            runtime: The graph runtime, whose context carries the run control.

        Returns:
            A state update with the new corrections, or ``None``.

        Raises:
            asyncio.CancelledError: The user stopped the run.
        """
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
        """Async version of :meth:`before_model`.

        Args:
            state: The agent state.
            runtime: The graph runtime, whose context carries the run control.

        Returns:
            A state update with the new corrections, or ``None``.

        Raises:
            asyncio.CancelledError: The user stopped the run.
        """
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
        """Record which corrections reach the model, then call it.

        Args:
            request: The model request.
            handler: Calls the model.

        Returns:
            The model's response.

        Raises:
            asyncio.CancelledError: The user stopped the run.
        """
        self._model_control(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        """Async version of :meth:`wrap_model_call`.

        Args:
            request: The model request.
            handler: Calls the model.

        Returns:
            The model's response.

        Raises:
            asyncio.CancelledError: The user stopped the run.
        """
        self._model_control(request)
        return await handler(request)

    def wrap_tool_call(self, request, handler):
        """Run a tool call unless the user stopped the run.

        Args:
            request: The tool-call request.
            handler: Runs the request.

        Returns:
            The handler's result.

        Raises:
            asyncio.CancelledError: The user stopped the run.
        """
        control = control_from(request.runtime.context)
        if control is not None:
            control.check()
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        """Async version of :meth:`wrap_tool_call`.

        Args:
            request: The tool-call request.
            handler: Runs the request.

        Returns:
            The handler's result.

        Raises:
            asyncio.CancelledError: The user stopped the run.
        """
        control = control_from(request.runtime.context)
        if control is not None:
            control.check()
        return await handler(request)
