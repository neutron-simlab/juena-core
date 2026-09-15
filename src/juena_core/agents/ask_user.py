"""Stub for 01/CP2. Ported from ``juena/agents/ask_user.py``. Needs
``CLARIFICATION_KIND``, which moves to ``schema/interrupts.py``
(00-BOUNDARY.md, *Moves whole*)."""

from __future__ import annotations

from langchain_core.tools import BaseTool

__all__ = ["MAX_OPTIONS", "ASK_USER_DESCRIPTION", "build_ask_user_tool"]

MAX_OPTIONS = 4
ASK_USER_DESCRIPTION = ""


def build_ask_user_tool(asked_by: str) -> BaseTool:
    raise NotImplementedError("juena_core.agents.ask_user.build_ask_user_tool lands in 01/CP2")
