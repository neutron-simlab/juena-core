"""The Streamlit UI shell, behind the ``[ui]`` extra. Importing
``juena_core`` itself must never require Streamlit — that is 01/CP5's
clean-room wheel test. See 00-BOUNDARY.md, decision 7."""

from __future__ import annotations

__all__ = ["math_rendering", "chat_storage", "client_setup", "streaming", "components"]
