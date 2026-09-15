"""Agent invocation, streaming, resume, thread deletion and artifact download.

See ``endpoints.build_api_router``. 00-BOUNDARY.md decision 10 left this module
in the applications; 01/CP4 moved it here instead, and records why.
"""

from __future__ import annotations

__all__ = ["endpoints"]
