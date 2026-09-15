"""Stub for 01/CP1. Ported from ``juena/core/log.py``.

Unlike the source, this must never import ``config`` at module scope — that
import-time coupling is exactly what makes importing any ``juena`` module
today load and validate ``.env`` (00-BOUNDARY.md, decision 1; Finding 1).
``_get_log_level`` instead reads ``config.settings().LOG_LEVEL`` **at call
time**, falling back to ``"INFO"`` when ``configure()`` has not run — a
logger must never be the thing that refuses to start.
"""

from __future__ import annotations

import logging
from typing import Union

__all__ = ["LOG_FORMAT", "LOG_DATE_FORMAT", "setup_logger", "get_logger"]

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_log_level(level: Union[int, str, None] = None) -> int:
    raise NotImplementedError("juena_core.log._get_log_level lands in 01/CP1")


def setup_logger(logger: logging.Logger, level: Union[int, str, None] = None) -> None:
    raise NotImplementedError("juena_core.log.setup_logger lands in 01/CP1")


def get_logger(name: str, level: Union[int, str, None] = None) -> logging.Logger:
    raise NotImplementedError("juena_core.log.get_logger lands in 01/CP1")
