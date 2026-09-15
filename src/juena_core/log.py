"""Consistent, import-safe logging configuration for :mod:`juena_core`."""

from __future__ import annotations

import logging

__all__ = ["LOG_FORMAT", "LOG_DATE_FORMAT", "setup_logger", "get_logger"]

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_log_level(level: int | str | None = None) -> int:
    """Convert a level to its logging constant.

    Configuration is read only when this function is called. Before an
    application configures core, logging safely defaults to ``INFO``.
    """

    if level is None:
        from juena_core.config import settings

        try:
            level = settings().LOG_LEVEL
        except RuntimeError:
            level = "INFO"

    if isinstance(level, int):
        return level

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level.upper(), logging.INFO)


def setup_logger(logger: logging.Logger, level: int | str | None = None) -> None:
    """Configure *logger* once with the shared format and requested level."""

    if logger.handlers:
        return

    resolved_level = _get_log_level(level)
    handler = logging.StreamHandler()
    handler.setLevel(resolved_level)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    logger.addHandler(handler)
    logger.setLevel(resolved_level)
    logger.propagate = False


def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """Return a consistently configured logger named *name*."""

    logger = logging.getLogger(name)
    setup_logger(logger, level)
    return logger
