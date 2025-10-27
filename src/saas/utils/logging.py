"""Logging helpers for the SAAS stack."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from saas.config import get_settings

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure global logging handlers.

    The first time this function is called it attaches a console handler
    and a rotating file handler pointing to the configured log directory.
    Subsequent invocations are no-ops.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_file = settings.log_dir / "saas.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    _CONFIGURED = True


__all__ = ["configure_logging"]
