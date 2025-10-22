"""Integração centralizada de logs usando :mod:`loguru`."""

from __future__ import annotations

import logging
import sys
from loguru import logger as _logger

from saas import config

_CONFIGURED = False


class _InterceptHandler(logging.Handler):
    """Redireciona logs do ``logging`` padrão para o Loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - cola técnica
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        _logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str | int = "INFO") -> None:
    """Configura os handlers globais apenas uma vez."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    config.ensure_runtime_directories()
    log_path = config.LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[context]} | {message}"
    _logger.add(sys.stdout, level=level, format=fmt, enqueue=True, backtrace=False, diagnose=False)
    _logger.add(
        log_path,
        level=level,
        format=fmt,
        enqueue=True,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)

    _CONFIGURED = True


def get_logger(name: str, level: str | int | None = None):
    """Retorna uma instância do Loguru com o campo ``context`` preenchido."""

    configure_logging(level or "INFO")
    bound = _logger.bind(context=name)
    return bound
