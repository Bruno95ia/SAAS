"""Utilitário de logging unificado.

Este módulo fornece uma função `get_logger` que aplica uma configuração
rotativa de arquivo (`runs/logs/saas.log`) e permite o uso consistente de logs
em todos os componentes da aplicação.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from saas import config

# Tamanho máximo de cada arquivo de log (~5 MB) com até 3 backups antigos.
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 3

# Armazenamos o estado de configuração para evitar duplicidade de handlers.
_configured = False


def _build_handler() -> RotatingFileHandler:
    """Cria o handler rotativo apontando para `runs/logs/saas.log`."""

    config.ensure_runtime_directories()
    log_path = config.LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    return handler


def configure_logging(level: int = logging.INFO) -> None:
    """Configura o logger raiz `saas` apenas uma vez."""

    global _configured
    if _configured:
        return

    handler = _build_handler()

    root = logging.getLogger("saas")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    # Também configuramos o logger raiz global para garantir que bibliotecas
    # externas (OpenCV, Ultralytics) escrevam no mesmo arquivo quando possível.
    logging.basicConfig(level=level, handlers=[handler])

    _configured = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Retorna um logger configurado com handler rotativo."""

    configure_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
