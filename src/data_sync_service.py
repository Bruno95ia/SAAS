"""Background dataset synchronisation worker."""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable, List, Tuple

from data_connector import sync_huggingface, sync_roboflow

logger = logging.getLogger(__name__)


def _parse_sources(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _roboflow_pairs(values: Iterable[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            logger.warning("Roboflow entry '%s' must be in 'workspace/project:version' format", value)
            continue
        project, version = value.split(":", 1)
        pairs.append((project, version))
    return pairs


def _huggingface_pairs(values: Iterable[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            logger.warning("HuggingFace entry '%s' must be in 'repo_id:filename' format", value)
            continue
        repo_id, filename = value.split(":", 1)
        pairs.append((repo_id, filename))
    return pairs


def run_sync_cycle() -> None:
    roboflow_sources = _roboflow_pairs(_parse_sources(os.getenv("ROBOFLOW_SOURCES", "")))
    hf_sources = _huggingface_pairs(_parse_sources(os.getenv("HUGGINGFACE_SOURCES", "")))

    if not roboflow_sources and not hf_sources:
        logger.info("Nenhuma fonte configurada para sincronização de datasets")
        return

    for project, version in roboflow_sources:
        try:
            sync_roboflow(project, version)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Falha ao sincronizar Roboflow %s:%s: %s", project, version, exc)

    for repo_id, filename in hf_sources:
        try:
            sync_huggingface(repo_id, filename)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Falha ao sincronizar HuggingFace %s:%s: %s", repo_id, filename, exc)


def main() -> None:
    interval = int(os.getenv("DATA_SYNC_INTERVAL", "3600"))
    interval = max(60, interval)
    logger.info("Data sync worker iniciado (intervalo: %ss)", interval)

    while True:
        run_sync_cycle()
        time.sleep(interval)


if __name__ == "__main__":
    main()
