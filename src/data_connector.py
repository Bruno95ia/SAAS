from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests

from saas_core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
DATASETS_DIR = settings.storage.data_dir / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    logger.info("Downloading %s", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with dest.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                fp.write(chunk)
    logger.info("Saved dataset to %s", dest)


def sync_roboflow(project: str, version: str, format: str = "zip") -> Optional[Path]:
    api_key = settings.roboflow_api_key
    if not api_key:
        logger.warning("ROBOFLOW_API_KEY not configured")
        return None
    url = f"https://universe.roboflow.com/{project}/{version}?format={format}&api_key={api_key}"
    dest = DATASETS_DIR / f"roboflow_{project.replace('/', '_')}_{version}.{format}"
    download_file(url, dest)
    return dest


def sync_huggingface(repo_id: str, filename: str) -> Optional[Path]:
    token = settings.hf_token
    if not token:
        logger.warning("HF_TOKEN not configured")
        return None
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}?token={token}"
    dest = DATASETS_DIR / f"hf_{repo_id.replace('/', '_')}_{filename.replace('/', '_')}"
    download_file(url, dest)
    return dest


__all__ = ["sync_roboflow", "sync_huggingface"]
