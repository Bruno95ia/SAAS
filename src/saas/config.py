"""Central configuration for the SAAS proof of concept.

This module exposes a :func:`get_settings` helper that loads sensible
defaults and allows overriding critical values through environment
variables.  The function is cached so every module in the stack imports
and shares the same configuration object.
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_WEIGHTS = Path("/mnt/data/SAAS/yolov8n.pt")
DEFAULT_CAMERAS = ["/mnt/data/sample_fall.mp4"]
DEFAULT_API_URL = "http://0.0.0.0:8000"
DEFAULT_API_KEY = "saas-poc-key"
DEFAULT_LOG_DIR = Path("/mnt/data/SAAS/runs/logs")
DEFAULT_STREAM_DIR = Path("/mnt/data/SAAS/runs/streams")
DEFAULT_DB_PATH = Path("/mnt/data/SAAS/runs/events.db")
DEFAULT_TEST_LOG = Path("/mnt/data/SAAS/runs/tests_poc.log")
DEFAULT_METRICS_PATH = Path("/mnt/data/SAAS/runs/metrics.json")


@dataclass
class Settings:
    """Container with the runtime configuration for the SAAS stack."""

    weights_path: Path = DEFAULT_WEIGHTS
    img_size: int = 640
    cameras: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERAS))
    api_url: str = DEFAULT_API_URL
    api_key: str = DEFAULT_API_KEY
    log_dir: Path = DEFAULT_LOG_DIR
    stream_dir: Path = DEFAULT_STREAM_DIR
    alerts_db_path: Path = DEFAULT_DB_PATH
    tests_log_path: Path = DEFAULT_TEST_LOG
    metrics_path: Path = DEFAULT_METRICS_PATH
    suspicious_labels: Sequence[str] = ("queda", "fall", "movimento suspeito")
    label_aliases: Dict[str, str] = field(default_factory=lambda: {"person": "fall"})

    def ensure_directories(self) -> None:
        """Create required directories for logs, streams and database."""

        for directory in {self.log_dir, self.stream_dir, self.alerts_db_path.parent}:
            directory.mkdir(parents=True, exist_ok=True)
        self.tests_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_assets(self) -> None:
        """Ensure that the default model weights and sample video exist."""

        package_root = Path(__file__).resolve().parent
        weights_source = package_root / "weights" / "best.pt"
        if not self.weights_path.exists() and weights_source.exists():
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            self.weights_path.write_bytes(weights_source.read_bytes())

        sample_source = package_root / "sample.mp4"
        sample_target = Path(DEFAULT_CAMERAS[0])
        if not sample_target.exists() and sample_source.exists():
            sample_target.parent.mkdir(parents=True, exist_ok=True)
            sample_target.write_bytes(sample_source.read_bytes())

    def to_json(self) -> str:
        data = {
            "weights_path": str(self.weights_path),
            "img_size": self.img_size,
            "cameras": self.cameras,
            "api_url": self.api_url,
            "api_key": "***masked***",
            "log_dir": str(self.log_dir),
            "stream_dir": str(self.stream_dir),
            "alerts_db_path": str(self.alerts_db_path),
            "tests_log_path": str(self.tests_log_path),
            "metrics_path": str(self.metrics_path),
            "suspicious_labels": list(self.suspicious_labels),
            "label_aliases": dict(self.label_aliases),
        }
        return json.dumps(data, indent=2)


def _parse_cameras(raw: str | None) -> List[str]:
    if not raw:
        return list(DEFAULT_CAMERAS)
    candidates: Iterable[str] = (segment.strip() for segment in raw.split(","))
    return [segment for segment in candidates if segment]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached :class:`Settings` built from the environment."""

    weights_env = os.getenv("SAAS_WEIGHTS")
    cameras_env = os.getenv("SAAS_CAMERAS")

    settings = Settings(
        weights_path=Path(weights_env) if weights_env else DEFAULT_WEIGHTS,
        img_size=int(os.getenv("SAAS_IMG_SIZE", "640")),
        cameras=_parse_cameras(cameras_env),
        api_url=os.getenv("SAAS_API_URL", DEFAULT_API_URL),
        api_key=os.getenv("SAAS_API_KEY", DEFAULT_API_KEY),
    )

    settings.ensure_directories()
    settings.ensure_assets()
    return settings


__all__ = ["Settings", "get_settings"]
