from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class DatabaseSettings:
    user: str = os.getenv("POSTGRES_USER", "saas")
    password: str = os.getenv("POSTGRES_PASSWORD", "saas")
    database: str = os.getenv("POSTGRES_DB", "saas")
    host: str = os.getenv("POSTGRES_HOST", "db")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


@dataclass(slots=True)
class StorageSettings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))

    def ensure(self) -> None:
        for folder in [
            self.data_dir,
            self.data_dir / "logs",
            self.data_dir / "clips",
            self.data_dir / "datasets",
        ]:
            folder.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class DetectionSettings:
    model_path: str = os.getenv("MODEL_PATH", "yolov8n-pose.pt")
    confidence: float = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    iou: float = float(os.getenv("DETECTION_IOU", "0.45"))
    device: Optional[str] = os.getenv("DETECTION_DEVICE")
    clip_length: int = int(os.getenv("EVENT_CLIP_LENGTH", "10"))


@dataclass(slots=True)
class AppSettings:
    debug: bool = _bool(os.getenv("DEBUG"), False)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    database: DatabaseSettings = DatabaseSettings()
    storage: StorageSettings = StorageSettings()
    detection: DetectionSettings = DetectionSettings()
    roboflow_api_key: Optional[str] = os.getenv("ROBOFLOW_API_KEY")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    stream_fps: int = int(os.getenv("STREAM_FPS", "15"))
    max_stream_backlog: int = int(os.getenv("MAX_STREAM_BACKLOG", "180"))

    def configure_logging(self) -> None:
        self.storage.ensure()
        log_file = self.storage.data_dir / "logs" / "saas-api.log"
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.configure_logging()
    return settings
