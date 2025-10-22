"""Configurações compartilhadas do projeto SAAS."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

# Diretórios -----------------------------------------------------------------

# ``config.py`` está localizado em ``src/saas``. Subindo duas pastas chegamos ao
# diretório raiz do repositório (onde vivem ``runs/`` e ``events.db``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNS_DIR = PROJECT_ROOT / "runs"
BUFFER_DIR = RUNS_DIR / "buffer"
CLIPS_DIR = RUNS_DIR / "clips"
LOG_DIR = RUNS_DIR / "logs"
RESULTS_DIR = RUNS_DIR / "results"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATABASE_PATH = Path(os.getenv("SAAS_DB_PATH", PROJECT_ROOT / "events.db"))

LOG_FILE = LOG_DIR / "saas.log"

DEFAULT_WEIGHTS = Path(os.getenv("SAAS_YOLO_WEIGHTS", WEIGHTS_DIR / "yolov8n.pt"))


# API ------------------------------------------------------------------------

@dataclass(frozen=True)
class APISettings:
    """Configurações básicas para autenticação na API."""

    url: str
    key: str


def load_api_settings() -> APISettings:
    url = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000").rstrip("/")
    key = os.getenv("SAAS_API_KEY", "minha-chave-forte")
    return APISettings(url=url, key=key)


# Diretórios utilitários -----------------------------------------------------

def ensure_runtime_directories() -> None:
    """Garante que a estrutura de diretórios esperada exista."""

    for folder in (RUNS_DIR, BUFFER_DIR, CLIPS_DIR, LOG_DIR, RESULTS_DIR, WEIGHTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def default_buffer_dir(camera_id: str) -> Path:
    return BUFFER_DIR / camera_id


def segment_template(base_dir: Path) -> Path:
    return base_dir / "%Y%m%d" / "%H%M%S.m4s"


# Fontes de vídeo ------------------------------------------------------------

def detect_source_type(identifier: str) -> str:
    """Classifica o tipo de origem usada na captura."""

    value = identifier.strip().lower()
    if value.startswith("rtsp://"):
        return "rtsp"
    if value.startswith("screen"):
        return "screen"
    if value.startswith("local"):
        return "local"
    return "custom"


def _extract_options(source: str) -> str:
    if source.startswith("rtsp://") or ":" not in source:
        return ""
    return source.split(":", 1)[1]


def parse_source_options(source: str) -> Dict[str, str]:
    """Interpreta pares ``chave=valor`` passados após o tipo da origem.

    Exemplos
    --------
    ``screen:display=:0.0,size=1920x1080,fps=25``
        → {"display":":0.0", "size":"1920x1080", "fps":"25"}

    ``local:/dev/video2,fps=60``
        → {"device":"/dev/video2", "fps":"60"}
    """

    options: Dict[str, str] = {}
    raw = _extract_options(source)
    if not raw:
        return options

    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            options[key.strip()] = value.strip()
            continue
        # Sem "=" → assume parâmetro principal (device/display)
        if token and "device" not in options:
            options["device"] = token
        elif token and "display" not in options:
            options["display"] = token
    return options


def resolve_weights_path(candidate: str) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.is_file():
        return candidate_path
    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS
    return candidate_path
