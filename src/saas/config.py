"""Configurações centrais do projeto SAAS.

Este módulo concentra caminhos e utilidades de configuração que são
compartilhadas pelos utilitários de captura e inferência. A ideia é manter
em um único lugar as convenções de diretórios e a leitura de variáveis de
ambiente, favorecendo organização e padronização.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Diretórios principais -----------------------------------------------------

# Base do projeto (../.. em relação a este arquivo localizado em src/saas).
BASE_DIR = Path(__file__).resolve().parents[1]

# Estrutura de diretórios esperada.
RUNS_DIR = BASE_DIR / "runs"
BUFFER_DIR = RUNS_DIR / "buffer"
LOG_DIR = RUNS_DIR / "logs"
RESULTS_DIR = RUNS_DIR / "results"
WEIGHTS_DIR = BASE_DIR / "weights"

# Arquivo de log padrão.
LOG_FILE = LOG_DIR / "saas.log"

# Pesos padrão do YOLO (permite sobrescrever via variável de ambiente).
DEFAULT_WEIGHTS = Path(
    os.getenv("SAAS_YOLO_WEIGHTS", WEIGHTS_DIR / "yolov8n.pt")
)


# Utilidades de ambiente ----------------------------------------------------

@dataclass(frozen=True)
class APISettings:
    """Configurações de conexão com a API de alertas."""

    url: str
    key: str


def load_api_settings() -> APISettings:
    """Carrega URL e chave de API a partir das variáveis de ambiente.

    Defaults mantêm compatibilidade com instalações locais, mas recomenda-se
    definir `SAAS_API_URL` e `SAAS_API_KEY` explicitamente em produção.
    """

    url = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000").rstrip("/")
    key = os.getenv("SAAS_API_KEY", "minha-chave-forte")
    return APISettings(url=url, key=key)


# Funções auxiliares -------------------------------------------------------

def ensure_runtime_directories() -> None:
    """Garante que a estrutura de diretórios necessária exista."""

    for folder in (RUNS_DIR, BUFFER_DIR, LOG_DIR, RESULTS_DIR, WEIGHTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def default_buffer_dir(camera_id: str) -> Path:
    """Retorna o diretório de buffer para uma câmera específica."""

    return BUFFER_DIR / camera_id


def detect_source_type(identifier: str) -> str:
    """Identifica o tipo da origem de vídeo a partir do argumento `--rtsp`.

    - URLs iniciando com ``rtsp://`` são tratadas como câmeras remotas.
    - Os valores ``screen`` e ``local`` ativam captura via AVFoundation
      (típico em macOS para tela ou webcam integradas).
    - Qualquer outro valor é retornado como ``custom`` para permitir
      manipulação futura (ex.: arquivos de teste).
    """

    value = identifier.strip().lower()
    if value.startswith("rtsp://"):
        return "rtsp"
    if value in {"screen", "local"}:
        return value
    return "custom"


def resolve_weights_path(candidate: str) -> Path:
    """Resolve o caminho do arquivo de pesos YOLO com fallback local."""

    candidate_path = Path(candidate)
    if candidate_path.is_file():
        return candidate_path

    if DEFAULT_WEIGHTS.is_file():
        return DEFAULT_WEIGHTS

    # Caso nenhum arquivo seja encontrado, retornamos o caminho informado para
    # permitir que a biblioteca `ultralytics` tente baixar automaticamente.
    return candidate_path


def segment_template(base_dir: Path) -> Path:
    """Retorna o template strftime usado para segmentação de vídeo."""

    return base_dir / "%Y%m%d" / "%H%M%S.m4s"
