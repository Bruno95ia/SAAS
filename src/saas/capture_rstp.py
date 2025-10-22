"""CLI de captura contínua utilizando :class:`CaptureManager`.

Este script mantém a compatibilidade com a versão anterior, porém agora delega
à classe `CaptureManager` a responsabilidade de montar o comando FFmpeg,
registrar logs e reconectar automaticamente em caso de queda da stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from saas import config
from saas.capture_manager import CaptureManager
from saas.utils.logger import get_logger


def build_argparser() -> argparse.ArgumentParser:
    """Constroi o parser de argumentos da CLI."""

    parser = argparse.ArgumentParser(
        description="Captura RTSP ou tela local e grava segmentos contínuos em disco",
    )
    parser.add_argument("--camera", required=True, help="Identificador lógico da câmera")
    parser.add_argument(
        "--rtsp",
        required=True,
        help="Origem do vídeo: URL rtsp://, 'screen' ou 'local'",
    )
    parser.add_argument(
        "--out",
        default=str(config.BUFFER_DIR),
        help="Diretório base onde os segmentos serão armazenados",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=2,
        help="Duração de cada segmento em segundos",
    )
    parser.add_argument(
        "--reconnect",
        type=float,
        default=5.0,
        help="Intervalo (s) antes de tentar reconectar após uma falha",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    logger = get_logger("saas.capture")
    config.ensure_runtime_directories()

    output_base = Path(args.out)
    output_base.mkdir(parents=True, exist_ok=True)

    manager = CaptureManager.from_args(
        camera_id=args.camera,
        source=args.rtsp,
        output_base=output_base,
        segment_seconds=args.segment,
        reconnect_seconds=args.reconnect,
    )

    logger.info(
        "Iniciando captura unificada camera=%s origem=%s destino=%s",
        args.camera,
        args.rtsp,
        manager.segment_template.parent,
    )

    manager.run()


if __name__ == "__main__":  # pragma: no cover - execução direta
    main()
