"""Ferramenta para gravar segmentos RTSP contínuos em disco.

Os arquivos gerados são usados posteriormente para montar clipes quando um
evento é detectado. Mantemos os segmentos curtos (ex.: 2s) para facilitar a
montagem de janelas com `pre`/`post` segundos.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Captura RTSP e grava segmentos contínuos em disco",
    )
    parser.add_argument("--camera", required=True, help="Identificador lógico da câmera")
    parser.add_argument("--rtsp", required=True, help="URL RTSP ou 'webcam'")
    parser.add_argument(
        "--out",
        default="runs/buffer",
        help="Diretório base onde os segmentos serão armazenados",
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=2,
        help="Duração de cada segmento em segundos",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    outdir = Path(args.out) / args.camera
    outdir.mkdir(parents=True, exist_ok=True)

    # Gravamos com formatação por data (YYYYMMDD/HHMMSS.m4s) para facilitar a
    # reconstrução de clipes posteriormente.
    segment_template = outdir / "%Y%m%d" / "%H%M%S.m4s"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-stimeout",
        "5000000",
        "-i",
        "0" if args.rtsp.lower() == "webcam" else args.rtsp,
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(args.segment),
        "-strftime",
        "1",
        str(segment_template),
    ]

    print("Comando FFmpeg:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg não encontrado. Instale o utilitário para capturar a stream RTSP."
        ) from exc


if __name__ == "__main__":
    main()