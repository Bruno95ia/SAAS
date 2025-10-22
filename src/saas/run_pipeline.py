"""Ferramentas para anotar vídeos e disparar alertas manualmente."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests

from saas import config
from saas.annotate import annotate_clip
from saas.utils.logger import get_logger

LOGGER = get_logger("saas.pipeline")


def load_events_sidecar(video_path: Path) -> Optional[List[Dict[str, Any]]]:
    sidecar = video_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text())
        if isinstance(data, list) and all("bbox" in item for item in data):
            return data  # type: ignore[return-value]
    except json.JSONDecodeError:
        LOGGER.warning("Arquivo JSON inválido: %s", sidecar)
    return None


def demo_event_for_video(video_path: Path) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (20 * fps)
    duration = float(nframes / fps)
    cap.release()

    t0 = 0.30 * duration
    t1 = 0.70 * duration
    return [
        {
            "t0": t0,
            "t1": t1,
            "bbox": [0.35, 0.35, 0.30, 0.30],
            "label": "fall",
            "score": 0.80,
        }
    ]


def post_alert(api_url: str, api_key: str, camera_id: str, clip_url: str, score: float, extra: dict) -> None:
    payload = {
        "camera_id": camera_id,
        "type": "fall",
        "score": float(score),
        "clip_path": clip_url,
        "extra": extra,
    }
    response = requests.post(
        f"{api_url.rstrip('/')}/alerts",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    LOGGER.info("POST /alerts OK camera=%s url=%s", camera_id, clip_url)


def process_video(
    video_path: Path,
    out_dir: Path,
    api_url: str,
    api_key: str,
    camera_id: str,
    publish: bool,
    overwrite: bool,
) -> None:
    if video_path.name.endswith("_annot.mp4"):
        LOGGER.debug("Pulando vídeo anotado: %s", video_path.name)
        return

    dst = out_dir / f"{video_path.stem}_annot.mp4"
    events = load_events_sidecar(video_path) or demo_event_for_video(video_path)

    if dst.exists() and not overwrite:
        LOGGER.info("Já existe anotação para %s", video_path.name)
    else:
        LOGGER.info("Anotando %s (%d eventos)", video_path.name, len(events))
        annotate_clip(str(video_path), dst_path=str(dst), events=events)
        LOGGER.info("Arquivo gerado: %s", dst)

    if publish:
        clip_url = f"{api_url.rstrip('/')}/clips/{dst.name}"
        extra = {"annotated": True, "src": video_path.name}
        try:
            post_alert(api_url, api_key, camera_id, clip_url, events[0].get("score", 0.9), extra)
        except Exception as exc:  # pragma: no cover - falhas externas
            LOGGER.exception("Falha ao publicar alerta para %s: %s", video_path, exc)


def run(
    input_dir: Path,
    pattern: str,
    api_url: str,
    api_key: str,
    camera_id: str,
    publish: bool,
    overwrite: bool,
) -> None:
    config.ensure_runtime_directories()
    videos = sorted(input_dir.rglob(pattern))
    if not videos:
        LOGGER.warning("Nenhum vídeo encontrado em %s (padrão=%s)", input_dir, pattern)
        return

    clips_dir = config.CLIPS_DIR
    clips_dir.mkdir(parents=True, exist_ok=True)

    for video in videos:
        process_video(video, clips_dir, api_url, api_key, camera_id, publish, overwrite)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anota vídeos e publica alertas de teste.")
    parser.add_argument("-i", "--input", type=Path, default=Path("runs/clips"))
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--camera", default="cam01")
    defaults = config.load_api_settings()
    parser.add_argument("--api-url", default=defaults.url)
    parser.add_argument("--api-key", default=defaults.key)
    parser.add_argument("--post", action="store_true", help="Publica alertas na API")
    parser.add_argument("--overwrite", action="store_true", help="Regenera arquivos *_annot.mp4")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args.input, args.pattern, args.api_url, args.api_key, args.camera, args.post, args.overwrite)


if __name__ == "__main__":  # pragma: no cover
    main()
