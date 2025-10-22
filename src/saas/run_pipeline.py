# src/saas/run_pipeline.py
from __future__ import annotations
import os, json, argparse, logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import cv2

from .annotate import annotate_clip

LOG = logging.getLogger("pipeline")

def load_events_sidecar(video_path: Path) -> Optional[List[Dict[str, Any]]]:
    sidecar = video_path.with_suffix(".json")
    if sidecar.exists():
        with open(sidecar, "r") as f:
            data = json.load(f)
        # valida formato básico
        if isinstance(data, list) and all("bbox" in d for d in data):
            return data
    return None

def demo_event_for_video(video_path: Path) -> List[Dict[str, Any]]:
    """Fallback: cria um evento 'fall' no meio do vídeo se não houver JSON."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (20 * fps)
    dur = float(nframes / fps)
    cap.release()

    # evento simples: bbox central por 40% da duração, começando aos 30%
    t0 = 0.30 * dur
    t1 = 0.70 * dur
    ev = {
        "t0": t0,
        "t1": t1,
        "bbox": [0.35, 0.35, 0.30, 0.30],  # x,y,w,h relativos
        "label": "fall",
        "score": 0.80,
    }
    return [ev]

def post_alert(api_url: str, api_key: str, camera_id: str, clip_url: str, score: float = 0.9, extra: dict | None = None):
    payload = {
        "camera_id": camera_id,
        "type": "fall",
        "score": float(score),
        "clip_path": clip_url,
        "extra": extra or {},
    }
    r = requests.post(
        f"{api_url.rstrip('/')}/alerts",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def run(
    input_dir: Path,
    pattern: str,
    api_url: str,
    api_key: str,
    camera_id: str,
    post: bool,
    overwrite: bool,
) -> None:
    LOG.info("Varredura: %s (padrao=%s)", input_dir, pattern)
    videos = sorted(input_dir.rglob(pattern))
    if not videos:
        LOG.warning("Nenhum vídeo encontrado.")
        return

    for vid in videos:
        if vid.name.endswith("_annot.mp4"):
            continue  # evita reprocessar anotados
        dst = vid.with_name(vid.stem + "_annot.mp4")

        if dst.exists() and not overwrite:
            LOG.info("Pulando (já anotado): %s", dst.name)
        else:
            events = load_events_sidecar(vid) or demo_event_for_video(vid)
            LOG.info("Anotando: %s (eventos=%d)", vid.name, len(events))
            out = annotate_clip(str(vid), dst_path=str(dst), events=events)
            LOG.info("Gerado: %s", out)

        if post:
            url = f"{api_url.rstrip('/')}/clips/{dst.name}"
            LOG.info("Publicando alerta: %s", url)
            try:
                resp = post_alert(api_url, api_key, camera_id, url, score=events[0].get("score", 0.9), extra={"annotated": True, "src": vid.name})
                LOG.info("POST /alerts OK: %s", resp)
            except Exception as e:
                LOG.exception("Falha ao publicar alerta: %s", e)

def main():
    parser = argparse.ArgumentParser(description="Pipeline: anotar vídeos e publicar alertas no painel.")
    parser.add_argument("-i", "--input", type=Path, default=Path("runs/clips"), help="Pasta base dos vídeos.")
    parser.add_argument("--pattern", type=str, default="*.mp4", help="Glob de busca (ex.: *.mp4).")
    parser.add_argument("--camera", type=str, default="cam01", help="ID da câmera para o alerta.")
    parser.add_argument("--api-url", type=str, default=os.getenv("SAAS_API_URL", "http://127.0.0.1:8000"), help="URL da API.")
    parser.add_argument("--api-key", type=str, default=os.getenv("SAAS_API_KEY", "dev-key"), help="API Key.")
    parser.add_argument("--post", action="store_true", help="Publica o alerta após anotar.")
    parser.add_argument("--overwrite", action="store_true", help="Regera o *_annot.mp4 mesmo se existir.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs verbosos.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    os.makedirs(args.input, exist_ok=True)

    run(args.input, args.pattern, args.api_url, args.api_key, args.camera, args.post, args.overwrite)

if __name__ == "__main__":
    main()