"""Pipeline online de detecção de quedas usando YOLOv8 pose."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import math
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from saas.annotate import annotate_clip
from saas.clipper import collect_clip
from saas.infer_tcn import TCNInfer
from saas.pose_features_yolo import features_from_kpts
from saas.pose_yolo import _best_person


def trunk_angle(keypoints: np.ndarray) -> float:
    """Ângulo do tronco em relação ao eixo vertical (radianos)."""

    L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
    if (
        keypoints[L_SHO, 2] < 0.2
        or keypoints[R_SHO, 2] < 0.2
        or keypoints[L_HIP, 2] < 0.2
        or keypoints[R_HIP, 2] < 0.2
    ):
        return 0.0

    shoulders = (keypoints[L_SHO, :2] + keypoints[R_SHO, :2]) / 2.0
    hips = (keypoints[L_HIP, :2] + keypoints[R_HIP, :2]) / 2.0
    vec = shoulders - hips
    return math.atan2(abs(vec[0]), max(1.0, abs(vec[1])))


def vy_norm(prev_hip_y: Optional[float], hip_y: float, bbox_h: float) -> float:
    if prev_hip_y is None or bbox_h <= 1:
        return 0.0
    return (hip_y - prev_hip_y) / bbox_h


def _open_detection_log(path: str) -> Optional[Tuple[csv.writer, io.TextIOWrapper]]:
    if not path:
        return None

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(file)
    if log_path.stat().st_size == 0:
        writer.writerow(
            [
                "ts_iso",
                "camera_id",
                "score",
                "trunk_angle_deg",
                "vy_norm",
                "flat_ratio",
                "probable",
                "confirmed",
                "tcn_prob",
            ]
        )

    return writer, file


def _close_detection_log(handle: Optional[Tuple[csv.writer, io.TextIOWrapper]]) -> None:
    if handle is None:
        return
    _, file = handle
    file.close()


def _tcn_probability(logits: np.ndarray) -> float:
    if logits.ndim == 0 or logits.size == 0:
        return 0.0
    if logits.size == 1:
        return float(logits.item())
    logits = logits.astype(np.float32)
    logits -= logits.max()
    exps = np.exp(logits)
    probs = exps / exps.sum()
    return float(probs[-1])


def run(args: argparse.Namespace) -> None:
    api_url = args.api_url.rstrip("/")
    api_key = args.api_key
    cam_id = args.camera
    buffer_dir = args.buffer

    tcn: Optional[TCNInfer] = None
    if args.use_tcn:
        tcn = TCNInfer(args.tcn_path)
        tcn.warmup(T=args.tcn_window, F=5)
        print(f"[live] TCN carregado de {args.tcn_path}, janela {args.tcn_window}")

    model = YOLO(args.weights)
    src = 0 if args.rtsp.lower() == "webcam" else args.rtsp
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {args.rtsp}")

    prev_hip: Optional[float] = None
    flat_since: Optional[float] = None
    last_alert = 0.0
    theta_rot = math.radians(args.theta_deg)
    conf_min = args.conf

    log_handle = _open_detection_log(args.log_detections)
    print(
        f"[live] camera={cam_id} src={args.rtsp} theta={args.theta_deg} "
        f"vy_min={args.vy_min} flat_sec={args.flat_sec}"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            result = model.predict(frame, imgsz=args.imgsz, conf=conf_min, verbose=False)[0]
            best = _best_person(result)
            height, width = frame.shape[:2]
            now = time.time()
            tcn_prob = 0.0

            if best is None:
                prev_hip = None
                flat_since = None
                if tcn is not None:
                    tcn_prob = _tcn_probability(tcn.push_and_score(np.zeros(5, dtype=np.float32)))
                cv2.imshow(cam_id, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            keypoints, box, score = best
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            bbox_h = max(1.0, y2 - y1)
            L_HIP, R_HIP = 11, 12
            if keypoints[L_HIP, 2] > 0.2 and keypoints[R_HIP, 2] > 0.2:
                hip = (keypoints[L_HIP, 1] + keypoints[R_HIP, 1]) / 2.0
            else:
                hip = (y1 + y2) / 2.0

            angle = trunk_angle(keypoints)
            vy = vy_norm(prev_hip, hip, bbox_h)
            prev_hip = hip

            probable = angle > theta_rot and vy > args.vy_min
            flat = (bbox_h / max(1.0, x2 - x1)) < args.flat_ratio

            if probable and flat_since is None:
                flat_since = now
            if not flat:
                flat_since = None

            confirmed = flat_since is not None and (now - flat_since) >= args.flat_sec

            if tcn is not None:
                feats, mask = features_from_kpts(
                    keypoints[None, ...].astype(np.float32),
                    np.array([[x1, y1, x2, y2]], dtype=np.float32),
                )
                feat_vec = feats[0] if mask[0] > 0.5 else np.zeros(5, dtype=np.float32)
                tcn_prob = _tcn_probability(tcn.push_and_score(feat_vec))

            color = (0, 0, 255) if confirmed else (0, 255, 0)
            cv2.putText(
                frame,
                f"ang={math.degrees(angle):.0f} vy={vy:.2f} flat={flat}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            if log_handle is not None:
                writer, file = log_handle
                ts_iso = dt.datetime.now(dt.timezone.utc).isoformat()
                writer.writerow(
                    [
                        ts_iso,
                        cam_id,
                        float(score),
                        math.degrees(angle),
                        float(vy),
                        bbox_h / max(1.0, x2 - x1),
                        int(probable),
                        int(confirmed),
                        float(tcn_prob),
                    ]
                )
                file.flush()

            if confirmed and (now - last_alert) > args.debounce:
                last_alert = now
                try:
                    event_time = dt.datetime.now(dt.timezone.utc)
                    local_path, _ = collect_clip(
                        buffer_dir=buffer_dir,
                        camera_id=cam_id,
                        when=event_time,
                        pre=args.pre,
                        post=args.post,
                    )

                    ev_bbox = [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]
                    events = [
                        {
                            "t0": 0.0,
                            "t1": args.pre + args.post,
                            "bbox": ev_bbox,
                            "label": "fall",
                            "score": float(score),
                        }
                    ]
                    annotated = annotate_clip(local_path, events=events)
                    clip_url = f"{api_url}/clips/{Path(annotated).name}"

                    payload = {
                        "camera_id": cam_id,
                        "type": "fall",
                        "score": float(score),
                        "clip_path": clip_url,
                        "extra": {
                            "source": "yolov8",
                            "angle_deg": math.degrees(angle),
                            "vy_norm": float(vy),
                            "tcn_prob": float(tcn_prob),
                        },
                    }
                    response = requests.post(
                        f"{api_url}/alerts",
                        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
                        json=payload,
                        timeout=5,
                    )
                    response.raise_for_status()
                    print("[ALERTA] publicado:", clip_url)
                except Exception as exc:  # pragma: no cover - erros de IO externos
                    print("[ERRO] gerar/publicar clipe:", exc)

            cv2.imshow(cam_id, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        _close_detection_log(log_handle)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8-pose live fall-detection (baseline)")
    parser.add_argument("--camera", default="cam01", help="ID lógico da câmera")
    parser.add_argument("--rtsp", required=True, help='"webcam" ou URL RTSP')
    parser.add_argument("--buffer", required=True, help="Pasta do ring buffer: runs/buffer/<camera>")
    parser.add_argument("--weights", default="yolov8n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--log-detections",
        default="runs/logs/detections.csv",
        help="CSV onde salvar timestamp, camera_id, score e features",
    )
    parser.add_argument("--theta-deg", type=float, default=55.0, help="Limiar de rotação do tronco (graus)")
    parser.add_argument("--vy-min", type=float, default=0.25, help="Velocidade vertical normalizada mínima")
    parser.add_argument("--flat-ratio", type=float, default=0.60, help="H/W < flat_ratio => deitado")
    parser.add_argument("--flat-sec", type=float, default=2.0, help="Persistência deitado para confirmar (s)")
    parser.add_argument("--pre", type=float, default=5.0, help="Segundos antes do evento no clipe")
    parser.add_argument("--post", type=float, default=5.0, help="Segundos após o evento no clipe")
    parser.add_argument("--debounce", type=float, default=10.0, help="Tempo mínimo entre alertas (s)")
    parser.add_argument("--api-url", default=os.getenv("SAAS_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("SAAS_API_KEY", "minha-chave-forte"))
    parser.add_argument("--use-tcn", action="store_true", help="Usa o TCN treinado (ONNX) em tempo real")
    parser.add_argument("--tcn-path", default="runs/models/tcn.onnx")
    parser.add_argument("--tcn-window", type=int, default=32, help="Tamanho da janela temporal")
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())

