"""Pipeline de detecção em tempo real com reconexão e fallback offline.

Compatível com Python 3.12 e Ubuntu 24.04.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import logging
import math
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from saas.annotate import annotate_clip
from saas.clipper import collect_clip
from saas.infer_tcn import TCNInfer
from saas.pose_features_yolo import features_from_kpts
from saas.pose_yolo import _best_person


LOGGER = logging.getLogger("saas.live")


def _setup_logging() -> None:
    log_dir = Path("runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "saas.log"

    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger("saas")
    if not root.handlers:
        root.addHandler(handler)
        root.setLevel(logging.INFO)

    if not LOGGER.handlers:
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False


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
                "source",
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


def _resolve_weights_path(candidate: str) -> str:
    """Resolve o caminho do peso do YOLO com fallback para `weights/yolov8n.pt`."""

    cand_path = Path(candidate)
    if cand_path.is_file():
        return str(cand_path)

    fallback = Path("weights") / "yolov8n.pt"
    if fallback.is_file():
        LOGGER.warning(
            "Peso %s não encontrado. Usando fallback local %s", candidate, fallback
        )
        return str(fallback)

    LOGGER.warning(
        "Peso %s não encontrado e fallback padrão ausente. YOLO tentará baixar o modelo.",
        candidate,
    )
    return candidate


class FrameSource:
    """Gerencia a leitura da stream RTSP e dos segmentos gravados."""

    SUPPORTED_EXTENSIONS = {".mp4", ".m4s", ".mkv", ".avi", ".mov"}

    def __init__(self, rtsp: str, buffer_dir: Path, reconnect_interval: float = 5.0):
        self._rtsp = rtsp
        self._buffer_dir = buffer_dir
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        self._processed_files: set[Path] = set()
        self._reconnect_interval = reconnect_interval

    def iter_buffer_frames(self) -> Iterator[Tuple[np.ndarray, str]]:
        """Percorre novos segmentos gravados no disco."""

        files = sorted(
            f
            for f in self._buffer_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        for file_path in files:
            if file_path in self._processed_files:
                continue

            LOGGER.info("Processando segmento gravado: %s", file_path)
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                LOGGER.warning("Não foi possível abrir o arquivo %s", file_path)
                self._processed_files.add(file_path)
                continue

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, "buffer"

            cap.release()
            self._processed_files.add(file_path)

    def iter_live_stream(self) -> Iterator[Tuple[np.ndarray, str]]:
        """Tenta manter a conexão com a stream RTSP continuamente."""

        backoff = self._reconnect_interval
        src = 0 if self._rtsp.lower() == "webcam" else self._rtsp

        while True:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                LOGGER.warning(
                    "Falha ao abrir stream %s. Tentando novamente em %.1fs",
                    self._rtsp,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                yield from self.iter_buffer_frames()
                continue

            LOGGER.info("Stream %s conectada com sucesso", self._rtsp)
            backoff = self._reconnect_interval

            failure_count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    failure_count += 1
                    if failure_count >= 5:
                        LOGGER.warning(
                            "Falha de leitura na stream %s. Reiniciando conexão.",
                            self._rtsp,
                        )
                        break
                    time.sleep(0.1)
                    continue

                failure_count = 0
                yield frame, "live"

            cap.release()
            LOGGER.info("Conexão com %s encerrada. Reabrindo...", self._rtsp)


class LiveYoloRunner:
    def __init__(self, args: argparse.Namespace):
        _setup_logging()
        self.args = args
        self.api_url = args.api_url.rstrip("/")
        self.api_key = args.api_key
        self.cam_id = args.camera
        self.buffer_dir = Path(args.buffer)
        self.theta_rot = math.radians(args.theta_deg)
        self.conf_min = args.conf
        self.last_alert = 0.0
        self.prev_hip: Optional[float] = None
        self.flat_since: Optional[float] = None
        self.tcn: Optional[TCNInfer] = None
        self.tcn_prob = 0.0
        self.model = self._load_model(args.weights)
        self.frame_source = FrameSource(args.rtsp, self.buffer_dir)
        self.log_handle = _open_detection_log(args.log_detections)

        if args.use_tcn:
            self.tcn = TCNInfer(args.tcn_path)
            self.tcn.warmup(T=args.tcn_window, F=5)
            LOGGER.info(
                "TCN carregado de %s com janela %d",
                args.tcn_path,
                args.tcn_window,
            )

        LOGGER.info(
            "Pipeline inicializado camera=%s src=%s theta=%.1f vy_min=%.2f flat_sec=%.1f",
            self.cam_id,
            args.rtsp,
            args.theta_deg,
            args.vy_min,
            args.flat_sec,
        )

    def _load_model(self, weights: str) -> YOLO:
        resolved = _resolve_weights_path(weights)
        LOGGER.info("Carregando modelo YOLO de %s", resolved)
        try:
            return YOLO(resolved)
        except Exception as exc:  # pragma: no cover - erro externo
            LOGGER.error("Falha ao carregar YOLO: %s", exc)
            time.sleep(1.0)
            return YOLO(resolved)

    def _reload_model(self) -> None:
        LOGGER.warning("Recarregando o modelo YOLO devido a erro de inferência")
        self.model = self._load_model(self.args.weights)

    def _log_detection(
        self,
        score: float,
        angle: float,
        vy: float,
        flat_ratio: float,
        probable: bool,
        confirmed: bool,
        source: str,
    ) -> None:
        if self.log_handle is None:
            return
        writer, file = self.log_handle
        ts_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        writer.writerow(
            [
                ts_iso,
                self.cam_id,
                float(score),
                math.degrees(angle),
                float(vy),
                flat_ratio,
                int(probable),
                int(confirmed),
                float(self.tcn_prob),
                source,
            ]
        )
        file.flush()

    def _push_tcn(self, keypoints: np.ndarray, box: np.ndarray) -> None:
        if self.tcn is None:
            self.tcn_prob = 0.0
            return

        feats, mask = features_from_kpts(
            keypoints[None, ...].astype(np.float32),
            np.array([box], dtype=np.float32),
        )
        feat_vec = feats[0] if mask[0] > 0.5 else np.zeros(5, dtype=np.float32)
        self.tcn_prob = _tcn_probability(self.tcn.push_and_score(feat_vec))

    def _post_alert(
        self,
        score: float,
        angle: float,
        vy: float,
        bbox: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> None:
        height, width = frame_size
        x1, y1, x2, y2 = bbox

        try:
            event_time = dt.datetime.now(dt.timezone.utc)
            local_path, _ = collect_clip(
                buffer_dir=str(self.buffer_dir),
                camera_id=self.cam_id,
                when=event_time,
                pre=self.args.pre,
                post=self.args.post,
            )

            ev_bbox = [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]
            events = [
                {
                    "t0": 0.0,
                    "t1": self.args.pre + self.args.post,
                    "bbox": ev_bbox,
                    "label": "fall",
                    "score": float(score),
                }
            ]
            annotated = annotate_clip(local_path, events=events)
            clip_url = f"{self.api_url}/clips/{Path(annotated).name}"

            payload = {
                "camera_id": self.cam_id,
                "type": "fall",
                "score": float(score),
                "clip_path": clip_url,
                "extra": {
                    "source": "yolov8",
                    "angle_deg": math.degrees(angle),
                    "vy_norm": float(vy),
                    "tcn_prob": float(self.tcn_prob),
                },
            }
            response = requests.post(
                f"{self.api_url}/alerts",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            LOGGER.info("Alerta publicado com sucesso: %s", clip_url)
        except Exception as exc:  # pragma: no cover - erros de IO externos
            LOGGER.exception("Erro ao gerar/publicar clipe: %s", exc)

    def _process_frame(self, frame: np.ndarray, source: str) -> None:
        height, width = frame.shape[:2]

        try:
            result = self.model.predict(
                frame,
                imgsz=self.args.imgsz,
                conf=self.conf_min,
                verbose=False,
            )[0]
        except Exception:  # pragma: no cover - erro no modelo
            LOGGER.exception("Erro na inferência do modelo")
            self._reload_model()
            return

        best = _best_person(result)
        now = time.time()

        if best is None:
            self.prev_hip = None
            self.flat_since = None
            if self.tcn is not None:
                self.tcn_prob = _tcn_probability(
                    self.tcn.push_and_score(np.zeros(5, dtype=np.float32))
                )
            return

        keypoints, box, score = best
        x1, y1, x2, y2 = box.astype(int)
        bbox_h = max(1.0, y2 - y1)

        L_HIP, R_HIP = 11, 12
        if keypoints[L_HIP, 2] > 0.2 and keypoints[R_HIP, 2] > 0.2:
            hip = (keypoints[L_HIP, 1] + keypoints[R_HIP, 1]) / 2.0
        else:
            hip = (y1 + y2) / 2.0

        angle = trunk_angle(keypoints)
        vy = vy_norm(self.prev_hip, hip, bbox_h)
        self.prev_hip = hip

        probable = angle > self.theta_rot and vy > self.args.vy_min
        flat_ratio = bbox_h / max(1.0, x2 - x1)
        flat = flat_ratio < self.args.flat_ratio

        if probable and flat and self.flat_since is None:
            self.flat_since = now
        if not flat:
            self.flat_since = None

        confirmed = self.flat_since is not None and (now - self.flat_since) >= self.args.flat_sec

        self._push_tcn(keypoints, box)
        self._log_detection(score, angle, vy, flat_ratio, probable, confirmed, source)

        if confirmed and (now - self.last_alert) > self.args.debounce:
            self.last_alert = now
            LOGGER.info(
                "Evento confirmado camera=%s score=%.2f angle=%.1f vy=%.2f fonte=%s",
                self.cam_id,
                score,
                math.degrees(angle),
                vy,
                source,
            )
            self._post_alert(score, angle, vy, (x1, y1, x2, y2), (height, width))

    def run(self) -> None:
        try:
            while True:
                for frame, source in self.frame_source.iter_buffer_frames():
                    self._process_frame(frame, source)

                for frame, source in self.frame_source.iter_live_stream():
                    self._process_frame(frame, source)
        finally:
            _close_detection_log(self.log_handle)


def run(args: argparse.Namespace) -> None:
    runner = LiveYoloRunner(args)
    runner.run()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8-pose live fall-detection (robusto)")
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


if __name__ == "__main__":  # pragma: no cover - execução como script
    run(build_argparser().parse_args())

