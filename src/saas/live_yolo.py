"""Pipeline de inferência em tempo real com YOLOv8.

Este módulo foi refatorado para privilegiar organização e clareza. O fluxo
agora utiliza o `CaptureManager` para gerar segmentos em `runs/buffer/<camera>`
e processa automaticamente tanto frames ao vivo quanto arquivos gravados.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from saas import config
from saas.annotate import annotate_clip
from saas.clipper import collect_clip
from saas.infer_tcn import TCNInfer
from saas.pose_features_yolo import features_from_kpts
from saas.pose_yolo import _best_person
from saas.utils.logger import get_logger

LOGGER = get_logger("saas.live")


# ---------------------------------------------------------------------------
# Helpers geométricos


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


# ---------------------------------------------------------------------------
# Fontes de frame


class FrameSource:
    """Combina leitura direta da stream e dos segmentos gravados."""

    SUPPORTED_EXTENSIONS = {".mp4", ".m4s", ".mkv", ".avi", ".mov"}

    def __init__(
        self,
        source: str,
        buffer_dir: Path,
        reconnect_interval: float = 5.0,
        buffer_interval: float = 1.0,
    ):
        self.source = source
        self.source_type = config.detect_source_type(source)
        self.buffer_dir = buffer_dir
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self._processed_files: set[Path] = set()
        self.reconnect_interval = reconnect_interval
        self.buffer_interval = max(0.5, buffer_interval)
        self.logger = get_logger("saas.live.frames")

        self.logger.info(
            "FrameSource configurado tipo=%s buffer=%s", self.source_type, self.buffer_dir
        )
        if self.source_type in {"screen", "local"}:
            self.logger.info(
                "Origem '%s' depende dos segmentos gravados pelo CaptureManager.",
                self.source_type,
            )

    # ------------------------------- buffer
    def iter_buffer_frames(self) -> Iterator[Tuple[np.ndarray, str, float]]:
        """Percorre novos segmentos armazenados no disco."""

        files = sorted(
            f
            for f in self.buffer_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        for file_path in files:
            if file_path in self._processed_files:
                continue

            self.logger.info("Processando segmento gravado: %s", file_path)
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                self.logger.warning("Não foi possível abrir o arquivo %s", file_path)
                self._processed_files.add(file_path)
                continue

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame, "buffer", time.time()

            cap.release()
            self._processed_files.add(file_path)

    # ------------------------------- live
    def iter_live_stream(self) -> Iterator[Tuple[np.ndarray, str, float]]:
        """Mantém conexão com a origem de vídeo, realizando reconexões."""

        if self.source_type not in {"rtsp", "custom"}:
            return

        source = self.source
        if self.source_type == "custom" and source.isdigit():
            source = int(source)

        backoff = self.reconnect_interval
        while True:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.logger.warning(
                    "Falha ao abrir stream %s. Nova tentativa em %.1fs", self.source, backoff
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                yield from self.iter_buffer_frames()
                continue

            self.logger.info("Stream %s conectada com sucesso", self.source)
            backoff = self.reconnect_interval
            failure_count = 0

            while True:
                ok, frame = cap.read()
                ts = time.time()
                if not ok:
                    failure_count += 1
                    if failure_count >= 5:
                        self.logger.warning(
                            "Falha de leitura na stream %s. Reiniciando conexão.", self.source
                        )
                        break
                    time.sleep(0.1)
                    continue

                failure_count = 0
                yield frame, "live", ts

            cap.release()
            self.logger.info("Conexão encerrada. Tentando reconectar...")
            time.sleep(backoff)

    # ------------------------------- combinado
    def iter_frames(self) -> Iterator[Tuple[np.ndarray, str, float]]:
        """Intercala frames ao vivo com os segmentos gravados."""

        live_iter: Optional[Iterator[Tuple[np.ndarray, str, float]]] = None
        if self.source_type in {"rtsp", "custom"}:
            live_iter = self.iter_live_stream()

        next_buffer_poll = 0.0
        while True:
            now = time.time()
            if now >= next_buffer_poll:
                for frame_info in self.iter_buffer_frames():
                    yield frame_info
                next_buffer_poll = now + self.buffer_interval

            if live_iter is None:
                time.sleep(self.buffer_interval)
                continue

            try:
                yield next(live_iter)
            except StopIteration:
                live_iter = self.iter_live_stream()
            except Exception as exc:  # pragma: no cover - segurança extra
                self.logger.exception("Erro lendo frame ao vivo: %s", exc)
                time.sleep(self.reconnect_interval)
                live_iter = self.iter_live_stream()


# ---------------------------------------------------------------------------
# Runner principal


class LiveYoloRunner:
    """Executa inferência em tempo real combinando stream e buffer."""

    def __init__(self, args: argparse.Namespace) -> None:
        config.ensure_runtime_directories()
        self.args = args
        buffer_path = Path(args.buffer) if args.buffer else config.default_buffer_dir(args.camera)
        self.buffer_dir = buffer_path
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.reconnect_seconds = max(0.5, args.reconnect)
        self.buffer_interval = max(0.5, args.buffer_interval)
        self.frame_source = FrameSource(
            args.rtsp,
            self.buffer_dir,
            reconnect_interval=self.reconnect_seconds,
            buffer_interval=self.buffer_interval,
        )

        self.api_settings = config.load_api_settings()
        self.api_url = (args.api_url or self.api_settings.url).rstrip("/")
        self.api_key = args.api_key or self.api_settings.key

        self.theta_rot = math.radians(args.theta_deg)
        self.conf_min = args.conf
        self.prev_hip: Optional[float] = None
        self.flat_since: Optional[float] = None
        self.tcn: Optional[TCNInfer] = None
        self.tcn_prob = 0.0
        self.last_alert = 0.0
        self.frame_counter = 0
        self.last_log = time.time()

        resolved_weights = config.resolve_weights_path(args.weights)
        LOGGER.info("Carregando YOLOv8 com pesos %s", resolved_weights)
        self.model = self._load_model(resolved_weights)

        if args.use_tcn:
            self.tcn = TCNInfer(args.tcn_path)
            self.tcn.warmup(T=args.tcn_window, F=5)
            LOGGER.info(
                "TCN carregado de %s com janela %d", args.tcn_path, args.tcn_window
            )

        LOGGER.info(
            "Pipeline Live YOLO inicializado camera=%s origem=%s buffer=%s reconnect=%.1fs buffer_poll=%.1fs",
            args.camera,
            args.rtsp,
            self.buffer_dir,
            self.reconnect_seconds,
            self.buffer_interval,
        )

    # ------------------------------- modelo
    def _load_model(self, weights_path: Path) -> YOLO:
        try:
            return YOLO(str(weights_path))
        except Exception as exc:  # pragma: no cover - falhas externas
            LOGGER.error("Falha ao carregar YOLO (%s). Tentando fallback...", exc)
            time.sleep(1.0)
            return YOLO(str(weights_path))

    def _reload_model(self) -> None:
        LOGGER.warning("Recarregando modelo YOLO após erro de inferência")
        self.model = self._load_model(config.resolve_weights_path(self.args.weights))

    # ------------------------------- métricas auxiliares
    def _log_metrics(self) -> None:
        now = time.time()
        elapsed = now - self.last_log
        if elapsed < 10:
            return
        fps = self.frame_counter / elapsed if elapsed else 0.0
        LOGGER.info("Taxa média de processamento: %.2f FPS", fps)
        self.frame_counter = 0
        self.last_log = now

    # ------------------------------- TCN opcional
    def _push_tcn(self, keypoints: np.ndarray, box: np.ndarray) -> None:
        if self.tcn is None:
            self.tcn_prob = 0.0
            return

        feats, mask = features_from_kpts(
            keypoints[None, ...].astype(np.float32),
            np.array([box], dtype=np.float32),
        )
        feat_vec = feats[0] if mask[0] > 0.5 else np.zeros(5, dtype=np.float32)
        logits = self.tcn.push_and_score(feat_vec)
        if logits.ndim == 0 or logits.size == 0:
            self.tcn_prob = 0.0
        else:
            logits = logits.astype(np.float32)
            logits -= logits.max()
            exps = np.exp(logits)
            probs = exps / exps.sum()
            self.tcn_prob = float(probs[-1])

    # ------------------------------- alertas
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
                camera_id=self.args.camera,
                when=event_time,
                pre=self.args.pre,
                post=self.args.post,
            )

            events = [
                {
                    "t0": 0.0,
                    "t1": self.args.pre + self.args.post,
                    "bbox": [
                        x1 / width,
                        y1 / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ],
                    "label": "fall",
                    "score": float(score),
                }
            ]
            annotated = annotate_clip(local_path, events=events)
            clip_url = f"{self.api_url}/clips/{Path(annotated).name}"

            payload = {
                "camera_id": self.args.camera,
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
        except Exception as exc:  # pragma: no cover - erros externos
            LOGGER.exception("Erro ao gerar/publicar clipe: %s", exc)

    # ------------------------------- processamento
    def _process_frame(self, frame: np.ndarray, source: str) -> None:
        self.frame_counter += 1
        self._log_metrics()

        height, width = frame.shape[:2]

        try:
            result = self.model.predict(
                frame,
                imgsz=self.args.imgsz,
                conf=self.conf_min,
                verbose=False,
            )[0]
        except Exception:  # pragma: no cover - falhas no modelo
            LOGGER.exception("Erro na inferência do modelo")
            self._reload_model()
            return

        best = _best_person(result)
        now = time.time()

        if best is None:
            self.prev_hip = None
            self.flat_since = None
            if self.tcn is not None:
                self._push_tcn(np.zeros((17, 3), dtype=np.float32), np.zeros(4))
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

        LOGGER.info(
            "Detecção camera=%s fonte=%s score=%.2f angle=%.1f vy=%.2f flat_ratio=%.2f tcn=%.2f",
            self.args.camera,
            source,
            score,
            math.degrees(angle),
            vy,
            flat_ratio,
            self.tcn_prob,
        )

        if confirmed and (now - self.last_alert) > self.args.debounce:
            self.last_alert = now
            LOGGER.info(
                "Evento confirmado camera=%s origem=%s score=%.2f", self.args.camera, source, score
            )
            self._post_alert(score, angle, vy, (x1, y1, x2, y2), (height, width))

    # ------------------------------- loop principal
    def run(self) -> None:
        try:
            for frame, source, _ts in self.frame_source.iter_frames():
                self._process_frame(frame, source)
        except KeyboardInterrupt:
            LOGGER.info("Execução interrompida pelo usuário")


# ---------------------------------------------------------------------------
# CLI


def run(args: argparse.Namespace) -> None:
    runner = LiveYoloRunner(args)
    runner.run()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8 live fall-detection padronizado")
    parser.add_argument("--camera", default="cam01", help="ID lógico da câmera")
    parser.add_argument("--rtsp", required=True, help="Origem: rtsp://, 'screen', 'local' ou dispositivo")
    parser.add_argument(
        "--buffer",
        default=None,
        help="Pasta do ring buffer (runs/buffer/<camera>)",
    )
    parser.add_argument("--weights", default=str(config.DEFAULT_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--theta-deg", type=float, default=55.0, help="Limiar de rotação do tronco (graus)")
    parser.add_argument("--vy-min", type=float, default=0.25, help="Velocidade vertical normalizada mínima")
    parser.add_argument("--flat-ratio", type=float, default=0.60, help="H/W < flat_ratio => deitado")
    parser.add_argument("--flat-sec", type=float, default=2.0, help="Persistência deitado para confirmar (s)")
    parser.add_argument("--pre", type=float, default=5.0, help="Segundos antes do evento no clipe")
    parser.add_argument("--post", type=float, default=5.0, help="Segundos após o evento no clipe")
    parser.add_argument("--debounce", type=float, default=10.0, help="Tempo mínimo entre alertas (s)")
    parser.add_argument("--reconnect", type=float, default=5.0, help="Intervalo para tentar reconectar a stream")
    parser.add_argument(
        "--buffer-interval",
        type=float,
        default=1.0,
        help="Intervalo em segundos para varrer novos segmentos do buffer",
    )
    api_defaults = config.load_api_settings()
    parser.add_argument("--api-url", default=api_defaults.url)
    parser.add_argument("--api-key", default=api_defaults.key)
    parser.add_argument("--use-tcn", action="store_true", help="Ativa o TCN treinado (ONNX)")
    parser.add_argument("--tcn-path", default="runs/models/tcn.onnx")
    parser.add_argument("--tcn-window", type=int, default=32, help="Tamanho da janela temporal")
    return parser


if __name__ == "__main__":  # pragma: no cover - execução direta
    run(build_argparser().parse_args())
